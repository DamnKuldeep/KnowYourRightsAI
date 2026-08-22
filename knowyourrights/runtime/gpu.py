"""All GPU work funnels through here.

Two jobs:

1. **Serialise.** A 4 GB card cannot absorb two concurrent rerank batches. Every GPU call runs
   on a single dedicated worker thread behind a one-permit lock, so two browser tabs asking
   questions at once queue instead of colliding. Using a thread (not the event loop) keeps
   FastAPI responsive while CUDA is busy.

2. **Survive OOM.** Out-of-memory is treated as a routine, recoverable condition: empty the
   cache, halve the batch, retry; then fall back to CPU. A user question must never surface a
   CUDA error.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

from .. import config

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class GpuOutOfMemory(RuntimeError):
    """Raised when even a batch of one will not fit."""


def is_oom(exc: BaseException) -> bool:
    """torch raises a dedicated class in 2.x but still a bare RuntimeError in some paths."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "cuda oom" in msg)


def empty_cache() -> None:
    """Return cached blocks to the driver. Cheap, and the first thing to try on OOM."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


class GpuExecutor:
    """One worker thread, one permit. Also tracks idle time for optional model eviction."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kyr-gpu")
        self._lock = asyncio.Lock()
        self._last_used = time.monotonic()
        self._closed = False
        self._evict_hooks: list[Callable[[], None]] = []
        self._hook_lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────────────
    def register_evict_hook(self, hook: Callable[[], None]) -> None:
        """Called when the executor has been idle past ``MODEL_IDLE_EVICT_S``."""
        with self._hook_lock:
            self._evict_hooks.append(hook)

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    def shutdown(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=False, cancel_futures=True)

    # ── execution ────────────────────────────────────────────────────────────────────
    async def run(self, fn: Callable[..., R], *args, **kwargs) -> R:
        """Run ``fn`` on the GPU thread, exclusively."""
        if self._closed:
            raise RuntimeError("GPU executor is shut down")
        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(self._pool, lambda: fn(*args, **kwargs))
            finally:
                self._last_used = time.monotonic()

    async def map_batches(
        self,
        fn: Callable[[Sequence[T]], Sequence[R]],
        items: Sequence[T],
        batch_size: int,
        *,
        cpu_fn: Callable[[Sequence[T]], Sequence[R]] | None = None,
        label: str = "gpu-op",
    ) -> list[R]:
        """Apply ``fn`` over ``items`` in batches, halving on OOM.

        ``fn`` receives a slice and must return one result per input, in order. On repeated
        OOM we call ``cpu_fn`` for the remainder if the caller supplied one; otherwise we
        raise :class:`GpuOutOfMemory` so the caller can pick a different backend.
        """
        if not items:
            return []

        results: list[R] = []
        size = max(1, min(batch_size, len(items)))
        index = 0
        halvings = 0

        while index < len(items):
            batch = items[index:index + size]
            try:
                out = await self.run(fn, batch)
                results.extend(out)
                index += len(batch)
                continue
            except Exception as exc:
                if not is_oom(exc):
                    raise
                empty_cache()
                if size > 1 and halvings < config.GPU_OOM_RETRIES:
                    halvings += 1
                    size = max(1, size // 2)
                    log.warning("%s hit CUDA OOM; retrying with batch=%d (halving %d/%d)",
                                label, size, halvings, config.GPU_OOM_RETRIES)
                    continue
                if cpu_fn is not None:
                    log.warning("%s still OOM at batch=%d; finishing %d item(s) on CPU",
                                label, size, len(items) - index)
                    tail = items[index:]
                    out = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: list(cpu_fn(tail))
                    )
                    results.extend(out)
                    return results
                raise GpuOutOfMemory(
                    f"{label}: out of memory even at batch=1 with "
                    f"{len(items) - index} item(s) remaining"
                ) from exc

        return results

    async def maybe_evict(self) -> bool:
        """Free model memory if we've been idle long enough. Returns True if anything ran."""
        if config.MODEL_IDLE_EVICT_S <= 0 or self.idle_seconds < config.MODEL_IDLE_EVICT_S:
            return False
        with self._hook_lock:
            hooks = list(self._evict_hooks)
        if not hooks:
            return False
        log.info("idle for %.0fs — evicting local models to free VRAM", self.idle_seconds)
        for hook in hooks:
            try:
                await self.run(hook)
            except Exception as exc:
                log.warning("evict hook failed: %s", exc)
        empty_cache()
        return True


_EXECUTOR: GpuExecutor | None = None


def get_executor() -> GpuExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = GpuExecutor()
    return _EXECUTOR


async def evict_loop(interval_s: float = 60.0) -> None:
    """Background task: periodically give VRAM back when nobody is asking questions."""
    if config.MODEL_IDLE_EVICT_S <= 0:
        return
    executor = get_executor()
    while True:
        try:
            await asyncio.sleep(interval_s)
            await executor.maybe_evict()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a background chore must never take the server down
            log.warning("evict loop error: %s", exc)


def chunked(items: Iterable[T], size: int) -> list[list[T]]:
    """Small helper used by callers that batch outside :meth:`GpuExecutor.map_batches`."""
    out: list[list[T]] = []
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            out.append(batch)
            batch = []
    if batch:
        out.append(batch)
    return out
