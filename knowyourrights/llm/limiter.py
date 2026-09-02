"""Per-model rate limiting that pauses and continues instead of failing.

NVIDIA's free tier caps requests per minute **per model**, so a bucket per model id gives us
genuinely independent budgets: the planner hammering the small model does not consume the
writer's allowance.

Two behaviours matter:

* **Proactive spacing.** A token bucket keeps us under the ceiling so most 429s never happen.
* **Reactive pausing.** When one does happen we honour ``Retry-After`` exactly, tell the UI how
  long the wait is, and shrink the bucket (AIMD) so the next minute is calmer. The rate creeps
  back up on clean minutes. Nothing here ever raises on a 429 — that is the whole point.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable

from .. import config

log = logging.getLogger(__name__)

PauseCallback = Callable[[str, float, str], Awaitable[None] | None]
"""Called as ``(model, seconds, reason)`` whenever a bucket makes callers wait noticeably."""


class TokenBucket:
    """Async token bucket with AIMD rate adaptation and hard pauses."""

    def __init__(self, model: str, rpm: int, burst: int = 2) -> None:
        self.model = model
        self.base_rpm = max(1, rpm)
        self.rpm = float(self.base_rpm)
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._paused_until = 0.0
        self._lock = asyncio.Lock()
        self._last_penalty = 0.0
        self._last_recovery = time.monotonic()
        self.waits = 0
        self.penalties = 0
        self.total_wait_s = 0.0

    # ── internals ────────────────────────────────────────────────────────────────────
    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(float(self.burst), self._tokens + elapsed * (self.rpm / 60.0))

    def _recover(self) -> None:
        """Creep the rate back up after a clean minute."""
        now = time.monotonic()
        if self.rpm >= self.base_rpm or now - self._last_penalty < 60.0:
            return
        if now - self._last_recovery >= 60.0:
            self._last_recovery = now
            before = self.rpm
            self.rpm = min(float(self.base_rpm), self.rpm + config.AIMD_INCREASE)
            log.info("%s: clean minute, rate %.1f -> %.1f rpm", self.model, before, self.rpm)

    # ── public API ───────────────────────────────────────────────────────────────────
    async def acquire(self, on_pause: PauseCallback | None = None) -> None:
        """Block until this model may be called again. Never raises."""
        async with self._lock:
            self._recover()

            now = time.monotonic()
            if now < self._paused_until:
                wait = self._paused_until - now
                await self._sleep(wait, "rate limit", on_pause)

            self._refill()
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) * 60.0 / self.rpm
                await self._sleep(wait, "pacing", on_pause)
                self._refill()

            self._tokens = max(0.0, self._tokens - 1.0)

    async def _sleep(self, seconds: float, reason: str, on_pause: PauseCallback | None) -> None:
        seconds = max(0.0, seconds)
        if seconds <= 0:
            return
        self.waits += 1
        self.total_wait_s += seconds
        # Only tell the user about waits long enough to look like a stall.
        if on_pause is not None and seconds >= 1.0:
            try:
                result = on_pause(self.model, seconds, reason)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.debug("pause callback failed: %s", exc)
        await asyncio.sleep(seconds)

    def penalize(self, retry_after: float | None = None) -> float:
        """Record a 429. Returns how long the caller should wait before retrying."""
        self.penalties += 1
        self._last_penalty = time.monotonic()
        before = self.rpm
        self.rpm = max(float(config.AIMD_FLOOR_RPM), self.rpm * config.AIMD_DECREASE)

        # Trust the server's own number when it gives one; otherwise wait out a bucket refill.
        wait = retry_after if retry_after and retry_after > 0 else 60.0 / max(1.0, self.rpm)
        wait += random.uniform(0, 0.5)  # de-synchronise concurrent callers
        self._paused_until = max(self._paused_until, time.monotonic() + wait)
        log.warning("%s: 429 — rate %.1f -> %.1f rpm, pausing %.1fs",
                    self.model, before, self.rpm, wait)
        return wait

    def status(self) -> dict:
        return {
            "model": self.model,
            "rpm": round(self.rpm, 1),
            "base_rpm": self.base_rpm,
            "throttled": self.rpm < self.base_rpm,
            "paused_for_s": round(max(0.0, self._paused_until - time.monotonic()), 1),
            "waits": self.waits,
            "penalties": self.penalties,
            "total_wait_s": round(self.total_wait_s, 1),
        }


class LimiterRegistry:
    """One bucket per model id, created on first use."""

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def get(self, model: str, rpm: int | None = None) -> TokenBucket:
        bucket = self._buckets.get(model)
        if bucket is None:
            bucket = TokenBucket(model, rpm or 30)
            self._buckets[model] = bucket
        return bucket

    def status(self) -> list[dict]:
        return [b.status() for b in self._buckets.values()]

    def any_throttled(self) -> bool:
        return any(b.rpm < b.base_rpm for b in self._buckets.values())


_REGISTRY: LimiterRegistry | None = None


def get_limiters() -> LimiterRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = LimiterRegistry()
    return _REGISTRY
