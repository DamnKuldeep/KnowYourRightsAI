"""The query embedder — local ``BAAI/bge-m3``, permanently.

This model is not a choice. The corpus was embedded with it, so swapping it means re-embedding
all 38,890 chunks (DB README §8). When the hosted copy disappears from a catalogue, local is
the only option, which is why nothing here reaches for the network.

Cost control, measured on an RTX 3050:

* fp16 weights are ~1090 MiB of VRAM (fp32 would be ~2.2 GB and would not leave room for a
  reranker).
* The first encode costs 1.86s of CUDA warmup and every one after it ~25 ms — so we warm up
  at startup rather than making the first user wait.
* Loading spikes host RAM to ~2.3 GB, which is the most likely way to wedge a busy laptop.
  We check available RAM first and decline rather than risk it; retrieval then runs keyword-only
  until memory frees up.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .. import config
from ..runtime import gpu, resources
from ..runtime.cache import get_cache, key_of

log = logging.getLogger(__name__)


class Embedder:
    """Single-flight lazy load, then a warm model behind the shared GPU executor."""

    def __init__(self, plan: resources.ResourcePlan | None = None) -> None:
        self._plan = plan
        self._model = None
        self._load_lock = asyncio.Lock()
        self._load_failed: str | None = None
        self._warm = False
        self.encodes = 0
        self.cache_hits = 0

    @property
    def plan(self) -> resources.ResourcePlan:
        if self._plan is None:
            self._plan = resources.get_plan()
        return self._plan

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._load_failed

    # ── loading ──────────────────────────────────────────────────────────────────────
    def _load_sync(self):
        """Runs on the GPU worker thread so the CUDA context is created there."""
        import torch
        from sentence_transformers import SentenceTransformer

        plan = self.plan
        started = time.time()
        dtype = getattr(torch, plan.embed_dtype)
        try:
            model = SentenceTransformer(
                config.EMBED_MODEL, trust_remote_code=True, device=plan.embed_device,
                model_kwargs={"dtype": dtype},
            )
        except TypeError:
            # Older transformers spell it `torch_dtype`; newer ones deprecate that spelling.
            model = SentenceTransformer(
                config.EMBED_MODEL, trust_remote_code=True, device=plan.embed_device,
                model_kwargs={"torch_dtype": dtype},
            )
        model.max_seq_length = config.EMBED_MAX_SEQ
        log.info("embedder loaded: %s on %s/%s in %.1fs",
                 config.EMBED_MODEL, plan.embed_device, plan.embed_dtype, time.time() - started)
        return model

    async def ensure_loaded(self) -> bool:
        """Load once. Returns False if the model is unavailable — never raises."""
        if self._model is not None:
            return True
        if not self.plan.use_embedder:
            # The `lite` profile runs on BM25 alone. This is a deliberate configuration, not a
            # failure, so it is recorded once and never retried.
            if self._load_failed is None:
                self._load_failed = "lite profile: semantic search is off by configuration"
                log.info("lite profile — no embedder will be loaded")
            return False
        if self._load_failed is not None:
            return False

        async with self._load_lock:
            if self._model is not None:
                return True
            if self._load_failed is not None:
                return False

            snapshot = resources.probe()
            if snapshot.ram_available_mb < config.RAM_FLOOR_MB:
                self._load_failed = (
                    f"only {snapshot.ram_available_mb} MB RAM available (floor is "
                    f"{config.RAM_FLOOR_MB} MB) — refusing to load a 2 GB model"
                )
                log.warning("embedder unavailable: %s", self._load_failed)
                return False

            try:
                self._model = await gpu.get_executor().run(self._load_sync)
                gpu.get_executor().register_evict_hook(self._unload_sync)
                return True
            except Exception as exc:
                self._load_failed = f"{type(exc).__name__}: {exc}"
                log.error("embedder failed to load: %s", self._load_failed)
                gpu.empty_cache()
                return False

    def _unload_sync(self) -> None:
        if self._model is not None:
            log.info("evicting embedder from %s", self.plan.embed_device)
            self._model = None
            self._warm = False

    async def retry_load(self) -> bool:
        """Clear a previous failure and try again — used once RAM frees up."""
        self._load_failed = None
        return await self.ensure_loaded()

    async def warmup(self) -> bool:
        """Pay the ~1.9s CUDA warmup at startup instead of on the first question."""
        if not await self.ensure_loaded():
            return False
        if self._warm:
            return True
        started = time.time()
        await self.encode(["warmup"], use_cache=False)
        self._warm = True
        log.info("embedder warm in %.2fs (subsequent queries ~25 ms)", time.time() - started)
        return True

    # ── encoding ─────────────────────────────────────────────────────────────────────
    def _encode_sync(self, texts):
        import numpy as np

        vectors = self._model.encode(list(texts), normalize_embeddings=True,
                                     batch_size=self.plan.embed_batch,
                                     show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    async def encode(self, texts, use_cache: bool = True):
        """Encode a batch. Returns an (n, 1024) float32 array, or None if unavailable.

        bge-m3 encodes queries and passages symmetrically, so there is no prefix to add —
        adding one would silently shift queries away from the indexed vectors.
        """
        import numpy as np

        items = [str(t or "") for t in texts]
        if not items:
            return np.zeros((0, config.EMBED_DIM), dtype="float32")
        if not self.plan.use_embedder:
            # Answering some queries from cache and not others would make retrieval quality
            # depend on what happened to be asked before. In lite mode semantic search is off,
            # consistently.
            return None

        cache = get_cache() if use_cache else None
        out: list = [None] * len(items)
        todo: list[int] = []

        if cache is not None:
            for i, text in enumerate(items):
                hit = cache.get_vector(key_of(config.EMBED_MODEL, text))
                if hit is not None:
                    out[i] = hit
                    self.cache_hits += 1
                else:
                    todo.append(i)
        else:
            todo = list(range(len(items)))

        if todo:
            if not await self.ensure_loaded():
                return None
            pending = [items[i] for i in todo]
            try:
                vectors = await gpu.get_executor().map_batches(
                    self._encode_sync, pending, self.plan.embed_batch, label="embed",
                )
            except gpu.GpuOutOfMemory as exc:
                log.error("embedding ran out of memory: %s", exc)
                return None
            except Exception as exc:
                log.error("embedding failed: %s", exc)
                return None

            self.encodes += len(pending)
            for slot, vector in zip(todo, vectors):
                vector = np.asarray(vector, dtype="float32")
                out[slot] = vector
                if cache is not None:
                    cache.set_vector(key_of(config.EMBED_MODEL, items[slot]), vector)

        return np.vstack([np.asarray(v, dtype="float32").reshape(-1) for v in out])

    async def encode_one(self, text: str):
        """One query vector, or None. The common case — cached, so repeats are free."""
        result = await self.encode([text])
        return None if result is None else result[0]

    def status(self) -> dict:
        return {
            "model": config.EMBED_MODEL,
            "loaded": self.loaded,
            "warm": self._warm,
            "device": self.plan.embed_device,
            "dtype": self.plan.embed_dtype,
            "encodes": self.encodes,
            "cache_hits": self.cache_hits,
            "unavailable_reason": self._load_failed,
        }


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER
