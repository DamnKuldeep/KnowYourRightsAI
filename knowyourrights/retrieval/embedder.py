"""Query embeddings: ``baai/bge-m3`` over OpenRouter.

The model is not a choice. The corpus was embedded with bge-m3, so a different model would mean
re-embedding all 38,000+ chunks. OpenRouter serves the same model, and its vectors were checked
against the stored ones at cosine 1.0000 (``scripts/verify_embeddings.py`` repeats the check).

When the API cannot answer, :meth:`Embedder.encode` returns None and records why. Search then
runs on keyword (BM25) retrieval alone and tells the user, which is a real answer, only a less
thorough one.
"""

from __future__ import annotations

import logging

from .. import config
from ..llm import retrieval_api
from ..runtime.cache import get_cache, key_of

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        self.encodes = 0
        self.cache_hits = 0
        self.api_calls = 0
        self.api_failures = 0
        self.last_error: str | None = None
        self.warm = False

    @property
    def available(self) -> bool:
        return retrieval_api.available()

    @property
    def unavailable_reason(self) -> str | None:
        if not self.available:
            return "no OPENROUTER_API_KEY is configured"
        return self.last_error

    async def warmup(self) -> bool:
        """One real call at startup, proving the key, the model and the vector width."""
        self.warm = await self.encode(["warmup"], use_cache=False) is not None
        if not self.warm:
            log.error("embedding API did not answer at startup (%s) — retrieval will be "
                      "keyword-only until it recovers", self.unavailable_reason)
        return self.warm

    async def encode(self, texts, use_cache: bool = True):
        """An (n, 1024) float32 array of unit vectors, or None if the API is unavailable."""
        import numpy as np

        items = [str(t or "") for t in texts]
        if not items:
            return np.zeros((0, config.EMBED_DIM), dtype="float32")
        if not self.available:
            return None

        cache = get_cache() if use_cache else None
        out, todo = self._from_cache(items, cache)
        if todo:
            vectors = await self._embed([items[i] for i in todo])
            if vectors is None:
                return None
            for slot, vector in zip(todo, vectors, strict=True):
                out[slot] = vector
                if cache is not None:
                    cache.set_vector(key_of(config.EMBED_API_MODEL, items[slot]), vector)
        return np.vstack([np.asarray(v, dtype="float32").reshape(-1) for v in out])

    def _from_cache(self, items: list[str], cache) -> tuple[list, list[int]]:
        out: list = [None] * len(items)
        todo: list[int] = []
        for i, text in enumerate(items):
            hit = cache.get_vector(key_of(config.EMBED_API_MODEL, text)) if cache else None
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit
                self.cache_hits += 1
        return out, todo

    async def _embed(self, texts: list[str]):
        try:
            vectors = await retrieval_api.embed(texts)
        except retrieval_api.RetrievalApiError as exc:
            self.api_failures += 1
            self.last_error = str(exc)[:200]
            log.warning("embedding API unavailable: %s", self.last_error)
            return None
        self.api_calls += 1
        self.encodes += len(texts)
        self.last_error = None
        return vectors

    async def encode_one(self, text: str):
        """One query vector, or None. Cached, so a repeated query costs nothing."""
        result = await self.encode([text])
        return None if result is None else result[0]

    def status(self) -> dict:
        return {
            "model": config.EMBED_API_MODEL,
            "available": self.available,
            "warm": self.warm,
            "api_calls": self.api_calls,
            "api_failures": self.api_failures,
            "encodes": self.encodes,
            "cache_hits": self.cache_hits,
            "last_error": self.last_error,
        }


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER
