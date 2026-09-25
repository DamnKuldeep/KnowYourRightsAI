"""Embedding and reranking over HTTP, through OpenRouter.

* **Embeddings** — ``baai/bge-m3``, the model the corpus was built with. Its vectors were checked
  against the stored ones at cosine 1.0000, so no re-embedding is needed
  (``scripts/verify_embeddings.py`` repeats the check).
* **Reranking** — ``cohere/rerank-v3.5`` by default. A different model has a different score
  scale, so its thresholds are calibrated separately (see ``retrieval/reranker.py``).

Failure is downward, never fatal: callers catch :class:`RetrievalApiError` and fall back to
keyword search or fused ranking, and the turn tells the user it did.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .. import config
from . import spend
from .limiter import PauseCallback, get_limiters

log = logging.getLogger(__name__)


class RetrievalApiError(RuntimeError):
    """The API could not serve this call. Callers degrade; they do not propagate."""


def _retry_after(response: httpx.Response) -> float | None:
    for header in ("retry-after", "x-ratelimit-reset-after"):
        raw = response.headers.get(header)
        if raw:
            try:
                return max(0.5, min(float(raw), 120.0))
            except ValueError:
                continue
    return None


def _backoff_delay(attempt: int) -> float:
    return min(config.RETRY_MAX_DELAY, config.RETRY_INITIAL_DELAY * (attempt + 1))


class _Session:
    """One shared HTTP client for embeddings and reranking.

    Separate from the chat client on purpose: shorter timeouts (a slow rerank should fall back
    rather than hold a turn open) and its own connection pool, so chat traffic cannot starve
    retrieval or the reverse.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self.embed_calls = 0
        self.rerank_calls = 0
        self.cost_usd = 0.0

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.RETRIEVAL_API_TIMEOUT_S, connect=10.0),
                headers={
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": config.OPENROUTER_APP_URL,
                    "X-Title": config.OPENROUTER_APP_NAME,
                },
                limits=httpx.Limits(max_connections=6, max_keepalive_connections=3),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _record(self, path: str, usage: dict | None) -> None:
        try:
            cost = max(0.0, float((usage or {}).get("cost") or 0.0))
        except (TypeError, ValueError):
            cost = 0.0
        self.cost_usd += cost
        if "embeddings" in path:
            self.embed_calls += 1
        else:
            self.rerank_calls += 1
        spend.charge(cost, call=False)

    async def post(self, path: str, payload: dict, *, model: str,
                   on_pause: PauseCallback | None = None, deadline: float | None = None) -> dict:
        """POST with the same pause-and-continue policy the chat client uses."""
        limiter = get_limiters().get(f"retrieval:{model}", config.RETRIEVAL_API_RPM)
        url = f"{config.OPENROUTER_BASE_URL}{path}"
        for attempt in range(config.RETRY_MAX_ATTEMPTS):
            if deadline is not None and time.monotonic() >= deadline:
                raise RetrievalApiError(f"{path}: out of time before the call could be made")
            await limiter.acquire(on_pause)
            try:
                response = await self.client.post(url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= 2:
                    raise RetrievalApiError(f"{path}: {type(exc).__name__}: {exc}") from exc
                await asyncio.sleep(_backoff_delay(attempt))
                continue
            if response.status_code == 200:
                data = response.json()
                self._record(path, data.get("usage"))
                return data
            await self._handle_failure(response, path, model, limiter, attempt, on_pause)
        raise RetrievalApiError(f"{path}: gave up after {config.RETRY_MAX_ATTEMPTS} attempts")

    @staticmethod
    async def _handle_failure(response: httpx.Response, path: str, model: str, limiter,
                              attempt: int, on_pause: PauseCallback | None) -> None:
        """Wait and return to retry, or raise when retrying cannot help."""
        status, body = response.status_code, response.text[:300]
        if status == 429:
            # A rate limit is the provider working as documented: wait it out and continue.
            wait = limiter.penalize(_retry_after(response))
            if on_pause:
                on_pause(model, wait, "rate limit")
            await asyncio.sleep(wait)
            return
        if 500 <= status < 600 and attempt < 3:
            await asyncio.sleep(_backoff_delay(attempt))
            return
        if status in (401, 403):
            raise RetrievalApiError(f"{path}: HTTP {status} — OPENROUTER_API_KEY was rejected")
        if status == 402:
            raise RetrievalApiError(f"{path}: HTTP 402 — the OpenRouter account is out of credit")
        raise RetrievalApiError(f"{path}: HTTP {status}: {body}")

    def stats(self) -> dict:
        return {"embed_calls": self.embed_calls, "rerank_calls": self.rerank_calls,
                "cost_usd": round(self.cost_usd, 6)}


_SESSION: _Session | None = None


def session() -> _Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _Session()
    return _SESSION


def available() -> bool:
    return bool(config.OPENROUTER_API_KEY)


# ── embeddings ────────────────────────────────────────────────────────────────────────
async def embed(texts: list[str], *, on_pause: PauseCallback | None = None,
                deadline: float | None = None):
    """Embed a batch, returning an (n, 1024) L2-normalised float32 array."""
    import numpy as np

    if not available():
        raise RetrievalApiError("no OPENROUTER_API_KEY, so the embedding API is unavailable")
    items = [str(t or "") for t in texts]
    if not items:
        return np.zeros((0, config.EMBED_DIM), dtype="float32")

    vectors: list = []
    size = max(1, config.RETRIEVAL_API_EMBED_BATCH)
    for start in range(0, len(items), size):
        batch = items[start:start + size]
        data = await session().post("/embeddings",
                                    {"model": config.EMBED_API_MODEL, "input": batch},
                                    model=config.EMBED_API_MODEL, on_pause=on_pause,
                                    deadline=deadline)
        rows = data.get("data") or []
        if len(rows) != len(batch):
            raise RetrievalApiError(f"/embeddings returned {len(rows)} vectors for {len(batch)}")
        # Trust the index the API reports over the position in the list.
        vectors.extend(r["embedding"] for r in sorted(rows, key=lambda d: d.get("index", 0)))
    return _normalised(np.asarray(vectors, dtype="float32"))


def _normalised(matrix):
    import numpy as np

    if matrix.ndim != 2 or matrix.shape[1] != config.EMBED_DIM:
        raise RetrievalApiError(
            f"{config.EMBED_API_MODEL} returned vectors of shape {matrix.shape}, but the corpus "
            f"is {config.EMBED_DIM}-dimensional — they are not comparable with it")
    # Cosine-as-dot-product needs unit vectors on both sides; do not assume the API's are.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


# ── reranking ─────────────────────────────────────────────────────────────────────────
async def rerank(query: str, documents: list[str], *, on_pause: PauseCallback | None = None,
                 deadline: float | None = None) -> list[float]:
    """One relevance score per document, in input order.

    The API replies sorted by relevance; scores are scattered back by index, and any document
    it declined to rank gets 0.0.
    """
    if not available():
        raise RetrievalApiError("no OPENROUTER_API_KEY, so the reranking API is unavailable")
    docs = [str(d or "") for d in documents]
    if not docs:
        return []
    payload = {"model": config.RERANK_API_MODEL, "query": query, "documents": docs,
               "top_n": len(docs)}
    data = await session().post("/rerank", payload, model=config.RERANK_API_MODEL,
                                on_pause=on_pause, deadline=deadline)
    scores = [0.0] * len(docs)
    for item in data.get("results") or []:
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(docs):
            try:
                scores[index] = float(item.get("relevance_score") or 0.0)
            except (TypeError, ValueError):
                scores[index] = 0.0
    return scores
