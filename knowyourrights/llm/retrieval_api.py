"""Embedding and reranking over HTTP, through OpenRouter.

Both were local models. Moving them to an API is the single largest latency win available on a
small box: reranking 8 documents on one physical CPU core measured **6,625 ms**, and the same
work over the network measures **~830 ms** — while also giving back the 3.4 GB of RAM the two
models occupied, which is what makes a 1 GB instance viable at all.

Two facts make this safe rather than a gamble:

* **The embedder is the same model.** OpenRouter serves ``baai/bge-m3``, and its vectors were
  verified against the vectors already stored in the corpus: cosine **1.0000** on real rows,
  against 0.60 for unrelated rows. So this is a transport change, not a model change, and no
  re-embedding of the 38,890 chunks is needed. That property is load-bearing — if it ever stops
  holding, retrieval silently degrades, so ``scripts/verify_embeddings.py`` checks it.
* **The reranker is a different model**, so its score distribution is different and its
  thresholds must be recalibrated. The threshold key includes the model name, which is what
  stops a calibration from being reused across them.

Failure is always downward, never fatal: no key, a dead endpoint, a timeout or a spent budget
degrades to the local model if one is loadable and to fused RRF scores if not.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .. import config
from .limiter import PauseCallback, get_limiters

log = logging.getLogger(__name__)


class RetrievalApiError(RuntimeError):
    """The API could not serve this call. Callers degrade; they do not propagate."""


class _Session:
    """One shared HTTP client for embeddings and reranking.

    Separate from the chat client on purpose: these calls have their own timeouts (short — a
    slow rerank should fall back rather than hold a turn) and their own concurrency, so a burst
    of chat traffic cannot starve retrieval or the reverse.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self.embed_calls = 0
        self.rerank_calls = 0
        self.embed_tokens = 0
        self.rerank_units = 0
        self.cost_usd = 0.0

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
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

    def _record(self, usage: dict | None, *, kind: str) -> None:
        usage = usage or {}
        try:
            self.cost_usd += float(usage.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
        if kind == "embed":
            self.embed_tokens += int(usage.get("total_tokens") or 0)
        else:
            self.rerank_units += int(usage.get("search_units") or 0)

    async def post(self, path: str, payload: dict, *, model: str, rpm: int,
                   on_pause: PauseCallback | None = None,
                   deadline: float | None = None) -> dict:
        """POST with the same pause-and-continue policy the chat client uses.

        A 429 here is not an error condition — it is the free tier working as documented — so it
        waits out ``Retry-After`` and continues rather than failing the turn. The deadline is
        what eventually stops it.
        """
        limiter = get_limiters().get(f"retrieval:{model}", rpm)
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
                await asyncio.sleep(min(config.RETRY_MAX_DELAY,
                                        config.RETRY_INITIAL_DELAY * (attempt + 1)))
                continue

            if response.status_code == 200:
                data = response.json()
                self._record(data.get("usage"), kind="embed" if "embeddings" in path else "rerank")
                if "embeddings" in path:
                    self.embed_calls += 1
                else:
                    self.rerank_calls += 1
                return data

            body = response.text[:300]
            if response.status_code == 429:
                # penalize() shrinks the bucket (AIMD) and returns how long to wait, so the
                # backoff and the rate adjustment stay one decision rather than two that can
                # disagree.
                wait = limiter.penalize(_retry_after(response)) or min(
                    config.RETRY_MAX_DELAY,
                    config.RETRY_INITIAL_DELAY * (config.RETRY_MULTIPLIER ** attempt))
                if on_pause:
                    on_pause(model, wait, "rate limit")
                log.info("%s rate-limited, waiting %.0fs", model, wait)
                await asyncio.sleep(wait)
                continue
            if response.status_code in (401, 403):
                raise RetrievalApiError(
                    f"{path}: HTTP {response.status_code} — OPENROUTER_API_KEY was rejected. "
                    f"{body}")
            if response.status_code in (402,):
                raise RetrievalApiError(f"{path}: HTTP 402 — the account is out of credit. {body}")
            if 500 <= response.status_code < 600 and attempt < 3:
                await asyncio.sleep(min(config.RETRY_MAX_DELAY,
                                        config.RETRY_INITIAL_DELAY * (attempt + 1)))
                continue
            raise RetrievalApiError(f"{path}: HTTP {response.status_code}: {body}")

        raise RetrievalApiError(f"{path}: gave up after {config.RETRY_MAX_ATTEMPTS} attempts")

    def stats(self) -> dict:
        return {
            "embed_calls": self.embed_calls, "embed_tokens": self.embed_tokens,
            "rerank_calls": self.rerank_calls, "rerank_units": self.rerank_units,
            "cost_usd": round(self.cost_usd, 6),
        }


def _retry_after(response: httpx.Response) -> float | None:
    for header in ("retry-after", "x-ratelimit-reset-after"):
        raw = response.headers.get(header)
        if raw:
            try:
                return max(0.5, min(float(raw), 120.0))
            except ValueError:
                continue
    return None


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
    """Embed a batch, returning an (n, 1024) L2-normalised float32 array.

    Sent in chunks because a whole corpus re-embedding would otherwise be one enormous request.
    Vectors are re-normalised on arrival: bge-m3 returns unit vectors already, but the stored
    corpus vectors are unit vectors, and dot-product-as-cosine depends on that being true of
    both sides rather than assumed of one.
    """
    import numpy as np

    if not available():
        raise RetrievalApiError("no OPENROUTER_API_KEY, so the embedding API is unavailable")
    items = [str(t or "") for t in texts]
    if not items:
        return np.zeros((0, config.EMBED_DIM), dtype="float32")

    out: list = []
    size = max(1, config.RETRIEVAL_API_EMBED_BATCH)
    for start in range(0, len(items), size):
        batch = items[start:start + size]
        data = await session().post(
            "/embeddings",
            {"model": config.EMBED_API_MODEL, "input": batch},
            model=config.EMBED_API_MODEL, rpm=config.RETRIEVAL_API_RPM,
            on_pause=on_pause, deadline=deadline)
        rows = data.get("data") or []
        if len(rows) != len(batch):
            raise RetrievalApiError(
                f"/embeddings returned {len(rows)} vectors for {len(batch)} inputs")
        # The API is documented to preserve order, but retrieval silently breaks if it ever
        # does not, so trust the index it reports over the position in the list.
        rows = sorted(rows, key=lambda d: d.get("index", 0))
        out.extend(r["embedding"] for r in rows)

    matrix = np.asarray(out, dtype="float32")
    if matrix.shape[1] != config.EMBED_DIM:
        raise RetrievalApiError(
            f"{config.EMBED_API_MODEL} returned {matrix.shape[1]} dimensions, but the corpus is "
            f"{config.EMBED_DIM}-dimensional — these vectors are not comparable with it")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


# ── reranking ─────────────────────────────────────────────────────────────────────────
async def rerank(query: str, documents: list[str], *, top_n: int | None = None,
                 on_pause: PauseCallback | None = None,
                 deadline: float | None = None) -> list[float]:
    """Score each document against the query. Returns one score per document, in input order.

    The API replies sorted by relevance and omits anything beyond ``top_n``; callers here need a
    score per candidate in the order they sent them, so the response is scattered back by index
    and anything unscored gets 0.0 — which is correct: a document the reranker declined to rank
    is one it did not consider relevant.
    """
    if not available():
        raise RetrievalApiError("no OPENROUTER_API_KEY, so the reranking API is unavailable")
    docs = [str(d or "") for d in documents]
    if not docs:
        return []

    payload = {
        "model": config.RERANK_API_MODEL,
        "query": query,
        "documents": docs,
        "top_n": min(len(docs), top_n or len(docs)),
    }
    data = await session().post("/rerank", payload, model=config.RERANK_API_MODEL,
                                rpm=config.RETRIEVAL_API_RPM, on_pause=on_pause,
                                deadline=deadline)

    scores = [0.0] * len(docs)
    for item in data.get("results") or []:
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(docs):
            try:
                scores[index] = float(item.get("relevance_score") or 0.0)
            except (TypeError, ValueError):
                scores[index] = 0.0
    return scores
