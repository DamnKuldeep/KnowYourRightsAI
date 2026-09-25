"""Cross-encoder reranking over OpenRouter, and the thresholds that go with it.

A reranker's scores live on its own scale, so the abstention and citation cut-offs are
calibrated per ranking method (``scripts/calibrate.py``) and stored in
``.runtime/thresholds.json`` under a key naming exactly what produced the scores. When the API is
unavailable, :meth:`Reranker.score` returns None and records why; search then ranks on fused
keyword/semantic scores under that method's own calibrated threshold.
"""

from __future__ import annotations

import json
import logging
import time

from .. import config
from ..llm import retrieval_api

log = logging.getLogger(__name__)


class Thresholds:
    """Abstention and citation cut-offs for one ranking method."""

    def __init__(self, low: float, cite: float, source: str = "config") -> None:
        self.low = low
        self.cite = cite
        self.source = source

    def __repr__(self) -> str:
        return f"Thresholds(low={self.low:.3f}, cite={self.cite:.3f}, from={self.source!r})"


def load_thresholds(method: str) -> Thresholds:
    """Calibrated values for this ranking method, or the configured defaults.

    A local calibration (``.runtime/thresholds.json``) wins over the one shipped with the
    package. The shipped file matters: ``.runtime`` is never committed, and without it every
    fresh deployment ran on uncalibrated defaults that silently dropped valid citations.
    """
    for path, origin in ((config.THRESHOLDS_FILE, "calibrated"),
                         (config.PACKAGED_THRESHOLDS, "shipped calibration")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8")).get(method)
            if entry:
                return Thresholds(float(entry["low"]), float(entry["cite"]),
                                  f"{origin} {entry.get('calibrated_at', '')}".strip())
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return Thresholds(config.LOW_SCORE, config.CITE_MIN_SCORE, "config default (uncalibrated)")


def save_thresholds(method: str, low: float, cite: float, extra: dict | None = None) -> None:
    config.ensure_runtime_dirs()
    try:
        data = json.loads(config.THRESHOLDS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data[method] = {"low": round(low, 4), "cite": round(cite, 4),
                    "calibrated_at": time.strftime("%Y-%m-%d"), **(extra or {})}
    config.THRESHOLDS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def rerank_method() -> str:
    """The threshold key for cross-encoder scores.

    What the model is shown moves its scores, so the document format is part of the key: adding
    each section's citizen questions changed the score distribution enough to need its own cut.
    """
    return "api:" + config.RERANK_API_MODEL + ("+q" if config.RERANK_WITH_QUESTIONS else "")


# Fused scores are not probabilities, so they get their own calibration. Which retrievers feed
# the fusion changes the distribution: with no embeddings only BM25 lists are fused.
FUSION_METHOD = "fusion"
FUSION_BM25_METHOD = "fusion-bm25"


class Reranker:
    def __init__(self) -> None:
        self.calls = 0
        self.docs_scored = 0
        self.failures = 0
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return retrieval_api.available()

    @property
    def unavailable_reason(self) -> str | None:
        if not self.available:
            return "no OPENROUTER_API_KEY is configured"
        return self.last_error

    async def score(self, query: str, documents, *, deadline: float | None = None,
                    on_pause=None) -> list[float] | None:
        """Relevance in [0, 1] per document, in order, or None if the reranker is unavailable."""
        docs = [str(d or "") for d in documents]
        if not docs:
            return []
        if not self.available:
            return None
        try:
            scores = await retrieval_api.rerank(query, docs, on_pause=on_pause, deadline=deadline)
        except retrieval_api.RetrievalApiError as exc:
            self.failures += 1
            self.last_error = str(exc)[:200]
            log.warning("reranking API unavailable (%s) — ranking on fused scores", self.last_error)
            return None
        self.calls += 1
        self.docs_scored += len(docs)
        self.last_error = None
        return scores

    def status(self) -> dict:
        thresholds = load_thresholds(rerank_method())
        return {
            "model": config.RERANK_API_MODEL,
            "available": self.available,
            "calls": self.calls,
            "docs_scored": self.docs_scored,
            "failures": self.failures,
            "last_error": self.last_error,
            "low_score": thresholds.low,
            "cite_min_score": thresholds.cite,
            "thresholds_source": thresholds.source,
        }


_RERANKER: Reranker | None = None


def get_reranker() -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER
