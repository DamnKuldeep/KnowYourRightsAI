"""Cross-encoder reranking, local first, NIM as relief.

Unlike the embedder, the reranker is genuinely swappable — it reads text and never touches the
corpus vectors — so it can move between backends freely. What is *not* portable is the score:

* the local cross-encoder emits one logit that we squash with sigmoid into [0, 1];
* NIM emits raw logits on a much wider scale (probe measured −15.9 … +6.8).

Both are mapped through sigmoid here so callers see a single [0, 1] convention, but the two
distributions are still shaped differently. That is why ``scripts/calibrate.py`` derives
thresholds per backend and writes them to ``.runtime/thresholds.json`` — reusing the notebook's
``LOW_SCORE=0.05`` across a different reranker would silently admit or drop citations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time

from .. import config
from ..runtime import gpu, resources

log = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


class Thresholds:
    """Abstention and citation cut-offs for whichever reranker is actually in use."""

    def __init__(self, low: float, cite: float, source: str = "config") -> None:
        self.low = low
        self.cite = cite
        self.source = source

    def __repr__(self) -> str:
        return f"Thresholds(low={self.low:.3f}, cite={self.cite:.3f}, from={self.source!r})"


def load_thresholds(backend: str, model: str | None) -> Thresholds:
    """Calibrated values if we have them for this exact backend+model, else config defaults."""
    key = f"{backend}:{model or '-'}"
    try:
        data = json.loads(config.THRESHOLDS_FILE.read_text(encoding="utf-8"))
        entry = data.get(key)
        if entry:
            return Thresholds(float(entry["low"]), float(entry["cite"]),
                              f"calibrated {entry.get('calibrated_at', '')}".strip())
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return Thresholds(config.LOW_SCORE, config.CITE_MIN_SCORE, "config default (uncalibrated)")


def save_thresholds(backend: str, model: str | None, low: float, cite: float,
                    extra: dict | None = None) -> None:
    config.ensure_runtime_dirs()
    key = f"{backend}:{model or '-'}"
    try:
        data = json.loads(config.THRESHOLDS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data[key] = {"low": round(low, 4), "cite": round(cite, 4),
                 "calibrated_at": time.strftime("%Y-%m-%d"), **(extra or {})}
    config.THRESHOLDS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


class Reranker:
    """Local cross-encoder with automatic degradation to NIM, then to nothing."""

    def __init__(self, plan: resources.ResourcePlan | None = None) -> None:
        self._plan = plan
        self._tokenizer = None
        self._model = None
        self._max_len: int | None = None
        self._load_lock = asyncio.Lock()
        self._local_failed: str | None = None
        self._nim_failed: str | None = None
        self._warm = False
        self.calls = 0
        self.docs_scored = 0

    @property
    def plan(self) -> resources.ResourcePlan:
        if self._plan is None:
            self._plan = resources.get_plan()
        return self._plan

    @property
    def backend(self) -> str:
        """What we will actually use on the next call."""
        want = self.plan.rerank_backend
        if want == "local" and self._local_failed is None:
            return "local"
        if want in ("local", "nim") and self._nim_failed is None and config.NVIDIA_API_KEY:
            return "nim"
        return "none"

    @property
    def model_name(self) -> str | None:
        if self.backend == "none":
            return "rrf"          # the fusion scores are the ranking signal, and get their own
        if self.backend == "local":
            return self.plan.rerank_model
        if self.backend == "nim":
            from ..llm import registry

            return registry.resolve("rerank")
        return None

    @property
    def thresholds(self) -> Thresholds:
        return load_thresholds(self.backend, self.model_name)

    # ── local model ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _position_limit(model) -> int:
        """Cap at the *reranker's* own context, not the embedder's.

        RoBERTa-family models reserve two position slots, and exceeding the limit throws a
        CUDA index error rather than a clean Python one — so this must be right.
        """
        cfg = model.config
        limit = int(getattr(cfg, "max_position_embeddings", 512) or 512)
        if getattr(cfg, "model_type", "") in ("xlm-roberta", "roberta", "camembert"):
            limit -= 2
        return max(8, min(limit, config.EMBED_MAX_SEQ))

    def _load_sync(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        plan = self.plan
        name = plan.rerank_model
        started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                name, trust_remote_code=True, dtype=getattr(torch, plan.rerank_dtype))
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                name, trust_remote_code=True, torch_dtype=getattr(torch, plan.rerank_dtype))
        model = model.to(plan.rerank_device).eval()
        log.info("reranker loaded: %s on %s/%s in %.1fs",
                 name, plan.rerank_device, plan.rerank_dtype, time.time() - started)
        return tokenizer, model

    async def _ensure_local(self) -> bool:
        if self._model is not None:
            return True
        if self._local_failed is not None or not self.plan.rerank_model:
            return False
        async with self._load_lock:
            if self._model is not None:
                return True
            if self._local_failed is not None:
                return False
            try:
                self._tokenizer, self._model = await gpu.get_executor().run(self._load_sync)
                self._max_len = self._position_limit(self._model)
                gpu.get_executor().register_evict_hook(self._unload_sync)
                return True
            except Exception as exc:
                self._local_failed = f"{type(exc).__name__}: {exc}"
                log.warning("local reranker unavailable (%s) — falling back to %s",
                            self._local_failed, "NIM" if config.NVIDIA_API_KEY else "RRF scores")
                gpu.empty_cache()
                return False

    def _unload_sync(self) -> None:
        if self._model is not None:
            log.info("evicting reranker from %s", self.plan.rerank_device)
            self._tokenizer = self._model = None
            self._warm = False

    def _score_sync(self, pairs):
        import torch

        encoded = self._tokenizer(list(pairs), padding=True, truncation=True,
                                  max_length=self._max_len, return_tensors="pt")
        # XLM-R / RoBERTa rerankers have no segment ids; passing them is an error.
        encoded.pop("token_type_ids", None)
        encoded = {k: v.to(self.plan.rerank_device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded).logits
        if logits.ndim == 2 and logits.shape[1] > 1:
            # Two-class head: the positive class is the relevance score.
            return torch.softmax(logits.float(), dim=1)[:, -1].cpu().tolist()
        return torch.sigmoid(logits.float().view(-1)).cpu().tolist()

    # ── public API ───────────────────────────────────────────────────────────────────
    async def score(self, query: str, documents, *, deadline: float | None = None,
                    on_pause=None, session: str = "") -> list[float] | None:
        """Relevance in [0, 1] per document, or None if no reranker is available.

        None is a meaningful answer: the caller falls back to fused RRF ranking with its own
        threshold rather than pretending it has cross-encoder quality.
        """
        docs = [str(d or "") for d in documents]
        if not docs:
            return []

        if self.plan.rerank_backend == "none":
            # The profile says rank on fusion scores alone. Reaching for a remote reranker here
            # would spend API calls the operator explicitly opted out of.
            return None

        if self.plan.rerank_backend == "local" and await self._ensure_local():
            pairs = [[query, d] for d in docs]
            try:
                scores = await gpu.get_executor().map_batches(
                    self._score_sync, pairs, self.plan.rerank_batch, label="rerank")
                self.calls += 1
                self.docs_scored += len(docs)
                return [float(s) for s in scores]
            except gpu.GpuOutOfMemory as exc:
                log.warning("reranker out of memory (%s) — trying NIM", exc)
            except Exception as exc:
                log.warning("local reranking failed (%s) — trying NIM", exc)

        if self._nim_failed is None and config.NVIDIA_API_KEY:
            try:
                from ..llm.client import get_client

                logits = await get_client().rerank(query, docs, deadline=deadline,
                                                   on_pause=on_pause, session=session)
                self.calls += 1
                self.docs_scored += len(docs)
                # Map NIM's wide logits onto the same [0,1] convention as the local head.
                return [sigmoid(x) if x != float("-inf") else 0.0 for x in logits]
            except Exception as exc:
                self._nim_failed = f"{type(exc).__name__}: {exc}"
                log.warning("NIM reranking unavailable (%s) — ranking will use RRF scores",
                            self._nim_failed)

        return None

    async def warmup(self) -> bool:
        if self.plan.rerank_backend != "local":
            return True
        if not await self._ensure_local():
            return False
        if self._warm:
            return True
        started = time.time()
        await self.score("warmup query", ["a short document about warming up"])
        self._warm = True
        log.info("reranker warm in %.2fs", time.time() - started)
        return True

    def status(self) -> dict:
        thresholds = self.thresholds
        return {
            "backend": self.backend,
            "model": self.model_name,
            "device": self.plan.rerank_device if self.backend == "local" else "remote",
            "calls": self.calls,
            "docs_scored": self.docs_scored,
            "low_score": thresholds.low,
            "cite_min_score": thresholds.cite,
            "thresholds_source": thresholds.source,
            "local_error": self._local_failed,
            "nim_error": self._nim_failed,
        }


_RERANKER: Reranker | None = None


def get_reranker() -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER
