"""The search contract: hybrid retrieve -> fuse -> de-duplicate -> rerank -> diversify.

Follows DB README §11, with four deliberate changes:

1. **Multi-query fusion.** Every reformulation contributes its own dense and BM25 ranked list
   and they all enter one RRF pass. A citizen's phrasing and the statute's phrasing rarely
   overlap; asking several ways and fusing is the cheapest fix (an extra query costs ~25 ms
   of GPU, and reranking still happens once).

2. **Rank-preserving de-duplication.** Collapsing to one row per section keeps the *best
   ranked* chunk rather than whichever row pandas happened to see first.

3. **Stored-vector MMR.** Diversity is computed against the vectors that were actually
   indexed, read back from LanceDB in ~25 ms. The notebook re-encoded ``chunk_text``, but the
   index holds vectors of ``embed_text`` — so its diversity was measured in a space that was
   never searched, and it cost a dozen long encodes per query.

4. **A degradation ladder.** No embedder means BM25 alone; no reranker means fused RRF
   ranking with its own threshold. Both are worse, both are honest, neither is an error.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .. import config, legal_terms
from .embedder import get_embedder
from .reranker import get_reranker
from .store import get_store, sql_quote

log = logging.getLogger(__name__)


@dataclass
class Hit:
    """One section, ready to cite."""

    unit_id: str
    chunk_id: str
    citation: str
    act_title: str
    section_label: str = ""
    section_name: str = ""
    category: str = ""
    status: str = ""
    effective_date: str = ""
    act_year: str = ""
    source_type: str = ""
    source_snapshot: str = ""
    full_text: str = ""
    chunk_text: str = ""
    score: float = 0.0
    rrf_score: float = 0.0
    state: str | None = None

    @property
    def is_state_law(self) -> bool:
        return self.state is not None

    @property
    def is_omitted(self) -> bool:
        return (self.status or "").lower() == "omitted"


@dataclass
class SearchResult:
    hits: list[Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    mode: str = "hybrid"          # hybrid | dense_only | fts_only | unavailable
    ranked_by: str = "rerank"     # rerank | rrf
    abstain: bool = True
    top_score: float = 0.0
    candidates: int = 0
    elapsed_ms: int = 0
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.hits)


def rrf(ranked_lists, k: int | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked id lists, optionally weighted.

    Accepts either bare lists or ``(list, weight)`` pairs. Weighting matters when the user
    names a statute: results filtered to that act are far more likely to be right than a free
    semantic match, and equal-weight fusion would let a crowd of loosely-similar sections from
    other acts outvote them.
    """
    k = config.RRF_K if k is None else k
    scores: dict[str, float] = {}
    for entry in ranked_lists:
        ranking, weight = entry if isinstance(entry, tuple) else (entry, 1.0)
        for position, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + position + 1)
    return scores


def mmr_order(vectors, relevance, lam: float | None = None, k: int = 5) -> list[int]:
    """Maximal Marginal Relevance: trade relevance against redundancy.

    Keeps the top-k from being five near-identical clauses of the same section.
    """
    import numpy as np

    lam = config.MMR_LAMBDA if lam is None else lam
    n = len(relevance)
    if n == 0:
        return []
    chosen: list[int] = []
    remaining = list(range(n))
    while remaining and len(chosen) < k:
        if not chosen:
            best = int(np.argmax(relevance))
            chosen.append(best)
            remaining.remove(best)
            continue
        best_value, best_idx = -1e9, remaining[0]
        for i in remaining:
            redundancy = max(float(vectors[i] @ vectors[c]) for c in chosen)
            value = lam * float(relevance[i]) - (1.0 - lam) * redundancy
            if value > best_value:
                best_value, best_idx = value, i
        chosen.append(best_idx)
        remaining.remove(best_idx)
    return chosen


def _is_general_criminal(queries: list[str]) -> bool:
    """Does this look like an ordinary crime/policing question rather than a sectoral one?"""
    blob = " ".join(queries).lower()
    return any(trigger in blob for trigger in config.CRIMINAL_TRIGGERS)


def _rerank_document(row) -> str:
    """What the cross-encoder reads for a candidate section.

    A marginal heading is prepended *only when the corpus actually has one*. Measured
    coverage: every one of the 1,059 criminal-code sections carries a heading, and none of
    the 34,111 Constitution/central-act sections do. Prepending a bare citation line to the
    latter is pure noise — it measurably pushed the correct Article 22 out of the top-5 for
    an arrest query — while the real headings lift the sanhitas sharply ("Cheating" is what
    makes BNS §318 findable from the word "cheating").
    """
    heading = _clean(getattr(row, "section_name", ""))
    body = _clean(getattr(row, "chunk_text", ""))
    return f"{heading}\n{body}" if heading else body


def _clean(value) -> str:
    """NaN / NaT / 'nan' all become an empty string so they never reach a prompt or the UI."""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (TypeError, ValueError, ImportError):
        pass
    text = str(value).strip()
    return "" if text.lower() in ("", "nan", "none", "nat", "<na>") else text


class SearchEngine:
    def __init__(self) -> None:
        self.store = get_store()
        self.embedder = get_embedder()
        self.reranker = get_reranker()

    async def warmup(self) -> dict:
        """Load and warm both models plus the section index, at startup."""
        results = await asyncio.gather(
            self.embedder.warmup(), self.reranker.warmup(), return_exceptions=True,
        )
        try:
            stats = self.store.stats()
        except Exception as exc:
            log.error("could not open the legal database: %s", exc)
            stats = {}
        return {
            "embedder": self.embedder.status(),
            "reranker": self.reranker.status(),
            "store": stats,
            "errors": [str(r) for r in results if isinstance(r, Exception)],
        }

    # ── candidate generation ─────────────────────────────────────────────────────────
    async def _ranked_lists(self, queries: list[str], fetch: int,
                            act_filter: list[str]) -> tuple[list, str, list[str]]:
        notes: list[str] = []

        vectors = await self.embedder.encode(queries)
        if vectors is None:
            reason = self.embedder.unavailable_reason or "embedder unavailable"
            notes.append(f"Semantic search is off ({reason}); using keyword search only.")

        loop = asyncio.get_running_loop()

        def dense_for(vector, where=None, k=fetch):
            return self.store.dense(vector, k, where)

        def fts_for(text, where=None, k=fetch):
            return self.store.fts(text, k, where)

        weights: list[float] = []
        tasks = []
        for i, query in enumerate(queries):
            if vectors is not None:
                tasks.append(loop.run_in_executor(None, dense_for, vectors[i], None, fetch))
                weights.append(1.0)
            tasks.append(loop.run_in_executor(None, fts_for, query, None, fetch))
            weights.append(1.0)

        # A named act contributes extra *filtered* lists, weighted up. Still additive rather
        # than a hard WHERE, so a wrong act guess degrades the ranking instead of erasing the
        # real answer — but heavy enough that a crowd of loosely-similar sections from
        # unrelated statutes cannot outvote the act the user actually asked about.
        for title in act_filter[:2]:
            where = f"act_title = {sql_quote(title)}"
            if vectors is not None:
                tasks.append(loop.run_in_executor(None, dense_for, vectors[0], where, fetch))
                weights.append(config.ACT_FILTER_WEIGHT)
            tasks.append(loop.run_in_executor(None, fts_for, queries[0], where, fetch))
            weights.append(config.ACT_FILTER_WEIGHT)

        # No Act named, but plainly a criminal/policing question: lift the general codes so a
        # forest officer's power of arrest cannot outrank the procedure code that actually
        # governs the person asking.
        if not act_filter and _is_general_criminal(queries):
            titles = ",".join(sql_quote(t) for t in config.GENERAL_CODES)
            where = f"act_title IN ({titles})"
            if vectors is not None:
                tasks.append(loop.run_in_executor(None, dense_for, vectors[0], where, fetch))
                weights.append(config.GENERAL_CODE_WEIGHT)
            tasks.append(loop.run_in_executor(None, fts_for, queries[0], where, fetch))
            weights.append(config.GENERAL_CODE_WEIGHT)

        lists: list[tuple[list[str], float]] = []
        for result, weight in zip(await asyncio.gather(*tasks, return_exceptions=True), weights):
            if isinstance(result, Exception):
                log.debug("a ranked list failed: %s", result)
                continue
            if result:
                lists.append((result, weight))

        if vectors is None:
            mode = "fts_only"
        elif not self.store._fts_available:
            mode = "dense_only"
            notes.append("Keyword search is unavailable; using semantic search only.")
        else:
            mode = "hybrid"
        return lists, mode, notes

    # ── the main entry point ─────────────────────────────────────────────────────────
    async def search(
        self,
        queries,
        *,
        top_k: int | None = None,
        fetch: int | None = None,
        rerank_with: str | None = None,
        deadline: float | None = None,
        on_pause=None,
        session: str = "",
    ) -> SearchResult:
        started = time.monotonic()
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in queries if q and q.strip()][:6]
        if not queries:
            return SearchResult(mode="unavailable", notes=["No search query was produced."])

        top_k = max(1, min(config.TOPK_MAX, top_k or config.TOP_K))
        fetch = fetch or config.FETCH_K
        acts = []
        for query in queries:
            for title in legal_terms.detect_acts(query):
                if title not in acts:
                    acts.append(title)

        lists, mode, notes = await self._ranked_lists(queries, fetch, acts)
        if not lists:
            return SearchResult(queries=queries, mode="unavailable", abstain=True,
                                notes=notes + ["Retrieval returned nothing at all."],
                                elapsed_ms=int((time.monotonic() - started) * 1000))

        fused = rrf(lists)
        ordered_ids = sorted(fused, key=fused.get, reverse=True)[:fetch * 2]

        rows = self.store.rows(ordered_ids)
        if rows.empty:
            return SearchResult(queries=queries, mode=mode, abstain=True,
                                notes=notes + ["No matching sections were found."],
                                elapsed_ms=int((time.monotonic() - started) * 1000))

        # De-duplicate to one chunk per section, keeping the best-ranked chunk of each.
        by_chunk = {row.chunk_id: row for row in rows.itertuples()}
        best_per_unit: dict[str, object] = {}
        for chunk_id in ordered_ids:
            row = by_chunk.get(chunk_id)
            if row is None:
                continue
            best_per_unit.setdefault(row.unit_id, row)

        candidates = list(best_per_unit.values())[:config.RERANK_POOL]
        if not candidates:
            return SearchResult(queries=queries, mode=mode, abstain=True, notes=notes,
                                elapsed_ms=int((time.monotonic() - started) * 1000))

        # Rerank against the *natural* phrasing, not an expanded one. Acronym expansion is a
        # retrieval aid — it turns "file an RTI" into "file an Right to Information Act, 2005",
        # which helps BM25 and the bi-encoder but reads as broken English to a cross-encoder
        # trained on natural queries.
        rerank_query = rerank_with or queries[0]
        scores = await self.reranker.score(
            rerank_query, [_rerank_document(c) for c in candidates],
            deadline=deadline, on_pause=on_pause, session=session,
        )
        if scores is None:
            ranked_by = "rrf"
            top_rrf = max((fused.get(c.chunk_id, 0.0) for c in candidates), default=1.0) or 1.0
            scores = [fused.get(c.chunk_id, 0.0) / top_rrf for c in candidates]
            notes.append("Reranking was unavailable; results are ordered by keyword/semantic "
                         "fusion, which is less precise.")
        else:
            ranked_by = "rerank"

        # A cross-encoder cannot separate "power to arrest without warrant" in the Essential
        # Services Maintenance Act from the same words in the procedure code — measured at
        # 0.994 against 0.992. The distinction is not textual, it is about which statute
        # governs the person asking, so it has to be applied here rather than hoped for from
        # the model. Ordering only: the score shown to the user stays the honest one.
        prefer_general = not acts and _is_general_criminal(queries)

        def boosted(candidate, score: float) -> float:
            if prefer_general and _clean(getattr(candidate, "act_title", "")) in config.GENERAL_CODES:
                return score + config.GENERAL_CODE_BOOST
            return score

        # Carry (candidate, reported_score, ranking_score) together. The boost has to reach
        # MMR too — ranking with it and then diversifying on the raw scores simply undoes it.
        ranked = [(c, s, boosted(c, s)) for c, s in zip(candidates, scores)]
        ranked.sort(key=lambda row: row[2], reverse=True)
        shortlist = [(c, s) for c, s, _ in ranked[:max(top_k * 2, 12)]]
        rank_scores = [r for _, _, r in ranked[:max(top_k * 2, 12)]]

        # When the user named a statute, spreading results across *different* acts is the
        # opposite of what they want — they asked about one Act, so several of its sections
        # is the right answer. Lean towards relevance in that case.
        lam = config.MMR_LAMBDA_FOCUSED if acts else config.MMR_LAMBDA
        order = await self._diversify(shortlist, top_k, lam, rank_scores)
        hits = [self._to_hit(shortlist[i][0], shortlist[i][1], fused) for i in order]

        thresholds = self.reranker.thresholds
        cutoff = thresholds.low if ranked_by == "rerank" else 0.15
        top_score = hits[0].score if hits else 0.0

        return SearchResult(
            hits=hits, queries=queries, mode=mode, ranked_by=ranked_by,
            abstain=top_score < cutoff, top_score=round(top_score, 4),
            candidates=len(candidates), notes=notes,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    async def _diversify(self, shortlist, top_k: int, lam: float | None = None,
                         rank_scores: list[float] | None = None) -> list[int]:
        """MMR over the *stored* vectors — the ones retrieval actually searched."""
        import numpy as np

        if len(shortlist) <= top_k:
            return list(range(len(shortlist)))

        chunk_ids = [c.chunk_id for c, _ in shortlist]
        loop = asyncio.get_running_loop()
        try:
            vector_map = await loop.run_in_executor(None, self.store.vectors, chunk_ids)
        except Exception as exc:
            log.debug("stored-vector fetch failed, skipping MMR: %s", exc)
            vector_map = {}

        if len(vector_map) < len(chunk_ids):
            # Without every vector, MMR's redundancy term is meaningless — take pure relevance.
            return list(range(min(top_k, len(shortlist))))

        vectors = np.vstack([vector_map[cid] for cid in chunk_ids])
        relevance = np.array(rank_scores if rank_scores is not None
                             else [s for _, s in shortlist], dtype="float32")
        return mmr_order(vectors, relevance, lam=lam, k=top_k)

    @staticmethod
    def _to_hit(row, score: float, fused: dict[str, float]) -> Hit:
        act_title = _clean(getattr(row, "act_title", ""))
        return Hit(
            unit_id=_clean(row.unit_id),
            chunk_id=_clean(row.chunk_id),
            citation=_clean(getattr(row, "citation", "")),
            act_title=act_title,
            section_label=_clean(getattr(row, "section_label", "")),
            section_name=_clean(getattr(row, "section_name", "")),
            category=_clean(getattr(row, "category", "")),
            status=_clean(getattr(row, "status", "")),
            effective_date=_clean(getattr(row, "effective_date", "")),
            act_year=_clean(getattr(row, "act_year", "")),
            source_type=_clean(getattr(row, "source_type", "")),
            source_snapshot=_clean(getattr(row, "source_snapshot", "")),
            full_text=_clean(getattr(row, "full_text", "")),
            chunk_text=_clean(getattr(row, "chunk_text", "")),
            score=round(float(score), 4),
            rrf_score=round(float(fused.get(row.chunk_id, 0.0)), 6),
            # Trust the title, never the `jurisdiction` column (DB README §9).
            state=legal_terms.is_state_law(act_title, config.STATE_PREFIXES),
        )

    # ── exact lookup ─────────────────────────────────────────────────────────────────
    def lookup(self, act: str | None, section_label: str | None,
               constitution: bool = False) -> list[Hit]:
        """Exact provision text. No embedding, no reranking — this is a fact, not a guess."""
        df = self.store.lookup(act, section_label, constitution)
        hits: list[Hit] = []
        for row in df.itertuples():
            hit = self._to_hit(row, 1.0, {})
            hit.score = 1.0
            hits.append(hit)
        # A long section is several chunks; they share one citation, so keep the first.
        seen: set[str] = set()
        unique = []
        for hit in hits:
            if hit.unit_id in seen:
                continue
            seen.add(hit.unit_id)
            unique.append(hit)
        return unique

    def status(self) -> dict:
        return {
            "embedder": self.embedder.status(),
            "reranker": self.reranker.status(),
            "store": self.store.stats(),
        }


_ENGINE: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SearchEngine()
    return _ENGINE
