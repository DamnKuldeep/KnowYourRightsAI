"""The search contract: hybrid retrieve -> fuse -> de-duplicate -> rerank -> diversify.

Four decisions shape it:

1. **Multi-query fusion.** Every phrasing contributes its own dense and BM25 ranked list and all
   of them enter one RRF pass. A citizen's wording and a statute's rarely overlap, and asking
   several ways is the cheapest fix. Reranking still happens once.
2. **Rank-preserving de-duplication.** One row per section, keeping its best-ranked chunk.
3. **Stored-vector MMR.** Diversity is computed against the vectors that were actually indexed,
   read back from LanceDB, not against re-encoded text in a space that was never searched.
4. **A degradation ladder.** No embeddings means BM25 alone; no reranker means ordering by fused
   scores with their own calibrated threshold. Both are worse, both are honest, and both are
   reported to the user through :attr:`SearchResult.degraded`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .. import config, legal_terms
from .embedder import get_embedder
from .ranking import clean, is_general_criminal, mmr_order, rerank_document, rrf
from .reranker import (
    FUSION_BM25_METHOD, FUSION_METHOD, get_reranker, load_thresholds, rerank_method,
)
from .store import get_store, sql_quote

log = logging.getLogger(__name__)

# What a reader is told when retrieval ran in a reduced mode.
NOTE_NO_EMBEDDINGS = ("Semantic search is unavailable right now, so this answer is based on "
                      "keyword search only and may miss relevant sections.")
NOTE_NO_RERANK = ("The relevance ranker is unavailable right now, so sources were ordered by a "
                  "simpler method and may be less precise.")
NOTE_NO_KEYWORDS = "Keyword search is unavailable; sources come from semantic search only."

MAX_QUERIES = 6
MAX_TOP_K = 12


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
    # The place this law is limited to, or None if it applies across India. Filled for state
    # Acts and for Acts Parliament passed for a Union Territory: both are territorially limited.
    state: str | None = None
    union_territory: bool = False    # True when Parliament enacted it for that territory

    @property
    def is_territorial(self) -> bool:
        return self.state is not None

    @property
    def is_state_law(self) -> bool:
        """Passed by a state legislature. Narrower than :attr:`is_territorial`."""
        return self.state is not None and not self.union_territory

    @property
    def is_omitted(self) -> bool:
        return (self.status or "").lower() == "omitted"


def extent(act_title: str) -> dict:
    """Where an Act applies, read from its title: a state, a Union Territory, or everywhere."""
    state = legal_terms.is_state_law(act_title, config.STATE_PREFIXES)
    if state:
        return {"state": state, "union_territory": False}
    territory = legal_terms.territory_of(act_title, config.TERRITORY_PREFIXES)
    if territory:
        return {"state": territory, "union_territory": True}
    return {"state": None, "union_territory": False}


@dataclass
class SearchResult:
    hits: list[Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    mode: str = "hybrid"          # hybrid | dense_only | fts_only | unavailable
    ranked_by: str = "rerank"     # rerank | fusion
    abstain: bool = True
    top_score: float = 0.0
    cite_floor: float = 0.0       # below this score a section is never cited
    candidates: int = 0
    elapsed_ms: int = 0
    notes: list[str] = field(default_factory=list)      # diagnostics, for logs and scripts
    degraded: list[str] = field(default_factory=list)   # reader-facing, see NOTE_* above

    def __bool__(self) -> bool:
        return bool(self.hits)


@dataclass
class _Ranked:
    """Candidate lists fused from every ranked list, before scoring."""

    lists: list[tuple[list[str], float]]
    mode: str
    bm25: dict[str, float]
    degraded: list[str]

    @property
    def total_weight(self) -> float:
        return sum(weight for _, weight in self.lists)


class SearchEngine:
    def __init__(self) -> None:
        self.store = get_store()
        self.embedder = get_embedder()
        self.reranker = get_reranker()

    async def warmup(self) -> dict:
        """Open the database and prove the embedding API answers, at startup."""
        errors: list[str] = []
        try:
            stats = await asyncio.to_thread(self.store.stats)
        except Exception as exc:
            log.error("could not open the legal database: %s", exc)
            stats, errors = {}, [f"database: {exc}"]
        await self.embedder.warmup()
        return {"store": stats, "embedder": self.embedder.status(),
                "reranker": self.reranker.status(), "errors": errors}

    # ── the main entry point ─────────────────────────────────────────────────────────
    async def search(self, queries, *, top_k: int | None = None, fetch: int | None = None,
                     rerank_with: str | None = None, deadline: float | None = None,
                     on_pause=None) -> SearchResult:
        started = time.monotonic()
        queries = _clean_queries(queries)
        if not queries:
            return SearchResult(mode="unavailable", notes=["No search query was produced."])
        top_k = max(1, min(MAX_TOP_K, top_k or config.TOP_K))
        fetch = fetch or config.FETCH_K
        acts = _named_acts(queries)

        ranked = await self._ranked_lists(queries, fetch, acts)
        candidates, fused = self._candidates(ranked, fetch)
        result = SearchResult(queries=queries, mode=ranked.mode, degraded=list(ranked.degraded),
                              candidates=len(candidates))
        if candidates:
            rerank_query = legal_terms.annotate(rerank_with or queries[0])
            scores, method = await self._score(rerank_query, candidates, ranked, fused,
                                               deadline, on_pause, result)
            result.hits = await self._order(candidates, scores, queries, acts, top_k,
                                            result.ranked_by)
            thresholds = load_thresholds(method)
            result.top_score = round(result.hits[0].score if result.hits else 0.0, 4)
            result.abstain = result.top_score < thresholds.low
            result.cite_floor = thresholds.cite
        else:
            result.notes.append("No matching sections were found.")
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        return result

    # ── candidate generation ─────────────────────────────────────────────────────────
    async def _ranked_lists(self, queries: list[str], fetch: int, acts: list[str]) -> _Ranked:
        bm25: dict[str, float] = {}
        vectors = await self.embedder.encode(queries)
        degraded = [] if vectors is not None else [NOTE_NO_EMBEDDINGS]
        jobs = _list_jobs(queries, vectors, acts)

        def run(kind: str, payload, where: str | None) -> list[str]:
            if kind == "dense":
                return self.store.dense(payload, fetch, where)
            return self.store.fts(payload, fetch, where, scores=bm25)

        results = await asyncio.gather(
            *(asyncio.to_thread(run, kind, payload, where) for kind, payload, where, _ in jobs),
            return_exceptions=True)
        lists: list[tuple[list[str], float]] = []
        for result, (_, _, _, weight) in zip(results, jobs, strict=True):
            if isinstance(result, Exception):
                log.debug("a ranked list failed: %s", result)
            elif result:
                lists.append((result, weight))

        mode = "hybrid"
        if vectors is None:
            mode = "fts_only"
        elif not self.store.fts_available:
            mode = "dense_only"
            degraded.append(NOTE_NO_KEYWORDS)
        return _Ranked(lists, mode, bm25, degraded)

    def _candidates(self, ranked: _Ranked, fetch: int) -> tuple[list, dict[str, float]]:
        """One row per section, best-ranked chunk first, capped at the rerank pool."""
        if not ranked.lists:
            return [], {}
        fused = rrf(ranked.lists)
        ordered_ids = sorted(fused, key=fused.get, reverse=True)[:fetch * 2]
        rows = self.store.rows(ordered_ids)
        by_chunk = {row.chunk_id: row for row in rows.itertuples()}
        best_per_unit: dict[str, object] = {}
        for chunk_id in ordered_ids:
            row = by_chunk.get(chunk_id)
            if row is not None:
                best_per_unit.setdefault(row.unit_id, row)
        return list(best_per_unit.values())[:config.RERANK_POOL], fused

    # ── scoring ──────────────────────────────────────────────────────────────────────
    async def _score(self, query: str, candidates: list, ranked: _Ranked,
                     fused: dict[str, float], deadline, on_pause,
                     result: SearchResult) -> tuple[list[float], str]:
        """Cross-encoder scores, or fused scores when the reranker is unavailable."""
        # Reranked against the natural phrasing: acronym expansion helps BM25 and the
        # bi-encoder but reads as broken English to a cross-encoder.
        scores = await self.reranker.score(query, [rerank_document(c) for c in candidates],
                                           deadline=deadline, on_pause=on_pause)
        if scores is not None:
            result.ranked_by = "rerank"
            return scores, rerank_method()
        result.ranked_by = "fusion"
        result.degraded.append(NOTE_NO_RERANK)
        method = FUSION_METHOD if ranked.mode != "fts_only" else FUSION_BM25_METHOD
        return _fusion_scores(candidates, ranked, fused), method

    async def _order(self, candidates: list, scores: list[float], queries: list[str],
                     acts: list[str], top_k: int, ranked_by: str) -> list[Hit]:
        ranked = _with_general_code_boost(candidates, scores, queries, acts)
        pool = max(top_k * 2, 12)
        shortlist = [(c, s) for c, s, _ in ranked[:pool]]
        rank_scores = [r for _, _, r in ranked[:pool]]
        # Diversity is paid for out of relevance. It is worth it with a cross-encoder ordering the
        # candidates; on fused scores the base order is weaker and spreading it costs recall. When
        # a statute is named, several of its sections is the right answer, so lean to relevance.
        if ranked_by != "rerank":
            lam = config.MMR_LAMBDA_NO_RERANK
        else:
            lam = config.MMR_LAMBDA_FOCUSED if acts else config.MMR_LAMBDA
        order = await self._diversify(shortlist, top_k, lam, rank_scores)
        return [_to_hit(shortlist[i][0], shortlist[i][1]) for i in order]

    async def _diversify(self, shortlist, top_k: int, lam: float,
                         rank_scores: list[float]) -> list[int]:
        """MMR over the stored vectors, the ones retrieval actually searched."""
        import numpy as np

        if len(shortlist) <= top_k:
            return list(range(len(shortlist)))
        chunk_ids = [c.chunk_id for c, _ in shortlist]
        try:
            vector_map = await asyncio.to_thread(self.store.vectors, chunk_ids)
        except Exception as exc:
            log.debug("stored-vector fetch failed, skipping MMR: %s", exc)
            vector_map = {}
        if len(vector_map) < len(chunk_ids):
            # Without every vector the redundancy term is meaningless: take pure relevance.
            return list(range(top_k))
        vectors = np.vstack([vector_map[cid] for cid in chunk_ids])
        return mmr_order(vectors, np.array(rank_scores, dtype="float32"), lam=lam, k=top_k)

    # ── exact lookup ─────────────────────────────────────────────────────────────────
    def lookup(self, act: str | None, section_label: str | None,
               constitution: bool = False) -> list[Hit]:
        """Exact provision text. No embedding, no reranking: this is a fact, not a guess."""
        hits: list[Hit] = []
        seen: set[str] = set()
        for row in self.store.lookup(act, section_label, constitution).itertuples():
            hit = _to_hit(row, 1.0)
            # A long section is several chunks sharing one citation; keep the first.
            if hit.unit_id not in seen:
                seen.add(hit.unit_id)
                hits.append(hit)
        return hits

    def status(self) -> dict:
        return {"embedder": self.embedder.status(), "reranker": self.reranker.status(),
                "store": self.store.stats()}


# ── helpers ───────────────────────────────────────────────────────────────────────────
def _clean_queries(queries) -> list[str]:
    if isinstance(queries, str):
        queries = [queries]
    return [q.strip() for q in queries if q and q.strip()][:MAX_QUERIES]


def _named_acts(queries: list[str]) -> list[str]:
    acts: list[str] = []
    for query in queries:
        for title in legal_terms.detect_acts(query):
            if title not in acts:
                acts.append(title)
    return acts


def _list_jobs(queries: list[str], vectors, acts: list[str]) -> list[tuple]:
    """Every ranked list to run, as ``(kind, query or vector, where, weight)``.

    A named Act contributes extra lists filtered to it, weighted up, and still additive rather
    than a hard filter, so a wrong guess costs ranking rather than erasing the real answer.
    With no Act named but plainly a policing question, the general codes get the same lift, so a
    forest officer's power of arrest cannot outrank the procedure code that governs the reader.
    """
    jobs: list[tuple] = []
    for i, query in enumerate(queries):
        if vectors is not None:
            jobs.append(("dense", vectors[i], None, 1.0))
        jobs.append(("fts", query, None, 1.0))

    filters = [(f"act_title = {sql_quote(title)}", config.ACT_FILTER_WEIGHT)
               for title in acts[:2]]
    if not acts and is_general_criminal(queries):
        titles = ",".join(sql_quote(t) for t in config.GENERAL_CODES)
        filters.append((f"act_title IN ({titles})", config.GENERAL_CODE_WEIGHT))
    for where, weight in filters:
        if vectors is not None:
            jobs.append(("dense", vectors[0], where, weight))
        jobs.append(("fts", queries[0], where, weight))
    return jobs


def _fusion_scores(candidates: list, ranked: _Ranked, fused: dict[str, float]) -> list[float]:
    """Scores for ordering and abstention without a reranker.

    Rank fusion alone cannot support abstention: the top of any list scores best whether the
    match is good or hopeless. BM25 magnitude is an absolute signal (a real legal query reaches
    ~30, an off-topic one ~13), so it leads, with fusion breaking ties.
    """
    best_possible = (ranked.total_weight / (config.RRF_K + 1)) or 1.0
    scores = []
    for c in candidates:
        relevance = min(1.0, ranked.bm25.get(c.chunk_id, 0.0) / config.BM25_FULL_SCORE)
        tiebreak = min(1.0, fused.get(c.chunk_id, 0.0) / best_possible)
        scores.append(round(relevance + 0.001 * tiebreak, 6))
    return scores


def _with_general_code_boost(candidates: list, scores: list[float], queries: list[str],
                             acts: list[str]) -> list[tuple]:
    """``(candidate, reported_score, ranking_score)``, best ranking first.

    A cross-encoder cannot separate "power to arrest without warrant" in a sectoral Act from the
    same words in the procedure code; which statute governs the reader is not a textual fact. So
    for a general criminal question the general codes win near-ties. The boost is relative to the
    best score and applies only to near-ties, so it means the same on every reranker's scale and
    can never lift an irrelevant row. The reported score stays the honest one.
    """
    prefer_general = not acts and is_general_criminal(queries)
    top = max(scores) if scores else 0.0
    boost, eligible = config.GENERAL_CODE_BOOST * top, config.GENERAL_CODE_ELIGIBLE * top

    def ranking(candidate, score: float) -> float:
        general = clean(getattr(candidate, "act_title", "")) in config.GENERAL_CODES
        return score + boost if prefer_general and general and score >= eligible else score

    ranked = [(c, s, ranking(c, s)) for c, s in zip(candidates, scores, strict=True)]
    ranked.sort(key=lambda row: row[2], reverse=True)
    return ranked


def _to_hit(row, score: float) -> Hit:
    act_title = clean(getattr(row, "act_title", ""))
    return Hit(
        unit_id=clean(row.unit_id),
        chunk_id=clean(row.chunk_id),
        citation=clean(getattr(row, "citation", "")),
        act_title=act_title,
        section_label=clean(getattr(row, "section_label", "")),
        section_name=clean(getattr(row, "section_name", "")),
        category=clean(getattr(row, "category", "")),
        status=clean(getattr(row, "status", "")),
        effective_date=clean(getattr(row, "effective_date", "")),
        act_year=clean(getattr(row, "act_year", "")),
        source_type=clean(getattr(row, "source_type", "")),
        source_snapshot=clean(getattr(row, "source_snapshot", "")),
        full_text=clean(getattr(row, "full_text", "")),
        chunk_text=clean(getattr(row, "chunk_text", "")),
        score=round(float(score), 4),
        # Trust the title, never the corpus's `jurisdiction` column (DB README §9).
        **extent(act_title),
    )


_ENGINE: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SearchEngine()
    return _ENGINE
