"""The statute tool — the only authoritative source in the system.

Three entry points, because three different questions deserve three different mechanisms:

* ``search`` — "what does the law say about X": hybrid retrieval, reranked.
* ``lookup`` — "what does Article 21 say": an exact fetch. Someone naming a provision should
  get *that* provision, not its nearest neighbour.
* ``browse`` — "walk me through the RTI Act": the section list, straight from the index.
"""

from __future__ import annotations

import logging

from .. import config, legal_terms
from ..evidence import Evidence, from_hit
from ..retrieval.search import get_engine

log = logging.getLogger(__name__)


async def search(query: str, *, top_k: int | None = None, variants: list[str] | None = None,
                 deadline: float | None = None, on_pause=None, session: str = "") -> list[Evidence]:
    """Retrieve statute sections for one sub-question.

    ``variants`` are alternative phrasings that all feed the same fusion. The caller's plain
    query is always what the reranker sees, since acronym-expanded text reads as broken
    English to a cross-encoder.
    """
    engine = get_engine()
    expanded = legal_terms.expand(query)
    queries = [query]
    if expanded != query:
        queries.append(expanded)
    for variant in variants or []:
        if variant and variant not in queries:
            queries.append(variant)

    result = await engine.search(queries, top_k=top_k, rerank_with=query,
                                 deadline=deadline, on_pause=on_pause, session=session)

    cite_floor = engine.reranker.thresholds.cite
    evidence: list[Evidence] = []
    for hit in result.hits:
        if hit.score < cite_floor:
            continue
        item = from_hit(hit, query)
        notes = []
        if hit.is_state_law:
            notes.append(
                f"[STATE LAW — this is {hit.state} legislation and applies only there; "
                f"the database holds central law, so the user's own state may differ.]"
            )
        if hit.is_omitted:
            notes.append("[OMITTED — this provision has been removed and is no longer in force.]")
        if notes:
            item.text = "\n".join(notes) + "\n" + item.text
        evidence.append(item)

    if result.abstain and evidence:
        log.debug("retrieval abstained for %r (top %.3f) but kept %d above the citation floor",
                  query, result.top_score, len(evidence))
    return evidence


async def search_result(query: str, **kwargs):
    """The raw :class:`SearchResult`, for callers that need ``abstain``/``mode``/notes."""
    engine = get_engine()
    expanded = legal_terms.expand(query)
    queries = [query] if expanded == query else [query, expanded]
    return await engine.search(queries, rerank_with=query, **kwargs)


def lookup(question: str) -> list[Evidence]:
    """Answer "what does <provision> say" exactly, or return nothing.

    Returning nothing is meaningful: it tells the caller this was not actually a citation
    lookup and should go through normal search instead.
    """
    refs = legal_terms.detect_section_refs(question)
    if not refs:
        return []
    engine = get_engine()
    evidence: list[Evidence] = []
    for ref in refs[:3]:
        hits = engine.lookup(ref.act, ref.label, ref.kind == "article")
        for hit in hits:
            item = from_hit(hit, question)
            item.score = 1.0
            item.meta["exact_lookup"] = True
            evidence.append(item)
    return evidence


def browse(act: str, limit: int = 60) -> tuple[list[dict], str | None]:
    """A table of contents for an Act: ``([{section_label, section_name, citation}], title)``."""
    engine = get_engine()
    rows, title = engine.store.browse_act(act, limit=limit)
    if title is None:
        return [], None
    listing = [
        {
            "section_label": str(row.section_label),
            "section_name": str(row.section_name or ""),
            "citation": str(row.citation),
            "category": str(row.category or ""),
            "status": str(row.status or ""),
        }
        for row in rows.itertuples()
    ]
    return listing, title


def corpus_notes(question: str) -> list[str]:
    """Caveats the answer layer must state up front.

    Covers the two things the corpus is silent about but users will ask anyway: statutes it
    does not hold, and codes that were repealed out of it.
    """
    notes = list(legal_terms.detect_gaps(question))
    for repeal in legal_terms.detect_repeals(question):
        notes.append(repeal.note)
    return notes


def snapshot() -> dict:
    return get_engine().status()
