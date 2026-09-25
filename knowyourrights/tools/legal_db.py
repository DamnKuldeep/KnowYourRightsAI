"""The statute tool — the only authoritative source in the system.

* :func:`search` — "what does the law say about X": hybrid retrieval, reranked.
* :func:`lookup` — "what does Article 21 say": an exact fetch. Someone naming a provision gets
  that provision, not its nearest neighbour.
* :func:`corpus_notes` — caveats about what the corpus cannot know: statutes it does not hold,
  codes repealed since it was built, and tenancy being state law.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .. import legal_terms
from ..evidence import Evidence, from_hit
from ..retrieval.search import Hit, get_engine

log = logging.getLogger(__name__)


@dataclass
class StatuteResults:
    items: list[Evidence] = field(default_factory=list)
    # Reader-facing notes when retrieval ran in a reduced mode (see retrieval.search.NOTE_*).
    degraded: list[str] = field(default_factory=list)


async def search(query: str, *, top_k: int | None = None, variants: list[str] | None = None,
                 deadline: float | None = None, on_pause=None) -> StatuteResults:
    """Statute sections for one sub-question, above the citation floor.

    ``variants`` are other phrasings that feed the same fusion. The plain query is what the
    reranker sees: acronym-expanded text reads as broken English to a cross-encoder.
    """
    queries = list(dict.fromkeys(q for q in (query, legal_terms.expand(query), *(variants or []))
                                 if q))
    result = await get_engine().search(queries, top_k=top_k, rerank_with=query,
                                       deadline=deadline, on_pause=on_pause)
    items = [_with_scope_notes(hit, query) for hit in result.hits
             if hit.score >= result.cite_floor]
    if result.abstain and items:
        log.debug("retrieval abstained for %r (top %.3f) but kept %d above the citation floor",
                  query, result.top_score, len(items))
    return StatuteResults(items, list(result.degraded))


def _with_scope_notes(hit: Hit, query: str) -> Evidence:
    """Evidence for a hit, prefixed with where it applies when that is not all of India."""
    item = from_hit(hit, query)
    notes = []
    if hit.is_state_law:
        notes.append(f"[STATE LAW — this is {hit.state} legislation and applies only there; "
                     f"the database holds central law, so the user's own state may differ.]")
    elif hit.is_territorial:
        notes.append(f"[{hit.state.upper()} ONLY — Parliament passed this for {hit.state}, and "
                     f"it does not apply anywhere else in India. Do not present it as all-India "
                     f"law.]")
    if hit.is_omitted:
        notes.append("[OMITTED — this provision has been removed and is no longer in force.]")
    if notes:
        item.text = "\n".join(notes) + "\n" + item.text
    return item


def lookup(question: str) -> list[Evidence]:
    """The exact text of any provision the question names, or nothing.

    Nothing is a meaningful answer: the question was not a citation lookup, so it goes through
    normal search instead.
    """
    evidence: list[Evidence] = []
    for ref in legal_terms.detect_section_refs(question)[:3]:
        for hit in get_engine().lookup(ref.act, ref.label, ref.kind == "article"):
            item = from_hit(hit, question)
            item.score = 1.0
            item.meta["exact_lookup"] = True
            evidence.append(item)
    return evidence


# ── caveats ───────────────────────────────────────────────────────────────────────────
# Notes are shown to the reader, so they state facts. The writer receives its own wording (see
# for_writer): given the reader's version it copied the note into answers word for word.
TENANCY_NOTE = ("Tenancy is governed by each state's own rent law. The Model Tenancy Act, 2021 "
                "is a template the Centre issued for states to adopt; it is not in force in a "
                "state unless that state has enacted it.")
_WRITER_VERSION = {
    TENANCY_NOTE: ("The Model Tenancy Act, 2021 is a model law, in force only where a state has "
                   "enacted it. Never say it applies centrally or in a state unless a source "
                   "shows that state enacted it. Mention it only if the answer relies on it."),
}
# A Mumbai deposit answer said the Model Tenancy Act "applies centrally", from a web page.
_TENANCY = re.compile(r"\b(landlord|tenant|tenancy|rent(ed|al)?|lease|security deposit|"
                      r"kiraye?dar|makaan malik|model tenancy)\b|किराय|मकान मालिक", re.I)


def corpus_notes(question: str) -> list[str]:
    """Caveats to show the reader before the answer."""
    notes = list(legal_terms.detect_gaps(question))
    notes += [repeal.note for repeal in legal_terms.detect_repeals(question)]
    # The section-level translation, stated outright. Knowing the IPC became the BNS was not
    # enough: the writer assumed Section 420 kept its number and cited a BNS section that does
    # not exist.
    mapped, unmapped = legal_terms.map_repealed_sections(question)
    notes += [m.note for m in mapped]
    notes += [f"Section {num} of the old {code} has no verified equivalent here. The numbering "
              f"changed in the new code, so the matching section could not be confirmed."
              for code, num in unmapped]
    if _TENANCY.search(question or ""):
        notes.append(TENANCY_NOTE)
    return notes


def for_writer(notes: list[str]) -> list[str]:
    """The same caveats, phrased as instructions for the writer."""
    return [_WRITER_VERSION.get(n, n) for n in notes]
