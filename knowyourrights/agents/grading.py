"""The relevance grader: which retrieved sources genuinely help answer this question.

A rerank score cannot tell topical relevance from shared vocabulary; a reader can. This is what
stops "power to arrest without warrant" in the Indian Forest Act being cited to someone asking
about their own arrest.
"""

from __future__ import annotations

import logging

from ..evidence import Evidence
from ..llm.client import get_client
from . import prompts
from .schemas import Grades

log = logging.getLogger(__name__)

# Retrieval this confident makes a blanket rejection more likely a grader misfire than a real
# absence of relevant law.
RESCUE_SCORE = 0.55
_EXCERPT_CHARS = 420


async def grade(question: str, items: list[Evidence], *, deadline: float | None = None,
                on_pause=None, session: str = "") -> list[Evidence]:
    """The relevant subset of ``items``. Raises ``LLMError`` if no model answers."""
    if not items:
        return []
    listing = "\n\n".join(f"[{item.id}] ({item.kind}) {item.label()}\n{item.text[:_EXCERPT_CHARS]}"
                          for item in items)
    result = await get_client().chat_json(
        [{"role": "system", "content": prompts.GRADER},
         {"role": "user", "content": f"QUESTION: {question}\n\nCANDIDATES:\n{listing}"}],
        Grades, Grades(grades=[]), role="fast", stage="grade", deadline=deadline,
        on_pause=on_pause, session=session, max_tokens=min(1400, 120 + 60 * len(items)),
    )
    if not result.grades:
        # Grading failed. Keeping everything is the safer failure: the writer still has to cite,
        # and dropping every source would produce a needlessly empty answer.
        log.warning("grader returned nothing for %d candidate(s); keeping all", len(items))
        for item in items:
            item.relevant = None
        return items
    return apply_grades(items, {g.id.strip(): g for g in result.grades})


def apply_grades(items: list[Evidence], verdicts: dict) -> list[Evidence]:
    """Keep what was graded relevant, and anything the grader skipped (marked unvetted)."""
    kept: list[Evidence] = []
    for item in items:
        verdict = verdicts.get(item.id)
        if verdict is None:
            item.relevant = None
            kept.append(item)
            continue
        item.relevant = verdict.relevant
        if verdict.relevant:
            kept.append(item)
    return kept or rescue(items)


def rescue(items: list[Evidence]) -> list[Evidence]:
    """Keep the strongest statutes when the grader rejects every single source.

    Observed live: a well-retrieved arrest question returned the right BNSS sections and the
    grader marked all six irrelevant, leaving the writer nothing. Rejecting everything is
    occasionally right, but an empty answer on top of confident retrieval never is.
    """
    confident = sorted((i for i in items if i.is_statute and i.score >= RESCUE_SCORE),
                       key=lambda i: -i.score)[:2]
    for item in confident:
        item.relevant = None
    if confident:
        log.warning("grader rejected all %d candidate(s); rescuing %d confident statute(s)",
                    len(items), len(confident))
    return confident
