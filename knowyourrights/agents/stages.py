"""The smaller model stages: query writing, gap analysis, procedure extraction, fact-checking
and conversation summaries. Planning and grading have their own modules.

Each stage is one focused model call with a safe default for a reply that will not parse.
Provider failures raise :class:`~knowyourrights.llm.errors.LLMError`, and the orchestrator
decides whether that stage can be skipped.
"""

from __future__ import annotations

import logging
import re

from .. import legal_terms
from ..evidence import Evidence
from ..llm.client import get_client
from . import prompts
from .schemas import Coverage, FactCheck, Procedure, SearchQueries, SubQuestion

log = logging.getLogger(__name__)


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ── query writer ──────────────────────────────────────────────────────────────────────
async def write_queries(question: str, *, deadline: float | None = None, on_pause=None,
                        session: str = "") -> SearchQueries:
    """Reformulations aimed at each source. Acronyms are expanded before the model sees them."""
    expanded = legal_terms.expand(question)
    default = SearchQueries(statute_queries=[expanded] if expanded != question else [],
                            web_queries=[f"{expanded} India official procedure"])
    result = await get_client().chat_json(
        _messages(prompts.QUERY_WRITER, f"QUESTION: {question}\nEXPANDED: {expanded}"),
        SearchQueries, default, role="fast", stage="queries",
        deadline=deadline, on_pause=on_pause, session=session,
    )
    result.statute_queries = [q for q in result.statute_queries if q.strip()][:3]
    result.web_queries = [q for q in result.web_queries if q.strip()][:2]
    return result


# ── gap analyst ───────────────────────────────────────────────────────────────────────
async def find_gaps(question: str, sub_questions: list[SubQuestion], items: list[Evidence],
                    *, deadline: float | None = None, on_pause=None,
                    session: str = "") -> Coverage:
    """Whether another research round is worth its time, and what it should look for."""
    if not sub_questions:
        return Coverage(enough=True)
    subs = "\n".join(f"{s.id}. {s.text}" for s in sub_questions)
    evidence = "\n".join(f"[{i.id}] ({i.kind}) {i.label()}: {i.text[:200]}" for i in items[:14])
    return await get_client().chat_json(
        _messages(prompts.GAP_ANALYST,
                  f"QUESTION: {question}\n\nSUB-QUESTIONS:\n{subs}\n\nEVIDENCE:\n{evidence}"),
        Coverage, Coverage(enough=True, note="gap analysis unavailable"),
        role="fast", stage="gaps", deadline=deadline, on_pause=on_pause, session=session,
    )


# ── procedure extractor ───────────────────────────────────────────────────────────────
# An amount is a currency marker next to a number. A bare digit is not enough: "as prescribed in
# the RTI Rules, 2012" contains one, and is exactly the text this exists to reject.
_AMOUNT = re.compile(r"(₹|\brs\.?|\binr)\s*\d|\d[\d,]*\s*(/-|rupees?\b)", re.I)
_NO_CHARGE = re.compile(r"\b(free|no fee|nil|no charge|exempt(ed)?)\b", re.I)
_DURATION = re.compile(r"\d+\s*(working\s+)?(hours?|days?|weeks?|months?|years?)\b", re.I)
# A field that says it has no value. Checked first: "amount not specified; free for BPL
# applicants is not mentioned" contains "free", and was shown on the card as the fee.
_NOT_GIVEN = re.compile(r"\bnot\s+(specified|mentioned|stated|given|provided|available|"
                        r"found|clear)\b|\bunknown\b|\bN/A\b|उल्लेख नहीं|नहीं (दी|बताई)", re.I)


async def extract_procedure(question: str, items: list[Evidence], *,
                            deadline: float | None = None, on_pause=None,
                            session: str = "") -> Procedure:
    """The at-a-glance facts of a procedure — fee, time limit, appeal, documents, portal."""
    sources = [i for i in items if i.kind in ("official", "web") and i.text]
    if not sources:
        return Procedure()
    # Eight pages at 1,400 characters rather than four at 2,200: the card and the answer must be
    # read from the same evidence or they disagree, and a fee sits near the top of its page.
    blocks = "\n\n".join(f"URL: {i.url}\n{i.text[:1400]}" for i in sources[:8])
    result = await get_client().chat_json(
        _messages(prompts.PROCEDURE_EXTRACTOR, f"QUESTION: {question}\n\nSOURCES:\n{blocks}"),
        Procedure, Procedure(), role="fast", stage="procedure",
        deadline=deadline, on_pause=on_pause, session=session, max_tokens=1200,
    )
    return keep_stated_facts(result, {i.url for i in sources})


def keep_stated_facts(result: Procedure, known_urls: set[str]) -> Procedure:
    """Drop anything the card should not show: unread links and facts without a value."""
    result.source_urls = [u for u in result.source_urls if u in known_urls] \
        or sorted(known_urls)[:3]
    if result.portal_url not in known_urls:
        result.portal_url = ""      # only a link we actually read; an invented URL is a harm
    # "Fee: as prescribed" tells the reader nothing and contradicts an answer naming the amount.
    result.fees = result.fees if states_a_value(result.fees, money=True) else ""
    result.timeline = result.timeline if states_a_value(result.timeline) else ""
    result.appeal_to = "" if _NOT_GIVEN.search(result.appeal_to or "") else result.appeal_to
    result.documents = [d for d in result.documents if not _NOT_GIVEN.search(d)]
    return result


def states_a_value(text: str, money: bool = False) -> bool:
    """Does this field state an amount (or "free") for a fee, or a duration for a time limit?"""
    text = (text or "").strip()
    if not text or _NOT_GIVEN.search(text):
        return False
    if money:
        return bool(_AMOUNT.search(text) or _NO_CHARGE.search(text))
    return bool(_DURATION.search(text))


# ── summariser ────────────────────────────────────────────────────────────────────────
async def summarise(turns_text: str, *, session: str = "") -> str:
    """A rolling summary of older turns. Returns "" on any failure: memory is optional."""
    if not turns_text.strip():
        return ""
    try:
        return await get_client().chat(_messages(prompts.SUMMARISER, turns_text), role="fast",
                                       stage="summarise", max_tokens=280, session=session)
    except Exception as exc:
        log.debug("summarisation failed: %s", exc)
        return ""


# ── self-verification ─────────────────────────────────────────────────────────────────
async def find_risky_claims(question: str, draft: str, items: list[Evidence], *,
                            deadline: float | None = None, on_pause=None,
                            session: str = "") -> FactCheck:
    """Ask what in its own draft the agent is not sure enough about.

    Run against a written draft rather than raw evidence: a model is far better at spotting "I
    asserted the fee is Rs 10 and only one blog says so" than at predicting in advance which
    retrieved facts will end up load-bearing.
    """
    if not draft.strip():
        return FactCheck(confident=True)
    evidence = "\n".join(
        f"[{i.id}] ({i.kind}{'/' + i.jurisdiction if i.jurisdiction else ''}) "
        f"{i.label()}: {i.text[:180]}" for i in items[:12])
    return await get_client().chat_json(
        _messages(prompts.FACT_CHECKER,
                  f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}\n\nDRAFT:\n{draft[:2500]}"),
        FactCheck, FactCheck(confident=True), role="fast", stage="factcheck",
        deadline=deadline, on_pause=on_pause, session=session, max_tokens=700,
    )


def verification_note(claims: list, findings: dict[str, list[Evidence]]) -> str:
    """The fact-check results, rendered for the writer's second pass."""
    if not findings:
        return ""
    lines = ["VERIFICATION PASS — these claims were checked against fresh web sources.",
             "Where a check CONFIRMS a claim, state it confidently. Where it CONTRADICTS or",
             "finds nothing, correct the claim or say it could not be confirmed."]
    for claim in claims:
        found = findings.get(claim.claim, [])
        lines.append(f"\nCLAIM: {claim.claim}")
        if not found:
            lines.append("  no confirming source found — soften this or drop it")
        lines.extend(f"  [{item.id}] {item.label()}: {item.text[:220]}" for item in found[:2])
    return "\n".join(lines)
