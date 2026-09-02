"""The pipeline stages. Each is one focused model call (or none at all).

Every stage degrades rather than raises: a planner that fails still produces a workable plan, a
grader that fails keeps everything, a gap analyst that fails ends the loop. The turn always
reaches the writer.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from .. import config, legal_terms
from ..evidence import Evidence
from ..llm.client import get_client
from . import prompts
from .schemas import (
    FactCheck,
    Coverage, Grades, Plan, Procedure, ResearchStep, SafetyCheck, SearchQueries, SubQuestion,
)

log = logging.getLogger(__name__)


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ── safety gate (no model call) ───────────────────────────────────────────────────────
# Deterministic on purpose. Someone typing "he is hitting me right now" must get helpline
# numbers immediately and reliably — not after a model round-trip that might be rate-limited.
_URGENT_PATTERNS = (
    # Match on the act and its object rather than on who is doing it. Requiring a specific
    # subject ("he is hitting") missed "my husband is hitting me" — which is precisely the
    # disclosure this gate exists for.
    (re.compile(r"(?i)\b((hit|hitt|beat|beating|attack|assault|abus|threaten)\w*\s+"
                r"(me|us|her|him|my \w+)|being\s+(beaten|hit|attacked|abused)|"
                r"maar\s+(rahe|raha|rahi)|peet\s+(rahe|raha|rahi)|ghar\s+mein\s+maar)\b"),
     "violence"),
    (re.compile(r"(?i)\b(kill myself|suicide|end my life|khudkushi|atmahatya|"
                r"want to die|self harm)\b"), "self_harm"),
    (re.compile(r"(?i)\b(being (raped|molested)|sexual(ly)? assault(ed|ing)?|"
                r"rape ho|chhed(chhad|khani))\b"), "sexual_violence"),
    (re.compile(r"(?i)\b(kidnapp?ed|abducted|trafficked|held (captive|hostage)|"
                r"forced (labour|labor|prostitution)|bandhak)\b"), "trafficking"),
    (re.compile(r"(?i)\b(missing child|child is missing|bacha gum|child labour|"
                r"child marriage|bal vivah)\b"), "child"),
    (re.compile(r"(?i)\b(arrest(ing|ed)? (me|him|her) (right )?now|police (are|is) here|"
                r"being detained|abhi arrest)\b"), "arrest_in_progress"),
)

_URGENT_ADVICE = {
    "violence": "If you are in immediate danger, call 112 now, or 181 for the women's helpline.",
    "self_harm": "If you are thinking about harming yourself, please call Tele-MANAS on 14416 — "
                 "someone is there right now.",
    "sexual_violence": "If this is happening now or just happened, call 112, or 1091 for the "
                       "women's helpline. You can report at any police station regardless of "
                       "where it happened.",
    "trafficking": "Call 112 immediately. Childline is 1098 if a child is involved.",
    "child": "Call Childline on 1098 — it operates 24 hours.",
    "arrest_in_progress": "Call 112 if you need help now. You have the right to know the "
                          "grounds of arrest, to inform someone, and to a lawyer — free legal "
                          "aid is on 15100.",
}


def safety_check(message: str) -> SafetyCheck:
    """Spot an emergency in the text itself, before any research begins."""
    for pattern, kind in _URGENT_PATTERNS:
        if pattern.search(message or ""):
            return SafetyCheck(urgent=True, kind=kind, reason=_URGENT_ADVICE.get(kind, ""))
    return SafetyCheck()


# ── planner ───────────────────────────────────────────────────────────────────────────
def _fallback_plan(message: str) -> Plan:
    """What we use when the planner is unavailable: research it as a normal legal question."""
    return Plan(
        kind="legal_question", depth="standard", answer_kind="mixed",
        normalized_query=legal_terms.expand(message),
        sub_questions=[SubQuestion(id=1, text=message)],
        steps=[ResearchStep(tool="legal_db", query=message, reason="fallback", sub_question=1)],
    )


_URL_RE = re.compile(r"^https?://\S+$", re.I)


def _dedupe_sub_questions(subs: list[SubQuestion]) -> list[SubQuestion]:
    """Drop restatements of the same sub-question.

    Planners split "how do I file an RTI" into two near-identical parts often enough that it
    is worth catching: each duplicate costs a research step and a slot in the gap analysis
    without adding anything. Compared on content words, so wording differences don't hide it.
    """
    seen: list[set[str]] = []
    out: list[SubQuestion] = []
    for sub in subs:
        words = {w for w in re.findall(r"[a-z]{4,}", (sub.text or "").lower())}
        if not words:
            continue
        if any(len(words & prior) / max(1, min(len(words), len(prior))) > 0.7 for prior in seen):
            continue
        seen.append(words)
        out.append(SubQuestion(id=len(out) + 1, text=sub.text))
    return out[:4]


def _clean_steps(steps: list[ResearchStep]) -> list[ResearchStep]:
    """Repair the two things planners reliably get wrong about steps.

    A URL in a search query makes us *search the web for a URL string* and then crawl whatever
    junk comes back — 30 wasted seconds per step, observed repeatedly. And duplicate steps
    burn the round's budget re-asking the same thing.
    """
    cleaned: list[ResearchStep] = []
    seen: set[str] = set()
    for step in list(steps):
        query = (step.query or "").strip()
        if not query:
            continue
        if _URL_RE.match(query) and step.tool != "navigate":
            # The planner clearly means "read this site" — treat it as navigation.
            step = ResearchStep(tool="navigate", query=query, reason=step.reason,
                                sub_question=step.sub_question)
        key = f"{step.tool}:{query.lower()}"
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(step)

    # Cap statute searches. Planners routinely emit three near-identical ones, and each is a
    # full hybrid search whose multi-query fusion already covers several phrasings.
    statute_steps = [s for s in cleaned if s.tool == "legal_db"]
    if len(statute_steps) > 2:
        keep = set(id(s) for s in statute_steps[:2])
        cleaned = [s for s in cleaned if s.tool != "legal_db" or id(s) in keep]
    return cleaned[:6]


async def make_plan(message: str, history: str = "", *, forced_depth: str | None = None,
                    deadline: float | None = None, on_pause=None,
                    session: str = "") -> Plan:
    client = get_client()
    user = f"{history}\n\nUSER MESSAGE: {message}".strip()
    plan = await client.chat_json(
        _messages(prompts.PLANNER, user), Plan, _fallback_plan(message),
        role="fast", stage="plan", deadline=deadline, on_pause=on_pause, session=session,
    )

    if forced_depth in ("quick", "standard", "deep"):
        plan.depth = forced_depth
    # Language is decided in code, not by the planner — it labelled plainly English questions
    # "hi" and the answer came back in Hindi.
    plan.language = legal_terms.detect_language(message)
    if not plan.normalized_query and plan.kind == "legal_question":
        plan.normalized_query = legal_terms.expand(message)
    plan.steps = _clean_steps(plan.steps)
    if plan.kind == "legal_question" and not plan.sub_questions:
        plan.sub_questions = [SubQuestion(id=1, text=plan.normalized_query or message)]
    plan.sub_questions = _dedupe_sub_questions(plan.sub_questions)
    if plan.kind == "legal_question" and not plan.steps:
        plan.steps = [ResearchStep(tool="legal_db", query=plan.normalized_query or message,
                                   sub_question=1, reason="default")]

    # Always consult the statute for a legal question. Planners asked for a *procedure* will
    # happily plan three web searches and no law at all — which produced a genuinely useful
    # RTI walkthrough that cited nine government pages and not one section of the RTI Act.
    # For a tool whose whole promise is "here is the actual provision", that is a miss.
    if plan.kind == "legal_question" and not any(s.tool == "legal_db" for s in plan.steps):
        plan.steps.insert(0, ResearchStep(
            tool="legal_db", query=plan.normalized_query or message, sub_question=1,
            reason="what the statute itself says"))
        plan.steps = plan.steps[:6]

    # A named provision is a lookup, not a research project — don't spend a deep budget on it.
    if plan.kind == "legal_question" and legal_terms.detect_section_refs(message) \
            and len(plan.sub_questions) <= 1 and forced_depth is None:
        plan.depth = "quick" if plan.depth == "standard" else plan.depth
    return plan


# ── query writer ──────────────────────────────────────────────────────────────────────
async def write_queries(question: str, *, deadline: float | None = None, on_pause=None,
                        session: str = "") -> SearchQueries:
    """Reformulations aimed at each source. Acronyms are expanded before the model sees it."""
    expanded = legal_terms.expand(question)
    default = SearchQueries(statute_queries=[expanded] if expanded != question else [],
                            web_queries=[f"{expanded} India official procedure"],
                            wikipedia_query="")
    client = get_client()
    result = await client.chat_json(
        _messages(prompts.QUERY_WRITER, f"QUESTION: {question}\nEXPANDED: {expanded}"),
        SearchQueries, default, role="fast", stage="queries",
        deadline=deadline, on_pause=on_pause, session=session,
    )
    result.statute_queries = [q for q in result.statute_queries if q.strip()][:3]
    result.web_queries = [q for q in result.web_queries if q.strip()][:2]
    return result


# ── grader ────────────────────────────────────────────────────────────────────────────
async def grade(question: str, items: list[Evidence], *, deadline: float | None = None,
                on_pause=None, session: str = "") -> list[Evidence]:
    """Keep only genuinely relevant sources.

    This is what stops "power to arrest without warrant" in the Indian Forest Act from being
    cited at someone asking about their own arrest. A rerank score cannot tell topical
    relevance from vocabulary overlap; a reader can.
    """
    if not items:
        return []
    listing = "\n\n".join(
        f"[{item.id}] ({item.kind}) {item.label()}\n{item.text[:420]}" for item in items
    )
    default = Grades(grades=[])          # only reached if parsing fails entirely
    client = get_client()
    result = await client.chat_json(
        _messages(prompts.GRADER, f"QUESTION: {question}\n\nCANDIDATES:\n{listing}"),
        Grades, default, role="fast", stage="grade",
        deadline=deadline, on_pause=on_pause, session=session,
        max_tokens=min(1400, 120 + 60 * len(items)),
    )

    if not result.grades:
        # Grading failed. Keeping everything is the safer failure: the writer still has to
        # cite, and dropping every source would produce a needlessly empty answer.
        log.warning("grader returned nothing for %d candidate(s); keeping all", len(items))
        for item in items:
            item.relevant = None
        return items

    verdicts = {g.id.strip(): g for g in result.grades}
    kept: list[Evidence] = []
    for item in items:
        verdict = verdicts.get(item.id)
        if verdict is None:
            item.relevant = None      # ungraded: keep, but mark it as unvetted
            kept.append(item)
            continue
        item.relevant = verdict.relevant
        item.grade_note = verdict.note
        if verdict.relevant:
            kept.append(item)

    if not kept:
        kept = _rescue(items)
    return kept


# Retrieval scoring above this is confident enough that a blanket rejection is more likely a
# grader misfire than a genuine absence of relevant law.
RESCUE_SCORE = 0.55


def _rescue(items: list[Evidence]) -> list[Evidence]:
    """Keep the strongest sources when the grader rejects every single one.

    Observed live: a well-retrieved arrest question returned the correct BNSS sections and the
    grader marked all six false, so the writer had nothing and produced an answer that helped
    nobody. Rejecting everything is occasionally right, but an empty answer built on top of
    *confident* retrieval is never the better outcome — so high-scoring statutes survive, and
    the writer is still free to say they are not quite on point.
    """
    confident = [i for i in items if i.is_statute and i.score >= RESCUE_SCORE]
    if not confident:
        return []
    confident.sort(key=lambda i: -i.score)
    rescued = confident[:2]
    for item in rescued:
        item.relevant = None
        item.grade_note = "kept despite grading: retrieval confidence was high"
    log.warning("grader rejected all %d candidate(s); rescuing %d high-confidence statute(s)",
                len(items), len(rescued))
    return rescued


# ── gap analyst ───────────────────────────────────────────────────────────────────────
async def find_gaps(question: str, sub_questions: list[SubQuestion], items: list[Evidence],
                    *, deadline: float | None = None, on_pause=None,
                    session: str = "") -> Coverage:
    if not sub_questions:
        return Coverage(enough=True)
    subs = "\n".join(f"{s.id}. {s.text}" for s in sub_questions)
    evidence = "\n".join(f"[{i.id}] ({i.kind}) {i.label()}: {i.text[:200]}" for i in items[:14])
    client = get_client()
    return await client.chat_json(
        _messages(prompts.GAP_ANALYST,
                  f"QUESTION: {question}\n\nSUB-QUESTIONS:\n{subs}\n\nEVIDENCE:\n{evidence}"),
        Coverage, Coverage(enough=True, note="gap analysis unavailable"),
        role="fast", stage="gaps", deadline=deadline, on_pause=on_pause, session=session,
    )


# ── procedure extractor ───────────────────────────────────────────────────────────────
async def extract_procedure(question: str, items: list[Evidence], *,
                            deadline: float | None = None, on_pause=None,
                            session: str = "") -> Procedure:
    sources = [i for i in items if i.kind in ("official", "web") and i.text]
    if not sources:
        return Procedure()
    blocks = "\n\n".join(f"URL: {i.url}\n{i.text[:2200]}" for i in sources[:4])
    client = get_client()
    result = await client.chat_json(
        _messages(prompts.PROCEDURE_EXTRACTOR, f"QUESTION: {question}\n\nSOURCES:\n{blocks}"),
        Procedure, Procedure(), role="fast", stage="procedure",
        deadline=deadline, on_pause=on_pause, session=session, max_tokens=1200,
    )
    known = {i.url for i in sources}
    result.source_urls = [u for u in result.source_urls if u in known] or list(known)[:3]
    if result.portal_url and result.portal_url not in known:
        # Only offer a link we actually read — a plausible-looking invented URL is a real harm.
        result.portal_url = ""
    return result


# ── summariser ────────────────────────────────────────────────────────────────────────
async def summarise(turns_text: str, *, deadline: float | None = None,
                    session: str = "") -> str:
    if not turns_text.strip():
        return ""
    client = get_client()
    try:
        return await client.chat(_messages(prompts.SUMMARISER, turns_text),
                                 role="fast", stage="summarise", max_tokens=280,
                                 deadline=deadline, session=session)
    except Exception as exc:
        log.debug("summarisation failed: %s", exc)
        return ""


# ── citation verification (no model call) ─────────────────────────────────────────────
_MARKER_RE = re.compile(r"\[([A-Z]{1,2}\d{1,2})\]")


def verify_citations(answer: str, items: list[Evidence]) -> tuple[str, list[str], int]:
    """Check every ``[S1]`` marker resolves to a source we actually supplied.

    Returns ``(cleaned_answer, unsupported_markers, verified_count)``. Unresolvable markers are
    removed rather than shown: a citation the user cannot click is worse than no marker, and
    silently leaving it implies support that does not exist.
    """
    known = {item.id for item in items}
    found = _MARKER_RE.findall(answer or "")
    unsupported = sorted({m for m in found if m not in known})

    cleaned = answer or ""
    for marker in unsupported:
        cleaned = cleaned.replace(f"[{marker}]", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)

    verified = len({m for m in found if m in known})
    return cleaned.strip(), unsupported, verified


def used_evidence(answer: str, items: list[Evidence]) -> list[Evidence]:
    """The sources the answer actually cited, in the order they first appear."""
    order = [m for m in _MARKER_RE.findall(answer or "")]
    by_id = {item.id: item for item in items}
    seen: set[str] = set()
    used: list[Evidence] = []
    for marker in order:
        item = by_id.get(marker)
        if item is not None and marker not in seen:
            seen.add(marker)
            used.append(item)
    return used


def today_str() -> str:
    return date.today().isoformat()


# ── self-verification ─────────────────────────────────────────────────────────────────
async def find_risky_claims(question: str, draft: str, items: list[Evidence], *,
                            deadline: float | None = None, on_pause=None,
                            session: str = "") -> FactCheck:
    """Ask the agent what in its own draft it is not sure enough about.

    Deliberately run against a *written draft* rather than raw evidence. A model is much better
    at spotting "I asserted the fee is Rs 10 and only one blog says so" than at predicting in
    advance which retrieved facts will end up load-bearing.
    """
    if not draft.strip():
        return FactCheck(confident=True)
    evidence = "\n".join(
        f"[{i.id}] ({i.kind}{'/' + i.jurisdiction if i.jurisdiction else ''}) "
        f"{i.label()}: {i.text[:180]}" for i in items[:12])
    client = get_client()
    return await client.chat_json(
        _messages(prompts.FACT_CHECKER,
                  f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}\n\nDRAFT:\n{draft[:2500]}"),
        FactCheck, FactCheck(confident=True), role="fast", stage="factcheck",
        deadline=deadline, on_pause=on_pause, session=session, max_tokens=700,
    )


def summarise_verification(claims: list, findings: dict[str, list[Evidence]]) -> str:
    """Render check results for the writer's second pass."""
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
            continue
        for item in found[:2]:
            lines.append(f"  [{item.id}] {item.label()}: {item.text[:220]}")
    return "\n".join(lines)
