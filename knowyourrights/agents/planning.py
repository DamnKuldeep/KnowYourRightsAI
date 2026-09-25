"""The planner: one model call, then rules applied in code.

The model decides what kind of question this is and how to research it. The rules after it are
the ones the model followed only sometimes; each exists because a real answer went wrong without
it, so they are enforced here rather than requested in the prompt.
"""

from __future__ import annotations

import re

from .. import config, legal_terms
from ..llm.client import get_client
from . import prompts
from .schemas import Plan, ResearchStep, SubQuestion

MAX_STEPS = 6
MAX_STATUTE_STEPS = 2
MAX_SUB_QUESTIONS = 4
_URL_RE = re.compile(r"^https?://\S+$", re.I)


def fallback_plan(message: str) -> Plan:
    """What is used when the planner is unavailable: research it as a normal legal question."""
    return Plan(
        kind="legal_question", depth="standard", answer_kind="mixed",
        normalized_query=legal_terms.expand(message),
        sub_questions=[SubQuestion(id=1, text=message)],
        steps=[ResearchStep(tool="legal_db", query=message, reason="fallback", sub_question=1)],
    )


async def make_plan(message: str, history: str = "", *, forced_depth: str | None = None,
                    deadline: float | None = None, on_pause=None, session: str = "") -> Plan:
    """Plan a turn. Raises :class:`~knowyourrights.llm.errors.LLMError` if no model answers."""
    plan = await get_client().chat_json(
        [{"role": "system", "content": prompts.PLANNER},
         {"role": "user", "content": f"{history}\n\nUSER MESSAGE: {message}".strip()}],
        Plan, fallback_plan(message), role="fast", stage="plan",
        deadline=deadline, on_pause=on_pause, session=session,
    )
    return apply_rules(plan, message, forced_depth)


def apply_rules(plan: Plan, message: str, forced_depth: str | None = None) -> Plan:
    """Everything decided in code rather than trusted to the planner."""
    if forced_depth in config.DEPTHS:
        plan.depth = forced_depth
    # Language is decided in code: the planner labelled plainly English questions "hi", and the
    # answer came back in Hindi.
    plan.language = legal_terms.detect_language(message)
    if plan.kind == "legal_question" and not plan.normalized_query:
        plan.normalized_query = legal_terms.expand(message)
    # A question that names its place has already answered "which state?".
    if plan.needs_state and legal_terms.place_named(f"{message} {plan.normalized_query}",
                                                    config.INDIAN_STATES):
        plan.needs_state = False
    plan.steps = clean_steps(plan.steps)
    if plan.kind == "legal_question":
        _ensure_research(plan, message)
        if legal_terms.detect_section_refs(message) and len(plan.sub_questions) <= 1 \
                and forced_depth is None and plan.depth == "standard":
            plan.depth = "quick"    # a named provision is a lookup, not a research project
    return plan


def _ensure_research(plan: Plan, message: str) -> None:
    query = plan.normalized_query or message
    if not plan.sub_questions:
        plan.sub_questions = [SubQuestion(id=1, text=query)]
    plan.sub_questions = dedupe_sub_questions(plan.sub_questions)
    # Always consult the statute. Asked for a procedure, planners will plan three web searches and
    # no law at all, which once produced an RTI walkthrough that cited no section of the RTI Act.
    if not any(s.tool == "legal_db" for s in plan.steps):
        plan.steps.insert(0, ResearchStep(tool="legal_db", query=query, sub_question=1,
                                          reason="what the statute itself says"))
    _ensure_deadline_and_appeal(plan, message)
    plan.steps = plan.steps[:MAX_STEPS]


def _ensure_deadline_and_appeal(plan: Plan, message: str) -> None:
    """A how-to about a named Act always searches that Act for its deadline and its appeal.

    A portal explains the form; the Act says how long the office has and what to do when it
    misses the deadline. One run of "how do I file an RTI" cited the 30-day rule, the next said
    "the sources do not state the response time" with Section 7 sitting in the corpus.
    """
    if plan.answer_kind not in ("procedure", "mixed"):
        return
    acts = legal_terms.detect_acts(f"{message} {plan.normalized_query}")
    if not acts or any(s.tool == "legal_db" and "appeal" in s.query.lower() for s in plan.steps):
        return
    plan.steps.insert(1, ResearchStep(
        tool="legal_db", sub_question=1,
        query=f"{acts[0]} time limit to decide the request and appeal if refused or no reply",
        reason="the deadline and the appeal route, from the Act itself"))


def dedupe_sub_questions(subs: list[SubQuestion]) -> list[SubQuestion]:
    """Drop restatements of the same sub-question, compared on content words.

    Each duplicate costs a research step and a slot in the gap analysis while adding nothing.
    """
    seen: list[set[str]] = []
    out: list[SubQuestion] = []
    for sub in subs:
        words = set(re.findall(r"[a-z]{4,}", (sub.text or "").lower()))
        if not words or any(len(words & prior) / max(1, min(len(words), len(prior))) > 0.7
                            for prior in seen):
            continue
        seen.append(words)
        out.append(SubQuestion(id=len(out) + 1, text=sub.text))
    return out[:MAX_SUB_QUESTIONS]


def clean_steps(steps: list[ResearchStep]) -> list[ResearchStep]:
    """Repair the two things planners reliably get wrong about steps.

    A URL as a search query means searching the web for a URL string and crawling whatever junk
    comes back; the planner clearly meant "read this site", so it becomes navigation. Duplicates
    burn the round's budget, and statute searches are capped because each is a full hybrid search
    whose multi-query fusion already covers several phrasings.
    """
    cleaned: list[ResearchStep] = []
    seen: set[str] = set()
    statute_steps = 0
    for step in steps:
        query = (step.query or "").strip()
        if not query:
            continue
        if _URL_RE.match(query) and step.tool != "navigate":
            step = ResearchStep(tool="navigate", query=query, reason=step.reason,
                                sub_question=step.sub_question)
        key = f"{step.tool}:{query.lower()}"
        if key in seen or (step.tool == "legal_db" and statute_steps >= MAX_STATUTE_STEPS):
            continue
        seen.add(key)
        statute_steps += step.tool == "legal_db"
        cleaned.append(step)
    return cleaned[:MAX_STEPS]
