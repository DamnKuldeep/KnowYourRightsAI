"""Research rounds: run the planned steps, grade what came back, and decide whether to go on.

Everything is budgeted against the turn's wall-clock deadline. When time runs out research stops
and the answer is written from what was found: a shallower answer beats an error.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .. import config, events
from ..agents import grading, stages
from ..agents.schemas import Coverage, ResearchStep
from ..context.reduce import reduce_pages
from ..evidence import Evidence, assign_ids, dedupe
from ..llm.errors import DeadlineExceeded, ProviderAuthError
from ..llm.ledger import get_ledger
from ..tools import crawl, legal_db, navigate, web, wikipedia
from .turn import TurnState, degradable

log = logging.getLogger(__name__)

STEP_LABELS = {
    "legal_db": "Searching Indian law",
    "web": "Searching the web",
    "official": "Checking official sources",
    "wikipedia": "Reading background",
    "navigate": "Navigating the official portal",
}
GRADE_LABEL = "Checking which sources are actually relevant"
GAPS_LABEL = "Deciding whether anything is still missing"
NOTE_NO_GRADER = ("The relevance check was unavailable, so some sources below may be only "
                  "loosely related to your question.")
MAX_RECALLED = 3
MAX_GAP_STEPS = 3
MAX_PAGES_PER_STEP = 3


async def research(turn: TurnState) -> None:
    """Run research rounds until the evidence is enough, the budget is spent, or time is up."""
    steps = list(turn.plan.steps)
    seen: set[str] = set()
    for round_no in range(1, turn.budget.max_rounds + 1):
        if turn.cancelled.is_set() or turn.budget.expired or not steps:
            return
        if round_no > 1:
            turn.stage(f"round{round_no}", f"Digging deeper (round {round_no})")
        gathered = await run_steps(turn, steps, seen)
        if round_no == 1:
            gathered += recall(turn)
        turn.evidence = assign_ids(dedupe(turn.evidence + gathered))
        await grade_round(turn)
        if round_no >= turn.budget.max_rounds or turn.budget.expired:
            return
        steps = await next_steps(turn)


def recall(turn: TurnState) -> list[Evidence]:
    """Sources vetted earlier in this conversation that may bear on this question.

    They go through the grader with everything else. Matched on the planner's standalone
    restatement, because "and how long can they keep me" shares almost no words with the
    sections it is asking about.
    """
    present = {e.dedupe_key() for e in turn.evidence}
    recalled = [e for e in turn.conversation.recall(turn.question)
                if e.dedupe_key() not in present]
    return recalled[:MAX_RECALLED]


async def grade_round(turn: TurnState) -> None:
    """Keep the relevant sources and show them. Exact citation lookups need no grading."""
    if not turn.evidence:
        return
    if all(e.meta.get("exact_lookup") for e in turn.evidence):
        turn.stage("grade", GRADE_LABEL, "done", "exact citation — no grading needed")
    else:
        turn.stage("grade", GRADE_LABEL)
        before = len(turn.evidence)
        # A how-to answer always has a fee, response-time and if-refused section, so sources
        # are graded against that too. Graded against "how do I file an RTI" alone, Section 19
        # (the appeal) was retrieved and then thrown out.
        graded_for = turn.question
        if turn.plan.answer_kind in ("procedure", "mixed"):
            graded_for += (" (the answer also covers the fee, the deadline to reply, and the "
                           "appeal if refused or not answered)")
        kept = await degradable(
            turn, grading.grade(graded_for, turn.evidence, **turn.call_kwargs()),
            turn.evidence, NOTE_NO_GRADER)
        turn.evidence = assign_ids(kept)
        turn.stage("grade", GRADE_LABEL, "done", f"kept {len(turn.evidence)} of {before}")
    for item in turn.evidence:
        turn.emit(events.source(item))


async def next_steps(turn: TurnState) -> list[ResearchStep]:
    """Ask what is still missing, as steps for another round. Empty when nothing is."""
    turn.stage("gaps", GAPS_LABEL)
    coverage = await degradable(
        turn, stages.find_gaps(turn.question, turn.plan.sub_questions, turn.evidence,
                               **turn.call_kwargs()),
        Coverage(enough=True, note="gap analysis unavailable"))
    if coverage.enough or not coverage.gaps:
        turn.stage("gaps", GAPS_LABEL, "done", coverage.note or "nothing important missing")
        return []
    turn.stage("gaps", GAPS_LABEL, "done", f"{len(coverage.gaps)} gap(s) to close")
    return [ResearchStep(tool=g.tool, query=g.query or g.missing, reason=g.missing,
                         sub_question=g.sub_question)
            for g in coverage.gaps[:MAX_GAP_STEPS] if g.query or g.missing]


async def run_steps(turn: TurnState, steps: list[ResearchStep], seen: set[str]) -> list[Evidence]:
    """One round's steps, in parallel, skipping any already run this turn."""
    fresh = []
    for step in steps:
        key = f"{step.tool}:{(step.query or '').lower().strip()}"
        if step.query and key not in seen:
            seen.add(key)
            fresh.append(step)
    results = await asyncio.gather(*(run_step(turn, s) for s in fresh), return_exceptions=True)
    gathered: list[Evidence] = []
    for result in results:
        if isinstance(result, (DeadlineExceeded, ProviderAuthError, asyncio.CancelledError)):
            raise result
        if isinstance(result, BaseException):
            log.warning("a research step failed: %s", result)
            continue
        gathered.extend(result)
    return gathered


async def run_step(turn: TurnState, step: ResearchStep) -> list[Evidence]:
    """One research step, reported to the reader as it starts and finishes."""
    turn.emit(events.tool(step.tool, step.query, "running",
                          detail=STEP_LABELS.get(step.tool, step.tool)))
    started = time.monotonic()
    try:
        items = await _dispatch(turn, step)
    except (DeadlineExceeded, ProviderAuthError, asyncio.CancelledError):
        raise
    except Exception as exc:
        log.warning("%s step failed: %s", step.tool, exc)
        get_ledger().record_tool(step.tool, ok=False)
        turn.emit(events.tool(step.tool, step.query, "error", detail="this source failed",
                              elapsed_ms=_ms_since(started)))
        return []
    get_ledger().record_tool(step.tool, ok=True)
    turn.emit(events.tool(step.tool, step.query, "done", count=len(items),
                          elapsed_ms=_ms_since(started)))
    return items


async def _dispatch(turn: TurnState, step: ResearchStep) -> list[Evidence]:
    if step.tool == "legal_db":
        return await _search_statutes(turn, step)
    if step.tool == "wikipedia":
        return await wikipedia.lookup(step.query)
    if step.tool == "navigate":
        return await _navigate(turn, step)
    finder = web.search_official if step.tool == "official" else web.search
    return await read_pages(turn, await finder(step.query), step.query)


async def _search_statutes(turn: TurnState, step: ResearchStep) -> list[Evidence]:
    # The corpus is English. A Hindi or Hinglish query searched verbatim retrieves noise, so
    # retrieval uses the planner's English restatement; the answer stays in the user's language.
    query = step.query
    if turn.plan.language != "en" and turn.plan.normalized_query:
        query = turn.plan.normalized_query
    # Reformulations are written once a turn, and only in deep mode: at standard depth they cost
    # ~1 s of first-token time and measurably bought nothing (Recall@5 is 100% without them).
    if turn.query_variants is None and turn.budget.depth == "deep":
        written = await degradable(turn, stages.write_queries(turn.question,
                                                              **turn.call_kwargs()), None)
        turn.query_variants = written.statute_queries if written else []
    found = await legal_db.search(query, variants=turn.query_variants,
                                  deadline=turn.budget.deadline, on_pause=turn.on_pause)
    for note in found.degraded:
        turn.degrade(note)
    return found.items


async def read_pages(turn: TurnState, found: list[Evidence], query: str) -> list[Evidence]:
    """Turn search snippets into read pages, best sources first, when the budget allows.

    A snippet says a fee exists; the page says what it is.
    """
    if not found or not turn.budget.can_crawl():
        return found
    ranked = sorted(found, key=lambda e: -e.tier)
    room = min(len(ranked), turn.budget.max_crawls - turn.budget.crawls, MAX_PAGES_PER_STEP)
    urls = [e.url for e in ranked[:room] if e.url]
    if not urls:
        return found
    turn.emit(events.tool("crawl", f"reading {len(urls)} page(s)", "running"))
    started = time.monotonic()
    pages = await crawl.get_crawler().fetch(urls, query)
    turn.budget.crawls += len(urls)
    read = crawl.to_evidence(pages, query)
    turn.emit(events.tool("crawl", f"read {len(read)} page(s)", "done", count=len(read),
                          elapsed_ms=_ms_since(started)))
    # Keep snippets only for pages that could not be read.
    read_urls = {e.url.rstrip("/") for e in read}
    leftovers = [e for e in found if e.url.rstrip("/") not in read_urls]
    return await reduce_pages(read + leftovers, query, deadline=turn.budget.deadline,
                              on_pause=turn.on_pause)


async def _navigate(turn: TurnState, step: ResearchStep) -> list[Evidence]:
    """Walk an official portal for an end-to-end procedure."""
    seed = step.query.strip()
    goal = step.reason or turn.question
    if not seed.startswith("http"):
        # The planner named a portal rather than a URL: find it first.
        found = await web.search_official(seed or goal, n=3)
        seed = next((e.url for e in found if e.tier >= config.TIER_OFFICIAL), "")
        if not seed:
            return found
    if not turn.budget.can_crawl(2):
        return []
    pages = await navigate.navigate(
        crawl.get_crawler(), seed, goal,
        max_pages=max(3, turn.budget.max_crawls - turn.budget.crawls),
        max_depth=turn.budget.nav_depth, should_cancel=lambda *_: turn.cancelled.is_set())
    turn.budget.crawls += len(pages)
    turn.emit(events.tool("navigate", seed, "done", count=len(pages),
                          detail=f"visited {len(pages)} page(s)"))
    return await reduce_pages(crawl.to_evidence(pages, goal), goal,
                              deadline=turn.budget.deadline, on_pause=turn.on_pause)


async def extract_procedure(turn: TurnState) -> None:
    """The at-a-glance card for a how-to: fee, time limit, appeal, documents, portal."""
    sources = [e for e in turn.evidence if e.kind in ("official", "web")]
    if not sources:
        return
    label = "Assembling the steps"
    turn.stage("procedure", label)
    procedure = await degradable(
        turn, stages.extract_procedure(turn.question, sources, **turn.call_kwargs()), None)
    if procedure is not None and procedure.is_useful:
        turn.procedure = procedure
        turn.emit(events.procedure(procedure.model_dump()))
        turn.stage("procedure", label, "done", f"{len(procedure.steps)} step(s)")
    else:
        turn.stage("procedure", label, "done", "no clear procedure found")


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
