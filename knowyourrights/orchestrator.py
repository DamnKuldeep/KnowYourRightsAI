"""The research state machine.

Runs as a background task feeding an event queue, rather than as a plain async generator.
That matters: a rate-limit pause happens deep inside the NIM client, in a callback that has no
way to ``yield``. With a queue, that callback can still put a countdown on the user's screen
the instant it happens, instead of the UI freezing until the next natural yield point.

Everything here is budgeted against a wall-clock deadline rather than a step count. When time
runs out the loop stops gathering and writes the answer with what it has — a shallower answer
is a far better outcome than an error, and it is the outcome a user of a free tier will
actually hit.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from . import config, events, legal_terms
from .agents import prompts, stages
from .agents.schemas import Plan, Procedure
from .context import budget as ctx_budget
from .context import packer
from .context.memory import Conversation
from .context.reduce import reduce_pages
from .evidence import Evidence, assign_ids, dedupe
from .nim.client import NimDeadlineExceeded, NimError, get_client
from .nim.ledger import get_ledger
from .tools import crawl, legal_db, web, wikipedia

log = logging.getLogger(__name__)

STAGE_LABELS = {
    "legal_db": "Searching Indian law",
    "web": "Searching the web",
    "official": "Checking official sources",
    "wikipedia": "Reading background",
    "navigate": "Navigating the official portal",
}


@dataclass
class TurnBudget:
    """What this turn may spend. Every field is a soft ceiling that degrades."""

    depth: str
    deadline: float
    max_rounds: int
    max_crawls: int
    nav_depth: int
    max_llm_calls: int
    llm_calls: int = 0
    crawls: int = 0

    @property
    def time_left(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.time_left <= 0

    def can_crawl(self, n: int = 1) -> bool:
        return self.crawls + n <= self.max_crawls and not self.expired

    @classmethod
    def for_depth(cls, depth: str) -> "TurnBudget":
        spec = config.DEPTHS.get(depth, config.DEPTHS["standard"])
        return cls(depth=spec.name, deadline=time.monotonic() + spec.deadline_s,
                   max_rounds=spec.max_rounds, max_crawls=spec.max_crawls,
                   nav_depth=spec.nav_depth, max_llm_calls=spec.max_llm_calls)


@dataclass
class TurnState:
    session_id: str
    message: str
    conversation: Conversation
    budget: TurnBudget
    # Ledger totals are process-lifetime; a turn needs its own delta or the UI reports the
    # server's whole history as the cost of one question.
    calls_at_start: int = 0
    plan: Plan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    procedure: Procedure | None = None
    notes: list[str] = field(default_factory=list)
    answer: str = ""
    query_variants: list[str] | None = None
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    started: float = field(default_factory=time.monotonic)


class Orchestrator:
    def __init__(self) -> None:
        self.client = get_client()
        self.ledger = get_ledger()
        self._active: dict[str, TurnState] = {}

    # ── public entry point ───────────────────────────────────────────────────────────
    async def stream(self, message: str, conversation: Conversation, *,
                     depth: str | None = None, state: str | None = None):
        """Yield :class:`Event` objects for one turn."""
        queue: asyncio.Queue = asyncio.Queue()
        turn_id = uuid.uuid4().hex[:12]

        if state:
            conversation.state = state
        turn = TurnState(
            session_id=conversation.session_id or turn_id,
            message=message.strip(),
            conversation=conversation,
            budget=TurnBudget.for_depth(depth if depth in config.DEPTHS else "standard"),
            calls_at_start=self.ledger.total_calls,
        )
        self._active[turn_id] = turn

        task = asyncio.create_task(self._run(turn, queue, requested_depth=depth))
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            self._active.pop(turn_id, None)
            if not task.done():
                turn.cancelled.set()
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def cancel(self, session_id: str) -> bool:
        for turn in self._active.values():
            if turn.session_id == session_id:
                turn.cancelled.set()
                return True
        return False

    # ── the turn ─────────────────────────────────────────────────────────────────────
    async def _run(self, turn: TurnState, queue: asyncio.Queue, requested_depth: str | None):
        emit = queue.put_nowait

        def on_pause(model: str, seconds: float, reason: str) -> None:
            """Called from inside the NIM client when a call has to wait."""
            if reason == "rate limit":
                emit(events.notice(
                    f"The AI service is rate-limited. Waiting {seconds:.0f}s, then continuing "
                    f"where it left off.", level="pause", resume_in_s=seconds, model=model))
            elif seconds >= 3:
                emit(events.notice(f"Pacing requests to stay within the free tier "
                                   f"({seconds:.0f}s).", level="info", resume_in_s=seconds))

        try:
            await self._pipeline(turn, emit, on_pause, requested_depth)
        except asyncio.CancelledError:
            emit(events.notice("Stopped.", level="info"))
        except NimDeadlineExceeded as exc:
            log.warning("turn ran out of time: %s", exc)
            emit(events.notice(
                "This took longer than the time budget allowed, so here is what was found.",
                level="warn"))
            await self._write_answer(turn, emit, on_pause, degraded=True)
        except Exception as exc:
            log.exception("turn failed")
            emit(events.error(f"Something went wrong: {str(exc)[:200]}"))
        finally:
            emit(events.usage(**self._usage(turn)))
            emit(events.done(elapsed_s=round(time.monotonic() - turn.started, 1)))
            emit(None)

    async def _pipeline(self, turn: TurnState, emit, on_pause, requested_depth):
        conversation = turn.conversation
        conversation.add_user(turn.message)

        # 1 — emergencies come before research, and without a model call.
        check = stages.safety_check(turn.message)
        if check.urgent:
            emit(events.safety(list(config.HELPLINES), check.reason))

        # 2 — plan
        emit(events.stage("plan", "Understanding your question"))
        history = conversation.history_block()
        turn.plan = await stages.make_plan(
            turn.message, history, forced_depth=requested_depth,
            deadline=turn.budget.deadline, on_pause=on_pause, session=turn.session_id)
        plan = turn.plan

        # The planner's own depth choice governs unless the user forced one.
        if requested_depth not in config.DEPTHS:
            turn.budget = TurnBudget.for_depth(plan.depth)
        emit(events.stage("plan", "Understanding your question", "done",
                          detail=f"{plan.kind} · {plan.depth}"))
        emit(events.plan(plan.depth, plan.answer_kind,
                         [s.text for s in plan.sub_questions],
                         [{"tool": s.tool, "query": s.query, "reason": s.reason}
                          for s in plan.steps],
                         language=plan.language, needs_state=plan.needs_state))

        if plan.is_conversational or not plan.steps:
            await self._concierge(turn, emit, on_pause)
            return

        turn.notes = legal_db.corpus_notes(turn.message)
        for note in turn.notes:
            emit(events.notice(note, level="warn"))
        if plan.needs_state and not conversation.state:
            emit(events.notice(
                "This depends on which state you are in, and the database holds central law. "
                "Set your state above for a more precise answer.", level="info",
                needs_state=True))

        # 3 — exact lookup short-circuit: a named provision deserves the provision.
        exact = legal_db.lookup(turn.message)
        if exact:
            emit(events.tool("legal_db", "exact citation lookup", "done", count=len(exact)))
            turn.evidence.extend(exact)

        # 4 — research rounds
        await self._research(turn, emit, on_pause)

        # 5 — procedure extraction, when the question is a how-to
        if plan.answer_kind in ("procedure", "mixed") and not turn.budget.expired:
            await self._extract_procedure(turn, emit, on_pause)

        # 6 — write
        await self._write_answer(turn, emit, on_pause)

    # ── research loop ────────────────────────────────────────────────────────────────
    async def _research(self, turn: TurnState, emit, on_pause) -> None:
        plan = turn.plan
        steps = list(plan.steps)
        seen_queries: set[str] = set()

        for round_no in range(1, turn.budget.max_rounds + 1):
            if turn.cancelled.is_set() or turn.budget.expired or not steps:
                break
            if round_no > 1:
                emit(events.stage(f"round{round_no}", f"Digging deeper (round {round_no})"))

            gathered = await self._run_steps(turn, steps, emit, on_pause, seen_queries)
            turn.evidence = dedupe(turn.evidence + gathered)
            assign_ids(turn.evidence)

            # An exact citation lookup cannot be off-topic — the user named the provision and
            # we fetched that provision. Grading it wastes a model call, which at ~2.6s per
            # call under load is a third of a quick turn.
            only_exact = turn.evidence and all(
                e.meta.get("exact_lookup") for e in turn.evidence)
            if only_exact:
                emit(events.stage("grade", "Checking which sources are actually relevant",
                                  "done", detail="exact citation — no grading needed"))
                assign_ids(turn.evidence)
                for item in turn.evidence:
                    emit(events.source(item))

            # Grade before deciding anything: an ungraded pile makes the gap analyst think it
            # has coverage it does not have.
            elif turn.evidence:
                emit(events.stage("grade", "Checking which sources are actually relevant"))
                before = len(turn.evidence)
                turn.evidence = await stages.grade(
                    plan.normalized_query or turn.message, turn.evidence,
                    deadline=turn.budget.deadline, on_pause=on_pause, session=turn.session_id)
                assign_ids(turn.evidence)
                emit(events.stage("grade", "Checking which sources are actually relevant",
                                  "done", detail=f"kept {len(turn.evidence)} of {before}"))
                for item in turn.evidence:
                    emit(events.source(item))

            if round_no >= turn.budget.max_rounds or turn.budget.expired:
                break

            emit(events.stage("gaps", "Deciding whether anything is still missing"))
            coverage = await stages.find_gaps(
                plan.normalized_query or turn.message, plan.sub_questions, turn.evidence,
                deadline=turn.budget.deadline, on_pause=on_pause, session=turn.session_id)
            if coverage.enough or not coverage.gaps:
                emit(events.stage("gaps", "Deciding whether anything is still missing", "done",
                                  detail=coverage.note or "nothing important missing"))
                break

            emit(events.stage("gaps", "Deciding whether anything is still missing", "done",
                              detail=f"{len(coverage.gaps)} gap(s) to close"))
            from .agents.schemas import ResearchStep

            steps = [ResearchStep(tool=g.tool, query=g.query or g.missing,
                                  reason=g.missing, sub_question=g.sub_question)
                     for g in coverage.gaps[:3] if (g.query or g.missing)]

    async def _run_steps(self, turn: TurnState, steps, emit, on_pause,
                         seen_queries: set[str]) -> list[Evidence]:
        """Execute one round's planned steps in parallel."""
        tasks = []
        for step in steps:
            key = f"{step.tool}:{(step.query or '').lower().strip()}"
            if not step.query or key in seen_queries:
                continue
            seen_queries.add(key)
            tasks.append(self._run_step(turn, step, emit, on_pause))
        if not tasks:
            return []

        gathered: list[Evidence] = []
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                if isinstance(result, (NimDeadlineExceeded, asyncio.CancelledError)):
                    raise result
                log.warning("a research step failed: %s", result)
                continue
            gathered.extend(result)
        return gathered

    async def _run_step(self, turn: TurnState, step, emit, on_pause) -> list[Evidence]:
        label = STAGE_LABELS.get(step.tool, step.tool)
        emit(events.tool(step.tool, step.query, "running", detail=label))
        started = time.monotonic()
        items: list[Evidence] = []

        try:
            if step.tool == "legal_db":
                # The corpus is English. A Hinglish or Hindi query searched verbatim retrieves
                # noise — "police ne bina warrant arrest kar liya" pulled in the Navy Act and
                # the Metro Railways Act — so retrieval uses the planner's English restatement
                # while the answer still comes back in the user's language.
                query = step.query
                if turn.plan and turn.plan.language != "en" and turn.plan.normalized_query:
                    query = turn.plan.normalized_query

                # Reformulations are written once per turn, not once per step. A planner that
                # emits three statute steps would otherwise cost three extra model calls for
                # variants of the same question, and multi-query fusion already searches every
                # variant against every step.
                if turn.query_variants is None and turn.budget.depth != "quick":
                    written = await stages.write_queries(
                        turn.plan.normalized_query or turn.message,
                        deadline=turn.budget.deadline, on_pause=on_pause,
                        session=turn.session_id)
                    turn.query_variants = written.statute_queries
                items = await legal_db.search(
                    query, variants=turn.query_variants,
                    deadline=turn.budget.deadline, on_pause=on_pause, session=turn.session_id)

            elif step.tool == "wikipedia":
                items = await wikipedia.lookup(step.query)

            elif step.tool in ("web", "official"):
                finder = web.search_official if step.tool == "official" else web.search
                items = await finder(step.query)
                items = await self._read_pages(turn, items, step.query, emit, on_pause)

            elif step.tool == "navigate":
                items = await self._navigate(turn, step, emit, on_pause)

        except (NimDeadlineExceeded, asyncio.CancelledError):
            raise
        except Exception as exc:
            self.ledger.record_tool(step.tool, ok=False)
            emit(events.tool(step.tool, step.query, "error", detail=str(exc)[:120],
                             elapsed_ms=int((time.monotonic() - started) * 1000)))
            return []

        self.ledger.record_tool(step.tool, ok=True)
        emit(events.tool(step.tool, step.query, "done", count=len(items),
                         elapsed_ms=int((time.monotonic() - started) * 1000)))
        return items

    async def _read_pages(self, turn: TurnState, found: list[Evidence], query: str,
                          emit, on_pause) -> list[Evidence]:
        """Turn search snippets into read pages, best sources first.

        A snippet says a fee exists; the page says what it is. Worth the seconds when the
        budget allows, skipped without ceremony when it does not.
        """
        if not found or not turn.budget.can_crawl():
            return found

        ranked = sorted(found, key=lambda e: -e.tier)
        room = min(len(ranked), turn.budget.max_crawls - turn.budget.crawls, 3)
        urls = [e.url for e in ranked[:room] if e.url]
        if not urls:
            return found

        emit(events.tool("crawl", f"reading {len(urls)} page(s)", "running"))
        started = time.monotonic()
        try:
            pages = await crawl.get_crawler().fetch(urls, query)
        except Exception as exc:
            emit(events.tool("crawl", "reading pages", "error", detail=str(exc)[:120]))
            return found

        turn.budget.crawls += len(urls)
        read = crawl.to_evidence(pages, query)
        emit(events.tool("crawl", f"read {len(read)} page(s)", "done", count=len(read),
                         elapsed_ms=int((time.monotonic() - started) * 1000)))

        # Keep snippets only for pages we could not read.
        read_urls = {e.url.rstrip("/") for e in read}
        leftovers = [e for e in found if e.url.rstrip("/") not in read_urls]
        combined = read + leftovers

        return await reduce_pages(combined, query, deadline=turn.budget.deadline,
                                  on_pause=on_pause, session=turn.session_id)

    async def _navigate(self, turn: TurnState, step, emit, on_pause) -> list[Evidence]:
        """Walk an official portal for an end-to-end procedure."""
        seed = step.query.strip()
        goal = step.reason or turn.plan.normalized_query or turn.message

        if not seed.startswith("http"):
            # The planner named a portal rather than a URL — find it first.
            found = await web.search_official(seed or goal, n=3)
            seed = next((e.url for e in found if e.tier >= config.TIER_OFFICIAL), "")
            if not seed:
                return found
        if not turn.budget.can_crawl(2):
            return []

        pages = await crawl.get_crawler().navigate(
            seed, goal, max_pages=max(3, turn.budget.max_crawls - turn.budget.crawls),
            max_depth=turn.budget.nav_depth,
            should_cancel=lambda *_: turn.cancelled.is_set())
        turn.budget.crawls += len(pages)
        emit(events.tool("navigate", seed, "done", count=len(pages),
                         detail=f"visited {len(pages)} page(s)"))
        items = crawl.to_evidence(pages, goal)
        return await reduce_pages(items, goal, deadline=turn.budget.deadline,
                                  on_pause=on_pause, session=turn.session_id)

    # ── procedure ────────────────────────────────────────────────────────────────────
    async def _extract_procedure(self, turn: TurnState, emit, on_pause) -> None:
        sources = [e for e in turn.evidence if e.kind in ("official", "web")]
        if not sources:
            return
        emit(events.stage("procedure", "Assembling the steps"))
        procedure = await stages.extract_procedure(
            turn.plan.normalized_query or turn.message, sources,
            deadline=turn.budget.deadline, on_pause=on_pause, session=turn.session_id)
        if procedure.is_useful:
            turn.procedure = procedure
            emit(events.procedure(procedure.model_dump()))
            emit(events.stage("procedure", "Assembling the steps", "done",
                              detail=f"{len(procedure.steps)} step(s)"))
        else:
            emit(events.stage("procedure", "Assembling the steps", "done",
                              detail="no clear procedure found"))

    # ── writing ──────────────────────────────────────────────────────────────────────
    async def _write_answer(self, turn: TurnState, emit, on_pause, degraded: bool = False):
        plan = turn.plan
        conversation = turn.conversation

        # Sections already vetted earlier in this conversation are free to reuse.
        recalled = [e for e in conversation.recall(turn.message)
                    if e.dedupe_key() not in {x.dedupe_key() for x in turn.evidence}]
        items = assign_ids(dedupe(turn.evidence + recalled[:3]))

        history = conversation.history_block(600)
        context = prompts.writer_context(plan, conversation.state, turn.notes,
                                         stages.today_str())
        reserved = ctx_budget.estimate_tokens(history + context + turn.message) + 200
        packed = packer.pack(items, ctx_budget.Budget.for_writer(), reserved_tokens=reserved)

        # The packer is the last word on ids, so this is the set the writer may cite and the
        # set the UI must be able to resolve a chip against.
        emit(events.sources_final(packed.included))

        if packed.dropped:
            emit(events.notice(
                f"Using the {len(packed.included)} strongest sources; "
                f"{len(packed.dropped)} more were set aside to stay within the context budget.",
                level="info"))

        sources_block = packed.text or packer.render_empty_note(turn.notes)
        procedure_block = ""
        if turn.procedure is not None:
            procedure_block = ("\n\nEXTRACTED PROCEDURE (from the official sources above):\n"
                               + turn.procedure.model_dump_json(indent=None))

        user = (f"{history}\n\nQUESTION: {turn.message}\n"
                f"ANSWER SHAPE: {plan.answer_kind}\n\n{context}\n\n"
                f"VETTED SOURCES — cite only these, by id:\n{sources_block}{procedure_block}")

        emit(events.stage("write", "Writing the answer"))
        collected: list[str] = []
        try:
            stream = self.client.chat_stream(
                [{"role": "system", "content": prompts.WRITER},
                 {"role": "user", "content": user}],
                role="writer", stage="write", deadline=turn.budget.deadline + 45,
                on_pause=on_pause, session=turn.session_id,
                on_reasoning=lambda delta: emit(events.reasoning(delta)),
            )
            async for delta in stream:
                if turn.cancelled.is_set():
                    break
                collected.append(delta)
                emit(events.token(delta))
        except (NimError, NimDeadlineExceeded) as exc:
            if not collected:
                # The research succeeded and only the prose failed. Handing back the provisions
                # we actually found is far more useful than an error — measured during a
                # provider slowdown, this is the difference between a usable answer and a
                # blank screen.
                log.warning("writer unavailable (%s) — falling back to a source digest", exc)
                emit(events.notice(
                    "The writing model could not be reached, so here are the provisions found "
                    "for your question, unedited.", level="warn"))
                digest = _source_digest(packed.included, turn.message)
                for line in digest.splitlines(keepends=True):
                    emit(events.token(line))
                turn.answer = digest
                conversation.add_assistant(digest, packed.included)
                emit(events.verdict(len(packed.included), [],
                                    coverage="written without the model", degraded=True))
                return

        answer = "".join(collected)
        cleaned, unsupported, verified = stages.verify_citations(answer, packed.included)
        if cleaned != answer:
            # The prose already streamed; tell the UI to use the corrected text.
            emit(events.Event("answer_revised", {"text": cleaned}))
        turn.answer = cleaned

        cited = stages.used_evidence(cleaned, packed.included)
        emit(events.verdict(verified, unsupported,
                            coverage=f"{len(cited)}/{len(packed.included)} sources cited",
                            degraded=degraded))
        if unsupported:
            emit(events.notice(
                f"Removed {len(unsupported)} citation marker(s) that did not match any source.",
                level="warn"))

        emit(events.stage("write", "Writing the answer", "done"))
        conversation.add_assistant(cleaned, packed.included)

        if conversation.needs_summary:
            pending = conversation.pending_for_summary()
            if pending:
                text = "\n".join(f"{t.role}: {t.content}" for t in pending)
                summary = await stages.summarise(text, session=turn.session_id)
                if summary:
                    conversation.set_summary(
                        (conversation.summary + "\n" + summary).strip(),
                        len(conversation.turns) - config.HISTORY_TURNS_VERBATIM * 2)

    async def _concierge(self, turn: TurnState, emit, on_pause) -> None:
        history = turn.conversation.history_block(400)
        collected: list[str] = []
        try:
            stream = self.client.chat_stream(
                [{"role": "system", "content": prompts.CONCIERGE},
                 {"role": "user", "content": f"{history}\nUSER: {turn.message}"}],
                role="writer", stage="concierge", max_tokens=220,
                deadline=turn.budget.deadline, on_pause=on_pause, session=turn.session_id)
            async for delta in stream:
                collected.append(delta)
                emit(events.token(delta))
        except Exception as exc:
            log.debug("concierge failed: %s", exc)
            if not collected:
                fallback = "Hello — ask me anything about your rights under Indian law."
                collected.append(fallback)
                emit(events.token(fallback))
        turn.answer = "".join(collected)
        turn.conversation.add_assistant(turn.answer, [])

    # ── reporting ────────────────────────────────────────────────────────────────────
    def _usage(self, turn: TurnState) -> dict:
        from .runtime import resources

        snapshot = self.ledger.snapshot()
        live = resources.live_usage()
        return {
            "llm_calls": max(0, snapshot["total_calls"] - turn.calls_at_start),
            "llm_calls_session": snapshot["total_calls"],
            "estimated_credits": snapshot["estimated_credits"],
            "budget_pressure": snapshot["budget_pressure"],
            "tools": snapshot["tools"],
            "elapsed_s": round(time.monotonic() - turn.started, 1),
            "depth": turn.budget.depth,
            "crawls": turn.budget.crawls,
            "sources": len(turn.evidence),
            "vram_free_mb": live.get("vram_free_mb"),
            "ram_available_mb": live.get("ram_available_mb"),
            "throttled": any(b.get("throttled") for b in self.client.status()["limiters"]),
        }


def _source_digest(items: list[Evidence], question: str) -> str:
    """A readable answer assembled without a model, for when the writer cannot be reached.

    Deliberately quotes rather than paraphrases: with no model in the loop there is nothing to
    do the summarising, and a verbatim provision with its citation is still a real answer.
    """
    if not items:
        return ("I could not reach the writing model, and no sources were found for this "
                "question. Please try again in a moment.")

    lines = [f"**Provisions found for:** {question}", ""]
    statutes = [i for i in items if i.is_statute]
    others = [i for i in items if not i.is_statute]

    for item in statutes[:4]:
        excerpt = " ".join(item.text.split())[:420]
        flags = []
        if item.state:
            flags.append(f"applies only in {item.state}")
        if item.is_omitted:
            flags.append("this provision has been omitted")
        note = f"  _({'; '.join(flags)})_" if flags else ""
        lines.append(f"**{item.label()}**{note}")
        lines.append(f"> {excerpt}…")
        lines.append("")

    if others:
        lines.append("**Also found:**")
        for item in others[:4]:
            link = f"[{item.title[:70]}]({item.url})" if item.url else item.title[:70]
            lines.append(f"- {link}")
        lines.append("")

    lines.append("_This is the raw statutory text, not an explanation — the model that writes "
                 "the plain-language answer was unavailable. Try again shortly._")
    return "\n".join(lines)


_ORCHESTRATOR: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = Orchestrator()
    return _ORCHESTRATOR
