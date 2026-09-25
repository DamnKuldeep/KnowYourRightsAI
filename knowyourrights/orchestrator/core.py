"""The turn lifecycle: plan, research, write, verify, commit.

A turn runs as a background task feeding an event queue rather than as a plain generator. A
rate-limit pause happens deep inside the model client, in a callback that cannot ``yield``; with
a queue that callback still puts a countdown on the reader's screen the moment it happens.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Callable

from .. import config, events, safety
from ..agents import planning, prompts
from ..context.memory import Conversation
from ..llm.client import get_client
from ..llm.errors import DeadlineExceeded, LLMError, ProviderAuthError
from ..llm.spend import SpendMeter, metered
from ..tools import legal_db
from .research import extract_procedure, research
from .turn import TurnBudget, TurnState, degradable
from .verify import self_verify
from .writer import write_answer

log = logging.getLogger(__name__)

PLAN_LABEL = "Understanding your question"
NOTE_NO_PLANNER = ("The planning model is unavailable, so your question is being researched "
                   "directly, without a tailored plan.")
NOTE_OUT_OF_TIME = "This took longer than the time budget allowed, so here is what was found."
AUTH_FAILURE = ("The service could not authenticate with its AI provider, so it cannot answer "
                "right now. The operator needs to check the API key.")
NOT_CONFIGURED = ("The service has no AI provider configured, so it cannot answer. The operator "
                  "needs to set OPENROUTER_API_KEY.")


class Orchestrator:
    def __init__(self) -> None:
        self._active: dict[str, tuple[TurnState, asyncio.Task]] = {}
        # asyncio keeps only weak references to tasks; hold ours until they finish.
        self._background: set[asyncio.Task] = set()

    # ── public entry points ──────────────────────────────────────────────────────────
    async def stream(self, message: str, conversation: Conversation, *,
                     depth: str | None = None, state: str | None = None,
                     on_charge: Callable[[float], None] | None = None,
                     ) -> AsyncIterator[events.Event]:
        """Yield the events of one turn.

        ``state`` is the reader's current selection: a state name, or "" for All India. None
        leaves the conversation's state unchanged. ``on_charge`` receives every billed dollar.
        """
        if state is not None:
            conversation.state = state
        # One turn per conversation: a second tab or a double submit replaces the first rather
        # than interleaving two answers into one history.
        self.cancel(conversation.session_id)
        queue: asyncio.Queue = asyncio.Queue()
        turn = TurnState(session_id=conversation.session_id or uuid.uuid4().hex[:12],
                         message=message.strip(), conversation=conversation,
                         budget=TurnBudget.for_depth(depth or "standard"),
                         meter=SpendMeter(on_charge=on_charge), emit=queue.put_nowait)
        task = asyncio.create_task(self._run(turn, depth))
        turn_id = uuid.uuid4().hex
        self._active[turn_id] = (turn, task)
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            self._active.pop(turn_id, None)
            if not task.done():
                turn.cancelled.set()
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def cancel(self, session_id: str) -> bool:
        """Stop every running turn of a conversation. True if there was one."""
        found = False
        for turn, _ in list(self._active.values()):
            if turn.session_id == session_id:
                turn.cancelled.set()
                found = True
        return found

    # ── the turn ─────────────────────────────────────────────────────────────────────
    async def _run(self, turn: TurnState, requested_depth: str | None) -> None:
        turn.on_pause = _pause_reporter(turn)
        with metered(turn.meter):
            try:
                await self._pipeline(turn, requested_depth)
            except asyncio.CancelledError:
                turn.emit(events.notice("Stopped.", level="info"))
            except DeadlineExceeded as exc:
                log.warning("turn ran out of time: %s", exc)
                turn.emit(events.notice(NOTE_OUT_OF_TIME, level="warn"))
                await self._write_safely(turn, degraded=True)
            except Exception as exc:
                self._report_failure(turn, exc)
            finally:
                self._commit(turn)
                turn.emit(events.usage(**self._usage(turn)))
                turn.emit(events.done(elapsed_s=turn.elapsed_s))
                turn.emit(None)

    async def _pipeline(self, turn: TurnState, requested_depth: str | None) -> None:
        turn.conversation.add_user(turn.message)
        safety_check = _start_safety_gate(turn)
        if not any(config.provider_available(p) for p in config.PROVIDERS):
            # Nothing downstream can work. The safety card, if any, has already been shown.
            turn.emit(events.error(NOT_CONFIGURED, recoverable=False))
            return
        try:
            await self._plan(turn, requested_depth)
        finally:
            # Settled before research begins, so a helpline card always precedes it.
            if safety_check is not None:
                await asyncio.gather(safety_check, return_exceptions=True)
        if turn.plan.is_conversational or not turn.plan.steps:
            await self._concierge(turn)
            return
        self._announce_caveats(turn)
        await self._gather(turn)
        if turn.plan.answer_kind in ("procedure", "mixed") and not turn.budget.expired:
            await extract_procedure(turn)
        await write_answer(turn)
        await self_verify(turn)

    @staticmethod
    async def _gather(turn: TurnState) -> None:
        """Fetch any provision the question names, then research unless that settles it."""
        exact = legal_db.lookup(turn.message)
        turn.evidence.extend(exact)
        if exact:
            turn.emit(events.tool("legal_db", "exact citation lookup", "done", count=len(exact)))
        # A quick question naming a provision is answered by the provision itself; searching
        # anyway only mixed fuzzy matches in with the verbatim text.
        if not (exact and turn.budget.depth == "quick"):
            await research(turn)
            return
        turn.stage("grade", "Checking which sources are actually relevant", "done",
                   "exact citation — no search or grading needed")
        for item in turn.evidence:
            turn.emit(events.source(item))

    async def _plan(self, turn: TurnState, requested_depth: str | None) -> None:
        turn.stage("plan", PLAN_LABEL)
        turn.plan = await degradable(
            turn, planning.make_plan(turn.message, turn.conversation.history_block(),
                                     forced_depth=requested_depth, **turn.call_kwargs()),
            None, NOTE_NO_PLANNER)
        if turn.plan is None:
            turn.plan = planning.apply_rules(planning.fallback_plan(turn.message), turn.message,
                                             requested_depth)
        plan = turn.plan
        # The planner's depth governs unless the reader forced one.
        if requested_depth not in config.DEPTHS:
            turn.budget = TurnBudget.for_depth(plan.depth, started=turn.started)
        turn.stage("plan", PLAN_LABEL, "done", f"{plan.kind} · {plan.depth}")
        turn.emit(events.plan(plan.depth, plan.answer_kind, [s.text for s in plan.sub_questions],
                              [{"tool": s.tool, "query": s.query, "reason": s.reason}
                               for s in plan.steps],
                              language=plan.language, needs_state=plan.needs_state))

    @staticmethod
    def _announce_caveats(turn: TurnState) -> None:
        turn.notes = legal_db.corpus_notes(turn.message)
        for note in turn.notes:
            turn.emit(events.notice(note, level="warn"))
        if turn.plan.needs_state and not turn.conversation.state:
            turn.emit(events.notice(
                "This depends on which state you are in, and the database holds central law. "
                "Set your state above for a more precise answer.", level="info",
                needs_state=True))

    async def _write_safely(self, turn: TurnState, degraded: bool) -> None:
        """Write from what exists after research was cut short; a plan may not even exist."""
        if turn.plan is None:
            turn.plan = planning.fallback_plan(turn.message)
        try:
            await write_answer(turn, degraded=degraded)
        except LLMError as exc:
            self._report_failure(turn, exc)

    @staticmethod
    def _report_failure(turn: TurnState, exc: Exception) -> None:
        """Say what failed without leaking internals; the log gets the detail and a reference."""
        if isinstance(exc, ProviderAuthError):
            log.error("provider rejected its key: %s", exc)
            turn.emit(events.error(AUTH_FAILURE, recoverable=False))
            return
        reference = uuid.uuid4().hex[:8]
        log.exception("turn failed (ref %s)", reference, exc_info=exc)
        turn.emit(events.error(f"Something went wrong on our side (ref {reference}). "
                               f"Please try again."))

    async def _concierge(self, turn: TurnState) -> None:
        """Small talk and questions about the assistant itself: a short streamed reply."""
        history = turn.conversation.history_block(400)
        chunks: list[str] = []
        try:
            async for delta in get_client().chat_stream(
                    [{"role": "system", "content": prompts.CONCIERGE},
                     {"role": "user", "content": f"{history}\nUSER: {turn.message}"}],
                    role="writer", stage="concierge", max_tokens=220,
                    deadline=turn.budget.deadline, on_pause=turn.on_pause,
                    session=turn.session_id):
                chunks.append(delta)
                turn.emit(events.token(delta))
        except LLMError as exc:
            log.debug("concierge failed: %s", exc)
        if not chunks:
            chunks = ["Hello — ask me anything about your rights under Indian law."]
            turn.emit(events.token(chunks[0]))
        turn.answer = "".join(chunks)

    # ── committing the turn ──────────────────────────────────────────────────────────
    def _commit(self, turn: TurnState) -> None:
        """Record the answer in the conversation once, whatever path the turn took.

        Deep mode writes twice (a draft, then a revision). Recording inside the writer stored
        both, feeding the draft that verification had just found wrong back into later turns.
        A turn the reader stopped keeps what they saw, since a follow-up may refer to it.
        """
        if turn.committed or not turn.answer.strip():
            return
        turn.committed = True
        conversation = turn.conversation
        conversation.add_assistant(turn.answer, turn.final_evidence)
        if conversation.needs_summary and conversation.pending_for_summary():
            # Off the critical path: awaiting it held the Stop button up after the answer.
            task = asyncio.create_task(self._summarise(conversation, turn.session_id))
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    @staticmethod
    async def _summarise(conversation: Conversation, session: str) -> None:
        """Fold older turns into a rolling summary, bounded rather than ever-growing."""
        from ..agents import stages

        pending = conversation.pending_for_summary()
        if not pending:
            return
        # Captured before the await: turns added meanwhile must not be skipped.
        upto = conversation.summarised_upto + len(pending)
        text = "\n".join(f"{t.role}: {t.content}" for t in pending)
        if conversation.summary:
            text = f"SUMMARY SO FAR:\n{conversation.summary}\n\nNEW TURNS TO FOLD IN:\n{text}"
        summary = await stages.summarise(text, session=session)
        if summary:
            conversation.set_summary(summary, upto)

    async def aclose(self) -> None:
        """Wait briefly for background summaries, then cancel what is left."""
        if self._background:
            _, pending = await asyncio.wait(self._background, timeout=5)
            for task in pending:
                task.cancel()

    # ── reporting ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _usage(turn: TurnState) -> dict:
        limiters = get_client().status()["limiters"]
        return {
            "llm_calls": turn.meter.calls,
            "cost_usd": round(turn.meter.cost_usd, 6),
            "elapsed_s": turn.elapsed_s,
            "depth": turn.budget.depth,
            "crawls": turn.budget.crawls,
            "sources": len(turn.evidence),
            "degraded": bool(turn.degraded),
            "throttled": any(b.get("throttled") for b in limiters),
        }


def _start_safety_gate(turn: TurnState) -> asyncio.Task | None:
    """Emergencies come before research.

    Literal phrasings decide instantly with no model or network, so a disclosure gets its
    helpline card before anything else. The meaning tier needs one embedding and runs alongside
    the planner rather than in front of it.
    """
    literal = safety.check_patterns(turn.message)
    if literal.urgent:
        turn.emit(events.safety(list(config.HELPLINES), literal.reason))
        return None
    task = asyncio.create_task(safety.check(turn.message))

    def announce(done: asyncio.Task) -> None:
        if not done.cancelled() and done.exception() is None and done.result().urgent:
            turn.emit(events.safety(list(config.HELPLINES), done.result().reason))

    task.add_done_callback(announce)
    return task


def _pause_reporter(turn: TurnState):
    """Turn a wait inside the model client into something the reader can see."""
    def on_pause(model: str, seconds: float, reason: str) -> None:
        if reason == "rate limit":
            turn.emit(events.notice(
                f"The AI service is rate-limited. Waiting {seconds:.0f}s, then continuing where "
                f"it left off.", level="pause", resume_in_s=seconds, model=model))
        elif seconds >= 3:
            turn.emit(events.notice(f"Pacing requests to stay within the provider's limits "
                                    f"({seconds:.0f}s).", level="info", resume_in_s=seconds))
    return on_pause


_ORCHESTRATOR: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = Orchestrator()
    return _ORCHESTRATOR
