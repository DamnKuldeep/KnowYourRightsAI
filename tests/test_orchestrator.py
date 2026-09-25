"""Whole turns, end to end, with the models and the statute search replaced by fakes.

These check behaviour a reader sees: what streams, what is reported when a stage fails, what is
remembered, and what each turn costs.
"""

from __future__ import annotations

import asyncio

import pytest

from knowyourrights import config, safety
from knowyourrights.agents import grading, planning, stages
from knowyourrights.agents.schemas import Plan, ResearchStep, SafetyCheck
from knowyourrights.context.memory import Conversation
from knowyourrights.evidence import Evidence
from knowyourrights.llm import spend
from knowyourrights.llm.errors import ModelUnavailable, ProviderAuthError
from knowyourrights.orchestrator import Orchestrator, TurnBudget, TurnState, core, writer
from knowyourrights.retrieval.search import NOTE_NO_EMBEDDINGS
from knowyourrights.tools import legal_db

PLAN = Plan(kind="legal_question", depth="standard", answer_kind="rights",
            normalized_query="rights on arrest",
            steps=[ResearchStep(tool="legal_db", query="rights on arrest")])


class FakeClient:
    def __init__(self, *, plan=PLAN, answer="You must be told the grounds [S1].",
                 plan_error=None, writer_error=None, cost=0.0, gate=None):
        self.plan, self.answer, self.cost, self.gate = plan, answer, cost, gate
        self.plan_error, self.writer_error = plan_error, writer_error

    async def chat_json(self, messages, model_cls, default, *, stage="", **_):
        if stage == "plan":
            if self.plan_error:
                raise self.plan_error
            return self.plan.model_copy(deep=True)
        return default

    async def chat_stream(self, messages, *, on_finish=None, **_):
        if self.writer_error:
            raise self.writer_error
        if self.gate is not None:
            await self.gate.wait()
        spend.charge(self.cost)
        for word in self.answer.split(" "):
            yield word + " "
        if on_finish:
            on_finish("stop")

    async def chat(self, *args, **kwargs):
        return ""

    def status(self):
        return {"limiters": []}


def statute() -> Evidence:
    return Evidence(kind="statute", title="Section 47, BNSS", text="grounds of arrest",
                    tier=config.TIER_STATUTE, score=0.9, unit_id="u47",
                    citation="Section 47, Bharatiya Nagarik Suraksha Sanhita, 2023")


@pytest.fixture
def fakes(monkeypatch):
    """Install a fake client everywhere it is used, and a fake statute search."""
    def install(client: FakeClient, degraded: list[str] | None = None):
        for module in (planning, grading, stages, writer, core):
            monkeypatch.setattr(module, "get_client", lambda: client)

        async def search(query, **_):
            return legal_db.StatuteResults([statute()], list(degraded or []))

        monkeypatch.setattr(legal_db, "search", search)
        monkeypatch.setattr(legal_db, "lookup", lambda question: [])

        async def no_emergency(message, *_args, **_kwargs):
            return SafetyCheck()

        monkeypatch.setattr(safety, "check", no_emergency)
        return client
    return install


async def run(orchestrator, message="can the police arrest me", *, conversation=None, **kw):
    conversation = conversation or Conversation(session_id="s1")
    events = [e async for e in orchestrator.stream(message, conversation, **kw)]
    return events, conversation


def of_type(events, kind):
    return [e.data for e in events if e.type == kind]


def text_of(events) -> str:
    return "".join(d["delta"] for d in of_type(events, "token"))


# ── the ordinary path ─────────────────────────────────────────────────────────────────
async def test_a_turn_streams_an_answer_and_commits_it_once(fakes):
    fakes(FakeClient(cost=0.002))
    events, conversation = await run(Orchestrator())
    assert "grounds" in text_of(events)
    assert of_type(events, "verdict")[0]["citations_verified"] == 1
    assert of_type(events, "usage")[0]["cost_usd"] == pytest.approx(0.002)
    assert events[-1].type == "done"
    assert [t.role for t in conversation.turns] == ["user", "assistant"]


async def test_switching_back_to_all_india_clears_the_state(fakes):
    """Regression: once a state was chosen, choosing All India again was silently ignored."""
    fakes(FakeClient())
    orchestrator = Orchestrator()
    _, conversation = await run(orchestrator, state="Kerala")
    assert conversation.state == "Kerala"
    await run(orchestrator, conversation=conversation, state="")
    assert conversation.state == ""


# ── what the reader is told when something fails ─────────────────────────────────────
async def test_a_planner_outage_is_reported_and_research_continues(fakes):
    fakes(FakeClient(plan_error=ModelUnavailable("no fast model")))
    events, _ = await run(Orchestrator())
    notices = [n["text"] for n in of_type(events, "notice")]
    assert core.NOTE_NO_PLANNER in notices
    assert "grounds" in text_of(events), "the question is still answered"


async def test_a_writer_outage_hands_back_the_provisions_found(fakes):
    fakes(FakeClient(writer_error=ModelUnavailable("no writer")))
    events, conversation = await run(Orchestrator())
    assert writer.NOTE_WRITER_DOWN in [n["text"] for n in of_type(events, "notice")]
    assert "Provisions found" in text_of(events)
    assert of_type(events, "verdict")[0]["degraded"] is True
    assert conversation.turns[-1].role == "assistant"


async def test_degraded_retrieval_is_reported_to_the_reader(fakes):
    fakes(FakeClient(), degraded=[NOTE_NO_EMBEDDINGS])
    events, _ = await run(Orchestrator())
    notices = [n for n in of_type(events, "notice") if n.get("degraded")]
    assert [n["text"] for n in notices] == [NOTE_NO_EMBEDDINGS]
    assert of_type(events, "usage")[0]["degraded"] is True


async def test_a_rejected_key_ends_the_turn_with_a_clear_error(fakes):
    fakes(FakeClient(plan_error=ProviderAuthError("openrouter rejected its API key")))
    events, _ = await run(Orchestrator())
    errors = of_type(events, "error")
    assert errors and errors[0]["message"] == core.AUTH_FAILURE
    assert errors[0]["recoverable"] is False
    assert not of_type(events, "token")


async def test_no_configured_provider_is_reported_as_fatal(fakes, monkeypatch):
    fakes(FakeClient())
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    events, _ = await run(Orchestrator())
    assert of_type(events, "error")[0]["message"] == core.NOT_CONFIGURED


async def test_an_unexpected_failure_shows_a_reference_not_internals(fakes, monkeypatch):
    fakes(FakeClient())

    async def broken(*_args, **_kwargs):
        raise RuntimeError("secret internal detail at /srv/app/x.py")

    monkeypatch.setattr(core, "research", broken)
    events, _ = await run(Orchestrator())
    message = of_type(events, "error")[0]["message"]
    assert "ref " in message and "secret" not in message


# ── cost, concurrency and memory ──────────────────────────────────────────────────────
class PricedClient(FakeClient):
    """Charges by question, and yields between words so two turns genuinely interleave."""

    async def chat_stream(self, messages, *, on_finish=None, **_):
        spend.charge(0.009 if "expensive" in messages[-1]["content"] else 0.001)
        for word in self.answer.split(" "):
            await asyncio.sleep(0)
            yield word + " "
        if on_finish:
            on_finish("stop")


async def test_concurrent_turns_each_report_their_own_cost(fakes):
    """Regression: a turn's cost was the change in a process-wide total, so two people asking
    at once each saw the other's spend."""
    fakes(PricedClient())

    async def ask(question: str, session: str) -> tuple[float, float]:
        charged: list[float] = []
        events, _ = await run(Orchestrator(), question,
                              conversation=Conversation(session_id=session),
                              on_charge=charged.append)
        return of_type(events, "usage")[0]["cost_usd"], sum(charged)

    cheap, dear = await asyncio.gather(ask("a cheap question", "a"),
                                       ask("an expensive question", "b"))
    assert cheap == (pytest.approx(0.001), pytest.approx(0.001))
    assert dear == (pytest.approx(0.009), pytest.approx(0.009))


async def test_a_new_question_replaces_the_running_one(fakes):
    gate = asyncio.Event()
    fakes(FakeClient(gate=gate))
    orchestrator = Orchestrator()
    conversation = Conversation(session_id="same")
    first = orchestrator.stream("first", conversation)
    await first.__anext__()                              # the turn is now running
    running = next(iter(orchestrator._active.values()))[0]
    second = orchestrator.stream("second", conversation)
    await second.__anext__()
    assert running.cancelled.is_set(), "the earlier turn must be stopped, not interleaved"
    gate.set()
    await first.aclose()
    await second.aclose()


def _turn(conversation: Conversation, answer: str) -> TurnState:
    turn = TurnState(session_id="t", message="q", conversation=conversation,
                     budget=TurnBudget.for_depth("quick"))
    turn.answer = answer
    return turn


def test_an_answer_is_committed_exactly_once():
    """Deep mode once wrote both the draft and the revision into history."""
    orchestrator, conversation = Orchestrator(), Conversation(session_id="t")
    conversation.add_user("q")
    turn = _turn(conversation, "the revised answer")
    orchestrator._commit(turn)
    orchestrator._commit(turn)
    assert [t.content for t in conversation.turns if t.role == "assistant"] == \
        ["the revised answer"]


async def test_the_summary_runs_off_the_critical_path(monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_summary(text, **_):
        started.set()
        await release.wait()
        return "rolled up"

    monkeypatch.setattr(stages, "summarise", slow_summary)
    orchestrator, conversation = Orchestrator(), Conversation(session_id="t")
    for i in range(config.HISTORY_SUMMARY_TRIGGER):
        conversation.add_user(f"q{i}")
        conversation.add_assistant(f"a{i}")
    conversation.add_user("one more")
    orchestrator._commit(_turn(conversation, "answer"))
    await asyncio.wait_for(started.wait(), 1)
    assert conversation.summary == "", "the commit must not wait for the summary"
    release.set()
    await orchestrator.aclose()
    assert conversation.summary == "rolled up"


async def test_a_revision_replaces_the_draft_instead_of_streaming_over_it(fakes):
    """The rewrite once streamed over the draft, clearing it mid-read."""
    fakes(FakeClient(answer="Corrected: the fee is ten rupees [S1]."))
    emitted = []
    turn = _turn(Conversation(session_id="t"), "Draft: the fee is Rs 50 [S1].")
    turn.plan, turn.evidence, turn.emit = PLAN, [statute()], emitted.append
    await writer.write_answer(turn, revision=True)
    kinds = [e.type for e in emitted]
    assert "token" not in kinds
    assert kinds.index("sources_final") < kinds.index("answer_revised")
    assert turn.answer.startswith("Corrected")
