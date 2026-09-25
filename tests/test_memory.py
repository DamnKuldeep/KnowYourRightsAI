"""Conversation memory — what a follow-up question can see.

Every test here is a regression from an audit of the real code. Each bug was silent: nothing
crashed, the answers just quietly got worse the longer a conversation ran.
"""

from __future__ import annotations

import asyncio

import pytest

from knowyourrights import config
from knowyourrights.context.memory import Conversation

LONG_ANSWER = ("Under Section 58 of the BNSS the police cannot hold you more than twenty-four "
               "hours without a magistrate's order. " * 40)          # ~3,500 chars, a real answer


def _chat(pairs: int, *, state: str = "") -> Conversation:
    c = Conversation(session_id="t", state=state)
    for i in range(pairs):
        c.add_user(f"question number {i}")
        c.add_assistant(f"ANSWER-{i} " + LONG_ANSWER)
    return c


def test_latest_exchange_survives_a_long_history():
    """Regression: history was built oldest-first and then trimmed by keeping the *start*, so
    the most recent exchange was the first thing cut — from about the second follow-up on."""
    c = _chat(4)
    c.add_user("what about the appeal?")
    block = c.history_block(900)
    assert "ANSWER-3" in block, "the answer a follow-up refers to must be kept"
    assert "question number 3" in block


def test_users_state_is_never_the_part_that_gets_cut():
    """It was appended last, so it went first. It decides which law applies."""
    c = _chat(4, state="Kerala")
    c.add_user("and the deposit?")
    assert "USER'S STATE: Kerala" in c.history_block(300)


def test_in_flight_question_is_not_repeated_in_its_own_history():
    """Regression: the current message appeared twice in every prompt."""
    c = _chat(1)
    c.add_user("what about the appeal?")
    assert "what about the appeal?" not in c.history_block(900)


def test_one_long_answer_cannot_evict_everything_before_it():
    c = Conversation(session_id="t")
    c.add_user("first question")
    c.add_assistant("short answer one")
    c.add_user("second question")
    c.add_assistant("ANSWER-LONG " + LONG_ANSWER * 4)          # ~14,000 chars
    c.add_user("follow-up")
    block = c.history_block(900)
    assert "second question" in block and "ANSWER-LONG" in block
    assert "first question" in block, "a per-answer cap must leave room for earlier turns"


def test_history_respects_its_budget():
    from knowyourrights.context.budget import estimate_tokens
    c = _chat(6, state="Delhi")
    c.add_user("next")
    for limit in (200, 400, 900):
        assert estimate_tokens(c.history_block(limit)) <= limit + 5


def test_summary_is_rolling_not_accumulating():
    """Regression: each new summary was appended to the old one and never re-compressed."""
    c = Conversation(session_id="t")
    c.set_summary("first summary", upto=2)
    c.set_summary("second summary", upto=4)
    assert c.summary == "second summary"
    assert "first summary" not in c.summary


def test_summarised_upto_never_moves_backwards():
    c = Conversation(session_id="t")
    c.set_summary("s", upto=6)
    c.set_summary("s2", upto=4)
    assert c._summarised_upto == 6


# ── the orchestrator's single commit ──────────────────────────────────────────────────
def _turn(conversation, answer="final answer"):
    from knowyourrights.orchestrator import TurnBudget, TurnState
    t = TurnState(session_id="t", message="q", conversation=conversation,
                  budget=TurnBudget.for_depth("quick"))
    t.answer = answer
    return t


def test_answer_is_committed_exactly_once():
    """Regression: deep mode wrote the draft *and* the revision into history — and the draft
    was the text self-verification had just found wrong."""
    from knowyourrights.orchestrator import Orchestrator
    orch = Orchestrator()
    c = Conversation(session_id="t")
    c.add_user("q")
    turn = _turn(c, "the revised answer")
    orch._commit(turn)
    orch._commit(turn)                         # a second exit path must be a no-op
    assistant = [t for t in c.turns if t.role == "assistant"]
    assert [t.content for t in assistant] == ["the revised answer"]


def test_empty_answer_is_not_committed():
    from knowyourrights.orchestrator import Orchestrator
    c = Conversation(session_id="t")
    c.add_user("q")
    Orchestrator()._commit(_turn(c, "   "))
    assert all(t.role == "user" for t in c.turns)


def test_writer_never_writes_history_itself():
    """The commit has to live in one place, or a second writer pass reintroduces the bug."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "knowyourrights" / "orchestrator.py"
           ).read_text(encoding="utf-8")
    assert src.count("add_assistant(") == 1, "only Orchestrator._commit may write an answer"


@pytest.mark.asyncio
async def test_summary_runs_off_the_critical_path(monkeypatch):
    """Regression: summarisation was awaited before `done`, holding every long turn open."""
    from knowyourrights.agents import stages
    from knowyourrights.orchestrator import Orchestrator

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_summary(text, **_):
        started.set()
        await release.wait()
        return "rolled up"

    monkeypatch.setattr(stages, "summarise", slow_summary)
    orch = Orchestrator()
    c = _chat(config.HISTORY_SUMMARY_TRIGGER)
    c.add_user("one more")
    orch._commit(_turn(c, "answer"))          # returns immediately
    await asyncio.wait_for(started.wait(), 1)
    assert c.summary == "", "commit must not have waited for the summary"
    release.set()
    await asyncio.gather(*orch._background)
    assert c.summary == "rolled up"


# ── recall: earlier turns' sources offered to a new question ──────────────────────────
def _src(title: str, text: str):
    from knowyourrights.evidence import Evidence
    return Evidence(id="S1", kind="web", title=title, citation=title, text=text,
                    score=0.6, url=f"https://example.org/{abs(hash(title))}")


def test_unrelated_evidence_is_not_recalled():
    """Regression: a domestic-violence question listed rent-deposit pages as its sources,
    because two shared words — "legal" and "under" — were the whole test."""
    c = Conversation(session_id="t")
    c.remember(_src("Recover Rental Security Deposit from Landlord in India",
                    "Legal steps to recover a security deposit from a landlord under the rent "
                    "agreement: send a legal notice, then file a suit under law."))
    assert c.recall("what legal protection does a woman have under the law if her husband "
                    "is hitting her") == []


def test_related_evidence_is_still_recalled():
    """The pool must keep doing its job for genuine follow-ups."""
    c = Conversation(session_id="t")
    c.remember(_src("Section 58, Bharatiya Nagarik Suraksha Sanhita, 2023",
                    "No police officer shall detain in custody a person arrested without "
                    "warrant for longer than twenty-four hours, exclusive of the journey to "
                    "the Magistrate's Court."))
    got = c.recall("how long can the police detain an arrested person before producing them "
                   "before a magistrate")
    assert got and "Section 58" in got[0].title


def test_recalled_evidence_goes_through_the_grader():
    """Recall must happen inside research, before grading — never in the writer after it."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "knowyourrights" / "orchestrator.py"
           ).read_text(encoding="utf-8")
    research = src[src.index("async def _research("):src.index("async def _run_steps(")]
    assert "self._recall(turn)" in research
    assert research.index("self._recall(turn)") < research.index("await stages.grade(")
    writer = src[src.index("async def _write_answer("):src.index("def _commit(")]
    assert ".recall(" not in writer, "the writer must not add ungraded evidence"


def test_jurisdiction_follows_the_matter_not_the_user():
    """Regression: with Kerala selected, a Mumbai flat was told Maharashtra law 'does not govern
    your tenancy' and sent to court in Kerala. Tenancy follows the property."""
    from knowyourrights.agents import prompts
    assert "WHERE THE MATTER IS" in prompts.WRITER
    assert "user lives" in prompts.WRITER


# ── the deep-mode rewrite and the "All India" default ─────────────────────────────────
def test_revision_is_swapped_in_not_streamed_over_the_draft():
    """Regression: the rewrite streamed over the draft — the UI cleared the answer mid-read and
    started again, and printed the sources notice a second time."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "knowyourrights" / "orchestrator.py"
           ).read_text(encoding="utf-8")
    writer = src[src.index("async def _write_answer("):src.index("def _commit(")]
    assert 'stage_id = "revise" if revision else "write"' in writer, \
        "a rewrite must not reuse the 'write' stage id, which the UI clears on"
    assert "emit_token = (lambda delta: None) if revision" in writer, \
        "a rewrite must not stream tokens over the draft"
    assert "if not revision:\n            emit(events.sources_final" in writer, \
        "the sources notice must be emitted once per turn, not once per writer pass"
    verify = src[src.index("async def _self_verify("):src.index("async def _concierge(")]
    assert 'turn.answer = ""' not in verify, "the draft must survive a failed rewrite"


def test_all_india_is_answered_for_all_india():
    """Regression: with no state selected, an RTI question came back with Maharashtra and Delhi
    portals and fees, because nothing said what "All India" meant."""
    from knowyourrights.agents import prompts
    from knowyourrights.agents.schemas import Plan

    plan = Plan(kind="legal_question", depth="standard", answer_kind="procedure",
                normalized_query="how to file an RTI", needs_state=False)
    ctx = prompts.writer_context(plan, None, [], "2026-09-25")
    assert "All India" in ctx and "one state" in ctx

    plan.needs_state = True
    ctx = prompts.writer_context(plan, None, [], "2026-09-25")
    assert "asking which state" in ctx, "when it truly varies, the writer must ask"

    ctx = prompts.writer_context(plan, "Kerala", [], "2026-09-25")
    assert "All India" not in ctx


def test_procedure_steps_appear_once():
    """Regression: the card listed the steps and the answer repeated them. The answer now owns
    the steps; the card carries only the facts."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent / "knowyourrights" / "web" / "app.js"
          ).read_text(encoding="utf-8")
    card = js[js.index("function addProcedure"):js.index("function renderSources")]
    assert "<ol>" not in card and "data.steps" not in card


def test_writer_is_not_the_unreliable_free_model():
    """The free 120B broke 12 of 22 streams in one session, 5 of them after text had started."""
    from knowyourrights import config
    assert not config.is_free_model(config.WRITER_MODELS[0].id)


# ── procedure card, links, and out-of-date statute text ───────────────────────────────
@pytest.mark.parametrize("text,money,ok", [
    ("As prescribed in the RTI Rules, 2012.", True, False),   # a year is not an amount
    ("Payment can be made through internet banking", True, False),
    ("Rs 10; free for BPL applicants", True, True),
    ("₹10", True, True),
    ("Free of charge", True, True),
    ("30 days", False, True),
    ("as soon as possible", False, False),
])
def test_card_facts_must_state_a_value(text, money, ok):
    """Regression: the card said 'Fee: as prescribed in the RTI Rules' beside an answer that
    said ₹10. A fact with no value is dropped rather than shown."""
    from knowyourrights.agents.stages import _states_a_value
    assert _states_a_value(text, money=money) is ok


def test_page_titles_are_not_used_as_link_text():
    from knowyourrights.agents.stages import tidy_link_labels
    out = tidy_link_labels("Visit [RTI Online:: Home | Submit RTI Request | Submit RTI First A]"
                           "(https://rtionline.gov.in/).")
    assert out == "Visit [RTI Online](https://rtionline.gov.in/)."
    assert tidy_link_labels("[RTI Online portal](https://rtionline.gov.in)") == \
        "[RTI Online portal](https://rtionline.gov.in)"


def test_the_2019_jk_change_travels_with_the_source():
    """The corpus still carries the pre-2019 extent clause; the correction must reach both
    the writer and the card, not depend on the model remembering a prompt rule."""
    from knowyourrights.evidence import Evidence
    ev = Evidence(id="S1", kind="statute", title="Right to Information Act, 2005",
                  citation="Section 1", score=0.9, url="", source_type="central_act",
                  text="It extends to the whole of India except the State of Jammu and Kashmir.")
    assert ev.caveats and "2019" in ev.caveats[0]
    assert ev.to_public()["caveats"] == ev.caveats


# ── rules the model follows only sometimes, moved into code ───────────────────────────
def test_procedure_about_an_act_always_searches_its_deadline_and_appeal(monkeypatch):
    """Regression: one RTI how-to cited the 30-day rule; the next, with the planner skipping
    the statute, said "the sources do not state the response time"."""
    from knowyourrights.agents import stages
    from knowyourrights.agents.schemas import Plan, ResearchStep

    planned = Plan(kind="legal_question", answer_kind="procedure",
                   normalized_query="how to file an RTI application",
                   steps=[ResearchStep(tool="web", query="rti online portal")])

    class Client:
        async def chat_json(self, *args, **kwargs):
            return planned.model_copy(deep=True)

    monkeypatch.setattr(stages, "get_client", lambda: Client())
    plan = asyncio.run(stages.make_plan("How do I file an RTI?"))
    statute = [s.query for s in plan.steps if s.tool == "legal_db"]
    assert any("appeal" in q and "Right to Information" in q for q in statute), statute

    planned.answer_kind = "rights"
    plan = asyncio.run(stages.make_plan("Can police arrest me at night?"))
    assert not any("appeal" in s.query for s in plan.steps)


def test_state_question_is_appended_when_the_answer_depends_on_it():
    """Regression: a Hindi deposit question with All India selected was routed to the National
    Consumer Helpline with no word that tenancy law is made by each state."""
    from knowyourrights.agents.stages import state_question
    from knowyourrights.agents.schemas import Plan

    plan = Plan(needs_state=True, language="hi")
    assert "राज्य" in state_question("जमा राशि वापस पाने के लिए…", plan, None)
    assert state_question("…", plan, "Kerala") == ""
    plan.language = "en"
    assert state_question("Which state is your flat in?", plan, None) == ""
    assert state_question("**Which state is your flat in?**", plan, None) == ""
    assert state_question("Which state are you in? [S1]\n", plan, None) == ""
    assert "Which state" in state_question("Send a legal notice.", plan, None)
    plan.needs_state = False
    assert state_question("Send a legal notice.", plan, None) == ""


def test_unstated_lines_are_dropped_but_real_lines_kept():
    from knowyourrights.agents.stages import drop_unstated_lines
    text = ("- **Fee:** ₹10\n"
            "- **Response time:** The sources do not state the response time.\n"
            "- Your sources do not state the fee.\n"
            "- The PIO must reply within 30 days [2].")
    kept = drop_unstated_lines(text)
    assert "₹10" in kept and "30 days" in kept
    assert "do not state" not in kept


def test_a_heading_left_empty_by_the_filter_is_removed():
    from knowyourrights.agents.stages import drop_unstated_lines
    text = ("Send a legal notice first [W1].\n\n**What it costs and how long**\n\n"
            "- **Fee:** The sources do not state the court fee.\n"
            "- **Response time:** Not stated in the provided sources.")
    assert "What it costs" not in drop_unstated_lines(text)
    kept = drop_unstated_lines("**Steps**\n\n- File online [G1].")
    assert "**Steps**" in kept


def test_a_named_place_answers_which_state():
    """Regression: the Delhi Rent Control Act question was answered and then asked which
    state the user was in."""
    from knowyourrights import config, legal_terms
    place = lambda t: legal_terms.place_named(t, config.INDIAN_STATES)   # noqa: E731
    assert place("What does the Delhi Rent Control Act say about eviction?") == "Delhi"
    assert place("My landlord in Mumbai won't return my deposit") == "Maharashtra"
    assert place("Jammu and Kashmir land law") == "Jammu & Kashmir"
    assert place("How do I file an RTI?") is None
    assert place("goals for a legal notice") is None


def test_hindi_unstated_line_is_dropped():
    from knowyourrights.agents.stages import drop_unstated_lines
    text = ("- **शुल्क:** ₹10\n"
            "- **प्रतिक्रिया समय:** स्रोतों में अधिकारी द्वारा प्रतिक्रिया देने की कोई विशिष्ट "
            "समय सीमा नहीं बताई गई है।")
    kept = drop_unstated_lines(text)
    assert "₹10" in kept and "स्रोतों" not in kept


def test_prompt_block_names_are_not_left_as_citations():
    from knowyourrights.agents.stages import normalize_markers
    out = normalize_markers("File a summary suit [EXTRACTED PROCEDURE][W2]. Pay ₹10 [G1].")
    assert "EXTRACTED" not in out and "[W2]" in out and "[G1]" in out


def test_source_card_does_not_show_writer_notes():
    """Regression: Delhi Act cards read "[DELHI ONLY — … Do not present it as all-India law.]"."""
    from knowyourrights.evidence import Evidence
    e = Evidence(kind="statute", title="Section 14", text=(
        "[DELHI ONLY — Parliament passed this for Delhi. Do not present it as all-India law.]\n"
        "14. Protection of tenant against eviction."))
    assert e.to_public()["snippet"].startswith("14. Protection")


def test_reference_to_a_repealed_code_is_corrected():
    """Regression: a domestic-violence answer said its proceedings "are governed by the Code of
    Criminal Procedure, 1973", quoting Section 28 of the PWDVA."""
    from knowyourrights.evidence import Evidence
    e = Evidence(kind="statute", title="Section 28", act_title="Protection of Women from "
                 "Domestic Violence Act, 2005", text="governed by the provisions of the Code of "
                 "Criminal Procedure, 1973 (2 of 1974)")
    assert any("Bharatiya Nagarik Suraksha Sanhita" in c for c in e.caveats)


def test_card_fields_that_say_they_are_empty_are_dropped():
    """Regression: the deep-mode card read "Fee: amount not specified in sources; free for BPL
    applicants is not mentioned" (it contains "free") and "Appeal to: Not specified"."""
    from knowyourrights.agents.stages import _states_a_value, _NOT_GIVEN
    assert not _states_a_value("Application fee (amount not specified in sources); free for "
                               "BPL applicants is not mentioned.", money=True)
    assert _states_a_value("₹10; free for BPL applicants", money=True)
    assert _NOT_GIVEN.search("Not specified in the provided sources.")
    assert not _NOT_GIVEN.search("First Appellate Authority, National Consumer Commission")
