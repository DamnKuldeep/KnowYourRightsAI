"""Conversation memory: what a follow-up question can see.

Each bug here was silent: nothing crashed, the answers just got worse as a conversation ran.
"""

from __future__ import annotations

from knowyourrights import config
from knowyourrights.context.budget import estimate_tokens
from knowyourrights.context.memory import Conversation
from knowyourrights.evidence import Evidence

LONG_ANSWER = ("Under Section 58 of the BNSS the police cannot hold you more than twenty-four "
               "hours without a magistrate's order. " * 40)


def _chat(pairs: int, *, state: str = "") -> Conversation:
    c = Conversation(session_id="t", state=state)
    for i in range(pairs):
        c.add_user(f"question number {i}")
        c.add_assistant(f"ANSWER-{i} " + LONG_ANSWER)
    return c


def _source(title: str, text: str, url: str | None = None) -> Evidence:
    return Evidence(kind="web", title=title, citation=title, text=text, score=0.6,
                    url=url or f"https://example.org/{abs(hash(title))}")


def test_latest_exchange_survives_a_long_history():
    """History was once trimmed from the end, cutting the exchange a follow-up refers to."""
    c = _chat(4)
    c.add_user("what about the appeal?")
    block = c.history_block(900)
    assert "ANSWER-3" in block and "question number 3" in block


def test_users_state_is_never_the_part_that_gets_cut():
    c = _chat(4, state="Kerala")
    c.add_user("and the deposit?")
    assert "USER'S STATE: Kerala" in c.history_block(300)


def test_in_flight_question_is_not_repeated_in_its_own_history():
    c = _chat(1)
    c.add_user("what about the appeal?")
    assert "what about the appeal?" not in c.history_block(900)


def test_one_long_answer_cannot_evict_everything_before_it():
    c = Conversation(session_id="t")
    c.add_user("first question")
    c.add_assistant("short answer one")
    c.add_user("second question")
    c.add_assistant("ANSWER-LONG " + LONG_ANSWER * 4)
    c.add_user("follow-up")
    block = c.history_block(900)
    assert "second question" in block and "ANSWER-LONG" in block and "first question" in block


def test_history_respects_its_budget():
    c = _chat(6, state="Delhi")
    c.add_user("next")
    for limit in (200, 400, 900):
        assert estimate_tokens(c.history_block(limit)) <= limit + 5


def test_summary_is_rolling_not_accumulating():
    c = Conversation(session_id="t")
    c.set_summary("first summary", upto=2)
    c.set_summary("second summary", upto=4)
    assert c.summary == "second summary"


def test_summarised_upto_never_moves_backwards():
    c = Conversation(session_id="t")
    c.set_summary("s", upto=6)
    c.set_summary("s2", upto=4)
    assert c.summarised_upto == 6


def test_the_source_pool_is_bounded(monkeypatch):
    """Unbounded, a long conversation kept every page it had read for six hours."""
    monkeypatch.setattr(config, "SESSION_POOL_MAX", 5)
    c = Conversation(session_id="t")
    for i in range(12):
        c.remember(_source(f"page {i}", "text", url=f"https://example.org/{i}"))
    assert len(c.pool) == 5
    assert any("/11" in key for key in c.pool), "the newest sources are the ones kept"


def test_unrelated_evidence_is_not_recalled():
    """A domestic-violence question once listed rent-deposit pages as its sources."""
    c = Conversation(session_id="t")
    c.remember(_source("Recover Rental Security Deposit from Landlord in India",
                       "Legal steps to recover a security deposit from a landlord under the "
                       "rent agreement: send a legal notice, then file a suit under law."))
    assert c.recall("what legal protection does a woman have under the law if her husband "
                    "is hitting her") == []


def test_related_evidence_is_still_recalled():
    c = Conversation(session_id="t")
    c.remember(_source("Section 58, Bharatiya Nagarik Suraksha Sanhita, 2023",
                       "No police officer shall detain in custody a person arrested without "
                       "warrant for longer than twenty-four hours, exclusive of the journey "
                       "to the Magistrate's Court."))
    got = c.recall("how long can the police detain an arrested person before producing them "
                   "before a magistrate")
    assert got and "Section 58" in got[0].title
