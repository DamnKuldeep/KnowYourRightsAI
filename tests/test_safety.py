"""The safety gate, scored against the same labelled set the calibration script uses.

Two failure directions, weighted very differently. A miss means someone describing violence
gets a statute lecture instead of 112. A false alarm means someone reading about the law gets a
helpline card they did not need — harmless once, and corrosive repeated, because it teaches
people to ignore the card that matters.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from knowyourrights import safety
from knowyourrights.safety_eval import DISCLOSURES, QUESTIONS


@pytest.mark.parametrize("case", [d for d in DISCLOSURES if d.literal],
                         ids=lambda c: c.text[:38])
def test_literal_disclosures_need_no_model(case):
    """These must reach a helpline with no embedder, no network and no provider available."""
    check = safety.check_patterns(case.text)
    assert check.urgent, f"pattern tier missed a literal disclosure: {case.text!r}"
    assert check.kind == case.kind
    assert check.reason, "an urgent result must carry advice to show"


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q[:38])
def test_legal_questions_never_trigger_a_helpline(question):
    """Asking *about* a crime is not disclosing one.

    Regression: bare topic nouns used to fire, so "is suicide a crime in India" and "child
    labour laws in India" both produced helpline cards. Weak patterns are now gated by the
    informational guard.
    """
    assert not safety.check_patterns(question).urgent, f"false alarm on: {question!r}"


def test_every_kind_has_advice():
    """A fired gate with no advice shows an empty card, which is worse than not firing."""
    kinds = {kind for _pattern, kind, _strong in safety._URGENT_PATTERNS}
    assert kinds <= set(safety.URGENT_ADVICE), "a pattern exists with no advice text"
    assert all(safety.URGENT_ADVICE[k].strip() for k in kinds)


def test_exemplars_cover_every_kind_the_patterns_know():
    """The meaning tier must be able to reach the same kinds the pattern tier can."""
    pattern_kinds = {kind for _p, kind, _s in safety._URGENT_PATTERNS}
    assert pattern_kinds == set(safety.CRISIS_EXEMPLARS), \
        "a crisis kind exists in one tier but not the other"


def test_gate_runs_before_any_model_call():
    """Ordering is the guarantee: a rate limit must never delay a helpline number."""
    src = (Path(__file__).resolve().parent.parent
           / "knowyourrights" / "orchestrator.py").read_text(encoding="utf-8")
    gate = src.index("safety.check(turn.message)")
    plan = src.index("await stages.make_plan(")
    assert gate < plan, "the safety gate must run before the planner"


def test_pattern_tier_is_synchronous():
    """It has to work when the embedder is absent, so it cannot be async or awaitable."""
    assert not inspect.iscoroutinefunction(safety.check_patterns)


@pytest.mark.asyncio
async def test_semantic_tier_degrades_to_patterns_when_embedder_fails():
    """A broken embedder must not cost us the pattern tier, and must never raise."""
    class Broken:
        plan = type("P", (), {"use_embedder": True})()

        async def encode(self, texts):
            raise RuntimeError("no embedder here")

        async def encode_one(self, text):
            raise RuntimeError("no embedder here")

    safety.reset_cache()
    try:
        caught = await safety.check("my husband is hitting me", embedder=Broken())
        assert caught.urgent, "patterns must still fire when the meaning tier is broken"
        quiet = await safety.check("how do I file an RTI", embedder=Broken())
        assert not quiet.urgent
    finally:
        safety.reset_cache()


@pytest.mark.asyncio
async def test_lite_profile_still_gets_the_pattern_tier():
    """`lite` loads no embedder at all; the gate must degrade rather than disappear."""
    class NoEmbedder:
        plan = type("P", (), {"use_embedder": False})()

    safety.reset_cache()
    try:
        check = await safety.check("I was raped", embedder=NoEmbedder())
        assert check.urgent and check.kind == "sexual_violence"
    finally:
        safety.reset_cache()
