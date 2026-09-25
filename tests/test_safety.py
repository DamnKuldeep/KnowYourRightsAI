"""The safety gate, scored against the same labelled set the calibration script uses.

A miss means someone describing violence gets a statute lecture instead of 112. A false alarm
means someone reading about the law gets a helpline card they did not need: harmless once, and
corrosive repeated, because it teaches people to ignore the card that matters.
"""

from __future__ import annotations

import inspect

import pytest

from knowyourrights import safety
from knowyourrights.orchestrator import core
from knowyourrights.safety_eval import DISCLOSURES, QUESTIONS


@pytest.mark.parametrize("case", [d for d in DISCLOSURES if d.literal],
                         ids=lambda c: c.text[:38])
def test_literal_disclosures_need_no_model(case):
    check = safety.check_patterns(case.text)
    assert check.urgent and check.kind == case.kind and check.reason


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q[:38])
def test_legal_questions_never_trigger_a_helpline(question):
    """Asking about a crime is not disclosing one."""
    assert not safety.check_patterns(question).urgent, f"false alarm on: {question!r}"


def test_every_kind_has_advice():
    kinds = {kind for _pattern, kind, _strong in safety._URGENT_PATTERNS}
    assert kinds <= set(safety.URGENT_ADVICE)
    assert all(safety.URGENT_ADVICE[k].strip() for k in kinds)


def test_exemplars_cover_every_kind_the_patterns_know():
    assert {kind for _p, kind, _s in safety._URGENT_PATTERNS} == set(safety.CRISIS_EXEMPLARS)


def test_the_gate_starts_before_the_planner():
    """A rate-limited planner must never delay a helpline number."""
    source = inspect.getsource(core.Orchestrator._pipeline)
    assert source.index("_start_safety_gate(turn)") < source.index("await self._plan(")


def test_the_pattern_tier_is_synchronous():
    assert not inspect.iscoroutinefunction(safety.check_patterns)


class _Broken:
    async def encode(self, texts):
        raise RuntimeError("no embeddings here")

    async def encode_one(self, text):
        raise RuntimeError("no embeddings here")


class _Unavailable:
    async def encode(self, texts):
        return None

    async def encode_one(self, text):
        return None


@pytest.mark.parametrize("embedder", [_Broken(), _Unavailable()], ids=["raises", "unavailable"])
async def test_the_meaning_tier_degrades_to_patterns(embedder):
    """A failing embedding API must not cost the pattern tier, and must never raise."""
    safety.reset_cache()
    try:
        assert (await safety.check("my husband is hitting me", embedder=embedder)).urgent
        assert not (await safety.check("I feel unsafe at home lately",
                                       embedder=embedder)).urgent
    finally:
        safety.reset_cache()
