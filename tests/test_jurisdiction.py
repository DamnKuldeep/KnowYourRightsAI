"""Where a law applies — the error this system must not make.

Telling someone in Mumbai that the Delhi Rent Act governs their deposit is worse than telling
them nothing. That happened: 44 Acts Parliament passed *for* a Union Territory were labelled
"Central law — applies across India", because the corpus notes they are "genuinely central" —
true of who enacted them, false of where they apply.
"""

from __future__ import annotations

import pytest

from knowyourrights import config, legal_terms
from knowyourrights.evidence import Evidence
from knowyourrights.retrieval.search import _extent


def _statute(title: str, **extent) -> Evidence:
    return Evidence(id="S1", kind="statute", title=title, citation=f"Section 1, {title}",
                    text="…", score=0.5, url="", source_type="central_act", **extent)


@pytest.mark.parametrize("title,place,ut", [
    ("Delhi Rent Act, 1995", "Delhi", True),
    ("Delhi Rent Control Act, 1958", "Delhi", True),
    ("National Capital Territory of Delhi (Recognition of Property Rights) Act, 2019", "Delhi", True),
    ("Chandigarh (Delegation of Powers) Act, 1987", "Chandigarh", True),
    ("Maharashtra Rent Control Act, 1999", "Maharashtra", False),
    ("Karnataka Stamp Act, 1957", "Karnataka", False),
])
def test_territorially_limited_acts_are_recognised(title, place, ut):
    extent = _extent(title)
    assert extent["state"] == place
    assert extent["union_territory"] is ut


@pytest.mark.parametrize("title", [
    "Indian Stamp Act, 1899",
    "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "Right to Information Act, 2005",
    "Consumer Protection Act, 2019",
])
def test_all_india_acts_stay_all_india(title):
    """The fix must not over-reach and start restricting national law."""
    assert _extent(title) == {"state": None, "union_territory": False}


def test_a_delhi_act_does_not_apply_in_maharashtra():
    """The regression itself: this surfaced first for a Mumbai deposit question."""
    ev = _statute("Delhi Rent Act, 1995", **_extent("Delhi Rent Act, 1995"))
    assert ev.jurisdiction == "TERRITORY"
    assert ev.applies_in("Maharashtra") is False
    assert ev.applies_in("Delhi") is True
    assert ev.applies_in(None) is None, "unknown location must stay unknown, not assumed"
    assert "across India" not in ev.jurisdiction_label
    assert "Delhi" in ev.jurisdiction_label


def test_territory_acts_are_not_called_state_law():
    """Parliament passed them — saying a state legislature did would be a different error."""
    ev = _statute("Delhi Rent Control Act, 1958", **_extent("Delhi Rent Control Act, 1958"))
    assert ev.jurisdiction != "STATE"
    assert "state law" not in ev.jurisdiction_label.lower()


def test_every_territory_is_selectable_in_the_ui():
    """applies_in() compares against what the user picks, so the names must line up."""
    places = {place for _prefix, place in config.TERRITORY_PREFIXES}
    assert places <= set(config.INDIAN_STATES), places - set(config.INDIAN_STATES)


def test_longest_prefix_wins():
    """'National Capital Territory of Delhi' must not be read as just 'Delhi …' by accident —
    the result is the same place here, but ordering bugs in this list would silently misfile
    other territories."""
    prefixes = [p for p, _ in config.TERRITORY_PREFIXES]
    for i, a in enumerate(prefixes):
        for b in prefixes[i + 1:]:
            assert not b.startswith(a) or a == b, f"{a!r} shadows the longer {b!r}"


def test_writer_prompt_knows_territory():
    """A label the writer is never told about is a label it will ignore."""
    from knowyourrights.agents import prompts
    text = prompts.WRITER if hasattr(prompts, "WRITER") else str(vars(prompts))
    assert "TERRITORY" in text
