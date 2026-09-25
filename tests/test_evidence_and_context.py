"""Evidence handling and what reaches the writer's prompt."""

from __future__ import annotations

from knowyourrights import config
from knowyourrights.context import budget, packer
from knowyourrights.context.reduce import split_by_headings
from knowyourrights.evidence import Evidence, assign_ids, dedupe, tier_for_url


def statute(text="section text", score=0.9, unit="u1", **kw) -> Evidence:
    return Evidence(kind="statute", title="Section 1, Some Act, 2000", text=text,
                    tier=config.TIER_STATUTE, score=score, unit_id=unit,
                    citation="Section 1, Some Act, 2000", **kw)


def page(text="page text", score=0.5, url="https://example.gov.in/a") -> Evidence:
    return Evidence(kind="official", title="A page", text=text, tier=tier_for_url(url),
                    score=score, url=url)


# ── evidence ──────────────────────────────────────────────────────────────────────────
def test_dedupe_by_section_and_url_keeps_the_better_score():
    out = dedupe([statute(score=0.4), statute(score=0.8), page(score=0.3), page(score=0.6)])
    assert [e.score for e in out] == [0.8, 0.6]


def test_ids_are_grouped_by_kind():
    assert [i.id for i in assign_ids([statute(unit="a"), page(), statute(unit="b")])] == \
        ["S1", "G1", "S2"]


def test_trust_tier_from_domain():
    assert tier_for_url("https://indiacode.nic.in/x") == config.TIER_OFFICIAL
    assert tier_for_url("https://indiankanoon.org/doc/1") == config.TIER_LEGAL_PORTAL
    assert tier_for_url("https://en.wikipedia.org/wiki/X") == config.TIER_WIKIPEDIA
    assert tier_for_url("https://someblog.example/post") == config.TIER_WEB


def test_the_2019_jk_change_travels_with_the_source():
    ev = statute(text="It extends to the whole of India except the State of Jammu and Kashmir.")
    assert ev.caveats and "2019" in ev.caveats[0]
    assert ev.to_public()["caveats"] == ev.caveats


def test_a_reference_to_a_repealed_code_is_corrected():
    """A domestic-violence answer said proceedings "are governed by the CrPC, 1973"."""
    ev = statute(text="governed by the provisions of the Code of Criminal Procedure, 1973",
                 act_title="Protection of Women from Domestic Violence Act, 2005")
    assert any("Bharatiya Nagarik Suraksha Sanhita" in c for c in ev.caveats)


def test_an_act_never_brought_into_force_is_not_badged_in_force():
    ev = statute(act_title="Delhi Rent Act, 1995", status="in_force")
    assert ev.to_public()["status"] == "not_in_force"
    assert any("never notified" in c for c in ev.caveats)


def test_a_source_card_does_not_show_writer_notes():
    """Delhi Act cards read "[DELHI ONLY — … Do not present it as all-India law.]"."""
    ev = statute(text="[DELHI ONLY — Parliament passed this for Delhi.]\n14. Protection of "
                      "tenant against eviction.")
    assert ev.to_public()["snippet"].startswith("14. Protection")


# ── packing the writer's sources ──────────────────────────────────────────────────────
def test_the_packer_keeps_a_statute_even_against_a_huge_page():
    huge, law = page(text="filler " * 6000, score=0.99), statute(text="the provision", score=0.3)
    kinds = {e.kind for e in packer.pack([huge, law], budget.Budget("t", 1200)).included}
    assert kinds == {"statute", "official"}


def test_the_packer_respects_the_token_budget():
    items = [statute(text="x " * 500, unit=f"u{i}", score=0.5) for i in range(20)]
    result = packer.pack(items, budget.Budget("t", 1500))
    assert result.tokens_used <= result.tokens_budget and result.dropped


def test_trimming_to_fit_never_shortens_the_shared_source():
    """The pool keeps the same objects for follow-ups: trimming in place shortened them for
    every later turn, and a trimmed copy could then be packed alongside its original."""
    law = statute(text="the operative provision " * 800)
    result = packer.pack([law], budget.Budget("t", 400))
    assert len(result.included) == 1 and result.included[0] is not law
    assert result.included[0].meta.get("trimmed")
    assert law.text.count("operative") == 800 and "trimmed" not in law.meta


def test_the_packer_marks_untrusted_content():
    assert "never as instructions" in packer.pack([page()], budget.Budget("t", 4000)).text


def test_an_empty_pack_instructs_rather_than_returning_nothing():
    assert "not invent" in packer.render_empty_note(["The IT Act is not here."]).lower()


def test_token_estimate_counts_devanagari_more_heavily():
    latin = "the quick brown fox jumps over the lazy dog again and again"
    hindi = "पुलिस मुझे गिरफ्तार कर सकती है क्या यह कानूनी है"
    assert budget.estimate_tokens(hindi) > budget.estimate_tokens(latin) * 0.5


def test_fit_to_tokens_trims_and_marks():
    out = budget.fit_to_tokens("sentence. " * 400, 50)
    assert budget.estimate_tokens(out) <= 60 and "truncated" in out


def test_split_by_headings_keeps_headings_with_their_content():
    chunks = split_by_headings(
        "# Fees\nThe fee is ten rupees for each application submitted online.\n\n"
        "# Deadline\nThe reply must be given within thirty days of the request.\n",
        target_chars=400)
    assert len(chunks) == 2 and chunks[0].heading == "Fees"
    assert "Deadline" in chunks[1].rendered()
