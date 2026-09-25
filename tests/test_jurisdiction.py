"""Where a law applies, and which law replaced which — the errors this system must not make.

Telling someone in Mumbai that the Delhi Rent Act governs their deposit is worse than telling
them nothing. So is citing "Section 420 of the BNS", which does not exist.
"""

from __future__ import annotations

import pytest

from knowyourrights import config, legal_terms
from knowyourrights.agents import prompts
from knowyourrights.evidence import Evidence
from knowyourrights.retrieval.search import extent

from .conftest import requires_corpus


def _statute(title: str) -> Evidence:
    return Evidence(id="S1", kind="statute", title=title, citation=f"Section 1, {title}",
                    text="…", score=0.5, source_type="central_act", **extent(title))


def test_a_delhi_act_does_not_apply_in_maharashtra():
    ev = _statute("Delhi Rent Act, 1995")
    assert ev.jurisdiction == "TERRITORY"
    assert ev.applies_in("Maharashtra") is False
    assert ev.applies_in("Delhi") is True
    assert ev.applies_in(None) is None, "an unknown location stays unknown, never assumed"
    assert "across India" not in ev.jurisdiction_label and "Delhi" in ev.jurisdiction_label


def test_territory_acts_are_not_called_state_law():
    """Parliament passed them; saying a state legislature did would be a different error."""
    ev = _statute("Delhi Rent Control Act, 1958")
    assert ev.jurisdiction != "STATE"
    assert "state law" not in ev.jurisdiction_label.lower()


def test_every_territory_is_selectable_in_the_ui():
    places = {place for _prefix, place in config.TERRITORY_PREFIXES}
    assert places <= set(config.INDIAN_STATES), places - set(config.INDIAN_STATES)


def test_longest_prefix_wins():
    prefixes = [p for p, _ in config.TERRITORY_PREFIXES]
    for i, shorter in enumerate(prefixes):
        for longer in prefixes[i + 1:]:
            assert not longer.startswith(shorter), f"{shorter!r} shadows {longer!r}"


def test_the_writer_is_told_about_territory_and_where_the_matter_is():
    assert "TERRITORY" in prompts.WRITER
    assert "WHERE THE MATTER IS" in prompts.WRITER, "tenancy follows the property"


def test_a_named_place_answers_which_state():
    place = lambda text: legal_terms.place_named(text, config.INDIAN_STATES)   # noqa: E731
    assert place("What does the Delhi Rent Control Act say about eviction?") == "Delhi"
    assert place("My landlord in Mumbai won't return my deposit") == "Maharashtra"
    assert place("Jammu and Kashmir land law") == "Jammu & Kashmir"
    assert place("How do I file an RTI?") is None
    assert place("goals for a legal notice") is None


# ── vocabulary ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("what are my rights if the police arrest me without a warrant?", "en"),
    ("what is the punishment for cheating", "en"),
    ("police ne mujhe bina warrant ke arrest kar liya, kya yeh legal hai?", "hinglish"),
    ("mera landlord deposit nahi de raha", "hinglish"),
    ("पुलिस मुझे गिरफ्तार कर सकती है क्या", "hi"),
])
def test_language_detection(text, expected):
    assert legal_terms.detect_language(text) == expected


def test_acronyms_expand_once_not_repeatedly():
    out = legal_terms.expand("what is the punishment under IPC 302")
    assert out.count("Bharatiya") == 1 and "Bharatiya Nyaya Sanhita, 2023" in out


def test_acronyms_are_named_beside_themselves_for_the_reranker():
    """A bare "RTI" let the Credit Information Companies Act outrank RTI Section 19."""
    assert legal_terms.annotate("my RTI was rejected") == \
        "my RTI (Right to Information Act, 2005) was rejected"
    assert legal_terms.annotate("what is Article 21") == "what is Article 21"


def test_rti_roles_imply_the_rti_act():
    for question in ("penalty on PIO", "the CPIO did not reply",
                     "appeal to the first appellate authority"):
        assert "Right to Information Act, 2005" in legal_terms.detect_acts(question), question


def test_section_reference_does_not_swallow_the_next_word():
    refs = legal_terms.detect_section_refs("read Section 6 of the RTI Act")
    assert (refs[0].label, refs[0].act) == ("6", "Right to Information Act, 2005")


def test_article_reference_with_subclauses():
    refs = legal_terms.detect_section_refs("what does article 19(1)(a) protect")
    assert (refs[0].kind, refs[0].label) == ("article", "19")


def test_known_gaps_are_reported():
    gaps = legal_terms.detect_gaps("does PMLA cover this")
    assert gaps and "Money Laundering" in gaps[0]


# ── old-code section numbers ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("question,new_act,new_section", [
    ("What is the punishment under Section 420 IPC?", "Bharatiya Nyaya Sanhita, 2023", "318"),
    ("IPC 302 kya hai", "Bharatiya Nyaya Sanhita, 2023", "103"),
    ("section 498A of the Indian Penal Code", "Bharatiya Nyaya Sanhita, 2023", "85"),
    ("anticipatory bail under 438 CrPC", "Bharatiya Nagarik Suraksha Sanhita, 2023", "482"),
    ("Section 65B Evidence Act certificate", "Bharatiya Sakshya Adhiniyam, 2023", "63"),
])
def test_repealed_sections_map_to_their_new_numbers(question, new_act, new_section):
    """Regression: "Section 420 IPC" was looked up as BNS 420, which does not exist."""
    refs = legal_terms.detect_section_refs(question)
    assert [(r.act, r.label) for r in refs] == [(new_act, new_section)]


def test_an_unmapped_old_section_is_never_guessed():
    assert legal_terms.detect_section_refs("what is section 999 IPC") == []
    _, unmapped = legal_terms.map_repealed_sections("what is section 999 IPC")
    assert unmapped == [("IPC", "999")]


def test_old_section_numbers_never_become_new_code_citations():
    assert "Section 420 IPC is Section 318 BNS" in " ".join(prompts.WRITER.split())


@requires_corpus
def test_every_mapping_matches_the_corpus_heading():
    """Each mapping was checked by hand; this re-checks it against the live database."""
    import lancedb

    table = lancedb.connect(str(config.DB_PATH)).open_table(config.TABLE)
    df = (table.search().where("source_type = 'criminal_code'").limit(5000)
          .select(["act_title", "section_label", "section_name"]).to_pandas())
    headings = {(r.act_title.split(" (")[0].strip(), str(r.section_label)):
                str(r.section_name).lower() for r in df.itertuples()}
    must = {"318": "cheat", "103": "murder", "64": "rape", "85": "cruelty", "80": "dowry",
            "482": "apprehending arrest", "173": "cognizable", "63": "electronic",
            "58": "twenty-four", "35": "without warrant", "109": "attempt to murder",
            "316": "breach of trust"}
    for m in legal_terms.SECTION_MAP.values():
        key = (m.new_act, m.new)
        assert key in headings, f"{m.code} {m.old} -> {m.new_act} {m.new}: not in the corpus"
        if m.new in must:
            assert must[m.new] in headings[key], f"{m.new}: heading is {headings[key]!r}"
