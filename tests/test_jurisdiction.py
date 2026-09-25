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


# ── retrieval regressions from the corpus repair ──────────────────────────────────────
def test_acronyms_are_named_beside_themselves_for_the_reranker():
    """A bare "RTI" let the Credit Information Companies Act outrank RTI Section 19."""
    from knowyourrights import legal_terms
    assert legal_terms.annotate("my RTI was rejected") == \
        "my RTI (Right to Information Act, 2005) was rejected"
    assert "(FIR)" not in legal_terms.annotate("how do I file an FIR")
    assert legal_terms.annotate("what is Article 21") == "what is Article 21"


def test_rti_roles_imply_the_rti_act():
    """"penalty on PIO" was treated as a general criminal question and BNS was boosted."""
    from knowyourrights import legal_terms
    for q in ("penalty on PIO", "the CPIO did not reply", "appeal to the first appellate authority"):
        assert "Right to Information Act, 2005" in legal_terms.detect_acts(q), q


def test_general_code_boost_is_relative_and_only_for_near_ties(monkeypatch):
    """A flat +0.25 tuned on a ~0.99 score scale swamped Cohere's ~0.3 scale and lifted
    Article 193 (0.056) over the Motor Vehicles Act (0.334)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "knowyourrights" / "retrieval" / "search.py"
           ).read_text(encoding="utf-8")
    assert "config.GENERAL_CODE_BOOST * top_score" in src
    assert "score >= eligible" in src


def test_reranker_sees_the_sections_own_citizen_questions():
    from knowyourrights.retrieval.search import _citizen_questions
    class Row:
        chunk_text = "(1) Subject to the proviso ... within thirty days of the receipt"
        embed_text = ("Right to Information Act, 2005 — Section 7 (Disposal of request)\n"
                      "How long does the government have to respond?\n"
                      "What if they do not reply?\n"
                      "RTI deadline response time\n" + chunk_text)
    q = _citizen_questions(Row())
    assert "How long does the government have to respond?" in q
    assert "RTI deadline" not in q, "the keyword line is noise to a cross-encoder"
    assert "Subject to the proviso" not in q


def test_rerank_calibration_is_keyed_to_the_document_format(monkeypatch):
    from knowyourrights import config
    from knowyourrights.retrieval.reranker import Reranker
    from knowyourrights.runtime import resources
    r = Reranker(resources.select_profile(requested="api"))
    monkeypatch.setattr(config, "RERANK_WITH_QUESTIONS", True)
    with_q = r.model_name
    monkeypatch.setattr(config, "RERANK_WITH_QUESTIONS", False)
    assert with_q != r.model_name, "a threshold must not be shared across document formats"


# ── old-code section numbers ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("question,new_act,new_section", [
    ("What is the punishment under Section 420 IPC?", "Bharatiya Nyaya Sanhita, 2023", "318"),
    ("IPC 302 kya hai", "Bharatiya Nyaya Sanhita, 2023", "103"),
    ("section 498A of the Indian Penal Code", "Bharatiya Nyaya Sanhita, 2023", "85"),
    ("anticipatory bail under 438 CrPC", "Bharatiya Nagarik Suraksha Sanhita, 2023", "482"),
    ("Section 65B Evidence Act certificate", "Bharatiya Sakshya Adhiniyam, 2023", "63"),
])
def test_repealed_sections_map_to_their_new_numbers(question, new_act, new_section):
    """Regression: "Section 420 IPC" was looked up as BNS 420, and the writer then invented
    "Section 420 of the BNS". The number changes; it must be translated, not carried."""
    refs = legal_terms.detect_section_refs(question)
    assert [(r.act, r.label) for r in refs] == [(new_act, new_section)]


def test_an_unmapped_old_section_is_never_guessed():
    refs = legal_terms.detect_section_refs("what is section 999 IPC")
    assert refs == [], "no verified mapping means no lookup — not BNS 999"
    _, unmapped = legal_terms.map_repealed_sections("what is section 999 IPC")
    assert unmapped == [("IPC", "999")]


def test_every_mapping_matches_the_corpus_heading():
    """Each row was checked by hand against the new section's heading; this re-checks it
    against the live database so a wrong row fails the build instead of reaching a user."""
    import lancedb
    db = lancedb.connect(str(config.DB_PATH)).open_table(config.TABLE)
    df = db.search().where("source_type = 'criminal_code'").limit(5000).select(
        ["act_title", "section_label", "section_name"]).to_pandas()
    headings = {(r.act_title.split(" (")[0].strip(), str(r.section_label)): str(r.section_name).lower()
                for r in df.itertuples()}
    # a word from the subject that must appear in the corpus heading of the new section
    must = {"318": "cheat", "103": "murder", "64": "rape", "85": "cruelty", "80": "dowry",
            "482": "apprehending arrest", "173": "cognizable", "63": "electronic", "58": "twenty-four",
            "35": "without warrant", "109": "attempt to murder", "316": "breach of trust"}
    for m in legal_terms.SECTION_MAP.values():
        key = (m.new_act, m.new)
        assert key in headings, f"{m.code} {m.old} -> {m.new_act} {m.new}: not in corpus"
        if m.new in must:
            assert must[m.new] in headings[key], f"{m.new}: heading is {headings[key]!r}"


def test_old_section_numbers_never_become_new_code_citations():
    from knowyourrights.agents import prompts
    assert "Section 420 IPC is Section 318 BNS" in " ".join(prompts.WRITER.split())
