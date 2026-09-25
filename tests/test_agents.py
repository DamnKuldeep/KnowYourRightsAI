"""The model stages' deterministic parts: planning rules, grading rescue, answer cleanup, and
the procedure card. Every case here is a regression from a real answer."""

from __future__ import annotations

import pytest

from knowyourrights import config
from knowyourrights.agents import answer_text, grading, planning, prompts, stages
from knowyourrights.agents.schemas import Plan, Procedure, ResearchStep
from knowyourrights.evidence import Evidence, assign_ids


def statute(score=0.9, unit="u1") -> Evidence:
    return Evidence(kind="statute", title="Section 1, Some Act, 2000", text="section text",
                    tier=config.TIER_STATUTE, score=score, unit_id=unit,
                    citation="Section 1, Some Act, 2000")


# ── planning ──────────────────────────────────────────────────────────────────────────
def test_a_url_in_a_search_query_becomes_navigation():
    """Searching the web for a URL string wasted 30 s a step and found nothing."""
    steps = planning.clean_steps([ResearchStep(tool="official",
                                               query="https://rtionline.gov.in/guidelines.php")])
    assert steps[0].tool == "navigate"


def test_duplicate_steps_are_collapsed():
    steps = planning.clean_steps([ResearchStep(tool="legal_db", query="arrest rights"),
                                  ResearchStep(tool="legal_db", query="Arrest Rights"),
                                  ResearchStep(tool="web", query="arrest rights")])
    assert len(steps) == 2


def test_a_legal_question_always_consults_the_statute():
    """An RTI walkthrough once cited nine government pages and no section of the RTI Act."""
    plan = planning.apply_rules(Plan(kind="legal_question", answer_kind="rights",
                                     steps=[ResearchStep(tool="web", query="x")]), "x")
    assert plan.steps[0].tool == "legal_db"


def test_a_procedure_about_an_act_always_searches_its_deadline_and_appeal():
    planned = Plan(kind="legal_question", answer_kind="procedure",
                   normalized_query="how to file an RTI application",
                   steps=[ResearchStep(tool="web", query="rti online portal")])
    plan = planning.apply_rules(planned, "How do I file an RTI?")
    statute_queries = [s.query for s in plan.steps if s.tool == "legal_db"]
    assert any("appeal" in q and "Right to Information" in q for q in statute_queries)

    rights = planning.apply_rules(Plan(kind="legal_question", answer_kind="rights"),
                                  "Can police arrest me at night?")
    assert not any("appeal" in s.query for s in rights.steps)


def test_a_question_naming_its_place_is_not_asked_which_state():
    plan = planning.apply_rules(Plan(kind="legal_question", needs_state=True),
                                "What does the Delhi Rent Control Act say about eviction?")
    assert plan.needs_state is False


def test_language_is_decided_in_code_not_by_the_planner():
    plan = planning.apply_rules(Plan(kind="legal_question", language="hi"),
                                "what is the punishment for cheating")
    assert plan.language == "en"


async def test_make_plan_applies_the_rules_to_the_models_plan(monkeypatch):
    class Client:
        async def chat_json(self, *args, **kwargs):
            return Plan(kind="legal_question", answer_kind="procedure")

    monkeypatch.setattr(planning, "get_client", lambda: Client())
    plan = await planning.make_plan("How do I file an RTI?")
    assert any(s.tool == "legal_db" for s in plan.steps)


# ── grading ───────────────────────────────────────────────────────────────────────────
def test_grader_rescue_keeps_confident_statutes():
    """The grader once rejected all six correct BNSS sections and the answer was empty."""
    items = assign_ids([statute(0.95, "a"), statute(0.9, "b"), statute(0.2, "c")])
    rescued = grading.rescue(items)
    assert len(rescued) == 2 and all(e.score >= grading.RESCUE_SCORE for e in rescued)


def test_grader_rescue_declines_when_nothing_is_confident():
    assert grading.rescue([statute(0.1)]) == []


def test_ungraded_sources_are_kept_but_marked_unvetted():
    items = assign_ids([statute(0.9, "a"), statute(0.8, "b")])
    verdicts = {"S1": type("G", (), {"relevant": False, "note": "off topic"})()}
    kept = grading.apply_grades(items, verdicts)
    assert [e.id for e in kept] == ["S2"] and kept[0].relevant is None


# ── answer cleanup ────────────────────────────────────────────────────────────────────
def test_fabricated_citation_markers_are_removed():
    items = assign_ids([statute(unit="a"), statute(unit="b")])
    cleaned, unsupported, verified = answer_text.finalise(
        "The law says X [S1] and also Y [S9], plus Z [S2].", items)
    assert unsupported == ["S9"] and "[S9]" not in cleaned and verified == 2


def test_invented_citation_shapes_are_normalised():
    assert answer_text.normalize_markers("see [S1(a)] and [S1, S2]") == "see [S1] and [S1][S2]"
    assert answer_text.normalize_markers("[S1 means something else]") == \
        "[S1 means something else]"


def test_prompt_block_names_are_not_left_as_citations():
    out = answer_text.normalize_markers("File a summary suit [EXTRACTED PROCEDURE][W2]. [G1]")
    assert "EXTRACTED" not in out and "[W2]" in out and "[G1]" in out


def test_used_evidence_reports_only_what_was_cited():
    items = assign_ids([statute(unit="a"), statute(unit="b"), statute(unit="c")])
    assert [e.id for e in answer_text.used_evidence("Only this [S2].", items)] == ["S2"]


def test_page_titles_are_not_used_as_link_text():
    out = answer_text.tidy_link_labels(
        "Visit [RTI Online:: Home | Submit RTI Request | Submit RTI First A]"
        "(https://rtionline.gov.in/).")
    assert out == "Visit [RTI Online](https://rtionline.gov.in/)."
    kept = "[RTI Online portal](https://rtionline.gov.in)"
    assert answer_text.tidy_link_labels(kept) == kept


def test_unstated_lines_are_dropped_but_real_lines_kept():
    kept = answer_text.drop_unstated_lines(
        "- **Fee:** ₹10\n"
        "- **Response time:** The sources do not state the response time.\n"
        "- Your sources do not state the fee.\n"
        "- The PIO must reply within 30 days [S2].")
    assert "₹10" in kept and "30 days" in kept and "do not state" not in kept


def test_a_hindi_unstated_line_is_dropped():
    kept = answer_text.drop_unstated_lines(
        "- **शुल्क:** ₹10\n- **प्रतिक्रिया समय:** स्रोतों में अधिकारी द्वारा प्रतिक्रिया देने की "
        "कोई विशिष्ट समय सीमा नहीं बताई गई है।")
    assert "₹10" in kept and "स्रोतों" not in kept


def test_a_heading_left_empty_by_the_filter_is_removed():
    text = ("Send a legal notice first [W1].\n\n**What it costs and how long**\n\n"
            "- **Fee:** The sources do not state the court fee.\n"
            "- **Response time:** Not stated in the provided sources.")
    assert "What it costs" not in answer_text.drop_unstated_lines(text)
    assert "**Steps**" in answer_text.drop_unstated_lines("**Steps**\n\n- File online [G1].")


def test_the_state_question_is_asked_exactly_once():
    plan = Plan(needs_state=True, language="hi")
    assert "राज्य" in answer_text.state_question("जमा राशि…", plan, None)
    assert answer_text.state_question("…", plan, "Kerala") == ""
    plan.language = "en"
    for already_asked in ("Which state is your flat in?", "**Which state is your flat in?**",
                          "Which state are you in? [S1]\n"):
        assert answer_text.state_question(already_asked, plan, None) == ""
    assert "Which state" in answer_text.state_question("Send a legal notice.", plan, None)
    plan.needs_state = False
    assert answer_text.state_question("Send a legal notice.", plan, None) == ""


# ── the writer's context ──────────────────────────────────────────────────────────────
def test_all_india_is_answered_for_all_india():
    """An RTI question with no state selected came back with one state's portal and fees."""
    plan = Plan(kind="legal_question", answer_kind="procedure", needs_state=False)
    assert "All India" in prompts.writer_context(plan, None, [], "2026-09-25")
    plan.needs_state = True
    assert "asking which state" in prompts.writer_context(plan, None, [], "2026-09-25")
    assert "All India" not in prompts.writer_context(plan, "Kerala", [], "2026-09-25")


def test_the_writer_is_not_the_unreliable_free_model():
    """The free 120B broke 12 of 22 streams in one session, 5 after text had started."""
    assert not config.is_free_model(config.WRITER_MODELS[0].id)


# ── the procedure card ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,money,ok", [
    ("As prescribed in the RTI Rules, 2012.", True, False),     # a year is not an amount
    ("Payment can be made through internet banking", True, False),
    ("Rs 10; free for BPL applicants", True, True),
    ("₹10", True, True),
    ("Free of charge", True, True),
    ("Application fee (amount not specified); free for BPL is not mentioned.", True, False),
    ("30 days", False, True),
    ("as soon as possible", False, False),
])
def test_card_facts_must_state_a_value(text, money, ok):
    assert stages.states_a_value(text, money=money) is ok


def test_the_card_keeps_only_links_that_were_read_and_facts_with_values():
    card = stages.keep_stated_facts(
        Procedure(portal_url="https://invented.example", fees="as prescribed",
                  appeal_to="Not specified in the provided sources.", timeline="30 days",
                  documents=["Rental agreement", "not mentioned"]),
        {"https://rtionline.gov.in"})
    assert card.portal_url == "" and card.fees == "" and card.appeal_to == ""
    assert card.timeline == "30 days" and card.documents == ["Rental agreement"]
    assert card.source_urls == ["https://rtionline.gov.in"]
