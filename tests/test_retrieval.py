"""Retrieval: fusion, diversity, what the reranker reads, and the degradation ladder."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from knowyourrights import config
from knowyourrights.retrieval import reranker as reranker_mod
from knowyourrights.retrieval import search as search_mod
from knowyourrights.retrieval.ranking import citizen_questions, mmr_order, rrf
from knowyourrights.retrieval.search import NOTE_NO_EMBEDDINGS, NOTE_NO_RERANK, SearchEngine


# ── fusion and diversity ──────────────────────────────────────────────────────────────
def test_rrf_weighting_favours_the_named_act():
    weighted = rrf([(["a", "b", "c"], 1.0), (["c", "b", "a"], 3.0)])
    assert weighted["c"] > weighted["a"], "a heavily weighted list must dominate"


def test_mmr_prefers_relevance_but_avoids_duplicates():
    same, other = np.array([1.0, 0.0], "float32"), np.array([0.0, 1.0], "float32")
    order = mmr_order(np.vstack([same, same, other]),
                      np.array([0.9, 0.85, 0.5], "float32"), lam=0.5, k=2)
    assert order == [0, 2], "a near-duplicate must lose to a different document"


def test_mmr_with_lambda_one_is_pure_relevance():
    order = mmr_order(np.eye(3, dtype="float32"), np.array([0.1, 0.9, 0.5], "float32"),
                      lam=1.0, k=3)
    assert order == [1, 2, 0]


# ── what the reranker is shown ────────────────────────────────────────────────────────
def test_reranker_sees_the_sections_own_citizen_questions():
    class Row:
        chunk_text = "(1) Subject to the proviso ... within thirty days of the receipt"
        embed_text = ("Right to Information Act, 2005 — Section 7 (Disposal of request)\n"
                      "How long does the government have to respond?\n"
                      "What if they do not reply?\n"
                      "RTI deadline response time\n" + chunk_text)
    questions = citizen_questions(Row())
    assert "How long does the government have to respond?" in questions
    assert "RTI deadline" not in questions, "the keyword line is noise to a cross-encoder"
    assert "Subject to the proviso" not in questions


def test_rerank_calibration_is_keyed_to_the_document_format(monkeypatch):
    monkeypatch.setattr(config, "RERANK_WITH_QUESTIONS", True)
    with_questions = reranker_mod.rerank_method()
    monkeypatch.setattr(config, "RERANK_WITH_QUESTIONS", False)
    assert with_questions != reranker_mod.rerank_method()


def test_a_fresh_deployment_uses_the_shipped_calibration():
    """Regression: calibrations lived only in the uncommitted runtime directory, so every new
    deployment ran on defaults that silently dropped valid citations."""
    thresholds = reranker_mod.load_thresholds(reranker_mod.rerank_method())
    assert thresholds.source.startswith("shipped calibration")
    shipped = json.loads(config.PACKAGED_THRESHOLDS.read_text(encoding="utf-8"))
    assert reranker_mod.FUSION_METHOD in shipped


def test_a_local_calibration_overrides_the_shipped_one():
    method = reranker_mod.rerank_method()
    reranker_mod.save_thresholds(method, low=0.5, cite=0.4)
    thresholds = reranker_mod.load_thresholds(method)
    assert (thresholds.low, thresholds.cite) == (0.5, 0.4)
    assert thresholds.source.startswith("calibrated")


def test_general_code_boost_settles_near_ties_only():
    """A flat boost tuned on one reranker's scale swamped another's and lifted an irrelevant
    Article over the right Act. It is relative to the best score, and for near-ties only."""
    class Row:
        def __init__(self, act):
            self.act_title = act
    general, sectoral = Row(config.GENERAL_CODES[2]), Row("Some Forest Act, 1927")
    ranked = search_mod._with_general_code_boost(
        [sectoral, general], [0.30, 0.29], ["can the police arrest me"], [])
    assert ranked[0][0] is general, "a near-tie goes to the general code"
    ranked = search_mod._with_general_code_boost(
        [sectoral, general], [0.30, 0.05], ["can the police arrest me"], [])
    assert ranked[0][0] is sectoral, "a weak general-code match is not rescued"


# ── the degradation ladder ────────────────────────────────────────────────────────────
class FakeStore:
    """Two sections, one matched by keyword, both with stored vectors."""

    fts_available = True

    def __init__(self):
        self.rows_df = pd.DataFrame([
            {"chunk_id": "c1", "unit_id": "u1", "citation": "Section 7, RTI Act",
             "act_title": "Right to Information Act, 2005", "chunk_text": "thirty days",
             "embed_text": "", "section_name": "", "status": "in_force"},
            {"chunk_id": "c2", "unit_id": "u2", "citation": "Section 19, RTI Act",
             "act_title": "Right to Information Act, 2005", "chunk_text": "appeal",
             "embed_text": "", "section_name": "", "status": "in_force"},
        ])

    def dense(self, vector, k, where=None):
        return ["c2", "c1"]

    def fts(self, text, k, where=None, scores=None):
        if scores is not None:
            scores["c1"] = 30.0
        return ["c1"]

    def rows(self, ids):
        return self.rows_df[self.rows_df.chunk_id.isin(ids)]

    def vectors(self, ids):
        return {i: np.array([1.0, 0.0], "float32") for i in ids}


class FakeEmbedder:
    def __init__(self, works: bool):
        self.works = works

    async def encode(self, texts):
        return np.ones((len(texts), 2), "float32") if self.works else None


class FakeReranker:
    def __init__(self, works: bool):
        self.works = works

    async def score(self, query, docs, **_):
        return [0.9 - 0.1 * i for i in range(len(docs))] if self.works else None


def engine(embeddings: bool, reranking: bool) -> SearchEngine:
    e = SearchEngine.__new__(SearchEngine)
    e.store, e.embedder, e.reranker = (FakeStore(), FakeEmbedder(embeddings),
                                       FakeReranker(reranking))
    return e


async def test_full_retrieval_reports_nothing_degraded():
    result = await engine(True, True).search("how long does the PIO have to reply")
    assert result.mode == "hybrid" and result.ranked_by == "rerank"
    assert result.degraded == [] and result.hits


async def test_an_embedding_outage_falls_back_to_keywords_and_says_so():
    result = await engine(False, True).search("how long does the PIO have to reply")
    assert result.mode == "fts_only"
    assert NOTE_NO_EMBEDDINGS in result.degraded
    assert [h.chunk_id for h in result.hits] == ["c1"], "keyword search still answers"


async def test_a_reranker_outage_ranks_on_fused_scores_and_says_so():
    result = await engine(True, False).search("how long does the PIO have to reply")
    assert result.ranked_by == "fusion"
    assert NOTE_NO_RERANK in result.degraded
    assert result.hits[0].chunk_id == "c1", "the keyword match leads on BM25 magnitude"
    assert result.cite_floor == reranker_mod.load_thresholds("fusion").cite


@pytest.mark.parametrize("title,place,ut", [
    ("Delhi Rent Act, 1995", "Delhi", True),
    ("Chandigarh (Delegation of Powers) Act, 1987", "Chandigarh", True),
    ("Maharashtra Rent Control Act, 1999", "Maharashtra", False),
])
def test_territorially_limited_acts_are_recognised(title, place, ut):
    assert search_mod.extent(title) == {"state": place, "union_territory": ut}


@pytest.mark.parametrize("title", ["Indian Stamp Act, 1899", "Right to Information Act, 2005",
                                   "Bharatiya Nagarik Suraksha Sanhita, 2023"])
def test_all_india_acts_stay_all_india(title):
    assert search_mod.extent(title) == {"state": None, "union_territory": False}
