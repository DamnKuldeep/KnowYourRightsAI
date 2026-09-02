"""Validated contracts between pipeline stages.

Pydantic rather than a provider-specific ``response_format``: it works on every model in the
catalogue, including ones without strict schema mode, and it gives each stage a safe default
so a single malformed reply degrades one step instead of failing the turn.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Intent = Literal["smalltalk", "capability", "legal_question", "out_of_scope"]
Depth = Literal["quick", "standard", "deep"]
AnswerKind = Literal["definition", "procedure", "rights", "punishment", "mixed", "none"]
ToolName = Literal["legal_db", "web", "official", "wikipedia", "navigate"]


class ResearchStep(BaseModel):
    tool: ToolName
    query: str = ""
    reason: str = ""
    sub_question: int = 0


class SubQuestion(BaseModel):
    id: int = 0
    text: str = ""


class Plan(BaseModel):
    kind: Intent = "legal_question"
    depth: Depth = "standard"
    answer_kind: AnswerKind = "mixed"
    normalized_query: str = ""
    language: str = "en"
    needs_state: bool = False
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    steps: list[ResearchStep] = Field(default_factory=list)

    @property
    def is_conversational(self) -> bool:
        return self.kind in ("smalltalk", "capability", "out_of_scope")


class SearchQueries(BaseModel):
    """Reformulations for one sub-question, aimed at the source each will be run against."""

    statute_queries: list[str] = Field(default_factory=list)
    web_queries: list[str] = Field(default_factory=list)
    wikipedia_query: str = ""


class SourceGrade(BaseModel):
    id: str = ""
    relevant: bool = False
    note: str = ""


class Grades(BaseModel):
    grades: list[SourceGrade] = Field(default_factory=list)


class Gap(BaseModel):
    sub_question: int = 0
    missing: str = ""
    tool: ToolName = "web"
    query: str = ""


class Coverage(BaseModel):
    """Whether another research round is worth its time."""

    answered: list[int] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    enough: bool = True
    note: str = ""


class ProcedureStep(BaseModel):
    n: int = 0
    text: str = ""


class Procedure(BaseModel):
    """A how-to extracted from official sources, rendered by the UI as a checklist.

    Accepts a bare list of steps as well as the full object. Models asked for
    ``{"steps": [...], "fees": ...}`` quite often return just the ``[...]`` — observed live,
    and losing the whole procedure card over a wrapper is a silly way to fail.
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_step_list(cls, value):
        if isinstance(value, list):
            return {"steps": value}
        return value

    title: str = ""
    steps: list[ProcedureStep] = Field(default_factory=list)
    portal_url: str = ""
    fees: str = ""
    documents: list[str] = Field(default_factory=list)
    timeline: str = ""
    appeal_to: str = ""
    source_urls: list[str] = Field(default_factory=list)

    @property
    def is_useful(self) -> bool:
        return bool(self.steps) or bool(self.fees) or bool(self.timeline)


class SafetyCheck(BaseModel):
    urgent: bool = False
    kind: str = ""
    reason: str = ""


class RiskyClaim(BaseModel):
    """A factual assertion the agent wants to confirm before standing behind it."""

    claim: str = ""
    risk: Literal["high", "medium"] = "medium"
    why: str = ""
    query: str = ""


class FactCheck(BaseModel):
    claims: list[RiskyClaim] = Field(default_factory=list)
    confident: bool = True

    @property
    def needs_checking(self) -> bool:
        return bool(self.claims) and not self.confident
