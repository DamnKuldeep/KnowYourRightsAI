"""Request bodies, validated at the edge.

Every field a browser sends is bounded. ``state`` in particular is checked against the known list
because it is written into the writer's prompt: an arbitrary string there is a prompt injection.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .. import config

SESSION_ID = Field(default="", max_length=64, pattern=r"^[A-Za-z0-9_-]*$")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = SESSION_ID
    depth: Literal["auto", "quick", "standard", "deep"] = "auto"
    state: str = Field(default="", max_length=64)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the message is empty")
        return value

    @field_validator("state")
    @classmethod
    def _known_state(cls, value: str) -> str:
        value = value.strip()
        if value and value not in config.INDIAN_STATES:
            raise ValueError("unknown state or union territory")
        return value


class SessionRequest(BaseModel):
    session_id: str = SESSION_ID


class FeedbackRequest(BaseModel):
    session_id: str = SESSION_ID
    rating: Literal["up", "down"]
    question: str = Field(default="", max_length=4000)
    answer: str = Field(default="", max_length=8000)
    comment: str = Field(default="", max_length=2000)
