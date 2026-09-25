"""The state of one turn, and the small helpers every pipeline step shares."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from .. import config, events
from ..agents.schemas import Plan, Procedure
from ..context.memory import Conversation
from ..evidence import Evidence
from ..llm.errors import DeadlineExceeded, LLMError, ProviderAuthError
from ..llm.spend import SpendMeter

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class TurnBudget:
    """What this turn may spend. Every field is a soft ceiling that degrades the answer."""

    depth: str
    deadline: float
    max_rounds: int
    max_crawls: int
    nav_depth: int
    crawls: int = 0

    @property
    def time_left(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.time_left <= 0

    def can_crawl(self, n: int = 1) -> bool:
        return self.crawls + n <= self.max_crawls and not self.expired

    @classmethod
    def for_depth(cls, depth: str, started: float | None = None) -> TurnBudget:
        spec = config.DEPTHS.get(depth, config.DEPTHS["standard"])
        start = time.monotonic() if started is None else started
        return cls(depth=spec.name, deadline=start + spec.deadline_s,
                   max_rounds=spec.max_rounds, max_crawls=spec.max_crawls,
                   nav_depth=spec.nav_depth)


def _ignore(_event) -> None:
    return None


@dataclass
class TurnState:
    session_id: str
    message: str
    conversation: Conversation
    budget: TurnBudget
    meter: SpendMeter = field(default_factory=SpendMeter)
    emit: Callable[[events.Event], None] = _ignore
    on_pause: Callable | None = None
    plan: Plan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    procedure: Procedure | None = None
    notes: list[str] = field(default_factory=list)          # caveats shown before the answer
    degraded: list[str] = field(default_factory=list)       # reduced-mode notices already shown
    answer: str = ""
    query_variants: list[str] | None = None
    verification_note: str = ""
    # What the final answer was written from, committed with it exactly once at the end.
    final_evidence: list[Evidence] = field(default_factory=list)
    committed: bool = False
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    started: float = field(default_factory=time.monotonic)

    @property
    def question(self) -> str:
        """The planner's standalone restatement, or the message itself."""
        if self.plan and self.plan.normalized_query:
            return self.plan.normalized_query
        return self.message

    @property
    def elapsed_s(self) -> float:
        return round(time.monotonic() - self.started, 1)

    def degrade(self, note: str) -> None:
        """Tell the reader, once, that part of the pipeline ran in a reduced mode."""
        if note and note not in self.degraded:
            self.degraded.append(note)
            self.emit(events.notice(note, level="warn", degraded=True))

    def stage(self, stage_id: str, label: str, status: str = "running", detail: str = "") -> None:
        self.emit(events.stage(stage_id, label, status, detail=detail))

    def call_kwargs(self) -> dict:
        """The deadline, pause callback and session every model call takes."""
        return {"deadline": self.budget.deadline, "on_pause": self.on_pause,
                "session": self.session_id}


async def degradable(turn: TurnState, work: Awaitable[T], fallback: T, note: str = "") -> T:
    """Run a stage that the turn can survive without.

    Running out of time and a rejected API key propagate: the first ends research, the second
    ends the turn. Any other model failure returns ``fallback`` and, if a note is given, tells
    the reader what ran without its model.
    """
    try:
        return await work
    except (DeadlineExceeded, ProviderAuthError):
        raise
    except LLMError as exc:
        log.warning("stage degraded: %s", exc)
        turn.degrade(note)
        return fallback
