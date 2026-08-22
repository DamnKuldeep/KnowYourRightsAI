"""Conversation state across turns.

Two things make follow-ups work. First, **history compaction**: recent turns verbatim, older
ones collapsed into a summary that is regenerated only when it has to be. Second, an
**evidence pool** — the sections and pages already vetted this conversation, kept so that
"what about the appeal?" reuses what we found rather than paying to search for it again.

The pool is also what lets the agent answer a follow-up when the provider is rate-limited: it
already has the sources.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .. import config
from ..evidence import Evidence
from .budget import estimate_tokens, fit_to_tokens


@dataclass
class Turn:
    role: str                       # "user" | "assistant"
    content: str
    at: float = field(default_factory=time.time)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    """Per-session memory. One instance per browser session."""

    session_id: str = ""
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    state: str = ""                 # the user's Indian state, if they told us
    pool: dict[str, Evidence] = field(default_factory=dict)
    topic: str = ""
    _summarised_upto: int = 0

    # ── turns ────────────────────────────────────────────────────────────────────────
    def add_user(self, text: str) -> None:
        self.turns.append(Turn("user", text.strip()))

    def add_assistant(self, text: str, evidence: list[Evidence] | None = None) -> None:
        self.turns.append(Turn("assistant", text.strip(),
                               evidence_ids=[e.id for e in evidence or []]))
        for item in evidence or []:
            self.remember(item)

    @property
    def needs_summary(self) -> bool:
        return len(self.turns) > config.HISTORY_SUMMARY_TRIGGER

    def recent(self, n: int | None = None) -> list[Turn]:
        return self.turns[-(n or config.HISTORY_TURNS_VERBATIM * 2):]

    def pending_for_summary(self) -> list[Turn]:
        """Turns old enough to fold into the summary but not yet folded."""
        keep = config.HISTORY_TURNS_VERBATIM * 2
        return self.turns[self._summarised_upto:max(self._summarised_upto, len(self.turns) - keep)]

    def set_summary(self, text: str, upto: int) -> None:
        self.summary = text.strip()
        self._summarised_upto = upto

    def history_block(self, max_tokens: int = 900) -> str:
        """Compact recent context so the planner can resolve "it" and "that fine".

        Citation markers are stripped from past answers. Ids are assigned per turn, so a
        ``[S6]`` in yesterday's answer means something different — or nothing — today, and the
        writer will happily copy it forward. Observed live: an answer cited [S6] and [S7] from
        the previous turn and annotated its own uncertainty about them mid-sentence.
        """
        parts: list[str] = []
        if self.summary:
            parts.append(f"EARLIER IN THIS CONVERSATION:\n{_strip_markers(self.summary)}")
        lines = [f"{t.role.upper()}: {_strip_markers(t.content)}"
                 for t in self.recent() if t.content]
        if lines:
            parts.append("RECENT TURNS:\n" + "\n".join(lines))
        if self.state:
            parts.append(f"USER'S STATE: {self.state}")
        block = "\n\n".join(parts)
        return fit_to_tokens(block, max_tokens) if block else ""

    # ── evidence pool ────────────────────────────────────────────────────────────────
    def remember(self, item: Evidence) -> None:
        key = "|".join(item.dedupe_key())
        existing = self.pool.get(key)
        if existing is None or item.score > existing.score:
            self.pool[key] = item

    def recall(self, question: str, limit: int = 6) -> list[Evidence]:
        """Previously-vetted sources that still look relevant to a follow-up.

        Word overlap rather than embeddings: this runs on every turn and only needs to decide
        whether last turn's sections are worth re-showing, which does not justify a GPU call.
        """
        words = {w for w in _words(question) if len(w) > 3}
        if not words:
            return []
        scored: list[tuple[float, Evidence]] = []
        for item in self.pool.values():
            haystack = _words(f"{item.label()} {item.text[:600]}")
            overlap = len(words & haystack)
            if overlap >= 2:
                scored.append((overlap / len(words), item))
        scored.sort(key=lambda pair: -pair[0])
        return [item for _, item in scored[:limit]]

    def cited_so_far(self) -> list[str]:
        return sorted({e.citation for e in self.pool.values() if e.citation})

    def tokens(self) -> int:
        return estimate_tokens(self.history_block(10_000))

    def reset(self) -> None:
        self.turns.clear()
        self.pool.clear()
        self.summary = ""
        self.topic = ""
        self._summarised_upto = 0


def _words(text: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower())}


def _strip_markers(text: str) -> str:
    """Remove ``[S1]``-style citation markers, which are only valid within their own turn."""
    import re

    cleaned = re.sub(r"\[[A-Z]{1,2}\d{1,2}\]", "", text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
