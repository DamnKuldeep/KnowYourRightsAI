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


@dataclass
class Conversation:
    """Per-session memory. One instance per browser session."""

    session_id: str = ""
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    state: str = ""                 # the user's Indian state, if they told us
    pool: dict[str, Evidence] = field(default_factory=dict)
    _summarised_upto: int = 0

    # ── turns ────────────────────────────────────────────────────────────────────────
    def add_user(self, text: str) -> None:
        self.turns.append(Turn("user", text.strip()))

    def add_assistant(self, text: str, evidence: list[Evidence] | None = None) -> None:
        self.turns.append(Turn("assistant", text.strip()))
        for item in evidence or []:
            self.remember(item)

    @property
    def needs_summary(self) -> bool:
        return len(self.turns) > config.HISTORY_SUMMARY_TRIGGER

    @property
    def summarised_upto(self) -> int:
        """How many turns, from the start, the rolling summary already covers."""
        return self._summarised_upto

    def pending_for_summary(self) -> list[Turn]:
        """Turns old enough to fold into the summary but not yet folded."""
        keep = config.HISTORY_TURNS_VERBATIM * 2
        return self.turns[self._summarised_upto:max(self._summarised_upto, len(self.turns) - keep)]

    def set_summary(self, text: str, upto: int) -> None:
        """Replace the summary. It is a *rolling* summary: each update is produced from the
        previous summary plus the newly folded turns, so it stays one bounded paragraph.

        It used to be appended to instead — old text, a newline, then the new text — and never
        re-compressed, so it grew without limit and eventually consumed the entire history
        budget.
        """
        self.summary = text.strip()
        self._summarised_upto = max(self._summarised_upto, upto)

    def history_block(self, max_tokens: int = 900) -> str:
        """Compact context so the planner and writer can resolve "it" and "that fine".

        **Newest first.** The budget is spent on the most recent exchange before anything
        older, and on the user's state before any of it. This used to be built oldest-first
        and then trimmed by keeping the *start* — so once history passed ~900 tokens (the
        second or third follow-up, with real answers) the most recent exchange and the user's
        state were the first things cut. "What about the appeal?" lost the answer it referred to.

        **The in-flight question is excluded.** The orchestrator records the user's message
        before planning, so without this the current question appeared twice — once in its own
        history and once as the question — in every prompt.

        Citation markers are stripped from past answers. Ids are assigned per turn, so a ``[S6]``
        in an earlier answer means something different — or nothing — now, and the writer will
        copy it forward. Observed live.
        """
        turns = list(self.turns[self._summarised_upto:])
        if turns and turns[-1].role == "user":
            turns = turns[:-1]                      # the question being answered right now
        head = f"USER'S STATE: {self.state}" if self.state else ""
        budget = max(0, max_tokens - estimate_tokens(head))
        kept, budget = _recent_lines(turns[-config.HISTORY_TURNS_VERBATIM * 2:], budget)

        parts: list[str] = [head] if head else []
        if self.summary and budget > 40:
            parts.append("EARLIER IN THIS CONVERSATION:\n"
                         + fit_to_tokens(_strip_markers(self.summary), budget))
        if kept:
            parts.append("RECENT TURNS:\n" + "\n".join(kept))
        return "\n\n".join(parts)

    # ── evidence pool ────────────────────────────────────────────────────────────────
    def remember(self, item: Evidence) -> None:
        """Keep a vetted source for follow-ups. The pool is bounded: the oldest go first.

        Unbounded, it held every section and crawled page of a long conversation in memory
        for the whole six-hour session lifetime.
        """
        key = "|".join(item.dedupe_key())
        existing = self.pool.pop(key, None)
        self.pool[key] = item if existing is None or item.score > existing.score else existing
        while len(self.pool) > config.SESSION_POOL_MAX:
            self.pool.pop(next(iter(self.pool)))

    def recall(self, question: str, limit: int = 6) -> list[Evidence]:
        """Previously-vetted sources that still look relevant to a follow-up.

        Word overlap rather than embeddings: this runs on every turn and only needs to decide
        whether last turn's sections are worth re-showing, which does not justify a GPU call.
        """
        words = {w for w in _words(question) if len(w) > 3} - _STOPWORDS
        if not words:
            return []
        scored: list[tuple[float, Evidence]] = []
        for item in self.pool.values():
            haystack = _words(f"{item.label()} {item.text[:600]}")
            overlap = len(words & haystack)
            share = overlap / len(words)
            # Two shared words was the whole test, and "legal" plus "under" clears that for
            # almost any pair of legal texts — so rent-deposit pages were recalled for a
            # domestic-violence question. Require a real share of the question's own words.
            # (The grader now also sees everything recalled, so this is a pre-filter.)
            if overlap >= 2 and share >= config.RECALL_MIN_SHARE:
                scored.append((share, item))
        scored.sort(key=lambda pair: -pair[0])
        return [item for _, item in scored[:limit]]

    def reset(self) -> None:
        self.turns.clear()
        self.pool.clear()
        self.summary = ""
        self._summarised_upto = 0


# Words that appear in nearly every legal question and prove nothing about topic.
_STOPWORDS = frozenset({
    "what", "which", "when", "where", "does", "have", "with", "that", "this", "from", "your",
    "they", "them", "their", "there", "about", "under", "legal", "law", "laws", "india",
    "indian", "right", "rights", "can", "will", "would", "should", "must", "been", "into",
    "section", "act", "person", "provision", "case", "file", "make", "take", "also",
})


def _recent_lines(turns: list[Turn], budget: int) -> tuple[list[str], int]:
    """The newest turns that fit in ``budget``, oldest first, and the budget left over.

    Walks backwards keeping whole turns. Each past answer is capped on its own first: a
    follow-up needs what was answered, not all 3,000 characters of it, and one long answer must
    not evict every turn before it. Something of the latest exchange is always kept.
    """
    kept: list[str] = []
    for turn in reversed(turns):
        if not turn.content:
            continue
        body = _strip_markers(turn.content)
        if turn.role == "assistant":
            body = fit_to_tokens(body, config.HISTORY_ANSWER_CAP_TOKENS)
        line = f"{turn.role.upper()}: {body}"
        cost = estimate_tokens(line) + 1
        if cost > budget:
            if not kept:
                kept.append(fit_to_tokens(line, budget))
            break
        kept.append(line)
        budget -= cost
    kept.reverse()
    return kept, budget


def _words(text: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower())}


def _strip_markers(text: str) -> str:
    """Remove ``[S1]``-style citation markers, which are only valid within their own turn."""
    import re

    cleaned = re.sub(r"\[[A-Z]{1,2}\d{1,2}\]", "", text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
