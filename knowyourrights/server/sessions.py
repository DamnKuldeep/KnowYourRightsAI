"""Conversations held in memory, bounded in number and in idle time."""

from __future__ import annotations

import time
import uuid

from .. import config
from ..context.memory import Conversation


class SessionStore:
    def __init__(self, max_sessions: int | None = None, ttl_s: float | None = None) -> None:
        self.max_sessions = max_sessions or config.SESSION_MAX
        self.ttl_s = ttl_s or config.SESSION_TTL_S
        self._sessions: dict[str, Conversation] = {}
        self._last_seen: dict[str, float] = {}

    def get(self, session_id: str) -> Conversation:
        """The conversation for ``session_id``, created if new. Evicts stale ones first."""
        self._evict_idle()
        session_id = session_id.strip() or uuid.uuid4().hex[:16]
        conversation = self._sessions.get(session_id)
        if conversation is None:
            while len(self._sessions) >= self.max_sessions:
                self.drop(min(self._last_seen, key=self._last_seen.get))
            conversation = Conversation(session_id=session_id)
            self._sessions[session_id] = conversation
        self._last_seen[session_id] = time.time()
        return conversation

    def drop(self, session_id: str) -> bool:
        """Forget a conversation entirely, returning its memory. True if it existed."""
        conversation = self._sessions.pop(session_id, None)
        self._last_seen.pop(session_id, None)
        if conversation is not None:
            conversation.reset()
        return conversation is not None

    def _evict_idle(self) -> None:
        now = time.time()
        for sid in [s for s, seen in self._last_seen.items() if now - seen > self.ttl_s]:
            self.drop(sid)

    def stats(self) -> dict:
        return {"active": len(self._sessions), "max": self.max_sessions,
                "ttl_hours": round(self.ttl_s / 3600, 1),
                "pooled_sources": sum(len(c.pool) for c in self._sessions.values())}
