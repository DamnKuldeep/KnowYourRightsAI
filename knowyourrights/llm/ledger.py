"""Usage accounting: calls, errors, billed dollars and the free tier's daily allowance.

Every call is appended to ``.runtime/usage.jsonl`` so a session can be audited afterwards, and
every billed call also charges the turn that made it (see :mod:`.spend`), which is how each
question's cost and each client's budget are known exactly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from . import spend

log = logging.getLogger(__name__)

# The audit log is rotated once it passes this size, keeping one previous file.
USAGE_LOG_MAX_BYTES = 10 * 1024 * 1024


def usage_file() -> Path:
    """Resolved on every use, never at import.

    A path fixed at import is baked in before the test suite can redirect ``RUNTIME_DIR``, so
    running the tests wrote to the real runtime directory. Every runtime path here is lazy.
    """
    return config.RUNTIME_DIR / "usage.jsonl"


def daily_file() -> Path:
    return config.RUNTIME_DIR / "daily_usage.json"


@dataclass
class ModelUsage:
    calls: int = 0
    errors: int = 0
    rate_limits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls, "errors": self.errors, "rate_limits": self.rate_limits,
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class Ledger:
    """Process-wide totals. Per-turn and per-client figures come from :mod:`.spend`."""

    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    tools: Counter = field(default_factory=Counter)
    tool_errors: Counter = field(default_factory=Counter)
    # OpenRouter's free models are capped per day, resetting at midnight UTC. The count is
    # persisted so restarting the server cannot quietly exceed the allowance.
    provider_day: str = ""
    provider_calls: Counter = field(default_factory=Counter)
    cost_usd: float = 0.0            # billed chat spend, summed from each response's usage.cost
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _persist: bool = True

    # ── recording ────────────────────────────────────────────────────────────────────
    def _usage(self, model: str) -> ModelUsage:
        return self.by_model.setdefault(model, ModelUsage())

    def record_call(self, model: str, *, seconds: float = 0.0, prompt_tokens: int = 0,
                    completion_tokens: int = 0, session: str = "", stage: str = "",
                    cost_usd: float = 0.0) -> None:
        cost = max(0.0, float(cost_usd or 0.0))
        with self._lock:
            usage = self._usage(model)
            usage.calls += 1
            usage.seconds += seconds
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
            self.cost_usd += cost
        spend.charge(cost)
        self._append({"t": time.time(), "kind": "call", "model": model, "stage": stage,
                      "session": session, "seconds": round(seconds, 3),
                      "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "cost_usd": round(cost, 6)})

    def record_error(self, model: str, kind: str = "error", session: str = "",
                     stage: str = "", detail: str = "") -> None:
        with self._lock:
            usage = self._usage(model)
            if kind == "rate_limit":
                usage.rate_limits += 1
            else:
                usage.errors += 1
        self._append({"t": time.time(), "kind": kind, "model": model, "stage": stage,
                      "session": session, "detail": detail[:300]})

    def record_tool(self, name: str, ok: bool = True) -> None:
        with self._lock:
            (self.tools if ok else self.tool_errors)[name] += 1

    # ── the free tier's daily allowance ──────────────────────────────────────────────
    def note_provider_call(self, provider: str, model: str = "") -> None:
        """Count a call against the daily allowance. Paid models spend money, not allowance."""
        if provider == "openrouter" and model and not config.is_free_model(model):
            return
        today = time.strftime("%Y-%m-%d", time.gmtime())
        with self._lock:
            if self.provider_day != today:
                self.provider_day = today
                self.provider_calls.clear()
            self.provider_calls[provider] += 1
        self._persist_daily()

    def daily_remaining(self, provider: str = "openrouter") -> int:
        """Free-model requests left today before this provider's free models are skipped."""
        if provider != "openrouter":
            return 10**9
        limit = config.OPENROUTER_DAILY_LIMIT - config.OPENROUTER_DAILY_RESERVE
        return max(0, limit - self.provider_calls.get(provider, 0))

    def daily_exhausted(self, provider: str = "openrouter") -> bool:
        return self.daily_remaining(provider) <= 0

    def load_daily(self) -> None:
        """Restore today's counts; a different day starts clean."""
        try:
            data = json.loads(daily_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if data.get("day") == today:
            self.provider_day = today
            self.provider_calls = Counter(data.get("calls", {}))

    def _persist_daily(self) -> None:
        if not self._persist:
            return
        try:
            config.ensure_runtime_dirs()
            daily_file().write_text(json.dumps(
                {"day": self.provider_day, "calls": dict(self.provider_calls)}), encoding="utf-8")
        except OSError as exc:
            log.debug("could not persist the daily allowance: %s", exc)

    # ── the audit log ────────────────────────────────────────────────────────────────
    def _append(self, row: dict) -> None:
        if not self._persist:
            return
        try:
            config.ensure_runtime_dirs()
            path = usage_file()
            if path.exists() and path.stat().st_size > USAGE_LOG_MAX_BYTES:
                path.replace(path.with_suffix(".jsonl.1"))
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:  # accounting must never break the request
            log.debug("could not append usage row: %s", exc)

    # ── reporting ────────────────────────────────────────────────────────────────────
    @property
    def total_calls(self) -> int:
        return sum(u.calls for u in self.by_model.values())

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "by_model": {m: u.as_dict() for m, u in self.by_model.items()},
                "tools": dict(self.tools),
                "tool_errors": dict(self.tool_errors),
                "total_calls": self.total_calls,
                "provider_calls_today": dict(self.provider_calls),
                "openrouter_free_remaining_today": self.daily_remaining("openrouter"),
                "chat_cost_usd": round(self.cost_usd, 6),
            }


_LEDGER: Ledger | None = None


def get_ledger() -> Ledger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = Ledger()
        _LEDGER.load_daily()
    return _LEDGER
