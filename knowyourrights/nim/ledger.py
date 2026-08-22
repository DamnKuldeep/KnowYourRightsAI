"""Usage accounting.

The free tier is metered in credits, so the agent needs to know how much it has spent in
order to downshift research depth before it runs out rather than after. Every call is also
appended to ``.runtime/usage.jsonl`` so a session can be audited after the fact.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from .. import config

log = logging.getLogger(__name__)

USAGE_FILE = config.RUNTIME_DIR / "usage.jsonl"


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
    """Process-wide totals plus a per-session view."""

    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    tools: Counter = field(default_factory=Counter)
    tool_errors: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _persist: bool = True

    # ── recording ────────────────────────────────────────────────────────────────────
    def _usage(self, model: str) -> ModelUsage:
        usage = self.by_model.get(model)
        if usage is None:
            usage = ModelUsage()
            self.by_model[model] = usage
        return usage

    def record_call(self, model: str, *, seconds: float = 0.0, prompt_tokens: int = 0,
                    completion_tokens: int = 0, session: str = "", stage: str = "") -> None:
        with self._lock:
            usage = self._usage(model)
            usage.calls += 1
            usage.seconds += seconds
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
        self._append({"t": time.time(), "kind": "call", "model": model, "stage": stage,
                      "session": session, "seconds": round(seconds, 3),
                      "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})

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
            if ok:
                self.tools[name] += 1
            else:
                self.tool_errors[name] += 1

    def _append(self, row: dict) -> None:
        if not self._persist:
            return
        try:
            config.ensure_runtime_dirs()
            with open(USAGE_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:  # accounting must never break the request
            log.debug("could not append usage row: %s", exc)

    # ── reporting ────────────────────────────────────────────────────────────────────
    @property
    def total_calls(self) -> int:
        return sum(u.calls for u in self.by_model.values())

    @property
    def estimated_credits(self) -> int:
        """One credit per request is the closest honest proxy we have for the free tier."""
        return self.total_calls

    def budget_pressure(self) -> float:
        """0.0 = plenty left, 1.0 = budget exhausted. 0.0 when no budget is configured."""
        if config.SESSION_CREDIT_BUDGET <= 0:
            return 0.0
        return min(1.0, self.estimated_credits / config.SESSION_CREDIT_BUDGET)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "by_model": {m: u.as_dict() for m, u in self.by_model.items()},
                "tools": dict(self.tools),
                "tool_errors": dict(self.tool_errors),
                "total_calls": self.total_calls,
                "estimated_credits": self.estimated_credits,
                "budget": config.SESSION_CREDIT_BUDGET or None,
                "budget_pressure": round(self.budget_pressure(), 3),
            }

    def report(self) -> str:
        lines = ["┌── NIM usage " + "─" * 52]
        for model, usage in sorted(self.by_model.items()):
            lines.append(f"│ {model:<38} calls={usage.calls:<4} 429s={usage.rate_limits:<3} "
                         f"errors={usage.errors:<3} {usage.seconds:.1f}s")
        if self.tools:
            lines.append("│ tools: " + ", ".join(f"{k}×{v}" for k, v in self.tools.items()))
        if self.tool_errors:
            lines.append("│ tool errors: " + ", ".join(f"{k}×{v}" for k, v in self.tool_errors.items()))
        lines.append(f"│ total calls {self.total_calls} (~{self.estimated_credits} credits)")
        lines.append("└" + "─" * 65)
        return "\n".join(lines)


_LEDGER: Ledger | None = None


def get_ledger() -> Ledger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = Ledger()
    return _LEDGER
