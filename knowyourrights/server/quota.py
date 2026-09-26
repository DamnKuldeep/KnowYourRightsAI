"""Per-client limits: a spending allowance, a request rate, and the service's daily ceiling.

A client is an IP address, and is stored only as a salted hash: the book records how much each
address has spent without keeping the address itself. The book is persisted, so a restart does
not hand every client a fresh allowance.

Budgets are checked before a question starts. A question already running is allowed to finish,
so a client can end slightly over its allowance by at most the cost of one answer (typically well
under a cent), which is far better than cutting an answer off mid-sentence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Request

from .. import config

log = logging.getLogger(__name__)


def book_file():
    return config.RUNTIME_DIR / "client_spend.json"


def client_ip(request: Request) -> str:
    """The caller's address. Proxy headers are trusted only when configured to be."""
    if config.TRUST_PROXY_HEADERS:
        for header in ("cf-connecting-ip", "x-real-ip"):
            if value := request.headers.get(header, "").strip():
                return value
        if forwarded := request.headers.get("x-forwarded-for", ""):
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@dataclass
class Verdict:
    allowed: bool
    kind: str = ""               # rate | client_budget | daily_budget
    message: str = ""
    retry_after_s: int = 0


class SpendBook:
    """What each client, and the whole service today, has spent. Thread-safe and persisted."""

    SAVE_INTERVAL_S = 2.0      # a crash loses at most this much of the record

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = self._load()
        self._dirty = False
        self._saved_at = 0.0

    def _load(self) -> dict:
        try:
            data = json.loads(book_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("salt", secrets.token_hex(16))
        data.setdefault("day", _today())
        data.setdefault("daily_usd", 0.0)
        data.setdefault("clients", {})
        return data

    def key(self, ip: str) -> str:
        return hashlib.sha256(f"{self._data['salt']}|{ip}".encode()).hexdigest()[:24]

    def spent(self, key: str) -> float:
        with self._lock:
            return self._entry(key)["usd"]

    def daily_spent(self) -> float:
        with self._lock:
            self._roll_day()
            return self._data["daily_usd"]

    def charge(self, key: str, usd: float) -> None:
        if usd <= 0:
            return
        with self._lock:
            self._roll_day()
            self._entry(key)["usd"] += usd
            self._data["daily_usd"] += usd
            self._dirty = True
            if time.monotonic() - self._saved_at >= self.SAVE_INTERVAL_S:
                self._save()

    def reset(self, key: str, *, day: bool = False) -> None:
        """Forget what this client has spent and restart its window; optionally today's total."""
        with self._lock:
            self._data["clients"][key] = {"usd": 0.0, "since": time.time()}
            if day:
                self._data["daily_usd"] = 0.0
            self._save()

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._save()

    def _entry(self, key: str) -> dict:
        """This client's record, reset once its budget window has passed."""
        now = time.time()
        entry = self._data["clients"].setdefault(key, {"usd": 0.0, "since": now})
        window_s = config.CLIENT_BUDGET_WINDOW_H * 3600
        if window_s > 0 and now - entry["since"] > window_s:
            entry.update(usd=0.0, since=now)
        return entry

    def _roll_day(self) -> None:
        if self._data["day"] != _today():
            self._data["day"], self._data["daily_usd"] = _today(), 0.0

    def _save(self) -> None:
        try:
            config.ensure_runtime_dirs()
            path = book_file()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data), encoding="utf-8")
            os.replace(tmp, path)
            self._dirty, self._saved_at = False, time.monotonic()
        except OSError as exc:
            log.warning("could not persist the spend book: %s", exc)

    def stats(self) -> dict:
        with self._lock:
            return {"clients": len(self._data["clients"]),
                    "daily_usd": round(self._data["daily_usd"], 4), "day": self._data["day"]}


class RateLimiter:
    """At most ``per_minute`` requests per client in any rolling minute."""

    def __init__(self, per_minute: int | None = None) -> None:
        self.per_minute = per_minute or config.CLIENT_RPM
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> int:
        """0 if allowed (and counted), else seconds until the next request would be."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return max(1, int(60 - (now - hits[0])) + 1)
        hits.append(now)
        if len(self._hits) > 10_000:           # forget clients idle for a minute
            for idle in [k for k, h in self._hits.items() if not h or now - h[-1] > 60]:
                del self._hits[idle]
        return 0


class Guard:
    """Everything that decides whether a client may ask a question right now."""

    def __init__(self) -> None:
        self.book = SpendBook()
        self.rate = RateLimiter()

    def check(self, key: str) -> Verdict:
        if self.day_exhausted():
            return Verdict(False, "daily_budget",
                           "This free service has reached today's usage limit. Please come back "
                           "tomorrow.")
        if self.exhausted(key):
            return Verdict(False, "client_budget", self._budget_message())
        wait = self.rate.check(key)
        if wait:
            return Verdict(False, "rate", f"You are asking questions very quickly. Please wait "
                                          f"{wait} seconds and try again.", retry_after_s=wait)
        return Verdict(True)

    def day_exhausted(self) -> bool:
        return 0 < config.DAILY_BUDGET_USD <= self.book.daily_spent()

    def exhausted(self, key: str) -> bool:
        return 0 < config.CLIENT_BUDGET_USD <= self.book.spent(key)

    def quota(self, key: str) -> dict:
        spent = self.book.spent(key)
        budget = config.CLIENT_BUDGET_USD
        return {"spent_usd": round(spent, 4), "budget_usd": budget or None,
                "remaining_usd": round(max(0.0, budget - spent), 4) if budget else None,
                "exhausted": self.exhausted(key),
                "resets_after_hours": config.CLIENT_BUDGET_WINDOW_H or None}

    @staticmethod
    def _budget_message() -> str:
        window = config.CLIENT_BUDGET_WINDOW_H
        when = (f" It resets {window:g} hours after your first question." if window
                else "")
        return (f"You have used this free service's allowance of "
                f"${config.CLIENT_BUDGET_USD:.2f} for your connection, so it cannot answer more "
                f"questions for you.{when}")


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())
