"""A small sqlite-backed cache with TTLs, plus an in-memory LRU in front of it.

Caching is the cheapest latency and rate-limit win available: repeated queries skip the GPU,
repeated pages skip the network, and repeated turns skip the LLM entirely. sqlite is stdlib,
survives restarts, and handles our write volume without ceremony — no extra dependency.

Namespaces in use: ``embed``, ``search``, ``web``, ``crawl``, ``turn``, ``wiki``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any

from .. import config

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    ns      TEXT NOT NULL,
    k       TEXT NOT NULL,
    v       BLOB NOT NULL,
    kind    TEXT NOT NULL DEFAULT 'json',
    created REAL NOT NULL,
    expires REAL,
    PRIMARY KEY (ns, k)
);
CREATE INDEX IF NOT EXISTS kv_expires ON kv(expires);
"""


def key_of(*parts: Any) -> str:
    """Stable short key for arbitrary inputs (queries, URLs, option dicts)."""
    blob = "\x1f".join(
        json.dumps(p, sort_keys=True, ensure_ascii=False, default=str) if not isinstance(p, str) else p
        for p in parts
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class Cache:
    """Thread-safe. One connection guarded by a lock — writes are tiny and infrequent."""

    def __init__(self, path=None, memory_items: int = 512) -> None:
        config.ensure_runtime_dirs()
        self.path = str(path or (config.CACHE_DIR / "kyr_cache.db"))
        self._lock = threading.Lock()
        self._mem: OrderedDict[tuple[str, str], tuple[float | None, Any]] = OrderedDict()
        self._mem_max = memory_items
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.executescript(_SCHEMA)
        # WAL keeps readers from blocking the single writer.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            pass
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    # ── memory tier ──────────────────────────────────────────────────────────────────
    def _mem_get(self, ns: str, k: str):
        item = self._mem.get((ns, k))
        if item is None:
            return None
        expires, value = item
        if expires is not None and expires < time.time():
            self._mem.pop((ns, k), None)
            return None
        self._mem.move_to_end((ns, k))
        return value

    def _mem_put(self, ns: str, k: str, value: Any, expires: float | None) -> None:
        self._mem[(ns, k)] = (expires, value)
        self._mem.move_to_end((ns, k))
        while len(self._mem) > self._mem_max:
            self._mem.popitem(last=False)

    # ── raw bytes ────────────────────────────────────────────────────────────────────
    def get_bytes(self, ns: str, k: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT v, expires FROM kv WHERE ns=? AND k=?", (ns, k)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        value, expires = row
        if expires is not None and expires < time.time():
            self.delete(ns, k)
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set_bytes(self, ns: str, k: str, value: bytes, ttl: float | None = None,
                  kind: str = "bytes") -> None:
        expires = time.time() + ttl if ttl else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv(ns,k,v,kind,created,expires) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(ns,k) DO UPDATE SET v=excluded.v, kind=excluded.kind, "
                "created=excluded.created, expires=excluded.expires",
                (ns, k, value, kind, time.time(), expires),
            )
            self._conn.commit()

    # ── json ─────────────────────────────────────────────────────────────────────────
    def get_json(self, ns: str, k: str):
        cached = self._mem_get(ns, k)
        if cached is not None:
            self.hits += 1
            return cached
        raw = self.get_bytes(ns, k)
        if raw is None:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.delete(ns, k)
            return None
        self._mem_put(ns, k, value, None)
        return value

    def set_json(self, ns: str, k: str, value: Any, ttl: float | None = None) -> None:
        raw = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self.set_bytes(ns, k, raw, ttl, kind="json")
        self._mem_put(ns, k, value, time.time() + ttl if ttl else None)

    # ── vectors ──────────────────────────────────────────────────────────────────────
    def get_vector(self, k: str):
        """Returns a float32 numpy array, or None. Used for the query-embedding cache."""
        cached = self._mem_get("embed", k)
        if cached is not None:
            self.hits += 1
            return cached
        raw = self.get_bytes("embed", k)
        if raw is None:
            return None
        import numpy as np

        vec = np.frombuffer(raw, dtype="float32")
        if vec.size != config.EMBED_DIM:  # stale entry from a different embedder
            self.delete("embed", k)
            return None
        self._mem_put("embed", k, vec, None)
        return vec

    def set_vector(self, k: str, vec) -> None:
        import numpy as np

        arr = np.asarray(vec, dtype="float32").reshape(-1)
        self.set_bytes("embed", k, arr.tobytes(), ttl=None, kind="vec")
        self._mem_put("embed", k, arr, None)

    # ── maintenance ──────────────────────────────────────────────────────────────────
    def delete(self, ns: str, k: str) -> None:
        self._mem.pop((ns, k), None)
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE ns=? AND k=?", (ns, k))
            self._conn.commit()

    def clear(self, ns: str | None = None) -> int:
        with self._lock:
            if ns:
                cur = self._conn.execute("DELETE FROM kv WHERE ns=?", (ns,))
                self._mem = OrderedDict((key, v) for key, v in self._mem.items() if key[0] != ns)
            else:
                cur = self._conn.execute("DELETE FROM kv")
                self._mem.clear()
            self._conn.commit()
            return cur.rowcount

    def purge_expired(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM kv WHERE expires IS NOT NULL AND expires < ?", (time.time(),)
            )
            self._conn.commit()
            return cur.rowcount

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ns, COUNT(*), SUM(LENGTH(v)) FROM kv GROUP BY ns"
            ).fetchall()
        total = self.hits + self.misses
        return {
            "namespaces": {ns: {"items": n, "bytes": int(b or 0)} for ns, n, b in rows},
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "memory_items": len(self._mem),
            "path": self.path,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


_CACHE: Cache | None = None


def get_cache() -> Cache:
    global _CACHE
    if _CACHE is None:
        _CACHE = Cache()
    return _CACHE
