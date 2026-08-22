"""Which model actually answers for each role.

Hosted catalogues change: models get renamed, retired, or moved behind a different tier
(``baai/bge-m3`` is being withdrawn as this is written). Rather than hard-code an id and
discover the problem mid-answer, each role has an ordered list — the configured id followed
by alternates — and the first one that responds wins. ``scripts/probe_nim.py`` resolves them
once and caches the answer; runtime failures update the same cache.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Literal

from .. import config

log = logging.getLogger(__name__)

ModelRole = Literal["fast", "writer", "rerank"]


def probe_file():
    """Resolved on each use, not at import.

    A module-level constant would bake in ``RUNTIME_DIR`` before tests get a chance to
    redirect it — which is exactly how a mocked 410-failover test once retired a perfectly
    healthy model in the developer's real runtime state.
    """
    return config.RUNTIME_DIR / "nim_probe.json"

_ROLE_SPECS: dict[str, config.ModelSpec] = {
    "fast": config.FAST_MODEL,
    "writer": config.WRITER_MODEL,
}

_lock = threading.Lock()
_state: dict | None = None


def candidates(role: ModelRole) -> list[str]:
    """Configured id first, then declared alternates."""
    if role == "rerank":
        return [config.NIM_RERANK_MODEL, *config.NIM_RERANK_ALTERNATES]
    spec = _ROLE_SPECS[role]
    return [spec.id, *spec.alternates]


def spec(role: ModelRole) -> config.ModelSpec:
    """The tuning parameters for a role, with ``id`` set to the resolved model."""
    if role == "rerank":
        return config.ModelSpec(id=resolve("rerank"), rpm=config.NIM_RERANK_RPM,
                                ctx=8192, max_out=0, temperature=0.0)
    base = _ROLE_SPECS[role]
    resolved = resolve(role)
    if resolved == base.id:
        return base
    return config.ModelSpec(id=resolved, rpm=base.rpm, ctx=base.ctx,
                            max_out=base.max_out, temperature=base.temperature,
                            alternates=base.alternates)


def _load() -> dict:
    global _state
    if _state is None:
        try:
            _state = json.loads(probe_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _state = {"resolved": {}, "unavailable": []}
    return _state


def _save() -> None:
    config.ensure_runtime_dirs()
    try:
        probe_file().write_text(json.dumps(_load(), indent=2), encoding="utf-8")
    except OSError as exc:
        log.debug("could not persist model probe results: %s", exc)


def resolve(role: ModelRole) -> str:
    """The model id to use for ``role`` right now."""
    with _lock:
        state = _load()
        pinned = state.get("resolved", {}).get(role)
        unavailable = set(state.get("unavailable", []))
        if pinned and pinned not in unavailable:
            return pinned
        for candidate in candidates(role):
            if candidate not in unavailable:
                return candidate
        # Everything is marked dead — fall back to the configured id and let the call fail
        # with a real error rather than silently doing nothing.
        return candidates(role)[0]


def pin(role: ModelRole, model_id: str) -> None:
    """Record that ``model_id`` answered for ``role`` (used by the probe script)."""
    with _lock:
        state = _load()
        state.setdefault("resolved", {})[role] = model_id
        if model_id in state.get("unavailable", []):
            state["unavailable"].remove(model_id)
        _save()


def mark_unavailable(model_id: str, reason: str = "") -> str | None:
    """Retire a model id after a 404/400 and return the next candidate for its role."""
    with _lock:
        state = _load()
        unavailable = set(state.get("unavailable", []))
        if model_id not in unavailable:
            unavailable.add(model_id)
            state["unavailable"] = sorted(unavailable)
            log.warning("model %s marked unavailable%s", model_id, f" ({reason})" if reason else "")
        for role, options in (("fast", candidates("fast")),
                              ("writer", candidates("writer")),
                              ("rerank", candidates("rerank"))):
            if model_id in options:
                if state.get("resolved", {}).get(role) == model_id:
                    state["resolved"].pop(role, None)
                _save()
                nxt = next((c for c in options if c not in unavailable), None)
                if nxt:
                    log.warning("role %r falling back to %s", role, nxt)
                return nxt
        _save()
        return None


def snapshot() -> dict:
    state = _load()
    return {
        "roles": {role: resolve(role) for role in ("fast", "writer", "rerank")},
        "unavailable": list(state.get("unavailable", [])),
        "probe_file": str(probe_file()),
    }
