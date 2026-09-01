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
import time
from typing import Literal

from .. import config

log = logging.getLogger(__name__)

ModelRole = Literal["fast", "writer", "rerank"]

# How many failures before a model's unavailability is written to disk. One is too few:
# NVIDIA returns 410 under load as well as for genuine retirement.
PERSIST_AFTER_FAILURES = 3
# How long a recorded failure counts for. After this, the model is tried again — hosted
# catalogues recover, and nothing should be retired permanently by an outage.
UNAVAILABLE_TTL_S = 3600.0

# Models sidelined for *this process only*, so the current request fails over immediately
# without condemning the model for everyone.
_sidelined: dict[str, float] = {}


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


def _usable(model_id: str, persisted: set[str]) -> bool:
    """Is this model worth trying right now?"""
    if model_id in _sidelined:
        if time.time() - _sidelined[model_id] < UNAVAILABLE_TTL_S:
            return False
        del _sidelined[model_id]     # cooled off — give it another chance
    return model_id not in persisted


def resolve(role: ModelRole) -> str:
    """The model id to use for ``role`` right now."""
    with _lock:
        state = _load()
        persisted = _expired_pruned(state)
        pinned = state.get("resolved", {}).get(role)
        if pinned and _usable(pinned, persisted):
            return pinned
        for candidate in candidates(role):
            if _usable(candidate, persisted):
                return candidate
        # Everything is sidelined. Rather than do nothing, clear the temporary sidelining and
        # try the configured model again — a real error is more useful than silent paralysis.
        _sidelined.clear()
        return candidates(role)[0]


def _expired_pruned(state: dict) -> set[str]:
    """Drop recorded failures that have aged out, so recovered models return by themselves."""
    now = time.time()
    failures = state.get("failures", {})
    persisted = set(state.get("unavailable", []))
    revived = [m for m in list(persisted)
               if now - failures.get(m, {}).get("last", 0) > UNAVAILABLE_TTL_S]
    if revived:
        for model_id in revived:
            persisted.discard(model_id)
            failures.pop(model_id, None)
            log.info("model %s has been sidelined for over an hour — trying it again", model_id)
        state["unavailable"] = sorted(persisted)
        _save()
    return persisted


def pin(role: ModelRole, model_id: str) -> None:
    """Record that ``model_id`` answered for ``role`` (used by the probe script)."""
    with _lock:
        state = _load()
        state.setdefault("resolved", {})[role] = model_id
        if model_id in state.get("unavailable", []):
            state["unavailable"].remove(model_id)
        _save()


def mark_unavailable(model_id: str, reason: str = "") -> str | None:
    """Take a model out of rotation after a 404/410 and return the next candidate.

    Sidelining is **temporary and in-memory first**. NVIDIA returns 410 transiently under
    load, not only for genuinely retired models — observed live: a model 410'd on one call and
    answered normally on the next. Persisting the first 410 to disk meant one blip retired a
    healthy model until somebody hand-edited the file, and the app quietly ran on its fallback
    from then on.

    So: fail over immediately (the current request still needs an answer), but only write the
    verdict to disk once a model has failed ``PERSIST_AFTER_FAILURES`` times, and let even that
    expire after ``UNAVAILABLE_TTL_S`` so a recovered model comes back on its own.
    """
    now = time.time()
    with _lock:
        state = _load()
        failures = state.setdefault("failures", {})
        record = failures.get(model_id) or {"count": 0, "first": now}
        # A failure long after the last one starts a new streak rather than compounding.
        if now - record.get("last", record["first"]) > UNAVAILABLE_TTL_S:
            record = {"count": 0, "first": now}
        record["count"] += 1
        record["last"] = now
        record["reason"] = reason
        failures[model_id] = record

        _sidelined[model_id] = now
        persisted = set(state.get("unavailable", []))
        if record["count"] >= PERSIST_AFTER_FAILURES and model_id not in persisted:
            persisted.add(model_id)
            state["unavailable"] = sorted(persisted)
            log.warning("model %s failed %d times — recording it as unavailable%s",
                        model_id, record["count"], f" ({reason})" if reason else "")
        else:
            log.warning("model %s unavailable this run (failure %d/%d)%s",
                        model_id, record["count"], PERSIST_AFTER_FAILURES,
                        f" ({reason})" if reason else "")

        if state.get("resolved", {}).get(model_id) == model_id:
            state["resolved"].pop(model_id, None)
        for role, options in (("fast", candidates("fast")),
                              ("writer", candidates("writer")),
                              ("rerank", candidates("rerank"))):
            if model_id in options:
                if state.get("resolved", {}).get(role) == model_id:
                    state["resolved"].pop(role, None)
                _save()
                nxt = next((c for c in options if _usable(c, persisted)), None)
                if nxt:
                    log.warning("role %r falling back to %s", role, nxt)
                else:
                    log.error("role %r has no reachable model left", role)
                return nxt
        _save()
        return None


def mark_available(model_id: str) -> None:
    """Clear a model's failure streak after it answers successfully."""
    with _lock:
        _sidelined.pop(model_id, None)
        state = _load()
        changed = False
        if state.get("failures", {}).pop(model_id, None) is not None:
            changed = True
        if model_id in state.get("unavailable", []):
            state["unavailable"].remove(model_id)
            log.info("model %s is answering again — returning it to rotation", model_id)
            changed = True
        if changed:
            _save()


def snapshot() -> dict:
    state = _load()
    return {
        "roles": {role: resolve(role) for role in ("fast", "writer", "rerank")},
        "unavailable": list(state.get("unavailable", [])),
        "probe_file": str(probe_file()),
    }
