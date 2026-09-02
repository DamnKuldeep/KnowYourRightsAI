"""Which model, on which provider, answers for each role.

Two providers rather than one, because a single hosted catalogue turned out to be an
unreliable dependency. Observed on NVIDIA during development: a healthy model returning 410
Gone on one call and answering normally on the next, every reranking endpoint returning 410 or
404, plain 503s, and per-call latency swinging from 1 s to 2.6 s. None of that is a reason to
stop working — it is a reason to have somewhere else to go.

So each role has an ordered list of ``ModelSpec`` spanning both NVIDIA NIM and OpenRouter, and
the first that answers wins. Sidelining is temporary: a failure takes a model out of rotation
for this process, and only three failures inside an hour are written to disk. NVIDIA's 410s are
often transient, and permanently retiring a healthy model over one blip is the bug this policy
exists to prevent.
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

PERSIST_AFTER_FAILURES = 3
UNAVAILABLE_TTL_S = 3600.0

_sidelined: dict[str, float] = {}
_lock = threading.RLock()
_state: dict | None = None


def probe_file():
    """Resolved on each use, not at import.

    A module-level constant would bake in ``RUNTIME_DIR`` before tests can redirect it — which
    is exactly how a mocked failover test once retired a healthy model in the real runtime.
    """
    return config.RUNTIME_DIR / "model_probe.json"


# ── candidates ────────────────────────────────────────────────────────────────────────
def _parse_override(raw: str, fallback: config.ModelSpec) -> config.ModelSpec | None:
    """Accept "provider:model-id" or a bare model id (assumed NIM)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    provider, _, model_id = raw.partition(":")
    if provider not in config.PROVIDERS:
        provider, model_id = "nim", raw
    return config.ModelSpec(model_id, provider, rpm=fallback.rpm, ctx=fallback.ctx,
                            max_out=fallback.max_out, temperature=fallback.temperature)


def candidates(role: ModelRole) -> list[config.ModelSpec]:
    """Every model that could serve this role, best first, configured providers only."""
    if role == "rerank":
        specs = [config.ModelSpec(m, "nim", rpm=config.NIM_RERANK_RPM, ctx=8192, max_out=0)
                 for m in (config.NIM_RERANK_MODEL, *config.NIM_RERANK_ALTERNATES)]
    elif role == "fast":
        override = _parse_override(config.FAST_MODEL_OVERRIDE, config.FAST_MODELS[0])
        specs = [override] if override else list(config.FAST_MODELS)
    else:
        override = _parse_override(config.WRITER_MODEL_OVERRIDE, config.WRITER_MODELS[0])
        specs = [override] if override else list(config.WRITER_MODELS)

    return [s for s in specs if config.provider_available(s.provider)]


def spec(role: ModelRole) -> config.ModelSpec:
    """The model to use for ``role`` right now."""
    with _lock:
        state = _load()
        persisted = _expired_pruned(state)
        options = candidates(role)
        if not options:
            raise RuntimeError(
                f"No provider is configured for role {role!r}. Set NVIDIA_API_KEY or "
                f"OPENROUTER_API_KEY in .env."
            )

        # A provider whose daily allowance is spent is skipped rather than tried and refused.
        # OpenRouter's free tier is capped per day, so hitting it is a certainty, not an error.
        affordable = [o for o in options if _within_budget(o.provider)]
        usable = affordable or options          # all spent: try anyway rather than do nothing

        pinned = state.get("resolved", {}).get(role)
        for option in usable:
            if option.key == pinned and _usable(option.key, persisted):
                return option
        for option in usable:
            if _usable(option.key, persisted):
                return option

        # Everything is sidelined. Clear the temporary marks and try the best option again —
        # a real error is more useful than silent paralysis.
        _sidelined.clear()
        return options[0]


def resolve(role: ModelRole) -> str:
    """The provider-qualified id for a role, for reporting."""
    return spec(role).key


# ── availability bookkeeping ──────────────────────────────────────────────────────────
def _within_budget(provider: str) -> bool:
    """Does this provider have daily allowance left?"""
    if provider != "openrouter":
        return True
    from .ledger import get_ledger

    return not get_ledger().daily_exhausted("openrouter")


def _usable(key: str, persisted: set[str]) -> bool:
    if key in _sidelined:
        if time.time() - _sidelined[key] < UNAVAILABLE_TTL_S:
            return False
        del _sidelined[key]
    return key not in persisted


def _load() -> dict:
    global _state
    if _state is None:
        try:
            _state = json.loads(probe_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _state = {"resolved": {}, "unavailable": [], "failures": {}}
    return _state


def _save() -> None:
    config.ensure_runtime_dirs()
    try:
        probe_file().write_text(json.dumps(_load(), indent=2), encoding="utf-8")
    except OSError as exc:
        log.debug("could not persist model probe results: %s", exc)


def _expired_pruned(state: dict) -> set[str]:
    """Drop failures that have aged out, so a recovered model returns by itself."""
    now = time.time()
    failures = state.get("failures", {})
    persisted = set(state.get("unavailable", []))
    revived = [k for k in list(persisted)
               if now - failures.get(k, {}).get("last", 0) > UNAVAILABLE_TTL_S]
    if revived:
        for key in revived:
            persisted.discard(key)
            failures.pop(key, None)
            log.info("%s has been sidelined over an hour — trying it again", key)
        state["unavailable"] = sorted(persisted)
        _save()
    return persisted


def pin(role: ModelRole, key: str) -> None:
    """Record that ``key`` answered for ``role``."""
    with _lock:
        state = _load()
        state.setdefault("resolved", {})[role] = key
        if key in state.get("unavailable", []):
            state["unavailable"].remove(key)
        state.get("failures", {}).pop(key, None)
        _sidelined.pop(key, None)
        _save()


def mark_unavailable(key: str, reason: str = "") -> config.ModelSpec | None:
    """Take a model out of rotation and return the next candidate for its role."""
    now = time.time()
    with _lock:
        state = _load()
        failures = state.setdefault("failures", {})
        record = failures.get(key) or {"count": 0, "first": now}
        if now - record.get("last", record["first"]) > UNAVAILABLE_TTL_S:
            record = {"count": 0, "first": now}          # a new streak, not a continuation
        record.update(count=record["count"] + 1, last=now, reason=reason)
        failures[key] = record
        _sidelined[key] = now

        persisted = set(state.get("unavailable", []))
        if record["count"] >= PERSIST_AFTER_FAILURES and key not in persisted:
            persisted.add(key)
            state["unavailable"] = sorted(persisted)
            log.warning("%s failed %d times — recording it as unavailable%s",
                        key, record["count"], f" ({reason})" if reason else "")
        else:
            log.warning("%s unavailable this run (failure %d/%d)%s",
                        key, record["count"], PERSIST_AFTER_FAILURES,
                        f" ({reason})" if reason else "")

        for role in ("fast", "writer", "rerank"):
            options = candidates(role)
            if any(o.key == key for o in options):
                if state.get("resolved", {}).get(role) == key:
                    state["resolved"].pop(role, None)
                _save()
                nxt = next((o for o in options if _usable(o.key, persisted)), None)
                if nxt:
                    log.warning("role %r falling back to %s", role, nxt.key)
                else:
                    log.error("role %r has no reachable model left", role)
                return nxt
        _save()
        return None


def mark_available(key: str) -> None:
    """Clear a model's failure streak after it answers."""
    with _lock:
        was_sidelined = _sidelined.pop(key, None) is not None
        state = _load()
        changed = state.get("failures", {}).pop(key, None) is not None
        if key in state.get("unavailable", []):
            state["unavailable"].remove(key)
            log.info("%s is answering again — returning it to rotation", key)
            changed = True
        if changed or was_sidelined:
            _save()


def snapshot() -> dict:
    state = _load()
    roles = {}
    for role in ("fast", "writer", "rerank"):
        try:
            roles[role] = resolve(role)
        except RuntimeError:
            roles[role] = None
    return {
        "roles": roles,
        "candidates": {role: [s.key for s in candidates(role)]
                       for role in ("fast", "writer", "rerank")},
        "unavailable": list(state.get("unavailable", [])),
        "sidelined_this_run": sorted(_sidelined),
        "providers": {p: config.provider_available(p) for p in config.PROVIDERS},
    }
