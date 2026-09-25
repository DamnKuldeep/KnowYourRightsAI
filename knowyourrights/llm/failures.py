"""What to do when a provider answers with anything but success.

One policy for every call, returning the model to try next or raising:

* **429** — honour ``Retry-After``, tell the caller how long the pause is, shrink the bucket and
  retry the same model. Bounded by the turn's deadline, not an attempt count.
* **404 / 410 / unknown model, 402 / 403 quota** — take that model out of rotation and move on.
* **400 about an optional parameter** — retry the same model once without it.
* **Other 400s** — this model cannot serve the request; try the next.
* **5xx / 408** — back off once; on a second failure switch to the next model.
* **401** — the key was rejected. Raise, because no retry can fix configuration.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

import httpx

from .. import config
from . import registry
from .errors import DeadlineExceeded, LLMError, ModelUnavailable, ProviderAuthError
from .ledger import get_ledger
from .limiter import PauseCallback, get_limiters

log = logging.getLogger(__name__)

_OPTIONAL_PARAM_WORDS = ("reasoning", "chat_template_kwargs", "unsupported param",
                         "unrecognized", "unknown field", "extra inputs", "not permitted",
                         "response_format")
_UNKNOWN_MODEL_WORDS = ("not found", "does not exist", "invalid", "unavailable", "deprecat",
                        "retired")


def retry_after(response: httpx.Response) -> float | None:
    for header in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset"):
        try:
            value = float(response.headers.get(header) or 0)
        except ValueError:
            continue
        if value > 0:
            return min(value, config.RETRY_MAX_DELAY * 2)
    return None


def is_unknown_model(status: int, body: str) -> bool:
    # 410 Gone is how a catalogue reports a retired model.
    if status in (404, 410):
        return True
    lowered = body.lower()
    return status in (400, 422) and "model" in lowered and any(
        word in lowered for word in _UNKNOWN_MODEL_WORDS)


def is_parameter_complaint(body: str) -> bool:
    """Does a 400 blame one of our optional request fields rather than the model?"""
    lowered = (body or "").lower()
    return any(word in lowered for word in _OPTIONAL_PARAM_WORDS)


def check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise DeadlineExceeded(f"{stage}: ran out of time waiting for the model")


async def backoff(attempt: int, stage: str, deadline: float | None) -> None:
    delay = min(config.RETRY_MAX_DELAY,
                config.RETRY_INITIAL_DELAY * (config.RETRY_MULTIPLIER ** attempt))
    delay += random.uniform(0, delay * 0.25)
    check_deadline(deadline, stage)
    if deadline is not None and time.monotonic() + delay > deadline:
        raise DeadlineExceeded(f"{stage}: out of time during backoff")
    await asyncio.sleep(delay)


@dataclass
class CallContext:
    """The per-call facts every failure decision needs."""

    role: registry.ModelRole
    stage: str
    deadline: float | None = None
    on_pause: PauseCallback | None = None
    session: str = ""
    # Models that rejected an optional parameter; they are retried once without it.
    no_optional_params: set[str] = field(default_factory=set)


async def handle_failure(response: httpx.Response, body: str, spec: config.ModelSpec,
                         call: CallContext, attempt: int) -> config.ModelSpec:
    """Decide what a failed response means. Returns the spec to try next, or raises."""
    status = response.status_code
    if status == 429:
        return await _rate_limited(response, spec, call)
    if is_unknown_model(status, body) or status in (402, 403):
        kind = "unknown_model" if is_unknown_model(status, body) else "quota"
        return _next_model(spec, call, kind, f"HTTP {status}", body)
    if status in (400, 422):
        if spec.key not in call.no_optional_params and is_parameter_complaint(body):
            call.no_optional_params.add(spec.key)
            log.info("%s rejected an optional parameter — retrying without it", spec.key)
            return spec
        return _next_model(spec, call, "bad_request", f"HTTP {status}: {body[:80]}", body)
    if status >= 500 or status == 408:
        return await _server_error(spec, call, attempt, status, body)
    get_ledger().record_error(spec.key, "error", call.session, call.stage, body)
    if status == 401:
        env = "NVIDIA_API_KEY" if spec.provider == "nim" else "OPENROUTER_API_KEY"
        raise ProviderAuthError(f"{spec.provider} rejected its API key — check {env}")
    raise LLMError(f"{call.stage}: {spec.provider} returned HTTP {status}: {body[:300]}")


async def _rate_limited(response: httpx.Response, spec: config.ModelSpec,
                        call: CallContext) -> config.ModelSpec:
    get_ledger().record_error(spec.key, "rate_limit", call.session, call.stage)
    wait = get_limiters().get(spec.key, spec.rpm).penalize(retry_after(response))
    if call.on_pause is not None:
        result = call.on_pause(spec.key, wait, "rate limit")
        if asyncio.iscoroutine(result):
            await result
    check_deadline(call.deadline, call.stage)
    if call.deadline is not None and time.monotonic() + wait > call.deadline:
        raise DeadlineExceeded(f"{call.stage}: rate limited for {wait:.0f}s, longer than the "
                               f"turn's remaining time")
    await asyncio.sleep(wait)
    return spec


def _next_model(spec: config.ModelSpec, call: CallContext, kind: str, reason: str,
                body: str) -> config.ModelSpec:
    get_ledger().record_error(spec.key, kind, call.session, call.stage, body)
    nxt = registry.mark_unavailable(spec.key, reason, call.role)
    if nxt is None:
        raise ModelUnavailable(f"{call.stage}: no usable model for role {call.role!r}")
    if nxt.key != spec.key:
        log.warning("%s failed (%s) — switching to %s", spec.key, reason, nxt.key)
    return nxt


async def _server_error(spec: config.ModelSpec, call: CallContext, attempt: int, status: int,
                        body: str) -> config.ModelSpec:
    """One 5xx is a blip worth a retry; a second means switch to the next model."""
    get_ledger().record_error(spec.key, "server_error", call.session, call.stage, body)
    if attempt >= 1:
        nxt = registry.mark_unavailable(spec.key, f"HTTP {status} (repeated)", call.role)
        if nxt is not None and nxt.key != spec.key:
            log.warning("%s keeps failing (%s) — switching to %s", spec.key, status, nxt.key)
            return nxt
    await backoff(attempt, call.stage, call.deadline)
    return spec
