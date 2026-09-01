"""The rate limiter's contract: a 429 costs time, never the turn.

These use httpx's MockTransport so they exercise the real client code — header parsing,
AIMD, retirement on 410 — without touching the network or spending credits.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from knowyourrights import config
from knowyourrights.nim import registry
from knowyourrights.nim.client import (
    NimClient, NimDeadlineExceeded, NimError, extract_json,
)
from knowyourrights.nim.limiter import TokenBucket


def chat_response(text: str) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


def make_client(handler) -> NimClient:
    client = NimClient(api_key="test-key")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                       headers={"Authorization": "Bearer test-key"})
    return client


# ── the headline behaviour ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rate_limit_pauses_then_succeeds():
    """Two 429s with Retry-After must produce a delay and then a real answer."""
    calls = {"n": 0}
    pauses: list[tuple[str, float, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, headers={"Retry-After": "0.2"}, json={"error": "slow down"})
        return chat_response("here is your answer")

    client = make_client(handler)
    started = time.monotonic()
    reply = await client.chat([{"role": "user", "content": "hi"}],
                              on_pause=lambda m, s, r: pauses.append((m, s, r)))
    elapsed = time.monotonic() - started

    assert reply == "here is your answer"
    assert calls["n"] == 3, "should have retried past both rate limits"
    assert elapsed >= 0.4, "must actually honour Retry-After rather than hammering"
    assert pauses, "the UI must be told about a pause so it can show a countdown"
    # Both pause kinds are legitimate here: the 429s themselves, plus the slower pacing that
    # AIMD imposes afterwards. What matters is that the rate limits were reported as such.
    assert any(p[2] == "rate limit" for p in pauses)
    assert all(p[2] in ("rate limit", "pacing") for p in pauses)
    await client.aclose()


@pytest.mark.asyncio
async def test_rate_limit_shrinks_the_bucket_then_recovers():
    """AIMD: a 429 lowers the rate; the base rate is never exceeded."""
    bucket = TokenBucket("m", rpm=30)
    assert bucket.rpm == 30

    bucket.penalize(retry_after=0.01)
    assert bucket.rpm == pytest.approx(30 * config.AIMD_DECREASE)
    bucket.penalize(retry_after=0.01)
    assert bucket.rpm == pytest.approx(30 * config.AIMD_DECREASE ** 2)
    assert bucket.rpm >= config.AIMD_FLOOR_RPM

    for _ in range(40):
        bucket.penalize(retry_after=0.01)
    assert bucket.rpm == pytest.approx(config.AIMD_FLOOR_RPM), "must not shrink below the floor"

    # Pretend the penalty and the last recovery were both long ago.
    bucket._last_penalty = time.monotonic() - 120
    bucket._last_recovery = time.monotonic() - 120
    bucket._recover()
    assert bucket.rpm == pytest.approx(config.AIMD_FLOOR_RPM + config.AIMD_INCREASE)


@pytest.mark.asyncio
async def test_deadline_beats_infinite_retrying():
    """A permanently rate-limited provider must surface as a deadline, not an infinite wait."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "30"})

    client = make_client(handler)
    with pytest.raises(NimDeadlineExceeded):
        await client.chat([{"role": "user", "content": "hi"}],
                          deadline=time.monotonic() + 0.5)
    await client.aclose()


@pytest.mark.asyncio
async def test_unreachable_model_falls_through_to_an_alternate():
    """A 410 must fail over immediately — the current request still needs an answer."""
    registry._state = {"resolved": {}, "unavailable": [], "failures": {}}
    registry._sidelined.clear()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == config.FAST_MODEL.id:
            return httpx.Response(410, json={"title": "Gone"})
        return chat_response("from the alternate")

    client = make_client(handler)
    reply = await client.chat([{"role": "user", "content": "hi"}], role="fast")

    assert reply == "from the alternate"
    assert seen[0] == config.FAST_MODEL.id
    assert seen[1] in config.FAST_MODEL.alternates
    registry._sidelined.clear()
    await client.aclose()


@pytest.mark.asyncio
async def test_a_single_410_does_not_permanently_retire_a_model():
    """Regression, and it happened for real.

    NVIDIA returns 410 transiently under load, not only for genuinely retired models — a model
    410'd on one call here and answered normally on the next. Persisting the first failure
    meant one blip retired a healthy model on disk, and the app silently ran on its fallback
    from then on.
    """
    registry._state = {"resolved": {}, "unavailable": [], "failures": {}}
    registry._sidelined.clear()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        model = json.loads(request.content)["model"]
        if model == config.FAST_MODEL.id and calls["n"] == 1:
            return httpx.Response(410, json={"title": "Gone"})
        return chat_response(f"ok from {model}")

    client = make_client(handler)
    await client.chat([{"role": "user", "content": "hi"}], role="fast")

    assert config.FAST_MODEL.id not in registry._state["unavailable"], \
        "one transient failure must not be written to disk"
    assert registry._state["failures"][config.FAST_MODEL.id]["count"] == 1

    # After the in-process cooldown the model is tried again, and succeeding clears its streak.
    registry._sidelined.clear()
    reply = await client.chat([{"role": "user", "content": "hi"}], role="fast")
    assert reply == f"ok from {config.FAST_MODEL.id}"
    assert not registry._state["failures"], "a success must clear the failure streak"
    await client.aclose()


@pytest.mark.asyncio
async def test_repeated_failures_are_eventually_recorded():
    """Persistent failure is different from a blip, and should be remembered."""
    registry._state = {"resolved": {}, "unavailable": [], "failures": {}}
    registry._sidelined.clear()

    for _ in range(registry.PERSIST_AFTER_FAILURES):
        registry._sidelined.clear()
        registry.mark_unavailable(config.FAST_MODEL.id, "HTTP 410")

    assert config.FAST_MODEL.id in registry._state["unavailable"]
    registry._state = {"resolved": {}, "unavailable": [], "failures": {}}
    registry._sidelined.clear()


@pytest.mark.asyncio
async def test_server_errors_back_off_and_recover():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="upstream unavailable")
        return chat_response("recovered")

    client = make_client(handler)
    reply = await client.chat([{"role": "user", "content": "hi"}],
                              deadline=time.monotonic() + 30)
    assert reply == "recovered"
    assert calls["n"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried():
    """A bad key is a configuration problem — retrying just wastes the turn's deadline."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")

    client = make_client(handler)
    with pytest.raises(NimError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1
    await client.aclose()


# ── thinking suppression ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_thinking_disabled_by_default():
    """Measured 6x token saving on the structured stages; must be on the wire by default."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.clear()
        captured.update(json.loads(request.content))
        return chat_response("ok")

    client = make_client(handler)
    await client.chat([{"role": "user", "content": "hi"}], role="fast")
    assert captured["chat_template_kwargs"] == {"thinking": False}

    await client.chat([{"role": "user", "content": "hi"}], role="fast", thinking=True)
    assert "chat_template_kwargs" not in captured, "explicit thinking=True must let the model reason"
    await client.aclose()


@pytest.mark.asyncio
async def test_reasoning_content_is_used_when_content_is_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "", "reasoning_content": "the model thought this"}}]
        })

    client = make_client(handler)
    assert await client.chat([{"role": "user", "content": "hi"}]) == "the model thought this"
    await client.aclose()


# ── structured output ─────────────────────────────────────────────────────────────────
def test_extract_json_survives_prose_and_fences():
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == '{"a": 1}'
    assert extract_json('{"a": {"b": [1,2]}}') == '{"a": {"b": [1,2]}}'
    # A brace inside a string must not confuse the depth counter.
    assert extract_json('{"q": "what about {this}?"}') == '{"q": "what about {this}?"}'
    assert extract_json('{"q": "a \\" quote {"}') == '{"q": "a \\" quote {"}'


@pytest.mark.asyncio
async def test_chat_json_falls_back_to_default_after_bad_replies():
    from pydantic import BaseModel

    class Shape(BaseModel):
        kind: str

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("I'm afraid I can't do that.")

    client = make_client(handler)
    result = await client.chat_json([{"role": "user", "content": "classify"}],
                                    Shape, Shape(kind="fallback"))
    assert result.kind == "fallback", "a malformed reply must not fail the turn"
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_json_recovers_on_the_retry():
    from pydantic import BaseModel

    class Shape(BaseModel):
        kind: str

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return chat_response("nope" if calls["n"] == 1 else '{"kind": "legal_question"}')

    client = make_client(handler)
    result = await client.chat_json([{"role": "user", "content": "classify"}],
                                    Shape, Shape(kind="fallback"))
    assert result.kind == "legal_question"
    assert calls["n"] == 2
    await client.aclose()
