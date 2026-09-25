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
from knowyourrights.llm import registry
from knowyourrights.llm.client import (
    NimClient, NimDeadlineExceeded, NimError, extract_json,
)
from knowyourrights.llm.limiter import TokenBucket


def chat_response(text: str) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


def _fast_candidates():
    """(first choice, the rest) for the fast role, as configured right now."""
    options = registry.candidates("fast")
    return options[0], options[1:]


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
    # Buckets are keyed by the provider-qualified id, so providers never share an allowance.
    assert all(":" in p[0] for p in pauses)
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
    first, alternates = _fast_candidates()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == first.id:
            return httpx.Response(410, json={"title": "Gone"})
        return chat_response("from the alternate")

    client = make_client(handler)
    reply = await client.chat([{"role": "user", "content": "hi"}], role="fast")

    assert reply == "from the alternate"
    assert seen[0] == first.id
    assert seen[1] in {spec.id for spec in alternates}
    registry._sidelined.clear()
    await client.aclose()


@pytest.mark.asyncio
async def test_failover_crosses_providers():
    """The whole point of two providers: NVIDIA going down must not take the answer with it.

    The fast role's candidate list deliberately alternates between NIM and OpenRouter, so one
    provider failing entirely still leaves somewhere to go.
    """
    registry._state = {"resolved": {}, "unavailable": [], "failures": {}}
    registry._sidelined.clear()
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if "nvidia" in request.url.host:
            return httpx.Response(503, text="upstream unavailable")
        return chat_response("answered by the other provider")

    client = make_client(handler)
    reply = await client.chat([{"role": "user", "content": "hi"}], role="fast",
                              deadline=time.monotonic() + 30)

    assert reply == "answered by the other provider"
    assert any("openrouter" in h for h in hosts), f"never reached OpenRouter: {hosts}"
    registry._sidelined.clear()
    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_gets_its_own_auth_and_attribution_headers():
    """Each provider needs its own key; OpenRouter also wants attribution on free models."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "openrouter" in request.url.host:
            captured.update(dict(request.headers))
            return chat_response("ok")
        return httpx.Response(410, json={"title": "Gone"})

    registry._state = {"resolved": {}, "unavailable": [], "failures": {}}
    registry._sidelined.clear()
    client = make_client(handler)
    await client.chat([{"role": "user", "content": "hi"}], role="fast")

    assert captured.get("x-title") == config.OPENROUTER_APP_NAME
    assert "http-referer" in captured
    registry._sidelined.clear()
    await client.aclose()


@pytest.mark.asyncio
async def test_spent_free_allowance_removes_only_free_models():
    """OpenRouter's 1,000/day cap covers ``:free`` models. Paid ones are metered in money.

    Regression: every OpenRouter call used to count against the free allowance, and a spent
    allowance removed *all* OpenRouter models — so ~6 paid fast calls a question would have
    locked the router out of the models it had just been told to prefer, after ~155 questions.
    """
    from knowyourrights.llm.ledger import get_ledger

    ledger = get_ledger()
    ledger.provider_calls.clear()
    ledger.provider_calls["openrouter"] = config.OPENROUTER_DAILY_LIMIT
    assert ledger.daily_exhausted("openrouter")

    fast = registry.spec("fast")
    assert fast.provider == "openrouter" and not config.is_free_model(fast.id),         "a spent *free* allowance must not take paid OpenRouter models out of rotation"

    writer = registry.spec("writer")
    assert not config.is_free_model(writer.id),         "with the free allowance spent, the writer must move off its free model"

    # and paid calls must not have been drawing it down in the first place
    ledger.provider_calls.clear()
    ledger.note_provider_call("openrouter", "google/gemini-2.5-flash-lite")
    assert ledger.provider_calls.get("openrouter", 0) == 0
    ledger.note_provider_call("openrouter", "nvidia/nemotron-3-super-120b-a12b:free")
    assert ledger.provider_calls.get("openrouter", 0) == 1
    ledger.provider_calls.clear()


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
    first, _ = _fast_candidates()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        model = json.loads(request.content)["model"]
        if model == first.id and calls["n"] == 1:
            return httpx.Response(410, json={"title": "Gone"})
        return chat_response(f"ok from {model}")

    client = make_client(handler)
    await client.chat([{"role": "user", "content": "hi"}], role="fast")

    assert first.key not in registry._state["unavailable"], \
        "one transient failure must not be written to disk"
    assert registry._state["failures"][first.key]["count"] == 1

    # After the in-process cooldown the model is tried again, and succeeding clears its streak.
    registry._sidelined.clear()
    reply = await client.chat([{"role": "user", "content": "hi"}], role="fast")
    assert reply == f"ok from {first.id}"
    assert not registry._state["failures"], "a success must clear the failure streak"
    await client.aclose()


@pytest.mark.asyncio
async def test_repeated_failures_are_eventually_recorded():
    """Persistent failure is different from a blip, and should be remembered."""
    registry._state = {"resolved": {}, "unavailable": [], "failures": {}}
    registry._sidelined.clear()

    first, _ = _fast_candidates()
    for _ in range(registry.PERSIST_AFTER_FAILURES):
        registry._sidelined.clear()
        registry.mark_unavailable(first.key, "HTTP 410")

    assert first.key in registry._state["unavailable"]
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
    """Measured 6x token saving on the structured stages; must be on the wire by default.

    Each provider takes its own spelling of the switch — NVIDIA a chat-template flag, OpenRouter
    a `reasoning` block — so check whichever the routed model actually needs.
    """
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.clear()
        captured.update(json.loads(request.content))
        return chat_response("ok")

    client = make_client(handler)
    provider = registry.spec("fast").provider
    await client.chat([{"role": "user", "content": "hi"}], role="fast")
    if provider == "openrouter":
        assert captured["reasoning"] == {"enabled": False}
    else:
        assert captured["chat_template_kwargs"] == {"thinking": False}

    await client.chat([{"role": "user", "content": "hi"}], role="fast", thinking=True)
    assert "chat_template_kwargs" not in captured and "reasoning" not in captured,         "explicit thinking=True must let the model reason"
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


@pytest.mark.asyncio
async def test_rejected_optional_parameter_retries_the_same_model_without_it():
    """Regression: a 400 over the `reasoning` switch used to raise, and inside chat_json that
    became the caller's *default* — a generic fallback plan, silently, on every call."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(("reasoning" in body) or ("chat_template_kwargs" in body))
        if seen[-1]:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: reasoning"}})
        return chat_response("ok")

    client = make_client(handler)
    out = await client.chat([{"role": "user", "content": "hi"}], role="fast")
    assert out == "ok"
    assert seen[:2] == [True, False], "must retry the same model once, without the parameter"
    seen.clear()
    await client.chat([{"role": "user", "content": "hi"}], role="fast")
    assert seen == [False], "the rejection must be remembered, not rediscovered every call"
    await client.aclose()


# ── streaming: how an answer ends ─────────────────────────────────────────────────────
def _sse(*frames: str) -> bytes:
    return "".join(f"data: {f}\n\n" for f in frames).encode()


def _delta(text: str, finish: str | None = None) -> str:
    choice = {"delta": {"content": text}}
    if finish:
        choice["finish_reason"] = finish
    return json.dumps({"choices": [choice]})


async def _drain(client, **kw) -> tuple[str, list[str]]:
    ending: list[str] = []
    out = []
    async for piece in client.chat_stream([{"role": "user", "content": "q"}],
                                          role="writer", on_finish=ending.append, **kw):
        out.append(piece)
    return "".join(out), ending


@pytest.mark.asyncio
async def test_a_normal_stream_reports_stop():
    def handler(request):
        return httpx.Response(200, content=_sse(_delta("Hello "), _delta("world", "stop"), "[DONE]"))
    client = make_client(handler)
    text, ending = await _drain(client)
    assert text == "Hello world" and ending == ["stop"]
    await client.aclose()


@pytest.mark.asyncio
async def test_hitting_the_token_limit_is_reported():
    """Regression: finish_reason was never read, so a cut-off answer looked complete."""
    def handler(request):
        return httpx.Response(200, content=_sse(_delta("Step one. Step tw", "length"), "[DONE]"))
    client = make_client(handler)
    text, ending = await _drain(client)
    assert ending == ["length"]
    await client.aclose()


@pytest.mark.asyncio
async def test_a_stream_that_just_stops_is_reported_as_interrupted():
    """Neither [DONE] nor a finish reason: the body ended early. That is a truncation."""
    def handler(request):
        return httpx.Response(200, content=_sse(_delta("Under Section 47 you")))
    client = make_client(handler)
    text, ending = await _drain(client)
    assert text == "Under Section 47 you" and ending == ["interrupted"]
    await client.aclose()


@pytest.mark.asyncio
async def test_a_dropped_connection_never_replays_the_answer():
    """Regression, and the worst of these: a transport error after the first token was retried
    by opening a fresh stream, which re-yielded the whole answer from the start."""
    calls = []

    class Dropping(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _sse(_delta("Under Section 47 you must be told "))
            raise httpx.ReadError("connection reset")

    def handler(request):
        calls.append(1)
        return httpx.Response(200, stream=Dropping())

    client = make_client(handler)
    text, ending = await _drain(client)
    assert len(calls) == 1, "must not open a second stream once text has been sent"
    assert text == "Under Section 47 you must be told "
    assert text.count("Section 47") == 1, "the answer must not start over"
    assert ending == ["interrupted"]
    await client.aclose()


@pytest.mark.asyncio
async def test_an_error_frame_before_any_text_fails_over():
    """An error inside a 200 with nothing sent yet is an ordinary failure — try again."""
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=_sse(json.dumps({"error": {"message": "overloaded"}})))
        return httpx.Response(200, content=_sse(_delta("ok", "stop"), "[DONE]"))

    client = make_client(handler)
    text, ending = await _drain(client)
    assert text == "ok" and ending == ["stop"] and len(calls) == 2
    await client.aclose()
