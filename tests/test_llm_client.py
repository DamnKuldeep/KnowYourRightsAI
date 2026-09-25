"""The model client's contract: a rate limit costs time, never the turn.

These use httpx's MockTransport, so they exercise the real client code (header parsing, AIMD,
failover, stream endings) without touching the network or spending anything.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
from pydantic import BaseModel

from knowyourrights import config
from knowyourrights.llm import registry, spend
from knowyourrights.llm.client import LLMClient, extract_json
from knowyourrights.llm.errors import DeadlineExceeded, LLMError, ProviderAuthError
from knowyourrights.llm.ledger import get_ledger
from knowyourrights.llm.limiter import TokenBucket

HI = [{"role": "user", "content": "hi"}]


class Shape(BaseModel):
    kind: str


def chat_response(text: str, cost: float = 0.0) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost},
    })


def make_client(handler) -> LLMClient:
    client = LLMClient(nvidia_api_key="test-key")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def model_of(request: httpx.Request) -> str:
    return json.loads(request.content)["model"]


# ── rate limits and deadlines ─────────────────────────────────────────────────────────
async def test_rate_limit_pauses_then_succeeds():
    """Two 429s with Retry-After produce a visible pause and then a real answer."""
    calls = {"n": 0}
    pauses: list[tuple[str, float, str]] = []

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, headers={"Retry-After": "0.2"}, json={"error": "slow"})
        return chat_response("here is your answer")

    client = make_client(handler)
    started = time.monotonic()
    reply = await client.chat(HI, on_pause=lambda m, s, r: pauses.append((m, s, r)))
    assert reply == "here is your answer"
    assert calls["n"] == 3
    assert time.monotonic() - started >= 0.4, "Retry-After must be honoured, not hammered"
    assert any(p[2] == "rate limit" for p in pauses), "the UI needs the pause to count down"
    assert all(":" in p[0] for p in pauses), "buckets are keyed by provider-qualified id"
    await client.aclose()


def test_rate_limit_shrinks_the_bucket_then_recovers():
    """AIMD: a 429 lowers the rate towards a floor; clean minutes raise it back."""
    bucket = TokenBucket("m", rpm=30)
    bucket.penalize(retry_after=0.01)
    assert bucket.rpm == pytest.approx(30 * config.AIMD_DECREASE)
    for _ in range(40):
        bucket.penalize(retry_after=0.01)
    assert bucket.rpm == pytest.approx(config.AIMD_FLOOR_RPM)
    bucket._last_penalty = bucket._last_recovery = time.monotonic() - 120
    bucket._recover()
    assert bucket.rpm == pytest.approx(config.AIMD_FLOOR_RPM + config.AIMD_INCREASE)


async def test_deadline_beats_infinite_retrying():
    client = make_client(lambda request: httpx.Response(429, headers={"Retry-After": "30"}))
    with pytest.raises(DeadlineExceeded):
        await client.chat(HI, deadline=time.monotonic() + 0.5)
    await client.aclose()


# ── failover ──────────────────────────────────────────────────────────────────────────
async def test_a_gone_model_falls_through_to_the_next():
    first = registry.candidates("fast")[0]
    seen: list[str] = []

    def handler(request):
        seen.append(model_of(request))
        if seen[-1] == first.id:
            return httpx.Response(410, json={"title": "Gone"})
        return chat_response("from the alternate")

    client = make_client(handler)
    assert await client.chat(HI, role="fast") == "from the alternate"
    assert seen[0] == first.id and seen[1] != first.id
    await client.aclose()


async def test_failover_crosses_providers():
    """OpenRouter failing outright must not take the answer with it."""
    hosts: list[str] = []

    def handler(request):
        hosts.append(request.url.host)
        if "openrouter" in request.url.host:
            return httpx.Response(503, text="upstream unavailable")
        return chat_response("answered by NVIDIA")

    client = make_client(handler)
    reply = await client.chat(HI, role="fast", deadline=time.monotonic() + 30)
    assert reply == "answered by NVIDIA"
    assert any("nvidia" in h for h in hosts)
    await client.aclose()


async def test_a_writer_failure_stays_on_the_writers_list():
    """Regression: a model serving two roles failed as the writer and the answer was handed to
    the planner's next model, because the fallback was looked up for the first role found."""
    shared = config.WRITER_MODELS[1]                  # also a fast-role model
    assert any(s.id == shared.id for s in config.FAST_MODELS)
    nxt = registry.mark_unavailable(shared.key, "test", role="writer")
    writer_ids = [s.id for s in config.WRITER_MODELS]
    assert nxt is not None and nxt.id in writer_ids


async def test_a_single_410_does_not_permanently_retire_a_model():
    """A provider returns 410 transiently under load. One blip must not be written to disk."""
    first = registry.candidates("fast")[0]
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if model_of(request) == first.id and calls["n"] == 1:
            return httpx.Response(410, json={"title": "Gone"})
        return chat_response(f"ok from {model_of(request)}")

    client = make_client(handler)
    await client.chat(HI, role="fast")
    assert first.key not in registry._state["unavailable"]
    assert registry._state["failures"][first.key]["count"] == 1
    registry._sidelined.clear()
    assert await client.chat(HI, role="fast") == f"ok from {first.id}"
    assert not registry._state["failures"], "a success clears the failure streak"
    await client.aclose()


def test_repeated_failures_are_eventually_recorded():
    first = registry.candidates("fast")[0]
    for _ in range(registry.PERSIST_AFTER_FAILURES):
        registry._sidelined.clear()
        registry.mark_unavailable(first.key, "HTTP 410", role="fast")
    assert first.key in registry._state["unavailable"]


async def test_server_errors_back_off_and_recover():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else chat_response("recovered")

    client = make_client(handler)
    assert await client.chat(HI, deadline=time.monotonic() + 30) == "recovered"
    assert calls["n"] == 2
    await client.aclose()


async def test_a_rejected_key_is_reported_not_retried():
    """A bad key is a configuration problem: retrying only wastes the turn's time."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")

    client = make_client(handler)
    with pytest.raises(ProviderAuthError, match="OPENROUTER_API_KEY"):
        await client.chat(HI)
    assert calls["n"] == 1
    await client.aclose()


def test_a_spent_free_allowance_removes_only_free_models():
    """Only ``:free`` models draw on OpenRouter's daily allowance; paid ones cost money."""
    ledger = get_ledger()
    ledger.provider_calls["openrouter"] = config.OPENROUTER_DAILY_LIMIT
    assert ledger.daily_exhausted("openrouter")
    fast = registry.spec("fast")
    assert fast.provider == "openrouter" and not config.is_free_model(fast.id)
    assert not config.is_free_model(registry.spec("writer").id)
    ledger.provider_calls.clear()
    ledger.note_provider_call("openrouter", "google/gemini-2.5-flash-lite")
    assert ledger.provider_calls.get("openrouter", 0) == 0
    ledger.note_provider_call("openrouter", "nvidia/nemotron-3-super-120b-a12b:free")
    assert ledger.provider_calls["openrouter"] == 1


# ── payloads and replies ──────────────────────────────────────────────────────────────
async def test_thinking_is_off_by_default():
    captured: dict = {}

    def handler(request):
        captured.clear()
        captured.update(json.loads(request.content))
        return chat_response("ok")

    client = make_client(handler)
    await client.chat(HI, role="fast")
    assert captured["reasoning"] == {"enabled": False}
    await client.chat(HI, role="fast", thinking=True)
    assert "reasoning" not in captured and "chat_template_kwargs" not in captured
    await client.aclose()


async def test_a_rejected_optional_parameter_is_dropped_once_and_remembered():
    seen: list[bool] = []

    def handler(request):
        body = json.loads(request.content)
        seen.append("reasoning" in body)
        if seen[-1]:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: "
                                                                  "reasoning"}})
        return chat_response("ok")

    client = make_client(handler)
    assert await client.chat(HI, role="fast") == "ok"
    assert seen[:2] == [True, False]
    seen.clear()
    await client.chat(HI, role="fast")
    assert seen == [False]
    await client.aclose()


async def test_reasoning_content_is_used_when_content_is_empty():
    client = make_client(lambda r: httpx.Response(200, json={
        "choices": [{"message": {"content": "", "reasoning_content": "thought"}}]}))
    assert await client.chat(HI) == "thought"
    await client.aclose()


async def test_billed_cost_is_charged_to_the_running_turn():
    client = make_client(lambda r: chat_response("ok", cost=0.0012))
    meter = spend.SpendMeter()
    with spend.metered(meter):
        await client.chat(HI)
    assert meter.cost_usd == pytest.approx(0.0012) and meter.calls == 1
    await client.aclose()


# ── structured output ─────────────────────────────────────────────────────────────────
def test_extract_json_survives_prose_and_fences():
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == '{"a": 1}'
    assert extract_json('{"q": "what about {this}?"}') == '{"q": "what about {this}?"}'
    assert extract_json('{"q": "a \\" quote {"}') == '{"q": "a \\" quote {"}'


async def test_chat_json_falls_back_to_its_default_when_a_reply_never_parses():
    client = make_client(lambda r: chat_response("I'm afraid I can't do that."))
    result = await client.chat_json(HI, Shape, Shape(kind="fallback"))
    assert result.kind == "fallback"
    await client.aclose()


async def test_chat_json_recovers_on_the_retry():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return chat_response("nope" if calls["n"] == 1 else '{"kind": "legal_question"}')

    client = make_client(handler)
    assert (await client.chat_json(HI, Shape, Shape(kind="x"))).kind == "legal_question"
    assert calls["n"] == 2
    await client.aclose()


async def test_chat_json_raises_provider_failures_for_the_caller_to_handle():
    """A failing provider is not a malformed reply: the caller decides how to degrade."""
    client = make_client(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(LLMError):
        await client.chat_json(HI, Shape, Shape(kind="fallback"))
    await client.aclose()


# ── streaming: how an answer ends ─────────────────────────────────────────────────────
def _sse(*frames: str) -> bytes:
    return "".join(f"data: {f}\n\n" for f in frames).encode()


def _delta(text: str, finish: str | None = None) -> str:
    choice = {"delta": {"content": text}}
    if finish:
        choice["finish_reason"] = finish
    return json.dumps({"choices": [choice]})


async def _drain(client) -> tuple[str, list[str]]:
    ending: list[str] = []
    out = [piece async for piece in client.chat_stream(HI, role="writer",
                                                       on_finish=ending.append)]
    return "".join(out), ending


async def test_a_normal_stream_reports_stop():
    client = make_client(lambda r: httpx.Response(
        200, content=_sse(_delta("Hello "), _delta("world", "stop"), "[DONE]")))
    assert await _drain(client) == ("Hello world", ["stop"])
    await client.aclose()


async def test_hitting_the_token_limit_is_reported():
    client = make_client(lambda r: httpx.Response(
        200, content=_sse(_delta("Step one. Step tw", "length"), "[DONE]")))
    assert (await _drain(client))[1] == ["length"]
    await client.aclose()


async def test_a_stream_that_just_stops_is_reported_as_interrupted():
    client = make_client(lambda r: httpx.Response(200, content=_sse(_delta("Under Sec"))))
    assert await _drain(client) == ("Under Sec", ["interrupted"])
    await client.aclose()


async def test_a_dropped_connection_never_replays_the_answer():
    """Regression: a transport error after the first token opened a fresh stream, which
    re-yielded the whole answer from the start."""
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
    assert len(calls) == 1 and text.count("Section 47") == 1
    assert ending == ["interrupted"]
    await client.aclose()


async def test_an_error_frame_before_any_text_fails_over():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=_sse(json.dumps({"error": {"message": "busy"}})))
        return httpx.Response(200, content=_sse(_delta("ok", "stop"), "[DONE]"))

    client = make_client(handler)
    assert await _drain(client) == ("ok", ["stop"])
    assert len(calls) == 2
    await client.aclose()
