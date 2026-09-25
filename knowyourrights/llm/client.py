"""The chat client: one shared connection pool, two OpenAI-compatible providers.

Built on plain ``httpx`` rather than an SDK: the raw ``Retry-After`` header, provider-specific
payload fields and control over exactly when a retry is worth attempting are all things we need
to own. Failure handling lives in :mod:`.failures`, stream parsing in :mod:`.streaming`.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence

import httpx

from .. import config
from . import registry
from .errors import DeadlineExceeded, LLMError, ModelUnavailable, ProviderAuthError
from .failures import CallContext, backoff, check_deadline, handle_failure
from .ledger import get_ledger
from .limiter import PauseCallback, get_limiters
from .streaming import StreamOutcome, read_stream

log = logging.getLogger(__name__)

__all__ = ["DeadlineExceeded", "LLMClient", "LLMError", "ModelUnavailable",
           "ProviderAuthError", "extract_json", "get_client"]

_TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.TransportError)


def extract_json(text: str) -> str:
    """The first balanced JSON object or array in a model reply.

    Models wrap JSON in prose or code fences often enough that scanning for the first balanced
    structure is more reliable than trusting the reply. String-aware, so a brace inside a quoted
    value does not throw off the depth count.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start != -1 and (end := _balanced_end(s, start, opener, closer)) is not None:
            return s[start:end + 1]
    return s


def _balanced_end(s: str, start: int, opener: str, closer: str) -> int | None:
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
    return None


class LLMClient:
    """Create once; it holds a connection pool."""

    def __init__(self, nvidia_api_key: str | None = None) -> None:
        # The override exists so tests can inject a key.
        self.nvidia_api_key = config.NVIDIA_API_KEY if nvidia_api_key is None else nvidia_api_key
        self._client: httpx.AsyncClient | None = None
        self._limiters = get_limiters()
        self._ledger = get_ledger()
        # Models that rejected an optional parameter; they are called without it from then on.
        self._no_optional_params: set[str] = set()

    # ── lifecycle ────────────────────────────────────────────────────────────────────
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.LLM_TIMEOUT_S, connect=15.0),
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _endpoint(self, provider: str) -> tuple[str, dict]:
        """Base URL and auth headers for a provider."""
        if provider == "openrouter":
            return config.OPENROUTER_BASE_URL, {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "HTTP-Referer": config.OPENROUTER_APP_URL,
                "X-Title": config.OPENROUTER_APP_NAME,
            }
        return config.NIM_BASE_URL, {"Authorization": f"Bearer {self.nvidia_api_key}"}

    def _payload(self, spec: config.ModelSpec, messages: Sequence[dict],
                 temperature: float | None, max_tokens: int | None, stream: bool,
                 thinking: bool | None) -> dict:
        payload = {
            "model": spec.id,
            "messages": list(messages),
            "temperature": spec.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or spec.max_out,
            "stream": stream,
        }
        if stream and spec.provider == "openrouter":
            payload["stream_options"] = {"include_usage": True}   # final frame carries the cost
        # Reasoning off unless asked for: each provider has its own switch for it.
        if not (spec.thinking if thinking is None else thinking) \
                and spec.key not in self._no_optional_params:
            if spec.provider == "openrouter":
                payload["reasoning"] = {"enabled": False}
            else:
                payload["chat_template_kwargs"] = {"thinking": False}
        return payload

    def _call(self, role: registry.ModelRole, stage: str, deadline, on_pause,
              session: str) -> CallContext:
        return CallContext(role, stage, deadline, on_pause, session, self._no_optional_params)

    async def _prepare(self, spec: config.ModelSpec, call: CallContext) -> tuple[str, dict]:
        check_deadline(call.deadline, call.stage)
        await self._limiters.get(spec.key, spec.rpm).acquire(call.on_pause)
        self._ledger.note_provider_call(spec.provider, spec.id)
        base, headers = self._endpoint(spec.provider)
        return f"{base}/chat/completions", headers

    # ── one completion ───────────────────────────────────────────────────────────────
    async def chat(self, messages: Sequence[dict], *, role: registry.ModelRole = "fast",
                   temperature: float | None = None, max_tokens: int | None = None,
                   stage: str = "chat", deadline: float | None = None,
                   on_pause: PauseCallback | None = None, session: str = "",
                   thinking: bool | None = None) -> str:
        """One completion, retried through rate limits. Returns the assistant's text."""
        call = self._call(role, stage, deadline, on_pause, session)
        spec = registry.spec(role)
        for attempt in range(config.RETRY_MAX_ATTEMPTS):
            url, headers = await self._prepare(spec, call)
            started = time.monotonic()
            payload = self._payload(spec, messages, temperature, max_tokens, False, thinking)
            try:
                response = await self.client.post(url, headers=headers, json=payload)
            except _TRANSPORT_ERRORS as exc:
                self._ledger.record_error(spec.key, "transport", session, stage, str(exc))
                await backoff(attempt, stage, deadline)
                continue
            if response.status_code == 200:
                return self._completion_text(spec, response.json(), started, call)
            spec = await handle_failure(response, response.text, spec, call, attempt)
        raise LLMError(f"{stage}: gave up after {config.RETRY_MAX_ATTEMPTS} attempts")

    def _completion_text(self, spec: config.ModelSpec, data: dict, started: float,
                         call: CallContext) -> str:
        # A success clears a failure streak, so a model that failed under load earns its way back.
        registry.mark_available(spec.key)
        usage = data.get("usage") or {}
        self._ledger.record_call(spec.key, seconds=time.monotonic() - started,
                                 prompt_tokens=usage.get("prompt_tokens", 0),
                                 completion_tokens=usage.get("completion_tokens", 0),
                                 session=call.session, stage=call.stage,
                                 cost_usd=usage.get("cost") or 0.0)
        message = ((data.get("choices") or [{}])[0].get("message")) or {}
        # Some models put everything in `reasoning_content` when asked for a very short answer.
        return message.get("content") or message.get("reasoning_content") or ""

    async def chat_json(self, messages: Sequence[dict], model_cls, default, *,
                        role: registry.ModelRole = "fast", stage: str = "structured",
                        retries: int = 1, deadline: float | None = None,
                        on_pause: PauseCallback | None = None, session: str = "",
                        max_tokens: int | None = None):
        """Ask for JSON, validate it with Pydantic, and fall back to ``default`` if it never parses.

        Validating here rather than relying on a provider's ``response_format`` works on every
        model in the catalogue. A reply that does not parse is retried once with a blunter
        instruction. Provider failures (:class:`LLMError`) are raised for the caller to handle.
        """
        prompt = list(messages)
        for attempt in range(retries + 1):
            raw = await self.chat(prompt, role=role, stage=stage, deadline=deadline,
                                  on_pause=on_pause, session=session, temperature=0.0,
                                  max_tokens=max_tokens)
            try:
                return model_cls.model_validate_json(extract_json(raw))
            except ValueError as exc:
                if attempt >= retries:
                    log.warning("%s: falling back to its default after a parse failure: %s",
                                stage, str(exc)[:160])
                    return default
                log.debug("%s: structured parse failed (%s), retrying", stage, str(exc)[:120])
                prompt = [*messages, {
                    "role": "user",
                    "content": "Your last reply was not valid JSON. Reply with ONLY the JSON "
                               "object described above — no prose, no markdown fences.",
                }]
        return default

    # ── streaming ────────────────────────────────────────────────────────────────────
    async def chat_stream(self, messages: Sequence[dict], *, role: registry.ModelRole = "writer",
                          temperature: float | None = None, max_tokens: int | None = None,
                          stage: str = "write", deadline: float | None = None,
                          on_pause: PauseCallback | None = None, session: str = "",
                          thinking: bool | None = None,
                          on_reasoning: Callable[[str], None] | None = None,
                          on_finish: Callable[[str], None] | None = None) -> AsyncIterator[str]:
        """Yield text deltas; ``on_finish`` receives ``stop``, ``length`` or ``interrupted``.

        Retries cover establishing the stream. Once text has gone out it cannot be rewound, so a
        failure mid-answer ends the stream and the caller keeps what arrived. Retrying then would
        re-yield the answer from its start, which a reader sees as it beginning again.
        """
        call = self._call(role, stage, deadline, on_pause, session)
        spec = registry.spec(role)
        for attempt in range(config.RETRY_MAX_ATTEMPTS):
            url, headers = await self._prepare(spec, call)
            payload = self._payload(spec, messages, temperature, max_tokens, True, thinking)
            outcome, started = StreamOutcome(), time.monotonic()
            try:
                async with self.client.stream("POST", url, headers=headers,
                                              json=payload) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "replace")
                        spec = await handle_failure(response, body, spec, call, attempt)
                        continue
                    async for text in read_stream(response, outcome, on_reasoning):
                        yield text
            except _TRANSPORT_ERRORS as exc:
                if await self._retry_after_transport_error(spec, call, attempt, outcome, exc):
                    continue
            if outcome.error and not outcome.emitted:
                # Nothing reached the reader, so this is an ordinary failure: try another model.
                spec = await self._after_empty_stream_error(spec, call, attempt, outcome.error)
                continue
            self._finish_stream(spec, call, outcome, started, on_finish)
            return
        raise LLMError(f"{stage}: could not open a stream after {config.RETRY_MAX_ATTEMPTS} "
                       f"attempts")

    async def _retry_after_transport_error(self, spec: config.ModelSpec, call: CallContext,
                                           attempt: int, outcome: StreamOutcome,
                                           exc: Exception) -> bool:
        """True to retry. Once text has been sent the stream is over; it is marked as cut."""
        if not outcome.emitted:
            self._ledger.record_error(spec.key, "transport", call.session, call.stage, str(exc))
            await backoff(attempt, call.stage, call.deadline)
            return True
        self._ledger.record_error(spec.key, "transport_midstream", call.session, call.stage,
                                  str(exc))
        outcome.error = outcome.error or "connection dropped"
        return False

    async def _after_empty_stream_error(self, spec: config.ModelSpec, call: CallContext,
                                        attempt: int, error: str) -> config.ModelSpec:
        self._ledger.record_error(spec.key, "stream_error", call.session, call.stage, error)
        nxt = registry.mark_unavailable(spec.key, f"stream error: {error[:60]}", call.role)
        await backoff(attempt, call.stage, call.deadline)
        return nxt if nxt is not None else spec

    def _finish_stream(self, spec: config.ModelSpec, call: CallContext, outcome: StreamOutcome,
                       started: float, on_finish: Callable[[str], None] | None) -> None:
        registry.mark_available(spec.key)
        self._ledger.record_call(spec.key, seconds=time.monotonic() - started,
                                 completion_tokens=outcome.emitted // 4, session=call.session,
                                 stage=call.stage, cost_usd=outcome.cost_usd)
        if on_finish is not None:
            on_finish(outcome.ending)

    # ── diagnostics ──────────────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "providers": {p: config.provider_available(p) for p in config.PROVIDERS},
            "models": registry.snapshot(),
            "limiters": self._limiters.status(),
            "usage": self._ledger.snapshot(),
        }


_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
