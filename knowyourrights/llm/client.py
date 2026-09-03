"""The NIM HTTP client.

Deliberately built on plain ``httpx`` rather than an SDK: we need the raw ``Retry-After``
header, the non-OpenAI reranking endpoint on a different host, and control over exactly when
a retry is worth attempting. The layers we would gain from an SDK are the ones we want to own.

Failure policy, in one place:

* **429** — honour ``Retry-After``, tell the caller how long the pause is, shrink the bucket,
  retry. Retrying is bounded by the turn's *deadline*, not by an attempt count, so a busy
  provider costs an answer some depth rather than the whole turn.
* **404 / unknown model** — retire that id and immediately try the role's next alternate.
* **5xx / timeouts** — exponential backoff with jitter, same deadline rule.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import AsyncIterator, Callable, Sequence

import httpx

from .. import config
from . import registry
from .ledger import get_ledger
from .limiter import PauseCallback, get_limiters

log = logging.getLogger(__name__)


class NimError(RuntimeError):
    """Any unrecoverable NIM failure."""


class NimDeadlineExceeded(NimError):
    """The turn ran out of time while waiting for the provider."""


class NimUnavailable(NimError):
    """No candidate model for the role could be reached."""


def extract_json(text: str) -> str:
    """Pull the first balanced JSON object or array out of a model reply.

    Models wrap JSON in prose or ``` fences often enough that scanning for the first balanced
    structure is more reliable than trusting the reply to be clean. String-aware, so a brace
    inside a quoted value doesn't throw off the depth count.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s


def _retry_after(response: httpx.Response) -> float | None:
    for header in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset"):
        raw = response.headers.get(header)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            return min(value, config.RETRY_MAX_DELAY * 2)
    return None


def _is_unknown_model(response: httpx.Response, body: str) -> bool:
    # 410 Gone is how the catalogue reports a *retired* model — which is exactly the case the
    # alternates list exists for. (Observed live: llama-3.2-nv-rerankqa-1b-v2 now returns 410.)
    if response.status_code in (404, 410):
        return True
    if response.status_code in (400, 422):
        lowered = body.lower()
        return "model" in lowered and ("not found" in lowered or "does not exist" in lowered
                                       or "invalid" in lowered or "unavailable" in lowered
                                       or "deprecat" in lowered or "retired" in lowered)
    return False


class NimClient:
    """One shared client. Create it once; it holds a connection pool."""

    def __init__(self, api_key: str | None = None) -> None:
        # `api_key` overrides NVIDIA's key only; it exists so tests can inject one.
        self.api_key = api_key if api_key is not None else config.NVIDIA_API_KEY
        self._client: httpx.AsyncClient | None = None
        self._limiters = get_limiters()
        self._ledger = get_ledger()
        self._rerank_url_cache: dict[str, str] = {}

    def _endpoint(self, provider: str) -> tuple[str, dict]:
        """Base URL and headers for a provider. Both speak OpenAI chat-completions."""
        if provider == "openrouter":
            return config.OPENROUTER_BASE_URL, {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                # OpenRouter uses these for attribution on free models.
                "HTTP-Referer": config.OPENROUTER_APP_URL,
                "X-Title": config.OPENROUTER_APP_NAME,
            }
        return config.NIM_BASE_URL, {"Authorization": f"Bearer {self.api_key}"}

    # ── lifecycle ────────────────────────────────────────────────────────────────────
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.NIM_TIMEOUT_S, connect=15.0),
                # Auth is per-provider and added per request; only the common bits here.
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    # ── shared retry machinery ───────────────────────────────────────────────────────
    async def _await_slot(self, model: str, rpm: int, on_pause: PauseCallback | None) -> None:
        await self._limiters.get(model, rpm).acquire(on_pause)

    def _check_deadline(self, deadline: float | None, stage: str) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise NimDeadlineExceeded(f"{stage}: ran out of time waiting for the model")

    async def _handle_failure(
        self, response: httpx.Response, body: str, spec: config.ModelSpec,
        role: registry.ModelRole | None, stage: str, attempt: int,
        deadline: float | None, on_pause: PauseCallback | None, session: str,
    ) -> config.ModelSpec:
        """Decide what to do about a failed response. Returns the spec to use next."""
        model = spec.key
        bucket = self._limiters.get(model, spec.rpm)

        if response.status_code == 429:
            self._ledger.record_error(model, "rate_limit", session, stage)
            wait = bucket.penalize(_retry_after(response))
            if on_pause is not None:
                result = on_pause(model, wait, "rate limit")
                if asyncio.iscoroutine(result):
                    await result
            self._check_deadline(deadline, stage)
            if deadline is not None and time.monotonic() + wait > deadline:
                raise NimDeadlineExceeded(
                    f"{stage}: rate limited for {wait:.0f}s, longer than the turn's remaining time"
                )
            await asyncio.sleep(wait)
            return spec

        if role is not None and _is_unknown_model(response, body):
            self._ledger.record_error(model, "unknown_model", session, stage, body)
            nxt = registry.mark_unavailable(model, f"HTTP {response.status_code}")
            if not nxt:
                raise NimUnavailable(f"{stage}: no usable model for role {role!r}")
            return nxt

        # OpenRouter answers 402 when the free allowance is spent, and 403 when a model is
        # gated. Neither is retryable on that model, but the other provider may still work.
        if role is not None and response.status_code in (402, 403):
            self._ledger.record_error(model, "quota", session, stage, body)
            nxt = registry.mark_unavailable(model, f"HTTP {response.status_code} (quota/gated)")
            if not nxt:
                raise NimUnavailable(f"{stage}: no usable model for role {role!r}")
            return nxt

        if response.status_code >= 500 or response.status_code == 408:
            self._ledger.record_error(model, "server_error", session, stage, body)
            # One 5xx is a blip worth retrying. A second in a row means this provider is having
            # a bad time, and continuing to back off against it just burns the turn's deadline
            # — switch to the other provider instead. This is the resilience the second
            # provider exists to buy.
            if attempt >= 1 and role is not None:
                nxt = registry.mark_unavailable(model, f"HTTP {response.status_code} (repeated)")
                if nxt is not None and nxt.key != model:
                    log.warning("%s keeps failing (%s) — switching to %s",
                                model, response.status_code, nxt.key)
                    return nxt
            await self._backoff(attempt, stage, deadline)
            return spec

        # 401/403 and other client errors are configuration problems; retrying won't help.
        self._ledger.record_error(model, "error", session, stage, body)
        # Name the provider that actually answered. Reporting an OpenRouter 401 as "NIM returned
        # HTTP 401" sent a real debugging session after the wrong key entirely.
        who = spec.provider if spec is not None else "provider"
        hint = ""
        if response.status_code in (401, 403):
            env = "NVIDIA_API_KEY" if who == "nim" else "OPENROUTER_API_KEY"
            hint = f" — check {env} in .env; the key was rejected, not the model"
        raise NimError(
            f"{stage}: {who} returned HTTP {response.status_code}{hint}: {body[:300]}")

    async def _backoff(self, attempt: int, stage: str, deadline: float | None) -> None:
        delay = min(config.RETRY_MAX_DELAY,
                    config.RETRY_INITIAL_DELAY * (config.RETRY_MULTIPLIER ** attempt))
        delay += random.uniform(0, delay * 0.25)
        self._check_deadline(deadline, stage)
        if deadline is not None and time.monotonic() + delay > deadline:
            raise NimDeadlineExceeded(f"{stage}: out of time during backoff")
        await asyncio.sleep(delay)

    # ── chat ─────────────────────────────────────────────────────────────────────────
    def _chat_payload(self, spec: config.ModelSpec, messages: Sequence[dict],
                      temperature: float | None, max_tokens: int | None,
                      stream: bool, thinking: bool | None = None) -> dict:
        payload = {
            "model": spec.id,
            "messages": list(messages),
            "temperature": spec.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or spec.max_out,
            "stream": stream,
        }
        # Nemotron reasons by default and returns it in `reasoning_content`. Turning it off is
        # a ~6x cut in completion tokens on our short structured stages. NVIDIA accepts the
        # chat-template flag; OpenRouter has its own `reasoning` block and rejects unknown keys
        # on some upstreams, so each provider gets the form it understands.
        want_thinking = spec.thinking if thinking is None else thinking
        if not want_thinking:
            if spec.provider == "openrouter":
                payload["reasoning"] = {"enabled": False}
            else:
                payload["chat_template_kwargs"] = {"thinking": False}
        return payload

    async def chat(
        self,
        messages: Sequence[dict],
        *,
        role: registry.ModelRole = "fast",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str = "chat",
        deadline: float | None = None,
        on_pause: PauseCallback | None = None,
        session: str = "",
        thinking: bool | None = None,
    ) -> str:
        """One completion, retried through rate limits. Returns the assistant's text."""
        spec = registry.spec(role)

        for attempt in range(config.RETRY_MAX_ATTEMPTS):
            self._check_deadline(deadline, stage)
            model = spec.key
            base, headers = self._endpoint(spec.provider)
            await self._await_slot(model, spec.rpm, on_pause)
            self._ledger.note_provider_call(spec.provider)
            started = time.monotonic()
            try:
                response = await self.client.post(
                    f"{base}/chat/completions", headers=headers,
                    json=self._chat_payload(spec, messages, temperature,
                                            max_tokens, stream=False, thinking=thinking)
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self._ledger.record_error(model, "transport", session, stage, str(exc))
                await self._backoff(attempt, stage, deadline)
                continue

            if response.status_code == 200:
                # A success clears any failure streak, so a model that 410'd under load
                # earns its way back instead of staying sidelined.
                registry.mark_available(model)
                data = response.json()
                usage = data.get("usage") or {}
                self._ledger.record_call(
                    model, seconds=time.monotonic() - started,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    session=session, stage=stage,
                )
                choices = data.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                content = message.get("content") or ""
                # Some models put everything in `reasoning_content` when asked for a very
                # short answer; fall back to it rather than returning an empty string.
                return content or (message.get("reasoning_content") or "")

            spec = await self._handle_failure(response, response.text, spec, role, stage,
                                              attempt, deadline, on_pause, session)

        raise NimError(f"{stage}: gave up after {config.RETRY_MAX_ATTEMPTS} attempts")

    async def chat_json(
        self,
        messages: Sequence[dict],
        model_cls,
        default,
        *,
        role: registry.ModelRole = "fast",
        stage: str = "structured",
        retries: int = 1,
        deadline: float | None = None,
        on_pause: PauseCallback | None = None,
        session: str = "",
        max_tokens: int | None = None,
    ):
        """Ask for JSON and validate it with Pydantic, degrading to ``default``.

        Validating ourselves rather than relying on a provider-specific ``response_format``
        keeps this working across every model in the catalogue — including ones that don't
        implement strict schema mode. A failed parse retries once with a blunter instruction,
        then returns the caller's safe default so one malformed reply can't fail a turn.
        """
        prompt = list(messages)
        for attempt in range(retries + 1):
            try:
                raw = await self.chat(prompt, role=role, stage=stage, deadline=deadline,
                                      on_pause=on_pause, session=session, temperature=0.0,
                                      max_tokens=max_tokens)
                return model_cls.model_validate_json(extract_json(raw))
            except (NimDeadlineExceeded, NimUnavailable):
                raise
            except Exception as exc:
                if attempt < retries:
                    log.debug("%s: structured parse failed (%s), retrying", stage, str(exc)[:120])
                    prompt = list(messages) + [{
                        "role": "user",
                        "content": "Your last reply was not valid JSON. Reply with ONLY the JSON "
                                   "object described above — no prose, no markdown fences.",
                    }]
                    continue
                log.warning("%s: falling back to default after parse failure: %s",
                            stage, str(exc)[:160])
                return default

    async def chat_stream(
        self,
        messages: Sequence[dict],
        *,
        role: registry.ModelRole = "writer",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str = "write",
        deadline: float | None = None,
        on_pause: PauseCallback | None = None,
        session: str = "",
        thinking: bool | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas.

        Retries cover *establishing* the stream. Once the first token is out we cannot rewind,
        so a mid-stream failure ends the stream and the caller keeps what arrived — which is
        still a partial answer rather than nothing.

        When reasoning is enabled the model emits ``reasoning_content`` deltas *before* any
        answer text. Those go to ``on_reasoning`` (so the UI can show a thinking stage instead
        of an apparently frozen screen) and are never mixed into the yielded answer.
        """
        spec = registry.spec(role)

        for attempt in range(config.RETRY_MAX_ATTEMPTS):
            self._check_deadline(deadline, stage)
            model = spec.key
            base, headers = self._endpoint(spec.provider)
            await self._await_slot(model, spec.rpm, on_pause)
            self._ledger.note_provider_call(spec.provider)
            started = time.monotonic()
            payload = self._chat_payload(spec, messages, temperature, max_tokens,
                                         stream=True, thinking=thinking)

            try:
                async with self.client.stream("POST", f"{base}/chat/completions",
                                              headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "replace")
                        spec = await self._handle_failure(response, body, spec, role, stage,
                                                          attempt, deadline, on_pause, session)
                        continue

                    emitted = 0
                    async for line in response.aiter_lines():
                        # OpenRouter sends ": OPENROUTER PROCESSING" comment frames as
                        # keep-alives; SSE comments start with ':' and carry no payload.
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            event = json.loads(chunk)
                        except ValueError:
                            continue
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            thought = delta.get("reasoning_content")
                            if thought and on_reasoning is not None:
                                on_reasoning(thought)
                            text = delta.get("content")
                            if text:
                                emitted += len(text)
                                yield text

                    registry.mark_available(model)
                    self._ledger.record_call(model, seconds=time.monotonic() - started,
                                             completion_tokens=emitted // 4,
                                             session=session, stage=stage)
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self._ledger.record_error(model, "transport", session, stage, str(exc))
                await self._backoff(attempt, stage, deadline)
                continue

        raise NimError(f"{stage}: could not open a stream after {config.RETRY_MAX_ATTEMPTS} attempts")

    # ── reranking (different host, non-OpenAI shape) ─────────────────────────────────
    def _rerank_urls(self, model_id: str) -> list[str]:
        """The catalogue is inconsistent about dots vs underscores in the path."""
        if model_id in self._rerank_url_cache:
            return [self._rerank_url_cache[model_id]]
        dotted = f"{config.NIM_RETRIEVAL_BASE}/{model_id}/reranking"
        underscored = f"{config.NIM_RETRIEVAL_BASE}/{model_id.replace('.', '_')}/reranking"
        return [dotted] if dotted == underscored else [dotted, underscored]

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        stage: str = "rerank",
        deadline: float | None = None,
        on_pause: PauseCallback | None = None,
        session: str = "",
    ) -> list[float]:
        """Relevance logits, one per passage, in the caller's order.

        Scores are raw logits (roughly -10..+5 depending on the model), *not* probabilities —
        callers must use thresholds calibrated for this backend, never the local reranker's.
        """
        if not passages:
            return []
        spec = registry.spec("rerank")
        model = spec.key

        for attempt in range(config.RETRY_MAX_ATTEMPTS):
            self._check_deadline(deadline, stage)
            await self._await_slot(model, spec.rpm, on_pause)
            # `model` is the provider-qualified key for bookkeeping; the wire needs the bare
            # id. Reranking is NVIDIA-only — OpenRouter exposes no reranking endpoint.
            payload = {
                "model": spec.id,
                "query": {"text": query[:4000]},
                "passages": [{"text": p[:8000]} for p in passages[:512]],
                "truncate": "END",
            }
            _, headers = self._endpoint("nim")
            started = time.monotonic()
            last_response: httpx.Response | None = None

            for url in self._rerank_urls(spec.id):
                try:
                    response = await self.client.post(url, headers=headers, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    self._ledger.record_error(model, "transport", session, stage, str(exc))
                    last_response = None
                    continue

                if response.status_code == 200:
                    registry.mark_available(model)
                    self._rerank_url_cache[spec.id] = url
                    self._ledger.record_call(model, seconds=time.monotonic() - started,
                                             session=session, stage=stage)
                    scores = [float("-inf")] * len(payload["passages"])
                    for item in response.json().get("rankings", []):
                        idx = int(item.get("index", -1))
                        if 0 <= idx < len(scores):
                            scores[idx] = float(item.get("logit", item.get("score", 0.0)))
                    # Pad if the caller sent more than the endpoint's 512-passage limit.
                    scores.extend([float("-inf")] * (len(passages) - len(scores)))
                    return scores

                last_response = response
                if response.status_code != 404:
                    break  # a real error, not just the wrong URL spelling

            if last_response is None:
                await self._backoff(attempt, stage, deadline)
                continue

            spec = await self._handle_failure(last_response, last_response.text, spec,
                                              "rerank", stage, attempt, deadline, on_pause, session)
            model = spec.key

        raise NimError(f"{stage}: reranking failed after {config.RETRY_MAX_ATTEMPTS} attempts")

    # ── diagnostics ──────────────────────────────────────────────────────────────────
    async def list_models(self, provider: str = "nim") -> list[str]:
        """Enumerate a provider's catalogue. Free on both, and spends no quota."""
        base, headers = self._endpoint(provider)
        response = await self.client.get(f"{base}/models", headers=headers)
        response.raise_for_status()
        return sorted(m["id"] for m in response.json().get("data", []))

    def status(self) -> dict:
        return {
            "providers": {p: config.provider_available(p) for p in config.PROVIDERS},
            "models": registry.snapshot(),
            "limiters": self._limiters.status(),
            "usage": self._ledger.snapshot(),
        }


_CLIENT: NimClient | None = None


def get_client() -> NimClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = NimClient()
    return _CLIENT
