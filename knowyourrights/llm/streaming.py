"""Reading an OpenAI-compatible chat stream.

Kept apart from the client because the rules are subtle and each one was learned the hard way:

* ``: OPENROUTER PROCESSING`` comment frames are keep-alives, not data.
* A failure after the HTTP 200 arrives as an ``error`` frame, not a status code. Skipping it made
  an aborted answer look finished.
* A stream that stops with neither ``[DONE]`` nor a finish reason was truncated.
* The final frame carries ``usage.cost``, which is how spend is measured.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx


@dataclass
class StreamOutcome:
    """What happened on one stream, filled in as it is read."""

    emitted: int = 0                 # characters of answer text yielded
    cost_usd: float = 0.0
    finish_reason: str | None = None
    saw_done: bool = False
    error: str = ""

    @property
    def ending(self) -> str:
        """``stop``, ``length`` or ``interrupted``, as the writer reports it to the reader."""
        if self.error:
            return "interrupted"
        if self.finish_reason == "length":
            return "length"
        if self.finish_reason or self.saw_done:
            return "stop"
        return "interrupted"


async def read_stream(response: httpx.Response, outcome: StreamOutcome,
                      on_reasoning: Callable[[str], None] | None = None) -> AsyncIterator[str]:
    """Yield answer text from a streaming response, recording how it ended in ``outcome``."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            outcome.saw_done = True
            return
        try:
            event = json.loads(chunk)
        except ValueError:
            continue
        if event.get("error"):
            outcome.error = str(event["error"])[:200]
            return
        if event.get("usage"):
            outcome.cost_usd = float(event["usage"].get("cost") or 0.0)
        for text in _deltas(event, outcome, on_reasoning):
            outcome.emitted += len(text)
            yield text


def _deltas(event: dict, outcome: StreamOutcome,
            on_reasoning: Callable[[str], None] | None) -> list[str]:
    texts: list[str] = []
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        thought = delta.get("reasoning_content")
        if thought and on_reasoning is not None:
            on_reasoning(thought)
        if delta.get("content"):
            texts.append(delta["content"])
        if choice.get("finish_reason"):
            outcome.finish_reason = choice["finish_reason"]
    return texts
