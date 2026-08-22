"""The streaming contract between the orchestrator and the UI.

Deliberately typed events rather than a bare token stream. A deep research turn can run for a
minute or more; without structure the user watches a blank screen and cannot tell the
difference between thinking, crawling, and being rate-limited. Every event here exists because
something in the pipeline is worth *seeing*:

* ``stage``/``tool``    — what is happening now, and how long it took
* ``source``            — populates the citations panel **before** the prose starts
* ``notice`` with ``resume_in_s`` — turns a rate-limit stall into a visible countdown
* ``verdict``           — whether the answer's citations actually check out
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        payload = {"type": self.type, "at": round(self.at, 3), **self.data}
        return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def stage(id: str, label: str, status: str = "running", detail: str = "", **extra) -> Event:
    return Event("stage", {"id": id, "label": label, "status": status,
                           "detail": detail, **extra})


def plan(depth: str, answer_kind: str, sub_questions: list[str],
         steps: list[dict], **extra) -> Event:
    return Event("plan", {"depth": depth, "answer_kind": answer_kind,
                          "sub_questions": sub_questions, "steps": steps, **extra})


def tool(name: str, query: str, status: str = "running", count: int = 0,
         elapsed_ms: int = 0, detail: str = "") -> Event:
    return Event("tool", {"tool": name, "query": query, "status": status,
                          "count": count, "elapsed_ms": elapsed_ms, "detail": detail})


def source(item) -> Event:
    return Event("source", item.to_public())


def sources_final(items) -> Event:
    """The definitive citable set, after packing re-assigns ids.

    Progressive ``source`` events show research happening, but ids are re-assigned once the
    packer decides what actually reaches the writer — and the writer may cite evidence
    recalled from earlier turns that was never streamed. Without this the UI can render a
    citation chip that points at nothing.
    """
    return Event("sources_final", {"sources": [item.to_public() for item in items]})


def procedure(data: dict) -> Event:
    return Event("procedure", data)


def notice(text: str, level: str = "info", resume_in_s: float | None = None, **extra) -> Event:
    payload: dict[str, Any] = {"level": level, "text": text, **extra}
    if resume_in_s is not None:
        payload["resume_in_s"] = round(resume_in_s, 1)
    return Event("notice", payload)


def safety(helplines: list[tuple[str, str]], text: str) -> Event:
    return Event("safety", {"text": text,
                            "helplines": [{"label": l, "number": n} for l, n in helplines]})


def token(delta: str) -> Event:
    return Event("token", {"delta": delta})


def reasoning(delta: str) -> Event:
    return Event("reasoning", {"delta": delta})


def verdict(citations_verified: int, unsupported: list[str], coverage: str = "",
            **extra) -> Event:
    return Event("verdict", {"citations_verified": citations_verified,
                             "unsupported": unsupported, "coverage": coverage, **extra})


def usage(**data) -> Event:
    return Event("usage", data)


def done(**data) -> Event:
    return Event("done", data)


def error(message: str, recoverable: bool = True) -> Event:
    return Event("error", {"message": message, "recoverable": recoverable})


def as_dict(event: Event) -> dict:
    return asdict(event)
