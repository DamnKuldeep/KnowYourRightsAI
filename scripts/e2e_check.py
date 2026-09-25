"""Run real questions through the whole pipeline and report what a user would experience.

Not a measure of retrieval — that is ``evaluate.py``. This drives the actual orchestrator,
turn by turn, over one conversation, and measures the things a person notices:

* **time to first token** — when the answer starts appearing, which is what "fast" feels like;
* **total time** and which stages spent it;
* **dollars**, read from OpenRouter's own account usage before and after, not estimated;
* the answer itself, the sources and their jurisdiction labels, and every notice shown.

The default set covers each path through the pipeline: small talk, an exact lookup, a standard
question, a follow-up that only makes sense with history, Hinglish, a jurisdiction trap, the
safety gate, and a deep procedure question.

    python scripts/e2e_check.py                   # the full set, ~2 minutes, ~3 US cents
    python scripts/e2e_check.py --quick           # skip the deep procedure question
    python scripts/e2e_check.py "your question"   # just one
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config
from knowyourrights.context.memory import Conversation
from knowyourrights.llm import retrieval_api
from knowyourrights.orchestrator import get_orchestrator
from knowyourrights.retrieval.search import get_engine
from knowyourrights.runtime.console import bold, dim, rule, setup_console

setup_console()

# (message, depth or None for auto, state or None, what to look for)
CASES = [
    ("hi, what can you help me with?", None, None, "concierge — no research"),
    ("what does Article 21 say", None, None, "exact lookup"),
    ("what are my rights if the police arrest me without a warrant?", None, None,
     "standard research, BNSS"),
    ("and how long can they keep me before a magistrate?", None, None,
     "follow-up — needs the previous turn"),
    ("Police ne bina warrant arrest kar liya, kya yeh legal hai?", None, None,
     "Hinglish in, Hinglish out"),
    ("my landlord in Mumbai won't return my deposit", None, "Kerala",
     "jurisdiction — must not present another state's law as governing"),
    ("my husband is hitting me", None, None, "safety card before research"),
]
DEEP_CASE = ("how do I file an RTI application, what does it cost, and how do I appeal?",
             "deep", None, "deep — reads the portal, procedure card")


def account_usage() -> float | None:
    """Total dollars this key has spent, straight from OpenRouter."""
    try:
        req = urllib.request.Request(f"{config.OPENROUTER_BASE_URL}/key",
                                     headers={"Authorization":
                                              f"Bearer {config.OPENROUTER_API_KEY}"})
        data = json.load(urllib.request.urlopen(req, timeout=30)).get("data") or {}
        return float(data.get("usage") or 0.0)
    except Exception:
        return None


class TurnRecord:
    """Everything a reader would notice about one turn, collected from its events."""

    def __init__(self, message: str) -> None:
        self.message, self.started = message, time.perf_counter()
        self.first_token: float | None = None
        self.answer: list[str] = []
        self.stages: dict[str, float] = {}
        self._stage_started: dict[str, float] = {}
        self.notices: list[str] = []
        self.sources: list = []
        self.errors: list[str] = []
        self.safety: str | None = None
        self.usage: dict = {}

    def add(self, kind: str, data: dict) -> None:
        now = time.perf_counter()
        if kind == "stage":
            self._stage(data, now)
        elif kind == "token":
            self.first_token = self.first_token or now - self.started
            self.answer.append(data.get("delta", ""))
        elif kind == "answer_revised":
            self.answer = [data.get("text", "")]
        elif kind == "sources_final":
            self.sources = data.get("sources", []) or self.sources
        elif kind == "notice":
            self.notices.append(f"[{data.get('level')}] {data.get('text')}")
        elif kind == "safety":
            self.safety = data.get("text") or "shown"
        elif kind == "error":
            self.errors.append(data.get("message"))
        elif kind == "usage":
            self.usage = data

    def _stage(self, data: dict, now: float) -> None:
        if data.get("status") == "running":
            self._stage_started[data["id"]] = now
        elif data.get("status") == "done" and data["id"] in self._stage_started:
            spent = now - self._stage_started[data["id"]]
            self.stages[data["id"]] = self.stages.get(data["id"], 0.0) + spent

    def result(self) -> dict:
        return {"message": self.message, "first_token": self.first_token,
                "total": time.perf_counter() - self.started, "answer": "".join(self.answer),
                "stages": self.stages, "notices": self.notices, "sources": self.sources,
                "safety": self.safety, "errors": self.errors, "usage": self.usage}


async def run_turn(orch, conversation, message, depth, state) -> dict:
    record = TurnRecord(message)
    async for event in orch.stream(message, conversation, depth=depth, state=state or ""):
        record.add(event.type, event.data)
    return record.result()


def report(row: dict, expect: str) -> None:
    rule(row["message"][:70])
    print(f"  {dim('expect:')} {expect}")
    ft = f"{row['first_token']:.1f}s" if row["first_token"] is not None else "—"
    print(f"  {bold('first token')} {ft}   {bold('total')} {row['total']:.1f}s   "
          f"llm calls {row['usage'].get('llm_calls', '?')}   "
          f"cost ${row['usage'].get('cost_usd', 0):.5f}")
    if row["stages"]:
        print("  stages: " + " · ".join(f"{k} {v:.1f}s" for k, v in row["stages"].items()))
    if row["safety"]:
        print(f"  {bold('SAFETY CARD:')} {row['safety']}")
    for n in row["notices"]:
        print(f"  notice {n[:150]}")
    for e in row["errors"]:
        print(f"  {bold('ERROR')} {e}")
    for s in row["sources"][:5]:
        juris = s.get("jurisdiction") or s.get("tier_label") or ""
        place = f" [{s['state']}]" if s.get("state") else ""
        print(f"    {s.get('id','?'):<4} {juris:<12}{place} {str(s.get('title',''))[:62]}")
    text = row["answer"].strip()
    print()
    for line in (text[:900] + (" …" if len(text) > 900 else "")).splitlines():
        print(f"  │ {line}")


def summarise(rows: list[dict], before: float | None, after: float | None) -> None:
    rule("summary")
    firsts = sorted(r["first_token"] for r in rows if r["first_token"] is not None)
    totals = sorted(r["total"] for r in rows)
    if firsts:
        print(f"  first token  median {firsts[len(firsts) // 2]:.1f}s   worst {firsts[-1]:.1f}s")
        print(f"  total        median {totals[len(totals) // 2]:.1f}s   worst {totals[-1]:.1f}s")
    print(f"  retrieval API {retrieval_api.session().stats()}")
    measured = sum(r["usage"].get("cost_usd", 0.0) for r in rows)
    print(f"  {bold('spent')} ${measured:.4f} across {len(rows)} turns "
          f"(${measured / max(1, len(rows)):.4f} per turn), summed from each response's own "
          f"usage.cost")
    if before is not None and after is not None:
        lag_note = f"account endpoint says ${after - before:.4f}; it lags, so trust the line above"
        print(f"  {dim(lag_note)}")


async def main_async(args) -> int:
    status = await get_engine().warmup()
    print(f"  embeddings {status['embedder']['model']} · reranker "
          f"{status['reranker']['model']} [{status['reranker']['thresholds_source']}]")
    cases = ([(m, args.depth, args.state, "") for m in args.messages] if args.messages
             else CASES + ([] if args.quick else [DEEP_CASE]))
    before = account_usage()
    conversation = Conversation(session_id="e2e-check")
    rows = []
    for message, depth, state, expect in cases:
        rows.append(await run_turn(get_orchestrator(), conversation, message, depth, state))
        report(rows[-1], expect)
    await asyncio.sleep(3)          # let a background summary settle before reading the bill
    summarise(rows, before, account_usage())
    return 1 if any(r["errors"] for r in rows) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("messages", nargs="*")
    ap.add_argument("--depth", choices=["quick", "standard", "deep"])
    ap.add_argument("--state")
    ap.add_argument("--quick", action="store_true", help="skip the deep procedure question")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
