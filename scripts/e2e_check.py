"""Run real questions through the whole pipeline and report what a user would experience.

Not a benchmark of retrieval — that is ``benchmark.py``. This drives the actual orchestrator,
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

from knowyourrights import config                                     # noqa: E402
from knowyourrights.context.memory import Conversation                # noqa: E402
from knowyourrights.llm import retrieval_api                          # noqa: E402
from knowyourrights.orchestrator import get_orchestrator              # noqa: E402
from knowyourrights.retrieval.search import get_engine                # noqa: E402
from knowyourrights.runtime.console import bold, dim, rule, setup_console  # noqa: E402

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
                                     headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"})
        data = json.load(urllib.request.urlopen(req, timeout=30)).get("data") or {}
        return float(data.get("usage") or 0.0)
    except Exception:                                        # noqa: BLE001
        return None


async def run_turn(orch, conversation, message, depth, state) -> dict:
    t0 = time.perf_counter()
    first_token = None
    answer: list[str] = []
    stages: dict[str, float] = {}
    stage_started: dict[str, float] = {}
    notices, sources, safety, errors = [], [], None, []
    usage = {}

    async for ev in orch.stream(message, conversation, depth=depth, state=state):
        kind, data = ev.type, ev.data
        now = time.perf_counter()
        if kind == "stage":
            if data.get("status") == "running":
                stage_started[data["id"]] = now
            elif data.get("status") == "done" and data["id"] in stage_started:
                stages[data["id"]] = stages.get(data["id"], 0.0) + (now - stage_started[data["id"]])
        elif kind == "token":
            if first_token is None:
                first_token = now - t0
            answer.append(data.get("delta", ""))
        elif kind == "answer_revised":
            answer = [data.get("text", "")]
        elif kind in ("sources", "sources_final"):
            sources = data.get("sources", []) or sources
        elif kind == "notice":
            notices.append(f"[{data.get('level')}] {data.get('text')}")
        elif kind == "safety":
            safety = data.get("text") or "shown"
        elif kind == "error":
            errors.append(data.get("message"))
        elif kind == "usage":
            usage = data
    return {"message": message, "first_token": first_token, "total": time.perf_counter() - t0,
            "answer": "".join(answer), "stages": stages, "notices": notices,
            "sources": sources, "safety": safety, "errors": errors, "usage": usage}


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


async def main_async(args) -> int:
    get_engine()
    status = await get_engine().warmup()
    print(f"  embedder {status['embedder'].get('backend')} · reranker "
          f"{status['reranker'].get('backend')} {status['reranker'].get('model')}")

    if args.messages:
        cases = [(m, args.depth, args.state, "") for m in args.messages]
    else:
        cases = CASES + ([] if args.quick else [DEEP_CASE])

    before = account_usage()
    orch = get_orchestrator()
    conversation = Conversation(session_id="e2e-check")
    rows = []
    for message, depth, state, expect in cases:
        row = await run_turn(orch, conversation, message, depth, state)
        rows.append(row)
        report(row, expect)
    await asyncio.sleep(3)          # let a background summary settle before reading the bill
    after = account_usage()

    rule("summary")
    answered = [r for r in rows if r["first_token"] is not None]
    if answered:
        fts = sorted(r["first_token"] for r in answered)
        print(f"  first token  median {fts[len(fts)//2]:.1f}s   worst {fts[-1]:.1f}s")
        tots = sorted(r["total"] for r in rows)
        print(f"  total        median {tots[len(tots)//2]:.1f}s   worst {tots[-1]:.1f}s")
    print(f"  retrieval API {retrieval_api.session().stats()}")
    measured = sum(r["usage"].get("cost_usd", 0.0) for r in rows)
    print(f"  {bold('spent')} ${measured:.4f} across {len(rows)} turns "
          f"(${measured / max(1, len(rows)):.4f} per turn) — summed from each response's "
          f"own usage.cost")
    if before is not None and after is not None:
        print(f"  {dim(f'account endpoint says ${after - before:.4f}; it lags, so trust the line above')}")
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
