"""Run the full agent from the terminal — the whole pipeline, no server, no browser.

    python scripts/try_agent.py                       # a scripted set of probes
    python scripts/try_agent.py "can police search my phone"
    python scripts/try_agent.py --depth deep "how do I file an RTI and what does it cost"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights.context.memory import Conversation                # noqa: E402
from knowyourrights.nim.ledger import get_ledger                      # noqa: E402
from knowyourrights.orchestrator import get_orchestrator              # noqa: E402
from knowyourrights.retrieval.search import get_engine                # noqa: E402
from knowyourrights.runtime.console import bold, dim, rule, setup_console  # noqa: E402
from knowyourrights.tools import crawl                                # noqa: E402

setup_console()

PROBES = [
    ("hi bro", None),
    ("what does Article 21 say", None),
    ("what are my rights if the police arrest me without a warrant?", None),
    ("police ne mujhe bina warrant ke arrest kar liya, kya yeh legal hai?", None),
    ("my landlord in Mumbai won't return my deposit", None),
]


async def run_one(orchestrator, conversation, message: str, depth: str | None,
                  state: str | None) -> None:
    print(f"\n{bold('USER')}  {message}")
    started = time.time()
    answer_started = False
    sources: list[str] = []

    async for event in orchestrator.stream(message, conversation, depth=depth, state=state):
        kind = event.type
        data = event.data
        if kind == "stage" and data["status"] == "running":
            print(dim(f"  · {data['label']}…"))
        elif kind == "stage" and data["status"] == "done" and data.get("detail"):
            print(dim(f"    {data['label']}: {data['detail']}"))
        elif kind == "plan":
            print(dim(f"  plan: {data['depth']} / {data['answer_kind']} / "
                      f"{len(data['sub_questions'])} sub-question(s) / "
                      f"lang={data.get('language')} / steps="
                      f"{[s['tool'] for s in data['steps']]}"))
        elif kind == "tool":
            if data["status"] == "done":
                print(dim(f"    {data['tool']}: {data['count']} result(s) "
                          f"in {data['elapsed_ms']} ms  ← {data['query'][:56]}"))
            elif data["status"] == "error":
                print(dim(f"    {data['tool']} FAILED: {data['detail']}"))
        elif kind == "source":
            sources.append(f"[{data['id']}] {data['tier_label']:<13} {data['title'][:70]}")
        elif kind == "procedure":
            print(dim(f"  procedure: {len(data.get('steps', []))} steps, "
                      f"fee={data.get('fees') or '-'}, timeline={data.get('timeline') or '-'}"))
        elif kind == "safety":
            print(f"  {bold('URGENT')} {data['text']}")
        elif kind == "notice":
            marker = "PAUSE" if data["level"] == "pause" else data["level"].upper()
            print(dim(f"  [{marker}] {data['text']}"))
        elif kind == "token":
            if not answer_started:
                print(f"\n{bold('ANSWER')}\n", end="")
                answer_started = True
            print(data["delta"], end="", flush=True)
        elif kind == "answer_revised":
            print(dim("\n  (answer revised: unresolvable citation markers removed)"))
        elif kind == "verdict":
            print(dim(f"\n  citations verified: {data['citations_verified']}, "
                      f"unsupported: {data['unsupported'] or 'none'}, {data['coverage']}"))
        elif kind == "usage":
            print(dim(f"  usage: {data['llm_calls']} LLM calls, {data['crawls']} crawls, "
                      f"{data['sources']} sources, {data['elapsed_s']}s"
                      f"{', THROTTLED' if data.get('throttled') else ''}"))
        elif kind == "error":
            print(f"\n  ERROR: {data['message']}")

    if sources:
        print(f"\n{bold('SOURCES')}")
        for line in sources:
            print(f"  {line}")
    print(dim(f"\n  wall clock {time.time() - started:.1f}s"))
    print("─" * 76)


async def main_async(args) -> int:
    rule("warmup")
    started = time.time()
    status = await get_engine().warmup()
    print(f"  models ready in {time.time() - started:.1f}s — "
          f"{status['embedder']['model']} / {status['reranker']['model']} "
          f"({status['reranker']['backend']})")
    print(f"  thresholds: low={status['reranker']['low_score']} "
          f"cite={status['reranker']['cite_min_score']} "
          f"[{status['reranker']['thresholds_source']}]")

    orchestrator = get_orchestrator()
    conversation = Conversation(session_id="cli")

    rule("turns")
    if args.messages:
        for message in args.messages:
            await run_one(orchestrator, conversation, message, args.depth, args.state)
    else:
        for message, depth in PROBES:
            await run_one(orchestrator, conversation, message, args.depth or depth, args.state)

    rule("session totals")
    print(get_ledger().report())
    await crawl.get_crawler().aclose()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("messages", nargs="*", help="questions to ask (default: a probe set)")
    ap.add_argument("--depth", choices=["quick", "standard", "deep"], help="force a depth")
    ap.add_argument("--state", help="the user's Indian state")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
