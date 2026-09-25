"""Ask questions from the terminal: the whole pipeline, or retrieval alone. No server, no browser.

    python scripts/ask.py "can police search my phone"
    python scripts/ask.py --depth deep --state Kerala "how do I get my deposit back"
    python scripts/ask.py --search "right to information appeal"     # statute search only

Questions in one run share a conversation, so later ones can be follow-ups. A full answer costs
about a third of a cent; ``--search`` costs a tenth of that.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import legal_terms
from knowyourrights.context.memory import Conversation
from knowyourrights.orchestrator import get_orchestrator
from knowyourrights.retrieval.search import get_engine
from knowyourrights.runtime.console import bold, dim, rule, setup_console
from knowyourrights.tools import crawl

setup_console()


def show_event(kind: str, data: dict, state: dict) -> None:
    """Print one pipeline event: progress dimmed, the answer plain."""
    if kind == "stage" and data["status"] == "running":
        print(dim(f"  · {data['label']}…"))
    elif kind == "tool" and data["status"] in ("done", "error"):
        outcome = f"{data['count']} found in {data['elapsed_ms']} ms" \
            if data["status"] == "done" else f"failed: {data['detail']}"
        print(dim(f"    {data['tool']}: {outcome}  ← {data['query'][:56]}"))
    elif kind == "source":
        state["sources"][data["id"]] = f"[{data['id']}] {data['tier_label']:<13} {data['title']}"
    elif kind == "safety":
        print(f"  {bold('URGENT')} {data['text']}")
    elif kind == "notice":
        print(dim(f"  [{data['level'].upper()}] {data['text']}"))
    elif kind == "token":
        if not state["answering"]:
            print(f"\n{bold('ANSWER')}")
            state["answering"] = True
        print(data["delta"], end="", flush=True)
    elif kind == "usage":
        print(dim(f"\n\n  {data['llm_calls']} model calls, {data['crawls']} pages read, "
                  f"${data['cost_usd']:.4f}, {data['elapsed_s']}s"))
    elif kind == "error":
        print(f"\n  ERROR: {data['message']}")


async def ask(conversation: Conversation, message: str, depth: str | None,
              state_name: str | None) -> None:
    print(f"\n{bold('YOU')}  {message}")
    state = {"answering": False, "sources": {}}
    async for event in get_orchestrator().stream(message, conversation, depth=depth,
                                                 state=state_name):
        show_event(event.type, event.data, state)
    if state["sources"]:
        print(f"\n{bold('SOURCES')}")
        for line in state["sources"].values():
            print(f"  {line}")
    print("─" * 76)


async def search_only(engine, query: str) -> None:
    started = time.monotonic()
    result = await engine.search([query, legal_terms.expand(query)], rerank_with=query)
    print(f"\n{bold(query)}  ({result.mode}, ranked by {result.ranked_by}, "
          f"{int((time.monotonic() - started) * 1000)} ms"
          f"{', would abstain' if result.abstain else ''})")
    for note in result.degraded:
        print(dim(f"  note: {note}"))
    for hit in result.hits:
        scope = f"  [{hit.state} only]" if hit.state else ""
        print(f"  {hit.score:6.3f}  {hit.citation}{scope}")


async def main_async(args) -> int:
    rule("warm-up")
    status = await get_engine().warmup()
    print(f"  embeddings {status['embedder']['model']}, reranker "
          f"{status['reranker']['model']} [{status['reranker']['thresholds_source']}]")
    try:
        if args.search:
            for query in args.questions:
                await search_only(get_engine(), query)
            return 0
        conversation = Conversation(session_id="cli")
        for question in args.questions:
            await ask(conversation, question, args.depth, args.state)
        return 0
    finally:
        await crawl.get_crawler().aclose()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("questions", nargs="+", help="one or more questions, asked in order")
    ap.add_argument("--depth", choices=["quick", "standard", "deep"], help="force a depth")
    ap.add_argument("--state", default="", help="the state the reader is in")
    ap.add_argument("--search", action="store_true", help="statute search only, no model")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
