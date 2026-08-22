"""Exercise the retrieval stack directly — no agents, no LLM.

    python scripts/try_search.py                       # the built-in probe set
    python scripts/try_search.py "police arrest rights"
    python scripts/try_search.py --multi               # show multi-query fusion gains
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import legal_terms                              # noqa: E402
from knowyourrights.retrieval.search import get_engine              # noqa: E402
from knowyourrights.runtime.console import rule, setup_console      # noqa: E402

setup_console()

PROBES = [
    "can the police arrest me without telling me why",
    "how do I file an RTI request for information",
    "my consumer complaint against a defective product",
    "maternity leave entitlement at work",
    "what is the punishment for cheating",
    "protection from domestic violence",
]

STRESS = [
    "my landlord in Mumbai won't return my deposit",
    "rules for my housing society in Karnataka",
    "what is the best pizza recipe",
]


def show(result, show_text: bool = False) -> None:
    flags = []
    if result.mode != "hybrid":
        flags.append(result.mode)
    if result.ranked_by != "rerank":
        flags.append(f"ranked by {result.ranked_by}")
    if result.abstain:
        flags.append("ABSTAIN")
    suffix = f"  [{', '.join(flags)}]" if flags else ""
    print(f"  {result.candidates} candidates -> {len(result.hits)} hits in "
          f"{result.elapsed_ms} ms{suffix}")
    for note in result.notes:
        print(f"    note: {note}")
    for hit in result.hits:
        badges = []
        if hit.is_omitted:
            badges.append("OMITTED")
        if hit.is_state_law:
            badges.append(f"STATE:{hit.state}")
        badge = ("  " + " ".join(badges)) if badges else ""
        print(f"    {hit.score:.3f}  {hit.citation}{badge}")
        if show_text:
            print(f"           {hit.full_text[:160].replace(chr(10), ' ')}…")


async def main_async(queries: list[str], multi: bool, text: bool) -> int:
    engine = get_engine()

    rule("warmup")
    started = time.time()
    status = await engine.warmup()
    print(f"took {time.time() - started:.1f}s")
    print(f"  embedder : {status['embedder']['model']} loaded={status['embedder']['loaded']} "
          f"{status['embedder']['device']}/{status['embedder']['dtype']}")
    rr = status["reranker"]
    print(f"  reranker : {rr['model']} backend={rr['backend']} "
          f"low={rr['low_score']} cite={rr['cite_min_score']} ({rr['thresholds_source']})")
    print(f"  store    : {status['store'].get('chunks'):,} chunks / "
          f"{status['store'].get('sections'):,} sections / {status['store'].get('acts')} acts")
    for err in status["errors"]:
        print(f"  ERROR    : {err}")

    rule("exact lookup (no embedding, no rerank)")
    for act, sec, con in [("Constitution", "21", True), ("RTI Act", "6", False)]:
        started = time.time()
        hits = engine.lookup(act, sec, con)
        label = f"{act} {sec}"
        print(f"  {label:<22} {len(hits)} hit(s) in {(time.time() - started) * 1000:.0f} ms"
              + (f" -> {hits[0].citation}" if hits else ""))

    rule("queries")
    for query in queries:
        expanded = legal_terms.expand(query)
        print(f"\n\033[1m{query}\033[0m")
        if expanded != query:
            print(f"  expanded: {expanded}")
        started = time.time()
        variants = [query] if expanded == query else [query, expanded]
        result = await engine.search(variants, rerank_with=query)
        show(result, text)
        print(f"  total {(time.time() - started) * 1000:.0f} ms")

    if multi:
        rule("single vs multi-query fusion")
        probe = "police arrested me and did not tell me why"
        variants = [probe, "grounds of arrest must be communicated to the arrested person",
                    "rights of an arrested person under Indian criminal procedure"]
        single = await engine.search([probe])
        fused = await engine.search(variants)
        print("  single query:")
        show(single)
        print("  three queries fused:")
        show(fused)

    rule("stress — these should abstain or flag state law")
    for query in STRESS:
        result = await engine.search([legal_terms.expand(query)])
        marker = "ABSTAIN" if result.abstain else f"answers ({result.top_score:.3f})"
        print(f"  {query:<48} -> {marker}")
        for hit in result.hits[:2]:
            state = f"  [STATE: {hit.state}]" if hit.is_state_law else ""
            print(f"      {hit.score:.3f} {hit.citation}{state}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*", help="queries to run (default: a built-in probe set)")
    ap.add_argument("--multi", action="store_true", help="compare single vs fused multi-query")
    ap.add_argument("--text", action="store_true", help="print a snippet of each section")
    args = ap.parse_args()
    return asyncio.run(main_async(args.queries or PROBES, args.multi, args.text))


if __name__ == "__main__":
    raise SystemExit(main())
