"""Measure retrieval quality on the labelled sets, instead of guessing at it.

    python scripts/evaluate.py                  # Recall@5, MRR, abstention, exact lookups
    python scripts/evaluate.py --verbose        # show every miss and what came back instead
    python scripts/evaluate.py --compare        # A/B the ranking settings against each other
    python scripts/evaluate.py --degraded fusion    # as if the reranker were down
    python scripts/evaluate.py --degraded keywords  # as if embeddings were down too

Retrieval only: it calls the embedding and reranking APIs (about $0.05 a full run) and no chat
model. Needs OPENROUTER_API_KEY and the corpus (git lfs pull).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config, legal_terms
from knowyourrights.eval_data import EXACT, GOLD, STRESS
from knowyourrights.retrieval.search import get_engine
from knowyourrights.runtime.console import rule, setup_console

setup_console()


@dataclass
class Outcome:
    query: str
    expect: str
    rank: int | None
    returned: list[str]
    elapsed_ms: int


def degrade(engine, mode: str | None) -> None:
    """Simulate an API outage, to measure what readers get when one happens."""
    async def unavailable(*_args, **_kwargs):
        return None

    if mode in ("fusion", "keywords"):
        engine.reranker.score = unavailable
    if mode == "keywords":
        engine.embedder.encode = unavailable


async def run_gold(engine) -> list[Outcome]:
    outcomes = []
    for case in GOLD:
        result = await engine.search([case.query, legal_terms.expand(case.query)],
                                     rerank_with=case.query)
        citations = [h.citation for h in result.hits]
        rank = next((i + 1 for i, c in enumerate(citations) if case.matches(c)), None)
        outcomes.append(Outcome(case.query, " | ".join(case.accepted), rank, citations,
                                result.elapsed_ms))
    return outcomes


def gold_metrics(outcomes: list[Outcome]) -> dict:
    n = len(outcomes)
    found = [o for o in outcomes if o.rank]
    return {"n": n, "hits": len(found), "recall": len(found) / n,
            "mrr": sum(1 / o.rank for o in found) / n,
            "at1": sum(1 for o in found if o.rank == 1) / n,
            "median_ms": sorted(o.elapsed_ms for o in outcomes)[n // 2]}


async def run_stress(engine) -> list[dict]:
    """Questions the corpus cannot answer: they must abstain, or be flagged as state law."""
    rows = []
    for case in STRESS:
        result = await engine.search([legal_terms.expand(case.query)], rerank_with=case.query)
        flagged = any(h.is_territorial for h in result.hits)
        rows.append({"query": case.query, "abstain": result.abstain, "top": result.top_score,
                     "flagged": flagged,
                     "handled": result.abstain or (case.expect_state and flagged)})
    return rows


def run_exact(engine) -> list[dict]:
    rows = []
    for question, label, act in EXACT:
        refs = legal_terms.detect_section_refs(question)
        hits = engine.lookup(refs[0].act, refs[0].label, refs[0].kind == "article") if refs \
            else []
        got = hits[0].citation if hits else "-"
        rows.append({"q": question, "got": got, "want": f"{label}, {act}",
                     "ok": label.lower() in got.lower() and act.lower() in got.lower()})
    return rows


async def evaluate(engine) -> dict:
    started = time.time()
    outcomes = await run_gold(engine)
    return {"outcomes": outcomes, "gold": gold_metrics(outcomes),
            "stress": await run_stress(engine), "exact": run_exact(engine),
            "seconds": time.time() - started}


def print_report(report: dict, verbose: bool) -> None:
    g, stress, exact = report["gold"], report["stress"], report["exact"]
    print(f"  Recall@{config.TOP_K} : {g['hits']}/{g['n']} = {g['recall']:.0%}   MRR "
          f"{g['mrr']:.3f}   top-1 {g['at1']:.0%}   median {g['median_ms']} ms")
    print(f"  exact lookup: {sum(r['ok'] for r in exact)}/{len(exact)}")
    print(f"  stress      : {sum(r['handled'] for r in stress)}/{len(stress)} handled "
          f"(abstained, or flagged as another state's law)")
    for row in stress:
        detail = "abstained" if row["abstain"] else f"answered at {row['top']:.3f}"
        print(f"     {'ok  ' if row['handled'] else 'MISS'} {row['query'][:46]:<48} {detail}")
    for row in exact:
        if not row["ok"]:
            print(f"     exact MISS: {row['q']} -> {row['got']!r}, wanted {row['want']!r}")
    if verbose:
        for o in report["outcomes"]:
            if not o.rank:
                print(f"\n  MISS {o.query}\n    wanted {o.expect}")
                for citation in o.returned[:5]:
                    print(f"    got    {citation}")


COMPARISONS = [
    ("baseline (current config)", {}),
    ("no MMR (pure relevance)", {"MMR_LAMBDA": 1.0, "MMR_LAMBDA_FOCUSED": 1.0}),
    ("heavier MMR diversity", {"MMR_LAMBDA": 0.4, "MMR_LAMBDA_FOCUSED": 0.4}),
    ("no act-filter boost", {"ACT_FILTER_WEIGHT": 1.0}),
    ("wider candidate pool", {"FETCH_K": 40, "RERANK_POOL": 40}),
]


async def compare(engine) -> None:
    for label, overrides in COMPARISONS:
        saved = {key: getattr(config, key) for key in overrides}
        for key, value in overrides.items():
            setattr(config, key, value)
        try:
            report = await evaluate(engine)
        finally:
            for key, value in saved.items():
                setattr(config, key, value)
        g = report["gold"]
        handled = sum(r["handled"] for r in report["stress"])
        print(f"  {label:<30} recall {g['recall']:.0%}  MRR {g['mrr']:.3f}  "
              f"top-1 {g['at1']:.0%}  stress {handled}/{len(report['stress'])}")


async def main_async(args) -> int:
    engine = get_engine()
    rule("warm-up")
    status = await engine.warmup()
    print(f"  embeddings {status['embedder']['model']}  reranker {status['reranker']['model']} "
          f"[{status['reranker']['thresholds_source']}]")
    degrade(engine, args.degraded)
    if args.degraded:
        print(f"  simulating an outage: {args.degraded}")
    if args.compare:
        rule("comparing ranking settings")
        await compare(engine)
        return 0
    rule(f"gold set ({len(GOLD)} questions)")
    report = await evaluate(engine)
    print_report(report, args.verbose)
    print(f"\n  ran in {report['seconds']:.1f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(report["gold"], indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0 if report["gold"]["recall"] >= 0.8 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", "-v", action="store_true", help="show every miss")
    ap.add_argument("--compare", action="store_true", help="A/B the ranking settings")
    ap.add_argument("--degraded", choices=["fusion", "keywords"],
                    help="measure a reduced mode, as if an API were down")
    ap.add_argument("--json", help="write the gold-set metrics to this path")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
