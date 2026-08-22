"""Measure retrieval instead of guessing at it.

    python scripts/eval.py                     # Recall@5 / MRR over the gold set
    python scripts/eval.py --verbose           # show every miss with what came back instead
    python scripts/eval.py --compare           # A/B the ranking knobs against each other
    python scripts/eval.py --reranker BAAI/bge-reranker-v2-m3

Retrieval-only, so it costs GPU time but no NIM credits.
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

from knowyourrights import config, legal_terms                     # noqa: E402
from knowyourrights.eval_data import EXACT, GOLD, STRESS           # noqa: E402
from knowyourrights.retrieval import search as search_mod          # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()


@dataclass
class Outcome:
    query: str
    expect: str
    rank: int | None
    top: str
    score: float
    returned: list[str]
    elapsed_ms: int


async def run_gold(engine, verbose: bool) -> dict:
    outcomes: list[Outcome] = []
    for case in GOLD:
        expanded = legal_terms.expand(case.query)
        variants = [case.query] if expanded == case.query else [case.query, expanded]
        result = await engine.search(variants, rerank_with=case.query)
        citations = [h.citation for h in result.hits]
        rank = next((i + 1 for i, c in enumerate(citations) if case.matches(c)), None)
        outcomes.append(Outcome(case.query, " | ".join(case.accepted), rank,
                                citations[0] if citations else "-",
                                result.top_score, citations, result.elapsed_ms))

    hits = sum(1 for o in outcomes if o.rank)
    mrr = sum(1 / o.rank for o in outcomes if o.rank) / len(outcomes)
    at1 = sum(1 for o in outcomes if o.rank == 1)
    median_ms = sorted(o.elapsed_ms for o in outcomes)[len(outcomes) // 2]

    if verbose:
        misses = [o for o in outcomes if not o.rank]
        if misses:
            print(f"\n  {bold('misses')} ({len(misses)}):")
            for o in misses:
                print(f"    {o.query}")
                print(f"      wanted : {o.expect}")
                for c in o.returned[:5]:
                    print(f"      got    : {c}")
        weak = [o for o in outcomes if o.rank and o.rank > 2]
        if weak:
            print(f"\n  {bold('found but ranked low')} ({len(weak)}):")
            for o in weak:
                print(f"    rank {o.rank}  {o.query[:50]:<52} wanted {o.expect}")

    return {
        "n": len(outcomes), "recall": hits / len(outcomes), "hits": hits,
        "mrr": mrr, "at1": at1 / len(outcomes), "median_ms": median_ms,
        "outcomes": outcomes,
    }


async def run_stress(engine) -> dict:
    rows = []
    for case in STRESS:
        result = await engine.search([legal_terms.expand(case.query)], rerank_with=case.query)
        flagged_state = any(h.is_state_law for h in result.hits)
        rows.append({
            "query": case.query, "abstain": result.abstain, "top": result.top_score,
            "state": flagged_state, "expect_state": case.expect_state,
            "handled": result.abstain or (case.expect_state and flagged_state),
            "reason": case.reason,
        })
    return {"rows": rows, "handled": sum(1 for r in rows if r["handled"]), "n": len(rows)}


def run_exact(engine) -> dict:
    rows = []
    for question, expect_label, expect_act in EXACT:
        refs = legal_terms.detect_section_refs(question)
        ok = False
        got = "-"
        if refs:
            ref = refs[0]
            hits = engine.lookup(ref.act, ref.label, ref.kind == "article")
            if hits:
                got = hits[0].citation
                ok = expect_label.lower() in got.lower() and expect_act.lower() in got.lower()
        rows.append({"q": question, "ok": ok, "got": got,
                     "want": f"{expect_label}, {expect_act}"})
    return {"rows": rows, "ok": sum(1 for r in rows if r["ok"]), "n": len(rows)}


async def evaluate(engine, verbose: bool = False, label: str = "") -> dict:
    started = time.time()
    gold = await run_gold(engine, verbose)
    stress = await run_stress(engine)
    exact = run_exact(engine)
    return {"label": label, "gold": gold, "stress": stress, "exact": exact,
            "seconds": time.time() - started}


def print_report(report: dict) -> None:
    g, s, e = report["gold"], report["stress"], report["exact"]
    print(f"  Recall@{config.TOP_K} : {g['hits']}/{g['n']} = {g['recall']:.0%}"
          f"   MRR {g['mrr']:.3f}   top-1 {g['at1']:.0%}   median {g['median_ms']} ms")
    print(f"  exact lookup: {e['ok']}/{e['n']}")
    print(f"  stress      : {s['handled']}/{s['n']} handled (abstained or flagged state law)")
    for row in s["rows"]:
        mark = "ok  " if row["handled"] else "MISS"
        detail = "abstained" if row["abstain"] else f"answered at {row['top']:.3f}"
        if row["state"]:
            detail += ", state law flagged"
        print(f"     {mark} {row['query'][:46]:<48} {detail}")
    for row in e["rows"]:
        if not row["ok"]:
            print(f"     exact MISS: {row['q']} -> got {row['got']!r}, wanted {row['want']!r}")


async def compare(engine, verbose: bool) -> None:
    """A/B the ranking knobs so choices are made on evidence."""
    variants = [
        ("baseline (current config)", {}),
        ("no MMR (pure relevance)", {"MMR_LAMBDA": 1.0, "MMR_LAMBDA_FOCUSED": 1.0}),
        ("heavier MMR diversity", {"MMR_LAMBDA": 0.4, "MMR_LAMBDA_FOCUSED": 0.4}),
        ("no act-filter boost", {"ACT_FILTER_WEIGHT": 1.0}),
        ("stronger act-filter boost", {"ACT_FILTER_WEIGHT": 4.0}),
        ("wider candidate pool", {"FETCH_K": 40, "RERANK_POOL": 40}),
    ]
    results = []
    for label, overrides in variants:
        saved = {k: getattr(config, k) for k in overrides}
        for key, value in overrides.items():
            setattr(config, key, value)
        try:
            report = await evaluate(engine, verbose=False, label=label)
            results.append(report)
            g = report["gold"]
            print(f"  {label:<30} recall {g['recall']:.0%}  MRR {g['mrr']:.3f}  "
                  f"top-1 {g['at1']:.0%}  stress {report['stress']['handled']}/{report['stress']['n']}")
        finally:
            for key, value in saved.items():
                setattr(config, key, value)

    best = max(results, key=lambda r: (r["gold"]["mrr"], r["gold"]["recall"]))
    print(f"\n  best by MRR: {bold(best['label'])}")


async def main_async(args) -> int:
    engine = search_mod.get_engine()

    rule("warmup")
    status = await engine.warmup()
    rr = status["reranker"]
    print(f"  embedder {status['embedder']['model']} on {status['embedder']['device']}"
          f"/{status['embedder']['dtype']}")
    print(f"  reranker {rr['model']} ({rr['backend']}) thresholds low={rr['low_score']} "
          f"cite={rr['cite_min_score']} [{rr['thresholds_source']}]")
    print(f"  knobs    FETCH_K={config.FETCH_K} RERANK_POOL={config.RERANK_POOL} "
          f"TOP_K={config.TOP_K} MMR={config.MMR_LAMBDA}/{config.MMR_LAMBDA_FOCUSED} "
          f"ACT_WEIGHT={config.ACT_FILTER_WEIGHT}")

    if args.compare:
        rule("comparing ranking configurations")
        await compare(engine, args.verbose)
        return 0

    rule(f"gold set ({len(GOLD)} queries)")
    report = await evaluate(engine, args.verbose)
    print_report(report)
    print(f"\n  ran in {report['seconds']:.1f}s")

    if args.json:
        payload = {k: v for k, v in report["gold"].items() if k != "outcomes"}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0 if report["gold"]["recall"] >= 0.8 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true", help="show misses and weak ranks")
    ap.add_argument("--compare", action="store_true", help="A/B the ranking knobs")
    ap.add_argument("--reranker", help="override the local reranker model")
    ap.add_argument("--profile", help="force a resource profile")
    ap.add_argument("--json", help="write summary metrics to this path")
    args = ap.parse_args()

    import os
    if args.reranker:
        os.environ["KYR_RERANK_BALANCED"] = args.reranker
        os.environ["KYR_RERANK_QUALITY"] = args.reranker
        config.RERANK_BALANCED = args.reranker
        config.RERANK_QUALITY = args.reranker
        config.PROFILES = tuple(
            config.Profile(p.name, p.rerank_backend,
                           args.reranker if p.rerank_backend == "local" else p.rerank_model,
                           p.model_vram_mb, p.embed_batch, p.rerank_batch, p.note)
            for p in config.PROFILES
        )
    if args.profile:
        config.PROFILE_REQUEST = args.profile

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
