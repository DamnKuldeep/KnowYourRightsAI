"""Derive the abstention and citation thresholds for a ranking method.

Scores are not portable between rerankers, or between a reranker and fused ranking, so reusing
one method's thresholds for another silently drops good citations or admits bad ones. This runs
the gold set (questions the corpus can answer) and the off-topic set (questions it cannot) and
picks the cut that best separates them.

    python scripts/calibrate.py                   # the reranker in use -> .runtime/thresholds.json
    python scripts/calibrate.py --method fusion   # ranking without the reranker
    python scripts/calibrate.py --method keywords # without embeddings either (BM25 only)
    python scripts/calibrate.py --dry-run         # measure and print, change nothing
    python scripts/calibrate.py --ship            # also update the shipped calibration

Costs about $0.05 in retrieval calls. Needs OPENROUTER_API_KEY and the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config, legal_terms
from knowyourrights.eval_data import GOLD, offtopic_cases
from knowyourrights.retrieval.reranker import (
    FUSION_BM25_METHOD, FUSION_METHOD, rerank_method, save_thresholds,
)
from knowyourrights.retrieval.search import get_engine
from knowyourrights.runtime.console import bold, rule, setup_console

setup_console()


def separation_threshold(answerable: list[float], unanswerable: list[float]) -> float:
    """The lowest cut that best separates the two populations: a wrong citation is the worse
    failure for a legal tool, but abstaining needlessly costs an answer, so ties break low."""
    best, best_accuracy = config.LOW_SCORE, -1.0
    total = max(1, len(answerable) + len(unanswerable))
    for cut in sorted({round(s, 4) for s in answerable + unanswerable}):
        correct = sum(s >= cut for s in answerable) + sum(s < cut for s in unanswerable)
        if correct / total > best_accuracy:
            best, best_accuracy = cut, correct / total
    return best


async def measure(engine) -> tuple[list[float], list[float], list[float]]:
    """(best score per answerable question, scores of correct hits, best per off-topic one)."""
    answerable, correct = [], []
    for case in GOLD:
        result = await engine.search([case.query, legal_terms.expand(case.query)],
                                     rerank_with=case.query)
        if result.hits:
            answerable.append(result.hits[0].score)
            correct += [h.score for h in result.hits if case.matches(h.citation)]
    unanswerable = []
    for case in offtopic_cases():
        result = await engine.search([legal_terms.expand(case.query)], rerank_with=case.query)
        if result.hits:
            unanswerable.append(result.hits[0].score)
    return answerable, correct, unanswerable


def choose(answerable: list[float], correct: list[float],
           unanswerable: list[float]) -> tuple[float, float]:
    """``(low, cite)``: the abstention cut and the more generous citation floor."""
    # A low percentile, not the minimum: a gold question whose best hit scores near zero is one
    # where retrieval genuinely failed, and letting it set the cut would disable abstention.
    ordered = sorted(answerable)
    floor = ordered[max(0, int(len(ordered) * 0.10) - 1)]
    ceiling = max(unanswerable) if unanswerable else 0.0
    if ceiling < floor:
        low = round(ceiling + (floor - ceiling) * 0.35, 4)     # inside the gap, nearer off-topic
    else:
        low = separation_threshold(answerable, unanswerable)
    # The citation floor only pre-filters for the grader, so it is generous: clamping it up to
    # the abstention cut once hid the correct provision for "what is anticipatory bail".
    correct = sorted(correct)
    percentile = correct[max(0, int(len(correct) * 0.05) - 1)] if correct else low * 0.4
    return low, round(max(0.02, min(percentile, low * 0.6)), 4)


def report(low: float, cite: float, answerable, correct, unanswerable) -> None:
    print(f"  {bold('low')}  = {low:.4f}   below this the best hit is too weak: abstain")
    print(f"  {bold('cite')} = {cite:.4f}   below this a hit never reaches the grader")
    print(f"  gold questions that would abstain: {sum(s < low for s in answerable)}/"
          f"{len(answerable)} (want 0); off-topic that would answer: "
          f"{sum(s >= low for s in unanswerable)}/{len(unanswerable)} (want 0); correct hits "
          f"kept: {sum(s >= cite for s in correct)}/{len(correct)}")


def ship(method: str, low: float, cite: float, extra: dict) -> None:
    path = config.PACKAGED_THRESHOLDS
    data = json.loads(path.read_text(encoding="utf-8"))
    data[method] = {"low": round(low, 4), "cite": round(cite, 4), **extra}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  shipped calibration updated: {path}")


def select_method(engine, name: str) -> str:
    """Put the engine into the ranking mode being calibrated, and return that mode's key.

    ``fusion`` is ranking with the reranker down; ``keywords`` is with embeddings down too.
    Each is what readers get during that outage, so each needs its own abstention cut.
    """
    async def unavailable(*_args, **_kwargs):
        return None

    if name == "rerank":
        return rerank_method()
    engine.reranker.score = unavailable
    if name == "keywords":
        engine.embedder.encode = unavailable
        return FUSION_BM25_METHOD
    return FUSION_METHOD


async def main_async(args) -> int:
    engine = get_engine()
    await engine.warmup()
    method = select_method(engine, args.method)
    rule(f"measuring {method}")
    answerable, correct, unanswerable = await measure(engine)
    low, cite = choose(answerable, correct, unanswerable)
    report(low, cite, answerable, correct, unanswerable)
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    extra = {"calibrated_at": time.strftime("%Y-%m-%d"), "gold_n": len(answerable),
             "offtopic_n": len(unanswerable)}
    save_thresholds(method, low, cite, extra)
    print(f"\n  written to {config.THRESHOLDS_FILE}")
    if args.ship:
        ship(method, low, cite, extra)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", choices=["rerank", "fusion", "keywords"], default="rerank")
    ap.add_argument("--dry-run", action="store_true", help="measure without writing")
    ap.add_argument("--ship", action="store_true",
                    help="also write knowyourrights/thresholds.json (commit it)")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
