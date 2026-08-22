"""Derive the abstention and citation thresholds for whichever reranker is in use.

This is not optional bookkeeping. ``LOW_SCORE = 0.05`` and ``CITE_MIN_SCORE = 0.20`` were
tuned in the build notebook against a *different* cross-encoder. Score distributions are not
portable between rerankers — the NIM endpoint emits logits measured from −15.9 to +6.8, a
local head emits sigmoid probabilities — so reusing those numbers silently drops good
citations or admits bad ones.

Method: run the gold set (questions the corpus *can* answer) and the off-topic stress set
(questions it cannot), then pick the cut that best separates them.

    python scripts/calibrate.py            # measure and write .runtime/thresholds.json
    python scripts/calibrate.py --dry-run  # measure and print, change nothing
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config, legal_terms                        # noqa: E402
from knowyourrights.eval_data import GOLD, offtopic_cases             # noqa: E402
from knowyourrights.retrieval.reranker import save_thresholds         # noqa: E402
from knowyourrights.retrieval.search import get_engine                # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()


def separation_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float, float]:
    """Pick the cut that best separates the two score populations.

    Returns ``(threshold, accuracy)``. Ties break low, because wrongly abstaining costs the
    user an answer they could have had, while wrongly answering costs them a bad citation —
    and for a legal tool the bad citation is the worse failure, so we still prefer the lowest
    cut that achieves the best score rather than dipping under the noise.
    """
    candidates = sorted({round(s, 4) for s in answerable + unanswerable})
    if not candidates:
        return config.LOW_SCORE, 0.0
    best, best_acc = candidates[0], -1.0
    for cut in candidates:
        correct = sum(1 for s in answerable if s >= cut) + sum(1 for s in unanswerable if s < cut)
        acc = correct / max(1, len(answerable) + len(unanswerable))
        if acc > best_acc:
            best, best_acc = cut, acc
    return best, best_acc


async def main_async(dry_run: bool) -> int:
    engine = get_engine()

    rule("warmup")
    status = await engine.warmup()
    backend = status["reranker"]["backend"]
    model = status["reranker"]["model"]
    print(f"  reranker: {model} ({backend})")
    print(f"  current : low={status['reranker']['low_score']} "
          f"cite={status['reranker']['cite_min_score']} "
          f"[{status['reranker']['thresholds_source']}]")
    if backend == "none":
        print("\n  No reranker is available, so there is nothing to calibrate.")
        return 1

    rule("measuring")
    answerable: list[float] = []
    correct_hit_scores: list[float] = []
    for case in GOLD:
        expanded = legal_terms.expand(case.query)
        variants = [case.query] if expanded == case.query else [case.query, expanded]
        result = await engine.search(variants, rerank_with=case.query)
        if result.hits:
            answerable.append(result.hits[0].score)
            for hit in result.hits:
                if case.matches(hit.citation):
                    correct_hit_scores.append(hit.score)
    print(f"  {len(answerable)} answerable queries  "
          f"top-score min {min(answerable):.4f} / median "
          f"{sorted(answerable)[len(answerable) // 2]:.4f} / max {max(answerable):.4f}")

    offtopic = list(offtopic_cases())
    unanswerable: list[float] = []
    for case in offtopic:
        result = await engine.search([legal_terms.expand(case.query)], rerank_with=case.query)
        if result.hits:
            unanswerable.append(result.hits[0].score)
    if unanswerable:
        print(f"  {len(unanswerable)} off-topic queries   "
              f"top-score max {max(unanswerable):.4f}")
    else:
        print("  no off-topic scores captured; falling back to a percentile of the gold set")

    rule("chosen thresholds")
    # Use a low percentile rather than the minimum. A gold query whose *best* hit scores 0.03
    # is one where retrieval genuinely failed — abstaining there and searching the web is the
    # correct outcome, not a threshold bug. Letting that outlier set the cut would drag it
    # under the off-topic noise floor and disable abstention altogether.
    ordered_gold = sorted(answerable)
    floor_index = max(0, int(len(ordered_gold) * 0.10) - 1)
    gold_floor = ordered_gold[floor_index]
    ceiling = max(unanswerable) if unanswerable else 0.0
    print(f"  answerable floor (10th pct): {gold_floor:.4f}   off-topic ceiling: {ceiling:.4f}")

    if ceiling < gold_floor:
        # Sit in the gap, nearer the off-topic side so a weak but genuine answer survives.
        low = round(ceiling + (gold_floor - ceiling) * 0.35, 4)
        print(f"  cleanly separated — placing the cut inside the gap")
    else:
        low, accuracy = separation_threshold(answerable, unanswerable)
        print(f"  populations overlap; best separating cut is {accuracy:.0%} accurate")

    # The two thresholds answer different questions and must not be tied together.
    #   LOW_SCORE  — is the *best* hit good enough to answer at all, or should we abstain?
    #   CITE_MIN_SCORE — may this individual hit be shown to the grader?
    # The grader is the real gate, so the pre-filter should be generous. Clamping it up to the
    # abstention floor made "what is anticipatory bail" return nothing at all, even though the
    # correct provision (BNSS §483) was retrieved — its score simply sat below the floor.
    ordered = sorted(correct_hit_scores)
    percentile = ordered[max(0, int(len(ordered) * 0.05) - 1)] if ordered else low * 0.4
    cite = round(max(0.02, min(percentile, low * 0.6)), 4)

    print(f"  {bold('LOW_SCORE')}      = {low:.4f}   (below this: abstain and search the web)")
    print(f"  {bold('CITE_MIN_SCORE')} = {cite:.4f}   (below this: never reaches the grader)")
    would_abstain = sum(1 for s in answerable if s < low)
    print(f"\n  sanity: {would_abstain}/{len(answerable)} gold queries would now abstain "
          f"(want 0), {sum(1 for s in unanswerable if s >= low)}/{len(unanswerable)} "
          f"off-topic queries would answer (want 0)")
    kept = sum(1 for s in correct_hit_scores if s >= cite)
    print(f"          {kept}/{len(correct_hit_scores)} correct hits survive the citation floor")

    if dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    save_thresholds(backend, model, low, cite, extra={
        "gold_n": len(answerable), "offtopic_n": len(unanswerable),
        "gold_min_top": round(min(answerable), 4),
        "offtopic_max_top": round(max(unanswerable), 4) if unanswerable else None,
    })
    print(f"\n  written to {config.THRESHOLDS_FILE}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="measure without writing")
    args = ap.parse_args()
    return asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
