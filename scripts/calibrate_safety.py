"""Measure the safety gate against a labelled set, and pick its semantic threshold.

The gate has two ways to be wrong and they are not symmetric:

* **A miss** — someone describing violence gets a statute lecture instead of 112.
* **A false alarm** — someone reading about the law gets a helpline card they did not need.
  Harmless once. Repeated, it is how people learn to ignore the card that matters.

So misses are weighted more heavily than false alarms, but false alarms are not free. This
sweeps the cosine threshold across the labelled set and reports the whole curve rather than a
single number, because the right cut is a judgement about that trade and should be made with
the curve visible.

    python scripts/calibrate_safety.py             # sweep and report
    python scripts/calibrate_safety.py --verbose   # name every case that is wrong
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import safety
from knowyourrights.retrieval.embedder import get_embedder
from knowyourrights.runtime.console import bold, rule, setup_console
from knowyourrights.safety_eval import DISCLOSURES, QUESTIONS

setup_console()

# A miss is this many times worse than a false alarm when scoring a threshold.
MISS_WEIGHT = 4.0


def pattern_tier(verbose: bool) -> tuple[int, int]:
    """(disclosures caught, false alarms) for the pattern tier alone."""
    caught = [d for d in DISCLOSURES if safety.check_patterns(d.text).urgent]
    false = [q for q in QUESTIONS if safety.check_patterns(q).urgent]
    print(f"  disclosures caught : {len(caught)}/{len(DISCLOSURES)}")
    print(f"  false alarms       : {len(false)}/{len(QUESTIONS)}")
    if verbose:
        for case in DISCLOSURES:
            if case not in caught:
                print(f"    missed: {case.text!r}")
    return len(caught), len(false)


async def meaning_scores(embedder) -> tuple[list[float], list[float]]:
    """Best exemplar similarity for every case the pattern tier left open.

    Questions the informational guard suppresses can never be false alarms whatever the cut,
    so only the ones that get past it are scored.
    """
    exemplars = await safety._exemplar_matrix(embedder)
    if exemplars is None:
        raise SystemExit("the embedding API did not answer; check OPENROUTER_API_KEY")
    _, matrix = exemplars

    async def best(text: str) -> float:
        return float((matrix @ safety._unit_rows(await embedder.encode_one(text))[0]).max())

    disclosures = [d.text for d in DISCLOSURES if not safety.check_patterns(d.text).urgent]
    questions = [q for q in QUESTIONS if not safety.check_patterns(q).urgent
                 and not safety.looks_informational(q)]
    return [await best(t) for t in disclosures], [await best(q) for q in questions]


def sweep(disclosure_scores: list[float], question_scores: list[float]) -> float:
    """Print the trade-off at every cut and return the cheapest, a miss weighted heavier."""
    print(f"  {'cut':>6}  {'caught':>10}  {'false':>8}  {'cost':>7}")
    best_cut, best_cost = safety.SEMANTIC_THRESHOLD, float("inf")
    for cut in [x / 100 for x in range(40, 91, 2)]:
        caught = sum(s >= cut for s in disclosure_scores)
        false = sum(s >= cut for s in question_scores)
        cost = MISS_WEIGHT * (len(disclosure_scores) - caught) + false
        if cost < best_cost:
            best_cut, best_cost = cut, cost
        print(f"  {cut:>6.2f}  {caught:>4}/{len(disclosure_scores):<5}  "
              f"{false:>4}/{len(question_scores):<3}  {cost:>7.1f}"
              f"{' <-' if cost == best_cost else ''}")
    return best_cut


async def main_async(verbose: bool) -> int:
    embedder = get_embedder()
    await embedder.warmup()
    rule("tier 1 — patterns only")
    t1_caught, t1_false = pattern_tier(verbose)
    rule("tier 2 — meaning, on the cases tier 1 left open")
    disclosure_scores, question_scores = await meaning_scores(embedder)
    rule("threshold sweep")
    best_cut = sweep(disclosure_scores, question_scores)

    rule("verdict")
    caught = t1_caught + sum(s >= best_cut for s in disclosure_scores)
    false = t1_false + sum(s >= best_cut for s in question_scores)
    print(f"  best cut (a miss weighted {MISS_WEIGHT:g}x): {bold(f'{best_cut:.2f}')}; "
          f"configured: {safety.SEMANTIC_THRESHOLD:.2f}")
    print(f"  both tiers at that cut: {caught}/{len(DISCLOSURES)} disclosures caught, "
          f"{false}/{len(QUESTIONS)} false alarms")
    if abs(best_cut - safety.SEMANTIC_THRESHOLD) > 0.005:
        print(f"\n  To adopt it, set SEMANTIC_THRESHOLD = {best_cut:.2f} "
              f"in knowyourrights/safety.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    return asyncio.run(main_async(ap.parse_args().verbose))


if __name__ == "__main__":
    raise SystemExit(main())
