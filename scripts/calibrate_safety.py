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

from knowyourrights import safety                        # noqa: E402
from knowyourrights.retrieval.embedder import get_embedder  # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402
from knowyourrights.safety_eval import DISCLOSURES, QUESTIONS  # noqa: E402

setup_console()

# A miss is this many times worse than a false alarm when scoring a threshold.
MISS_WEIGHT = 4.0


async def main_async(verbose: bool) -> int:
    embedder = get_embedder()
    await embedder.warmup()

    rule("tier 1 — patterns only")
    t1_fire = [d for d in DISCLOSURES if safety.check_patterns(d.text).urgent]
    t1_false = [q for q in QUESTIONS if safety.check_patterns(q).urgent]
    print(f"  disclosures caught : {len(t1_fire)}/{len(DISCLOSURES)}")
    print(f"  false alarms       : {len(t1_false)}/{len(QUESTIONS)}")
    if verbose:
        for case in DISCLOSURES:
            if not safety.check_patterns(case.text).urgent:
                print(f"    missed: {case.text!r}")

    rule("tier 2 — scoring every case")
    kinds, _texts, matrix = await safety._exemplar_matrix(embedder)

    import numpy as np

    async def best_score(text: str) -> tuple[float, str]:
        vec = np.asarray(await embedder.encode_one(text), dtype="float32")
        vec = vec / max(float(np.linalg.norm(vec)), 1e-9)
        scores = matrix @ vec
        i = int(scores.argmax())
        return float(scores[i]), kinds[i]

    # Only cases tier 1 did not already decide are in play for the threshold.
    open_disclosures = [d for d in DISCLOSURES if not safety.check_patterns(d.text).urgent]
    open_questions = [q for q in QUESTIONS if not safety.check_patterns(q).urgent]

    dis_scores = [(await best_score(d.text))[0] for d in open_disclosures]
    # The informational guard suppresses tier 2 entirely for these, so a question it catches can
    # never be a false alarm no matter the threshold. Score only the ones that get through.
    guarded = [q for q in open_questions if not safety.looks_informational(q)]
    q_scores = [(await best_score(q))[0] for q in guarded]

    print(f"  disclosures still open after tier 1 : {len(open_disclosures)}")
    print(f"  questions not caught by the guard   : {len(guarded)}/{len(open_questions)}")
    if dis_scores:
        print(f"  disclosure similarity  min {min(dis_scores):.3f} / "
              f"median {sorted(dis_scores)[len(dis_scores) // 2]:.3f} / max {max(dis_scores):.3f}")
    if q_scores:
        print(f"  question similarity    max {max(q_scores):.3f}")

    rule("threshold sweep")
    print(f"  {'cut':>6}  {'caught':>10}  {'false':>8}  {'cost':>7}")
    best_cut, best_cost = safety.SEMANTIC_THRESHOLD, float("inf")
    for cut in [x / 100 for x in range(40, 91, 2)]:
        caught = sum(1 for s in dis_scores if s >= cut)
        false = sum(1 for s in q_scores if s >= cut)
        missed = len(dis_scores) - caught
        cost = MISS_WEIGHT * missed + false
        if cost < best_cost:
            best_cut, best_cost = cut, cost
        flag = " <-" if cost == best_cost else ""
        print(f"  {cut:>6.2f}  {caught:>4}/{len(dis_scores):<5}  {false:>4}/{len(q_scores):<3}"
              f"  {cost:>7.1f}{flag}")

    rule("verdict")
    total_caught = len(t1_fire) + sum(1 for s in dis_scores if s >= best_cut)
    total_false = len(t1_false) + sum(1 for s in q_scores if s >= best_cut)
    print(f"  best cut by cost (miss weighted {MISS_WEIGHT:g}x): {bold(f'{best_cut:.2f}')}")
    print(f"  currently configured                    : {safety.SEMANTIC_THRESHOLD:.2f}")
    print()
    print(f"  both tiers at that cut:")
    print(f"    disclosures caught : {total_caught}/{len(DISCLOSURES)} "
          f"({total_caught / len(DISCLOSURES):.0%})")
    print(f"    false alarms       : {total_false}/{len(QUESTIONS)} "
          f"({total_false / len(QUESTIONS):.0%})")
    if abs(best_cut - safety.SEMANTIC_THRESHOLD) > 0.005:
        print(f"\n  To adopt it, set SEMANTIC_THRESHOLD = {best_cut:.2f} in knowyourrights/safety.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    return asyncio.run(main_async(ap.parse_args().verbose))


if __name__ == "__main__":
    raise SystemExit(main())
