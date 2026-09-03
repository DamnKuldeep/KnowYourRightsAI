"""Turn a benchmark run into a report you can paste, commit, or compare.

The published numbers come from a laptop with a GPU. What a deployment actually delivers on
two shared vCPUs is a different question, and the honest answer is to measure it there rather
than extrapolate. This renders ``.runtime/benchmark.json`` as markdown, and given two files it
diffs them so the two machines sit side by side.

    python scripts/benchmark.py --all                    # on the box: produces the json
    python scripts/deploy_report.py                      # render it as markdown
    python scripts/deploy_report.py --out DEPLOYED.md    # write it to a file
    python scripts/deploy_report.py --baseline docs/benchmark-laptop.json   # compare

Accuracy should match the laptop almost exactly — the same two models rank the same corpus, so
only latency is expected to move. A recall difference is a signal that something is wrong with
the deployment, not that the hardware is slower.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                        # noqa: E402
from knowyourrights.runtime.console import setup_console  # noqa: E402

setup_console()


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"No benchmark at {path}. Run:  python scripts/benchmark.py --all")
    except ValueError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _ms(x: float | None) -> str:
    return "—" if x is None else (f"{x / 1000:.1f} s" if x >= 1000 else f"{x:.0f} ms")


def _delta(now: float | None, base: float | None, *, higher_is_better: bool) -> str:
    """A change worth reading, or a dash. Sub-1% moves are noise and are reported as such."""
    if now is None or base is None:
        return "—"
    if base == 0:
        return "—"
    change = (now - base) / abs(base)
    if abs(change) < 0.01:
        return "="
    better = (change > 0) == higher_is_better
    return f"{'+' if change > 0 else ''}{change * 100:.0f}% {'✓' if better else '✗'}"


def render(data: dict, baseline: dict | None) -> str:
    env, corpus = data.get("environment", {}), data.get("corpus", {})
    ret, abst = data.get("retrieval", {}), data.get("abstention", {})
    exact = data.get("exact_lookup", {})
    # benchmark.py nests the per-stage medians one level down; tolerate either shape so an
    # older or newer benchmark.json still renders instead of silently dropping the section.
    stage_blob = data.get("stage_latency", {})
    stages = stage_blob.get("stages", stage_blob)
    cfg = data.get("config", {})
    b_ret = (baseline or {}).get("retrieval", {})
    _b_blob = (baseline or {}).get("stage_latency", {})
    b_stages = _b_blob.get("stages", _b_blob)
    cmp_on = baseline is not None

    out: list[str] = []
    add = out.append

    add("# Measured on the deployment target\n")
    add(f"Generated {data.get('generated_at', '—')} · "
        f"profile `{env.get('profile', '?')}` · "
        f"rerank pool {cfg.get('rerank_pool', '?')}\n")

    add("## Machine\n")
    add("| | |")
    add("|---|---|")
    add(f"| Device | {env.get('device') or 'CPU only'} |")
    add(f"| CPU cores (physical) | {env.get('cpu_physical', '?')} |")
    add(f"| RAM | {env.get('ram_total_mb', 0) / 1024:.1f} GB |")
    add(f"| Embedder | `{env.get('embedder', '?')}` on {env.get('embed_device', '?')}"
        f"/{env.get('embed_dtype', '?')} |")
    add(f"| Reranker | `{env.get('reranker') or 'none'}` ({env.get('rerank_backend', '?')}) |")
    add(f"| Cold start | {data.get('warmup_seconds', '—')} s |")
    add(f"| Corpus | {corpus.get('chunks', 0):,} chunks · {corpus.get('sections', 0):,} sections "
        f"· {corpus.get('acts', 0):,} Acts |\n")

    recall, counts = ret.get("recall_at", {}), ret.get("recall_at_counts", {})
    b_recall = b_ret.get("recall_at", {})
    n = ret.get("n_questions", 0)

    add("## Retrieval accuracy\n")
    add(f"| Metric | This machine |{' Laptop (GPU) | Change |' if cmp_on else ''}")
    add(f"|---|---|{'---|---|' if cmp_on else ''}")
    for k in ("1", "3", "5", "10"):
        if k not in recall:
            continue
        row = f"| Recall@{k} | **{_pct(recall[k])}** ({counts.get(k, '?')}/{n}) |"
        if cmp_on:
            row += (f" {_pct(b_recall.get(k))} |"
                    f" {_delta(recall[k], b_recall.get(k), higher_is_better=True)} |")
        add(row)
    row = f"| MRR | **{ret.get('mrr', '—')}** |"
    if cmp_on:
        row += (f" {b_ret.get('mrr', '—')} |"
                f" {_delta(ret.get('mrr'), b_ret.get('mrr'), higher_is_better=True)} |")
    add(row)
    row = f"| Exact lookups | {exact.get('correct', '?')}/{exact.get('n', '?')} |"
    if cmp_on:
        row += " — | — |"
    add(row)
    add("")

    add("## Abstention — refusing what it cannot answer\n")
    off, ans = abst.get("offtopic", {}), abst.get("answerable", {})
    st = abst.get("state_subject", {})
    add("| | |")
    add("|---|---|")
    add(f"| Off-topic wrongly answered | **{off.get('wrongly_answered', '?')} / {off.get('n', '?')}** |")
    add(f"| Answerable wrongly refused | {ans.get('wrongly_abstained', '?')} / {ans.get('n', '?')} |")
    add(f"| State-subject flagged | {st.get('flagged_as_state_law', '?')} / {st.get('n', '?')} |")
    add(f"| Threshold | `{abst.get('threshold_low', '?')}` "
        f"({abst.get('threshold_source', 'unknown')}) |\n")

    if stages:
        add("## Where the time goes\n")
        add(f"| Stage | This machine |{' Laptop (GPU) | Change |' if cmp_on else ''}")
        add(f"|---|---|{'---|---|' if cmp_on else ''}")
        for name, value in stages.items():
            median = value.get("median_ms") if isinstance(value, dict) else value
            row = f"| {name.replace('_', ' ')} | {_ms(median)} |"
            if cmp_on:
                bv = b_stages.get(name)
                b_median = bv.get("median_ms") if isinstance(bv, dict) else bv
                row += f" {_ms(b_median)} | {_delta(median, b_median, higher_is_better=False)} |"
            add(row)
        add("")

    lat = ret.get("latency", {})
    if lat:
        add(f"Whole retrieval: median **{_ms(lat.get('median_ms'))}**, "
            f"p90 {_ms(lat.get('p90_ms'))}, max {_ms(lat.get('max_ms'))}.\n")

    if cmp_on:
        same = all(abs(recall.get(k, 0) - b_recall.get(k, 0)) < 1e-9 for k in recall if k in b_recall)
        add("> **Reading this comparison:** accuracy is expected to be identical — the same two "
            "models rank the same corpus, so hardware changes speed, not ranking. "
            + ("Accuracy matched, so only latency moved, which is the expected result."
               if same else
               "**Accuracy did not match.** That points at a deployment problem — an incomplete "
               "corpus, a different profile, or thresholds calibrated for another reranker — "
               "not at the hardware.") + "\n")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.RUNTIME_DIR / "benchmark.json")
    ap.add_argument("--baseline", type=Path, help="another benchmark.json to compare against")
    ap.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    args = ap.parse_args()

    report = render(_load(args.input), _load(args.baseline) if args.baseline else None)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
