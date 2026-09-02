"""Measure the system properly, and write the numbers to a file.

Three layers, because they answer different questions and cost different amounts:

  --retrieval   (free)   quality and latency of search alone. GPU only, no API calls.
  --stages      (free)   where the milliseconds actually go inside one retrieval.
  --e2e         (costs)  full turns through the orchestrator, per depth. ~5 API calls each.

Everything lands in .runtime/benchmark.json so a report can be regenerated without re-running.

    python scripts/benchmark.py --retrieval --stages
    python scripts/benchmark.py --all --e2e-questions 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config, legal_terms                       # noqa: E402
from knowyourrights.eval_data import EXACT, GOLD, STRESS, offtopic_cases  # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()

OUT = config.RUNTIME_DIR / "benchmark.json"


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
    return ordered[k]


def summarise(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "min_ms": round(min(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "p90_ms": round(pct(values, 90), 1),
        "max_ms": round(max(values), 1),
        "mean_ms": round(statistics.fmean(values), 1),
    }


# ── corpus ────────────────────────────────────────────────────────────────────────────
def corpus_profile() -> dict:
    from knowyourrights.retrieval.store import get_store

    store = get_store()
    idx = store.index
    by_type = idx["source_type"].value_counts().to_dict()
    by_cat = idx["category"].value_counts().to_dict()
    by_status = idx["status"].value_counts().to_dict()

    years = idx["act_year"].dropna()
    state_acts = sum(
        1 for t in idx["act_title"].unique()
        if legal_terms.is_state_law(str(t), config.STATE_PREFIXES)
    )
    return {
        "chunks": int(store.count_rows()),
        "sections": int(len(idx)),
        "acts": int(idx["act_title"].nunique()),
        "resident_index_mb": round(idx.memory_usage(deep=True).sum() / 1e6, 1),
        "by_source_type": {str(k): int(v) for k, v in by_type.items()},
        "by_status": {str(k): int(v) for k, v in by_status.items()},
        "by_category": {str(k): int(v) for k, v in sorted(by_cat.items(), key=lambda p: -p[1])},
        "act_year_range": [int(years.min()), int(years.max())] if len(years) else None,
        "state_acts_incidentally_present": state_acts,
        "sections_with_marginal_heading": int(
            (idx["section_name"].fillna("").astype(str).str.strip() != "").sum()),
    }


# ── retrieval quality ─────────────────────────────────────────────────────────────────
async def retrieval_quality(engine, ks=(1, 3, 5, 10)) -> dict:
    hits_at = {k: 0 for k in ks}
    reciprocal = []
    latencies = []
    per_category: dict[str, list[bool]] = defaultdict(list)
    misses = []

    max_k = max(ks)
    for case in GOLD:
        expanded = legal_terms.expand(case.query)
        variants = [case.query] if expanded == case.query else [case.query, expanded]
        started = time.perf_counter()
        result = await engine.search(variants, top_k=max_k, rerank_with=case.query)
        latencies.append((time.perf_counter() - started) * 1000)

        citations = [h.citation for h in result.hits]
        rank = next((i + 1 for i, c in enumerate(citations) if case.matches(c)), None)
        for k in ks:
            if rank and rank <= k:
                hits_at[k] += 1
        reciprocal.append(1 / rank if rank else 0.0)
        per_category[case.category].append(bool(rank and rank <= 5))
        if not rank or rank > 5:
            misses.append({"query": case.query, "expected": list(case.accepted),
                           "rank": rank, "got": citations[:3]})

    n = len(GOLD)
    return {
        "n_questions": n,
        "recall_at": {str(k): round(hits_at[k] / n, 4) for k in ks},
        "recall_at_counts": {str(k): hits_at[k] for k in ks},
        "mrr": round(statistics.fmean(reciprocal), 4),
        "latency": summarise(latencies),
        "per_category_recall_at_5": {
            cat: {"n": len(v), "recall": round(sum(v) / len(v), 3)}
            for cat, v in sorted(per_category.items())
        },
        "misses_or_low_rank": misses,
    }


async def abstention_quality(engine) -> dict:
    """Does it decline what it cannot answer, and answer what it can?"""
    answerable_tops, offtopic_tops = [], []
    for case in GOLD:
        r = await engine.search([legal_terms.expand(case.query)], rerank_with=case.query)
        answerable_tops.append(r.top_score)
    for case in offtopic_cases():
        r = await engine.search([legal_terms.expand(case.query)], rerank_with=case.query)
        offtopic_tops.append(r.top_score)

    threshold = engine.reranker.thresholds
    false_abstain = sum(1 for s in answerable_tops if s < threshold.low)
    false_answer = sum(1 for s in offtopic_tops if s >= threshold.low)

    state_flagged = 0
    state_cases = [c for c in STRESS if c.expect_state]
    for case in state_cases:
        r = await engine.search([legal_terms.expand(case.query)], rerank_with=case.query)
        if any(h.is_state_law for h in r.hits):
            state_flagged += 1

    return {
        "threshold_low": threshold.low,
        "threshold_cite": threshold.cite,
        "threshold_source": threshold.source,
        "answerable": {"n": len(answerable_tops),
                       "top_score_median": round(statistics.median(answerable_tops), 4),
                       "top_score_min": round(min(answerable_tops), 4),
                       "wrongly_abstained": false_abstain},
        "offtopic": {"n": len(offtopic_tops),
                     "top_score_max": round(max(offtopic_tops), 4) if offtopic_tops else None,
                     "wrongly_answered": false_answer},
        "state_subject": {"n": len(state_cases), "flagged_as_state_law": state_flagged},
        "separation_margin": round(min(answerable_tops) - max(offtopic_tops), 4)
        if offtopic_tops else None,
    }


def exact_lookup_quality(engine) -> dict:
    rows = []
    for question, expect_label, expect_act in EXACT:
        refs = legal_terms.detect_section_refs(question)
        started = time.perf_counter()
        got, ok = "-", False
        if refs:
            hits = engine.lookup(refs[0].act, refs[0].label, refs[0].kind == "article")
            if hits:
                got = hits[0].citation
                ok = expect_label.lower() in got.lower() and expect_act.lower() in got.lower()
        rows.append({"question": question, "ok": ok, "got": got,
                     "ms": round((time.perf_counter() - started) * 1000, 1)})
    return {"n": len(rows), "correct": sum(1 for r in rows if r["ok"]), "cases": rows}


# ── stage-level latency ───────────────────────────────────────────────────────────────
async def stage_latency(engine, repeats: int = 5) -> dict:
    """Where the milliseconds go inside one retrieval."""
    from knowyourrights.retrieval.search import _rerank_document

    store, embedder, reranker = engine.store, engine.embedder, engine.reranker
    query = "what are my rights if the police arrest me without a warrant"
    timings: dict[str, list[float]] = defaultdict(list)
    loop = asyncio.get_running_loop()

    for i in range(repeats):
        t = time.perf_counter()
        vec = await embedder.encode([f"{query} {i}"], use_cache=False)
        timings["embed_1_query"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        dense_ids = await loop.run_in_executor(None, store.dense, vec[0], config.FETCH_K, None)
        timings["dense_search"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        fts_ids = await loop.run_in_executor(None, store.fts, query, config.FETCH_K, None)
        timings["bm25_search"].append((time.perf_counter() - t) * 1000)

        ids = list(dict.fromkeys(dense_ids + fts_ids))[:config.RERANK_POOL]
        t = time.perf_counter()
        rows = await loop.run_in_executor(None, store.rows, ids)
        timings["fetch_rows"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        await loop.run_in_executor(None, store.vectors, ids)
        timings["fetch_stored_vectors"].append((time.perf_counter() - t) * 1000)

        docs = [_rerank_document(r) for r in rows.itertuples()]
        t = time.perf_counter()
        await reranker.score(query, docs)
        timings[f"rerank_{len(docs)}_docs"].append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    await engine.search([query], rerank_with=query)
    whole = (time.perf_counter() - t) * 1000

    return {"stages": {k: summarise(v) for k, v in timings.items()},
            "whole_search_ms": round(whole, 1),
            "fetch_k": config.FETCH_K, "rerank_pool": config.RERANK_POOL}


# ── end to end ────────────────────────────────────────────────────────────────────────
E2E_QUESTIONS = [
    ("quick", "what does Article 21 say"),
    ("quick", "what is the punishment for cheating"),
    ("standard", "what are my rights if the police arrest me without a warrant?"),
    ("standard", "my employer has not paid my salary for two months"),
    ("standard", "police ne bina warrant arrest kar liya, kya yeh legal hai?"),
    ("deep", "how do I file an RTI application, what does it cost, and what is the deadline?"),
]


async def end_to_end(limit: int) -> dict:
    from knowyourrights.context.memory import Conversation
    from knowyourrights.llm.ledger import get_ledger
    from knowyourrights.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    ledger = get_ledger()
    runs = []

    for depth, question in E2E_QUESTIONS[:limit]:
        conversation = Conversation(session_id=f"bench-{len(runs)}")
        calls_before = ledger.total_calls
        started = time.perf_counter()
        first_token_at = None
        marks: dict[str, float] = {}
        answer_chars = 0
        sources = 0
        verified = 0
        unsupported = 0
        tools: dict[str, int] = defaultdict(int)

        async for event in orchestrator.stream(question, conversation, depth=depth):
            now = (time.perf_counter() - started) * 1000
            if event.type == "stage" and event.data["status"] == "done":
                marks[f"stage:{event.data['id']}"] = round(now, 1)
            elif event.type == "tool" and event.data["status"] == "done":
                tools[event.data["tool"]] += 1
            elif event.type == "sources_final":
                sources = len(event.data["sources"])
                marks["sources_ready"] = round(now, 1)
            elif event.type == "token":
                if first_token_at is None:
                    first_token_at = now
                answer_chars += len(event.data["delta"])
            elif event.type == "verdict":
                verified = event.data["citations_verified"]
                unsupported = len(event.data["unsupported"])

        total = (time.perf_counter() - started) * 1000
        runs.append({
            "depth": depth, "question": question,
            "total_ms": round(total, 1),
            "time_to_first_token_ms": round(first_token_at, 1) if first_token_at else None,
            "llm_calls": ledger.total_calls - calls_before,
            "sources": sources, "citations_verified": verified,
            "citations_unsupported": unsupported,
            "answer_chars": answer_chars,
            "tools": dict(tools),
            "milestones_ms": marks,
        })
        print(f"  {depth:<9} {round(total):>6} ms  ttft {round(first_token_at or 0):>5} ms  "
              f"{runs[-1]['llm_calls']} calls  {sources} sources  {verified} cited  "
              f"{question[:44]}")

    by_depth: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_depth[r["depth"]].append(r)

    return {
        "runs": runs,
        "by_depth": {
            d: {"n": len(v),
                "total_ms": summarise([x["total_ms"] for x in v]),
                "ttft_ms": summarise([x["time_to_first_token_ms"] for x in v
                                      if x["time_to_first_token_ms"]]),
                "mean_llm_calls": round(statistics.fmean([x["llm_calls"] for x in v]), 1)}
            for d, v in by_depth.items()
        },
        "totals": {
            "llm_calls": sum(r["llm_calls"] for r in runs),
            "citations_verified": sum(r["citations_verified"] for r in runs),
            "citations_unsupported": sum(r["citations_unsupported"] for r in runs),
        },
    }


# ── driver ────────────────────────────────────────────────────────────────────────────
async def main_async(args) -> int:
    from knowyourrights.retrieval.search import get_engine
    from knowyourrights.runtime import resources

    # Merge into any previous run rather than replacing it: the layers are meant to be run
    # separately (the free ones often, the credit-spending one rarely), and clobbering meant
    # a --e2e run silently erased the retrieval results.
    report: dict = {}
    if OUT.exists():
        try:
            report = json.loads(OUT.read_text(encoding="utf-8"))
        except ValueError:
            report = {}
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M")

    rule("environment")
    snapshot = resources.probe()
    plan = resources.get_plan()
    print(" ", snapshot.describe())
    print(f"  profile: {plan.name}  embedder {config.EMBED_MODEL} on {plan.embed_device}/"
          f"{plan.embed_dtype}  reranker {plan.rerank_model or 'NIM'} ({plan.rerank_backend})")
    report["environment"] = {
        "device": snapshot.device_name or "cpu",
        "vram_total_mb": snapshot.vram_total_mb,
        "ram_total_mb": snapshot.ram_total_mb,
        "cpu_physical": snapshot.cpu_physical,
        "profile": plan.name,
        "embedder": config.EMBED_MODEL,
        "embed_device": plan.embed_device,
        "embed_dtype": plan.embed_dtype,
        "reranker": plan.rerank_model,
        "rerank_backend": plan.rerank_backend,
    }
    report["config"] = {
        "fetch_k": config.FETCH_K, "top_k": config.TOP_K,
        "rerank_pool": config.RERANK_POOL, "rrf_k": config.RRF_K,
        "mmr_lambda": config.MMR_LAMBDA, "mmr_lambda_focused": config.MMR_LAMBDA_FOCUSED,
        "act_filter_weight": config.ACT_FILTER_WEIGHT,
        "general_code_weight": config.GENERAL_CODE_WEIGHT,
        "general_code_boost": config.GENERAL_CODE_BOOST,
        "fast_model": config.FAST_MODEL.id, "writer_model": config.WRITER_MODEL.id,
    }

    rule("warmup")
    t = time.perf_counter()
    engine = get_engine()
    await engine.warmup()
    warmup_s = time.perf_counter() - t
    print(f"  models ready in {warmup_s:.1f}s")
    report["warmup_seconds"] = round(warmup_s, 1)

    rule("corpus")
    report["corpus"] = corpus_profile()
    c = report["corpus"]
    print(f"  {c['chunks']:,} chunks · {c['sections']:,} sections · {c['acts']:,} acts · "
          f"{c['resident_index_mb']} MB resident index")
    print(f"  by type: {c['by_source_type']}")
    print(f"  with a marginal heading: {c['sections_with_marginal_heading']:,} "
          f"({c['sections_with_marginal_heading'] / c['sections']:.0%})")

    if args.retrieval or args.all:
        rule("retrieval quality")
        report["retrieval"] = await retrieval_quality(engine)
        r = report["retrieval"]
        for k, v in r["recall_at"].items():
            print(f"  Recall@{k:<3} {v:.1%}  ({r['recall_at_counts'][k]}/{r['n_questions']})")
        print(f"  MRR       {r['mrr']:.3f}")
        print(f"  latency   median {r['latency']['median_ms']} ms · "
              f"p90 {r['latency']['p90_ms']} ms")

        rule("abstention")
        report["abstention"] = await abstention_quality(engine)
        a = report["abstention"]
        print(f"  threshold {a['threshold_low']} [{a['threshold_source']}]")
        print(f"  answerable: {a['answerable']['wrongly_abstained']}/{a['answerable']['n']} "
              f"wrongly abstained")
        print(f"  off-topic : {a['offtopic']['wrongly_answered']}/{a['offtopic']['n']} "
              f"wrongly answered")
        print(f"  state law : {a['state_subject']['flagged_as_state_law']}/"
              f"{a['state_subject']['n']} flagged")

        rule("exact lookup")
        report["exact_lookup"] = exact_lookup_quality(engine)
        e = report["exact_lookup"]
        print(f"  {e['correct']}/{e['n']} correct · "
              f"median {statistics.median([c['ms'] for c in e['cases']]):.0f} ms")

    if args.stages or args.all:
        rule("stage latency")
        report["stage_latency"] = await stage_latency(engine)
        for name, s in report["stage_latency"]["stages"].items():
            print(f"  {name:<24} median {s['median_ms']:>7.1f} ms   p90 {s['p90_ms']:>7.1f} ms")
        print(f"  {'whole search':<24} {report['stage_latency']['whole_search_ms']:>13.1f} ms")

    if args.e2e:
        rule(f"end to end ({args.e2e_questions} questions — this spends API credits)")
        report["end_to_end"] = await end_to_end(args.e2e_questions)
        for depth, s in report["end_to_end"]["by_depth"].items():
            print(f"  {depth:<9} median {s['total_ms']['median_ms']:>7.0f} ms  "
                  f"ttft {s['ttft_ms'].get('median_ms', 0):>6.0f} ms  "
                  f"{s['mean_llm_calls']} calls")
        t = report["end_to_end"]["totals"]
        print(f"  citations: {t['citations_verified']} verified, "
              f"{t['citations_unsupported']} unsupported")

    config.ensure_runtime_dirs()
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rule("done")
    print(f"  wrote {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--e2e", action="store_true", help="full turns — spends API credits")
    ap.add_argument("--e2e-questions", type=int, default=len(E2E_QUESTIONS))
    ap.add_argument("--all", action="store_true", help="retrieval + stages (still free)")
    args = ap.parse_args()
    if not any([args.retrieval, args.stages, args.e2e, args.all]):
        args.all = True
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
