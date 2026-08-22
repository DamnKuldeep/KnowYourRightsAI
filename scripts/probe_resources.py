"""What can this machine run, right now?

Run this before anything else. It prints the probed machine, the profile that will be chosen,
and — with --load — actually loads the planned models and measures the real cost, so the
numbers in the plan stay honest rather than aspirational.

    python scripts/probe_resources.py
    python scripts/probe_resources.py --load        # load models and measure
    python scripts/probe_resources.py --oom-test    # exercise the OOM-halving path
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                          # noqa: E402
from knowyourrights.runtime import gpu, resources          # noqa: E402
from knowyourrights.runtime.console import rule, setup_console  # noqa: E402

setup_console()


def show_plan() -> resources.ResourcePlan:
    snap = resources.probe()
    rule("machine")
    print(snap.describe())

    rule("profile selection")
    plan = resources.select_profile(snap)
    print(plan.describe())

    rule("what each profile would need")
    for p in config.PROFILES:
        need = p.model_vram_mb + config.VRAM_RESERVE_MB
        fits = "fits" if (p.name == "cpu" or (snap.cuda_available and snap.vram_free_mb >= need)) else "too big"
        mark = "►" if p.name == plan.name else " "
        print(f" {mark} {p.name:<9} models {p.model_vram_mb:>5} MiB + reserve "
              f"{config.VRAM_RESERVE_MB} MiB = {need:>5} MiB   {fits}")
    return plan


def measure_load(plan: resources.ResourcePlan) -> None:
    """Load exactly what the plan says and report the true cost."""
    import torch
    import psutil

    proc = psutil.Process()
    def vram_used() -> int:
        if not torch.cuda.is_available():
            return 0
        free, total = torch.cuda.mem_get_info()
        return (total - free) // (1024 * 1024)

    rule("loading (measured)")
    base_vram, base_rss = vram_used(), proc.memory_info().rss // 1_000_000
    print(f"baseline           VRAM {base_vram:>5} MiB   RSS {base_rss:>5} MB")

    if not plan.ram_ok:
        print(f"\n  refusing to load: {plan.snapshot.ram_available_mb} MB RAM available, "
              f"{config.RAM_LOAD_HEADROOM_MB} MB needed for the load spike.")
        print("  (the server would start in FTS-only mode rather than risk the machine)")
        return

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    emb = SentenceTransformer(
        config.EMBED_MODEL, trust_remote_code=True, device=plan.embed_device,
        model_kwargs={"dtype": getattr(torch, plan.embed_dtype)},
    )
    emb.max_seq_length = config.EMBED_MAX_SEQ
    peak_rss = proc.memory_info().rss // 1_000_000
    print(f"{config.EMBED_MODEL:<18} VRAM {vram_used():>5} MiB   RSS {peak_rss:>5} MB   "
          f"load {time.time() - t0:.1f}s   (+{vram_used() - base_vram} MiB)")

    t0 = time.time()
    emb.encode(["warmup"], normalize_embeddings=True)
    warm = time.time() - t0
    t0 = time.time()
    emb.encode(["can the police arrest me without a warrant"], normalize_embeddings=True)
    print(f"  encode            first {warm:.2f}s (CUDA warmup)  then {time.time() - t0:.3f}s")

    if plan.rerank_backend == "local" and plan.rerank_model:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(plan.rerank_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            plan.rerank_model, dtype=getattr(torch, plan.rerank_dtype)
        ).to(plan.rerank_device).eval()
        load_s = time.time() - t0

        docs = ["a long statutory passage about arrest procedure and rights " * 30] * plan.rerank_batch
        t0 = time.time()
        with torch.inference_mode():
            enc = tok([["arrest without a warrant", d] for d in docs], padding=True,
                      truncation=True, max_length=510, return_tensors="pt")
            enc.pop("token_type_ids", None)
            enc = {k: v.to(plan.rerank_device) for k, v in enc.items()}
            torch.sigmoid(model(**enc).logits.view(-1)).float().cpu().numpy()
        print(f"{plan.rerank_model:<18} VRAM {vram_used():>5} MiB   RSS "
              f"{proc.memory_info().rss // 1_000_000:>5} MB   load {load_s:.1f}s   "
              f"rerank {plan.rerank_batch} docs {time.time() - t0:.3f}s")

    rule("result")
    free_now = plan.snapshot.vram_total_mb - vram_used()
    print(f"VRAM in use {vram_used()} MiB of {plan.snapshot.vram_total_mb} MiB — "
          f"{free_now} MiB free (reserve target {config.VRAM_RESERVE_MB} MiB)")
    print("headroom OK" if free_now >= config.VRAM_RESERVE_MB else
          "HEADROOM BREACHED — consider a lighter profile")


async def oom_test() -> None:
    """Force the halving path without needing a real OOM."""
    rule("OOM-halving path")
    executor = gpu.get_executor()
    attempts: list[int] = []

    def flaky(batch):
        attempts.append(len(batch))
        if len(batch) > 4:
            raise RuntimeError("CUDA error: out of memory")
        return [len(batch)] * len(batch)

    out = await executor.map_batches(flaky, list(range(16)), batch_size=16, label="oom-test")
    print(f"batch sizes attempted : {attempts}")
    print(f"items processed       : {len(out)} of 16")
    assert len(out) == 16, "map_batches lost items while recovering"

    def always(batch):
        raise RuntimeError("CUDA error: out of memory")

    def on_cpu(batch):
        return ["cpu"] * len(batch)

    out = await executor.map_batches(always, list(range(6)), batch_size=6,
                                     cpu_fn=on_cpu, label="oom-fallback")
    print(f"CPU fallback          : {len(out)} item(s) recovered -> {out[:3]}…")
    assert out == ["cpu"] * 6
    print("\nOOM recovery works: halves, then falls back, never raises to the caller.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", action="store_true", help="load the planned models and measure")
    ap.add_argument("--oom-test", action="store_true", help="exercise OOM-halving recovery")
    ap.add_argument("--profile", help="force a profile instead of auto-selecting")
    args = ap.parse_args()

    if args.profile:
        import os
        os.environ["KYR_PROFILE"] = args.profile

    plan = show_plan()
    if args.load:
        measure_load(plan)
    if args.oom_test:
        asyncio.run(oom_test())

    rule("data")
    for label, path in (("LanceDB", config.DB_PATH), ("parquet", config.PARQUET),
                        ("enrichment", config.ENRICH_CACHE)):
        print(f"  {label:<11} {'OK ' if path.exists() else 'MISSING'} {path}")
    print(f"  NVIDIA key  {'set' if config.NVIDIA_API_KEY else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
