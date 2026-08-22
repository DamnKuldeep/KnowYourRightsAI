"""Resolve which NIM models this API key can actually reach, once, and cache the answer.

Catalogues change under you. Rather than discover a retired model id in the middle of
answering someone's question, we probe each role's candidates here and pin the winner into
``.runtime/nim_probe.json``.

It also reports the reranker's **logit scale**, which matters: the local cross-encoder emits a
sigmoid score in [0,1] and NIM emits raw logits, so thresholds calibrated for one are
meaningless for the other.

    python scripts/probe_nim.py              # list models + probe every role (~5 calls)
    python scripts/probe_nim.py --list-only  # just enumerate the catalogue (free)
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                                    # noqa: E402
from knowyourrights.nim import registry                              # noqa: E402
from knowyourrights.nim.client import NimClient, NimError            # noqa: E402
from knowyourrights.runtime.console import rule, setup_console       # noqa: E402

setup_console()

INTERESTING = re.compile(r"bge|embed|rerank|nemotron|llama-3\.3", re.I)


async def probe_chat_role(client: NimClient, role: str) -> str | None:
    for candidate in registry.candidates(role):
        registry.pin(role, candidate)
        try:
            reply = await client.chat(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                role=role, max_tokens=8, temperature=0.0, stage=f"probe:{role}",
            )
            print(f"  {candidate:<44} OK  -> {reply.strip()[:40]!r}")
            return candidate
        except NimError as exc:
            print(f"  {candidate:<44} unavailable ({str(exc)[:70]})")
        except Exception as exc:
            print(f"  {candidate:<44} failed ({type(exc).__name__}: {str(exc)[:60]})")
    return None


async def probe_rerank(client: NimClient) -> str | None:
    query = "what are my rights if the police arrest me"
    passages = [
        "Every person arrested shall be informed of the grounds of arrest and of the right to bail.",
        "The annual report of the institute shall be laid before both Houses of Parliament.",
    ]
    for candidate in registry.candidates("rerank"):
        registry.pin("rerank", candidate)
        try:
            scores = await client.rerank(query, passages, stage="probe:rerank")
            print(f"  {candidate:<44} OK  -> relevant={scores[0]:+.3f}  irrelevant={scores[1]:+.3f}")
            if scores[0] <= scores[1]:
                print("      warning: the relevant passage did not outscore the irrelevant one")
            lo, hi = min(scores), max(scores)
            print(f"      logit scale observed: {lo:+.3f} … {hi:+.3f} "
                  f"({'looks like raw logits' if lo < 0 or hi > 1 else 'looks like [0,1] probabilities'})")
            print("      -> run scripts/calibrate.py before trusting LOW_SCORE / CITE_MIN_SCORE")
            return candidate
        except NimError as exc:
            print(f"  {candidate:<44} unavailable ({str(exc)[:70]})")
        except Exception as exc:
            print(f"  {candidate:<44} failed ({type(exc).__name__}: {str(exc)[:60]})")
    return None


async def main_async(list_only: bool) -> int:
    if not config.NVIDIA_API_KEY:
        print("NVIDIA_API_KEY is not set. Put it in .env or the environment.")
        return 1

    client = NimClient()
    try:
        rule("catalogue")
        try:
            models = await client.list_models()
        except Exception as exc:
            print(f"could not list models: {exc}")
            return 1
        print(f"{len(models)} models visible to this key. Relevant ones:")
        for model in models:
            if INTERESTING.search(model):
                print(f"  {model}")

        embed_hosted = config.EMBED_MODEL.lower() in {m.lower() for m in models}
        rule("embedder")
        print(f"  corpus is locked to {config.EMBED_MODEL} (DB README §8) — it runs locally.")
        print(f"  hosted copy currently {'present' if embed_hosted else 'ABSENT'} on the catalogue"
              f"{'' if embed_hosted else ' — as expected after deprecation; nothing to do'}.")

        if list_only:
            return 0

        rule("chat roles")
        results: dict[str, str | None] = {}
        for role in ("fast", "writer"):
            print(f"{role}:")
            results[role] = await probe_chat_role(client, role)

        rule("reranker (used only by the 'lean' profile / as a local fallback)")
        results["rerank"] = await probe_rerank(client)

        rule("resolved")
        for role, model in results.items():
            if model:
                registry.pin(role, model)
                print(f"  {role:<8} -> {model}")
            else:
                print(f"  {role:<8} -> NONE REACHABLE")
        print(f"\nwritten to {registry.probe_file()}")
        print(client._ledger.report())
        return 0 if results.get("fast") and results.get("writer") else 1
    finally:
        await client.aclose()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-only", action="store_true", help="enumerate models without calling any")
    args = ap.parse_args()
    return asyncio.run(main_async(args.list_only))


if __name__ == "__main__":
    raise SystemExit(main())
