"""Find out which models each provider can actually serve, and cache the answer.

Hosted catalogues change underneath you. Rather than discover a retired id mid-answer, probe
each role's candidates here and pin the winner into ``.runtime/model_probe.json``.

Deliberately frugal: **one call per role**, stopping at the first model that answers. A full
run is 2-3 requests. Listing catalogues costs nothing at all.

    python scripts/probe_models.py --list-only   # free: enumerate both catalogues
    python scripts/probe_models.py               # ~3 requests total
    python scripts/probe_models.py --all         # try every candidate (more requests)
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                                     # noqa: E402
from knowyourrights.llm import registry                               # noqa: E402
from knowyourrights.llm.client import NimClient, NimError             # noqa: E402
from knowyourrights.llm.ledger import get_ledger                      # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()

INTERESTING = re.compile(r"bge|embed|rerank|nemotron|gemma|glm|ling|minimax|inkling", re.I)


async def probe_role(client: NimClient, role: str, try_all: bool) -> str | None:
    """One call per candidate, stopping at the first success unless --all."""
    winner = None
    for spec in registry.candidates(role):
        if winner and not try_all:
            break
        registry.pin(role, spec.key)
        try:
            reply = await client.chat([{"role": "user", "content": "Reply with exactly: OK"}],
                                      role=role, max_tokens=8, temperature=0.0,
                                      stage=f"probe:{role}")
            text = (reply or "").strip().replace("\n", " ")[:34]
            print(f"  {spec.key:<52} OK   {text!r}")
            winner = winner or spec.key
        except NimError as exc:
            print(f"  {spec.key:<52} no   ({str(exc)[:58]})")
        except Exception as exc:
            print(f"  {spec.key:<52} err  ({type(exc).__name__}: {str(exc)[:44]})")
    return winner


async def probe_rerank(client: NimClient) -> str | None:
    """Reranking is NVIDIA-only; OpenRouter has no reranking endpoint."""
    query = "what are my rights if the police arrest me"
    passages = ["Every person arrested shall be informed of the grounds of arrest.",
                "The annual report of the institute shall be laid before Parliament."]
    for spec in registry.candidates("rerank"):
        registry.pin("rerank", spec.key)
        try:
            scores = await client.rerank(query, passages, stage="probe:rerank")
            print(f"  {spec.key:<52} OK   relevant={scores[0]:+.2f} other={scores[1]:+.2f}")
            if scores[0] <= scores[1]:
                print("       warning: the relevant passage did not score higher")
            print("       -> run scripts/calibrate.py; thresholds are per reranker")
            return spec.key
        except NimError as exc:
            print(f"  {spec.key:<52} no   ({str(exc)[:58]})")
        except Exception as exc:
            print(f"  {spec.key:<52} err  ({type(exc).__name__}: {str(exc)[:44]})")
    return None


async def main_async(args) -> int:
    if not (config.NVIDIA_API_KEY or config.OPENROUTER_API_KEY):
        print("No provider configured. Set NVIDIA_API_KEY and/or OPENROUTER_API_KEY in .env.")
        return 1

    client = NimClient()
    ledger = get_ledger()
    ledger.load_daily()
    try:
        rule("providers")
        for name in config.PROVIDERS:
            have = config.provider_available(name)
            extra = ""
            if name == "openrouter" and have:
                extra = f"  ({ledger.daily_remaining('openrouter')} requests left today)"
            print(f"  {name:<12} {'configured' if have else 'no key'}{extra}")

        rule("catalogues (free to list)")
        for name in config.PROVIDERS:
            if not config.provider_available(name):
                continue
            try:
                models = await client.list_models(name)
            except Exception as exc:
                print(f"  {name}: could not list ({str(exc)[:60]})")
                continue
            if name == "openrouter":
                models = [m for m in models if m.endswith(":free")]
                print(f"  {name}: {len(models)} free models")
            else:
                print(f"  {name}: {len(models)} models")
            for model in models:
                if INTERESTING.search(model):
                    print(f"      {model}")

        rule("embedder")
        print(f"  {config.EMBED_MODEL} runs locally and cannot be swapped —")
        print("  the corpus is embedded with it (see the DB README §8).")

        if args.list_only:
            return 0

        results: dict[str, str | None] = {}
        rule("chat roles (one call per candidate, stops at the first success)")
        for role in ("fast", "writer"):
            print(f"{role}:")
            results[role] = await probe_role(client, role, args.all)

        rule("reranker (NVIDIA only; used by the lean profiles)")
        results["rerank"] = await probe_rerank(client)

        rule("resolved")
        for role, key in results.items():
            if key:
                registry.pin(role, key)
                print(f"  {role:<8} -> {bold(key)}")
            else:
                print(f"  {role:<8} -> none reachable")
        print(f"\n  written to {registry.probe_file()}")
        print(f"  OpenRouter requests left today: {ledger.daily_remaining('openrouter')}")
        return 0 if results.get("fast") and results.get("writer") else 1
    finally:
        await client.aclose()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-only", action="store_true", help="enumerate catalogues, call nothing")
    ap.add_argument("--all", action="store_true", help="try every candidate, not just the first")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
