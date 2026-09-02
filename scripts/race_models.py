"""Time the candidate models against each other so the preference order is measured, not guessed.

Deliberately frugal: two calls per model on a realistic prompt for its role. Ordering the
candidate lists by real latency matters because these stages are on the critical path — the
fast role runs 4-6 times per question.

    python scripts/race_models.py            # both roles
    python scripts/race_models.py --role fast
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                                     # noqa: E402
from knowyourrights.llm.client import NimClient                       # noqa: E402
from knowyourrights.llm.ledger import get_ledger                      # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()

# Realistic work, not "say OK" — a trivial prompt hides the difference that matters.
FAST_PROMPT = (
    'Classify this and reply with ONLY JSON of the form '
    '{"kind":"legal_question","depth":"standard","sub_questions":["..."]}: '
    '"what are my rights if the police arrest me without a warrant?"'
)
WRITER_PROMPT = (
    "In two sentences, plainly explain what Section 47 of the Bharatiya Nagarik Suraksha "
    "Sanhita, 2023 requires of a police officer making an arrest."
)


async def time_model(client: NimClient, spec: config.ModelSpec, prompt: str,
                     max_tokens: int, runs: int) -> dict:
    latencies: list[float] = []
    chars = 0
    error = ""
    base, headers = client._endpoint(spec.provider)

    for i in range(runs):
        payload = client._chat_payload(spec, [{"role": "user", "content": f"{prompt} ({i})"}],
                                       temperature=0.0, max_tokens=max_tokens, stream=False)
        started = time.perf_counter()
        try:
            response = await client.client.post(f"{base}/chat/completions",
                                                headers=headers, json=payload)
            if response.status_code != 200:
                error = f"HTTP {response.status_code}"
                break
            body = response.json()["choices"][0]["message"]
            chars += len((body.get("content") or "") + (body.get("reasoning_content") or ""))
            latencies.append((time.perf_counter() - started) * 1000)
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:40]}"
            break

    # A model that answered once then 429'd is not "fast" — it is unreliable. Only report a
    # timing when every run completed, or the summary flatters models that half-worked.
    complete = not error and len(latencies) == runs
    return {"spec": spec, "error": error, "runs": len(latencies), "chars": chars,
            "median_ms": statistics.median(latencies) if complete else None,
            "partial_ms": statistics.median(latencies) if latencies else None}


async def race(client: NimClient, role: str, runs: int) -> list[dict]:
    specs = list(config.FAST_MODELS if role == "fast" else config.WRITER_MODELS)
    prompt = FAST_PROMPT if role == "fast" else WRITER_PROMPT
    max_tokens = 200 if role == "fast" else 220

    rows = []
    for spec in specs:
        if not config.provider_available(spec.provider):
            continue
        result = await time_model(client, spec, prompt, max_tokens, runs)
        rows.append(result)
        if result["error"]:
            print(f"  {spec.key:<52} {result['error']}")
        else:
            print(f"  {spec.key:<52} {result['median_ms']:>7.0f} ms   "
                  f"{result['chars'] // max(1, result['runs']):>4} chars")
    return rows


async def main_async(args) -> int:
    client = NimClient()
    ledger = get_ledger()
    ledger.load_daily()
    try:
        print(f"OpenRouter requests left today: {ledger.daily_remaining('openrouter')}")
        roles = [args.role] if args.role else ["fast", "writer"]
        results = {}
        for role in roles:
            rule(f"{role} role — {args.runs} calls each, realistic prompt")
            results[role] = await race(client, role, args.runs)

        rule("recommended order (fastest working model first)")
        for role, rows in results.items():
            ok = sorted([r for r in rows if r["median_ms"] is not None],
                        key=lambda r: r["median_ms"])
            print(f"\n{bold(role)}:")
            for i, r in enumerate(ok, 1):
                print(f"  {i}. {r['spec'].key:<50} {r['median_ms']:>7.0f} ms")
            for r in [x for x in rows if x["median_ms"] is None]:
                partial = f"  (1st call {r['partial_ms']:.0f} ms)" if r["partial_ms"] else ""
                print(f"  -- {r['spec'].key:<50} {r['error']}{partial}")
        print(f"\nOpenRouter requests left today: {ledger.daily_remaining('openrouter')}")
        return 0
    finally:
        await client.aclose()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["fast", "writer"])
    ap.add_argument("--runs", type=int, default=2)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
