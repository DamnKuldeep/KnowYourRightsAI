"""Race OpenRouter models on the *real* prompts, so role routing is measured rather than guessed.

The older race used "classify this, reply with JSON" — a toy that hides the two things that
actually decide a model's fitness here:

* **fast role** — the full planner system prompt and a strict pydantic schema. A model that is
  quick but produces JSON that fails validation is slower in practice, because the stage then
  retries or falls back. So each run is scored on latency *and* on whether the output parses into
  a valid :class:`Plan`.
* **writer role** — streamed prose. What a person feels is **time to first token**, not total
  time, so both are measured.

Frugal by design. Each candidate gets a couple of calls on realistic inputs, and the cost of every
call is read back from OpenRouter's own ``usage.cost`` rather than estimated. A full run of both
roles costs about one US cent.

    python scripts/race_openrouter.py                 # both roles
    python scripts/race_openrouter.py --role fast
    python scripts/race_openrouter.py --only openai/gpt-oss-20b,qwen/qwen3.7-flash
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                                     # noqa: E402
from knowyourrights.agents import prompts                             # noqa: E402
from knowyourrights.agents.schemas import Plan                        # noqa: E402
from knowyourrights.llm.client import extract_json                    # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()

FAST_CANDIDATES = [
    # paid, cheap — the point is to keep the 4-6 calls per turn OFF the shared free limit
    "openai/gpt-oss-20b",
    "mistralai/mistral-nemo",
    "qwen/qwen3.7-flash",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4-flash-0731",
    "openai/gpt-5-nano",
    "nvidia/nemotron-3-nano-30b-a3b",
    "inception/mercury-2.5",
    # free, for comparison — what the fast role would cost us in rate limit
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
]

WRITER_CANDIDATES = [
    # free — the writer is one call per turn, so the shared free limit can carry it
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "nex-agi/nex-n2.5-pro:free",
    # paid fallbacks, for when the free tier is throttled
    "deepseek/deepseek-v4-flash-0731",
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3.7-flash",
    "openai/gpt-oss-120b",
]

PLANNER_QUESTIONS = [
    "what are my rights if the police arrest me without a warrant?",
    "how do I file an RTI application, what does it cost, and how do I appeal if it is rejected?",
]

WRITER_EVIDENCE = """[S1] STATUTE — Section 47, Bharatiya Nagarik Suraksha Sanhita, 2023
jurisdiction: CENTRAL — Central law — applies across India
(1) Every police officer or other person arresting any person without warrant shall forthwith
communicate to him full particulars of the offence for which he is arrested or other grounds for
such arrest. (2) Where a police officer arrests without warrant any person other than a person
accused of a non-bailable offence, he shall inform the person arrested that he is entitled to be
released on bail and that he may arrange for sureties on his behalf.

[S2] STATUTE — Section 58, Bharatiya Nagarik Suraksha Sanhita, 2023
jurisdiction: CENTRAL — Central law — applies across India
No police officer shall detain in custody a person arrested without warrant for a longer period
than under all the circumstances of the case is reasonable, and such period shall not exceed
twenty-four hours exclusive of the time necessary for the journey from the place of arrest to the
Magistrate's Court.

[S3] STATUTE — Article 22, Constitution of India
jurisdiction: CONSTITUTION — Constitution of India — applies nationwide
(1) No person who is arrested shall be detained in custody without being informed, as soon as
may be, of the grounds for such arrest nor shall he be denied the right to consult, and to be
defended by, a legal practitioner of his choice."""

WRITER_QUESTION = "What are my rights if the police arrest me without a warrant?"


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.OPENROUTER_APP_URL, "X-Title": config.OPENROUTER_APP_NAME}


def _payload(model: str, messages: list[dict], max_tokens: int, stream: bool,
             json_mode: bool) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.1 if json_mode else 0.3, "stream": stream,
            # Reasoning is latency. Every stage here is short and grounded in supplied text, so
            # thinking buys nothing a reader would notice and costs seconds they would.
            "reasoning": {"enabled": False}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


async def _post(client: httpx.AsyncClient, body: dict) -> tuple[dict | None, str]:
    """POST, retrying once without reasoning/json_mode if the upstream rejects those keys."""
    for attempt in range(2):
        r = await client.post(f"{config.OPENROUTER_BASE_URL}/chat/completions", json=body)
        if r.status_code == 200:
            return r.json(), ""
        if r.status_code == 400 and attempt == 0:
            body = {k: v for k, v in body.items() if k not in ("reasoning", "response_format")}
            continue
        return None, f"HTTP {r.status_code}: {r.text[:90]}"
    return None, "rejected"


async def race_fast(client: httpx.AsyncClient, model: str) -> dict:
    times, valid, costs, errors = [], 0, [], []
    for question in PLANNER_QUESTIONS:
        messages = [{"role": "system", "content": prompts.PLANNER},
                    {"role": "user", "content": f"USER MESSAGE: {question}"}]
        t = time.perf_counter()
        data, err = await _post(client, _payload(model, messages, 900, False, True))
        elapsed = (time.perf_counter() - t) * 1000
        if data is None:
            errors.append(err)
            continue
        times.append(elapsed)
        costs.append(float((data.get("usage") or {}).get("cost") or 0.0))
        text = (data["choices"][0]["message"].get("content") or "")
        try:
            Plan.model_validate(json.loads(extract_json(text)))
            valid += 1
        except Exception as exc:                          # noqa: BLE001
            errors.append(f"schema: {type(exc).__name__}")
        await asyncio.sleep(0.5)
    return {"model": model, "times": times, "valid": valid, "n": len(PLANNER_QUESTIONS),
            "cost": sum(costs), "errors": errors}


async def race_writer(client: httpx.AsyncClient, model: str) -> dict:
    messages = [{"role": "system", "content": prompts.WRITER},
                {"role": "user", "content": f"SOURCES:\n{WRITER_EVIDENCE}\n\n"
                                            f"QUESTION: {WRITER_QUESTION}"}]
    body = _payload(model, messages, 700, True, False)
    t0 = time.perf_counter()
    ttft, total_chars, cost, err = None, 0, 0.0, ""
    for attempt in range(2):
        try:
            async with client.stream("POST", f"{config.OPENROUTER_BASE_URL}/chat/completions",
                                     json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode(errors="replace")
                    if r.status_code == 400 and attempt == 0:
                        body = {k: v for k, v in body.items() if k != "reasoning"}
                        continue
                    err = f"HTTP {r.status_code}: {text[:90]}"
                    break
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except ValueError:
                        continue
                    if obj.get("usage"):
                        cost = float(obj["usage"].get("cost") or 0.0)
                    for choice in obj.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content") or ""
                        if delta:
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000
                            total_chars += len(delta)
                break
        except httpx.HTTPError as exc:
            err = f"{type(exc).__name__}"
            break
    total = (time.perf_counter() - t0) * 1000
    return {"model": model, "ttft": ttft, "total": total, "chars": total_chars,
            "cost": cost, "error": err if ttft is None else ""}


async def main_async(args) -> int:
    if not config.OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY is not set.")
        return 1
    only = set(filter(None, (args.only or "").split(",")))
    grand = 0.0
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0),
                                 headers=_headers()) as client:
        if args.role in (None, "fast"):
            rule("fast role — real planner prompt, output must validate as a Plan")
            rows = []
            for m in [m for m in FAST_CANDIDATES if not only or m in only]:
                row = await race_fast(client, m)
                rows.append(row)
                grand += row["cost"]
                if row["times"]:
                    print(f"  {m:<40} median {statistics.median(row['times']):>6.0f} ms  "
                          f"valid {row['valid']}/{row['n']}  ${row['cost']:.5f}"
                          + (f"  [{row['errors'][0]}]" if row["errors"] else ""))
                else:
                    print(f"  {m:<40} FAILED  {row['errors'][:1]}")
            good = [r for r in rows if r["times"] and r["valid"] == r["n"]]
            if good:
                best = min(good, key=lambda r: statistics.median(r["times"]))
                print(f"\n  fastest with every plan valid: {bold(best['model'])}")

        if args.role in (None, "writer"):
            rule("writer role — streamed, time to first token is what a reader feels")
            for m in [m for m in WRITER_CANDIDATES if not only or m in only]:
                row = await race_writer(client, m)
                grand += row["cost"]
                if row["ttft"] is not None:
                    print(f"  {m:<40} first token {row['ttft']:>6.0f} ms  "
                          f"total {row['total']:>6.0f} ms  {row['chars']:>4} chars  "
                          f"${row['cost']:.5f}")
                else:
                    print(f"  {m:<40} FAILED  {row['error']}")
                await asyncio.sleep(1.0)
    print(f"\n  total spent on this race: ${grand:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["fast", "writer"])
    ap.add_argument("--only", help="comma-separated model ids to restrict the race to")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
