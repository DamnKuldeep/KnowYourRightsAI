"""Drive the real web UI through every kind of question, and screenshot each answer.

The API-level checks (``e2e_check.py``) prove the pipeline; this proves what a person sees —
the answer as rendered, the safety card, the jurisdiction badges, the notices — in a real
browser against the running server. Each case records a screenshot plus a machine-readable
summary, and a short list of expectations is checked so a regression is visible without
reading every image.

    python -m knowyourrights.server             # in one terminal
    python scripts/ui_check.py                  # in another; screenshots in .runtime/ui_check/
    python scripts/ui_check.py --only rti       # cases whose name contains "rti"

Costs roughly 5 US cents for the full set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config


@dataclass
class Case:
    name: str
    question: str
    state: str = ""                 # "" = All India
    depth: str = "auto"
    new_chat: bool = True
    # expectations: every string in `must` appears in the answer (case-insensitive), none in
    # `must_not`; `safety` says whether the helpline card should appear
    must: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    safety: bool | None = None
    note: str = ""


CASES = [
    Case("rti", "How do I file an RTI, and what does it cost?",
         must=["10", "Section 7", "Section 19"],
         must_not=["Maharashtra", "except the State of Jammu", "do not state"],
         note="All India selected: central portal, ₹10, no state bleed"),
    Case("rti_followup", "and if they don't reply in time, what can I do?", new_chat=False,
         must=["appeal"], note="follow-up resolved from history; §19 appeal"),
    Case("hinglish", "Police ne bina warrant arrest kar liya, kya yeh legal hai?",
         must=["warrant"], note="Hinglish in, Hinglish out, BNSS"),
    Case("hindi", "मेरा मकान मालिक सिक्योरिटी डिपॉजिट वापस नहीं कर रहा, मैं क्या करूँ?",
         must=["राज्य"], note="Devanagari in, Devanagari out; asks which state"),
    Case("place_vs_state", "My landlord in Mumbai won't return my deposit", state="Kerala",
         must=["Maharashtra"], must_not=["applies centrally"],
         note="flat in Mumbai, user in Kerala: Maharashtra law governs"),
    Case("state_subject", "What are the rules for my housing society?", state="Karnataka",
         must=["Karnataka"], note="state subject; corpus has no Karnataka Act"),
    Case("union_territory", "What does the Delhi Rent Control Act say about eviction?",
         must=["Delhi"], must_not=["which state are you"],
         note="Parliament's Act for Delhi: TERRITORY, Delhi only"),
    Case("safety_literal", "my husband is hitting me", safety=True,
         note="helpline card before any research"),
    Case("safety_paraphrase", "my partner keeps hurting me and I'm scared to go home",
         safety=True, note="no literal pattern — the meaning tier must catch it"),
    Case("not_a_disclosure", "what is the punishment for rape in India?", safety=False,
         note="a legal question about a crime, not a disclosure — no helpline card"),
    Case("off_topic", "who won the cricket world cup in 2011?",
         must_not=["Section", "Act,"], note="not law: decline, cite nothing"),
    Case("exact", "What does Article 21 say?", depth="quick",
         must=["personal liberty"], note="exact lookup, no search"),
    Case("repealed", "What is the punishment under Section 420 IPC?",
         must=["318"], must_not=["420 of the Bharatiya", "Section 420 BNS", "never cite"],
         note="IPC repealed: map to BNS 318, never cite IPC as current"),
    Case("not_in_corpus", "What does the DPDP Act say about consent for my personal data?",
         note="Act not in the corpus: say so, do not invent sections"),
    Case("deep_compare", "Should I use an RTI or a consumer complaint to chase my delayed "
                         "passport? Compare both.", depth="deep",
         must=["RTI", "consumer"], note="deep mode, multi-part comparison"),
]


SOURCES_JS = ("els => els.map(e => ({title: e.querySelector('.title')?.innerText || '', "
              "badges: [...e.querySelectorAll('.badge')].map(b => b.innerText), "
              "warn: [...e.querySelectorAll('.warn-line')].map(w => w.innerText)}))")


def ask(page, case: Case) -> tuple[object, float]:
    """Type the question and wait for the answer to finish. Returns (its message, seconds)."""
    if case.new_chat:
        page.click("#reset")
        page.wait_for_timeout(400)
    page.select_option("#state", case.state or "")
    page.click(f"button[data-depth='{case.depth}']")
    before = page.locator(".msg:not(.user)").count()
    page.fill("#input", case.question)
    started = time.time()
    page.press("#input", "Enter")
    page.wait_for_selector("#send.stop", timeout=20_000)
    page.wait_for_selector("#send:not(.stop)", timeout=300_000)
    elapsed = time.time() - started
    page.wait_for_timeout(600)
    messages = page.locator(".msg:not(.user)")
    return (messages.nth(before) if messages.count() > before else page.locator(".msg").last,
            elapsed)


def observe(page, turn) -> dict:
    """What the reader sees for one answer."""
    return {
        "answer": turn.locator(".answer").inner_text() if turn.locator(".answer").count() else "",
        "safety_card": turn.locator(".safety").count() > 0,
        "notices": [n.inner_text() for n in turn.locator(".notice").all()],
        "sources": page.eval_on_selector_all(".src", SOURCES_JS),
        "stat": page.inner_text("#stat"),
    }


def screenshot(page, turn, path: Path) -> None:
    """Scroll so the answer starts at the top of the viewport, then photograph it."""
    turn.scroll_into_view_if_needed()
    page.evaluate("(el) => el.scrollIntoView({block: 'start'})", turn.element_handle())
    page.wait_for_timeout(250)
    page.screenshot(path=str(path))


def problems_with(case: Case, seen: dict) -> list[str]:
    low = seen["answer"].lower()
    # A "must" can be met by a cited source's title: the writer often cites Section 7 as a
    # citation without spelling the number out.
    cited = (low + " " + " ".join(s["title"] for s in seen["sources"])).lower()
    problems = [f"missing {m!r}" for m in case.must if m.lower() not in cited]
    problems += [f"contains {m!r}" for m in case.must_not if m.lower() in low]
    if case.safety is not None and seen["safety_card"] != case.safety:
        problems.append(f"safety card {'absent' if case.safety else 'shown'}")
    if not low.strip():
        problems.append("no answer")
    return problems


def run(args) -> int:
    from playwright.sync_api import sync_playwright

    out = config.RUNTIME_DIR / "ui_check"
    out.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until="networkidle")
        for case in [c for c in CASES if not args.only or args.only in c.name]:
            turn, elapsed = ask(page, case)
            seen = observe(page, turn)
            shot = out / f"{len(results) + 1:02d}_{case.name}.png"
            screenshot(page, turn, shot)
            problems = problems_with(case, seen)
            results.append({"case": case.name, "question": case.question, "state": case.state,
                            "depth": case.depth, "note": case.note, "seconds": round(elapsed, 1),
                            **seen, "sources": seen["sources"][:6], "problems": problems,
                            "screenshot": str(shot)})
            print(f"  {'FAIL' if problems else 'PASS'}  {case.name:<20} {elapsed:5.1f}s  "
                  f"{seen['stat']:<42} {'; '.join(problems)}")
        browser.close()
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    failed = sum(bool(r["problems"]) for r in results)
    print(f"\n  {len(results) - failed}/{len(results)} passed · js errors: "
          f"{errors or 'none'} · screenshots in {out}")
    return 1 if failed or errors else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"http://127.0.0.1:{config.PORT}")
    ap.add_argument("--only", help="run only cases whose name contains this")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
