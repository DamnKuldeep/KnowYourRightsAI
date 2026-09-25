"""Deep mode's self-check: confirm the draft's time-sensitive claims against the live web.

The statute corpus cannot know that a fee changed, a deadline moved or a procedure went online.
The agent names what in its own draft it is unsure of; if it is confident, nothing is spent. When
checks come back, the answer is rewritten with them rather than corrected in a footnote.
"""

from __future__ import annotations

import logging

from .. import events
from ..agents import stages
from ..agents.schemas import FactCheck
from ..evidence import Evidence, assign_ids, dedupe
from ..tools import web
from .turn import TurnState, degradable
from .writer import write_answer

log = logging.getLogger(__name__)

LABEL = "Checking my own answer against live sources"
MAX_CLAIMS = 2        # a check, not a second research round


async def self_verify(turn: TurnState) -> None:
    if turn.budget.depth != "deep" or turn.budget.expired or turn.cancelled.is_set():
        return
    if not turn.answer.strip():
        return
    turn.stage("verify", LABEL)
    check = await degradable(
        turn, stages.find_risky_claims(turn.question, turn.answer, turn.evidence,
                                       **turn.call_kwargs()),
        FactCheck(confident=True))
    if not check.needs_checking:
        turn.stage("verify", LABEL, "done", "nothing needed checking")
        return
    claims = check.claims[:MAX_CLAIMS]
    turn.stage("verify", LABEL, "done", f"{len(claims)} claim(s) to confirm")
    findings = await _check_claims(turn, claims)
    confirmed = [item for found in findings.values() for item in found]
    if not confirmed:
        turn.emit(events.notice(
            "I could not independently confirm one or more time-sensitive details, so treat the "
            "fees and deadlines above as worth double-checking.", level="warn"))
        return
    turn.evidence = assign_ids(dedupe(turn.evidence + confirmed))
    turn.verification_note = stages.verification_note(claims, findings)
    turn.emit(events.notice(f"Confirmed {len(claims)} detail(s) against official sources; "
                            f"rewriting with what they said.", level="info"))
    # The draft stays in turn.answer until a rewrite replaces it, so a failed rewrite leaves the
    # reader with the draft rather than nothing.
    await write_answer(turn, revision=True)


async def _check_claims(turn: TurnState, claims: list) -> dict[str, list[Evidence]]:
    findings: dict[str, list[Evidence]] = {}
    for claim in claims:
        if turn.budget.expired or turn.cancelled.is_set():
            break
        turn.emit(events.tool("verify", claim.claim[:70], "running", detail=claim.why[:80]))
        found = await web.search_official(claim.query, n=3)
        findings[claim.claim] = found
        turn.emit(events.tool("verify", claim.claim[:70], "done", count=len(found)))
    return findings
