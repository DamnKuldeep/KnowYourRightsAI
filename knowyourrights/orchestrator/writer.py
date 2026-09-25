"""Writing the answer: build the prompt, stream it, check it, and fall back when the model fails.

A deep-mode revision is written silently and swapped in whole when it is done. Streaming it over
the draft cleared the answer the reader was part-way through and started again from nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from .. import events
from ..agents import answer_text, prompts
from ..context import budget as ctx_budget
from ..context import packer
from ..evidence import assign_ids, dedupe
from ..llm.client import get_client
from ..llm.errors import LLMError, ProviderAuthError
from ..tools import legal_db
from .digest import source_digest
from .turn import TurnState

log = logging.getLogger(__name__)

NOTE_WRITER_DOWN = ("The writing model could not be reached, so here are the provisions found "
                    "for your question, unedited.")
NOTE_WRITER_EMPTY = ("The writing model returned nothing, so here are the provisions found for "
                     "your question, unedited.")
NOTE_REWRITE_FAILED = ("Could not rewrite with the confirmed details, so the answer above is the "
                       "original — treat its fees and deadlines as worth double-checking.")
_ENDING_NOTES = {
    "length": "This answer reached its length limit and may be cut short. Ask a narrower "
              "follow-up for the rest.",
    "interrupted": "The connection dropped while this answer was being written, so it may be "
                   "incomplete.",
}


@dataclass
class _Draft:
    """One writing pass: what the model streamed, and how the stream ended."""

    packed: packer.PackResult
    revision: bool
    stage_id: str
    label: str
    prompt: str = ""
    chunks: list[str] = field(default_factory=list)
    endings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.chunks)


async def write_answer(turn: TurnState, *, degraded: bool = False,
                       revision: bool = False) -> None:
    """Write (or, with ``revision``, silently rewrite) the answer from the vetted evidence."""
    draft = _start(turn, revision)
    try:
        await _stream(turn, draft)
    except ProviderAuthError:
        raise
    except LLMError as exc:
        log.warning("writer failed (%s)", exc)
        if not draft.chunks:
            _fail(turn, draft, NOTE_WRITER_DOWN)
            return
    if not draft.text.strip():
        _fail(turn, draft, NOTE_WRITER_EMPTY)
        return
    _finalise(turn, draft, degraded)


def _start(turn: TurnState, revision: bool) -> _Draft:
    """Pack the sources and announce the stage. A revision keeps the draft's sources on screen."""
    history = turn.conversation.history_block(600)
    context = prompts.writer_context(turn.plan, turn.conversation.state,
                                     legal_db.for_writer(turn.notes), date.today().isoformat())
    reserved = ctx_budget.estimate_tokens(history + context + turn.message) + 200
    packed = packer.pack(assign_ids(dedupe(turn.evidence)), ctx_budget.Budget.for_writer(),
                         reserved_tokens=reserved)
    draft = _Draft(packed, revision, "revise" if revision else "write",
                   "Rewriting with the confirmed details" if revision else "Writing the answer")
    draft.prompt = _prompt(turn, packed, history, context, revision)
    if not revision:
        # The packer is the last word on ids, so this is the set the writer may cite and the UI
        # must be able to resolve a citation against.
        turn.emit(events.sources_final(packed.included))
        if packed.dropped:
            turn.emit(events.notice(
                f"Using the {len(packed.included)} strongest sources; {len(packed.dropped)} more "
                f"were set aside to stay within the context budget.", level="info"))
    turn.stage(draft.stage_id, draft.label)
    return draft


def _prompt(turn: TurnState, packed: packer.PackResult, history: str, context: str,
            revision: bool) -> str:
    sources = packed.text or packer.render_empty_note(turn.notes)
    procedure = ""
    if turn.procedure is not None:
        procedure = ("\n\nEXTRACTED PROCEDURE (a summary of the sources above, not a source itself"
                     " — cite the ids it came from, never this block):\n"
                     + turn.procedure.model_dump_json())
    # On a revision the fact-check results ride along, so the model corrects its own draft
    # against what the web actually said rather than restating it.
    verification = f"\n\n{turn.verification_note}" if revision and turn.verification_note else ""
    return (f"{history}\n\nQUESTION: {turn.message}\nANSWER SHAPE: {turn.plan.answer_kind}\n\n"
            f"{context}\n\nVETTED SOURCES — cite only these, by id:\n{sources}"
            f"{procedure}{verification}")


async def _stream(turn: TurnState, draft: _Draft) -> None:
    stream = get_client().chat_stream(
        [{"role": "system", "content": prompts.WRITER},
         {"role": "user", "content": draft.prompt}],
        role="writer", stage="write", deadline=turn.budget.deadline + 45,
        on_pause=turn.on_pause, session=turn.session_id,
        on_reasoning=lambda delta: turn.emit(events.reasoning(delta)),
        on_finish=draft.endings.append,
    )
    async for delta in stream:
        if turn.cancelled.is_set():
            break
        draft.chunks.append(delta)
        if not draft.revision:
            turn.emit(events.token(delta))


def _fail(turn: TurnState, draft: _Draft, note: str) -> None:
    """The model produced nothing. A revision keeps the draft; a first answer gets a digest.

    Handing back the provisions actually found is far more useful than an error: the research
    succeeded and only the prose failed.
    """
    if draft.revision:
        turn.stage(draft.stage_id, draft.label, "done", "kept the original answer")
        turn.emit(events.notice(NOTE_REWRITE_FAILED, level="warn"))
        return
    turn.degrade(note)
    digest = source_digest(draft.packed.included, turn.message)
    for line in digest.splitlines(keepends=True):
        turn.emit(events.token(line))
    turn.answer = digest
    turn.final_evidence = list(draft.packed.included)
    turn.emit(events.verdict(len(draft.packed.included), [],
                             coverage="written without the model", degraded=True))
    turn.stage(draft.stage_id, draft.label, "done")


def _finalise(turn: TurnState, draft: _Draft, degraded: bool) -> None:
    """Clean and check the text, add the state question if needed, and settle the UI's copy."""
    raw = draft.text
    if draft.endings and draft.endings[-1] in _ENDING_NOTES:
        turn.emit(events.notice(_ENDING_NOTES[draft.endings[-1]], level="warn"))
    included = draft.packed.included
    cleaned, unsupported, verified = answer_text.finalise(raw, included)
    follow_up = answer_text.state_question(cleaned, turn.plan, turn.conversation.state)
    if follow_up and not draft.revision and cleaned == raw:
        turn.emit(events.token(follow_up))
        raw += follow_up
    cleaned += follow_up
    if draft.revision:
        # Sources first, so every citation in the new text resolves the moment it appears.
        turn.emit(events.sources_final(included))
    if draft.revision or cleaned != raw:
        turn.emit(events.Event("answer_revised", {"text": cleaned}))
    turn.answer = cleaned
    turn.final_evidence = list(included)
    cited = answer_text.used_evidence(cleaned, included)
    turn.emit(events.verdict(verified, unsupported, degraded=degraded,
                             coverage=f"{len(cited)}/{len(included)} sources cited"))
    if unsupported:
        turn.emit(events.notice(f"Removed {len(unsupported)} citation marker(s) that did not "
                                f"match any source.", level="warn"))
    turn.stage(draft.stage_id, draft.label, "done")
