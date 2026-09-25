"""Post-processing the written answer. No model calls: every rule here is deterministic.

Each rule fixes something a writer model did in a real answer and that no prompt instruction
reliably prevented: citation shapes nothing can link, page titles pasted as link text, lines
announcing that the sources say nothing, headings left empty, and a missing question about which
state the reader is in.
"""

from __future__ import annotations

import re

from ..evidence import Evidence

_MARKER_RE = re.compile(r"\[([A-Z]{1,2}\d{1,2})\]")
# Anything bracketed that starts with a source id ("[S1(a)]", "[S1, S2]"), but not a markdown
# link, whose "]" is followed by "(".
_COMPOUND_RE = re.compile(r"\[([A-Z]{1,2}\d{1,2}[^\[\]\n]{0,40})\](?!\()")
_ID_RE = re.compile(r"[A-Z]{1,2}\d{1,2}")
_CLAUSE_RE = re.compile(r"\([^()]{1,8}\)")
_JOINERS = re.compile(r"^[\s,;&\-–—/]*(?:and[\s,;&\-–—/]*)*$", re.I)
# The prompt's own block names, which the writer sometimes cites as if they were sources.
_BLOCK_LABEL_RE = re.compile(
    r"\s?\[(?:EXTRACTED PROCEDURE|PROCEDURE|SOURCES?|VERIFICATION|HISTORY|CONTEXT)\]")
_MD_LINK = re.compile(r"\[([^\[\]\n]{2,160})\]\((https?://[^\s)]+)\)")
_TITLE_SPLIT = re.compile(r"\s*(?:::|\s\|\s|\s[-–—]\s)\s*")
_HEADING = re.compile(r"^(#{1,6}\s+\S.*|\*\*[^*]{1,80}\*\*:?)$")
_LIST_OR_LABEL = re.compile(r"^([-*•]|\d+[.)])\s+|^\*\*[^*]{1,40}:?\*\*")
# "Not stated in the sources" and "the sources do not state", in English and Hindi.
_UNSTATED = re.compile(
    r"(\b(not|isn't|is not|are not)\s+(explicitly\s+|clearly\s+)?"
    r"(stated|mentioned|specified|given|provided|available|found)\b.{0,40}\b(sources?|documents?)\b"
    r"|\b(sources?|documents?)\b.{0,30}\b(do|does|did)\s+not\s+"
    r"(state|mention|specify|say|provide|include|cover|give)\b"
    r"|स्रोत\S*\s.{0,90}नहीं)",
    re.I)


def finalise(answer: str, items: list[Evidence]) -> tuple[str, list[str], int]:
    """Every cleanup, then citation checking: ``(answer, unsupported_markers, verified_count)``.

    A marker that resolves to no supplied source is removed rather than shown: a citation the
    reader cannot open implies support that does not exist.
    """
    answer = drop_unstated_lines(tidy_link_labels(normalize_markers(answer or "")))
    known = {item.id for item in items}
    found = _MARKER_RE.findall(answer)
    unsupported = sorted({m for m in found if m not in known})
    for marker in unsupported:
        answer = answer.replace(f"[{marker}]", "")
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r" +([.,;:])", r"\1", answer)
    return answer.strip(), unsupported, len({m for m in found if m in known})


def normalize_markers(text: str) -> str:
    """Rewrite invented citation shapes into the one shape that can be verified and linked.

    ``[S1(a)]`` and ``[S1, S2]`` become ``[S1]`` and ``[S1][S2]``. Conservative: a bracket is
    rewritten only when nothing but ids, short clause groups and separators is inside it.
    """
    def fix(match: re.Match) -> str:
        inner = match.group(1)
        ids = _ID_RE.findall(inner)
        rest = _CLAUSE_RE.sub("", _ID_RE.sub("", inner))
        if not ids or not _JOINERS.match(rest):
            return match.group(0)
        return "".join(f"[{m}]" for m in dict.fromkeys(ids))

    return _BLOCK_LABEL_RE.sub("", _COMPOUND_RE.sub(fix, text or ""))


def tidy_link_labels(text: str) -> str:
    """Shorten link text that is really a page's <title>, breadcrumbs and all.

    The first segment of such a title is the site's own name, which is what a reader clicks.
    """
    def fix(m: re.Match) -> str:
        label, url = m.group(1).strip(), m.group(2)
        if not (_TITLE_SPLIT.search(label) or len(label) > 60):
            return m.group(0)
        short = _TITLE_SPLIT.split(label)[0].strip() or label
        if len(short) > 60:
            short = " ".join(short[:60].split()[:-1]) or short[:60]
        return f"[{short}]({url})"

    return _MD_LINK.sub(fix, text or "")


def drop_unstated_lines(text: str) -> str:
    """Remove short "Fee: not stated in the provided sources" lines, and headings they empty.

    Such a line reads as "the law says nothing about this" when the real problem is that we did
    not find it. Only list or label lines are removed; real prose that mentions sources stays.
    """
    kept: list[str] = []
    heading_at: int | None = None     # index in `kept` of the current section's heading
    listed = dropped = 0              # list lines seen and removed in that section
    emptied: set[int] = set()         # headings whose every list line was removed
    for line in (text or "").splitlines():
        stripped = line.strip()
        if _HEADING.match(stripped):
            if heading_at is not None and listed and dropped == listed:
                emptied.add(heading_at)
            heading_at, listed, dropped = len(kept), 0, 0
        elif _LIST_OR_LABEL.match(stripped):
            listed += 1
            if len(stripped) < 180 and _UNSTATED.search(stripped):
                dropped += 1
                continue
        kept.append(line)
    if heading_at is not None and listed and dropped == listed:
        emptied.add(heading_at)
    out = "\n".join(line for i, line in enumerate(kept) if i not in emptied)
    return re.sub(r"\n{3,}", "\n\n", out) if emptied else out


def used_evidence(answer: str, items: list[Evidence]) -> list[Evidence]:
    """The sources the answer actually cited, in order of first appearance."""
    by_id = {item.id: item for item in items}
    cited = dict.fromkeys(m for m in _MARKER_RE.findall(answer or "") if m in by_id)
    return [by_id[m] for m in cited]


_STATE_QUESTION = {
    "en": "Which state is this in? The rules here differ from state to state — tell me, or set "
          "your state above, and I can point you to the exact law and authority.",
    "hi": "यह मामला किस राज्य का है? इस विषय के नियम हर राज्य में अलग हैं — राज्य बताइए या ऊपर "
          "चुनिए, तो मैं सही कानून और प्राधिकरण बता सकूँगा।",
    "hinglish": "Yeh kis state ka mamla hai? Is par rules har state mein alag hain — state "
                "bataiye ya upar select kijiye, toh main sahi law aur authority bata sakta hoon.",
}
# The answer already asks when its last line mentions a state and ends in a question, possibly
# followed by markdown or citation markers.
_ASKS_STATE = re.compile(r"(state|राज्य|rajya)[^\n]{0,160}\?(\s*\[[SGW]\d+\])*[\s*_)]*$", re.I)


def state_question(answer: str, plan, state: str | None) -> str:
    """The question to append when the answer depends on a state nobody has named, else "".

    The writer is told to ask and does so only some of the time: a Hindi deposit question with
    "All India" selected was routed to the consumer helpline without a word about state law.
    """
    if state or not getattr(plan, "needs_state", False):
        return ""
    if _ASKS_STATE.search((answer or "").rstrip()[-400:]):
        return ""
    return "\n\n" + _STATE_QUESTION.get(getattr(plan, "language", "en"), _STATE_QUESTION["en"])
