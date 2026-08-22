"""Shrinking crawled pages before they reach a prompt.

A government FAQ can be 40 KB of which 300 characters answer the question. Four stages, each
cheaper than the one after it, so the expensive tool only ever sees a small candidate set:

1. BM25 filtering at crawl time (in ``tools/crawl.py``) — free, already done.
2. Heading-aware chunking here, keeping the heading path as a breadcrumb so a fragment stays
   interpretable once separated from its page.
3. The cross-encoder we have already loaded, scoring chunks against the sub-question.
4. Hard caps, as a floor rather than the main mechanism.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .. import config
from ..evidence import Evidence
from ..retrieval.reranker import get_reranker

log = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


@dataclass
class Chunk:
    text: str
    heading: str
    order: int
    score: float = 0.0

    def rendered(self) -> str:
        return f"**{self.heading}**\n{self.text}" if self.heading else self.text


def split_by_headings(markdown: str, target_chars: int | None = None) -> list[Chunk]:
    """Split on markdown headings, then pack sections up to ``target_chars``.

    Heading-aware rather than fixed-width because a procedure's steps and its fee table live
    under different headings, and slicing across them produces fragments that answer nothing.
    """
    target = target_chars or config.PAGE_CHUNK_CHARS
    text = (markdown or "").strip()
    if not text:
        return []

    matches = list(_HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        sections.append(("", text))
    else:
        if matches[0].start() > 0:
            sections.append(("", text[: matches[0].start()].strip()))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end():end].strip()
            if body:
                sections.append((match.group(2).strip(), body))

    chunks: list[Chunk] = []
    for heading, body in sections:
        if len(body) <= target:
            chunks.append(Chunk(body, heading, len(chunks)))
            continue
        # Long section: break on blank lines, keeping the heading on every piece.
        buffer = ""
        for paragraph in re.split(r"\n\s*\n", body):
            if len(buffer) + len(paragraph) + 2 > target and buffer:
                chunks.append(Chunk(buffer.strip(), heading, len(chunks)))
                buffer = paragraph
            else:
                buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if buffer.strip():
            chunks.append(Chunk(buffer.strip(), heading, len(chunks)))

    return [c for c in chunks if len(c.text.strip()) > 40]


async def reduce_pages(items: list[Evidence], question: str, *,
                       keep_per_page: int | None = None,
                       max_chars: int | None = None,
                       deadline: float | None = None,
                       on_pause=None, session: str = "") -> list[Evidence]:
    """Replace each crawled page's text with only the parts that answer ``question``.

    Statutes are passed through untouched — a section is already the right unit, and trimming
    it would risk dropping the very proviso that changes the answer.
    """
    keep_per_page = keep_per_page or config.PAGE_CHUNKS_KEPT
    max_chars = max_chars or config.WEB_TEXT_CAP

    targets = [e for e in items if e.kind in ("web", "official") and len(e.text) > max_chars]
    if not targets:
        return items

    chunked: list[tuple[Evidence, list[Chunk]]] = []
    flat: list[str] = []
    for item in targets:
        chunks = split_by_headings(item.text)
        if len(chunks) <= 1:
            continue
        chunked.append((item, chunks))
        flat.extend(c.rendered() for c in chunks)

    if not chunked:
        for item in targets:
            item.text = item.text[:max_chars]
        return items

    reranker = get_reranker()
    scores = await reranker.score(question, flat, deadline=deadline,
                                 on_pause=on_pause, session=session)

    cursor = 0
    for item, chunks in chunked:
        window = scores[cursor:cursor + len(chunks)] if scores else None
        cursor += len(chunks)
        if window:
            for chunk, score in zip(chunks, window):
                chunk.score = float(score)
            ordered = sorted(chunks, key=lambda c: -c.score)[:keep_per_page]
        else:
            # No reranker: the opening of a page is the least-bad heuristic for its subject.
            ordered = chunks[:keep_per_page]

        ordered.sort(key=lambda c: c.order)
        kept = "\n\n".join(c.rendered() for c in ordered)[:max_chars]
        item.meta["reduced_from"] = len(item.text)
        item.meta["chunks_kept"] = f"{len(ordered)}/{len(chunks)}"
        item.text = kept
        if window:
            item.score = max(item.score, min(0.9, max(window)))

    for item in targets:
        if len(item.text) > max_chars:
            item.text = item.text[:max_chars]
    return items


def cap_text(item: Evidence) -> str:
    """The per-kind hard ceiling. Statutes get the most room — they are what gets cited."""
    if item.is_statute:
        return item.text[:config.STATUTE_TEXT_CAP]
    if item.kind == "wikipedia":
        return item.text[:config.WIKI_TEXT_CAP]
    return item.text[:config.WEB_TEXT_CAP]
