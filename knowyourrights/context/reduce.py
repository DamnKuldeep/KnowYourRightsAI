"""Shrinking crawled pages before they reach a prompt.

A government FAQ can be 40 KB of which 300 characters answer the question. Four stages, each
cheaper than the one after it, so the expensive tool only ever sees a small candidate set:

1. BM25 filtering at crawl time (in ``tools/crawl.py``) — free, already done.
2. Heading-aware chunking here, keeping the heading path as a breadcrumb so a fragment stays
   interpretable once separated from its page.
3. The reranker, scoring chunks against the sub-question.
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
    chunks: list[Chunk] = []
    for heading, body in _sections((markdown or "").strip()):
        # A long section is broken on blank lines, keeping its heading on every piece.
        pieces = [body] if len(body) <= target else _paragraph_groups(body, target)
        chunks += [Chunk(piece, heading, len(chunks) + i) for i, piece in enumerate(pieces)]
    return [c for c in chunks if len(c.text.strip()) > 40]


def _sections(text: str) -> list[tuple[str, str]]:
    """``(heading, body)`` pairs; any text before the first heading has an empty heading."""
    if not text:
        return []
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections = [("", text[: matches[0].start()].strip())] if matches[0].start() > 0 else []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if body := text[match.end():end].strip():
            sections.append((match.group(2).strip(), body))
    return sections


def _paragraph_groups(body: str, target: int) -> list[str]:
    """Consecutive paragraphs packed into pieces of at most ``target`` characters."""
    pieces: list[str] = []
    buffer = ""
    for paragraph in re.split(r"\n\s*\n", body):
        if buffer and len(buffer) + len(paragraph) + 2 > target:
            pieces.append(buffer.strip())
            buffer = paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
    if buffer.strip():
        pieces.append(buffer.strip())
    return pieces


async def reduce_pages(items: list[Evidence], question: str, *,
                       keep_per_page: int | None = None, max_chars: int | None = None,
                       deadline: float | None = None, on_pause=None) -> list[Evidence]:
    """Replace each long crawled page's text with only the parts that answer ``question``.

    Statutes pass through untouched: a section is already the right unit, and trimming it would
    risk dropping the very proviso that changes the answer.
    """
    keep_per_page = keep_per_page or config.PAGE_CHUNKS_KEPT
    max_chars = max_chars or config.WEB_TEXT_CAP
    targets = [e for e in items if e.kind in ("web", "official") and len(e.text) > max_chars]
    chunked = [(item, chunks) for item in targets
               if len(chunks := split_by_headings(item.text)) > 1]
    if chunked:
        flat = [c.rendered() for _, chunks in chunked for c in chunks]
        scores = await get_reranker().score(question, flat, deadline=deadline,
                                            on_pause=on_pause)
        _keep_best_chunks(chunked, scores, keep_per_page, max_chars)
    for item in targets:
        item.text = item.text[:max_chars]
    return items


def _keep_best_chunks(chunked: list[tuple[Evidence, list[Chunk]]], scores: list[float] | None,
                      keep_per_page: int, max_chars: int) -> None:
    """Keep each page's best chunks, in page order. Without scores, keep its opening chunks."""
    cursor = 0
    for item, chunks in chunked:
        window = scores[cursor:cursor + len(chunks)] if scores else None
        cursor += len(chunks)
        if window:
            for chunk, score in zip(chunks, window, strict=True):
                chunk.score = float(score)
            kept = sorted(chunks, key=lambda c: -c.score)[:keep_per_page]
            item.score = max(item.score, min(0.9, max(window)))
        else:
            # The opening of a page is the least-bad guess at its subject.
            kept = chunks[:keep_per_page]
        kept.sort(key=lambda c: c.order)
        item.meta["reduced_from"] = len(item.text)
        item.meta["chunks_kept"] = f"{len(kept)}/{len(chunks)}"
        item.text = "\n\n".join(c.rendered() for c in kept)[:max_chars]
