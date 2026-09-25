"""Pure ranking functions: fusion, diversity, and what the reranker is shown.

No I/O here, so every function is directly testable.
"""

from __future__ import annotations

from .. import config


def rrf(ranked_lists, k: int | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked id lists, optionally weighted.

    Accepts bare lists or ``(list, weight)`` pairs. Weighting matters when the user names a
    statute: results filtered to that Act are far more likely to be right than a free semantic
    match, and equal weights would let a crowd of loosely similar sections outvote them.
    """
    k = config.RRF_K if k is None else k
    scores: dict[str, float] = {}
    for entry in ranked_lists:
        ranking, weight = entry if isinstance(entry, tuple) else (entry, 1.0)
        for position, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + position + 1)
    return scores


def mmr_order(vectors, relevance, lam: float | None = None, k: int = 5) -> list[int]:
    """Maximal Marginal Relevance: trade relevance against redundancy.

    Keeps the top-k from being five near-identical clauses of one section.
    """
    import numpy as np

    lam = config.MMR_LAMBDA if lam is None else lam
    if len(relevance) == 0:
        return []
    chosen = [int(np.argmax(relevance))]
    remaining = [i for i in range(len(relevance)) if i != chosen[0]]
    while remaining and len(chosen) < k:
        def value(i: int) -> float:
            redundancy = max(float(vectors[i] @ vectors[c]) for c in chosen)
            return lam * float(relevance[i]) - (1.0 - lam) * redundancy

        best = max(remaining, key=value)
        chosen.append(best)
        remaining.remove(best)
    return chosen


def is_general_criminal(queries: list[str]) -> bool:
    """Does this read as an ordinary crime or policing question rather than a sectoral one?"""
    blob = " ".join(queries).lower()
    return any(trigger in blob for trigger in config.CRIMINAL_TRIGGERS)


def rerank_document(row) -> str:
    """What the cross-encoder reads for a candidate section: heading, questions, text.

    The heading is included only when the corpus has one (every criminal-code section does; no
    other section does). A bare citation line in its place is noise that measurably pushed the
    right Article out of the top five.
    """
    heading = clean(getattr(row, "section_name", ""))
    body = clean(getattr(row, "chunk_text", ""))
    questions = citizen_questions(row) if config.RERANK_WITH_QUESTIONS else ""
    return "\n".join(p for p in (heading, questions, body) if p)


def citizen_questions(row) -> str:
    """The questions this section answers, written for it when the corpus was built.

    Statute text often buries its answer: RTI Section 7 opens with a page of provisos before it
    says "thirty days", and scored 0.240 against 0.667 for a section that merely mentions "five
    days". Its build-time questions ("How long does the government have to respond to my
    information request?") restore its meaning in the reader's own words.

    ``embed_text`` is heading, questions, keywords, chunk. This keeps the question lines and
    drops the keyword line, which reads as noise to a cross-encoder.
    """
    embed = clean(getattr(row, "embed_text", ""))
    chunk = clean(getattr(row, "chunk_text", ""))
    if not embed:
        return ""
    _, _, rest = embed.partition("\n")
    if chunk and rest.endswith(chunk):
        rest = rest[: -len(chunk)]
    lines = (line.strip() for line in rest.splitlines())
    return "\n".join(line for line in lines if line.endswith("?"))


def clean(value) -> str:
    """NaN, NaT and 'nan' all become "" so they never reach a prompt or the UI."""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (TypeError, ValueError, ImportError):
        pass
    text = str(value).strip()
    return "" if text.lower() in ("", "nan", "none", "nat", "<na>") else text
