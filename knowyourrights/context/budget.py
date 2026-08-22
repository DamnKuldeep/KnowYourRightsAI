"""Token counting and per-model input budgets.

The writer model advertises an enormous context window, and we deliberately use a small
fraction of it. Long prompts cost latency and credits, and accuracy falls off well before the
limit does — a fact worth designing around rather than discovering. The budget is therefore a
quality setting, not a capacity one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .. import config

log = logging.getLogger(__name__)

_encoder = None
_encoder_tried = False

# Devanagari and other Indic scripts tokenise far less efficiently than Latin text, so a
# single characters-per-token ratio would badly under-count a Hindi prompt.
_INDIC_RE = re.compile(r"[ऀ-ॿঀ-௿ఀ-ൿ]")
CHARS_PER_TOKEN_LATIN = 3.8
CHARS_PER_TOKEN_INDIC = 1.6


def _get_encoder():
    global _encoder, _encoder_tried
    if not _encoder_tried:
        _encoder_tried = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    return _encoder


def estimate_tokens(text: str) -> int:
    """Approximate token count. Deliberately errs high — under-counting overflows a prompt."""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:
            pass
    indic = len(_INDIC_RE.findall(text))
    latin = len(text) - indic
    return int(latin / CHARS_PER_TOKEN_LATIN + indic / CHARS_PER_TOKEN_INDIC) + 1


def fit_to_tokens(text: str, max_tokens: int) -> str:
    """Trim to a token budget, cutting at a paragraph or sentence boundary where possible."""
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    # Binary search on characters — cheaper than re-encoding progressively longer prefixes.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    cut = text[:low]
    for boundary in ("\n\n", ". ", "\n", " "):
        index = cut.rfind(boundary)
        if index > low * 0.6:
            cut = cut[:index]
            break
    return cut.rstrip() + " …[truncated]"


@dataclass
class Budget:
    """What a given call may spend on input."""

    name: str
    total_tokens: int
    reserved_output: int = 0
    safety: int = 0

    @property
    def usable(self) -> int:
        return max(256, self.total_tokens - self.reserved_output - self.safety)

    @classmethod
    def for_writer(cls) -> "Budget":
        return cls("writer", config.WRITER_INPUT_BUDGET_TOKENS,
                   reserved_output=config.WRITER_MODEL.max_out,
                   safety=config.CONTEXT_SAFETY_TOKENS)

    @classmethod
    def for_fast(cls) -> "Budget":
        return cls("fast", config.FAST_INPUT_BUDGET_TOKENS,
                   reserved_output=config.FAST_MODEL.max_out,
                   safety=config.CONTEXT_SAFETY_TOKENS)
