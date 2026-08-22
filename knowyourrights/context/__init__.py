"""Keeping prompts inside a budget.

Crawled pages are the one input that grows without bound, so reduction happens in stages that
get progressively more expensive: a free BM25 filter at crawl time, then heading-aware
chunking, then the reranker we already have, and only then hard truncation.
"""

from .budget import Budget, estimate_tokens, fit_to_tokens
from .memory import Conversation, Turn
from .packer import PackResult, pack
from .reduce import reduce_pages

__all__ = ["Budget", "estimate_tokens", "fit_to_tokens", "Conversation", "Turn",
           "PackResult", "pack", "reduce_pages"]
