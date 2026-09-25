"""A crawled page, how its text is cleaned, and how it becomes evidence.

Crawled text is untrusted: a page can contain instructions aimed at the model. It is stripped of
markup, anything that reads like a prompt injection is removed and flagged, and it reaches the
writer inside a block labelled as data (see ``context/packer.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import config
from ..evidence import Evidence, tier_for_url

_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANKS_RE = re.compile(r"\n{3,}")
# Phrases that only appear when a page is trying to talk to a model rather than a person.
_INJECTION_RE = re.compile(
    r"(?i)\b(ignore (all )?previous instructions|disregard (the )?above|"
    r"you are now|system prompt|new instructions?:|act as (an? )?\w+ (ai|assistant)|"
    r"</?(system|assistant|user)>)")


def sanitize(text: str) -> tuple[str, bool]:
    """Strip markup and flag anything that reads like a prompt-injection attempt."""
    cleaned = _SCRIPT_RE.sub(" ", text or "")
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = _BLANKS_RE.sub("\n\n", cleaned)
    suspicious = bool(_INJECTION_RE.search(cleaned))
    if suspicious:
        cleaned = _INJECTION_RE.sub("[removed]", cleaned)
    return cleaned.strip(), suspicious


def norm_url(url: str) -> str:
    """Collapse the spellings of one page: trailing slash, fragment, bare index file."""
    cleaned = (url or "").split("#")[0].rstrip("/")
    for suffix in ("/index.php", "/index.html", "/index.htm"):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned.lower()


@dataclass
class Page:
    url: str
    title: str
    markdown: str
    depth: int = 0
    score: float = 0.0
    suspicious: bool = False
    via: str = "http"           # http | browser
    links: list[str] = field(default_factory=list)

    @property
    def is_thin(self) -> bool:
        return len(self.markdown) < config.CRAWL_MIN_CHARS


def page_from_result(result, *, depth: int = 0, via: str = "http") -> Page | None:
    """Convert a crawl4ai result into a :class:`Page`, or None if it holds no usable text."""
    if not getattr(result, "success", False):
        return None
    md = getattr(result, "markdown", None)
    markdown = ""
    if md is not None:
        markdown = getattr(md, "fit_markdown", "") or getattr(md, "raw_markdown", "") or str(md)
    text, suspicious = sanitize(markdown)
    if not text:
        return None
    meta = getattr(result, "metadata", None) or {}
    # The final address after redirects, so the URL guard can check where the text came from.
    url = getattr(result, "redirected_url", None) or getattr(result, "url", "")
    return Page(
        url=url,
        title=(meta.get("title") or "").strip() or url,
        markdown=text,
        depth=int(meta.get("depth", depth) or depth),
        score=float(meta.get("score", 0.0) or 0.0),
        suspicious=suspicious,
        via=via,
    )


def unique_pages(pages: list[Page]) -> list[Page]:
    """One page per address, keeping the longest copy of each."""
    unique: dict[str, Page] = {}
    for page in pages:
        key = norm_url(page.url)
        if key not in unique or len(page.markdown) > len(unique[key].markdown):
            unique[key] = page
    return list(unique.values())


def to_evidence(pages: list[Page], query: str, max_chars: int | None = None) -> list[Evidence]:
    """Turn crawled pages into evidence, tiered by host."""
    max_chars = max_chars or config.WEB_TEXT_CAP * 3
    items: list[Evidence] = []
    for page in pages:
        if not page.markdown.strip():
            continue
        tier = tier_for_url(page.url)
        official = tier >= config.TIER_OFFICIAL
        items.append(Evidence(
            kind="official" if official else "web",
            title=page.title or page.url,
            text=page.markdown[:max_chars],
            url=page.url,
            tier=tier,
            # A read page is much stronger evidence than a search snippet.
            score=0.62 if official else 0.45,
            query=query,
            meta={"crawled": True, "depth": page.depth, "via": page.via,
                  "suspicious": page.suspicious, "chars": len(page.markdown)},
        ))
    return items
