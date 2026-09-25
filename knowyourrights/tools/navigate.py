"""Walking a government portal towards a procedure.

Procedures live several clicks inside a portal: "apply" leads to "guidelines" leads to "fees"
leads to "appeal". Answering "how do I file an RTI and what does it cost" means following links,
not reading one page.

Best-first rather than breadth-first: links are ranked by how much they look like a procedure
page, so the crawl spends its budget on "apply" and "fees" instead of "about us". When crawl4ai's
deep crawl stays on the seed page — common on Indian government portals — the highest-ranked
links are followed directly as a predictable backstop.
"""

from __future__ import annotations

import asyncio
import logging
import re

from .. import config
from ..evidence import domain_of
from . import url_safety
from .crawl import Crawler, run_config
from .pages import Page, norm_url, page_from_result, unique_pages

log = logging.getLogger(__name__)

# Link text and URL words that tend to lead towards the actual procedure.
PROCEDURE_KEYWORDS = (
    "apply", "application", "procedure", "how to", "guideline", "instruction", "faq",
    "form", "fee", "charges", "submit", "register", "grievance", "appeal", "complaint",
    "eligibility", "documents", "download", "status", "portal", "citizen", "service",
)
_DEAD_END = re.compile(r"login|signin|register\.php|logout|contact")
_BINARY = re.compile(r"\.(pdf|jpe?g|png|gif|zip|docx?|xlsx?)$", re.I)


async def navigate(crawler: Crawler, seed_url: str, goal: str, *, max_pages: int = 6,
                   max_depth: int = 2, should_cancel=None) -> list[Page]:
    """Walk a site from ``seed_url`` towards pages that answer ``goal``."""
    if not seed_url or not await url_safety.is_public_url(seed_url):
        return []
    strategy = _strategy(seed_url, goal, max_pages, max_depth, should_cancel)
    if strategy is None:
        return await crawler.fetch([seed_url], goal)

    links, via = await discover_links(crawler, seed_url)
    pages = await _deep_crawl(crawler, seed_url, goal, strategy, via, max_pages)
    if len({norm_url(p.url) for p in pages}) <= 1 and links:
        pages += await _follow_ranked_links(crawler, links, goal, seed_url, via, max_pages, pages)
    if not pages:
        pages = await crawler.fetch([seed_url], goal)

    pages = [p for p in unique_pages(pages) if await url_safety.is_public_url(p.url)]
    crawler.pages_fetched += len(pages)
    return pages


def _strategy(seed_url: str, goal: str, max_pages: int, max_depth: int, should_cancel):
    """crawl4ai's best-first strategy, or None when deep crawling is not installed."""
    try:
        from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
        from crawl4ai.deep_crawling.filters import ContentTypeFilter, DomainFilter, FilterChain
        from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
    except ImportError as exc:
        log.warning("deep crawling unavailable (%s); reading the seed page only", exc)
        return None

    # Only hard constraints belong in the filter chain: it is applied to the start URL too, so a
    # URL-pattern filter for "*apply*" would reject a portal's home page and end the crawl before
    # a single link was followed. Preferring procedure-shaped links is the scorer's job.
    filters = [ContentTypeFilter(allowed_types=["text/html"])]
    domain = domain_of(seed_url)
    if domain:
        filters.append(DomainFilter(allowed_domains=[domain]))
    strategy = BestFirstCrawlingStrategy(
        max_depth=max_depth, max_pages=max_pages, include_external=False,
        filter_chain=FilterChain(filters),
        url_scorer=KeywordRelevanceScorer(keywords=keywords_for(goal), weight=0.8),
    )
    if should_cancel is not None and hasattr(strategy, "should_cancel"):
        strategy.should_cancel = should_cancel
    return strategy


async def _deep_crawl(crawler: Crawler, seed_url: str, goal: str, strategy, via: str,
                      max_pages: int) -> list[Page]:
    engine = await crawler.engine(browser=(via == "browser")) or await crawler.http_engine()
    try:
        results = await asyncio.wait_for(
            engine.arun(url=seed_url, config=run_config(goal, deep=strategy)),
            timeout=config.CRAWL_TIMEOUT_S * max_pages)
    except TimeoutError:
        log.warning("navigation of %s timed out", seed_url)
        return []
    except Exception as exc:
        log.warning("navigation of %s failed: %s", seed_url, str(exc)[:160])
        return []
    batch = results if isinstance(results, list) else [results]
    return [p for p in (page_from_result(r, via=via) for r in batch) if p]


async def _follow_ranked_links(crawler: Crawler, links, goal: str, seed_url: str, via: str,
                               max_pages: int, already: list[Page]) -> list[Page]:
    """The deep crawl did not spread: follow the highest-ranked links directly."""
    best = rank_links(links, goal, seed_url)[:max_pages - 1]
    if not best:
        return []
    log.info("deep crawl stayed on one page; following %d ranked link(s) from %s",
             len(best), domain_of(seed_url))
    seen = {norm_url(p.url) for p in already}
    extra: list[Page] = []
    for page in await crawler.fetch_batch(best, goal, browser=(via == "browser")):
        if norm_url(page.url) not in seen:
            page.depth = 1
            seen.add(norm_url(page.url))
            extra.append(page)
    return extra


async def discover_links(crawler: Crawler, seed_url: str) -> tuple[list, str]:
    """``(internal_links, engine)`` for the seed, escalating to a browser if HTTP sees none.

    The engine is chosen by whether it can see the site's navigation, not by how much text came
    back: rtionline.gov.in over HTTP fails outright (0 links), and crawl4ai's cache would replay
    an earlier browser fetch to an HTTP request and hide that. So this probe bypasses the cache,
    and the verdict is remembered per domain.
    """
    domain = domain_of(seed_url)
    if crawler.engine_by_domain.get(domain) == "browser":
        return await _links_from(crawler, seed_url, browser=True), "browser"
    internal = await _links_from(crawler, seed_url, browser=False)
    if len(internal) >= 3 or not config.CRAWL_USE_BROWSER:
        crawler.engine_by_domain.setdefault(domain, "http")
        return internal, "http"
    crawler.browser_escalations += 1
    log.info("%s exposes %d link(s) over plain HTTP; using a browser to see its navigation",
             domain, len(internal))
    crawler.engine_by_domain[domain] = "browser"
    return await _links_from(crawler, seed_url, browser=True), "browser"


async def _links_from(crawler: Crawler, seed_url: str, browser: bool) -> list:
    from crawl4ai import CacheMode, CrawlerRunConfig

    engine = await crawler.engine(browser)
    if engine is None:
        return []
    probe = CrawlerRunConfig(cache_mode=CacheMode.BYPASS,
                             page_timeout=int(config.CRAWL_TIMEOUT_S * 1000),
                             exclude_all_images=True, exclude_social_media_links=True,
                             verbose=False)
    try:
        result = await asyncio.wait_for(engine.arun(url=seed_url, config=probe),
                                        timeout=config.CRAWL_TIMEOUT_S + 10)
    except Exception as exc:
        log.debug("link probe (%s) failed: %s", "browser" if browser else "http", str(exc)[:120])
        return []
    if not getattr(result, "success", False):
        return []
    links = getattr(result, "links", None) or {}
    return links.get("internal", []) if isinstance(links, dict) else []


def rank_links(internal_links, goal: str, seed_url: str) -> list[str]:
    """Order a page's internal links by how much they look like the procedure we want."""
    domain = domain_of(seed_url)
    goal_words = re.findall(r"[a-z]{4,}", (goal or "").lower())[:8]
    seed_key = norm_url(seed_url)
    scored: list[tuple[float, str]] = []
    for link in internal_links:
        href = (link.get("href") or "") if isinstance(link, dict) else str(link)
        text = (link.get("text") or "") if isinstance(link, dict) else ""
        if (not href.startswith("http") or domain_of(href) != domain
                or norm_url(href) == seed_key or _BINARY.search(href)):
            continue
        haystack = f"{href} {text}".lower()
        score = sum(1.0 for word in PROCEDURE_KEYWORDS if word in haystack)
        score += sum(0.6 for word in goal_words if word in haystack)
        if _DEAD_END.search(haystack):       # login and contact pages explain no process
            score -= 1.5
        if score > 0:
            scored.append((score, href.split("#")[0]))
    scored.sort(key=lambda pair: -pair[0])
    ordered: list[str] = []
    for _, href in scored:
        if all(norm_url(href) != norm_url(existing) for existing in ordered):
            ordered.append(href)
    return ordered


def keywords_for(goal: str) -> list[str]:
    """Words worth steering the crawl towards: the question's own terms plus procedure cues."""
    words = re.findall(r"[a-zA-Z]{4,}", (goal or "").lower())[:8]
    return list(dict.fromkeys(words + list(PROCEDURE_KEYWORDS[:10])))
