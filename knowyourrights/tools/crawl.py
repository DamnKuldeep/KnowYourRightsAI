"""Reading web pages with crawl4ai.

A search snippet says a fee exists; the page says what the fee is. Two decisions shape this:

**HTTP first, browser only when needed.** Most ``gov.in`` pages are static HTML, and Chromium
costs 300-500 MB of memory. Pages are fetched over plain HTTP and escalated to a browser only
when HTTP came back too thin to be real content. The browser is reused, and closed when idle.

**Query-focused extraction at the source.** Every fetch runs a BM25 content filter keyed to the
user's question, so a page is cut down to its relevant parts before it enters the pipeline.

Following links inside a portal lives in :mod:`.navigate`; the page model in :mod:`.pages`.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .. import config
from ..evidence import domain_of
from ..runtime.cache import get_cache, key_of
from . import url_safety
from .pages import Page, norm_url, page_from_result, to_evidence

log = logging.getLogger(__name__)

__all__ = ["Crawler", "Page", "get_crawler", "to_evidence"]


def run_config(query: str, *, stream: bool = False, deep=None):
    """crawl4ai settings: query-focused when there is a question, structural pruning if not."""
    from crawl4ai import CacheMode, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    content_filter = (BM25ContentFilter(user_query=query[:400], bm25_threshold=1.0)
                      if query else
                      PruningContentFilter(threshold=0.45, threshold_type="dynamic",
                                           min_word_threshold=25))
    return CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
        cache_mode=CacheMode.ENABLED,
        check_robots_txt=config.CRAWL_RESPECT_ROBOTS,
        page_timeout=int(config.CRAWL_TIMEOUT_S * 1000),
        exclude_all_images=True,
        exclude_social_media_links=True,
        remove_overlay_elements=True,
        word_count_threshold=10,
        stream=stream,
        deep_crawl_strategy=deep,
        verbose=False,
    )


class Crawler:
    """One long-lived crawl4ai instance per engine. Starting a browser per call is the cost."""

    def __init__(self) -> None:
        self._http = None
        self._browser = None
        self._lock = asyncio.Lock()
        self._browser_last_used = 0.0
        # Which engine works per host, learned once and reused.
        self.engine_by_domain: dict[str, str] = {}
        self.pages_fetched = 0
        self.browser_escalations = 0
        self.failures = 0

    # ── engines ──────────────────────────────────────────────────────────────────────
    async def http_engine(self):
        """Browserless crawler: no Chromium, no 300 MB."""
        if self._http is None:
            async with self._lock:
                if self._http is None:
                    self._http = await _start_http_engine()
        return self._http

    async def browser_engine(self):
        """Chromium, for pages that need JavaScript. None when browsing is disabled."""
        if not config.CRAWL_USE_BROWSER:
            return None
        if self._browser is None:
            async with self._lock:
                if self._browser is None:
                    self._browser = await _start_browser_engine()
        self._browser_last_used = time.monotonic()
        return self._browser

    async def engine(self, browser: bool):
        return await (self.browser_engine() if browser else self.http_engine())

    async def close_browser_if_idle(self) -> bool:
        """Give the memory back when nobody is crawling JavaScript-heavy pages."""
        if self._browser is None:
            return False
        if time.monotonic() - self._browser_last_used < config.CRAWL_BROWSER_IDLE_S:
            return False
        async with self._lock:
            crawler, self._browser = self._browser, None
        try:
            await crawler.close()
            log.info("browser closed after the idle timeout")
            return True
        except Exception as exc:
            log.debug("browser close failed: %s", exc)
            return False

    async def aclose(self) -> None:
        for attr in ("_http", "_browser"):
            crawler = getattr(self, attr)
            setattr(self, attr, None)
            if crawler is not None:
                try:
                    await crawler.close()
                except Exception as exc:
                    log.debug("closing the %s crawler failed: %s", attr.strip("_"), exc)

    # ── fetching ─────────────────────────────────────────────────────────────────────
    async def fetch(self, urls: list[str], query: str = "", use_cache: bool = True) -> list[Page]:
        """Read pages, escalating individually to a browser only where HTTP came back thin."""
        wanted = [u for u in dict.fromkeys(urls) if u]
        urls = await url_safety.public_only(wanted)
        if not urls:
            return []
        cache = get_cache() if use_cache else None
        pages, todo = self._from_cache(urls, query, cache)
        if todo:
            fetched = await self._fetch_fresh(todo, query)
            fetched = [p for p in fetched if await url_safety.is_public_url(p.url)]
            self._store(fetched, query, cache)
            pages.extend(fetched)
            self.pages_fetched += len(fetched)
        return pages

    @staticmethod
    def _from_cache(urls: list[str], query: str, cache) -> tuple[list[Page], list[str]]:
        pages: list[Page] = []
        todo: list[str] = []
        for url in urls:
            cached = cache.get_json("crawl", key_of("page", url, query)) if cache else None
            if cached:
                pages.append(Page(**cached))
            else:
                todo.append(url)
        return pages, todo

    async def _fetch_fresh(self, urls: list[str], query: str) -> list[Page]:
        # Hosts already known to need a browser skip the doomed HTTP attempt entirely.
        known_browser = [u for u in urls if self.engine_by_domain.get(domain_of(u)) == "browser"]
        try_http = [u for u in urls if u not in known_browser]
        fetched = await self.fetch_batch(try_http, query, browser=False) if try_http else []

        got = {norm_url(p.url) for p in fetched}
        missing = [u for u in try_http if norm_url(u) not in got]
        for url in missing:
            self.engine_by_domain.setdefault(domain_of(url), "browser")
        escalate = list(dict.fromkeys(known_browser + [p.url for p in fetched if p.is_thin]
                                      + missing))
        if not escalate or not config.CRAWL_USE_BROWSER:
            return fetched
        return await self._escalate(escalate, fetched, query)

    async def _escalate(self, urls: list[str], fetched: list[Page], query: str) -> list[Page]:
        self.browser_escalations += len(urls)
        log.info("escalating %d page(s) to a browser (thin or unreachable over HTTP)", len(urls))
        by_url = {norm_url(p.url): p for p in fetched}
        for page in await self.fetch_batch(urls, query, browser=True):
            if not page.is_thin or norm_url(page.url) not in by_url:
                by_url[norm_url(page.url)] = page
        return list(by_url.values())

    @staticmethod
    def _store(pages: list[Page], query: str, cache) -> None:
        if cache is None:
            return
        for page in pages:
            if not page.is_thin:
                cache.set_json("crawl", key_of("page", page.url, query), page.__dict__,
                               ttl=config.CRAWL_CACHE_TTL)

    async def fetch_batch(self, urls: list[str], query: str, browser: bool) -> list[Page]:
        """Fetch a batch, keeping whatever has arrived when the time budget runs out.

        Waiting for the whole batch meant one slow page held every fast one hostage — and on a
        timeout, discarded them all. Streaming results as they complete makes a slow page cost
        only itself.
        """
        crawler = await self.engine(browser)
        if crawler is None or not urls:
            return []
        from crawl4ai import MemoryAdaptiveDispatcher

        via = "browser" if browser else "http"
        dispatcher = MemoryAdaptiveDispatcher(memory_threshold_percent=88.0,
                                              max_session_permit=config.CRAWL_MAX_CONCURRENT)
        pages: list[Page] = []
        try:
            async with asyncio.timeout(config.CRAWL_BATCH_BUDGET_S):
                stream = await crawler.arun_many(urls=urls, config=run_config(query, stream=True),
                                                 dispatcher=dispatcher)
                async for result in stream:
                    page = page_from_result(result, via=via)
                    if page:
                        pages.append(page)
        except TimeoutError:
            log.info("crawl budget of %.0fs reached: kept %d page(s), gave up on %d",
                     config.CRAWL_BATCH_BUDGET_S, len(pages), len(urls) - len(pages))
            self.failures += len(urls) - len(pages)
        except Exception as exc:
            log.warning("crawl failed (%s): %s — keeping %d page(s) already read",
                        via, str(exc)[:160], len(pages))
            self.failures += len(urls) - len(pages)
        return pages

    def status(self) -> dict:
        return {
            "pages_fetched": self.pages_fetched,
            "browser_escalations": self.browser_escalations,
            "failures": self.failures,
            "browser_running": self._browser is not None,
            "http_ready": self._http is not None,
        }


async def _start_http_engine():
    from crawl4ai import AsyncWebCrawler, BrowserConfig
    from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

    strategy = AsyncHTTPCrawlerStrategy(
        browser_config=BrowserConfig(headers={"User-Agent": config.CRAWL_USER_AGENT}),
        max_connections=config.CRAWL_MAX_CONCURRENT * 2,
    )
    crawler = AsyncWebCrawler(crawler_strategy=strategy, config=BrowserConfig(verbose=False))
    await crawler.start()
    log.info("http crawler ready (no browser)")
    return crawler


async def _start_browser_engine():
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    crawler = AsyncWebCrawler(config=BrowserConfig(
        headless=True, verbose=False, user_agent=config.CRAWL_USER_AGENT,
        java_script_enabled=True,
        extra_args=["--disable-dev-shm-usage", "--disable-gpu", "--no-sandbox", "--mute-audio"],
    ))
    await crawler.start()
    log.info("browser crawler started (adds ~300-500 MB of memory)")
    return crawler


_CRAWLER: Crawler | None = None


def get_crawler() -> Crawler:
    global _CRAWLER
    if _CRAWLER is None:
        _CRAWLER = Crawler()
    return _CRAWLER
