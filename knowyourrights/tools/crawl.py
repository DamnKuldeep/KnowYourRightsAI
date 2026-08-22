"""Reading and navigating the web with crawl4ai.

A search snippet says a fee exists; it does not say what the fee *is*. Procedures live several
clicks inside a government portal — "apply" leads to "guidelines" leads to "fees" leads to
"appeal" — so answering "how do I file an RTI and what does it cost" means walking the site,
not reading one page.

Three decisions shape this module:

**HTTP first, browser only when needed.** Most ``gov.in`` and ``indiacode.nic.in`` pages are
static HTML. Chromium costs 300–500 MB of RSS, which is a lot on a machine with under 3 GB
free, so we fetch over plain HTTP and escalate to a browser only when the result comes back
too thin to be real content. The browser, once started, is reused and shut down when idle.

**Query-focused extraction at the source.** Every crawl runs a BM25 content filter keyed to
the user's question and reads ``fit_markdown``. Cutting a page down *before* it enters the
pipeline is the cheapest possible form of context management — free, and it happens once.

**Crawled text is untrusted.** A page can contain instructions aimed at the model. Content is
sanitised and delivered inside a labelled block; combined with the fact that the orchestration
is plain Python, a web page cannot cause a tool call no matter what it says.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from .. import config
from ..evidence import Evidence, domain_of, tier_for_url
from ..runtime.cache import get_cache, key_of

log = logging.getLogger(__name__)

# Link text/URLs that tend to lead towards the actual procedure.
PROCEDURE_KEYWORDS = (
    "apply", "application", "procedure", "how to", "guideline", "instruction", "faq",
    "form", "fee", "charges", "submit", "register", "grievance", "appeal", "complaint",
    "eligibility", "documents", "download", "status", "portal", "citizen", "service",
)
PROCEDURE_URL_PATTERNS = tuple(
    f"*{word}*" for word in
    ("apply", "application", "procedure", "how", "guideline", "faq", "form", "fee",
     "submit", "register", "grievance", "appeal", "complaint", "document", "service")
)

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


class Crawler:
    """One long-lived crawl4ai instance. Starting a browser per call is the dominant cost."""

    def __init__(self) -> None:
        self._http = None
        self._browser = None
        self._lock = asyncio.Lock()
        self._browser_last_used = 0.0
        # Which engine actually works per host, learned once and reused.
        self._engine_by_domain: dict[str, str] = {}
        self.pages_fetched = 0
        self.browser_escalations = 0
        self.failures = 0

    # ── lifecycle ────────────────────────────────────────────────────────────────────
    async def _get_http(self):
        """Browserless crawler: no Chromium, no 300 MB."""
        if self._http is None:
            async with self._lock:
                if self._http is None:
                    from crawl4ai import AsyncWebCrawler, BrowserConfig
                    from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

                    strategy = AsyncHTTPCrawlerStrategy(
                        browser_config=BrowserConfig(headers={"User-Agent": config.CRAWL_USER_AGENT}),
                        max_connections=config.CRAWL_MAX_CONCURRENT * 2,
                    )
                    crawler = AsyncWebCrawler(crawler_strategy=strategy, config=BrowserConfig(verbose=False))
                    await crawler.start()
                    self._http = crawler
                    log.info("http crawler ready (no browser)")
        return self._http

    async def _get_browser(self):
        if not config.CRAWL_USE_BROWSER:
            return None
        if self._browser is None:
            async with self._lock:
                if self._browser is None:
                    from crawl4ai import AsyncWebCrawler, BrowserConfig

                    crawler = AsyncWebCrawler(config=BrowserConfig(
                        headless=True, verbose=False,
                        user_agent=config.CRAWL_USER_AGENT,
                        java_script_enabled=True,
                        extra_args=["--disable-dev-shm-usage", "--disable-gpu",
                                    "--no-sandbox", "--mute-audio"],
                    ))
                    await crawler.start()
                    self._browser = crawler
                    log.info("browser crawler started (adds ~300-500 MB RSS)")
        self._browser_last_used = time.monotonic()
        return self._browser

    async def close_browser_if_idle(self) -> bool:
        """Give the memory back when nobody is crawling JS-heavy pages."""
        if self._browser is None:
            return False
        if time.monotonic() - self._browser_last_used < config.CRAWL_BROWSER_IDLE_S:
            return False
        async with self._lock:
            crawler, self._browser = self._browser, None
        try:
            await crawler.close()
            log.info("browser closed after idle timeout, RSS returned")
            return True
        except Exception as exc:
            log.debug("browser close failed: %s", exc)
            return False

    async def aclose(self) -> None:
        for attr in ("_http", "_browser"):
            crawler = getattr(self, attr)
            if crawler is not None:
                try:
                    await crawler.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    # ── configuration ────────────────────────────────────────────────────────────────
    @staticmethod
    def _run_config(query: str, *, stream: bool = False, deep=None):
        from crawl4ai import CacheMode, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        # Query-focused when we have a question, structural pruning when we don't.
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

    # ── fetching ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_page(result, depth: int = 0, via: str = "http") -> Page | None:
        if not getattr(result, "success", False):
            return None
        markdown = ""
        md = getattr(result, "markdown", None)
        if md is not None:
            markdown = (getattr(md, "fit_markdown", "") or getattr(md, "raw_markdown", "")
                        or str(md))
        text, suspicious = sanitize(markdown)
        if not text:
            return None
        meta = getattr(result, "metadata", None) or {}
        depth = int((meta or {}).get("depth", depth) or depth)
        return Page(
            url=getattr(result, "url", ""),
            title=(meta.get("title") or "").strip() or getattr(result, "url", ""),
            markdown=text,
            depth=depth,
            score=float((meta or {}).get("score", 0.0) or 0.0),
            suspicious=suspicious,
            via=via,
        )

    async def fetch(self, urls: list[str], query: str = "",
                    use_cache: bool = True) -> list[Page]:
        """Read pages, escalating individually to a browser only where HTTP came back thin."""
        urls = [u for u in dict.fromkeys(urls) if u and u.startswith(("http://", "https://"))]
        if not urls:
            return []

        cache = get_cache() if use_cache else None
        pages: list[Page] = []
        todo: list[str] = []
        for url in urls:
            cached = cache.get_json("crawl", key_of("page", url, query)) if cache else None
            if cached:
                pages.append(Page(**cached))
            else:
                todo.append(url)

        if todo:
            # Hosts already known to need a browser skip the doomed HTTP attempt entirely.
            known_browser = [u for u in todo
                             if self._engine_by_domain.get(domain_of(u)) == "browser"]
            try_http = [u for u in todo if u not in known_browser]

            fetched = await self._fetch_via(try_http, query, browser=False) if try_http else []
            thin = [p.url for p in fetched if p.is_thin]
            missing = [u for u in try_http
                       if not any(_norm_url(p.url) == _norm_url(u) for p in fetched)]
            for url in missing:
                self._engine_by_domain.setdefault(domain_of(url), "browser")
            escalate = list(dict.fromkeys(known_browser + thin + missing))

            if escalate and config.CRAWL_USE_BROWSER:
                self.browser_escalations += len(escalate)
                log.info("escalating %d page(s) to a browser (thin or unreachable over HTTP)",
                         len(escalate))
                richer = await self._fetch_via(escalate, query, browser=True)
                by_url = {_norm_url(p.url): p for p in fetched}
                for page in richer:
                    if not page.is_thin or _norm_url(page.url) not in by_url:
                        by_url[_norm_url(page.url)] = page
                fetched = list(by_url.values())

            for page in fetched:
                if cache is not None and not page.is_thin:
                    cache.set_json("crawl", key_of("page", page.url, query),
                                   page.__dict__, ttl=config.CRAWL_CACHE_TTL)
            pages.extend(fetched)
            self.pages_fetched += len(fetched)

        return pages

    async def _fetch_via(self, urls: list[str], query: str, browser: bool) -> list[Page]:
        crawler = await (self._get_browser() if browser else self._get_http())
        if crawler is None:
            return []
        run_config = self._run_config(query)
        try:
            from crawl4ai import MemoryAdaptiveDispatcher

            dispatcher = MemoryAdaptiveDispatcher(
                memory_threshold_percent=88.0,
                max_session_permit=config.CRAWL_MAX_CONCURRENT,
            )
            results = await asyncio.wait_for(
                crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher),
                timeout=config.CRAWL_TIMEOUT_S * 2 + 20,
            )
        except asyncio.TimeoutError:
            self.failures += len(urls)
            log.warning("crawl timed out for %d url(s)", len(urls))
            return []
        except Exception as exc:
            self.failures += len(urls)
            log.warning("crawl failed (%s): %s", "browser" if browser else "http", str(exc)[:160])
            return []

        via = "browser" if browser else "http"
        pages = []
        for result in results or []:
            page = self._to_page(result, via=via)
            if page:
                pages.append(page)
        return pages

    # ── navigation ───────────────────────────────────────────────────────────────────
    async def navigate(self, seed_url: str, goal: str, *, max_pages: int = 6,
                       max_depth: int = 2, should_cancel=None) -> list[Page]:
        """Walk a site towards an answer, following the most promising links.

        Best-first rather than breadth-first: a scorer ranks candidate links by how much their
        URL looks like a procedure page, so the crawl spends its budget on "apply" and "fees"
        instead of "about us" and "press releases".
        """
        if not seed_url:
            return []
        try:
            from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
            from crawl4ai.deep_crawling.filters import (
                ContentTypeFilter, DomainFilter, FilterChain, URLPatternFilter,
            )
            from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
        except ImportError as exc:
            log.warning("deep crawling unavailable (%s); reading the seed page only", exc)
            return await self.fetch([seed_url], goal)

        keywords = _keywords_for(goal)
        domain = domain_of(seed_url)
        # Only *hard* constraints belong in the filter chain — the chain is applied to the
        # start URL as well, so a URL-pattern filter for "*apply*" would reject a portal
        # homepage and end the crawl at depth 0 before a single link was followed. Preference
        # for procedure-shaped links is the scorer's job, which ranks rather than rejects.
        filters = [ContentTypeFilter(allowed_types=["text/html"])]
        if domain:
            filters.append(DomainFilter(allowed_domains=[domain]))

        strategy = BestFirstCrawlingStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=False,
            filter_chain=FilterChain(filters),
            url_scorer=KeywordRelevanceScorer(keywords=keywords, weight=0.8),
        )
        if should_cancel is not None:
            try:
                strategy.should_cancel = should_cancel
            except Exception:
                pass

        # Choose the engine by whether it can actually see the site's navigation, not by how
        # much text came back. Content length is a misleading signal here: crawl4ai's cache
        # happily serves an HTTP request from an earlier browser fetch, so a page can look
        # healthy while the HTTP engine is in fact failing outright — which is exactly what
        # rtionline.gov.in does (HTTP: success=False, 0 links; browser: 15 links).
        links, via = await self._discover_links(seed_url, goal)
        crawler = await (self._get_browser() if via == "browser" else self._get_http())
        if crawler is None:
            crawler, via = await self._get_http(), "http"

        run_config = self._run_config(goal, deep=strategy)
        pages: list[Page] = []
        try:
            results = await asyncio.wait_for(
                crawler.arun(url=seed_url, config=run_config),
                timeout=config.CRAWL_TIMEOUT_S * max_pages,
            )
            for result in (results if isinstance(results, list) else [results]):
                page = self._to_page(result, via=via)
                if page:
                    pages.append(page)
        except asyncio.TimeoutError:
            log.warning("navigation of %s timed out after %d page(s)", seed_url, len(pages))
        except Exception as exc:
            log.warning("navigation of %s failed: %s", seed_url, str(exc)[:160])

        seen = {_norm_url(p.url) for p in pages}
        if len({_norm_url(p.url) for p in pages}) <= 1 and links:
            # The strategy did not spread. Follow the highest-scoring links ourselves — a
            # plain, predictable backstop, needed often enough on Indian government portals
            # that it is a normal path rather than an edge case.
            best = _rank_links(links, goal, seed_url)[:max_pages - 1]
            if best:
                log.info("deep crawl stayed on one page; following %d ranked link(s) from %s",
                         len(best), domain)
                for page in await self._fetch_via(best, goal, browser=(via == "browser")):
                    if _norm_url(page.url) not in seen:
                        page.depth = 1
                        seen.add(_norm_url(page.url))
                        pages.append(page)

        if not pages:
            pages = await self.fetch([seed_url], goal)

        # Collapse near-duplicates (trailing slash, index.php) that add tokens but no content.
        unique: dict[str, Page] = {}
        for page in pages:
            key = _norm_url(page.url)
            if key not in unique or len(page.markdown) > len(unique[key].markdown):
                unique[key] = page
        pages = list(unique.values())

        self.pages_fetched += len(pages)
        return pages

    async def _discover_links(self, seed_url: str, goal: str) -> tuple[list, str]:
        """Return ``(internal_links, engine)`` for the seed, escalating if HTTP sees nothing.

        The probe **bypasses crawl4ai's cache**. That is essential rather than fussy: the
        cache is keyed by URL and not by engine, so a page previously fetched with the browser
        is happily replayed to an HTTP request. Reading it would tell us HTTP works on a site
        where it does not, and every subsequent link fetch would then fail. The verdict is
        remembered per domain so we pay for this probe once.
        """
        from crawl4ai import CacheMode, CrawlerRunConfig

        domain = domain_of(seed_url)
        probe_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, page_timeout=int(config.CRAWL_TIMEOUT_S * 1000),
            exclude_all_images=True, exclude_social_media_links=True, verbose=False,
        )

        async def links_from(browser: bool) -> list:
            crawler = await (self._get_browser() if browser else self._get_http())
            if crawler is None:
                return []
            try:
                result = await asyncio.wait_for(
                    crawler.arun(url=seed_url, config=probe_config),
                    timeout=config.CRAWL_TIMEOUT_S + 10)
            except Exception as exc:
                log.debug("link probe (%s) failed: %s",
                          "browser" if browser else "http", str(exc)[:120])
                return []
            if not getattr(result, "success", False):
                return []
            links = getattr(result, "links", None) or {}
            return links.get("internal", []) if isinstance(links, dict) else []

        known = self._engine_by_domain.get(domain)
        if known == "browser":
            return await links_from(browser=True), "browser"

        internal = await links_from(browser=False)
        if len(internal) >= 3:
            self._engine_by_domain[domain] = "http"
            return internal, "http"
        if not config.CRAWL_USE_BROWSER:
            return internal, "http"

        self.browser_escalations += 1
        log.info("%s exposes %d link(s) over plain HTTP; using a browser to see its navigation",
                 domain, len(internal))
        self._engine_by_domain[domain] = "browser"
        return await links_from(browser=True), "browser"

    def status(self) -> dict:
        return {
            "pages_fetched": self.pages_fetched,
            "browser_escalations": self.browser_escalations,
            "failures": self.failures,
            "browser_running": self._browser is not None,
            "http_ready": self._http is not None,
        }


def _norm_url(url: str) -> str:
    """Collapse the spellings of one page: trailing slash, fragment, bare index file."""
    cleaned = (url or "").split("#")[0].rstrip("/")
    for suffix in ("/index.php", "/index.html", "/index.htm"):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned.lower()


def _rank_links(internal_links, goal: str, seed_url: str) -> list[str]:
    """Order a page's internal links by how much they look like the procedure we want."""
    domain = domain_of(seed_url)
    goal_words = [w for w in re.findall(r"[a-z]{4,}", (goal or "").lower())][:8]
    seed_key = _norm_url(seed_url)
    scored: list[tuple[float, str]] = []

    for link in internal_links:
        href = (link.get("href") or "") if isinstance(link, dict) else str(link)
        text = (link.get("text") or "") if isinstance(link, dict) else ""
        if not href.startswith("http") or domain_of(href) != domain:
            continue
        if _norm_url(href) == seed_key:
            continue
        if re.search(r"\.(pdf|jpe?g|png|gif|zip|docx?|xlsx?)$", href, re.I):
            continue
        haystack = f"{href} {text}".lower()
        score = sum(1.0 for word in PROCEDURE_KEYWORDS if word in haystack)
        score += sum(0.6 for word in goal_words if word in haystack)
        # Login and status-check pages are dead ends for someone asking how a process works.
        if re.search(r"login|signin|register\.php|logout|contact", haystack):
            score -= 1.5
        if score > 0:
            scored.append((score, href.split("#")[0]))

    scored.sort(key=lambda pair: -pair[0])
    ordered: list[str] = []
    for _, href in scored:
        if not any(_norm_url(href) == _norm_url(existing) for existing in ordered):
            ordered.append(href)
    return ordered


def _keywords_for(goal: str) -> list[str]:
    """Words worth steering the crawl towards: the question's own terms plus procedure cues."""
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", (goal or "").lower())][:8]
    return list(dict.fromkeys(words + list(PROCEDURE_KEYWORDS[:10])))


def to_evidence(pages: list[Page], query: str, max_chars: int | None = None) -> list[Evidence]:
    """Turn crawled pages into evidence, tiered by host."""
    max_chars = max_chars or config.WEB_TEXT_CAP * 3
    items: list[Evidence] = []
    for page in pages:
        if not page.markdown.strip():
            continue
        tier = tier_for_url(page.url)
        items.append(Evidence(
            kind="official" if tier >= config.TIER_OFFICIAL else "web",
            title=page.title or page.url,
            text=page.markdown[:max_chars],
            url=page.url,
            tier=tier,
            # A read page is much stronger evidence than a search snippet.
            score=0.62 if tier >= config.TIER_OFFICIAL else 0.45,
            query=query,
            meta={"crawled": True, "depth": page.depth, "via": page.via,
                  "suspicious": page.suspicious, "chars": len(page.markdown)},
        ))
    return items


_CRAWLER: Crawler | None = None


def get_crawler() -> Crawler:
    global _CRAWLER
    if _CRAWLER is None:
        _CRAWLER = Crawler()
    return _CRAWLER
