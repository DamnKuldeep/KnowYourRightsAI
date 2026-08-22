"""Web discovery — finding *which pages* to read. Reading them is ``crawl.py``'s job.

The corpus is central statute as of a snapshot. Everything else a citizen needs — the current
fee, the portal to file on, the deadline, this year's amendment, a state's own rules — lives
on the open web. So discovery matters, and *which* part of the web matters more: a fee quoted
by a blog and a fee published on the department's own site are not equal evidence, and the
trust tier travels with the result so the writer and the UI can say which is which.

DuckDuckGo needs no key and is therefore the default, but it rate-limits aggressively; the
provider interface exists so a Tavily/Brave/Serper key can be dropped into ``.env`` later
without touching call sites.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from .. import config
from ..evidence import Evidence, tier_for_url
from ..runtime.cache import get_cache, key_of

log = logging.getLogger(__name__)

# Site-restricted passes for questions where officialness is the whole point.
OFFICIAL_SITE_FILTER = " OR ".join(f"site:{d}" for d in
                                   ("gov.in", "nic.in", "indiacode.nic.in"))


class _MinuteLimiter:
    """Politeness for a free endpoint that will block us if we lean on it."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            now = time.time()
            self._calls = [t for t in self._calls if now - t < 60]
            if len(self._calls) >= self.per_minute:
                return False
            self._calls.append(now)
            return True


_limiter = _MinuteLimiter(config.WEB_MAX_PER_MIN)


# ── providers ─────────────────────────────────────────────────────────────────────────
def _ddg_search(query: str, n: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS(timeout=int(config.WEB_TIMEOUT)) as client:
        raw = client.text(query, max_results=n, region="in-en", safesearch="moderate")
    return [
        {"title": r.get("title") or "", "url": r.get("href") or r.get("url") or "",
         "snippet": r.get("body") or ""}
        for r in (raw or [])
    ]


def _tavily_search(query: str, n: int) -> list[dict]:
    import httpx

    key = os.environ.get("TAVILY_API_KEY", "")
    resp = httpx.post("https://api.tavily.com/search", timeout=config.WEB_TIMEOUT,
                      json={"api_key": key, "query": query, "max_results": n,
                            "search_depth": "basic", "country": "india"})
    resp.raise_for_status()
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("content", "")} for r in resp.json().get("results", [])]


def _brave_search(query: str, n: int) -> list[dict]:
    import httpx

    key = os.environ.get("BRAVE_API_KEY", "")
    resp = httpx.get("https://api.search.brave.com/res/v1/web/search",
                     timeout=config.WEB_TIMEOUT,
                     headers={"X-Subscription-Token": key, "Accept": "application/json"},
                     params={"q": query, "count": n, "country": "IN"})
    resp.raise_for_status()
    results = resp.json().get("web", {}).get("results", [])
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("description", "")} for r in results]


def active_provider() -> tuple[str, callable]:
    """Prefer a keyed provider when one is configured; otherwise DuckDuckGo."""
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily", _tavily_search
    if os.environ.get("BRAVE_API_KEY"):
        return "brave", _brave_search
    return "ddg", _ddg_search


# ── public API ────────────────────────────────────────────────────────────────────────
async def search(query: str, *, n: int | None = None, official_only: bool = False,
                 use_cache: bool = True) -> list[Evidence]:
    """Search the web. Returns snippet-level evidence; deep reading happens in ``crawl``."""
    query = (query or "").strip()
    if not query:
        return []
    n = n or config.WEB_MAX_RESULTS
    if official_only:
        query = f"{query} ({OFFICIAL_SITE_FILTER})"

    name, provider = active_provider()
    cache = get_cache() if use_cache else None
    cache_key = key_of("web", name, query, n)

    if cache is not None:
        hit = cache.get_json("web", cache_key)
        if hit is not None:
            return [_to_evidence(r, query) for r in hit]

    if not await _limiter.allow():
        log.warning("web search skipped: %d/min budget spent", config.WEB_MAX_PER_MIN)
        return []

    loop = asyncio.get_running_loop()
    try:
        rows = await asyncio.wait_for(
            loop.run_in_executor(None, provider, query, n),
            timeout=config.WEB_TIMEOUT + 5,
        )
    except asyncio.TimeoutError:
        log.warning("web search timed out for %r", query[:60])
        return []
    except Exception as exc:
        log.warning("web search (%s) failed for %r: %s", name, query[:60], str(exc)[:140])
        return []

    rows = [r for r in rows if r.get("url")]
    if cache is not None and rows:
        cache.set_json("web", cache_key, rows, ttl=config.WEB_CACHE_TTL)
    return [_to_evidence(r, query) for r in rows]


async def search_official(query: str, n: int | None = None) -> list[Evidence]:
    """Government sources first, then a plain search if that came back thin.

    Two passes rather than one because ``site:`` filters are precise but brittle — some
    departments simply are not indexed under the obvious domain.
    """
    results = await search(query, n=n, official_only=True)
    if len(results) >= 2:
        return results
    fallback = await search(query, n=n)
    seen = {r.url for r in results}
    return results + [r for r in fallback if r.url not in seen]


def _to_evidence(row: dict, query: str) -> Evidence:
    url = row.get("url", "")
    tier = tier_for_url(url)
    return Evidence(
        kind="official" if tier >= config.TIER_OFFICIAL else "web",
        title=row.get("title", "") or url,
        text=(row.get("snippet") or "")[:900],
        url=url,
        tier=tier,
        query=query,
        # Snippets are a discovery signal, not evidence; crawling raises this.
        score=0.30 if tier >= config.TIER_OFFICIAL else 0.20,
        meta={"snippet_only": True},
    )
