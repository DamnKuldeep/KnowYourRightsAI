"""Wikipedia: plain-language background, never the citation.

Useful for "what is an FIR" or "what does the Consumer Commission do", where the statute gives
the rule but not the concept. Capped at the summary extract, and its trust tier keeps it from
ever being cited as law.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .. import config
from ..evidence import Evidence
from ..runtime.cache import get_cache, key_of

log = logging.getLogger(__name__)

API = "https://en.wikipedia.org/w/api.php"
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
CACHE_TTL = 7 * 24 * 3600  # encyclopaedia summaries are not time-critical
EXTRACT_CHARS = 1400


async def lookup(query: str, n: int | None = None) -> list[Evidence]:
    """Summaries of the best-matching articles. Empty on any failure."""
    query = (query or "").strip()
    if not query:
        return []
    n = n or config.WIKI_MAX_RESULTS
    cache, cache_key = get_cache(), key_of("wiki", query, n)
    rows = cache.get_json("wiki", cache_key)
    if rows is None:
        try:
            rows = await _fetch(query, n)
        except Exception as exc:
            log.warning("wikipedia lookup failed for %r: %s", query[:60], str(exc)[:140])
            return []
        if rows:
            cache.set_json("wiki", cache_key, rows, ttl=CACHE_TTL)
    return [_to_evidence(r, query) for r in rows]


async def _fetch(query: str, n: int) -> list[dict]:
    headers = {"User-Agent": config.CRAWL_USER_AGENT}
    async with httpx.AsyncClient(timeout=config.WIKI_TIMEOUT, headers=headers) as client:
        response = await client.get(API, params={"action": "query", "list": "search",
                                                 "srsearch": query, "format": "json",
                                                 "srlimit": n})
        response.raise_for_status()
        titles = [h.get("title", "") for h in response.json().get("query", {})
                  .get("search", [])[:n]]
        summaries = await asyncio.gather(*(_summary(client, t) for t in titles))
    return [s for s in summaries if s]


async def _summary(client: httpx.AsyncClient, title: str) -> dict | None:
    slug = title.replace(" ", "_")
    try:
        response = await client.get(SUMMARY + slug)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    url = (data.get("content_urls", {}).get("desktop", {}).get("page")
           or f"https://en.wikipedia.org/wiki/{slug}")
    return {"title": title, "url": url, "text": extract[:EXTRACT_CHARS]}


def _to_evidence(row: dict, query: str) -> Evidence:
    return Evidence(kind="wikipedia", title=row["title"], text=row["text"], url=row["url"],
                    tier=config.TIER_WIKIPEDIA, score=0.25, query=query,
                    meta={"background_only": True})
