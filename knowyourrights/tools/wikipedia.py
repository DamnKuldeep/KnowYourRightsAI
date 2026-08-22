"""Wikipedia — plain-language background, never the citation.

Useful for "what *is* an FIR" or "what does the Consumer Commission do", where the statute
gives the rule but not the concept. Deliberately capped at the summary extract: this is
scaffolding for the explanation, and the trust tier keeps it from being cited as law.
"""

from __future__ import annotations

import asyncio
import logging

from .. import config
from ..evidence import Evidence
from ..runtime.cache import get_cache, key_of

log = logging.getLogger(__name__)

API = "https://en.wikipedia.org/w/api.php"
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
HEADERS = {"User-Agent": config.CRAWL_USER_AGENT}
CACHE_TTL = 7 * 24 * 3600  # encyclopaedia summaries are not time-critical


async def lookup(query: str, n: int | None = None) -> list[Evidence]:
    query = (query or "").strip()
    if not query:
        return []
    n = n or config.WIKI_MAX_RESULTS

    cache = get_cache()
    cache_key = key_of("wiki", query, n)
    hit = cache.get_json("wiki", cache_key)
    if hit is not None:
        return [_to_evidence(r, query) for r in hit]

    try:
        import httpx

        async with httpx.AsyncClient(timeout=config.WIKI_TIMEOUT, headers=HEADERS) as client:
            response = await client.get(API, params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": n,
            })
            response.raise_for_status()
            titles = [h.get("title", "") for h in
                      response.json().get("query", {}).get("search", [])[:n]]

            async def summary(title: str) -> dict | None:
                slug = title.replace(" ", "_")
                try:
                    resp = await client.get(SUMMARY + slug)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    return None
                extract = (data.get("extract") or "").strip()
                if not extract:
                    return None
                url = (data.get("content_urls", {}).get("desktop", {}).get("page")
                       or f"https://en.wikipedia.org/wiki/{slug}")
                return {"title": title, "url": url, "text": extract[:1400]}

            rows = [r for r in await asyncio.gather(*(summary(t) for t in titles)) if r]
    except Exception as exc:
        log.warning("wikipedia lookup failed for %r: %s", query[:60], str(exc)[:140])
        return []

    if rows:
        cache.set_json("wiki", cache_key, rows, ttl=CACHE_TTL)
    return [_to_evidence(r, query) for r in rows]


def _to_evidence(row: dict, query: str) -> Evidence:
    return Evidence(
        kind="wikipedia",
        title=row["title"],
        text=row["text"],
        url=row["url"],
        tier=config.TIER_WIKIPEDIA,
        score=0.25,
        query=query,
        meta={"background_only": True},
    )
