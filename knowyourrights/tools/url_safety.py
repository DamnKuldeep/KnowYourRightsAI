"""Which URLs the crawler may fetch: public http(s) addresses only.

The crawler reads URLs that come from search results, from links on those pages and from the
planner, which copies URLs out of the user's own message. Unchecked, "read
http://169.254.169.254/latest/meta-data" would fetch a cloud instance's credentials and hand
them to the writer, and "http://localhost:8000/api/status" would read the app's own internals.
So every URL is resolved first, and it is refused unless *every* address it resolves to is a
public one. A page is checked again by its final URL after redirects, before its text is used.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import time
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_BLOCKED_HOSTS = {"localhost", "localhost.localdomain", "metadata.google.internal"}
# Resolutions are cached briefly: a research round fetches several pages from one host.
_CACHE_TTL_S = 300.0
_cache: dict[str, tuple[float, bool]] = {}


def _is_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _host_of(url: str) -> str | None:
    """The host of an http(s) URL, or None if the URL is not one we would ever fetch."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not host or host in _BLOCKED_HOSTS:
        return None
    if host.endswith((".localhost", ".internal", ".local")):
        return None
    return host


async def _resolves_publicly(host: str) -> bool:
    cached = _cache.get(host)
    if cached is not None and time.monotonic() - cached[0] < _CACHE_TTL_S:
        return cached[1]
    try:
        ipaddress.ip_address(host)
        verdict = _is_public_ip(host)
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, None, type=socket.SOCK_STREAM)
            addresses = {info[4][0] for info in infos}
            verdict = bool(addresses) and all(_is_public_ip(a) for a in addresses)
        except (OSError, UnicodeError):
            verdict = False
    _cache[host] = (time.monotonic(), verdict)
    return verdict


async def is_public_url(url: str) -> bool:
    """True when ``url`` is http(s) and its host resolves only to public addresses."""
    host = _host_of(url)
    if host is None:
        return False
    if not await _resolves_publicly(host):
        log.warning("refusing to fetch %s: it does not resolve to a public address", url[:120])
        return False
    return True


async def public_only(urls: list[str]) -> list[str]:
    """The subset of ``urls`` that is safe to fetch, in the original order."""
    verdicts = await asyncio.gather(*(is_public_url(u) for u in urls))
    return [u for u, ok in zip(urls, verdicts, strict=True) if ok]
