"""The single currency every tool produces and the writer consumes.

Retrieval, Wikipedia and the web all return different shapes; the pipeline needs one. An
``Evidence`` also carries the two things a legal answer depends on and prose alone cannot
convey: **where it came from** (trust tier) and **whether it is still good law** (status,
effective date, jurisdiction).

Ids are stable and human-readable — ``S1`` for a statute, ``W2`` for the web — because the
writer cites them inline and the UI turns them into clickable chips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import config

KIND_PREFIX = {
    "statute": "S",
    "official": "G",
    "web": "W",
    "wikipedia": "K",
    "procedure": "P",
}

TIER_LABEL = {
    config.TIER_STATUTE: "statute",
    config.TIER_OFFICIAL: "official",
    config.TIER_LEGAL_PORTAL: "legal portal",
    config.TIER_WIKIPEDIA: "background",
    config.TIER_WEB: "web",
}


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def tier_for_url(url: str) -> int:
    """Trust tier from the host. A gov.in page outranks a blog, and both outrank nothing."""
    host = domain_of(url)
    if not host:
        return config.TIER_WEB
    if any(host == d or host.endswith("." + d) for d in config.OFFICIAL_DOMAINS):
        return config.TIER_OFFICIAL
    if any(host == d or host.endswith("." + d) for d in config.LEGAL_PORTAL_DOMAINS):
        return config.TIER_LEGAL_PORTAL
    if host.endswith("wikipedia.org"):
        return config.TIER_WIKIPEDIA
    return config.TIER_WEB


@dataclass
class Evidence:
    kind: str                      # statute | official | web | wikipedia | procedure
    title: str
    text: str
    url: str = ""
    tier: int = config.TIER_WEB
    score: float = 0.0
    id: str = ""
    query: str = ""                # the search that produced it — useful for the trace
    # statute-only provenance
    citation: str = ""
    act_title: str = ""
    unit_id: str = ""
    source_type: str = ""        # constitution | central_act | criminal_code
    status: str = ""
    effective_date: str = ""
    source_snapshot: str = ""
    state: str | None = None
    category: str = ""
    # grading
    relevant: bool | None = None
    grade_note: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def tier_label(self) -> str:
        return TIER_LABEL.get(self.tier, "web")

    @property
    def is_statute(self) -> bool:
        return self.kind == "statute"

    @property
    def is_omitted(self) -> bool:
        return (self.status or "").lower() == "omitted"

    @property
    def jurisdiction(self) -> str:
        """CENTRAL, STATE, CONSTITUTION or empty — never a guess.

        Read from the Act's own title, because the corpus's ``jurisdiction`` column says
        ``central`` for every row including the state Acts that leaked into it (DB README §9).
        Getting this wrong is the worst kind of error this system can make: telling someone in
        Kerala that a Maharashtra rent law governs them is worse than telling them nothing.
        """
        if not self.is_statute:
            return ""
        if self.source_type == "constitution":
            return "CONSTITUTION"
        return "STATE" if self.state else "CENTRAL"

    @property
    def jurisdiction_label(self) -> str:
        """A phrase safe to show a non-lawyer."""
        if self.jurisdiction == "CONSTITUTION":
            return "Constitution of India — applies nationwide"
        if self.jurisdiction == "STATE":
            return f"{self.state} state law — applies only in {self.state}"
        if self.jurisdiction == "CENTRAL":
            return "Central law — applies across India"
        return ""

    def applies_in(self, user_state: str | None) -> bool | None:
        """Does this apply where the user is? ``None`` when we cannot tell."""
        if self.jurisdiction in ("CENTRAL", "CONSTITUTION"):
            return True
        if self.jurisdiction == "STATE":
            if not user_state:
                return None
            return user_state.strip().lower() == (self.state or "").strip().lower()
        return None

    @property
    def domain(self) -> str:
        return domain_of(self.url)

    def dedupe_key(self) -> tuple[str, str]:
        """Sections de-duplicate by unit_id, pages by URL."""
        if self.is_statute and self.unit_id:
            return ("statute", self.unit_id)
        if self.url:
            return ("url", self.url.split("#")[0].rstrip("/"))
        return ("title", f"{self.kind}:{self.title}".lower())

    def label(self) -> str:
        """What the writer sees as the source's name."""
        return self.citation or self.title or self.url or self.id

    def to_public(self) -> dict:
        """The shape the UI receives. Never includes raw prompt scaffolding."""
        return {
            "id": self.id,
            "kind": self.kind,
            "tier": self.tier,
            "tier_label": self.tier_label,
            "title": self.label(),
            "url": self.url,
            "domain": self.domain,
            "snippet": self.text[:320].strip(),
            "score": round(self.score, 3),
            "citation": self.citation,
            "status": self.status,
            "effective_date": self.effective_date,
            "state": self.state,
            "jurisdiction": self.jurisdiction,
            "jurisdiction_label": self.jurisdiction_label,
            "category": self.category,
            "source_snapshot": self.source_snapshot,
        }


def assign_ids(items: list[Evidence]) -> list[Evidence]:
    """Give every item a stable, readable id, grouped by kind."""
    counters: dict[str, int] = {}
    for item in items:
        prefix = KIND_PREFIX.get(item.kind, "E")
        counters[prefix] = counters.get(prefix, 0) + 1
        item.id = f"{prefix}{counters[prefix]}"
    return items


def dedupe(items: list[Evidence]) -> list[Evidence]:
    """First occurrence wins; later duplicates only contribute a better score."""
    seen: dict[tuple[str, str], Evidence] = {}
    ordered: list[Evidence] = []
    for item in items:
        key = item.dedupe_key()
        existing = seen.get(key)
        if existing is None:
            seen[key] = item
            ordered.append(item)
        elif item.score > existing.score:
            existing.score = item.score
    return ordered


def from_hit(hit, query: str = "") -> Evidence:
    """Convert a retrieval :class:`~knowyourrights.retrieval.search.Hit`."""
    return Evidence(
        kind="statute",
        title=hit.citation,
        text=hit.full_text or hit.chunk_text,
        tier=config.TIER_STATUTE,
        score=hit.score,
        query=query,
        citation=hit.citation,
        act_title=hit.act_title,
        unit_id=hit.unit_id,
        source_type=hit.source_type,
        status=hit.status,
        effective_date=hit.effective_date,
        source_snapshot=hit.source_snapshot,
        state=hit.state,
        category=hit.category,
    )
