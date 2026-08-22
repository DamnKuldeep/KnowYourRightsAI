"""LanceDB access.

The corpus is 38,890 chunks across 35,170 sections. Holding all of that in a pandas frame
costs ~300 MB of RSS, which is a lot on a machine with under 3 GB free — and unnecessary,
because LanceDB answers by-id queries fast enough to do on demand (measured: 25 full rows in
0.14s, 25 stored vectors in 0.025s).

So we keep only a small **section index** resident — one row per section with just the
identifying fields, about 10 MB — and pull text and vectors per query.

Reading stored vectors back is also a correctness fix, not only a memory one. The notebook
re-encoded each candidate's ``chunk_text`` to compute MMR diversity, but the index holds
vectors of ``embed_text`` (heading + generated questions + keywords + chunk). Those are
different spaces, so diversity was being measured against vectors that were never searched.
"""

from __future__ import annotations

import functools
import logging
import re
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

from .. import config, legal_terms

log = logging.getLogger(__name__)


def pd_isna(value) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None

# Everything the answer layer needs about a section. `embed_text` is deliberately absent:
# it is a build-time artefact and must never be shown to a user (DB README §7).
DISPLAY_COLUMNS = [
    "chunk_id", "unit_id", "citation", "act_title", "act_year", "section_label",
    "section_name", "chapter", "category", "status", "effective_date",
    "source_type", "source_snapshot", "chunk_text", "full_text",
]

# The resident index: identity only, no text bodies.
INDEX_COLUMNS = [
    "unit_id", "act_title", "act_short_name", "act_year", "section_label", "section_num",
    "section_name", "category", "status", "citation", "source_type",
]


def sql_quote(value: str) -> str:
    """Single-quoted SQL literal with quotes doubled.

    Act titles genuinely contain apostrophes ("Farmers Rights", "Children's ...") and unit_ids
    embed the title, so naive interpolation would produce broken filters.
    """
    return "'" + str(value).replace("'", "''") + "'"


def sql_in(column: str, values: Sequence[str]) -> str:
    return f"{column} IN ({','.join(sql_quote(v) for v in values)})"


def normalize_title(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


_STOPWORDS = {"the", "of", "and", "act", "a", "an", "for", "to", "in", "india", "indian"}


@dataclass(frozen=True)
class ActMatch:
    act_title: str
    score: float
    sections: int


class LegalStore:
    """Thread-safe reader over the LanceDB table. Connects lazily."""

    def __init__(self, db_path=None, table: str | None = None) -> None:
        self.db_path = str(db_path or config.DB_PATH)
        self.table_name = table or config.TABLE
        self._table = None
        self._index = None
        # Reentrant: building the index needs the table, and both are guarded by this lock.
        self._lock = threading.RLock()
        self._fts_available = True

    # ── connection ───────────────────────────────────────────────────────────────────
    @property
    def table(self):
        if self._table is None:
            with self._lock:
                if self._table is None:
                    import lancedb

                    db = lancedb.connect(self.db_path)
                    self._table = db.open_table(self.table_name)
                    log.info("LanceDB open: %s rows at %s",
                             f"{self._table.count_rows():,}", self.db_path)
        return self._table

    @property
    def index(self):
        """One row per section. ~10 MB, built in about 0.2s."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    df = (self.table.search()
                          .select(INDEX_COLUMNS)
                          .limit(self.table.count_rows())
                          .to_pandas()
                          .drop_duplicates("unit_id")
                          .reset_index(drop=True))
                    df["_title_norm"] = df["act_title"].map(normalize_title)
                    self._index = df
                    mb = df.memory_usage(deep=True).sum() / 1e6
                    log.info("section index: %s sections, %.0f MB resident", f"{len(df):,}", mb)
        return self._index

    def count_rows(self) -> int:
        return self.table.count_rows()

    def stats(self) -> dict:
        idx = self.index
        return {
            "chunks": self.count_rows(),
            "sections": len(idx),
            "acts": int(idx["act_title"].nunique()),
            "index_mb": round(idx.memory_usage(deep=True).sum() / 1e6, 1),
            "fts_available": self._fts_available,
            "db_path": self.db_path,
        }

    # ── ranked retrieval ─────────────────────────────────────────────────────────────
    def dense(self, vector, k: int = 25, where: str | None = None) -> list[str]:
        """Vector search. Returns chunk_ids, best first."""
        try:
            query = self.table.search(vector).limit(k)
            if where:
                query = query.where(where, prefilter=True)
            return query.to_pandas()["chunk_id"].tolist()
        except Exception as exc:
            log.warning("dense search failed: %s", exc)
            return []

    def fts(self, text: str, k: int = 25, where: str | None = None) -> list[str]:
        """BM25 over `embed_text`. Needs no model and no network — our last-resort path."""
        if not self._fts_available or not text.strip():
            return []
        try:
            query = self.table.search(_fts_sanitize(text), query_type="fts").limit(k)
            if where:
                query = query.where(where)
            return query.to_pandas()["chunk_id"].tolist()
        except Exception as exc:
            # Distinguish "this query upset the parser" from "there is no FTS index at all".
            if "index" in str(exc).lower() and "fts" in str(exc).lower():
                self._fts_available = False
                log.warning("FTS index unavailable, disabling keyword search: %s", exc)
            else:
                log.debug("FTS query failed for %r: %s", text[:60], exc)
            return []

    # ── by-id access ─────────────────────────────────────────────────────────────────
    def rows(self, chunk_ids: Sequence[str], columns: Sequence[str] | None = None):
        """Full rows for the given chunk_ids, in LanceDB's order (not the caller's)."""
        import pandas as pd

        ids = [c for c in dict.fromkeys(chunk_ids) if c]
        if not ids:
            return pd.DataFrame(columns=list(columns or DISPLAY_COLUMNS))
        cols = list(columns or DISPLAY_COLUMNS)
        try:
            return (self.table.search()
                    .where(sql_in("chunk_id", ids))
                    .select(cols)
                    .limit(len(ids) + 8)
                    .to_pandas())
        except Exception as exc:
            log.warning("row fetch failed for %d ids: %s", len(ids), exc)
            return pd.DataFrame(columns=cols)

    def vectors(self, chunk_ids: Sequence[str]):
        """``{chunk_id: np.ndarray}`` straight from the index — the vectors actually searched."""
        import numpy as np

        ids = [c for c in dict.fromkeys(chunk_ids) if c]
        if not ids:
            return {}
        try:
            df = (self.table.search()
                  .where(sql_in("chunk_id", ids))
                  .select(["chunk_id", "vector"])
                  .limit(len(ids) + 8)
                  .to_pandas())
        except Exception as exc:
            log.warning("vector fetch failed: %s", exc)
            return {}
        return {row.chunk_id: np.asarray(row.vector, dtype="float32") for row in df.itertuples()}

    def chunks_of(self, unit_ids: Sequence[str], columns: Sequence[str] | None = None):
        """Every chunk of the given sections, ordered by section then chunk."""
        import pandas as pd

        ids = [u for u in dict.fromkeys(unit_ids) if u]
        if not ids:
            return pd.DataFrame(columns=list(columns or DISPLAY_COLUMNS))
        try:
            df = (self.table.search()
                  .where(sql_in("unit_id", ids))
                  .select(list(columns or DISPLAY_COLUMNS))
                  .limit(config.__dict__.get("MAX_CHUNKS_PER_SECTION", 25) * len(ids) + 32)
                  .to_pandas())
            return df.sort_values(["unit_id", "chunk_id"]).reset_index(drop=True)
        except Exception as exc:
            log.warning("chunk fetch failed: %s", exc)
            return pd.DataFrame(columns=list(columns or DISPLAY_COLUMNS))

    # ── exact lookup (no embedding, no rerank) ───────────────────────────────────────
    def find_acts(self, query: str, limit: int = 5) -> list[ActMatch]:
        """Resolve an act name to titles that exist in the corpus.

        Alias table first, token overlap second. Deliberately string matching rather than an
        embedding call: "RTI Act" and "Right to Information Act, 2005" must land on the same
        statute, and that is a vocabulary problem, not a semantic one.
        """
        idx = self.index
        query = (query or "").strip()
        if not query:
            return []

        # An explicit alias is an exact answer — don't let fuzzy scoring second-guess it.
        for title in legal_terms.detect_acts(query):
            if (idx["act_title"] == title).any():
                sections = int((idx["act_title"] == title).sum())
                others = [m for m in self._fuzzy_acts(query, limit) if m.act_title != title]
                # Score above any fuzzy result so callers that re-sort keep the alias on top
                # ("companies act" fuzzy-matches the 1956 Act more strongly than the 2013 one).
                return [ActMatch(title, 2.0, sections), *others][:limit]

        return self._fuzzy_acts(query, limit)

    def _fuzzy_acts(self, query: str, limit: int) -> list[ActMatch]:
        idx = self.index
        normalized = normalize_title(query)
        wanted = set(normalized.split()) - _STOPWORDS
        if not wanted:
            return []

        wants_amendment = "amendment" in wanted
        counts = idx.groupby("act_title").size()
        scored: list[ActMatch] = []
        titles = idx[["act_title", "_title_norm", "act_year"]].drop_duplicates("act_title")

        for title, norm, year in titles.itertuples(index=False):
            tokens = set(norm.split()) - _STOPWORDS
            if not tokens:
                continue
            overlap = len(wanted & tokens)
            if not overlap:
                continue
            # Cover the query first, then the title, then reward exact containment.
            score = overlap / len(wanted) * 0.7 + overlap / len(tokens) * 0.3
            if normalized and normalized in norm:
                score += 0.25
            # Amendment acts only amend; the principal act is what someone asking about
            # "the Maternity Benefit Act" actually wants.
            if "amendment" in tokens and not wants_amendment:
                score -= 0.30
            # Several statutes exist in an old and a current version (Consumer Protection
            # 1986 vs 2019). Nudge towards the current one when nothing else separates them.
            try:
                if year and not pd_isna(year):
                    score += min(0.05, max(0.0, (int(year) - 1950) / 1500))
            except (TypeError, ValueError):
                pass
            scored.append(ActMatch(title, round(score, 4), int(counts.get(title, 0))))

        scored.sort(key=lambda m: (-m.score, -m.sections, m.act_title))
        return scored[:limit]

    def lookup(self, act: str | None, section_label: str | None,
               constitution: bool = False):
        """Exact section text. Returns a DataFrame of the matching section's chunks.

        This is the path for "what does Article 21 say" — a question that deserves the actual
        provision, not the nearest neighbour of a fuzzy search.
        """
        import pandas as pd

        idx = self.index
        candidates = idx

        if constitution or (act and "constitution" in act.lower()):
            candidates = idx[idx["source_type"] == "constitution"]
        elif act:
            matches = self.find_acts(act, limit=1)
            if not matches:
                return pd.DataFrame(columns=DISPLAY_COLUMNS)
            candidates = idx[idx["act_title"] == matches[0].act_title]

        if section_label:
            wanted = str(section_label).strip().lstrip("0").upper() or "0"
            labels = candidates["section_label"].astype(str).str.strip().str.lstrip("0").str.upper()
            candidates = candidates[labels == wanted]

        if candidates.empty:
            return pd.DataFrame(columns=DISPLAY_COLUMNS)
        return self.chunks_of(candidates["unit_id"].tolist()[:4])

    def browse_act(self, act: str, limit: int = 400):
        """The section list for an act — a table of contents, from the resident index."""
        import pandas as pd

        matches = self.find_acts(act, limit=1)
        if not matches:
            return pd.DataFrame(columns=INDEX_COLUMNS), None
        title = matches[0].act_title
        rows = self.index[self.index["act_title"] == title].copy()
        rows = rows.sort_values("section_num", na_position="last").head(limit)
        return rows.drop(columns=["_title_norm"], errors="ignore"), title


# LanceDB's FTS parser treats these as syntax; a citizen's question is not a query language.
_FTS_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/]|&&|\|\|')


def _fts_sanitize(text: str) -> str:
    cleaned = _FTS_SPECIAL.sub(" ", str(text))
    return re.sub(r"\s+", " ", cleaned).strip()[:400]


@functools.lru_cache(maxsize=1)
def get_store() -> LegalStore:
    return LegalStore()
