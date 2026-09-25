"""Maintain the corpus: build its vector index, and remove superseded versions.

The corpus ships with a BM25 index. Without a vector index every semantic query scans the whole
table (~159 MB read, 175-300 ms); an HNSW-over-IVF index with scalar quantisation takes that to
about 23 ms with no loss of quality on the gold set, costs ~42 MB and ~25 s to build.

Every change to a LanceDB table (a repair, an index rebuild) writes a new version and keeps the
old files. Two ways to clear them:

* ``--prune`` deletes only files the current version no longer uses, and writes nothing. Use it
  before committing: it shrinks a clone without adding anything to Git LFS. Needs ``pylance``.
* ``--compact`` also merges fragments, which rewrites the data files (~350 MB of new files, all
  uploaded again to Git LFS). Worth it only after many small edits.

    python scripts/build_index.py              # build the vector index if it is missing
    python scripts/build_index.py --check      # report what exists, change nothing
    python scripts/build_index.py --rebuild    # replace the vector index
    python scripts/build_index.py --prune      # delete superseded versions (pip install pylance)
    python scripts/build_index.py --compact    # merge fragments too; rewrites the data files
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config
from knowyourrights.runtime.console import bold, rule, setup_console

setup_console()

# 64 partitions for ~39k rows keeps each around 600 vectors: enough for HNSW to search well
# inside one, few enough that probing stays cheap.
NUM_PARTITIONS = 64
INDEX_TYPE = "IVF_HNSW_SQ"
METRIC = "cosine"        # the corpus vectors are L2-normalised


def open_table():
    import lancedb

    if not config.DB_PATH.exists():
        raise SystemExit(f"No database at {config.DB_PATH}. Clone with Git LFS (git lfs pull) "
                         f"or build it with notebooks/01-building-database.ipynb.")
    table = lancedb.connect(str(config.DB_PATH)).open_table(config.TABLE)
    print(f"  {table.count_rows():,} rows at {config.DB_PATH}")
    return table


def report_indices(table) -> bool:
    """Print the indices; True if a vector index exists."""
    existing = list(table.list_indices())
    for index in existing or ["(none)"]:
        print(f"  {index}")
    has_vector = any("vector" in str(i) for i in existing)
    has_fts = any("FTS" in str(i) for i in existing)
    print(f"\n  full-text (BM25): {'present' if has_fts else 'MISSING: keyword search is off'}")
    print(f"  vector (ANN)    : {'present' if has_vector else 'MISSING: every query scans'}")
    return has_vector


def build(table) -> None:
    print(f"  {INDEX_TYPE}, {NUM_PARTITIONS} partitions, metric={METRIC}")
    started = time.time()
    table.create_index(metric=METRIC, index_type=INDEX_TYPE,
                       num_partitions=NUM_PARTITIONS, replace=True)
    print(f"  built in {time.time() - started:.1f}s")


def prune(table) -> None:
    try:
        stats = table.cleanup_old_versions(older_than=timedelta(0), delete_unverified=True)
    except ImportError as exc:
        raise SystemExit("--prune needs the lance library: pip install pylance") from exc
    print(f"  removed {stats.bytes_removed / 1e6:.1f} MB of superseded versions")


def compact(table) -> None:
    before = table.version
    table.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)
    print(f"  compacted: version {before} -> {table.version}; superseded files removed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report existing indices only")
    ap.add_argument("--rebuild", action="store_true", help="replace the vector index")
    ap.add_argument("--prune", action="store_true", help="delete superseded versions only")
    ap.add_argument("--compact", action="store_true", help="merge fragments; rewrites data")
    args = ap.parse_args()

    rule("corpus")
    table = open_table()
    rule("indices")
    has_vector = report_indices(table)
    if args.check:
        return 0
    if not has_vector or args.rebuild:
        rule("building")
        build(table)
    if args.prune:
        rule("pruning")
        prune(table)
    if args.compact:
        rule("compacting")
        compact(table)
    print(f"\n  Verify with: {bold('python scripts/evaluate.py')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
