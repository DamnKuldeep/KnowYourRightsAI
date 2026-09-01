"""Build the vector index on the corpus.

The bundle produced by ``notebooks/01-building-database.ipynb`` ships with a BM25 index but no
vector index, so every dense query brute-force scans the whole table — 38,890 x 1024 float32,
about 159 MB read per question. Measured cost of that: **175-300 ms per query**, roughly half
of total retrieval time.

An HNSW-over-IVF index with scalar quantisation takes that to **~23 ms** with no loss of
quality on the gold set (Recall@5 held at 97.6%, MRR went 0.837 -> 0.849). It costs ~42 MB on
disk and about 25 seconds to build.

Run once after building or downloading the corpus:

    python scripts/build_index.py
    python scripts/build_index.py --check     # report what exists, build nothing
    python scripts/build_index.py --rebuild   # replace an existing index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                                     # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()

# 64 partitions for ~39k rows keeps each partition around 600 vectors — enough for HNSW to
# search well inside one, few enough that probing stays cheap.
NUM_PARTITIONS = 64
INDEX_TYPE = "IVF_HNSW_SQ"
# The corpus vectors are L2-normalised, so cosine is the metric the embeddings were built for.
METRIC = "cosine"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report existing indices only")
    ap.add_argument("--rebuild", action="store_true", help="replace an existing vector index")
    args = ap.parse_args()

    import lancedb

    rule("corpus")
    if not config.DB_PATH.exists():
        print(f"  no database at {config.DB_PATH}")
        print("  build it with notebooks/01-building-database.ipynb, or clone with git-lfs.")
        return 1

    table = lancedb.connect(str(config.DB_PATH)).open_table(config.TABLE)
    print(f"  {table.count_rows():,} rows at {config.DB_PATH}")

    existing = list(table.list_indices())
    has_vector = any("vector" in str(i) for i in existing)
    has_fts = any("FTS" in str(i) for i in existing)

    rule("indices")
    for index in existing:
        print(f"  {index}")
    if not existing:
        print("  (none)")
    print(f"\n  full-text (BM25): {'present' if has_fts else 'MISSING — keyword search will be off'}")
    print(f"  vector (ANN)    : {'present' if has_vector else 'MISSING — dense search scans every row'}")

    if args.check:
        return 0
    if has_vector and not args.rebuild:
        print("\n  Vector index already present. Use --rebuild to replace it.")
        return 0

    rule("building")
    print(f"  {INDEX_TYPE}, {NUM_PARTITIONS} partitions, metric={METRIC}")
    started = time.time()
    table.create_index(metric=METRIC, index_type=INDEX_TYPE,
                       num_partitions=NUM_PARTITIONS, replace=True)
    print(f"  built in {time.time() - started:.1f}s")

    rule("done")
    for index in table.list_indices():
        print(f"  {index}")
    print(f"\n  Verify with: {bold('python scripts/benchmark.py --all')}")
    print("  Expect dense_search around 20-30 ms and Recall@5 unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
