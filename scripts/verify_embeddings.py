"""Check that the embedding API still produces the vectors the corpus was built with.

Semantic search compares a question's embedding with vectors computed when the corpus was built.
That only works while the API serves the same model: if OpenRouter's ``baai/bge-m3`` ever changed,
search would quietly get worse rather than fail. This re-embeds a sample of stored rows and
compares them with their stored vectors.

    python scripts/verify_embeddings.py            # 20 random rows
    python scripts/verify_embeddings.py --rows 50

Healthy: every matching row at cosine ~1.000, unrelated pairs far lower. Costs a fraction of a
cent. Needs OPENROUTER_API_KEY and the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config
from knowyourrights.llm import retrieval_api
from knowyourrights.runtime.console import bold, rule, setup_console

setup_console()

MATCH_FLOOR = 0.99       # a row embedded again must match its stored vector at least this well


def sample_rows(n: int):
    import lancedb

    table = lancedb.connect(str(config.DB_PATH)).open_table(config.TABLE)
    df = table.search().select(["chunk_id", "embed_text", "vector"]).limit(5000).to_pandas()
    return df.sample(n=min(n, len(df)), random_state=7)


async def main_async(n: int) -> int:
    import numpy as np

    rule(f"re-embedding {n} stored rows with {config.EMBED_API_MODEL}")
    rows = sample_rows(n)
    fresh = await retrieval_api.embed(rows["embed_text"].tolist())
    stored = np.vstack(rows["vector"].to_numpy()).astype("float32")
    stored /= np.clip(np.linalg.norm(stored, axis=1, keepdims=True), 1e-9, None)
    same = (fresh * stored).sum(axis=1)
    unrelated = (fresh * np.roll(stored, 1, axis=0)).sum(axis=1)
    print(f"  matching rows : min {same.min():.4f}  mean {same.mean():.4f}")
    print(f"  unrelated rows: mean {unrelated.mean():.4f}")
    healthy = bool(same.min() >= MATCH_FLOOR)
    print(f"\n  {bold('OK') if healthy else bold('MISMATCH')}: the API "
          f"{'matches' if healthy else 'no longer matches'} the corpus's vectors")
    return 0 if healthy else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=20)
    return asyncio.run(main_async(ap.parse_args().rows))


if __name__ == "__main__":
    raise SystemExit(main())
