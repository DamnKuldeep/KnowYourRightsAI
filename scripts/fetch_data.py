"""Fetch the database bundle onto a fresh machine.

The bundle is ~330 MB and deliberately not in git. Hosting it in a Hugging Face **dataset**
repo is the practical answer: free, unmetered, and already close to wherever you deploy.

    huggingface-cli login
    huggingface-cli upload <you>/knowyourrights-db data/ . --repo-type=dataset

then on the server:

    KYR_DATA_REPO=<you>/knowyourrights-db python scripts/fetch_data.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config                                 # noqa: E402
from knowyourrights.runtime.console import rule, setup_console    # noqa: E402

setup_console()

REQUIRED = [
    (config.DB_PATH, "LanceDB directory"),
    (config.PARQUET, "chunk metadata"),
]


def present() -> bool:
    return all(path.exists() for path, _ in REQUIRED)


def main() -> int:
    rule("data bundle")
    for path, label in REQUIRED:
        print(f"  {label:<20} {'OK ' if path.exists() else 'missing'} {path}")

    if present():
        print("\n  Everything is here; nothing to fetch.")
        return 0

    repo = os.environ.get("KYR_DATA_REPO", "").strip()
    if not repo:
        print("\n  The database is missing and KYR_DATA_REPO is not set.")
        print("  Either copy data/ onto this machine, or publish it once:")
        print("    huggingface-cli upload <you>/knowyourrights-db data/ . --repo-type=dataset")
        print("  then set KYR_DATA_REPO=<you>/knowyourrights-db and run this again.")
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n  pip install huggingface_hub  (needed to fetch from a dataset repo)")
        return 1

    print(f"\n  downloading {repo} -> {config.DATA_DIR}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset",
                      local_dir=str(config.DATA_DIR),
                      token=os.environ.get("HF_TOKEN") or None)

    ok = present()
    print("  done." if ok else "  download finished but expected files are still missing.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
