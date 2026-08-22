"""Shared test setup.

The retry and backoff constants are production values measured in seconds. Left alone they
make the suite spend most of its time asleep, so they are scaled down globally here — the
*logic* under test is unchanged, only the wall-clock cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config  # noqa: E402


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(config, "RETRY_INITIAL_DELAY", 0.01)
    monkeypatch.setattr(config, "RETRY_MAX_DELAY", 0.05)
    monkeypatch.setattr(config, "RETRY_MULTIPLIER", 1.5)


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """Keep tests off the developer's real ``.runtime`` directory.

    Not merely tidiness. The model-failover test deliberately makes a model return 410, and
    the registry persists that verdict — so without isolation a mocked failure retires a
    healthy production model, and the app quietly runs on its fallback afterwards. This
    happened, which is why the registry now resolves its path lazily.
    """
    from knowyourrights.nim import registry

    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(config, "CACHE_DIR", runtime / "cache")
    monkeypatch.setattr(config, "THRESHOLDS_FILE", runtime / "thresholds.json")
    monkeypatch.setattr(registry, "_state", None)
    yield runtime
    registry._state = None
