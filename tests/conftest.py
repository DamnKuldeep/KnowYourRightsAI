"""Shared test setup: no network, no real keys, and no writes outside a temporary directory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowyourrights import config


def corpus_available() -> bool:
    """True when the LanceDB corpus has been downloaded, not just its Git LFS pointers."""
    data = config.DB_PATH / f"{config.TABLE}.lance" / "data"
    return any(p.stat().st_size > 1024 for p in data.glob("*.lance")) if data.is_dir() else False


requires_corpus = pytest.mark.skipif(not corpus_available(),
                                     reason="the corpus is not downloaded (git lfs pull)")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Production retry delays are seconds; the logic under test is the same at milliseconds."""
    monkeypatch.setattr(config, "RETRY_INITIAL_DELAY", 0.01)
    monkeypatch.setattr(config, "RETRY_MAX_DELAY", 0.05)
    monkeypatch.setattr(config, "RETRY_MULTIPLIER", 1.5)


@pytest.fixture(autouse=True)
def stub_provider_keys(monkeypatch):
    """Fake keys, so every role resolves on a machine with no ``.env`` (CI included).

    Requests are served by ``httpx.MockTransport`` or stubbed out, so these are never sent.
    """
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-test-key-not-real")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-v1-test-key-not-real")
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """Keep every test off the real ``.runtime`` directory and every singleton fresh.

    Not tidiness: a mocked model failure used to be persisted by the registry and retire a
    healthy model in the developer's real runtime.
    """
    from knowyourrights.llm import ledger, limiter, registry
    from knowyourrights.runtime import cache

    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(config, "CACHE_DIR", runtime / "cache")
    monkeypatch.setattr(config, "THRESHOLDS_FILE", runtime / "thresholds.json")
    monkeypatch.setattr(registry, "_state", None)
    registry._sidelined.clear()
    monkeypatch.setattr(ledger, "_LEDGER", None)
    monkeypatch.setattr(cache, "_CACHE", None)
    # Rate-limit buckets carried over between tests made later tests wait on earlier ones.
    monkeypatch.setattr(limiter, "_REGISTRY", None)
    yield runtime
    registry._state = None
    registry._sidelined.clear()
