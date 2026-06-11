"""
Tests for TradingAgents path resolution (F-01-005).

The resolver must prefer, in order:
1. TRADINGAGENTS_PATH env var (when it points at an existing directory)
2. The internal archived copy (backend/_archive_TradingAgents_fork_pre_2026-05-12)
3. A sibling stockanalysistool checkout (<repo>/../stockanalysistool/TradingAgents)
4. The legacy backend/TradingAgents location

It must never return a non-existent path and must return None when no
candidate exists, so callers fall back to stubs instead of polluting sys.path.
"""

import importlib.util
from pathlib import Path

import pytest

# Load the module directly (repo pattern, see test_trading_agents.py) to
# bypass backend/analytics/agents/__init__.py which imports heavy deps.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "agents"
    / "tradingagents_path.py"
)
_spec = importlib.util.spec_from_file_location("tradingagents_path", _MODULE_PATH)
tradingagents_path = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tradingagents_path)
resolve_tradingagents_path = tradingagents_path.resolve_tradingagents_path


@pytest.fixture
def fake_tree(tmp_path):
    """Build a fake repo tree with all candidate locations present."""
    backend_dir = tmp_path / "repo" / "backend"
    archive = backend_dir / "_archive_TradingAgents_fork_pre_2026-05-12"
    legacy = backend_dir / "TradingAgents"
    sibling = tmp_path / "stockanalysistool" / "TradingAgents"
    for d in (archive, legacy, sibling):
        d.mkdir(parents=True)
    return {
        "backend_dir": backend_dir,
        "archive": archive,
        "legacy": legacy,
        "sibling": sibling,
    }


def test_env_var_wins_when_it_exists(fake_tree, tmp_path, monkeypatch):
    override = tmp_path / "custom" / "TradingAgents"
    override.mkdir(parents=True)
    monkeypatch.setenv("TRADINGAGENTS_PATH", str(override))
    assert resolve_tradingagents_path(fake_tree["backend_dir"]) == str(override)


def test_env_var_ignored_when_missing(fake_tree, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_PATH", str(tmp_path / "does-not-exist"))
    assert resolve_tradingagents_path(fake_tree["backend_dir"]) == str(
        fake_tree["archive"]
    )


def test_internal_archive_preferred_over_sibling_and_legacy(fake_tree, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_PATH", raising=False)
    assert resolve_tradingagents_path(fake_tree["backend_dir"]) == str(
        fake_tree["archive"]
    )


def test_sibling_checkout_when_archive_absent(fake_tree, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_PATH", raising=False)
    fake_tree["archive"].rmdir()
    assert resolve_tradingagents_path(fake_tree["backend_dir"]) == str(
        fake_tree["sibling"]
    )


def test_legacy_location_last(fake_tree, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_PATH", raising=False)
    fake_tree["archive"].rmdir()
    fake_tree["sibling"].rmdir()
    assert resolve_tradingagents_path(fake_tree["backend_dir"]) == str(
        fake_tree["legacy"]
    )


def test_none_when_nothing_exists(fake_tree, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_PATH", raising=False)
    for key in ("archive", "sibling", "legacy"):
        fake_tree[key].rmdir()
    assert resolve_tradingagents_path(fake_tree["backend_dir"]) is None


def test_real_repo_resolves_to_existing_directory():
    """Against the actual repo layout the resolver must find a real directory."""
    import os

    resolved = resolve_tradingagents_path(tradingagents_path.default_backend_dir())
    assert resolved is not None
    assert os.path.isdir(resolved)
