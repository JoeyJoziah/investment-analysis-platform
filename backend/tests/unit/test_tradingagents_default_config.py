"""
Regression tests for TradingAgents default_config.

F-04-001 (audit 2026-04, G2a sub-theme B step 4): ``data_dir`` was
hardcoded to ``/Users/yluo/Documents/Code/ScAI/FR1-data``, an absolute
path on a different developer's machine. The fail-first tests below
assert that ``data_dir`` is sourced from the ``TRADINGAGENTS_DATA_DIR``
environment variable and never contains the legacy hardcoded path.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest


_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "TradingAgents"
    / "tradingagents"
    / "default_config.py"
)


def _load_config(monkeypatch_env: dict[str, str] | None = None) -> dict:
    """Load default_config in isolation, optionally with patched env."""

    if monkeypatch_env is not None:
        for k, v in monkeypatch_env.items():
            os.environ[k] = v

    spec = importlib.util.spec_from_file_location(
        "tradingagents_default_config_under_test", _CONFIG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DEFAULT_CONFIG


def test_data_dir_is_not_hardcoded_yluo_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-04-001: legacy ``/Users/yluo/...`` path must be gone."""

    monkeypatch.delenv("TRADINGAGENTS_DATA_DIR", raising=False)
    cfg = _load_config()
    assert "yluo" not in cfg["data_dir"], (
        f"hardcoded developer path leaked into data_dir: {cfg['data_dir']!r}"
    )


def test_data_dir_respects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-04-001: ``TRADINGAGENTS_DATA_DIR`` must control ``data_dir``."""

    monkeypatch.setenv("TRADINGAGENTS_DATA_DIR", "/tmp/loki-test-data")
    cfg = _load_config()
    assert cfg["data_dir"] == "/tmp/loki-test-data"
