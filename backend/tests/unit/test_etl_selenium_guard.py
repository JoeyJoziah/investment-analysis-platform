"""
Regression tests for the optional-selenium import guard.

F-05-001 (audit 2026-04, G2a sub-theme C step 19):
``backend/etl/unlimited_data_extractor.py`` had top-level ``from selenium ...``
imports. In environments without selenium installed, the whole module
failed to import, which cascaded through
``unlimited_extractor_with_fallbacks`` → ``data_extractor``, making
``from backend.etl.data_extractor import DataExtractor`` raise
ModuleNotFoundError.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest


def test_module_imports_without_selenium(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-05-001: module must import even when selenium is missing.

    Synthesises a ``loki_test_etl`` package, copies just the two files
    we care about (``types.py`` and ``unlimited_data_extractor.py``)
    into it, and imports under that name. This avoids both pulling in
    the heavy real ``backend.etl`` package and dealing with relative
    import edge cases under spec_from_file_location.
    """

    # Drop any cached selenium-touched modules.
    for name in list(sys.modules):
        if name.startswith("selenium") or "unlimited_data_extractor" in name:
            del sys.modules[name]

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "selenium" or name.startswith("selenium."):
            raise ImportError(f"simulated: {name} not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    etl_dir = Path(__file__).resolve().parents[2] / "etl"
    for heavy in ("aiohttp", "bs4", "pandas", "numpy", "yfinance", "requests"):
        pytest.importorskip(heavy, reason=f"{heavy} not installed")

    # Build a synthetic package on disk to host the two source files.
    import tempfile
    import shutil
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "loki_test_etl"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        shutil.copy(etl_dir / "types.py", pkg / "types.py")
        shutil.copy(
            etl_dir / "unlimited_data_extractor.py",
            pkg / "unlimited_data_extractor.py",
        )

        monkeypatch.syspath_prepend(tmp)
        module = importlib.import_module(
            "loki_test_etl.unlimited_data_extractor"
        )

    assert hasattr(module, "SELENIUM_AVAILABLE")
    assert module.SELENIUM_AVAILABLE is False


def test_selenium_available_flag_exists_in_source() -> None:
    """F-05-001: source-level guarantee of the SELENIUM_AVAILABLE flag."""

    path = (
        Path(__file__).resolve().parents[2]
        / "etl"
        / "unlimited_data_extractor.py"
    )
    text = path.read_text()
    assert "SELENIUM_AVAILABLE" in text, (
        "unlimited_data_extractor.py must export a SELENIUM_AVAILABLE flag"
    )
    assert "try:" in text and "from selenium import webdriver" in text, (
        "selenium imports must live inside a try/except block"
    )
    assert "except ImportError" in text
