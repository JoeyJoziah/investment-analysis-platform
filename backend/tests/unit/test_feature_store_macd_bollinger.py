"""
Regression tests for MACD + Bollinger Bands builtin features.

F-03-008 (audit 2026-04, G2a sub-theme A step 31):
ML_PIPELINE_DOCUMENTATION.md advertised ``macd``, ``macd_signal``,
``bollinger_upper_20d``, and ``bollinger_lower_20d`` as built-in
features but the corresponding compute methods did not exist on
``FeatureStore``. Calls to compute them silently returned ``None``
(``builtin_features.get(...)``).

This test verifies that the four feature names are now registered AND
their compute methods produce sane numeric values on a synthetic price
series.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


_FS_PATH = (
    Path(__file__).resolve().parents[2]
    / "ml"
    / "feature_store.py"
)


def _load_feature_store(monkeypatch: pytest.MonkeyPatch):
    """Load feature_store with heavy deps stubbed where possible."""

    spec = importlib.util.spec_from_file_location(
        "feature_store_under_test", _FS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synth_close_series(start: float = 100.0, n: int = 60, drift: float = 0.5):
    """A deterministic price series long enough for MACD+signal (>=35)."""
    return [start + drift * i for i in range(n)]


def _price_frame(ticker: str = "ACME"):
    closes = _synth_close_series()
    return pd.DataFrame({
        "ticker": [ticker] * len(closes),
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
        "close": closes,
    })


def test_macd_and_bollinger_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-03-008: builtin_features dict must include the 4 documented names."""

    text = _FS_PATH.read_text()
    for name in ("macd", "macd_signal", "bollinger_upper_20d", "bollinger_lower_20d"):
        assert f"'{name}'" in text, (
            f"builtin_features must register {name!r}"
        )
        assert f"_compute_{name}" in text or f"_compute_bollinger_band" in text, (
            f"compute method for {name!r} must exist"
        )


def test_macd_produces_finite_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-03-008: MACD = EMA(12)-EMA(26) over a 60-day series → finite float."""

    mod = _load_feature_store(monkeypatch)
    fs = mod.FeatureStore.__new__(mod.FeatureStore)
    fs.computation_cache = {}

    df = _price_frame("ACME")
    result = fs._compute_macd(["ACME"], datetime.now(), {"price_data": df}, pd.DataFrame())
    val = result.loc["ACME"]
    assert np.isfinite(val), f"MACD must be finite for a healthy series, got {val!r}"


def test_macd_signal_produces_finite_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-03-008: MACD signal = EMA(9) of MACD on a >=35-day series."""

    mod = _load_feature_store(monkeypatch)
    fs = mod.FeatureStore.__new__(mod.FeatureStore)
    fs.computation_cache = {}

    df = _price_frame("ACME")
    result = fs._compute_macd_signal(["ACME"], datetime.now(), {"price_data": df}, pd.DataFrame())
    val = result.loc["ACME"]
    assert np.isfinite(val)


def test_bollinger_bands_bracket_sma(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-03-008: upper > SMA(20) > lower for any positive-volatility window."""

    mod = _load_feature_store(monkeypatch)
    fs = mod.FeatureStore.__new__(mod.FeatureStore)
    fs.computation_cache = {}

    # Use a series with real volatility (drift + noise) so std > 0.
    closes = [100.0 + 0.5 * i + (3.0 if i % 2 == 0 else -3.0) for i in range(60)]
    df = pd.DataFrame({
        "ticker": ["ACME"] * len(closes),
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
        "close": closes,
    })

    sma = fs._compute_sma_20d(["ACME"], datetime.now(), {"price_data": df}, pd.DataFrame()).loc["ACME"]
    upper = fs._compute_bollinger_upper_20d(["ACME"], datetime.now(), {"price_data": df}, pd.DataFrame()).loc["ACME"]
    lower = fs._compute_bollinger_lower_20d(["ACME"], datetime.now(), {"price_data": df}, pd.DataFrame()).loc["ACME"]

    assert lower < sma < upper, (
        f"expected lower < sma < upper; got lower={lower}, sma={sma}, upper={upper}"
    )


def test_short_series_returns_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-03-008: insufficient data → NaN, not crash."""

    mod = _load_feature_store(monkeypatch)
    fs = mod.FeatureStore.__new__(mod.FeatureStore)
    fs.computation_cache = {}

    # Only 10 days of data — too short for any of the 4 features.
    df = pd.DataFrame({
        "ticker": ["X"] * 10,
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "close": [100.0 + i for i in range(10)],
    })
    for method in (
        fs._compute_macd,
        fs._compute_macd_signal,
        fs._compute_bollinger_upper_20d,
        fs._compute_bollinger_lower_20d,
    ):
        out = method(["X"], datetime.now(), {"price_data": df}, pd.DataFrame())
        assert np.isnan(out.loc["X"]), f"{method.__name__} on short series should be NaN"
