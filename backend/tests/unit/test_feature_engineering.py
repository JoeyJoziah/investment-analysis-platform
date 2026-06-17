"""
Unit tests for backend/ml/feature_engineering.py (#208 item 1).

Verifies the inference-time feature builders produce the exact tensor/row shape
each model expects and that they fail loud (no fabrication) on empty history.

Uses the importlib file-loading bypass so the module is exercised source-level
without pulling SQLAlchemy / the backend package graph into the test process.

Run (source-level, no conftest):
    ENVIRONMENT=test JWT_SECRET_KEY=x SECRET_KEY=y \
      MASTER_SECRET_KEY=... DATABASE_URL=... REDIS_URL=... \
      POSTGRES_HOST=localhost POSTGRES_DB=test \
      python3 -m pytest backend/tests/unit/test_feature_engineering.py --noconftest -q
"""

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ML_DIR = Path(__file__).resolve().parents[2] / "ml"


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _ML_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_fe = _load("feature_engineering_mod", "feature_engineering.py")

build_lstm_features = _fe.build_lstm_features
build_xgboost_features = _fe.build_xgboost_features
InsufficientHistoryError = _fe.InsufficientHistoryError


# The real persisted model contract: 56 features, LSTM sequence_length 60.
_FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume", "dividends", "stock splits",
    "market_cap", "pe_ratio", "beta", "dividend_yield",
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist", "rsi_14", "rsi_7",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr_14", "stoch_k", "stoch_d", "williams_r", "cci", "adx", "obv", "mfi",
    "roc_10", "momentum_10", "vwap", "resistance_1", "support_1",
    "returns", "log_returns", "volatility_20", "intraday_range",
    "close_to_open", "gap", "volume_sma_20", "volume_ratio",
    "trend_sma", "trend_macd", "above_sma_200",
    "return_lag_1", "return_lag_5", "return_lag_10", "return_lag_20",
]
_LSTM_CONFIG = {"feature_columns": _FEATURE_COLUMNS, "sequence_length": 60}
_XGB_CONFIG = {"feature_columns": _FEATURE_COLUMNS}
_N_FEATURES = len(_FEATURE_COLUMNS)


def _history(n_days: int):
    """Build ``n_days`` of monotonically-varying, *real* OHLCV rows."""
    rows = []
    base = date(2025, 1, 1)
    for i in range(n_days):
        c = 100.0 + i * 0.5
        rows.append(
            SimpleNamespace(
                date=base + timedelta(days=i),
                open=c - 0.3,
                high=c + 0.8,
                low=c - 0.9,
                close=c,
                volume=1_000_000 + i * 1000,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# LSTM shape contract
# ---------------------------------------------------------------------------

def test_lstm_features_exact_3d_shape():
    tensor = build_lstm_features(_history(120), _LSTM_CONFIG)
    assert tensor.shape == (1, 60, _N_FEATURES)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()  # no nan/inf leaked through


def test_lstm_features_left_pads_short_history_to_seq_len():
    # Fewer rows than the 60-step window -> shape still exact (real rows padded).
    tensor = build_lstm_features(_history(10), _LSTM_CONFIG)
    assert tensor.shape == (1, 60, _N_FEATURES)


def test_lstm_features_respect_sequence_length_override():
    tensor = build_lstm_features(_history(40), _LSTM_CONFIG, sequence_length=30)
    assert tensor.shape == (1, 30, _N_FEATURES)


# ---------------------------------------------------------------------------
# XGBoost shape contract
# ---------------------------------------------------------------------------

def test_xgboost_features_exact_2d_shape_single_row():
    matrix = build_xgboost_features(_history(120), _XGB_CONFIG)
    assert matrix.shape == (1, _N_FEATURES)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()


def test_xgboost_features_multiple_rows():
    matrix = build_xgboost_features(_history(120), _XGB_CONFIG, n_rows=7)
    assert matrix.shape == (7, _N_FEATURES)


# ---------------------------------------------------------------------------
# Fail-loud: empty history must raise, never fabricate (#200)
# ---------------------------------------------------------------------------

def test_lstm_empty_history_raises():
    with pytest.raises(InsufficientHistoryError):
        build_lstm_features([], _LSTM_CONFIG)


def test_xgboost_empty_history_raises():
    with pytest.raises(InsufficientHistoryError):
        build_xgboost_features([], _XGB_CONFIG)


def test_missing_feature_columns_config_raises():
    with pytest.raises(ValueError):
        build_lstm_features(_history(60), {"sequence_length": 60})


# ---------------------------------------------------------------------------
# Real-signal: derivable columns are non-trivial, non-derivable are zero-filled
# ---------------------------------------------------------------------------

def test_derivable_columns_carry_real_signal_non_derivable_zeroed():
    matrix = build_xgboost_features(_history(120), _XGB_CONFIG, n_rows=1)
    row = matrix[0]
    idx = {c: i for i, c in enumerate(_FEATURE_COLUMNS)}

    # Real price/volume signal present.
    assert row[idx["close"]] > 0
    assert row[idx["volume"]] > 0
    assert row[idx["sma_20"]] > 0  # derived indicator

    # Non-derivable fundamentals are zero-filled, never fabricated.
    assert row[idx["pe_ratio"]] == 0.0
    assert row[idx["market_cap"]] == 0.0
    assert row[idx["cci"]] == 0.0  # indicator we don't compute -> zero
