"""
Regression test for #208 item 1 (#200 follow-up): real feature_data wiring.

`create_prediction` / `_run_ensemble_prediction` used to call
`_run_single_model_prediction(..., feature_data=None)` with a
``TODO(#200-follow-up)``.  They now engineer *real* windowed feature tensors
from the ticker's price history via ``_fetch_feature_data`` and pass them in.

The critical contracts verified here:
  * ``_fetch_feature_data`` returns the exact model-shaped tensor (LSTM 3D /
    XGBoost 2D) when real price history exists,
  * it returns ``None`` (fail-loud, #200) when there is genuinely no history —
    never fabricating data, and
  * the 503 no-data guard in ``_run_single_model_prediction`` still fires when
    ``feature_data`` is None and BOOTSTRAP_MODELS is unset.

Run with::

    ENVIRONMENT=test JWT_SECRET_KEY=x SECRET_KEY=y MASTER_SECRET_KEY=z \
      DATABASE_URL=postgresql://u:p@localhost/db REDIS_URL=redis://localhost \
      POSTGRES_HOST=localhost POSTGRES_DB=test \
      pytest backend/tests/test_ml_feature_data_wiring.py --noconftest
"""
import asyncio
import importlib
import sys
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

ml = importlib.import_module("backend.api.routers.ml")

# Real persisted model contract (subset mirrors ml_models/{lstm,xgboost}_config.json).
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
_N_FEATURES = len(_FEATURE_COLUMNS)
_LSTM_CONFIG = {"feature_columns": _FEATURE_COLUMNS, "sequence_length": 60}
_XGB_CONFIG = {"feature_columns": _FEATURE_COLUMNS}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _rows(n):
    out = []
    base = date(2025, 1, 1)
    for i in range(n):
        c = 100.0 + i * 0.4
        out.append(
            SimpleNamespace(
                date=base + timedelta(days=i),
                open=c - 0.2, high=c + 0.7, low=c - 0.8, close=c,
                volume=1_000_000 + i * 500,
            )
        )
    return out


def _manager_with_config(config):
    mgr = MagicMock()
    mgr.get_model = MagicMock(return_value={"config": config} if config else None)
    return mgr


def _patch_repo(monkeypatch, rows):
    """Patch the lazily-imported price_repository.get_price_history."""
    fake_repo = SimpleNamespace(get_price_history=AsyncMock(return_value=rows))
    fake_mod = SimpleNamespace(price_repository=fake_repo)
    monkeypatch.setitem(
        sys.modules, "backend.repositories.price_repository", fake_mod
    )
    return fake_repo


# ---------------------------------------------------------------------------
# _fetch_feature_data produces real, model-shaped tensors
# ---------------------------------------------------------------------------

def test_fetch_feature_data_lstm_shape(monkeypatch):
    _patch_repo(monkeypatch, _rows(120))
    mgr = _manager_with_config(_LSTM_CONFIG)
    tensor = _run(
        ml._fetch_feature_data(mgr, ml.MLModelType.LSTM, "AAPL", 7, MagicMock())
    )
    assert tensor is not None
    assert tensor.shape == (1, 60, _N_FEATURES)
    assert np.isfinite(tensor).all()


def test_fetch_feature_data_xgboost_shape_matches_horizon(monkeypatch):
    _patch_repo(monkeypatch, _rows(120))
    mgr = _manager_with_config(_XGB_CONFIG)
    matrix = _run(
        ml._fetch_feature_data(mgr, ml.MLModelType.XGBOOST, "AAPL", 5, MagicMock())
    )
    assert matrix is not None
    assert matrix.shape == (5, _N_FEATURES)


def test_fetch_feature_data_prophet_returns_none(monkeypatch):
    # Prophet consumes future dates, not a feature tensor.
    _patch_repo(monkeypatch, _rows(120))
    mgr = _manager_with_config(_LSTM_CONFIG)
    assert _run(
        ml._fetch_feature_data(mgr, ml.MLModelType.PROPHET, "AAPL", 7, MagicMock())
    ) is None


# ---------------------------------------------------------------------------
# Fail-loud: no history / no config -> None (so caller raises 503)
# ---------------------------------------------------------------------------

def test_fetch_feature_data_none_on_empty_history(monkeypatch):
    _patch_repo(monkeypatch, [])
    mgr = _manager_with_config(_LSTM_CONFIG)
    assert _run(
        ml._fetch_feature_data(mgr, ml.MLModelType.LSTM, "AAPL", 7, MagicMock())
    ) is None


def test_fetch_feature_data_none_when_model_has_no_config(monkeypatch):
    _patch_repo(monkeypatch, _rows(120))
    mgr = _manager_with_config(None)  # fallback/dummy model, no persisted contract
    assert _run(
        ml._fetch_feature_data(mgr, ml.MLModelType.XGBOOST, "AAPL", 7, MagicMock())
    ) is None


# ---------------------------------------------------------------------------
# The 503 no-data guard still fires when feature_data is None
# ---------------------------------------------------------------------------

def test_503_guard_still_fires_when_feature_data_none(monkeypatch):
    monkeypatch.delenv("BOOTSTRAP_MODELS", raising=False)
    mgr = MagicMock()
    mgr.get_model = MagicMock(return_value={"config": _LSTM_CONFIG})
    with pytest.raises(ml.HTTPException) as exc_info:
        _run(
            ml._run_single_model_prediction(
                mgr, ml.MLModelType.LSTM, "AAPL", 7, 100.0, feature_data=None
            )
        )
    assert exc_info.value.status_code == 503


def test_real_feature_data_bypasses_503_guard(monkeypatch):
    monkeypatch.delenv("BOOTSTRAP_MODELS", raising=False)
    mgr = MagicMock()
    mgr.get_model = MagicMock(return_value={"config": _LSTM_CONFIG})
    mgr.predict = MagicMock(return_value=np.array([101.0]))
    feature = np.zeros((1, 60, _N_FEATURES), dtype=np.float32)
    raw, key = _run(
        ml._run_single_model_prediction(
            mgr, ml.MLModelType.LSTM, "AAPL", 7, 100.0, feature_data=feature
        )
    )
    # No 503 raised; the supplied real tensor was passed straight to predict.
    mgr.predict.assert_called_once()
    assert key == "lstm_price_predictor"
