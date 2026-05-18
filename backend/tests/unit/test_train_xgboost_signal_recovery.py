"""
Regression tests for F-03-004 (XGBoost R²<0, all-zero feature importances).

Per workpaper §3 step 30:
- Add gate test asserting ``r2 > 0.0`` AND at least one non-zero feature
  importance — currently fails on main.
- Audit pipeline: target construction, train/test split look-ahead,
  feature preprocessing. Likely cross-cuts F-05-004 (mock data) and
  F-06-008.

Root cause: ``XGBoostTrainer.prepare_features`` runs
``np.nan_to_num(y, nan=0.0)`` on the future-return target. Future
returns at the end of every time series are NaN (no look-ahead data).
Filling them with 0 means the best constant predictor is ``predict=0``,
the model collapses to that, R²→0 or negative, and gradient-based
feature importances all sit at zero.

This test exercises the real ``prepare_features`` and a minimal
training step on synthetic data with KNOWN signal:
``y = 0.7*X[:,0] + 0.3*X[:,1] + small noise``. After the fix:
- prepare_features must drop NaN-target rows rather than fill them.
- A short XGBoost fit on the cleaned data must produce r2 > 0 AND
  at least one non-zero feature importance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


_TRAIN_XGB = (
    Path(__file__).resolve().parents[2]
    / "ml"
    / "training"
    / "train_xgboost.py"
)


def _load_trainer_module(monkeypatch: pytest.MonkeyPatch):
    """Load train_xgboost in isolation with logger configured.

    We stub out backend.ml.gpu_utils so the module loads in environments
    without the gpu_utils helpers; the trainer's fallback path handles
    the rest.
    """
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    # Stub the optional gpu_utils chain so the import doesn't fail.
    backend_pkg = MagicMock(); backend_pkg.__path__ = []
    ml_pkg = MagicMock(); ml_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "backend", backend_pkg)
    monkeypatch.setitem(sys.modules, "backend.ml", ml_pkg)

    name = "train_xgboost_under_test"
    spec = importlib.util.spec_from_file_location(name, _TRAIN_XGB)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _synth_frame_with_signal_and_nan_tail(
    n_per_ticker: int = 100,
    n_tickers: int = 8,
    seed: int = 7,
) -> pd.DataFrame:
    """Frame mirroring real multi-ticker shape with realistic NaN tails.

    Real training data has one row per (ticker, date) and each ticker
    has its own 5-day NaN tail at ``future_return_5d`` (the last 5 days
    of the series have no future data). With 8 tickers × 5 NaN tail
    rows = 40 NaN targets out of 800 = 5%, the bug's signal corruption
    is severe enough that XGBoost collapses to predict=0 (the dominant
    bin after nan_to_num zero-fill) and importances all sit at zero.

    Includes:
    - 4 non-degenerate float features ``feat_a..d``.
    - The canonical target ``future_return_5d`` with NaN tail per ticker.
    - Non-numeric ``date``, ``ticker`` columns that must be excluded.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for ti in range(n_tickers):
        feat_a = rng.normal(0, 1, n_per_ticker)
        feat_b = rng.normal(0, 1, n_per_ticker)
        feat_c = rng.normal(0, 1, n_per_ticker)
        feat_d = rng.normal(0, 1, n_per_ticker)
        # KNOWN signal: y = 0.7*a + 0.3*b + noise.
        y = 0.7 * feat_a + 0.3 * feat_b + rng.normal(0, 0.1, n_per_ticker)
        # Per-ticker NaN tail mirrors the look-ahead truncation reality.
        y[-5:] = np.nan
        frames.append(pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n_per_ticker, freq="D"),
            "ticker": [f"TKR{ti:02d}"] * n_per_ticker,
            "feat_a": feat_a,
            "feat_b": feat_b,
            "feat_c": feat_c,
            "feat_d": feat_d,
            "future_return_5d": y,
        }))
    return pd.concat(frames, ignore_index=True)


def test_prepare_features_drops_nan_target_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-03-004: prepare_features must drop rows with NaN target, not synthesize 0."""

    mod = _load_trainer_module(monkeypatch)
    trainer = mod.XGBoostTrainer.__new__(mod.XGBoostTrainer)
    trainer.target_column = "future_return_5d"
    trainer.feature_columns = None

    df = _synth_frame_with_signal_and_nan_tail(n_per_ticker=100, n_tickers=8)
    n_rows = len(df)
    n_nan_tails = 8 * 5  # n_tickers * 5-day tail
    X, y = trainer.prepare_features(df)

    assert len(y) == n_rows - n_nan_tails, (
        f"prepare_features must drop NaN-target rows; expected "
        f"{n_rows - n_nan_tails}, got {len(y)}"
    )
    # No zero-target spike from nan_to_num filling.
    assert np.sum(y == 0.0) < 3, (
        f"prepare_features still synthesizes 0 for NaN targets; "
        f"got {np.sum(y == 0.0)} zero targets"
    )


def test_xgboost_recovers_r2_and_nonzero_importance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-03-004 ACCEPTANCE GATE: r2 > 0.0 AND ≥1 non-zero feature importance."""

    mod = _load_trainer_module(monkeypatch)
    trainer = mod.XGBoostTrainer.__new__(mod.XGBoostTrainer)
    trainer.target_column = "future_return_5d"
    trainer.feature_columns = None
    # Standard preprocessing path used by the trainer.
    from sklearn.preprocessing import StandardScaler
    trainer.scaler = StandardScaler()

    # Build a small, low-SNR multi-ticker frame where the NaN-zero-fill
    # bug actually dominates: 12 tickers × 60 days with σ=0.4 noise →
    # ~12% NaN-zero-filled targets and SNR low enough that the
    # nan_to_num spike at 0 swamps the real signal.
    rng = np.random.default_rng(11)
    frames = []
    for ti in range(12):
        feat_a = rng.normal(0, 1, 60)
        feat_b = rng.normal(0, 1, 60)
        feat_c = rng.normal(0, 1, 60)
        feat_d = rng.normal(0, 1, 60)
        y = 0.5 * feat_a + 0.2 * feat_b + rng.normal(0, 0.4, 60)
        y[-7:] = np.nan
        frames.append(pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=60, freq="D"),
            "ticker": [f"T{ti:02d}"] * 60,
            "feat_a": feat_a, "feat_b": feat_b,
            "feat_c": feat_c, "feat_d": feat_d,
            "future_return_5d": y,
        }))
    df = pd.concat(frames, ignore_index=True)
    X, y = trainer.prepare_features(df)
    X_scaled = trainer.scaler.fit_transform(X)

    # Train/test split (chronological — last 20% is the held-out tail).
    cut = int(0.8 * len(X_scaled))
    X_tr, X_te = X_scaled[:cut], X_scaled[cut:]
    y_tr, y_te = y[:cut], y[cut:]

    import xgboost as xgb
    from sklearn.metrics import r2_score

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr, verbose=False)
    preds = model.predict(X_te)
    r2 = r2_score(y_te, preds)
    importances = model.feature_importances_

    assert r2 > 0.0, (
        f"XGBoost R² is non-positive ({r2:.4f}). Signal is recoverable on "
        f"this synthetic data ONLY if prepare_features stops the nan_to_num "
        f"target-fill."
    )
    assert np.any(importances > 0.0), (
        f"All feature importances are zero ({importances!r}). XGBoost failed "
        f"to find a split — consistent with constant labels (post-nan_to_num)."
    )


def test_prepare_features_drops_pure_nan_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-03-004 supporting: all-NaN feature columns must be dropped, not zero-filled.

    nan_to_num turns an all-NaN column into all zeros — a constant
    feature — which gets zero importance and pollutes feature_columns.
    """
    mod = _load_trainer_module(monkeypatch)
    trainer = mod.XGBoostTrainer.__new__(mod.XGBoostTrainer)
    trainer.target_column = "future_return_5d"
    trainer.feature_columns = None

    df = _synth_frame_with_signal_and_nan_tail(n_per_ticker=100, n_tickers=2)
    df["dead_col"] = np.nan  # entirely NaN
    df["const_col"] = 1.0    # zero variance
    X, y = trainer.prepare_features(df)

    # ``feature_columns`` must not contain the dead or constant columns.
    assert "dead_col" not in trainer.feature_columns, (
        "prepare_features kept an all-NaN column — it'll become a zero-importance constant"
    )
    assert "const_col" not in trainer.feature_columns, (
        "prepare_features kept a zero-variance column — useless for learning"
    )
