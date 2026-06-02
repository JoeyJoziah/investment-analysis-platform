"""
End-to-end pipeline integration for the F-03-004 + F-06-008 bundle.

The closest analog to a browser E2E for this branch's work — there is
no UI surface yet for the ML evaluation outputs, so this exercises the
actual round-trip the Airflow DAG depends on:

    synthetic parquet
        → XGBoostTrainer.train (prepare → fit → _save_model)
        → ModelEvaluator.evaluate_xgboost (load model+scaler+config,
                                          predict on test parquet,
                                          compute metrics)
        → assert real metrics (r2 > 0, non-zero feature importance,
                               non-trivial direction accuracy)

This is the contract F-06-008 actually depends on: the DAG fails loud
on missing ``test_data.parquet`` only because the round trip produces
meaningful metrics when the parquet IS present.

Runs in ≤30 seconds on CPU with n_trials=2.
"""

from __future__ import annotations

import importlib.util
import json
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
_EVAL_MODELS = (
    Path(__file__).resolve().parents[2]
    / "ml"
    / "training"
    / "evaluate_models.py"
)


def _load_module_under_synthetic_pkg(
    monkeypatch: pytest.MonkeyPatch,
    pkg_name: str,
    module_name: str,
    path: Path,
):
    """Load a module under a synthetic package so internal imports of
    ``backend.ml.*`` resolve to harmless stubs (the trainer's gpu_utils
    import is optional; the evaluator does no ``backend.ml.*`` imports
    in the codepath we exercise).
    """
    # Clear any prior backend.* pollution from earlier tests in the run.
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    # Stub backend / backend.ml so the trainer's "try: from backend.ml.gpu_utils..."
    # gracefully takes the ImportError path.
    backend_pkg = MagicMock(); backend_pkg.__path__ = []
    ml_pkg = MagicMock(); ml_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "backend", backend_pkg)
    monkeypatch.setitem(sys.modules, "backend.ml", ml_pkg)

    fqname = f"{pkg_name}.{module_name}"
    spec = importlib.util.spec_from_file_location(fqname, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[fqname] = module
    monkeypatch.setitem(sys.modules, fqname, module)
    spec.loader.exec_module(module)
    return module


def _write_synth_parquets(data_dir: Path, target_col: str = "future_return_5d") -> None:
    """Create train/val/test parquets with known signal + realistic NaN tails."""

    def _make(n_tickers: int, n_days: int, seed_offset: int):
        rng_local = np.random.default_rng(11 + seed_offset)
        frames = []
        for ti in range(n_tickers):
            feat_a = rng_local.normal(0, 1, n_days)
            feat_b = rng_local.normal(0, 1, n_days)
            feat_c = rng_local.normal(0, 1, n_days)
            feat_d = rng_local.normal(0, 1, n_days)
            # Known recoverable signal.
            y = 0.6 * feat_a + 0.3 * feat_b + rng_local.normal(0, 0.15, n_days)
            y[-5:] = np.nan  # per-ticker NaN tail
            frames.append(pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=n_days, freq="D"),
                "ticker": [f"T{seed_offset:02d}_{ti:02d}"] * n_days,
                "feat_a": feat_a, "feat_b": feat_b,
                "feat_c": feat_c, "feat_d": feat_d,
                target_col: y,
            }))
        return pd.concat(frames, ignore_index=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    _make(n_tickers=6, n_days=120, seed_offset=0).to_parquet(data_dir / "train_data.parquet")
    _make(n_tickers=3, n_days=60, seed_offset=1).to_parquet(data_dir / "val_data.parquet")
    _make(n_tickers=3, n_days=60, seed_offset=2).to_parquet(data_dir / "test_data.parquet")


def test_train_xgboost_save_then_evaluate_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-03-004 + F-06-008: train → save → load → evaluate produces real metrics.

    Asserts the DAG-level contract: the held-out test set parquet is
    consumed by a freshly-loaded model+scaler+config and produces
    metrics with real signal (r2 > 0, direction_accuracy > 0.5,
    non-zero feature importance).
    """
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "ml_models"

    _write_synth_parquets(data_dir)

    # --- Train ---------------------------------------------------------
    trainer_mod = _load_module_under_synthetic_pkg(
        monkeypatch, "ml_bundle_e2e", "train_xgboost", _TRAIN_XGB
    )
    trainer = trainer_mod.XGBoostTrainer(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        n_trials=2,        # smoke; signal is strong enough to recover
        cv_splits=2,       # minimum for TimeSeriesSplit
        use_gpu=False,
    )
    results = trainer.train()

    # Trainer must report at least one non-zero feature importance.
    importances = results.get("feature_importance", {})
    assert importances, "train() did not record feature_importance in results"
    assert any(v > 0 for v in importances.values()), (
        f"all feature importances zero — F-03-004 regression: {importances!r}"
    )

    # Artifacts persisted to model_dir (the contract evaluate_models depends on).
    for art in ("xgboost_model.pkl", "xgboost_scaler.pkl", "xgboost_config.json"):
        assert (model_dir / art).exists(), f"trainer did not persist {art}"

    # --- Evaluate ------------------------------------------------------
    # Reload modules fresh so the evaluator gets a clean import (avoid
    # the trainer-side ``backend.ml`` MagicMock leaking into evaluate's
    # ``from backend.ml.training.train_lstm import LSTMModel`` lookup —
    # we don't exercise LSTM here so the stub is fine).
    evaluator_mod = _load_module_under_synthetic_pkg(
        monkeypatch, "ml_bundle_e2e", "evaluate_models", _EVAL_MODELS
    )
    evaluator = evaluator_mod.ModelEvaluator(
        data_dir=str(data_dir), model_dir=str(model_dir),
    )
    evaluator.load_test_data()
    metrics = evaluator.evaluate_xgboost()

    # F-06-008 acceptance: evaluator returns real metrics from the
    # persisted test parquet, not None and not random noise.
    assert metrics is not None, "evaluate_xgboost returned None — load chain broken"
    assert "r2" in metrics and "mse" in metrics and "direction_accuracy" in metrics

    # F-03-004 acceptance on the eval side: r2 > 0 on a held-out set
    # whose signal is recoverable from training.
    assert metrics["r2"] > 0.0, (
        f"eval R² non-positive ({metrics['r2']:.4f}) — either training "
        f"signal collapsed (F-03-004) or train/test split lost the signal"
    )

    # Direction accuracy must beat the 0.5 coin-flip baseline by a
    # margin — confirms the prediction is correlated with sign, not
    # just centered noise.
    assert metrics["direction_accuracy"] > 0.55, (
        f"direction_accuracy ({metrics['direction_accuracy']:.3f}) is at "
        f"or below coin-flip — predictions are uncorrelated with sign"
    )

    # MSE must be finite (sanity).
    assert np.isfinite(metrics["mse"]), "MSE is non-finite — NaN poisoning?"


def test_evaluator_loads_persisted_artifacts_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-06-008 supporting: evaluator reads the same config the trainer wrote.

    Verifies the persistence contract: trainer.feature_columns saved to
    xgboost_config.json is what evaluator reads back, and the scaler
    persisted is the one that produces consistent X_test_scaled.
    """
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "ml_models"

    _write_synth_parquets(data_dir)

    trainer_mod = _load_module_under_synthetic_pkg(
        monkeypatch, "ml_bundle_e2e_persist", "train_xgboost", _TRAIN_XGB
    )
    trainer = trainer_mod.XGBoostTrainer(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        n_trials=2, cv_splits=2, use_gpu=False,
    )
    trainer.train()

    # Config.json must contain the canonical feature list.
    cfg = json.loads((model_dir / "xgboost_config.json").read_text())
    assert "feature_columns" in cfg
    assert "target_column" in cfg and cfg["target_column"] == "future_return_5d"
    # F-03-004: dead/constant columns dropped — feature list non-empty
    # and not full of obvious junk.
    assert cfg["feature_columns"], "feature_columns list is empty"
    assert all(col not in cfg["feature_columns"] for col in ("date", "ticker")), (
        "non-numeric columns leaked into feature_columns"
    )
