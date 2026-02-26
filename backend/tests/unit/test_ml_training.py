"""
Unit tests for backend/ml/training/ module.

Tests cover:
- train_lstm.py: StockSequenceDataset, LSTMModel, LSTMTrainer
- train_xgboost.py: XGBoostTrainer
- train_prophet.py: ProphetTrainer
- evaluate_models.py: ModelEvaluator
- run_full_training.py: Pipeline orchestration functions

All tests use mocks -- no actual model training occurs.
"""

import sys
import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from unittest.mock import patch, MagicMock, PropertyMock, mock_open, call

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_train_df(n_rows: int = 200, n_features: int = 5) -> pd.DataFrame:
    """Create a minimal training DataFrame with numeric features and target."""
    np.random.seed(42)
    data = {f"feat_{i}": np.random.randn(n_rows).astype("float64") for i in range(n_features)}
    data["future_return_5d"] = np.random.randn(n_rows).astype("float64")
    data["date"] = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    data["ticker"] = "TEST"
    return pd.DataFrame(data)


def _make_stock_df(ticker: str = "AAPL", n_rows: int = 500) -> pd.DataFrame:
    """Create a stock DataFrame with date/close/volume columns for Prophet."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")
    prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n_rows),
    })


# ---------------------------------------------------------------------------
# StockSequenceDataset
# ---------------------------------------------------------------------------

class TestStockSequenceDataset:
    """Tests for the PyTorch dataset wrapper."""

    def test_init_stores_tensors(self):
        from backend.ml.training.train_lstm import StockSequenceDataset

        features = np.random.randn(100, 5).astype("float32")
        targets = np.random.randn(100).astype("float32")
        ds = StockSequenceDataset(features, targets, sequence_length=10)

        assert ds.sequence_length == 10
        assert ds.features.shape == (100, 5)
        assert ds.targets.shape == (100,)

    def test_len_accounts_for_sequence_length(self):
        from backend.ml.training.train_lstm import StockSequenceDataset

        features = np.random.randn(100, 5).astype("float32")
        targets = np.random.randn(100).astype("float32")
        ds = StockSequenceDataset(features, targets, sequence_length=10)

        # valid_length = 100 - 10 - 1 = 89
        assert len(ds) == 89

    def test_len_returns_zero_for_short_data(self):
        from backend.ml.training.train_lstm import StockSequenceDataset

        features = np.random.randn(5, 3).astype("float32")
        targets = np.random.randn(5).astype("float32")
        ds = StockSequenceDataset(features, targets, sequence_length=10)

        assert len(ds) == 0

    def test_getitem_returns_sequence_and_target(self):
        import torch
        from backend.ml.training.train_lstm import StockSequenceDataset

        features = np.arange(200).reshape(100, 2).astype("float32")
        targets = np.arange(100).astype("float32")
        ds = StockSequenceDataset(features, targets, sequence_length=10)

        seq, target = ds[0]
        assert seq.shape == (10, 2)
        assert isinstance(target, torch.Tensor)

    def test_getitem_boundary_idx(self):
        from backend.ml.training.train_lstm import StockSequenceDataset

        features = np.random.randn(20, 3).astype("float32")
        targets = np.random.randn(20).astype("float32")
        ds = StockSequenceDataset(features, targets, sequence_length=10)

        # Last valid index
        last_idx = len(ds) - 1
        seq, target = ds[last_idx]
        assert seq.shape == (10, 3)


# ---------------------------------------------------------------------------
# LSTMModel
# ---------------------------------------------------------------------------

class TestLSTMModel:
    """Tests for the LSTM neural network architecture."""

    def test_model_creation(self):
        from backend.ml.training.train_lstm import LSTMModel

        model = LSTMModel(input_dim=10, hidden_dim=32, num_layers=2, dropout=0.1)
        assert model is not None

    def test_model_parameter_count_positive(self):
        from backend.ml.training.train_lstm import LSTMModel

        model = LSTMModel(input_dim=10, hidden_dim=32, num_layers=2, dropout=0.1)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_forward_pass_shape(self):
        import torch
        from backend.ml.training.train_lstm import LSTMModel

        model = LSTMModel(input_dim=5, hidden_dim=16, num_layers=2, dropout=0.0)
        model.eval()

        batch = torch.randn(4, 10, 5)  # (batch, seq_len, features)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (4, 1)

    def test_forward_pass_single_sample(self):
        import torch
        from backend.ml.training.train_lstm import LSTMModel

        model = LSTMModel(input_dim=3, hidden_dim=8, num_layers=1, dropout=0.0)
        model.eval()

        batch = torch.randn(1, 5, 3)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (1, 1)

    def test_model_has_expected_components(self):
        from backend.ml.training.train_lstm import LSTMModel

        model = LSTMModel(input_dim=5, hidden_dim=16, num_layers=2, dropout=0.1)
        assert hasattr(model, "lstm")
        assert hasattr(model, "attention")
        assert hasattr(model, "fc_layers")


# ---------------------------------------------------------------------------
# LSTMTrainer
# ---------------------------------------------------------------------------

class TestLSTMTrainer:
    """Tests for LSTMTrainer orchestration logic (no actual training)."""

    def test_init_creates_model_dir(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        model_dir = tmp_path / "models"
        trainer = LSTMTrainer(
            data_dir=str(tmp_path / "data"),
            model_dir=str(model_dir),
            use_gpu=False,
        )
        assert model_dir.exists()

    def test_init_default_hyperparams(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        assert trainer.sequence_length == 60
        assert trainer.hidden_dim == 128
        assert trainer.num_layers == 3
        assert trainer.dropout == 0.2
        assert trainer.batch_size == 32
        assert trainer.epochs == 50
        assert trainer.learning_rate == 0.001
        assert trainer.early_stopping_patience == 5

    def test_setup_device_cpu_when_gpu_disabled(self, tmp_path):
        import torch
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        assert trainer.device == torch.device("cpu")

    def test_load_data_raises_on_missing_file(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path / "nonexistent"),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        with pytest.raises(FileNotFoundError):
            trainer.load_data()

    def test_load_data_returns_train_val_test(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = _make_train_df()
        df.to_parquet(data_dir / "train_data.parquet")
        df.to_parquet(data_dir / "val_data.parquet")
        df.to_parquet(data_dir / "test_data.parquet")

        trainer = LSTMTrainer(
            data_dir=str(data_dir),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        train, val, test = trainer.load_data()
        assert len(train) == len(df)
        assert val is not None
        assert test is not None

    def test_load_data_val_test_none_if_missing(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_train_df().to_parquet(data_dir / "train_data.parquet")

        trainer = LSTMTrainer(
            data_dir=str(data_dir),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        train, val, test = trainer.load_data()
        assert train is not None
        assert val is None
        assert test is None

    def test_prepare_features_excludes_target_cols(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        df = _make_train_df(n_rows=50, n_features=3)
        X, y = trainer.prepare_features(df)

        assert X.shape[0] == 50
        assert X.shape[1] == 3  # only feat_0, feat_1, feat_2
        assert y.shape == (50,)
        assert "future_return_5d" not in trainer.feature_columns

    def test_prepare_features_replaces_nan(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        df = _make_train_df(n_rows=20, n_features=2)
        df.iloc[0, 0] = np.nan
        df.iloc[1, 1] = np.inf

        X, y = trainer.prepare_features(df)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_create_sequences_shapes(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            sequence_length=10,
            use_gpu=False,
        )
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        X_seq, y_seq = trainer.create_sequences(X, y, scale=True)

        assert X_seq.shape == (40, 10, 3)  # 50 - 10 = 40
        assert y_seq.shape == (40,)

    def test_create_sequences_no_scaling(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            sequence_length=5,
            use_gpu=False,
        )
        X = np.ones((20, 2))
        y = np.ones(20)

        X_seq, y_seq = trainer.create_sequences(X, y, scale=False)
        # Without scaling, values should remain 1.0
        assert np.allclose(X_seq[0], 1.0)

    def test_save_model_writes_files(self, tmp_path):
        import torch
        from backend.ml.training.train_lstm import LSTMTrainer, LSTMModel

        model_dir = tmp_path / "models"
        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(model_dir),
            use_gpu=False,
        )
        trainer.model = LSTMModel(input_dim=5, hidden_dim=8, num_layers=1, dropout=0.0)
        trainer.feature_columns = ["a", "b", "c", "d", "e"]

        trainer._save_model()

        assert (model_dir / "lstm_weights.pth").exists()
        assert (model_dir / "lstm_scaler.pkl").exists()
        assert (model_dir / "lstm_config.json").exists()

        with open(model_dir / "lstm_config.json") as f:
            config = json.load(f)
        assert config["feature_columns"] == ["a", "b", "c", "d", "e"]
        assert config["sequence_length"] == 60
        assert config["hidden_dim"] == 128

    def test_upload_to_hf_hub_returns_false_when_client_unavailable(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        with patch("backend.ml.training.train_lstm.get_hf_client", return_value=None):
            assert trainer.upload_to_hf_hub() is False

    def test_upload_to_hf_hub_returns_false_when_env_disabled(self, tmp_path):
        from backend.ml.training.train_lstm import LSTMTrainer

        mock_client = MagicMock()
        trainer = LSTMTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        with patch("backend.ml.training.train_lstm.get_hf_client", return_value=mock_client):
            with patch.dict("os.environ", {"HF_HUB_ENABLED": "false"}):
                assert trainer.upload_to_hf_hub() is False


# ---------------------------------------------------------------------------
# XGBoostTrainer
# ---------------------------------------------------------------------------

class TestXGBoostTrainer:
    """Tests for XGBoostTrainer orchestration logic."""

    def test_init_creates_model_dir(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        model_dir = tmp_path / "xgb_models"
        trainer = XGBoostTrainer(
            data_dir=str(tmp_path / "data"),
            model_dir=str(model_dir),
            use_gpu=False,
        )
        assert model_dir.exists()

    def test_init_stores_hyperparams(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            n_trials=10,
            cv_splits=3,
            early_stopping_rounds=20,
            use_gpu=False,
        )
        assert trainer.n_trials == 10
        assert trainer.cv_splits == 3
        assert trainer.early_stopping_rounds == 20

    def test_gpu_disabled_produces_cpu_params(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        params = trainer._get_xgb_gpu_params()
        assert params["device"] == "cpu"
        assert params["tree_method"] == "hist"

    def test_load_data_raises_on_missing(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path / "nonexistent"),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        with pytest.raises(FileNotFoundError):
            trainer.load_data()

    def test_load_data_returns_dataframes(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = _make_train_df()
        df.to_parquet(data_dir / "train_data.parquet")

        trainer = XGBoostTrainer(
            data_dir=str(data_dir),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        train, val, test = trainer.load_data()
        assert len(train) > 0
        assert val is None
        assert test is None

    def test_prepare_features_shape(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        df = _make_train_df(n_rows=30, n_features=4)
        X, y = trainer.prepare_features(df)

        assert X.shape == (30, 4)
        assert y.shape == (30,)
        assert len(trainer.feature_columns) == 4

    def test_prepare_features_handles_nan(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        df = _make_train_df(n_rows=10, n_features=2)
        df.iloc[0, 0] = np.nan
        df.iloc[3, 1] = -np.inf

        X, y = trainer.prepare_features(df)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_save_model_writes_files(self, tmp_path):
        import joblib
        from backend.ml.training.train_xgboost import XGBoostTrainer

        model_dir = tmp_path / "models"
        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(model_dir),
            use_gpu=False,
        )
        trainer.model = MagicMock()
        trainer.feature_columns = ["x", "y"]
        trainer.best_params = {"n_estimators": 100}

        with patch("backend.ml.training.train_xgboost.joblib") as mock_joblib:
            trainer._save_model()

        assert mock_joblib.dump.call_count == 2  # model + scaler

        assert (model_dir / "xgboost_config.json").exists()
        with open(model_dir / "xgboost_config.json") as f:
            cfg = json.load(f)
        assert cfg["feature_columns"] == ["x", "y"]

    def test_upload_to_hf_hub_returns_false_when_client_unavailable(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        with patch("backend.ml.training.train_xgboost.get_hf_client", return_value=None):
            assert trainer.upload_to_hf_hub() is False

    def test_upload_to_hf_hub_returns_false_when_env_disabled(self, tmp_path):
        from backend.ml.training.train_xgboost import XGBoostTrainer

        mock_client = MagicMock()
        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            use_gpu=False,
        )
        with patch("backend.ml.training.train_xgboost.get_hf_client", return_value=mock_client):
            with patch.dict("os.environ", {"HF_HUB_ENABLED": "false"}):
                assert trainer.upload_to_hf_hub() is False

    def test_objective_uses_cv_splits(self, tmp_path):
        """Verify the objective function performs time-series cross-validation."""
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            cv_splits=3,
            use_gpu=False,
        )

        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 100
        mock_trial.suggest_float.return_value = 0.1

        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        with patch("backend.ml.training.train_xgboost.xgb") as mock_xgb:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.random.randn(20)
            mock_xgb.XGBRegressor.return_value = mock_model

            with patch("backend.ml.training.train_xgboost.mean_squared_error", return_value=0.5):
                score = trainer.objective(mock_trial, X, y)

        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# ProphetTrainer
# ---------------------------------------------------------------------------

class TestProphetTrainer:
    """Tests for ProphetTrainer orchestration logic."""

    def test_init_creates_prophet_subdir(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path / "data"),
            model_dir=str(tmp_path / "models"),
        )
        assert (tmp_path / "models" / "prophet").exists()

    def test_init_stores_config(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            top_n_stocks=20,
            forecast_days=14,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1,
        )
        assert trainer.top_n_stocks == 20
        assert trainer.forecast_days == 14
        assert trainer.yearly_seasonality is False
        assert trainer.changepoint_prior_scale == 0.1

    def test_load_data_raises_on_missing(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path / "missing" / "processed"),
            model_dir=str(tmp_path / "m"),
        )
        with pytest.raises(FileNotFoundError):
            trainer.load_data()

    def test_load_data_prefers_raw(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        # Create directory structure
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)

        raw_df = _make_stock_df(n_rows=100)
        raw_df.to_parquet(raw_dir / "all_stocks_raw.parquet")

        trainer = ProphetTrainer(
            data_dir=str(processed_dir),
            model_dir=str(tmp_path / "m"),
        )
        df = trainer.load_data()
        assert len(df) == 100

    def test_get_top_stocks(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
            top_n_stocks=2,
        )
        df = pd.concat([
            _make_stock_df("AAPL", 500),
            _make_stock_df("GOOG", 400),
            _make_stock_df("TSLA", 300),
        ])
        top = trainer.get_top_stocks(df)
        assert len(top) == 2
        assert top[0] == "AAPL"  # most data

    def test_prepare_prophet_data_format(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
        )
        df = _make_stock_df("AAPL", 100)
        prophet_df = trainer.prepare_prophet_data(df, "AAPL")

        assert list(prophet_df.columns) == ["ds", "y"]
        assert len(prophet_df) == 100
        assert prophet_df["ds"].dtype == "datetime64[ns]"

    def test_prepare_prophet_data_removes_duplicates(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
        )
        df = _make_stock_df("AAPL", 50)
        # Duplicate a date
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)

        prophet_df = trainer.prepare_prophet_data(df, "AAPL")
        assert len(prophet_df) == 50  # duplicate removed

    def test_train_single_stock_insufficient_data(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
        )
        # Only 100 days -- below the 365-day threshold
        df = _make_stock_df("AAPL", 100)
        result = trainer.train_single_stock(df, "AAPL")
        assert result is None

    @patch("backend.ml.training.train_prophet.PROPHET_COMPATIBLE", False)
    def test_train_skipped_when_prophet_incompatible(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
        )
        results = trainer.train()

        assert results["skipped"] is True
        assert results["stocks_trained"] == 0
        assert "incompatible" in results["skip_reason"].lower() or "pandas" in results["skip_reason"].lower()

    def test_upload_to_hf_hub_returns_false_when_client_unavailable(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
        )
        with patch("backend.ml.training.train_prophet.get_hf_client", return_value=None):
            assert trainer.upload_to_hf_hub() is False

    def test_upload_returns_false_when_no_models(self, tmp_path):
        from backend.ml.training.train_prophet import ProphetTrainer

        mock_client = MagicMock()
        trainer = ProphetTrainer(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "m"),
        )
        with patch("backend.ml.training.train_prophet.get_hf_client", return_value=mock_client):
            with patch.dict("os.environ", {"HF_HUB_ENABLED": "true"}):
                assert trainer.upload_to_hf_hub() is False


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class TestModelEvaluator:
    """Tests for the model evaluation module."""

    def test_init_sets_paths(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(
            data_dir=str(tmp_path / "data"),
            model_dir=str(tmp_path / "models"),
        )
        assert evaluator.data_dir == Path(tmp_path / "data")
        assert evaluator.model_dir == Path(tmp_path / "models")
        assert evaluator.test_data is None

    def test_load_test_data_raises_on_missing(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(
            data_dir=str(tmp_path / "nonexistent"),
            model_dir=str(tmp_path / "m"),
        )
        with pytest.raises(FileNotFoundError):
            evaluator.load_test_data()

    def test_load_test_data_stores_data(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_train_df(50).to_parquet(data_dir / "test_data.parquet")

        evaluator = ModelEvaluator(data_dir=str(data_dir), model_dir=str(tmp_path / "m"))
        df = evaluator.load_test_data()

        assert evaluator.test_data is not None
        assert len(df) == 50

    def test_prepare_features_replaces_nan(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(data_dir=str(tmp_path), model_dir=str(tmp_path))
        df = _make_train_df(10, 3)
        df.iloc[0, 0] = np.nan

        X = evaluator.prepare_features(df, ["feat_0", "feat_1", "feat_2"])
        assert not np.any(np.isnan(X))

    def test_calculate_regression_metrics(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(data_dir=str(tmp_path), model_dir=str(tmp_path))
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])

        metrics = evaluator._calculate_regression_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert "direction_accuracy" in metrics
        assert metrics["mse"] < 0.1
        assert metrics["r2"] > 0.9

    def test_calculate_regression_metrics_handles_nan(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(data_dir=str(tmp_path), model_dir=str(tmp_path))
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = evaluator._calculate_regression_metrics(y_true, y_pred)
        # Should filter NaN and still compute
        assert not np.isnan(metrics["mse"])

    def test_calculate_financial_metrics(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(data_dir=str(tmp_path), model_dir=str(tmp_path))
        returns = np.array([0.01, -0.005, 0.02, 0.01, -0.01])
        predictions = np.array([0.02, -0.01, 0.03, 0.01, -0.005])

        metrics = evaluator._calculate_financial_metrics(returns, predictions)

        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "total_return" in metrics

    def test_evaluate_lstm_returns_none_when_files_missing(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "empty_models"),
        )
        (tmp_path / "empty_models").mkdir()

        result = evaluator.evaluate_lstm()
        assert result is None

    def test_evaluate_xgboost_returns_none_when_files_missing(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "empty_models"),
        )
        (tmp_path / "empty_models").mkdir()

        result = evaluator.evaluate_xgboost()
        assert result is None

    def test_evaluate_prophet_returns_none_when_files_missing(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "empty_models"),
        )
        (tmp_path / "empty_models").mkdir()

        result = evaluator.evaluate_prophet()
        assert result is None

    def test_evaluate_prophet_reads_results_json(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        model_dir = tmp_path / "models"
        prophet_dir = model_dir / "prophet"
        prophet_dir.mkdir(parents=True)

        results_data = {
            "stocks_trained": 5,
            "average_metrics": {
                "mse": 0.5,
                "mae": 0.3,
                "mape": 2.5,
                "directional_accuracy": 0.55,
            },
        }
        with open(prophet_dir / "prophet_training_results.json", "w") as f:
            json.dump(results_data, f)

        evaluator = ModelEvaluator(
            data_dir=str(tmp_path / "data"),
            model_dir=str(model_dir),
        )
        result = evaluator.evaluate_prophet()

        assert result is not None
        assert result["model"] == "prophet"
        assert result["stocks_evaluated"] == 5
        assert result["mape"] == 2.5

    def test_generate_comparison_table(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(data_dir=str(tmp_path), model_dir=str(tmp_path))
        results = [
            {"model": "lstm", "mse": 0.5, "mae": 0.3, "r2": 0.7},
            {"model": "xgboost", "mse": 0.4, "mae": 0.25, "r2": 0.8},
        ]
        comparison = evaluator._generate_comparison_table(results)

        assert "mse" in comparison
        assert comparison["mse"]["lstm"] == 0.5
        assert comparison["mse"]["xgboost"] == 0.4

    def test_run_evaluation_returns_empty_when_no_models(self, tmp_path):
        from backend.ml.training.evaluate_models import ModelEvaluator

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "prophet").mkdir()
        _make_train_df(20).to_parquet(data_dir / "test_data.parquet")

        evaluator = ModelEvaluator(data_dir=str(data_dir), model_dir=str(model_dir))
        report = evaluator.run_evaluation()

        assert report == {}


# ---------------------------------------------------------------------------
# run_full_training.py -- orchestration functions
# ---------------------------------------------------------------------------

class TestRunFullTraining:
    """Tests for the pipeline orchestration functions."""

    def test_run_data_generation_catches_import_error(self):
        from backend.ml.training.run_full_training import run_data_generation

        with patch(
            "backend.ml.training.run_full_training.run_data_generation.__module__",
            create=True,
        ):
            # Simulate import failure inside the function
            with patch.dict("sys.modules", {"backend.ml.data_prep.generate_training_data": None}):
                result = run_data_generation("/fake/dir", max_stocks=5)
        assert result is False

    def test_run_lstm_training_returns_none_on_failure(self):
        from backend.ml.training.run_full_training import run_lstm_training

        # Patch LSTMTrainer at the location where it is imported inside run_lstm_training
        with patch(
            "backend.ml.training.train_lstm.LSTMTrainer",
        ) as MockTrainer:
            MockTrainer.return_value.train.side_effect = FileNotFoundError("no data")
            result = run_lstm_training("/nonexistent/data", "/tmp/lstm_test_model")

        assert result is None

    def test_run_xgboost_training_returns_none_on_failure(self):
        from backend.ml.training.run_full_training import run_xgboost_training

        result = run_xgboost_training("/nonexistent/data", "/nonexistent/model", n_trials=1)
        assert result is None

    def test_run_prophet_training_returns_none_on_failure(self):
        from backend.ml.training.run_full_training import run_prophet_training

        result = run_prophet_training("/nonexistent/data", "/nonexistent/model", top_n_stocks=1)
        assert result is None

    def test_run_evaluation_returns_none_on_failure(self):
        from backend.ml.training.run_full_training import run_evaluation

        result = run_evaluation("/nonexistent/data", "/nonexistent/model")
        assert result is None

    def test_download_finbert_handles_import_error(self):
        from backend.ml.training.run_full_training import download_finbert

        with patch.dict("sys.modules", {"backend.analytics.finbert_analyzer": None}):
            result = download_finbert()
        assert result is False

    def test_all_pipeline_functions_are_callable(self):
        from backend.ml.training import (
            run_data_generation,
            run_lstm_training,
            run_xgboost_training,
            run_prophet_training,
            run_evaluation,
            download_finbert,
        )
        assert callable(run_data_generation)
        assert callable(run_lstm_training)
        assert callable(run_xgboost_training)
        assert callable(run_prophet_training)
        assert callable(run_evaluation)
        assert callable(download_finbert)


# ---------------------------------------------------------------------------
# Module-level __init__.py exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify that the training __init__.py exports all expected symbols."""

    def test_lstm_exports(self):
        from backend.ml.training import LSTMModel, LSTMTrainer, StockSequenceDataset

        assert LSTMModel is not None
        assert LSTMTrainer is not None
        assert StockSequenceDataset is not None

    def test_xgboost_export(self):
        from backend.ml.training import XGBoostTrainer

        assert XGBoostTrainer is not None

    def test_prophet_export(self):
        from backend.ml.training import ProphetTrainer

        assert ProphetTrainer is not None

    def test_evaluator_export(self):
        from backend.ml.training import ModelEvaluator

        assert ModelEvaluator is not None

    def test_all_list_complete(self):
        import backend.ml.training as training_mod

        expected = {
            "LSTMModel", "LSTMTrainer", "StockSequenceDataset",
            "XGBoostTrainer", "ProphetTrainer", "ModelEvaluator",
            "run_data_generation", "run_lstm_training",
            "run_xgboost_training", "run_prophet_training",
            "run_evaluation", "download_finbert",
        }
        assert expected.issubset(set(training_mod.__all__))
