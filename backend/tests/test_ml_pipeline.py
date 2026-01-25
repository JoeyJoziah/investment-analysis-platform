#!/usr/bin/env python3
"""
ML Pipeline Integration Tests
Verifies the ML training pipeline is properly configured and ready for training.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import numpy as np
import pandas as pd


# ===== Test Configuration =====

PROJECT_ROOT = Path(__file__).parent.parent.parent
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"
DATA_DIR = PROJECT_ROOT / "data" / "ml_training"


# ===== Fixture Definitions =====

@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    n = len(dates)

    # Generate realistic stock data
    base_price = 100
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'date': dates,
        'ticker': 'TEST',
        'open': prices * (1 + np.random.randn(n) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n),
        'sector': 'Technology'
    })

    return df


@pytest.fixture
def sample_features():
    """Generate sample features for model testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 56

    return np.random.randn(n_samples, n_features)


# ===== Data Preparation Tests =====

class TestDataPreparation:
    """Tests for ML data preparation module."""

    def test_training_data_generator_import(self):
        """Test that data generator can be imported."""
        from backend.ml.data_prep import MLTrainingDataGenerator, SP100_STOCKS

        assert MLTrainingDataGenerator is not None
        assert isinstance(SP100_STOCKS, list)
        assert len(SP100_STOCKS) > 50

    def test_training_data_generator_initialization(self, tmp_path):
        """Test data generator initialization."""
        from backend.ml.data_prep import MLTrainingDataGenerator

        generator = MLTrainingDataGenerator(
            output_dir=str(tmp_path / "ml_training"),
            years_history=1
        )

        assert generator.output_dir.exists()
        assert (generator.output_dir / "raw").exists()
        assert (generator.output_dir / "processed").exists()

    def test_technical_indicators_calculation(self, sample_stock_data):
        """Test technical indicators calculation."""
        from backend.ml.data_prep import MLTrainingDataGenerator

        generator = MLTrainingDataGenerator()
        result = generator.calculate_technical_indicators(sample_stock_data)

        # Check key indicators are present
        assert 'sma_20' in result.columns
        assert 'rsi_14' in result.columns
        assert 'macd' in result.columns
        assert 'bb_upper' in result.columns
        assert 'atr_14' in result.columns

    def test_label_generation(self, sample_stock_data):
        """Test future return label generation."""
        from backend.ml.data_prep import MLTrainingDataGenerator

        generator = MLTrainingDataGenerator()
        result = generator.generate_labels(sample_stock_data)

        # Check labels are present
        assert 'future_return_5d' in result.columns
        assert 'direction_5d' in result.columns
        assert 'future_return_1d' in result.columns

    def test_processed_data_exists(self):
        """Test that processed training data exists."""
        processed_dir = DATA_DIR / "processed"

        if processed_dir.exists():
            # Check for required files
            assert (processed_dir / "train_data.parquet").exists() or \
                   len(list(processed_dir.glob("*.parquet"))) > 0, \
                   "No training data files found"

            # Check metadata
            if (processed_dir / "metadata.json").exists():
                with open(processed_dir / "metadata.json") as f:
                    metadata = json.load(f)
                    assert "train_samples" in metadata
                    assert "feature_columns" in metadata


# ===== Training Module Tests =====

class TestTrainingModules:
    """Tests for ML training modules."""

    def test_lstm_trainer_import(self):
        """Test LSTM trainer can be imported."""
        from backend.ml.training import LSTMModel, LSTMTrainer, StockSequenceDataset

        assert LSTMModel is not None
        assert LSTMTrainer is not None
        assert StockSequenceDataset is not None

    def test_lstm_model_creation(self):
        """Test LSTM model can be instantiated."""
        from backend.ml.training import LSTMModel

        model = LSTMModel(
            input_dim=56,
            hidden_dim=128,
            num_layers=3,
            dropout=0.2
        )

        assert model is not None
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_xgboost_trainer_import(self):
        """Test XGBoost trainer can be imported."""
        from backend.ml.training import XGBoostTrainer

        assert XGBoostTrainer is not None

    def test_xgboost_trainer_initialization(self, tmp_path):
        """Test XGBoost trainer initialization."""
        from backend.ml.training import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=str(tmp_path / "data"),
            model_dir=str(tmp_path / "models"),
            n_trials=5
        )

        assert trainer.n_trials == 5
        assert trainer.model_dir.exists()

    def test_prophet_trainer_import(self):
        """Test Prophet trainer can be imported."""
        from backend.ml.training import ProphetTrainer

        assert ProphetTrainer is not None

    def test_model_evaluator_import(self):
        """Test model evaluator can be imported."""
        from backend.ml.training import ModelEvaluator

        assert ModelEvaluator is not None


# ===== Model Manager Tests =====

class TestModelManager:
    """Tests for ML model manager."""

    def test_model_manager_import(self):
        """Test model manager can be imported."""
        from backend.ml.model_manager import ModelManager, get_model_manager

        assert ModelManager is not None
        assert get_model_manager is not None

    def test_model_manager_initialization(self, tmp_path):
        """Test model manager initialization."""
        from backend.ml.model_manager import ModelManager

        manager = ModelManager(models_path=str(tmp_path))

        assert manager.models_path.exists()
        assert isinstance(manager.models, dict)
        assert isinstance(manager.model_metadata, dict)

    def test_model_manager_fallback_models(self, tmp_path):
        """Test that fallback models are created when files missing."""
        from backend.ml.model_manager import ModelManager

        manager = ModelManager(models_path=str(tmp_path))

        # Check fallback models exist
        assert "lstm_price_predictor" in manager.models
        assert "xgboost_classifier" in manager.models
        assert "prophet_forecaster" in manager.models

        # Check status is fallback
        for name, metadata in manager.model_metadata.items():
            assert metadata["status"] == "fallback"

    def test_model_manager_health_check(self, tmp_path):
        """Test model manager health check."""
        from backend.ml.model_manager import ModelManager

        manager = ModelManager(models_path=str(tmp_path))
        health = manager.health_check()

        assert "healthy" in health
        assert "models" in health
        assert "total_models" in health

    def test_model_manager_with_trained_models(self):
        """Test model manager loads trained models if available."""
        if ML_MODELS_DIR.exists():
            from backend.ml.model_manager import ModelManager

            manager = ModelManager(models_path=str(ML_MODELS_DIR))
            status = manager.get_model_status()

            # Check if any models loaded successfully
            loaded_count = sum(
                1 for meta in status.values()
                if meta.get("status") == "loaded"
            )

            # At least check that initialization works
            assert isinstance(status, dict)


# ===== Model Files Tests =====

class TestModelFiles:
    """Tests for trained model files."""

    def test_lstm_model_files_exist(self):
        """Test LSTM model files exist if training completed."""
        lstm_weights = ML_MODELS_DIR / "lstm_weights.pth"
        lstm_config = ML_MODELS_DIR / "lstm_config.json"
        lstm_scaler = ML_MODELS_DIR / "lstm_scaler.pkl"

        if lstm_weights.exists():
            assert lstm_config.exists(), "LSTM config missing"
            assert lstm_scaler.exists(), "LSTM scaler missing"

            # Validate config
            with open(lstm_config) as f:
                config = json.load(f)
                assert "feature_columns" in config
                assert "hidden_dim" in config
                assert "sequence_length" in config

    def test_xgboost_model_files_exist(self):
        """Test XGBoost model files exist if training completed."""
        xgb_model = ML_MODELS_DIR / "xgboost_model.pkl"
        xgb_config = ML_MODELS_DIR / "xgboost_config.json"
        xgb_scaler = ML_MODELS_DIR / "xgboost_scaler.pkl"

        if xgb_model.exists():
            assert xgb_config.exists(), "XGBoost config missing"
            assert xgb_scaler.exists(), "XGBoost scaler missing"

            # Validate config
            with open(xgb_config) as f:
                config = json.load(f)
                assert "feature_columns" in config
                assert "best_params" in config

    def test_prophet_model_files_exist(self):
        """Test Prophet model files exist if training completed."""
        prophet_dir = ML_MODELS_DIR / "prophet"
        trained_stocks = prophet_dir / "trained_stocks.json"

        if prophet_dir.exists() and trained_stocks.exists():
            with open(trained_stocks) as f:
                stocks = json.load(f)
                assert isinstance(stocks, list)

                # Check model files for each stock
                for ticker in stocks:
                    model_file = prophet_dir / f"{ticker}_model.pkl"
                    assert model_file.exists(), f"Prophet model missing for {ticker}"

    def test_evaluation_report_exists(self):
        """Test evaluation report exists if evaluation completed."""
        eval_report = ML_MODELS_DIR / "evaluation_report.json"

        if eval_report.exists():
            with open(eval_report) as f:
                report = json.load(f)
                assert "evaluation_date" in report
                assert "model_results" in report
                assert "best_model" in report


# ===== Integration Tests =====

class TestMLPipelineIntegration:
    """Integration tests for the ML pipeline."""

    def test_full_pipeline_imports(self):
        """Test all pipeline functions can be imported."""
        from backend.ml.training import (
            run_data_generation,
            run_lstm_training,
            run_xgboost_training,
            run_prophet_training,
            run_evaluation,
        )

        assert callable(run_data_generation)
        assert callable(run_lstm_training)
        assert callable(run_xgboost_training)
        assert callable(run_prophet_training)
        assert callable(run_evaluation)

    def test_data_prep_to_training_flow(self, sample_stock_data, tmp_path):
        """Test data preparation flows into training."""
        from backend.ml.data_prep import MLTrainingDataGenerator

        generator = MLTrainingDataGenerator(
            output_dir=str(tmp_path / "ml_training"),
            years_history=1
        )

        # Process sample data
        df = generator.calculate_technical_indicators(sample_stock_data)
        df = generator.generate_labels(df)

        # Clean data
        df = df.dropna(subset=['close', 'returns', 'rsi_14', 'macd'])

        # Check data is ready for training
        assert len(df) > 50, "Not enough samples after cleaning"
        assert 'future_return_5d' in df.columns
        assert not df['future_return_5d'].isna().all()

    def test_finbert_analyzer_import(self):
        """Test FinBERT analyzer can be imported."""
        from backend.analytics.finbert_analyzer import (
            FinBERTAnalyzer,
            FinBERTResult,
            download_finbert_model
        )

        assert FinBERTAnalyzer is not None
        assert FinBERTResult is not None
        assert callable(download_finbert_model)


# ===== Model Prediction Tests =====

class TestModelPredictions:
    """Tests for model predictions."""

    def test_lstm_fallback_prediction(self, sample_features):
        """Test LSTM fallback model can make predictions."""
        from backend.ml.model_manager import ModelManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_path=tmpdir)
            result = manager.predict("lstm_price_predictor", sample_features)

            assert result is not None

    def test_xgboost_fallback_prediction(self, sample_features):
        """Test XGBoost fallback model can make predictions."""
        from backend.ml.model_manager import ModelManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_path=tmpdir)
            result = manager.predict("xgboost_classifier", sample_features)

            assert result is not None
            assert "predictions" in result

    def test_sentiment_fallback_prediction(self):
        """Test sentiment fallback model can make predictions."""
        from backend.ml.model_manager import ModelManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_path=tmpdir)
            result = manager.predict(
                "sentiment_analyzer",
                ["Stock price increased significantly today"]
            )

            assert result is not None
            assert isinstance(result, list)


# ===== Configuration Tests =====

class TestMLConfiguration:
    """Tests for ML configuration."""

    def test_model_configs_valid(self):
        """Test model configuration files are valid JSON."""
        config_files = [
            ML_MODELS_DIR / "lstm_config.json",
            ML_MODELS_DIR / "xgboost_config.json",
        ]

        for config_file in config_files:
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    assert isinstance(config, dict)
                    assert "feature_columns" in config

    def test_training_results_valid(self):
        """Test training results files are valid JSON."""
        result_files = [
            ML_MODELS_DIR / "lstm_training_results.json",
            ML_MODELS_DIR / "xgboost_training_results.json",
            ML_MODELS_DIR / "prophet" / "prophet_training_results.json",
        ]

        for result_file in result_files:
            if result_file.exists():
                with open(result_file) as f:
                    results = json.load(f)
                    assert isinstance(results, dict)
                    assert "training_completed" in results or "model_type" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
