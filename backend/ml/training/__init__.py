"""
ML Training Scripts Module
Provides training pipelines for various ML models used in stock prediction.
"""

from .train_lstm import LSTMModel, LSTMTrainer, StockSequenceDataset
from .train_xgboost import XGBoostTrainer
from .train_prophet import ProphetTrainer
from .evaluate_models import ModelEvaluator
from .run_full_training import (
    run_data_generation,
    run_lstm_training,
    run_xgboost_training,
    run_prophet_training,
    run_evaluation,
    download_finbert
)

__all__ = [
    # LSTM
    'LSTMModel',
    'LSTMTrainer',
    'StockSequenceDataset',

    # XGBoost
    'XGBoostTrainer',

    # Prophet
    'ProphetTrainer',

    # Evaluation
    'ModelEvaluator',

    # Pipeline functions
    'run_data_generation',
    'run_lstm_training',
    'run_xgboost_training',
    'run_prophet_training',
    'run_evaluation',
    'download_finbert',
]
