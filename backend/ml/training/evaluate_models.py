#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates trained ML models on test data and generates comparison report.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained ML models on test data."""

    def __init__(
        self,
        data_dir: str = 'data/ml_training/processed',
        model_dir: str = 'ml_models'
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

        self.test_data = None
        self.results = {}

    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        test_path = self.data_dir / 'test_data.parquet'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")

        self.test_data = pd.read_parquet(test_path)
        logger.info(f"Loaded {len(self.test_data)} test samples")

        return self.test_data

    def prepare_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Prepare features for prediction."""
        X = df[feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def evaluate_lstm(self) -> Optional[Dict[str, Any]]:
        """Evaluate LSTM model."""
        logger.info("Evaluating LSTM model...")

        model_path = self.model_dir / 'lstm_weights.pth'
        config_path = self.model_dir / 'lstm_config.json'
        scaler_path = self.model_dir / 'lstm_scaler.pkl'

        if not all(p.exists() for p in [model_path, config_path, scaler_path]):
            logger.warning("LSTM model files not found")
            return None

        try:
            # Load config
            with open(config_path) as f:
                config = json.load(f)

            # Import LSTM model class
            from backend.ml.training.train_lstm import LSTMModel

            # Load model
            model = LSTMModel(
                input_dim=len(config['feature_columns']),
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

            # Load scaler
            scaler = joblib.load(scaler_path)

            # Prepare test data
            X_test = self.prepare_features(self.test_data, config['feature_columns'])
            X_test_scaled = scaler.transform(X_test)
            y_test = self.test_data[config['target_column']].values

            # Create sequences
            seq_length = config['sequence_length']
            X_seq, y_seq = [], []
            for i in range(len(X_test_scaled) - seq_length):
                X_seq.append(X_test_scaled[i:i + seq_length])
                y_seq.append(y_test[i + seq_length])

            X_seq = torch.FloatTensor(np.array(X_seq))
            y_seq = np.array(y_seq)

            # Predict
            with torch.no_grad():
                predictions = model(X_seq).numpy().flatten()

            # Calculate metrics
            metrics = self._calculate_regression_metrics(y_seq, predictions)
            metrics['model'] = 'lstm'

            logger.info(f"LSTM - MSE: {metrics['mse']:.6f}, R²: {metrics['r2']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating LSTM: {e}")
            return None

    def evaluate_xgboost(self) -> Optional[Dict[str, Any]]:
        """Evaluate XGBoost model."""
        logger.info("Evaluating XGBoost model...")

        model_path = self.model_dir / 'xgboost_model.pkl'
        config_path = self.model_dir / 'xgboost_config.json'
        scaler_path = self.model_dir / 'xgboost_scaler.pkl'

        if not all(p.exists() for p in [model_path, config_path, scaler_path]):
            logger.warning("XGBoost model files not found")
            return None

        try:
            # Load config
            with open(config_path) as f:
                config = json.load(f)

            # Load model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Prepare test data
            X_test = self.prepare_features(self.test_data, config['feature_columns'])
            X_test_scaled = scaler.transform(X_test)
            y_test = self.test_data[config['target_column']].values
            y_test = np.nan_to_num(y_test, nan=0.0)

            # Predict
            predictions = model.predict(X_test_scaled)

            # Calculate metrics
            metrics = self._calculate_regression_metrics(y_test, predictions)
            metrics['model'] = 'xgboost'

            # Add classification metrics (direction prediction)
            y_direction = (y_test > 0).astype(int)
            pred_direction = (predictions > 0).astype(int)
            metrics['direction_accuracy'] = accuracy_score(y_direction, pred_direction)
            metrics['direction_f1'] = f1_score(y_direction, pred_direction)

            logger.info(f"XGBoost - MSE: {metrics['mse']:.6f}, R²: {metrics['r2']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating XGBoost: {e}")
            return None

    def evaluate_prophet(self) -> Optional[Dict[str, Any]]:
        """Evaluate Prophet models (aggregate across stocks)."""
        logger.info("Evaluating Prophet models...")

        prophet_dir = self.model_dir / 'prophet'
        results_path = prophet_dir / 'prophet_training_results.json'

        if not results_path.exists():
            logger.warning("Prophet results not found")
            return None

        try:
            with open(results_path) as f:
                prophet_results = json.load(f)

            metrics = {
                'model': 'prophet',
                'stocks_evaluated': prophet_results['stocks_trained'],
                'mse': prophet_results['average_metrics']['mse'],
                'mae': prophet_results['average_metrics']['mae'],
                'mape': prophet_results['average_metrics']['mape'],
                'direction_accuracy': prophet_results['average_metrics']['directional_accuracy']
            }

            logger.info(f"Prophet - MAPE: {metrics['mape']:.2f}%, Dir Acc: {metrics['direction_accuracy']:.2%}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating Prophet: {e}")
            return None

    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        # Filter out NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE (handle division by zero)
        nonzero_mask = y_true != 0
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        else:
            mape = 0.0

        # Directional accuracy
        y_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        direction_accuracy = np.mean(y_direction == pred_direction)

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }

    def _calculate_financial_metrics(
        self,
        returns: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate financial performance metrics."""
        # Simulated strategy: go long when prediction > 0, else stay out
        strategy_returns = np.where(predictions > 0, returns, 0)

        # Sharpe ratio (annualized, assuming daily returns)
        if strategy_returns.std() > 0:
            sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = drawdowns.max()

        # Win rate
        win_rate = (strategy_returns > 0).mean()

        # Profit factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        return {
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_return': float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation on all models."""
        logger.info("="*60)
        logger.info("Starting Model Evaluation")
        logger.info("="*60)

        # Load test data
        self.load_test_data()

        # Evaluate each model
        lstm_results = self.evaluate_lstm()
        xgboost_results = self.evaluate_xgboost()
        prophet_results = self.evaluate_prophet()

        # Compile results
        all_results = []
        if lstm_results:
            all_results.append(lstm_results)
        if xgboost_results:
            all_results.append(xgboost_results)
        if prophet_results:
            all_results.append(prophet_results)

        if not all_results:
            logger.error("No models could be evaluated")
            return {}

        # Determine best model (by MSE for regression)
        regression_results = [r for r in all_results if 'mse' in r]
        if regression_results:
            best_model = min(regression_results, key=lambda x: x['mse'])
        else:
            best_model = all_results[0]

        # Generate comparison report
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'test_samples': len(self.test_data),
            'models_evaluated': len(all_results),
            'model_results': all_results,
            'best_model': {
                'name': best_model['model'],
                'mse': best_model.get('mse'),
                'direction_accuracy': best_model.get('direction_accuracy')
            },
            'comparison': self._generate_comparison_table(all_results)
        }

        # Save report
        report_path = self.model_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("="*60)
        logger.info(f"Evaluation complete!")
        logger.info(f"Best model: {best_model['model']} (MSE: {best_model.get('mse', 'N/A')})")
        logger.info(f"Report saved to {report_path}")
        logger.info("="*60)

        return report

    def _generate_comparison_table(self, results: List[Dict]) -> Dict:
        """Generate model comparison summary."""
        comparison = {}

        for metric in ['mse', 'mae', 'r2', 'direction_accuracy']:
            values = {}
            for r in results:
                if metric in r:
                    values[r['model']] = r[metric]
            if values:
                comparison[metric] = values

        return comparison


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--data-dir', type=str, default='data/ml_training/processed')
    parser.add_argument('--model-dir', type=str, default='ml_models')

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        data_dir=args.data_dir,
        model_dir=args.model_dir
    )

    report = evaluator.run_evaluation()

    if report:
        print(f"\nBest model: {report['best_model']['name']}")
        print(f"MSE: {report['best_model'].get('mse', 'N/A')}")


if __name__ == '__main__':
    main()
