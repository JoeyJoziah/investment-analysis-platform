#!/usr/bin/env python3
"""
XGBoost Model Training Script
Trains XGBoost with Optuna hyperparameter optimization.

Supports automatic upload to HuggingFace Hub for centralized model storage.
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HuggingFace Hub integration (lazy import to avoid import errors if not installed)
def get_hf_client():
    """Lazy load HF Hub client."""
    try:
        from backend.ml.hf_hub_client import get_hf_hub_client
        return get_hf_hub_client()
    except ImportError:
        return None


class XGBoostTrainer:
    """Trainer class for XGBoost model with Optuna optimization."""

    def __init__(
        self,
        data_dir: str = 'data/ml_training/processed',
        model_dir: str = 'ml_models',
        n_trials: int = 50,
        cv_splits: int = 5,
        early_stopping_rounds: int = 50
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'future_return_5d'
        self.best_params = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training, validation, and test data."""
        train_path = self.data_dir / 'train_data.parquet'
        val_path = self.data_dir / 'val_data.parquet'
        test_path = self.data_dir / 'test_data.parquet'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path) if val_path.exists() else None
        test_df = pd.read_parquet(test_path) if test_path.exists() else None

        logger.info(f"Loaded train: {len(train_df)}, val: {len(val_df) if val_df is not None else 0}")

        return train_df, val_df, test_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        # Define feature columns (exclude non-numeric and target columns)
        exclude_cols = [
            'date', 'ticker', 'sector', 'industry',
            'future_return_1d', 'future_return_5d', 'future_return_10d', 'future_return_20d',
            'direction_1d', 'direction_5d', 'direction_10d', 'direction_20d',
            'risk_adj_return_1d', 'risk_adj_return_5d', 'risk_adj_return_10d', 'risk_adj_return_20d'
        ]

        self.feature_columns = [c for c in df.columns if c not in exclude_cols
                                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        logger.info(f"Using {len(self.feature_columns)} features")

        # Extract features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function for hyperparameter optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                verbose=False
            )

            preds = model.predict(X_val_cv)
            mse = mean_squared_error(y_val_cv, preds)
            scores.append(mse)

        return np.mean(scores)

    def train(self) -> Dict[str, Any]:
        """Train the XGBoost model with Optuna optimization."""
        logger.info("="*60)
        logger.info("Starting XGBoost Training with Optuna")
        logger.info("="*60)

        # Load data
        train_df, val_df, test_df = self.load_data()

        # Prepare features
        X_train, y_train = self.prepare_features(train_df)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        logger.info(f"Training samples: {len(X_train)}, Features: {len(self.feature_columns)}")

        # Optuna optimization
        logger.info(f"Running Optuna optimization with {self.n_trials} trials...")

        study = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
        study.optimize(
            lambda trial: self.objective(trial, X_train_scaled, y_train),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        self.best_params['random_state'] = 42
        self.best_params['n_jobs'] = -1

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best CV score (MSE): {study.best_value:.6f}")

        # Train final model with best params
        logger.info("Training final model with best parameters...")

        self.model = xgb.XGBRegressor(**self.best_params)

        # If validation set exists, use it for early stopping
        if val_df is not None:
            X_val, y_val = self.prepare_features(val_df)
            X_val_scaled = self.scaler.transform(X_val)

            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=True
            )

            # Evaluate on validation set
            val_preds = self.model.predict(X_val_scaled)
            val_mse = mean_squared_error(y_val, val_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)

            logger.info(f"Validation MSE: {val_mse:.6f}")
            logger.info(f"Validation MAE: {val_mae:.6f}")
            logger.info(f"Validation R²: {val_r2:.4f}")
        else:
            self.model.fit(X_train_scaled, y_train)
            val_mse, val_mae, val_r2 = None, None, None

        # Get feature importance
        feature_importance = {
            col: float(imp) for col, imp in zip(
                self.feature_columns,
                self.model.feature_importances_
            )
        }
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        logger.info("\nTop 20 Features:")
        for feat, imp in top_features:
            logger.info(f"  {feat}: {imp:.4f}")

        # Save model and results
        self._save_model()

        results = {
            'model_type': 'xgboost',
            'training_completed': datetime.now().isoformat(),
            'best_params': self.best_params,
            'optuna_best_score': study.best_value,
            'validation_metrics': {
                'mse': val_mse,
                'mae': val_mae,
                'r2': val_r2
            } if val_mse else None,
            'n_trials': self.n_trials,
            'feature_importance': {k: float(v) for k, v in top_features},
            'feature_columns': self.feature_columns
        }

        # Save results
        results_path = self.model_dir / 'xgboost_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save feature importance separately
        fi_path = self.model_dir / 'xgboost_feature_importance.json'
        with open(fi_path, 'w') as f:
            json.dump(feature_importance, f, indent=2)

        logger.info("="*60)
        mse_str = f'{val_mse:.6f}' if val_mse else 'N/A'
        logger.info(f"Training complete! Val MSE: {mse_str}")
        logger.info(f"Model saved to {self.model_dir}")
        logger.info("="*60)

        return results

    def upload_to_hf_hub(self, version: str = None, version_type: str = "patch") -> bool:
        """
        Upload trained model to HuggingFace Hub.

        Args:
            version: Explicit version string (e.g., "1.0.0"). If None, auto-increment.
            version_type: Type of version bump if auto-incrementing ("major", "minor", "patch")

        Returns:
            True if upload successful, False otherwise
        """
        hf_client = get_hf_client()
        if not hf_client:
            logger.warning("HuggingFace Hub client not available. Skipping upload.")
            return False

        if not os.getenv("HF_HUB_ENABLED", "false").lower() == "true":
            logger.info("HF Hub upload disabled (HF_HUB_ENABLED != true)")
            return False

        logger.info("Uploading XGBoost model to HuggingFace Hub...")

        # Create temp directory with model files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy model files with clean names
            files_to_copy = [
                ("xgboost_model.pkl", "model.pkl"),
                ("xgboost_scaler.pkl", "scaler.pkl"),
                ("xgboost_config.json", "config.json"),
                ("xgboost_training_results.json", "training_results.json"),
                ("xgboost_feature_importance.json", "feature_importance.json"),
            ]

            for src_name, dst_name in files_to_copy:
                src = self.model_dir / src_name
                if src.exists():
                    shutil.copy(src, temp_path / dst_name)
                    logger.info(f"  Prepared {src_name} -> {dst_name}")

            # Determine version
            if not version:
                versions = hf_client.list_versions("xgboost")
                if versions:
                    latest = max(versions, key=lambda v: [int(x) for x in v.split(".")])
                    parts = [int(x) for x in latest.split(".")]
                    if version_type == "major":
                        parts = [parts[0] + 1, 0, 0]
                    elif version_type == "minor":
                        parts = [parts[0], parts[1] + 1, 0]
                    else:  # patch
                        parts = [parts[0], parts[1], parts[2] + 1]
                    version = ".".join(str(p) for p in parts)
                else:
                    version = "1.0.0"

            # Upload to HF Hub
            result = hf_client.upload_model(
                model_name="xgboost",
                version=version,
                local_dir=temp_path,
                commit_message=f"XGBoost model v{version} - trained {datetime.now(timezone.utc).isoformat()}",
                metadata={
                    "model_type": "xgboost",
                    "task": "price_direction_prediction",
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "best_params": self.best_params,
                }
            )

            if result:
                logger.info(f"XGBoost model uploaded to HF Hub as v{version}")
                return True
            else:
                logger.error("Failed to upload XGBoost model to HF Hub")
                return False

    def _save_model(self):
        """Save model, scaler, and config."""
        # Save model
        model_path = self.model_dir / 'xgboost_model.pkl'
        joblib.dump(self.model, model_path)

        # Save scaler
        scaler_path = self.model_dir / 'xgboost_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)

        # Save config
        config_path = self.model_dir / 'xgboost_config.json'
        config = {
            'feature_columns': self.feature_columns,
            'best_params': self.best_params,
            'target_column': self.target_column
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {model_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--data-dir', type=str, default='data/ml_training/processed')
    parser.add_argument('--model-dir', type=str, default='ml_models')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--upload-to-hf', action='store_true',
                        help='Upload model to HuggingFace Hub after training')
    parser.add_argument('--hf-version', type=str, default=None,
                        help='Specific version for HF upload (e.g., "1.2.0")')
    parser.add_argument('--hf-version-type', type=str, default='patch',
                        choices=['major', 'minor', 'patch'],
                        help='Version bump type if auto-incrementing')

    args = parser.parse_args()

    trainer = XGBoostTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_trials=args.n_trials,
        cv_splits=args.cv_splits
    )

    results = trainer.train()
    val_mse = results.get('validation_metrics', {})
    print(f"\nTraining complete! Val MSE: {val_mse.get('mse', 'N/A') if val_mse else 'N/A'}")

    # Upload to HuggingFace Hub if requested
    if args.upload_to_hf:
        trainer.upload_to_hf_hub(
            version=args.hf_version,
            version_type=args.hf_version_type
        )


if __name__ == '__main__':
    main()
