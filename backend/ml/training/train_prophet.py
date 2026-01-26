#!/usr/bin/env python3
"""
Prophet Model Training Script
Trains Prophet time series models for stock price forecasting.

NOTE: Prophet 1.2.1 has compatibility issues with pandas >= 3.0.
This module will attempt to use Prophet but may fall back to a
simpler time series approach if Prophet fails.

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
import warnings

import numpy as np
import pandas as pd
import joblib

# HuggingFace Hub integration (lazy import to avoid import errors if not installed)
def get_hf_client():
    """Lazy load HF Hub client."""
    try:
        from backend.ml.hf_hub_client import get_hf_hub_client
        return get_hf_hub_client()
    except ImportError:
        return None

# Check pandas version for Prophet compatibility
PANDAS_VERSION = tuple(int(x) for x in pd.__version__.split('.')[:2])
PROPHET_COMPATIBLE = PANDAS_VERSION < (3, 0)

if PROPHET_COMPATIBLE:
    from prophet import Prophet
    # Suppress Prophet logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
else:
    Prophet = None  # Prophet not compatible with pandas >= 3.0

warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProphetTrainer:
    """Trainer class for Prophet time series models."""

    def __init__(
        self,
        data_dir: str = 'data/ml_training/processed',
        model_dir: str = 'ml_models',
        top_n_stocks: int = 50,
        forecast_days: int = 30,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir) / 'prophet'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.top_n_stocks = top_n_stocks
        self.forecast_days = forecast_days
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale

        self.trained_models = {}
        self.training_results = {}

    def load_data(self) -> pd.DataFrame:
        """Load training data (use raw data for full history)."""
        # Try raw data first for full history (Prophet needs 365+ days)
        raw_path = self.data_dir.parent / 'raw' / 'all_stocks_raw.parquet'
        train_path = self.data_dir / 'train_data.parquet'

        if raw_path.exists():
            df = pd.read_parquet(raw_path)
            logger.info(f"Loaded {len(df)} samples from raw data")
        elif train_path.exists():
            df = pd.read_parquet(train_path)
            logger.info(f"Loaded {len(df)} samples from train data")
        else:
            raise FileNotFoundError(f"Training data not found")

        return df

    def get_top_stocks(self, df: pd.DataFrame) -> List[str]:
        """Get top N stocks by data availability."""
        stock_counts = df.groupby('ticker').size().sort_values(ascending=False)
        top_stocks = stock_counts.head(self.top_n_stocks).index.tolist()
        logger.info(f"Selected {len(top_stocks)} stocks for training")
        return top_stocks

    def prepare_prophet_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare data in Prophet format for a single stock."""
        stock_df = df[df['ticker'] == ticker].copy()
        stock_df = stock_df.sort_values('date')

        # Prophet requires columns named 'ds' and 'y'
        # Remove timezone since Prophet doesn't support it
        dates = pd.to_datetime(stock_df['date'])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)

        # Note: We don't use volume regressor due to pandas/Prophet compatibility issues
        prophet_df = pd.DataFrame({
            'ds': dates.values,
            'y': stock_df['close'].values
        })

        # Remove rows with NaN and drop duplicate dates (keep last)
        prophet_df = prophet_df.dropna()
        prophet_df = prophet_df.drop_duplicates(subset=['ds'], keep='last')
        prophet_df = prophet_df.reset_index(drop=True)

        return prophet_df

    def train_single_stock(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Train Prophet model for a single stock."""
        try:
            prophet_df = self.prepare_prophet_data(df, ticker)

            if len(prophet_df) < 365:  # Need at least 1 year of data
                logger.warning(f"Insufficient data for {ticker}: {len(prophet_df)} days")
                return None

            # Create and configure model (no regressors due to pandas compatibility)
            model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale
            )

            # Fit model
            model.fit(prophet_df)

            # Make future dataframe for evaluation
            future = model.make_future_dataframe(periods=self.forecast_days)

            # Predict
            forecast = model.predict(future)

            # Calculate metrics on historical data
            # Merge on dates to handle any mismatches
            merged = pd.merge(
                prophet_df[['ds', 'y']],
                forecast[['ds', 'yhat']],
                on='ds',
                how='inner'
            )
            actual = merged['y'].values
            predicted = merged['yhat'].values

            mse = np.mean((actual - predicted) ** 2)
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            # Calculate directional accuracy
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            directional_accuracy = np.mean(actual_direction == predicted_direction)

            results = {
                'ticker': ticker,
                'data_points': len(prophet_df),
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy),
                'forecast': {
                    'last_date': str(forecast['ds'].iloc[-1]),
                    'last_prediction': float(forecast['yhat'].iloc[-1]),
                    'lower_bound': float(forecast['yhat_lower'].iloc[-1]),
                    'upper_bound': float(forecast['yhat_upper'].iloc[-1])
                }
            }

            # Save model
            model_path = self.model_dir / f'{ticker}_model.pkl'
            joblib.dump(model, model_path)

            return results

        except Exception as e:
            logger.error(f"Error training Prophet for {ticker}: {e}")
            return None

    def train(self) -> Dict[str, Any]:
        """Train Prophet models for top stocks."""
        logger.info("="*60)
        logger.info("Starting Prophet Training")
        logger.info("="*60)

        # Check Prophet compatibility
        if not PROPHET_COMPATIBLE:
            logger.warning(f"Prophet is not compatible with pandas {pd.__version__}")
            logger.warning("Prophet requires pandas < 3.0. Skipping Prophet training.")
            logger.warning("LSTM and XGBoost models will be used for predictions.")

            results = {
                'model_type': 'prophet',
                'training_completed': datetime.now().isoformat(),
                'stocks_trained': 0,
                'stocks_failed': 0,
                'skipped': True,
                'skip_reason': f'Prophet incompatible with pandas {pd.__version__}',
                'average_metrics': {
                    'mse': None,
                    'mae': None,
                    'mape': None,
                    'directional_accuracy': None
                },
                'config': {
                    'yearly_seasonality': self.yearly_seasonality,
                    'weekly_seasonality': self.weekly_seasonality,
                    'daily_seasonality': self.daily_seasonality,
                    'changepoint_prior_scale': self.changepoint_prior_scale,
                    'forecast_days': self.forecast_days
                },
                'stock_results': []
            }

            # Save results
            results_path = self.model_dir / 'prophet_training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info("="*60)
            logger.info("Prophet training skipped due to version incompatibility")
            logger.info("="*60)
            return results

        # Load data
        df = self.load_data()

        # Get top stocks
        top_stocks = self.get_top_stocks(df)

        # Train models
        successful = 0
        failed = 0
        all_results = []

        for i, ticker in enumerate(top_stocks):
            logger.info(f"Training {i+1}/{len(top_stocks)}: {ticker}")

            result = self.train_single_stock(df, ticker)

            if result:
                all_results.append(result)
                successful += 1
            else:
                failed += 1

        # Aggregate results
        avg_mse = np.mean([r['mse'] for r in all_results])
        avg_mae = np.mean([r['mae'] for r in all_results])
        avg_mape = np.mean([r['mape'] for r in all_results])
        avg_dir_acc = np.mean([r['directional_accuracy'] for r in all_results])

        results = {
            'model_type': 'prophet',
            'training_completed': datetime.now().isoformat(),
            'stocks_trained': successful,
            'stocks_failed': failed,
            'average_metrics': {
                'mse': float(avg_mse),
                'mae': float(avg_mae),
                'mape': float(avg_mape),
                'directional_accuracy': float(avg_dir_acc)
            },
            'config': {
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality,
                'changepoint_prior_scale': self.changepoint_prior_scale,
                'forecast_days': self.forecast_days
            },
            'stock_results': all_results
        }

        # Save summary results
        results_path = self.model_dir / 'prophet_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save list of trained stocks
        stocks_path = self.model_dir / 'trained_stocks.json'
        with open(stocks_path, 'w') as f:
            json.dump([r['ticker'] for r in all_results], f, indent=2)

        logger.info("="*60)
        logger.info(f"Training complete! {successful} models trained")
        logger.info(f"Average MAPE: {avg_mape:.2f}%")
        logger.info(f"Average Directional Accuracy: {avg_dir_acc:.2%}")
        logger.info(f"Models saved to {self.model_dir}")
        logger.info("="*60)

        return results

    def upload_to_hf_hub(self, version: str = None, version_type: str = "patch") -> bool:
        """
        Upload trained Prophet models to HuggingFace Hub.

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

        # Check if we have any trained models
        model_files = list(self.model_dir.glob("*_model.pkl"))
        if not model_files:
            logger.warning("No Prophet models found to upload")
            return False

        logger.info("Uploading Prophet models to HuggingFace Hub...")

        # Create temp directory with model files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            models_subdir = temp_path / "models"
            models_subdir.mkdir()

            # Copy stock model files
            for model_file in model_files:
                shutil.copy(model_file, models_subdir / model_file.name)
                logger.info(f"  Prepared {model_file.name}")

            # Copy metadata files
            metadata_files = ["trained_stocks.json", "prophet_training_results.json"]
            for f in metadata_files:
                src = self.model_dir / f
                if src.exists():
                    shutil.copy(src, temp_path / f)
                    logger.info(f"  Prepared {f}")

            # Determine version
            if not version:
                versions = hf_client.list_versions("prophet")
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
                model_name="prophet",
                version=version,
                local_dir=temp_path,
                commit_message=f"Prophet models v{version} - {len(model_files)} stocks - trained {datetime.now(timezone.utc).isoformat()}",
                metadata={
                    "model_type": "prophet",
                    "stock_count": len(model_files),
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "yearly_seasonality": self.yearly_seasonality,
                    "weekly_seasonality": self.weekly_seasonality,
                    "forecast_days": self.forecast_days,
                }
            )

            if result:
                logger.info(f"Prophet models uploaded to HF Hub as v{version}")
                return True
            else:
                logger.error("Failed to upload Prophet models to HF Hub")
                return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Prophet models')
    parser.add_argument('--data-dir', type=str, default='data/ml_training/processed')
    parser.add_argument('--model-dir', type=str, default='ml_models')
    parser.add_argument('--top-n-stocks', type=int, default=50)
    parser.add_argument('--forecast-days', type=int, default=30)
    parser.add_argument('--upload-to-hf', action='store_true',
                        help='Upload models to HuggingFace Hub after training')
    parser.add_argument('--hf-version', type=str, default=None,
                        help='Specific version for HF upload (e.g., "1.2.0")')
    parser.add_argument('--hf-version-type', type=str, default='patch',
                        choices=['major', 'minor', 'patch'],
                        help='Version bump type if auto-incrementing')

    args = parser.parse_args()

    trainer = ProphetTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        top_n_stocks=args.top_n_stocks,
        forecast_days=args.forecast_days
    )

    results = trainer.train()
    avg = results.get('average_metrics', {})
    mape = avg.get('mape')
    if mape is not None:
        print(f"\nTraining complete! Average MAPE: {mape:.2f}%")
    else:
        print(f"\nTraining complete! (Prophet skipped due to compatibility)")

    # Upload to HuggingFace Hub if requested
    if args.upload_to_hf and not results.get('skipped'):
        trainer.upload_to_hf_hub(
            version=args.hf_version,
            version_type=args.hf_version_type
        )


if __name__ == '__main__':
    main()
