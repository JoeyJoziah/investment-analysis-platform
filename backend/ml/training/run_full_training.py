#!/usr/bin/env python3
"""
Master Training Orchestrator
Runs the complete ML training pipeline: data generation, model training, and evaluation.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_generation(output_dir: str, max_stocks: Optional[int] = None) -> bool:
    """Run training data generation."""
    logger.info("="*60)
    logger.info("PHASE 1: Data Generation")
    logger.info("="*60)

    try:
        from backend.ml.data_prep.generate_training_data import MLTrainingDataGenerator

        generator = MLTrainingDataGenerator(
            output_dir=output_dir,
            years_history=2
        )

        result = generator.run_data_generation(max_stocks=max_stocks)

        if result and 'train' in result:
            logger.info(f"Data generation complete: {len(result['train'])} training samples")
            return True
        else:
            logger.error("Data generation failed")
            return False

    except Exception as e:
        logger.error(f"Data generation error: {e}")
        return False


def run_lstm_training(data_dir: str, model_dir: str) -> Optional[Dict]:
    """Run LSTM model training."""
    logger.info("="*60)
    logger.info("PHASE 2a: LSTM Training")
    logger.info("="*60)

    try:
        from backend.ml.training.train_lstm import LSTMTrainer

        trainer = LSTMTrainer(
            data_dir=data_dir,
            model_dir=model_dir,
            epochs=50,
            batch_size=32,
            sequence_length=60
        )

        results = trainer.train()
        logger.info(f"LSTM training complete: Val loss = {results.get('final_val_loss', 'N/A')}")
        return results

    except Exception as e:
        logger.error(f"LSTM training error: {e}")
        return None


def run_xgboost_training(data_dir: str, model_dir: str, n_trials: int = 30) -> Optional[Dict]:
    """Run XGBoost model training."""
    logger.info("="*60)
    logger.info("PHASE 2b: XGBoost Training")
    logger.info("="*60)

    try:
        from backend.ml.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer(
            data_dir=data_dir,
            model_dir=model_dir,
            n_trials=n_trials
        )

        results = trainer.train()
        val_metrics = results.get('validation_metrics', {})
        logger.info(f"XGBoost training complete: Val MSE = {val_metrics.get('mse', 'N/A')}")
        return results

    except Exception as e:
        logger.error(f"XGBoost training error: {e}")
        return None


def run_prophet_training(data_dir: str, model_dir: str, top_n_stocks: int = 30) -> Optional[Dict]:
    """Run Prophet model training."""
    logger.info("="*60)
    logger.info("PHASE 2c: Prophet Training")
    logger.info("="*60)

    try:
        from backend.ml.training.train_prophet import ProphetTrainer

        trainer = ProphetTrainer(
            data_dir=data_dir,
            model_dir=model_dir,
            top_n_stocks=top_n_stocks
        )

        results = trainer.train()
        avg_metrics = results.get('average_metrics', {})
        logger.info(f"Prophet training complete: {results.get('stocks_trained', 0)} models")
        return results

    except Exception as e:
        logger.error(f"Prophet training error: {e}")
        return None


def run_evaluation(data_dir: str, model_dir: str) -> Optional[Dict]:
    """Run model evaluation."""
    logger.info("="*60)
    logger.info("PHASE 3: Model Evaluation")
    logger.info("="*60)

    try:
        from backend.ml.training.evaluate_models import ModelEvaluator

        evaluator = ModelEvaluator(
            data_dir=data_dir,
            model_dir=model_dir
        )

        report = evaluator.run_evaluation()
        logger.info(f"Evaluation complete: Best model = {report.get('best_model', {}).get('name', 'N/A')}")
        return report

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return None


def download_finbert() -> bool:
    """Download FinBERT model."""
    logger.info("="*60)
    logger.info("Downloading FinBERT Model")
    logger.info("="*60)

    try:
        from backend.analytics.finbert_analyzer import download_finbert_model

        success = download_finbert_model(cache_dir='ml_models/finbert')
        if success:
            logger.info("FinBERT download complete")
        return success

    except Exception as e:
        logger.error(f"FinBERT download error: {e}")
        return False


def main():
    """Main entry point for full training pipeline."""
    parser = argparse.ArgumentParser(description='Run full ML training pipeline')
    parser.add_argument('--data-dir', type=str, default='data/ml_training',
                        help='Directory for training data')
    parser.add_argument('--model-dir', type=str, default='ml_models',
                        help='Directory for trained models')
    parser.add_argument('--max-stocks', type=int, default=50,
                        help='Maximum stocks to process for data generation')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Optuna trials for XGBoost')
    parser.add_argument('--top-n-prophet', type=int, default=30,
                        help='Top N stocks for Prophet training')
    parser.add_argument('--skip-data-gen', action='store_true',
                        help='Skip data generation (use existing data)')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM training')
    parser.add_argument('--skip-xgboost', action='store_true',
                        help='Skip XGBoost training')
    parser.add_argument('--skip-prophet', action='store_true',
                        help='Skip Prophet training')
    parser.add_argument('--download-finbert', action='store_true',
                        help='Download FinBERT model')

    args = parser.parse_args()

    start_time = datetime.now()

    logger.info("="*60)
    logger.info("ML TRAINING PIPELINE")
    logger.info(f"Started at: {start_time}")
    logger.info("="*60)

    results = {
        'start_time': start_time.isoformat(),
        'phases': {}
    }

    processed_dir = f"{args.data_dir}/processed"

    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Phase 0: Download FinBERT (optional)
    if args.download_finbert:
        finbert_success = download_finbert()
        results['phases']['finbert_download'] = finbert_success

    # Phase 1: Data Generation
    if not args.skip_data_gen:
        data_success = run_data_generation(args.data_dir, args.max_stocks)
        results['phases']['data_generation'] = data_success

        if not data_success:
            logger.error("Data generation failed. Aborting pipeline.")
            return 1
    else:
        logger.info("Skipping data generation (--skip-data-gen)")

    # Phase 2: Model Training
    training_results = {}

    # LSTM
    if not args.skip_lstm:
        lstm_results = run_lstm_training(processed_dir, args.model_dir)
        training_results['lstm'] = lstm_results is not None
        results['phases']['lstm_training'] = lstm_results
    else:
        logger.info("Skipping LSTM training (--skip-lstm)")

    # XGBoost
    if not args.skip_xgboost:
        xgb_results = run_xgboost_training(processed_dir, args.model_dir, args.n_trials)
        training_results['xgboost'] = xgb_results is not None
        results['phases']['xgboost_training'] = xgb_results
    else:
        logger.info("Skipping XGBoost training (--skip-xgboost)")

    # Prophet
    if not args.skip_prophet:
        prophet_results = run_prophet_training(processed_dir, args.model_dir, args.top_n_prophet)
        training_results['prophet'] = prophet_results is not None
        results['phases']['prophet_training'] = prophet_results
    else:
        logger.info("Skipping Prophet training (--skip-prophet)")

    # Phase 3: Evaluation
    if any(training_results.values()):
        eval_results = run_evaluation(processed_dir, args.model_dir)
        results['phases']['evaluation'] = eval_results
    else:
        logger.warning("No models trained, skipping evaluation")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    results['end_time'] = end_time.isoformat()
    results['duration_seconds'] = duration.total_seconds()

    # Save pipeline results
    pipeline_results_path = Path(args.model_dir) / 'pipeline_results.json'
    with open(pipeline_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Duration: {duration}")
    logger.info(f"Results saved to: {pipeline_results_path}")
    logger.info("="*60)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Duration: {duration}")
    print(f"\nModels trained:")
    for model, success in training_results.items():
        status = "OK" if success else "FAILED"
        print(f"  - {model}: {status}")

    if 'evaluation' in results['phases'] and results['phases']['evaluation']:
        best = results['phases']['evaluation'].get('best_model', {})
        print(f"\nBest model: {best.get('name', 'N/A')}")
        print(f"Best MSE: {best.get('mse', 'N/A')}")

    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
