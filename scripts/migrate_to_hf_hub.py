#!/usr/bin/env python3
"""
Migration Script: Upload Existing Models and Datasets to HuggingFace Hub

This script migrates existing local models and datasets to HuggingFace Hub
for centralized cloud storage and version management.

Usage:
    python scripts/migrate_to_hf_hub.py --all           # Migrate everything
    python scripts/migrate_to_hf_hub.py --models        # Migrate models only
    python scripts/migrate_to_hf_hub.py --datasets      # Migrate datasets only
    python scripts/migrate_to_hf_hub.py --model lstm    # Migrate specific model
"""

import os
import sys
import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_lstm_model(hf_client, models_dir: Path, version: str = "1.0.0") -> bool:
    """Migrate LSTM model to HuggingFace Hub"""
    logger.info("Migrating LSTM model...")

    # Required files
    required_files = ["lstm_weights.pth", "lstm_scaler.pkl", "lstm_config.json"]
    optional_files = ["lstm_training_results.json"]

    # Check if files exist
    missing = [f for f in required_files if not (models_dir / f).exists()]
    if missing:
        logger.warning(f"Missing required LSTM files: {missing}")
        return False

    # Create temp directory with model files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy files
        for f in required_files + optional_files:
            src = models_dir / f
            if src.exists():
                # Rename to cleaner names
                dst_name = f.replace("lstm_", "")
                shutil.copy(src, temp_path / dst_name)
                logger.info(f"  Copied {f} -> {dst_name}")

        # Upload to HF Hub
        result = hf_client.upload_model(
            model_name="lstm",
            version=version,
            local_dir=temp_path,
            commit_message=f"Migrate LSTM model v{version} from local storage",
            metadata={
                "model_type": "pytorch",
                "architecture": "LSTM with attention",
                "migrated_at": datetime.utcnow().isoformat(),
                "source": "local migration"
            }
        )

        if result:
            logger.info(f"LSTM model uploaded as v{version}")
            return True
        else:
            logger.error("Failed to upload LSTM model")
            return False


def migrate_xgboost_model(hf_client, models_dir: Path, version: str = "1.0.0") -> bool:
    """Migrate XGBoost model to HuggingFace Hub"""
    logger.info("Migrating XGBoost model...")

    # Required files
    required_files = ["xgboost_model.pkl", "xgboost_scaler.pkl", "xgboost_config.json"]
    optional_files = ["xgboost_training_results.json", "xgboost_feature_importance.json"]

    # Check if files exist
    missing = [f for f in required_files if not (models_dir / f).exists()]
    if missing:
        logger.warning(f"Missing required XGBoost files: {missing}")
        return False

    # Create temp directory with model files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy files
        for f in required_files + optional_files:
            src = models_dir / f
            if src.exists():
                # Rename to cleaner names
                dst_name = f.replace("xgboost_", "")
                shutil.copy(src, temp_path / dst_name)
                logger.info(f"  Copied {f} -> {dst_name}")

        # Upload to HF Hub
        result = hf_client.upload_model(
            model_name="xgboost",
            version=version,
            local_dir=temp_path,
            commit_message=f"Migrate XGBoost model v{version} from local storage",
            metadata={
                "model_type": "xgboost",
                "task": "price_direction_prediction",
                "migrated_at": datetime.utcnow().isoformat(),
                "source": "local migration"
            }
        )

        if result:
            logger.info(f"XGBoost model uploaded as v{version}")
            return True
        else:
            logger.error("Failed to upload XGBoost model")
            return False


def migrate_prophet_models(hf_client, models_dir: Path, version: str = "1.0.0") -> bool:
    """Migrate Prophet models to HuggingFace Hub"""
    logger.info("Migrating Prophet models...")

    prophet_dir = models_dir / "prophet"
    if not prophet_dir.exists():
        logger.warning("Prophet directory not found")
        return False

    # Get list of stock model files
    model_files = list(prophet_dir.glob("*_model.pkl"))
    metadata_files = ["trained_stocks.json", "prophet_training_results.json"]

    if not model_files:
        logger.warning("No Prophet model files found")
        return False

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        models_subdir = temp_path / "models"
        models_subdir.mkdir()

        # Copy stock model files
        for model_file in model_files:
            shutil.copy(model_file, models_subdir / model_file.name)
            logger.info(f"  Copied {model_file.name}")

        # Copy metadata files
        for f in metadata_files:
            src = prophet_dir / f
            if src.exists():
                shutil.copy(src, temp_path / f)
                logger.info(f"  Copied {f}")

        # Upload to HF Hub
        result = hf_client.upload_model(
            model_name="prophet",
            version=version,
            local_dir=temp_path,
            commit_message=f"Migrate Prophet models v{version} ({len(model_files)} stocks)",
            metadata={
                "model_type": "prophet",
                "stock_count": len(model_files),
                "migrated_at": datetime.utcnow().isoformat(),
                "source": "local migration"
            }
        )

        if result:
            logger.info(f"Prophet models uploaded as v{version} ({len(model_files)} stocks)")
            return True
        else:
            logger.error("Failed to upload Prophet models")
            return False


def migrate_dataset(dataset_manager, data_dir: Path, version_type: str = "patch") -> bool:
    """Migrate dataset to HuggingFace Hub"""
    logger.info("Migrating training dataset...")

    processed_dir = data_dir / "ml_training" / "processed"
    if not processed_dir.exists():
        logger.warning(f"Processed data directory not found: {processed_dir}")
        return False

    result = dataset_manager.upload_from_local(
        local_processed_dir=processed_dir,
        version_type=version_type
    )

    if result:
        logger.info(f"Dataset uploaded as v{result.version}")
        logger.info(f"  Total samples: {result.total_samples}")
        logger.info(f"  Stocks: {result.stock_count}")
        logger.info(f"  Features: {result.feature_count}")
        return True
    else:
        logger.error("Failed to upload dataset")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate models and datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Migrate all models and datasets"
    )
    parser.add_argument(
        "--models", action="store_true",
        help="Migrate all models only"
    )
    parser.add_argument(
        "--datasets", action="store_true",
        help="Migrate datasets only"
    )
    parser.add_argument(
        "--model", type=str, choices=["lstm", "xgboost", "prophet"],
        help="Migrate a specific model"
    )
    parser.add_argument(
        "--version", type=str, default="1.0.0",
        help="Version for migrated models (default: 1.0.0)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be migrated without uploading"
    )

    args = parser.parse_args()

    # Default to --all if no specific option given
    if not any([args.all, args.models, args.datasets, args.model]):
        args.all = True

    # Paths
    models_dir = PROJECT_ROOT / "ml_models"
    data_dir = PROJECT_ROOT / "data"

    if args.dry_run:
        logger.info("DRY RUN - No files will be uploaded")
        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Data directory: {data_dir}")

        # List what would be migrated
        if args.all or args.models or args.model == "lstm":
            lstm_files = [f for f in ["lstm_weights.pth", "lstm_scaler.pkl", "lstm_config.json"]
                         if (models_dir / f).exists()]
            logger.info(f"LSTM files found: {lstm_files}")

        if args.all or args.models or args.model == "xgboost":
            xgb_files = [f for f in ["xgboost_model.pkl", "xgboost_scaler.pkl", "xgboost_config.json"]
                        if (models_dir / f).exists()]
            logger.info(f"XGBoost files found: {xgb_files}")

        if args.all or args.models or args.model == "prophet":
            prophet_files = list((models_dir / "prophet").glob("*_model.pkl"))
            logger.info(f"Prophet model files found: {len(prophet_files)}")

        if args.all or args.datasets:
            processed_dir = data_dir / "ml_training" / "processed"
            if processed_dir.exists():
                data_files = list(processed_dir.glob("*.parquet"))
                logger.info(f"Dataset files found: {[f.name for f in data_files]}")
            else:
                logger.info("No processed dataset found")

        return

    # Initialize HF clients
    logger.info("Initializing HuggingFace Hub clients...")

    try:
        from backend.ml.hf_hub_client import HuggingFaceHubClient
        from backend.ml.dataset_hub import HuggingFaceDatasetManager

        hf_client = HuggingFaceHubClient()
        dataset_manager = HuggingFaceDatasetManager()
    except Exception as e:
        logger.error(f"Failed to initialize HF clients: {e}")
        logger.error("Make sure HF_TOKEN is set and huggingface_hub is installed")
        sys.exit(1)

    # Track results
    results = {}

    # Migrate models
    if args.all or args.models or args.model:
        if args.all or args.models or args.model == "lstm":
            results["lstm"] = migrate_lstm_model(hf_client, models_dir, args.version)

        if args.all or args.models or args.model == "xgboost":
            results["xgboost"] = migrate_xgboost_model(hf_client, models_dir, args.version)

        if args.all or args.models or args.model == "prophet":
            results["prophet"] = migrate_prophet_models(hf_client, models_dir, args.version)

    # Migrate dataset
    if args.all or args.datasets:
        # Use "minor" for first migration since it's a significant change
        results["dataset"] = migrate_dataset(dataset_manager, data_dir, "minor")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 50)

    for item, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {item}: {status}")

    # Overall status
    all_success = all(results.values())
    if all_success:
        logger.info("\nAll migrations completed successfully!")
        logger.info(f"\nVerify at:")
        logger.info(f"  Models: https://huggingface.co/{os.getenv('HF_MODEL_REPO')}")
        logger.info(f"  Datasets: https://huggingface.co/datasets/{os.getenv('HF_DATASET_REPO')}")
    else:
        logger.warning("\nSome migrations failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
