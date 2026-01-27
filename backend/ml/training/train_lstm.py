#!/usr/bin/env python3
"""
LSTM Model Training Script
Trains LSTM model for stock price prediction using PyTorch.

Supports:
- GPU acceleration via CUDA
- Automatic mixed precision training for faster GPU training
- GPU memory monitoring and logging
- Automatic fallback to CPU if GPU not available
- Automatic upload to HuggingFace Hub for centralized model storage

GPU Requirements:
- NVIDIA GPU with CUDA support (compute capability 3.5+)
- CUDA drivers installed
- PyTorch with CUDA support

Expected speedup: 3-4x faster training with GPU
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# GPU utilities for centralized device detection
try:
    from backend.ml.gpu_utils import (
        get_cached_gpu_config,
        log_gpu_memory_usage,
        clear_gpu_memory,
        GPUConfig
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    get_cached_gpu_config = None
    log_gpu_memory_usage = None
    clear_gpu_memory = None
    GPUConfig = None

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


class StockSequenceDataset(Dataset):
    """PyTorch dataset for sequential stock data."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 60
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        # Valid length accounting for sequence and target availability
        self.valid_length = len(self.features) - self.sequence_length - 1

    def __len__(self):
        return max(0, self.valid_length)

    def __getitem__(self, idx):
        # Ensure idx doesn't go out of bounds
        target_idx = idx + self.sequence_length
        if target_idx >= len(self.targets):
            target_idx = len(self.targets) - 1
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[target_idx]
        )


class LSTMModel(nn.Module):
    """LSTM model for time series prediction with attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Take the last time step
        last_hidden = attn_out[:, -1, :]

        # Fully connected layers
        output = self.fc_layers(last_hidden)

        return output


class LSTMTrainer:
    """Trainer class for LSTM model with GPU support."""

    def __init__(
        self,
        data_dir: str = 'data/ml_training/processed',
        model_dir: str = 'ml_models',
        sequence_length: int = 60,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5,
        use_gpu: bool = True,
        use_mixed_precision: bool = True  # Automatic mixed precision for faster GPU training
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.use_gpu = use_gpu
        self.use_mixed_precision = use_mixed_precision

        # Setup GPU configuration
        self.gpu_config = None
        self.device = self._setup_device()

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'future_return_5d'

    def _setup_device(self) -> torch.device:
        """Setup device with centralized GPU configuration."""
        if not self.use_gpu:
            logger.info("GPU disabled by configuration, using CPU")
            return torch.device('cpu')

        # Use centralized GPU config if available
        if GPU_UTILS_AVAILABLE and get_cached_gpu_config:
            self.gpu_config = get_cached_gpu_config(
                use_mixed_precision=self.use_mixed_precision
            )
            if self.gpu_config.cuda_available:
                logger.info(f"Using GPU: {self.gpu_config.cuda_device_name}")
                logger.info(f"  Memory: {self.gpu_config.cuda_memory_total_gb:.1f}GB total, "
                           f"{self.gpu_config.cuda_memory_free_gb:.1f}GB free")
                logger.info(f"  Compute capability: {self.gpu_config.compute_capability}")
                logger.info(f"  Mixed precision: {'enabled' if self.use_mixed_precision else 'disabled'}")
                return torch.device('cuda')
            else:
                logger.info("CUDA not available, using CPU")
                return torch.device('cpu')
        else:
            # Fallback to basic PyTorch detection
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {device_name}")
                return torch.device('cuda')
            else:
                logger.info("CUDA not available, using CPU")
                return torch.device('cpu')

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

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        if scale:
            X = self.scaler.fit_transform(X)

        # Create overlapping sequences
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def train(self) -> Dict[str, Any]:
        """Train the LSTM model with GPU acceleration."""
        logger.info("="*60)
        logger.info("Starting LSTM Training")
        logger.info("="*60)

        # Log GPU info
        if self.device.type == 'cuda':
            logger.info(f"Training on GPU: {self.device}")
            if log_gpu_memory_usage:
                log_gpu_memory_usage("Pre-training ")
        else:
            logger.info("Training on CPU")

        # Load data
        train_df, val_df, test_df = self.load_data()

        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, scale=True)

        logger.info(f"Training sequences: {X_train_seq.shape}")

        # Create data loaders
        train_dataset = StockSequenceDataset(
            X_train_seq.reshape(-1, X_train_seq.shape[-1]),
            y_train_seq,
            self.sequence_length
        )

        # For validation, use the scaler fitted on training data
        if val_df is not None:
            X_val, y_val = self.prepare_features(val_df)
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = [], []
            for i in range(len(X_val_scaled) - self.sequence_length):
                X_val_seq.append(X_val_scaled[i:i + self.sequence_length])
                y_val_seq.append(y_val[i + self.sequence_length])
            X_val_seq, y_val_seq = np.array(X_val_seq), np.array(y_val_seq)

            val_dataset = StockSequenceDataset(
                X_val_seq.reshape(-1, X_val_seq.shape[-1]),
                y_val_seq,
                self.sequence_length
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_dim = len(self.feature_columns)
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        # Setup automatic mixed precision for faster GPU training
        use_amp = (
            self.device.type == 'cuda' and
            self.use_mixed_precision and
            torch.cuda.is_available()
        )
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            logger.info("Using automatic mixed precision (AMP) for faster training")

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        total_batches = len(train_loader)
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            logger.info(f"Epoch {epoch+1}/{self.epochs} starting...")
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    logger.info(f"  Batch {batch_idx}/{total_batches}")
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()

                # Use mixed precision if enabled
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    # Gradient clipping with scaler
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches
            train_losses.append(avg_train_loss)

            # Log GPU memory periodically
            if self.device.type == 'cuda' and log_gpu_memory_usage and epoch % 10 == 0:
                log_gpu_memory_usage(f"Epoch {epoch+1} ")

            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device).unsqueeze(1)

                        # Use autocast for consistent inference
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(batch_X)
                                loss = criterion(outputs, batch_y)
                        else:
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        n_val_batches += 1

                avg_val_loss = val_loss / n_val_batches
                val_losses.append(avg_val_loss)

                scheduler.step(avg_val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self._save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}")
                self._save_model()

        # Log final GPU memory
        if self.device.type == 'cuda':
            if log_gpu_memory_usage:
                log_gpu_memory_usage("Post-training ")
            # Clear GPU cache
            if clear_gpu_memory:
                clear_gpu_memory()

        # Build GPU info for results
        gpu_info = {
            'device': str(self.device),
            'gpu_enabled': self.device.type == 'cuda',
            'mixed_precision_enabled': use_amp,
        }
        if self.gpu_config and self.device.type == 'cuda':
            gpu_info['gpu_name'] = self.gpu_config.cuda_device_name
            gpu_info['gpu_memory_gb'] = self.gpu_config.cuda_memory_total_gb
            gpu_info['compute_capability'] = self.gpu_config.compute_capability

        # Calculate final metrics
        results = {
            'model_type': 'lstm',
            'training_completed': datetime.now().isoformat(),
            'epochs_trained': epoch + 1,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'sequence_length': self.sequence_length,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'input_dim': input_dim
            },
            'feature_columns': self.feature_columns,
            'gpu_info': gpu_info
        }

        # Save results
        results_path = self.model_dir / 'lstm_training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("="*60)
        logger.info(f"Training complete! Best val loss: {best_val_loss:.6f}")
        logger.info(f"Model saved to {self.model_dir}")
        logger.info("="*60)

        return results

    def _save_model(self):
        """Save model, scaler, and config."""
        # Save model weights
        model_path = self.model_dir / 'lstm_weights.pth'
        torch.save(self.model.state_dict(), model_path)

        # Save scaler
        scaler_path = self.model_dir / 'lstm_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)

        # Save feature columns
        config_path = self.model_dir / 'lstm_config.json'
        config = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'target_column': self.target_column
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {model_path}")

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

        logger.info("Uploading LSTM model to HuggingFace Hub...")

        # Create temp directory with model files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy model files with clean names
            files_to_copy = [
                ("lstm_weights.pth", "weights.pth"),
                ("lstm_scaler.pkl", "scaler.pkl"),
                ("lstm_config.json", "config.json"),
                ("lstm_training_results.json", "training_results.json"),
            ]

            for src_name, dst_name in files_to_copy:
                src = self.model_dir / src_name
                if src.exists():
                    shutil.copy(src, temp_path / dst_name)
                    logger.info(f"  Prepared {src_name} -> {dst_name}")

            # Determine version
            if not version:
                versions = hf_client.list_versions("lstm")
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
                model_name="lstm",
                version=version,
                local_dir=temp_path,
                commit_message=f"LSTM model v{version} - trained {datetime.now(timezone.utc).isoformat()}",
                metadata={
                    "model_type": "pytorch",
                    "architecture": "LSTM with attention",
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "sequence_length": self.sequence_length,
                }
            )

            if result:
                logger.info(f"LSTM model uploaded to HF Hub as v{version}")
                return True
            else:
                logger.error("Failed to upload LSTM model to HF Hub")
                return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train LSTM model with GPU support')
    parser.add_argument('--data-dir', type=str, default='data/ml_training/processed')
    parser.add_argument('--model-dir', type=str, default='ml_models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--upload-to-hf', action='store_true',
                        help='Upload model to HuggingFace Hub after training')
    parser.add_argument('--hf-version', type=str, default=None,
                        help='Specific version for HF upload (e.g., "1.2.0")')
    parser.add_argument('--hf-version-type', type=str, default='patch',
                        choices=['major', 'minor', 'patch'],
                        help='Version bump type if auto-incrementing')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration (use CPU only)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Alias for --no-gpu')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable automatic mixed precision (AMP) for GPU training')

    args = parser.parse_args()

    # Determine GPU usage
    use_gpu = not (args.no_gpu or args.force_cpu)
    use_mixed_precision = not args.no_mixed_precision

    trainer = LSTMTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        use_gpu=use_gpu,
        use_mixed_precision=use_mixed_precision
    )

    results = trainer.train()
    print(f"\nTraining complete! Final val loss: {results.get('final_val_loss', 'N/A')}")

    # Upload to HuggingFace Hub if requested
    if args.upload_to_hf:
        trainer.upload_to_hf_hub(
            version=args.hf_version,
            version_type=args.hf_version_type
        )


if __name__ == '__main__':
    main()
