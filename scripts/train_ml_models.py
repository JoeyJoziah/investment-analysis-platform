#!/usr/bin/env python3
"""
Quick ML Models Training Script
Generates trained model artifacts for immediate deployment
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from datetime import datetime
import logging

# Add backend to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from backend.models.ml_models import LSTMModel, TransformerModel
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_stock_data(n_samples: int = 10000, n_features: int = 100) -> tuple:
    """
    Generate synthetic stock data for initial model training
    This allows immediate deployment while real historical data is being processed
    """
    np.random.seed(42)  # For reproducible results
    
    # Create realistic stock-like features
    features = []
    
    # Price-based features (20 features)
    price_base = 100 + np.random.randn(n_samples) * 10
    returns = np.random.randn(n_samples) * 0.02  # 2% daily volatility
    
    features.extend([
        price_base,
        returns,
        np.cumsum(returns),  # Cumulative returns
        pd.Series(returns).rolling(5).mean().fillna(0),   # 5-day MA
        pd.Series(returns).rolling(20).mean().fillna(0),  # 20-day MA
        pd.Series(returns).rolling(20).std().fillna(0),   # Volatility
        pd.Series(price_base).rolling(14).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min())).fillna(0.5),  # RSI-like
    ])
    
    # Technical indicators (30 features)
    for i in range(23):  # Additional technical features
        features.append(np.random.randn(n_samples) * 0.1)
    
    # Fundamental features (20 features)
    for i in range(20):
        features.append(np.random.randn(n_samples) * 0.05)
    
    # Market microstructure features (15 features)
    for i in range(15):
        features.append(np.random.randn(n_samples) * 0.03)
    
    # Time-based features (15 features)
    day_of_week = np.random.randint(0, 5, n_samples)  # Weekdays only
    month = np.random.randint(1, 13, n_samples)
    features.extend([
        day_of_week / 4.0,  # Normalized
        month / 12.0,       # Normalized
    ])
    for i in range(13):  # Additional time features
        features.append(np.random.randn(n_samples) * 0.02)
    
    # Pad to reach n_features if needed
    while len(features) < n_features:
        features.append(np.random.randn(n_samples) * 0.01)
    
    X = np.column_stack(features[:n_features])
    
    # Create realistic target (5-day forward return)
    # Based on momentum + mean reversion + noise
    momentum = pd.Series(returns).rolling(5).mean().fillna(0)
    mean_reversion = -pd.Series(returns).rolling(20).mean().fillna(0) * 0.3
    noise = np.random.randn(n_samples) * 0.01
    
    y = momentum + mean_reversion + noise
    y = np.clip(y, -0.1, 0.1)  # Clip extreme values
    
    return X, y


def train_pytorch_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray, epochs: int = 20) -> None:
    """Train PyTorch model with early stopping"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # For sequence models, reshape input
        if hasattr(model, 'lstm') or hasattr(model, 'transformer'):
            # Reshape for sequence input (batch, seq_len, features)
            seq_len = 60
            batch_size = len(X_train_tensor) // seq_len
            X_reshaped = X_train_tensor[:batch_size * seq_len].view(batch_size, seq_len, -1)
            y_reshaped = y_train_tensor[:batch_size * seq_len:seq_len]  # Take every seq_len-th element
            
            outputs = model(X_reshaped).squeeze()
        else:
            outputs = model(X_train_tensor).squeeze()
            y_reshaped = y_train_tensor
        
        loss = criterion(outputs, y_reshaped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'lstm') or hasattr(model, 'transformer'):
                batch_size_val = len(X_val_tensor) // seq_len
                X_val_reshaped = X_val_tensor[:batch_size_val * seq_len].view(batch_size_val, seq_len, -1)
                y_val_reshaped = y_val_tensor[:batch_size_val * seq_len:seq_len]
                val_outputs = model(X_val_reshaped).squeeze()
            else:
                val_outputs = model(X_val_tensor).squeeze()
                y_val_reshaped = y_val_tensor
            
            val_loss = criterion(val_outputs, y_val_reshaped)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            logger.info(f'Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # Early stopping
        if patience_counter >= 5:
            logger.info(f'Early stopping at epoch {epoch}')
            break


def main():
    """Main training function"""
    logger.info("Starting ML model training with synthetic data...")
    
    # Create models directory
    models_dir = Path("/app/ml_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    logger.info("Generating synthetic training data...")
    X, y = generate_synthetic_stock_data(n_samples=10000, n_features=100)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set: {X_train_scaled.shape}, Validation set: {X_val_scaled.shape}")
    
    # 1. Train LSTM Model
    logger.info("Training LSTM model...")
    lstm_model = LSTMModel(input_dim=100, hidden_dim=128, num_layers=3, dropout=0.2)
    train_pytorch_model(lstm_model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=20)
    
    # Save LSTM
    lstm_path = models_dir / "lstm_model.pt"
    torch.save(lstm_model.state_dict(), lstm_path)
    logger.info(f"Saved LSTM model to {lstm_path}")
    
    # 2. Train Transformer Model
    logger.info("Training Transformer model...")
    transformer_model = TransformerModel(input_dim=100, d_model=128, nhead=8, num_layers=4)
    train_pytorch_model(transformer_model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=20)
    
    # Save Transformer
    transformer_path = models_dir / "transformer_model.pt"
    torch.save(transformer_model.state_dict(), transformer_path)
    logger.info(f"Saved Transformer model to {transformer_path}")
    
    # 3. Train XGBoost Model
    logger.info("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_val_scaled, y_val)], 
                  early_stopping_rounds=20, 
                  verbose=False)
    
    # Save XGBoost
    xgb_path = models_dir / "xgboost_model.pkl"
    joblib.dump(xgb_model, xgb_path)
    logger.info(f"Saved XGBoost model to {xgb_path}")
    
    # 4. Train LightGBM Model
    logger.info("Training LightGBM model...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_val_scaled, y_val)], 
                  callbacks=[lgb.early_stopping(20)], 
                  verbose=False)
    
    # Save LightGBM
    lgb_path = models_dir / "lightgbm_model.pkl"
    joblib.dump(lgb_model, lgb_path)
    logger.info(f"Saved LightGBM model to {lgb_path}")
    
    # 5. Train Random Forest Model
    logger.info("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Save Random Forest
    rf_path = models_dir / "rf_model.joblib"
    joblib.dump(rf_model, rf_path)
    logger.info(f"Saved Random Forest model to {rf_path}")
    
    # 6. Save Scaler and Feature Names
    scaler_path = models_dir / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Generate feature names
    feature_names = [f'feature_{i}' for i in range(100)]
    features_path = models_dir / "feature_names.pkl"
    joblib.dump(feature_names, features_path)
    logger.info(f"Saved feature names to {features_path}")
    
    # 7. Test all models
    logger.info("Testing trained models...")
    
    # Test predictions
    test_sample = X_test_scaled[:1]  # Single sample for testing
    
    # PyTorch models
    lstm_model.eval()
    transformer_model.eval()
    
    with torch.no_grad():
        # Reshape for sequence models
        test_reshaped = torch.FloatTensor(test_sample).unsqueeze(0).repeat(1, 60, 1)
        lstm_pred = lstm_model(test_reshaped).cpu().numpy()[0, 0]
        transformer_pred = transformer_model(test_reshaped).cpu().numpy()[0, 0]
    
    # Tree models
    xgb_pred = xgb_model.predict(test_sample)[0]
    lgb_pred = lgb_model.predict(test_sample)[0]
    rf_pred = rf_model.predict(test_sample)[0]
    
    logger.info("Model predictions on test sample:")
    logger.info(f"  LSTM: {lstm_pred:.4f}")
    logger.info(f"  Transformer: {transformer_pred:.4f}")
    logger.info(f"  XGBoost: {xgb_pred:.4f}")
    logger.info(f"  LightGBM: {lgb_pred:.4f}")
    logger.info(f"  Random Forest: {rf_pred:.4f}")
    logger.info(f"  Actual: {y_test[0]:.4f}")
    
    # 8. Create model registry
    registry = {
        "created_at": datetime.utcnow().isoformat(),
        "model_versions": {
            "lstm": "1.0",
            "transformer": "1.0", 
            "xgboost": "1.0",
            "lightgbm": "1.0",
            "random_forest": "1.0"
        },
        "feature_count": 100,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "data_type": "synthetic",
        "performance": {
            "lstm": {"val_mse": "calculated_during_training"},
            "transformer": {"val_mse": "calculated_during_training"}, 
            "xgboost": {"val_mse": "calculated_during_training"},
            "lightgbm": {"val_mse": "calculated_during_training"},
            "random_forest": {"val_mse": "calculated_during_training"}
        }
    }
    
    import json
    registry_path = models_dir / "model_registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Created model registry at {registry_path}")
    
    logger.info("✅ All models trained and saved successfully!")
    logger.info(f"Models saved to: {models_dir}")
    
    # List all created files
    logger.info("Created files:")
    for file_path in models_dir.glob("*"):
        logger.info(f"  {file_path.name}")


if __name__ == "__main__":
    main()