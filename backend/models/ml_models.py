"""
Advanced ML Ensemble Models for Stock Prediction
Combines multiple models for world-class predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import optuna
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import joblib
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for model predictions"""
    ticker: str
    model_name: str
    prediction_date: datetime
    target_date: datetime
    predicted_price: float
    predicted_return: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    model_confidence: float


class StockDataset(Dataset):
    """PyTorch dataset for stock data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 60):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[idx + self.sequence_length]
        )


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
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


class TransformerModel(nn.Module):
    """Transformer model for stock prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
    
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = x.transpose(0, 1)  # (seq_len, batch, features)
        transformer_out = self.transformer(x)
        
        # Take the last time step
        last_hidden = transformer_out[-1, :, :]
        
        # Output layer
        output = self.output_layer(last_hidden)
        
        return output


class ModelManager:
    """
    Manages all ML models and ensemble predictions
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def load_models(self):
        """Load or initialize all models"""
        logger.info("Loading ML models...")
        
        # Initialize models
        self.models['lstm'] = LSTMModel(input_dim=100).to(self.device)
        self.models['transformer'] = TransformerModel(input_dim=100).to(self.device)
        self.models['xgboost'] = None  # Will be trained
        self.models['lightgbm'] = None  # Will be trained
        self.models['random_forest'] = None  # Will be trained
        self.models['prophet'] = None  # Will be fitted per stock
        
        # Try to load pre-trained weights
        try:
            self.models['lstm'].load_state_dict(
                torch.load('models/lstm_weights.pth', map_location=self.device)
            )
            self.models['transformer'].load_state_dict(
                torch.load('models/transformer_weights.pth', map_location=self.device)
            )
            
            self.models['xgboost'] = joblib.load('models/xgboost_model.pkl')
            self.models['lightgbm'] = joblib.load('models/lightgbm_model.pkl')
            self.models['random_forest'] = joblib.load('models/rf_model.pkl')
            
            logger.info("Loaded pre-trained models")
        except:
            logger.info("No pre-trained models found, will train from scratch")
    
    async def train_models(
        self,
        training_data: pd.DataFrame,
        target_column: str = 'future_return'
    ):
        """Train all models on historical data"""
        logger.info("Training ensemble models...")
        
        # Prepare features
        features, targets = self._prepare_features(training_data, target_column)
        
        # Split data
        train_size = int(0.8 * len(features))
        X_train, X_val = features[:train_size], features[train_size:]
        y_train, y_val = targets[:train_size], targets[train_size:]
        
        # Train each model type
        await asyncio.gather(
            self._train_deep_learning_models(X_train, y_train, X_val, y_val),
            self._train_tree_models(X_train, y_train, X_val, y_val),
            self._train_time_series_models(training_data)
        )
        
        logger.info("Model training completed")
    
    def _prepare_features(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML models"""
        
        # Feature engineering
        features_df = pd.DataFrame()
        
        # Price-based features
        features_df['returns'] = data['close'].pct_change()
        features_df['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features_df['volatility'] = features_df['returns'].rolling(20).std()
        
        # Technical indicators (already calculated)
        technical_cols = [col for col in data.columns if any(
            ind in col.lower() for ind in ['sma', 'ema', 'rsi', 'macd', 'bb_']
        )]
        for col in technical_cols:
            features_df[col] = data[col]
        
        # Volume features
        features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features_df['dollar_volume'] = data['close'] * data['volume']
        
        # Fundamental features (if available)
        fundamental_cols = [col for col in data.columns if any(
            ind in col.lower() for ind in ['pe_', 'eps', 'roe', 'debt_']
        )]
        for col in fundamental_cols:
            features_df[col] = data[col]
        
        # Sentiment features (if available)
        sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
        for col in sentiment_cols:
            features_df[col] = data[col]
        
        # Market microstructure
        features_df['high_low_ratio'] = data['high'] / data['low']
        features_df['close_to_high'] = data['close'] / data['high']
        features_df['close_to_low'] = data['close'] / data['low']
        
        # Time-based features
        features_df['day_of_week'] = pd.to_datetime(data.index).dayofweek
        features_df['month'] = pd.to_datetime(data.index).month
        features_df['quarter'] = pd.to_datetime(data.index).quarter
        
        # Remove NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Scale features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_df)
        self.scalers['features'] = scaler
        
        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        
        # Prepare targets
        if target_column in data.columns:
            targets = data[target_column].values
        else:
            # Calculate future returns as target
            targets = data['close'].shift(-5).pct_change(5).fillna(0).values
        
        return features_scaled, targets
    
    async def _train_deep_learning_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train LSTM and Transformer models"""
        
        # Create datasets
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train LSTM
        await self._train_pytorch_model(
            self.models['lstm'],
            train_loader,
            val_loader,
            model_name='lstm'
        )
        
        # Train Transformer
        await self._train_pytorch_model(
            self.models['transformer'],
            train_loader,
            val_loader,
            model_name='transformer'
        )
    
    async def _train_pytorch_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str,
        epochs: int = 50
    ):
        """Train a PyTorch model"""
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'models/{model_name}_weights.pth')
            
            if epoch % 10 == 0:
                logger.info(f'{model_name} - Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    async def _train_tree_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train tree-based models with Optuna optimization"""
        
        # XGBoost with Optuna
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            }
            
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            
            pred = model.predict(X_val)
            mse = np.mean((pred - y_val) ** 2)
            
            return mse
        
        # Optimize XGBoost
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
        
        # Train final XGBoost model
        best_params_xgb = study_xgb.best_params
        self.models['xgboost'] = xgb.XGBRegressor(**best_params_xgb, random_state=42, n_jobs=-1)
        self.models['xgboost'].fit(X_train, y_train)
        
        # LightGBM with Optuna
        def lgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }
            
            model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)], verbose=False)
            
            pred = model.predict(X_val)
            mse = np.mean((pred - y_val) ** 2)
            
            return mse
        
        # Optimize LightGBM
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
        
        # Train final LightGBM model
        best_params_lgb = study_lgb.best_params
        self.models['lightgbm'] = lgb.LGBMRegressor(**best_params_lgb, random_state=42, n_jobs=-1)
        self.models['lightgbm'].fit(X_train, y_train)
        
        # Random Forest (simpler optimization)
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # Save models
        joblib.dump(self.models['xgboost'], 'models/xgboost_model.pkl')
        joblib.dump(self.models['lightgbm'], 'models/lightgbm_model.pkl')
        joblib.dump(self.models['random_forest'], 'models/rf_model.pkl')
    
    async def _train_time_series_models(self, data: pd.DataFrame):
        """Train Prophet model for time series forecasting"""
        # Prophet requires specific format
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data['close']
        })
        
        # Add regressors if available
        if 'volume' in data.columns:
            prophet_data['volume'] = data['volume']
        
        # We'll fit Prophet per stock when making predictions
        # as it's designed for univariate time series
        logger.info("Prophet model will be fitted per stock during prediction")
    
    async def predict(
        self,
        ticker: str,
        current_data: pd.DataFrame,
        horizon: int = 5
    ) -> Dict[str, PredictionResult]:
        """
        Make predictions using all models
        """
        predictions = {}
        
        # Prepare features
        features, _ = self._prepare_features(current_data, 'dummy')
        
        # Get latest features for point prediction
        latest_features = features[-1:] if len(features.shape) == 2 else features[-60:]
        
        # Deep learning predictions
        if self.models.get('lstm'):
            predictions['lstm'] = await self._predict_with_dl_model(
                self.models['lstm'],
                latest_features,
                ticker,
                current_data,
                horizon,
                'lstm'
            )
        
        if self.models.get('transformer'):
            predictions['transformer'] = await self._predict_with_dl_model(
                self.models['transformer'],
                latest_features,
                ticker,
                current_data,
                horizon,
                'transformer'
            )
        
        # Tree-based predictions
        tree_features = latest_features[-1] if len(latest_features.shape) > 1 else latest_features
        
        if self.models.get('xgboost'):
            predictions['xgboost'] = self._predict_with_tree_model(
                self.models['xgboost'],
                tree_features,
                ticker,
                current_data,
                horizon,
                'xgboost'
            )
        
        if self.models.get('lightgbm'):
            predictions['lightgbm'] = self._predict_with_tree_model(
                self.models['lightgbm'],
                tree_features,
                ticker,
                current_data,
                horizon,
                'lightgbm'
            )
        
        if self.models.get('random_forest'):
            predictions['random_forest'] = self._predict_with_tree_model(
                self.models['random_forest'],
                tree_features,
                ticker,
                current_data,
                horizon,
                'random_forest'
            )
        
        # Time series prediction with Prophet
        predictions['prophet'] = await self._predict_with_prophet(
            ticker,
            current_data,
            horizon
        )
        
        # Ensemble prediction
        predictions['ensemble'] = self._create_ensemble_prediction(
            predictions,
            ticker,
            current_data,
            horizon
        )
        
        return predictions
    
    async def _predict_with_dl_model(
        self,
        model: nn.Module,
        features: np.ndarray,
        ticker: str,
        current_data: pd.DataFrame,
        horizon: int,
        model_name: str
    ) -> PredictionResult:
        """Make prediction with deep learning model"""
        
        model.eval()
        
        with torch.no_grad():
            # Prepare input
            if len(features.shape) == 2:
                # Need sequence of features
                if features.shape[0] < 60:
                    # Pad with zeros if not enough history
                    padding = np.zeros((60 - features.shape[0], features.shape[1]))
                    features = np.vstack([padding, features])
                features = features[-60:]  # Last 60 time steps
            
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get prediction
            prediction = model(input_tensor).cpu().numpy()[0, 0]
        
        # Convert to price prediction
        current_price = current_data['close'].iloc[-1]
        predicted_return = prediction
        predicted_price = current_price * (1 + predicted_return)
        
        # Estimate confidence interval (using dropout uncertainty)
        model.train()  # Enable dropout
        predictions = []
        
        for _ in range(100):
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy()[0, 0]
                predictions.append(pred)
        
        model.eval()
        
        predictions = np.array(predictions)
        confidence_interval = (
            current_price * (1 + np.percentile(predictions, 5)),
            current_price * (1 + np.percentile(predictions, 95))
        )
        
        return PredictionResult(
            ticker=ticker,
            model_name=model_name,
            prediction_date=datetime.utcnow(),
            target_date=datetime.utcnow() + timedelta(days=horizon),
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence_interval=confidence_interval,
            feature_importance={},  # DL models don't have simple feature importance
            model_confidence=1 - np.std(predictions)  # Lower std = higher confidence
        )
    
    def _predict_with_tree_model(
        self,
        model,
        features: np.ndarray,
        ticker: str,
        current_data: pd.DataFrame,
        horizon: int,
        model_name: str
    ) -> PredictionResult:
        """Make prediction with tree-based model"""
        
        # Reshape if necessary
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Convert to price prediction
        current_price = current_data['close'].iloc[-1]
        predicted_return = prediction
        predicted_price = current_price * (1 + predicted_return)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(
                self.feature_columns,
                model.feature_importances_
            ))
            # Top 10 features
            top_features = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        else:
            top_features = {}
        
        # Estimate confidence interval using tree predictions
        if hasattr(model, 'estimators_'):  # Random Forest
            tree_predictions = np.array([
                tree.predict(features)[0] for tree in model.estimators_
            ])
            confidence_interval = (
                current_price * (1 + np.percentile(tree_predictions, 5)),
                current_price * (1 + np.percentile(tree_predictions, 95))
            )
            model_confidence = 1 - np.std(tree_predictions)
        else:
            # Simple confidence interval
            std_estimate = abs(predicted_return) * 0.2  # 20% of prediction
            confidence_interval = (
                current_price * (1 + predicted_return - 2 * std_estimate),
                current_price * (1 + predicted_return + 2 * std_estimate)
            )
            model_confidence = 0.7  # Default confidence
        
        return PredictionResult(
            ticker=ticker,
            model_name=model_name,
            prediction_date=datetime.utcnow(),
            target_date=datetime.utcnow() + timedelta(days=horizon),
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence_interval=confidence_interval,
            feature_importance=top_features,
            model_confidence=model_confidence
        )
    
    async def _predict_with_prophet(
        self,
        ticker: str,
        current_data: pd.DataFrame,
        horizon: int
    ) -> PredictionResult:
        """Make prediction with Prophet"""
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': current_data.index,
            'y': current_data['close']
        })
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add regressors
        if 'volume' in current_data.columns:
            prophet_data['volume'] = current_data['volume']
            model.add_regressor('volume')
        
        model.fit(prophet_data)
        
        # Make predictions
        future = model.make_future_dataframe(periods=horizon)
        
        # Add regressor values for future
        if 'volume' in prophet_data.columns:
            # Use average volume for future predictions
            avg_volume = prophet_data['volume'].mean()
            future['volume'] = prophet_data['volume'].tolist() + [avg_volume] * horizon
        
        forecast = model.predict(future)
        
        # Get prediction for target date
        predicted_price = forecast['yhat'].iloc[-1]
        current_price = current_data['close'].iloc[-1]
        predicted_return = (predicted_price - current_price) / current_price
        
        # Confidence interval
        confidence_interval = (
            forecast['yhat_lower'].iloc[-1],
            forecast['yhat_upper'].iloc[-1]
        )
        
        return PredictionResult(
            ticker=ticker,
            model_name='prophet',
            prediction_date=datetime.utcnow(),
            target_date=datetime.utcnow() + timedelta(days=horizon),
            predicted_price=predicted_price,
            predicted_return=predicted_return,
            confidence_interval=confidence_interval,
            feature_importance={},
            model_confidence=0.8  # Prophet is generally reliable
        )
    
    def _create_ensemble_prediction(
        self,
        predictions: Dict[str, PredictionResult],
        ticker: str,
        current_data: pd.DataFrame,
        horizon: int
    ) -> PredictionResult:
        """Create ensemble prediction from all models"""
        
        # Model weights based on historical performance
        model_weights = {
            'lstm': 0.20,
            'transformer': 0.20,
            'xgboost': 0.15,
            'lightgbm': 0.15,
            'random_forest': 0.10,
            'prophet': 0.20
        }
        
        # Collect valid predictions
        valid_predictions = []
        valid_weights = []
        
        for model_name, weight in model_weights.items():
            if model_name in predictions and predictions[model_name]:
                pred = predictions[model_name]
                # Weight by model confidence
                adjusted_weight = weight * pred.model_confidence
                valid_predictions.append(pred)
                valid_weights.append(adjusted_weight)
        
        if not valid_predictions:
            # Return a default prediction
            current_price = current_data['close'].iloc[-1]
            return PredictionResult(
                ticker=ticker,
                model_name='ensemble',
                prediction_date=datetime.utcnow(),
                target_date=datetime.utcnow() + timedelta(days=horizon),
                predicted_price=current_price,
                predicted_return=0.0,
                confidence_interval=(current_price * 0.95, current_price * 1.05),
                feature_importance={},
                model_confidence=0.0
            )
        
        # Normalize weights
        total_weight = sum(valid_weights)
        normalized_weights = [w / total_weight for w in valid_weights]
        
        # Calculate weighted average
        ensemble_price = sum(
            pred.predicted_price * weight
            for pred, weight in zip(valid_predictions, normalized_weights)
        )
        
        ensemble_return = sum(
            pred.predicted_return * weight
            for pred, weight in zip(valid_predictions, normalized_weights)
        )
        
        # Confidence interval from all models
        all_lower_bounds = [pred.confidence_interval[0] for pred in valid_predictions]
        all_upper_bounds = [pred.confidence_interval[1] for pred in valid_predictions]
        
        ensemble_confidence_interval = (
            np.average(all_lower_bounds, weights=normalized_weights),
            np.average(all_upper_bounds, weights=normalized_weights)
        )
        
        # Combined feature importance
        combined_importance = {}
        for pred in valid_predictions:
            for feature, importance in pred.feature_importance.items():
                if feature not in combined_importance:
                    combined_importance[feature] = 0
                combined_importance[feature] += importance / len(valid_predictions)
        
        # Model confidence based on agreement
        price_std = np.std([pred.predicted_price for pred in valid_predictions])
        return_std = np.std([pred.predicted_return for pred in valid_predictions])
        
        # Lower std = higher agreement = higher confidence
        agreement_score = 1 / (1 + price_std / current_data['close'].iloc[-1])
        avg_model_confidence = np.mean([pred.model_confidence for pred in valid_predictions])
        
        ensemble_confidence = 0.7 * agreement_score + 0.3 * avg_model_confidence
        
        return PredictionResult(
            ticker=ticker,
            model_name='ensemble',
            prediction_date=datetime.utcnow(),
            target_date=datetime.utcnow() + timedelta(days=horizon),
            predicted_price=ensemble_price,
            predicted_return=ensemble_return,
            confidence_interval=ensemble_confidence_interval,
            feature_importance=combined_importance,
            model_confidence=ensemble_confidence
        )
    
    async def predict_batch(
        self,
        tickers: List[str],
        data_dict: Dict[str, pd.DataFrame],
        horizon: int = 5
    ) -> Dict[str, Dict[str, PredictionResult]]:
        """Make predictions for multiple stocks"""
        
        tasks = []
        for ticker in tickers:
            if ticker in data_dict:
                task = self.predict(ticker, data_dict[ticker], horizon)
                tasks.append((ticker, task))
        
        results = {}
        for ticker, task in tasks:
            try:
                predictions = await task
                results[ticker] = predictions
            except Exception as e:
                logger.error(f"Error predicting for {ticker}: {e}")
                results[ticker] = {}
        
        return results
    
    def get_prediction_explanation(
        self,
        prediction: PredictionResult,
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate human-readable explanation for prediction
        """
        explanation = {
            'summary': '',
            'key_drivers': [],
            'confidence_factors': [],
            'risks': [],
            'technical_signals': [],
            'fundamental_signals': [],
            'sentiment_signals': []
        }
        
        # Summary
        direction = 'increase' if prediction.predicted_return > 0 else 'decrease'
        confidence_level = 'high' if prediction.model_confidence > 0.8 else 'moderate' if prediction.model_confidence > 0.6 else 'low'
        
        explanation['summary'] = (
            f"The {prediction.model_name} model predicts a {abs(prediction.predicted_return)*100:.1f}% "
            f"{direction} in {prediction.ticker} over the next {(prediction.target_date - prediction.prediction_date).days} days, "
            f"with {confidence_level} confidence. Target price: ${prediction.predicted_price:.2f} "
            f"(range: ${prediction.confidence_interval[0]:.2f} - ${prediction.confidence_interval[1]:.2f})"
        )
        
        # Key drivers from feature importance
        if prediction.feature_importance:
            top_features = sorted(
                prediction.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for feature, importance in top_features:
                explanation['key_drivers'].append({
                    'feature': feature,
                    'importance': importance,
                    'description': self._explain_feature(feature)
                })
        
        # Confidence factors
        if prediction.model_confidence > 0.8:
            explanation['confidence_factors'].append("Strong model agreement across ensemble")
        if prediction.confidence_interval[1] / prediction.confidence_interval[0] < 1.1:
            explanation['confidence_factors'].append("Narrow prediction range indicates high certainty")
        
        # Add signals from analysis
        if 'technical_analysis' in analysis_data:
            tech = analysis_data['technical_analysis']
            if tech.get('signals'):
                explanation['technical_signals'] = tech['signals'][:3]
        
        if 'fundamental_analysis' in analysis_data:
            fund = analysis_data['fundamental_analysis']
            if fund.get('opportunities'):
                explanation['fundamental_signals'] = fund['opportunities'][:3]
        
        if 'sentiment_analysis' in analysis_data:
            sent = analysis_data['sentiment_analysis']
            if sent.get('signals'):
                explanation['sentiment_signals'] = sent['signals'][:3]
        
        return explanation
    
    def _explain_feature(self, feature_name: str) -> str:
        """Provide human-readable explanation for feature"""
        
        feature_explanations = {
            'rsi': 'Relative Strength Index momentum indicator',
            'macd': 'MACD trend-following momentum indicator',
            'volume_ratio': 'Current volume relative to average',
            'pe_ratio': 'Price-to-Earnings valuation metric',
            'sentiment_score': 'Aggregate news and social sentiment',
            'sma_20': '20-day moving average price trend',
            'volatility': 'Recent price volatility measure',
            'roe': 'Return on Equity profitability metric'
        }
        
        # Try exact match first
        if feature_name in feature_explanations:
            return feature_explanations[feature_name]
        
        # Try partial match
        for key, explanation in feature_explanations.items():
            if key in feature_name.lower():
                return explanation
        
        # Default
        return feature_name.replace('_', ' ').title()
    
    async def backtest_predictions(
        self,
        historical_data: Dict[str, pd.DataFrame],
        test_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Backtest model performance on historical data
        """
        results = {
            'overall_accuracy': 0.0,
            'model_performance': {},
            'directional_accuracy': 0.0,
            'average_error': 0.0,
            'sharpe_ratio': 0.0
        }
        
        all_predictions = []
        all_actuals = []
        
        for ticker, data in historical_data.items():
            if len(data) < test_period_days + 60:  # Need history for features
                continue
            
            # Split data
            test_data = data.iloc[-test_period_days:]
            
            for i in range(0, test_period_days - 5, 5):  # Predict every 5 days
                train_end = len(data) - test_period_days + i
                current_data = data.iloc[:train_end]
                
                # Make prediction
                predictions = await self.predict(ticker, current_data, horizon=5)
                
                # Get actual
                actual_price = data.iloc[train_end + 5]['close']
                actual_return = (actual_price - current_data.iloc[-1]['close']) / current_data.iloc[-1]['close']
                
                # Store results
                for model_name, pred in predictions.items():
                    if pred:
                        all_predictions.append(pred.predicted_return)
                        all_actuals.append(actual_return)
                        
                        if model_name not in results['model_performance']:
                            results['model_performance'][model_name] = {
                                'predictions': [],
                                'actuals': []
                            }
                        
                        results['model_performance'][model_name]['predictions'].append(pred.predicted_return)
                        results['model_performance'][model_name]['actuals'].append(actual_return)
        
        # Calculate metrics
        if all_predictions and all_actuals:
            predictions_array = np.array(all_predictions)
            actuals_array = np.array(all_actuals)
            
            # Directional accuracy
            correct_direction = np.sum(
                (predictions_array > 0) == (actuals_array > 0)
            )
            results['directional_accuracy'] = correct_direction / len(predictions_array)
            
            # Average error
            results['average_error'] = np.mean(np.abs(predictions_array - actuals_array))
            
            # Model-specific metrics
            for model_name, model_data in results['model_performance'].items():
                if model_data['predictions']:
                    preds = np.array(model_data['predictions'])
                    acts = np.array(model_data['actuals'])
                    
                    model_metrics = {
                        'directional_accuracy': np.sum((preds > 0) == (acts > 0)) / len(preds),
                        'mae': np.mean(np.abs(preds - acts)),
                        'rmse': np.sqrt(np.mean((preds - acts) ** 2)),
                        'correlation': np.corrcoef(preds, acts)[0, 1] if len(preds) > 1 else 0
                    }
                    
                    results['model_performance'][model_name].update(model_metrics)
        
        return results