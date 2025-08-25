"""
Concrete implementations of ML pipelines for different model types
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ModelPipeline, PipelineConfig, PipelineStep, ModelType, ModelArtifact

logger = logging.getLogger(__name__)


class DataLoadingStep(PipelineStep):
    """Load and validate data"""
    
    def __init__(self):
        super().__init__("data_loading")
    
    async def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load data from source"""
        config = context["config"]
        
        # Load data based on source type
        if config.data_source.endswith('.csv'):
            df = pd.read_csv(config.data_source)
        elif config.data_source.endswith('.parquet'):
            df = pd.read_parquet(config.data_source)
        elif config.data_source.startswith('postgresql://'):
            from sqlalchemy import create_engine
            engine = create_engine(config.data_source)
            df = pd.read_sql(config.data_source, engine)
        else:
            # Assume it's a DataFrame passed directly
            df = config.data_source
        
        self.logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Validate data
        if config.target_column not in df.columns:
            raise ValueError(f"Target column {config.target_column} not found")
        
        # Store in context
        context["raw_data"] = df
        context["data_loading_result"] = {
            "samples": len(df),
            "features": len(df.columns),
            "missing_values": df.isnull().sum().sum()
        }
        
        return df, context


class DataPreprocessingStep(PipelineStep):
    """Preprocess and clean data"""
    
    def __init__(self):
        super().__init__("data_preprocessing")
        self.scaler = None
    
    async def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess data"""
        config = context["config"]
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        data[categorical_columns] = data[categorical_columns].fillna('missing')
        
        # Encode categorical variables
        for col in categorical_columns:
            if col != config.target_column:
                data[col] = pd.Categorical(data[col]).codes
        
        # Remove outliers (optional)
        if config.hyperparameters.get("remove_outliers", False):
            for col in numeric_columns:
                if col != config.target_column:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 3 * iqr
                    upper = q3 + 3 * iqr
                    data = data[(data[col] >= lower) & (data[col] <= upper)]
        
        # Scale features
        if config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif config.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            feature_cols = [col for col in data.columns if col != config.target_column]
            data[feature_cols] = self.scaler.fit_transform(data[feature_cols])
            context["artifacts"]["scaler"] = self.scaler
        
        context["preprocessed_data"] = data
        context["data_preprocessing_result"] = {
            "samples_after_cleaning": len(data),
            "scaling_method": config.scaling_method
        }
        
        return data, context


class FeatureEngineeringStep(PipelineStep):
    """Engineer and select features"""
    
    def __init__(self):
        super().__init__("feature_engineering")
        self.feature_selector = None
    
    async def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Engineer features"""
        config = context["config"]
        
        # Separate features and target
        X = data.drop(columns=[config.target_column])
        y = data[config.target_column]
        
        # Create polynomial features (optional)
        if config.hyperparameters.get("polynomial_features", False):
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            X = pd.DataFrame(X_poly, columns=[f"poly_{i}" for i in range(X_poly.shape[1])])
        
        # Feature selection
        if config.feature_selection_method == "mutual_info":
            self.feature_selector = SelectKBest(
                mutual_info_regression,
                k=min(config.max_features, X.shape[1])
            )
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            X = pd.DataFrame(X_selected, columns=selected_features)
            
            context["artifacts"]["feature_selector"] = self.feature_selector
            context["artifacts"]["selected_features"] = selected_features
        
        # Combine back with target
        data = pd.concat([X, y], axis=1)
        
        context["engineered_data"] = data
        context["feature_engineering_result"] = {
            "features_created": X.shape[1],
            "features_selected": len(selected_features) if config.feature_selection_method else X.shape[1]
        }
        
        return data, context


class DataSplittingStep(PipelineStep):
    """Split data into train/validation/test sets"""
    
    def __init__(self):
        super().__init__("data_splitting")
    
    async def execute(self, data: pd.DataFrame, context: Dict[str, Any]) -> Tuple[Dict, Dict[str, Any]]:
        """Split data"""
        config = context["config"]
        
        # Separate features and target
        X = data.drop(columns=[config.target_column])
        y = data[config.target_column]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=1 - config.train_test_split,
            random_state=42
        )
        
        # Second split: train vs val
        val_size = config.validation_split / config.train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42
        )
        
        splits = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }
        
        context["data_splits"] = splits
        context["data_splitting_result"] = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        }
        
        return splits, context


class ModelTrainingStep(PipelineStep):
    """Train ML models"""
    
    def __init__(self, model_type: str = "xgboost"):
        super().__init__(f"model_training_{model_type}")
        self.model_type = model_type
        self.model = None
    
    async def execute(self, data: Dict, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train model"""
        config = context["config"]
        
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        
        # Train based on model type
        if self.model_type == "xgboost":
            self.model = self._train_xgboost(X_train, y_train, X_val, y_val, config)
        elif self.model_type == "lightgbm":
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val, config)
        elif self.model_type == "random_forest":
            self.model = self._train_random_forest(X_train, y_train, config)
        elif self.model_type == "neural_network":
            self.model = await self._train_neural_network(X_train, y_train, X_val, y_val, config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        context["artifacts"][f"model_{self.model_type}"] = self.model
        context[f"model_training_{self.model_type}_result"] = {
            "model_type": self.model_type,
            "training_samples": len(X_train)
        }
        
        return self.model, context
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, config):
        """Train XGBoost model"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': config.hyperparameters.get('max_depth', 6),
            'learning_rate': config.learning_rate,
            'n_estimators': config.hyperparameters.get('n_estimators', 100),
            'subsample': config.hyperparameters.get('subsample', 0.8),
            'colsample_bytree': config.hyperparameters.get('colsample_bytree', 0.8),
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=config.early_stopping_patience,
            verbose=False
        )
        
        return model
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, config):
        """Train LightGBM model"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': config.hyperparameters.get('max_depth', -1),
            'learning_rate': config.learning_rate,
            'n_estimators': config.hyperparameters.get('n_estimators', 100),
            'num_leaves': config.hyperparameters.get('num_leaves', 31),
            'feature_fraction': config.hyperparameters.get('feature_fraction', 0.8),
            'bagging_fraction': config.hyperparameters.get('bagging_fraction', 0.8),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(config.early_stopping_patience)],
            eval_metric='rmse'
        )
        
        return model
    
    def _train_random_forest(self, X_train, y_train, config):
        """Train Random Forest model"""
        params = {
            'n_estimators': config.hyperparameters.get('n_estimators', 100),
            'max_depth': config.hyperparameters.get('max_depth', None),
            'min_samples_split': config.hyperparameters.get('min_samples_split', 2),
            'min_samples_leaf': config.hyperparameters.get('min_samples_leaf', 1),
            'random_state': 42,
            'n_jobs': config.num_workers
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    async def _train_neural_network(self, X_train, y_train, X_val, y_val, config):
        """Train neural network model"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
        y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
        X_val_tensor = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val)
        y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Define model architecture
        input_size = X_train.shape[1]
        hidden_sizes = config.hyperparameters.get('hidden_sizes', [128, 64, 32])
        
        class NeuralNet(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, 1))
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        model = NeuralNet()
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    break
        
        return model


class ModelEvaluationStep(PipelineStep):
    """Evaluate trained models"""
    
    def __init__(self):
        super().__init__("model_evaluation")
    
    async def execute(self, data: Dict, context: Dict[str, Any]) -> Tuple[Dict, Dict[str, Any]]:
        """Evaluate models"""
        config = context["config"]
        
        X_test = data["X_test"]
        y_test = data["y_test"]
        
        evaluation_results = {}
        
        # Evaluate each model in artifacts
        for key, model in context["artifacts"].items():
            if key.startswith("model_"):
                model_type = key.replace("model_", "")
                
                # Make predictions
                if isinstance(model, nn.Module):
                    model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
                        y_pred = model(X_test_tensor).squeeze().numpy()
                else:
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                evaluation_results[model_type] = {
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2)
                }
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    feature_names = X_test.columns.tolist()
                    importance = model.feature_importances_
                    feature_importance = dict(zip(feature_names, importance))
                    evaluation_results[model_type]["feature_importance"] = feature_importance
        
        context["evaluation_results"] = evaluation_results
        context["model_evaluation_result"] = evaluation_results
        
        return evaluation_results, context


class ModelSavingStep(PipelineStep):
    """Save trained models and artifacts"""
    
    def __init__(self):
        super().__init__("model_saving")
    
    async def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[ModelArtifact, Dict[str, Any]]:
        """Save models"""
        config = context["config"]
        
        # Create output directory
        output_path = Path(config.output_path) / config.name / config.version
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find best model based on evaluation
        best_model_type = None
        best_score = -float('inf')
        
        if "evaluation_results" in context:
            for model_type, metrics in context["evaluation_results"].items():
                score = metrics.get(config.primary_metric, metrics.get("r2", 0))
                if score > best_score:
                    best_score = score
                    best_model_type = model_type
        
        # Save best model
        if best_model_type:
            model = context["artifacts"].get(f"model_{best_model_type}")
            if model:
                model_path = output_path / f"model_{best_model_type}.pkl"
                
                if isinstance(model, nn.Module):
                    torch.save(model.state_dict(), model_path)
                else:
                    joblib.dump(model, model_path)
                
                # Save preprocessor
                preprocessor_path = None
                if "scaler" in context["artifacts"]:
                    preprocessor_path = output_path / "preprocessor.pkl"
                    joblib.dump(context["artifacts"]["scaler"], preprocessor_path)
                
                # Save feature columns
                feature_columns_path = None
                if "selected_features" in context["artifacts"]:
                    feature_columns_path = output_path / "feature_columns.pkl"
                    joblib.dump(context["artifacts"]["selected_features"], feature_columns_path)
                
                # Create model artifact
                model_artifact = ModelArtifact(
                    model_id=f"{config.name}_{config.version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    name=config.name,
                    version=config.version,
                    model_type=config.model_type,
                    model_path=model_path,
                    preprocessor_path=preprocessor_path,
                    feature_columns_path=feature_columns_path,
                    created_at=datetime.utcnow(),
                    training_samples=context.get("data_splits", {}).get("X_train", []).shape[0] if "data_splits" in context else 0,
                    feature_count=len(context["artifacts"].get("selected_features", [])),
                    metrics=context["evaluation_results"].get(best_model_type, {}),
                    config_hash=config.get_hash(),
                    hyperparameters=config.hyperparameters
                )
                
                context["model_artifact"] = model_artifact
                context["model_saving_result"] = {
                    "model_saved": str(model_path),
                    "best_model_type": best_model_type,
                    "best_score": best_score
                }
                
                return model_artifact, context
        
        raise ValueError("No model to save")


class StockPredictionPipeline(ModelPipeline):
    """Complete pipeline for stock price prediction"""
    
    def _setup_pipeline(self):
        """Setup pipeline steps"""
        # Add all steps in order
        self.add_step(DataLoadingStep())
        self.add_step(DataPreprocessingStep())
        self.add_step(FeatureEngineeringStep())
        self.add_step(DataSplittingStep())
        
        # Train multiple models
        self.add_step(ModelTrainingStep("xgboost"))
        self.add_step(ModelTrainingStep("lightgbm"))
        self.add_step(ModelTrainingStep("random_forest"))
        
        # Evaluate and save
        self.add_step(ModelEvaluationStep())
        self.add_step(ModelSavingStep())


def create_pipeline(config: PipelineConfig) -> ModelPipeline:
    """Factory function to create appropriate pipeline"""
    
    if config.model_type == ModelType.TIME_SERIES:
        return StockPredictionPipeline(config)
    elif config.model_type == ModelType.REGRESSION:
        return StockPredictionPipeline(config)  # Reuse for now
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


# Integration with existing training pipeline
async def run_ml_pipeline(data_source: str, target_column: str = "future_return") -> ModelArtifact:
    """Run ML pipeline with data"""
    
    config = PipelineConfig(
        name="stock_prediction",
        version="1.0.0",
        model_type=ModelType.TIME_SERIES,
        data_source=data_source,
        feature_columns=[],  # Will be determined automatically
        target_column=target_column,
        train_test_split=0.8,
        validation_split=0.1,
        batch_size=32,
        epochs=100,
        early_stopping_patience=10,
        learning_rate=0.001,
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 6,
            "subsample": 0.8
        }
    )
    
    pipeline = create_pipeline(config)
    result = await pipeline.execute()
    
    if result.status == PipelineStatus.COMPLETED and result.model_artifact:
        return result.model_artifact
    else:
        raise Exception(f"Pipeline failed: {result.error_message}")