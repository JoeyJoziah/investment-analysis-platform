#!/usr/bin/env python3
"""
Simplified ML Training Pipeline Script - Phase 4 Testing
Main entry point for training ML models without complex dependencies
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'backend/ml_logs/simple_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class SimpleMLTrainingPipeline:
    """Simplified ML Training Pipeline for Testing"""
    
    def __init__(self):
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment"""
        return {
            'models_path': os.getenv('ML_MODELS_PATH', 'backend/ml_models'),
            'logs_path': os.getenv('ML_LOGS_PATH', 'backend/ml_logs'),
            'training_data_path': os.getenv('ML_TRAINING_DATA_PATH', 'data/training'),
            'predictions_path': os.getenv('ML_PREDICTIONS_PATH', 'data/predictions'),
        }
    
    def initialize(self):
        """Initialize the ML pipeline"""
        logger.info("Initializing Simple ML Training Pipeline...")
        
        # Create directories if they don't exist
        for path in [self.config['models_path'], self.config['logs_path'], 
                     self.config['training_data_path'], self.config['predictions_path']]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Simple ML Pipeline initialized successfully")
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data from database or files"""
        logger.info("Loading training data...")
        
        # Check for existing training data
        training_file = Path(self.config['training_data_path']) / 'training_data.csv'
        
        if training_file.exists():
            logger.info(f"Loading data from {training_file}")
            return pd.read_csv(training_file)
        
        # If no data exists, generate sample data for testing
        logger.warning("No training data found, generating sample data...")
        return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample training data for testing"""
        np.random.seed(42)
        n_samples = 5000
        
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
        
        data = {
            'timestamp': dates,
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], n_samples),
            'open': 100 + np.random.randn(n_samples) * 10,
            'high': 105 + np.random.randn(n_samples) * 10,
            'low': 95 + np.random.randn(n_samples) * 10,
            'close': 100 + np.random.randn(n_samples) * 10,
            'volume': np.random.randint(1000000, 10000000, n_samples),
            'returns': np.random.randn(n_samples) * 0.02,
        }
        
        df = pd.DataFrame(data)
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
        df['rsi_14'] = 50 + np.random.randn(n_samples) * 20
        df['macd'] = np.random.randn(n_samples) * 2
        df['bollinger_upper'] = df['sma_20'] + 2 * df['close'].rolling(20, min_periods=1).std()
        df['bollinger_lower'] = df['sma_20'] - 2 * df['close'].rolling(20, min_periods=1).std()
        
        # Add target variable (future return)
        df['future_return'] = df['returns'].shift(-1)
        df = df.dropna()
        
        # Save sample data
        save_path = Path(self.config['training_data_path']) / 'sample_training_data.csv'
        df.to_csv(save_path, index=False)
        logger.info(f"Sample data saved to {save_path}")
        
        return df
    
    def train_simple_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train a simple ML model using sklearn"""
        logger.info("Training simple model...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Prepare features and target
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'macd']
            target_column = 'future_return'
            
            # Filter data and handle missing values
            df_clean = data[feature_columns + [target_column]].dropna()
            
            X = df_clean[feature_columns]
            y = df_clean[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            model_path = Path(self.config['models_path']) / 'simple_random_forest.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            results = {
                'model_type': 'RandomForestRegressor',
                'model_path': str(model_path),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': {
                    'mse': float(mse),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                },
                'feature_importance': dict(zip(feature_columns, model.feature_importances_.tolist())),
                'status': 'completed'
            }
            
            logger.info(f"Model trained successfully - R2: {r2:.3f}, RMSE: {np.sqrt(mse):.6f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_training_pipeline(self):
        """Run the simplified training pipeline"""
        logger.info("="*60)
        logger.info("Starting Simple ML Training Pipeline")
        logger.info("="*60)
        
        try:
            # Step 1: Initialize
            self.initialize()
            
            # Step 2: Load data
            data = self.load_training_data()
            logger.info(f"Loaded {len(data)} training samples")
            
            # Step 3: Train simple model
            training_results = self.train_simple_model(data)
            
            # Step 4: Save results
            results_file = Path(self.config['logs_path']) / f'simple_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_type': 'simple',
                    'training_results': training_results,
                    'data_shape': list(data.shape),
                    'data_columns': data.columns.tolist()
                }, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
            logger.info("="*60)
            logger.info("Simple ML Training Pipeline completed successfully")
            logger.info("="*60)
            
            return training_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main entry point"""
    pipeline = SimpleMLTrainingPipeline()
    return pipeline.run_training_pipeline()

if __name__ == "__main__":
    # Run the pipeline
    results = main()
    print(f"Training completed with status: {results.get('status', 'unknown')}")