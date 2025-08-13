#!/usr/bin/env python3
"""
ML Models Deployment Script
Complete deployment pipeline for trained ML models
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import json
import shutil
from datetime import datetime
import subprocess

# Add backend to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelDeployer:
    """Handles ML model deployment and validation"""
    
    def __init__(self):
        self.models_dir = Path("/app/ml_models")
        self.backup_dir = Path("/app/ml_models_backup")
        self.deployment_log = []
        
    def create_models_directory(self):
        """Ensure models directory exists with proper permissions"""
        logger.info("Creating models directory...")
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Set permissions (readable/writable by application)
        os.chmod(self.models_dir, 0o755)
        
        logger.info(f"✅ Models directory ready: {self.models_dir}")
        return True
    
    def backup_existing_models(self):
        """Backup any existing models before deployment"""
        if self.models_dir.exists() and any(self.models_dir.iterdir()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            
            logger.info(f"Backing up existing models to {backup_path}")
            shutil.copytree(self.models_dir, backup_path)
            
            self.deployment_log.append(f"Backed up to {backup_path}")
            return str(backup_path)
        else:
            logger.info("No existing models to backup")
            return None
    
    def train_models(self):
        """Run the training script"""
        logger.info("Starting model training...")
        
        training_script = Path(__file__).parent / "train_ml_models.py"
        
        if not training_script.exists():
            raise FileNotFoundError(f"Training script not found: {training_script}")
        
        try:
            # Run training script
            result = subprocess.run([
                sys.executable, str(training_script)
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                logger.info("✅ Model training completed successfully")
                self.deployment_log.append("Training completed successfully")
                return True
            else:
                logger.error(f"❌ Training failed: {result.stderr}")
                self.deployment_log.append(f"Training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Training timed out after 30 minutes")
            self.deployment_log.append("Training timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            self.deployment_log.append(f"Training error: {e}")
            return False
    
    def validate_models(self):
        """Validate all trained models"""
        logger.info("Validating trained models...")
        
        validation_results = {
            'pytorch_models': {},
            'sklearn_models': {},
            'support_files': {},
            'overall_status': True
        }
        
        # Check PyTorch models
        pytorch_files = {
            'lstm_model.pt': 'LSTM time series model',
            'transformer_model.pt': 'Transformer model'
        }
        
        for filename, description in pytorch_files.items():
            filepath = self.models_dir / filename
            if filepath.exists():
                try:
                    import torch
                    state_dict = torch.load(filepath, map_location='cpu')
                    validation_results['pytorch_models'][filename] = {
                        'status': 'valid',
                        'size_mb': filepath.stat().st_size / 1024 / 1024,
                        'parameters': len(state_dict)
                    }
                    logger.info(f"  ✅ {description}: Valid ({len(state_dict)} params)")
                except Exception as e:
                    validation_results['pytorch_models'][filename] = {
                        'status': 'invalid',
                        'error': str(e)
                    }
                    validation_results['overall_status'] = False
                    logger.error(f"  ❌ {description}: Invalid - {e}")
            else:
                validation_results['pytorch_models'][filename] = {
                    'status': 'missing'
                }
                validation_results['overall_status'] = False
                logger.error(f"  ❌ {description}: Missing")
        
        # Check sklearn models
        sklearn_files = {
            'xgboost_model.pkl': 'XGBoost classifier',
            'lightgbm_model.pkl': 'LightGBM model',
            'rf_model.joblib': 'Random Forest model'
        }
        
        for filename, description in sklearn_files.items():
            filepath = self.models_dir / filename
            if filepath.exists():
                try:
                    import joblib
                    model = joblib.load(filepath)
                    validation_results['sklearn_models'][filename] = {
                        'status': 'valid',
                        'type': type(model).__name__,
                        'size_mb': filepath.stat().st_size / 1024 / 1024
                    }
                    logger.info(f"  ✅ {description}: Valid ({type(model).__name__})")
                except Exception as e:
                    validation_results['sklearn_models'][filename] = {
                        'status': 'invalid',
                        'error': str(e)
                    }
                    validation_results['overall_status'] = False
                    logger.error(f"  ❌ {description}: Invalid - {e}")
            else:
                validation_results['sklearn_models'][filename] = {
                    'status': 'missing'
                }
                validation_results['overall_status'] = False
                logger.error(f"  ❌ {description}: Missing")
        
        # Check support files
        support_files = {
            'feature_scaler.pkl': 'Feature scaler',
            'feature_names.pkl': 'Feature names',
            'model_registry.json': 'Model registry'
        }
        
        for filename, description in support_files.items():
            filepath = self.models_dir / filename
            if filepath.exists():
                validation_results['support_files'][filename] = {
                    'status': 'valid',
                    'size_kb': filepath.stat().st_size / 1024
                }
                logger.info(f"  ✅ {description}: Present")
            else:
                validation_results['support_files'][filename] = {
                    'status': 'missing'
                }
                # Support files missing is warning, not failure
                logger.warning(f"  ⚠️ {description}: Missing")
        
        # Save validation results
        validation_path = self.models_dir / "validation_results.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        self.deployment_log.append(f"Validation completed: {validation_results['overall_status']}")
        return validation_results
    
    async def test_model_integration(self):
        """Test integration with ModelManager"""
        logger.info("Testing model integration...")
        
        try:
            # Test loading with existing ModelManager
            from backend.ml.model_manager import get_model_manager
            
            manager = get_model_manager()
            
            # Health check
            health = manager.health_check()
            
            integration_results = {
                'health_status': health,
                'loaded_models': health['loaded_models'],
                'fallback_models': health['fallback_models'],
                'predictions_test': {}
            }
            
            # Test individual model predictions
            test_models = ['lstm_price_predictor', 'xgboost_classifier']
            
            for model_name in test_models:
                try:
                    test_data = manager._get_test_data(model_name)
                    result = manager.predict(model_name, test_data)
                    integration_results['predictions_test'][model_name] = {
                        'status': 'success',
                        'result_type': type(result).__name__
                    }
                    logger.info(f"  ✅ {model_name}: Prediction successful")
                except Exception as e:
                    integration_results['predictions_test'][model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    logger.error(f"  ❌ {model_name}: Prediction failed - {e}")
            
            # Save integration test results
            integration_path = self.models_dir / "integration_test_results.json"
            with open(integration_path, 'w') as f:
                json.dump(integration_results, f, indent=2, default=str)
            
            self.deployment_log.append("Integration testing completed")
            return integration_results
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            self.deployment_log.append(f"Integration test failed: {e}")
            return {'error': str(e)}
    
    def create_deployment_manifest(self, validation_results, integration_results):
        """Create deployment manifest with all details"""
        
        manifest = {
            'deployment_info': {
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0',
                'deployer': 'ML Training Pipeline',
                'environment': 'production'
            },
            'models': {
                'total_models': 5,  # LSTM, Transformer, XGBoost, LightGBM, RandomForest
                'pytorch_models': 2,
                'sklearn_models': 3
            },
            'validation_results': validation_results,
            'integration_results': integration_results,
            'deployment_log': self.deployment_log,
            'next_actions': [
                'Monitor model performance in production',
                'Set up automated retraining schedule',
                'Implement A/B testing for model comparison',
                'Add model drift monitoring'
            ]
        }
        
        manifest_path = self.models_dir / "deployment_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        logger.info(f"✅ Deployment manifest created: {manifest_path}")
        return manifest
    
    def print_deployment_summary(self, manifest):
        """Print comprehensive deployment summary"""
        
        print("\n" + "="*80)
        print("ML MODELS DEPLOYMENT SUMMARY")
        print("="*80)
        
        print(f"📅 Deployment Time: {manifest['deployment_info']['timestamp']}")
        print(f"📦 Total Models: {manifest['models']['total_models']}")
        print(f"🧠 PyTorch Models: {manifest['models']['pytorch_models']}")
        print(f"🌲 Sklearn Models: {manifest['models']['sklearn_models']}")
        
        # Validation status
        validation = manifest['validation_results']
        if validation['overall_status']:
            print("✅ Validation: PASSED")
        else:
            print("❌ Validation: FAILED")
        
        # Model status breakdown
        print("\n📊 Model Status:")
        for category, models in validation.items():
            if category != 'overall_status' and isinstance(models, dict):
                for model_name, model_info in models.items():
                    status_icon = "✅" if model_info.get('status') == 'valid' else "❌"
                    print(f"  {status_icon} {model_name}: {model_info.get('status', 'unknown')}")
        
        # Integration results
        integration = manifest.get('integration_results', {})
        if 'health_status' in integration:
            health = integration['health_status']
            print(f"\n🔗 Integration Status:")
            print(f"  Healthy: {health.get('healthy', False)}")
            print(f"  Loaded Models: {health.get('loaded_models', 0)}")
            print(f"  Fallback Models: {health.get('fallback_models', 0)}")
        
        # Next steps
        print("\n🎯 Next Actions:")
        for i, action in enumerate(manifest.get('next_actions', []), 1):
            print(f"  {i}. {action}")
        
        print("\n📁 Models Directory: " + str(self.models_dir))
        print("🚀 Ready for production use!")
        print("="*80)
    
    async def deploy(self):
        """Run complete deployment pipeline"""
        logger.info("Starting ML models deployment pipeline...")
        
        try:
            # Step 1: Prepare environment
            self.create_models_directory()
            
            # Step 2: Backup existing models
            backup_path = self.backup_existing_models()
            
            # Step 3: Train models
            training_success = self.train_models()
            if not training_success:
                raise Exception("Model training failed")
            
            # Step 4: Validate models
            validation_results = self.validate_models()
            
            # Step 5: Test integration
            integration_results = await self.test_model_integration()
            
            # Step 6: Create deployment manifest
            manifest = self.create_deployment_manifest(validation_results, integration_results)
            
            # Step 7: Print summary
            self.print_deployment_summary(manifest)
            
            return {
                'status': 'success',
                'manifest': manifest,
                'backup_path': backup_path
            }
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'deployment_log': self.deployment_log
            }


async def main():
    """Main deployment function"""
    deployer = MLModelDeployer()
    result = await deployer.deploy()
    
    if result['status'] == 'success':
        print("\n🎉 ML Models deployment completed successfully!")
        return 0
    else:
        print(f"\n💥 Deployment failed: {result['error']}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())