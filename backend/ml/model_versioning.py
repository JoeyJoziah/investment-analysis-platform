"""
ML Model Versioning System
Provides comprehensive model versioning with semantic versioning and model registry
"""

import os
import json
import pickle
import joblib
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    RETIRED = "retired"
    ARCHIVED = "archived"


class ModelType(Enum):
    """Supported model types"""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_name: str
    version: str
    model_type: ModelType
    stage: ModelStage
    created_at: datetime
    created_by: str
    description: str
    tags: List[str]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    model_size: int
    training_data_hash: str
    feature_names: List[str]
    model_path: str
    metadata_path: str
    performance_benchmark: Dict[str, float]
    dependencies: Dict[str, str]
    git_commit: Optional[str] = None
    parent_version: Optional[str] = None
    is_champion: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['model_type'] = self.model_type.value
        data['stage'] = self.stage.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        data['model_type'] = ModelType(data['model_type'])
        data['stage'] = ModelStage(data['stage'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    name: str
    description: str
    champion_version: str
    challenger_version: str
    traffic_split: float  # Percentage for challenger (0-100)
    start_date: datetime
    end_date: datetime
    success_metrics: List[str]
    minimum_sample_size: int
    statistical_significance: float = 0.95
    status: str = "active"  # active, completed, paused, cancelled


class ModelVersionManager:
    """
    Comprehensive model version management system
    """
    
    def __init__(self, 
                 registry_path: str = "/app/ml_models/registry",
                 storage_path: str = "/app/ml_models/versions",
                 enable_git_tracking: bool = False):
        self.registry_path = Path(registry_path)
        self.storage_path = Path(storage_path)
        self.enable_git_tracking = enable_git_tracking
        self.lock = threading.Lock()
        
        # Create directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self.registry_file = self.registry_path / "registry.json"
        self.ab_tests_file = self.registry_path / "ab_tests.json"
        self.model_registry = self._load_registry()
        self.ab_tests = self._load_ab_tests()
        
        logger.info(f"Model version manager initialized with {len(self.model_registry)} models")
    
    def _load_registry(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load model registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                registry = {}
                for model_name, versions_data in data.items():
                    registry[model_name] = {}
                    for version, version_data in versions_data.items():
                        registry[model_name][version] = ModelVersion.from_dict(version_data)
                
                return registry
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model registry to disk"""
        try:
            data = {}
            for model_name, versions in self.model_registry.items():
                data[model_name] = {}
                for version, version_obj in versions.items():
                    data[model_name][version] = version_obj.to_dict()
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _load_ab_tests(self) -> Dict[str, ABTestConfig]:
        """Load A/B test configurations"""
        if self.ab_tests_file.exists():
            try:
                with open(self.ab_tests_file, 'r') as f:
                    data = json.load(f)
                
                ab_tests = {}
                for test_name, test_data in data.items():
                    test_data['start_date'] = datetime.fromisoformat(test_data['start_date'])
                    test_data['end_date'] = datetime.fromisoformat(test_data['end_date'])
                    ab_tests[test_name] = ABTestConfig(**test_data)
                
                return ab_tests
            except Exception as e:
                logger.error(f"Error loading A/B tests: {e}")
                return {}
        return {}
    
    def _save_ab_tests(self):
        """Save A/B test configurations"""
        try:
            data = {}
            for test_name, test_config in self.ab_tests.items():
                config_dict = asdict(test_config)
                config_dict['start_date'] = config_dict['start_date'].isoformat()
                config_dict['end_date'] = config_dict['end_date'].isoformat()
                data[test_name] = config_dict
            
            with open(self.ab_tests_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving A/B tests: {e}")
    
    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute hash of model file"""
        hasher = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _compute_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """Compute hash of training data"""
        hasher = hashlib.sha256()
        if isinstance(data, pd.DataFrame):
            hasher.update(data.to_csv(index=False).encode())
        else:
            hasher.update(data.tobytes())
        return hasher.hexdigest()
    
    def _get_next_version(self, model_name: str, version_type: str = "patch") -> str:
        """Generate next semantic version"""
        if model_name not in self.model_registry:
            return "1.0.0"
        
        versions = list(self.model_registry[model_name].keys())
        latest_version = max(versions, key=lambda x: tuple(map(int, x.split('.'))))
        
        major, minor, patch = map(int, latest_version.split('.'))
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def _save_model_artifact(self, 
                           model: Any, 
                           model_path: Path, 
                           model_type: ModelType) -> int:
        """Save model artifact and return file size"""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_type == ModelType.PYTORCH:
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        elif model_type in [ModelType.SKLEARN, ModelType.ENSEMBLE]:
            joblib.dump(model, model_path)
        elif model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.PROPHET]:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            # Generic pickle fallback
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        return model_path.stat().st_size
    
    def register_model(self,
                      model_name: str,
                      model: Any,
                      model_type: ModelType,
                      description: str,
                      training_data: Union[pd.DataFrame, np.ndarray],
                      feature_names: List[str],
                      metrics: Dict[str, float],
                      parameters: Dict[str, Any],
                      tags: List[str] = None,
                      version_type: str = "patch",
                      created_by: str = "system") -> str:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model
            model: Trained model object
            model_type: Type of model
            description: Description of this version
            training_data: Training data used
            feature_names: List of feature names
            metrics: Performance metrics
            parameters: Model parameters/hyperparameters
            tags: Optional tags
            version_type: Version increment type (major, minor, patch)
            created_by: Who created this version
            
        Returns:
            Version string of registered model
        """
        with self.lock:
            try:
                # Generate version
                version = self._get_next_version(model_name, version_type)
                
                # Create paths
                version_dir = self.storage_path / model_name / version
                model_path = version_dir / f"model.{model_type.value}"
                metadata_path = version_dir / "metadata.json"
                
                # Save model
                model_size = self._save_model_artifact(model, model_path, model_type)
                
                # Compute hashes
                training_data_hash = self._compute_data_hash(training_data)
                
                # Get git commit if enabled
                git_commit = None
                if self.enable_git_tracking:
                    try:
                        import subprocess
                        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            git_commit = result.stdout.strip()
                    except:
                        pass
                
                # Get dependencies
                dependencies = self._get_current_dependencies()
                
                # Create model version
                model_version = ModelVersion(
                    model_name=model_name,
                    version=version,
                    model_type=model_type,
                    stage=ModelStage.DEVELOPMENT,
                    created_at=datetime.utcnow(),
                    created_by=created_by,
                    description=description,
                    tags=tags or [],
                    metrics=metrics,
                    parameters=parameters,
                    model_size=model_size,
                    training_data_hash=training_data_hash,
                    feature_names=feature_names,
                    model_path=str(model_path),
                    metadata_path=str(metadata_path),
                    performance_benchmark=self._compute_benchmark_metrics(metrics),
                    dependencies=dependencies,
                    git_commit=git_commit
                )
                
                # Save metadata
                with open(metadata_path, 'w') as f:
                    json.dump(model_version.to_dict(), f, indent=2)
                
                # Update registry
                if model_name not in self.model_registry:
                    self.model_registry[model_name] = {}
                self.model_registry[model_name][version] = model_version
                
                # Save registry
                self._save_registry()
                
                logger.info(f"Registered model {model_name} version {version}")
                return version
                
            except Exception as e:
                logger.error(f"Error registering model {model_name}: {e}")
                raise
    
    def _get_current_dependencies(self) -> Dict[str, str]:
        """Get current package dependencies"""
        try:
            import pkg_resources
            dependencies = {}
            for pkg in ['scikit-learn', 'torch', 'xgboost', 'lightgbm', 'prophet', 'numpy', 'pandas']:
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    dependencies[pkg] = version
                except:
                    pass
            return dependencies
        except:
            return {}
    
    def _compute_benchmark_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute standardized benchmark metrics"""
        benchmark = {}
        
        # Classification metrics
        if 'accuracy' in metrics:
            benchmark['accuracy_percentile'] = min(metrics['accuracy'] * 100, 100)
        if 'f1_score' in metrics:
            benchmark['f1_percentile'] = min(metrics['f1_score'] * 100, 100)
        
        # Regression metrics
        if 'r2_score' in metrics:
            benchmark['r2_percentile'] = min(max(metrics['r2_score'] * 100, 0), 100)
        if 'mse' in metrics:
            # Lower is better for MSE
            benchmark['mse_inverse_percentile'] = max(100 - metrics['mse'] * 100, 0)
        
        # Financial metrics
        if 'sharpe_ratio' in metrics:
            # Normalize Sharpe ratio (2.0 = excellent)
            benchmark['sharpe_percentile'] = min(metrics['sharpe_ratio'] / 2.0 * 100, 100)
        
        if 'directional_accuracy' in metrics:
            benchmark['directional_accuracy_percentile'] = metrics['directional_accuracy'] * 100
        
        return benchmark
    
    def promote_model(self, model_name: str, version: str, target_stage: ModelStage) -> bool:
        """Promote model to a different stage"""
        with self.lock:
            try:
                if model_name not in self.model_registry:
                    raise ValueError(f"Model {model_name} not found")
                
                if version not in self.model_registry[model_name]:
                    raise ValueError(f"Version {version} not found for model {model_name}")
                
                model_version = self.model_registry[model_name][version]
                
                # Validation rules
                if target_stage == ModelStage.PRODUCTION:
                    # Must have passed staging and meet minimum requirements
                    if model_version.stage not in [ModelStage.STAGING]:
                        raise ValueError("Model must be in STAGING before promotion to PRODUCTION")
                    
                    # Check benchmark metrics
                    if not self._validate_production_requirements(model_version):
                        raise ValueError("Model does not meet production requirements")
                
                # Update stage
                old_stage = model_version.stage
                model_version.stage = target_stage
                
                # If promoting to production, demote current champion
                if target_stage == ModelStage.PRODUCTION:
                    self._update_champion_model(model_name, version)
                
                # Save changes
                self._save_registry()
                
                logger.info(f"Promoted model {model_name}:{version} from {old_stage.value} to {target_stage.value}")
                return True
                
            except Exception as e:
                logger.error(f"Error promoting model {model_name}:{version}: {e}")
                return False
    
    def _validate_production_requirements(self, model_version: ModelVersion) -> bool:
        """Validate model meets production requirements"""
        benchmarks = model_version.performance_benchmark
        
        # Minimum accuracy requirements
        if 'accuracy_percentile' in benchmarks and benchmarks['accuracy_percentile'] < 60:
            return False
        if 'directional_accuracy_percentile' in benchmarks and benchmarks['directional_accuracy_percentile'] < 55:
            return False
        if 'sharpe_percentile' in benchmarks and benchmarks['sharpe_percentile'] < 30:
            return False
        
        return True
    
    def _update_champion_model(self, model_name: str, new_champion_version: str):
        """Update champion model designation"""
        if model_name in self.model_registry:
            # Remove champion flag from all versions
            for version_obj in self.model_registry[model_name].values():
                version_obj.is_champion = False
            
            # Set new champion
            if new_champion_version in self.model_registry[model_name]:
                self.model_registry[model_name][new_champion_version].is_champion = True
    
    def rollback_model(self, model_name: str, target_version: str) -> bool:
        """Rollback to a previous model version"""
        with self.lock:
            try:
                if model_name not in self.model_registry:
                    raise ValueError(f"Model {model_name} not found")
                
                if target_version not in self.model_registry[model_name]:
                    raise ValueError(f"Version {target_version} not found")
                
                target_model = self.model_registry[model_name][target_version]
                
                # Promote target version to production
                target_model.stage = ModelStage.PRODUCTION
                self._update_champion_model(model_name, target_version)
                
                # Save changes
                self._save_registry()
                
                logger.info(f"Rolled back model {model_name} to version {target_version}")
                return True
                
            except Exception as e:
                logger.error(f"Error rolling back model {model_name}: {e}")
                return False
    
    def load_model(self, model_name: str, version: str = None) -> Tuple[Any, ModelVersion]:
        """Load a specific model version"""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found")
            
            # Get version (default to champion or latest production)
            if version is None:
                version = self._get_default_version(model_name)
            
            if version not in self.model_registry[model_name]:
                raise ValueError(f"Version {version} not found for model {model_name}")
            
            model_version = self.model_registry[model_name][version]
            model_path = Path(model_version.model_path)
            
            # Load model based on type
            if model_version.model_type == ModelType.PYTORCH:
                model = torch.load(model_path, map_location='cpu')
            elif model_version.model_type in [ModelType.SKLEARN, ModelType.ENSEMBLE]:
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            return model, model_version
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}:{version}: {e}")
            raise
    
    def _get_default_version(self, model_name: str) -> str:
        """Get default version (champion or latest production)"""
        versions = self.model_registry[model_name]
        
        # Look for champion first
        for version, version_obj in versions.items():
            if version_obj.is_champion and version_obj.stage == ModelStage.PRODUCTION:
                return version
        
        # Look for latest production
        production_versions = [
            (version, version_obj) for version, version_obj in versions.items()
            if version_obj.stage == ModelStage.PRODUCTION
        ]
        
        if production_versions:
            # Sort by version number
            latest = max(production_versions, key=lambda x: tuple(map(int, x[0].split('.'))))
            return latest[0]
        
        # Fallback to latest version
        return max(versions.keys(), key=lambda x: tuple(map(int, x.split('.'))))
    
    def create_ab_test(self, 
                      name: str,
                      description: str,
                      champion_version: str,
                      challenger_version: str,
                      model_name: str,
                      traffic_split: float = 10.0,
                      duration_days: int = 14,
                      success_metrics: List[str] = None) -> bool:
        """Create A/B test comparing two model versions"""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found")
            
            if champion_version not in self.model_registry[model_name]:
                raise ValueError(f"Champion version {champion_version} not found")
            
            if challenger_version not in self.model_registry[model_name]:
                raise ValueError(f"Challenger version {challenger_version} not found")
            
            if name in self.ab_tests:
                raise ValueError(f"A/B test {name} already exists")
            
            ab_test = ABTestConfig(
                name=name,
                description=description,
                champion_version=f"{model_name}:{champion_version}",
                challenger_version=f"{model_name}:{challenger_version}",
                traffic_split=traffic_split,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=duration_days),
                success_metrics=success_metrics or ['accuracy', 'directional_accuracy'],
                minimum_sample_size=1000
            )
            
            self.ab_tests[name] = ab_test
            self._save_ab_tests()
            
            logger.info(f"Created A/B test {name}: {champion_version} vs {challenger_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            return False
    
    def get_ab_test_model(self, test_name: str, user_id: str = None) -> str:
        """Get model version for A/B test based on traffic split"""
        if test_name not in self.ab_tests:
            return None
        
        test_config = self.ab_tests[test_name]
        
        # Check if test is active
        now = datetime.utcnow()
        if now < test_config.start_date or now > test_config.end_date:
            return test_config.champion_version
        
        # Simple hash-based assignment
        if user_id:
            hash_value = hash(f"{test_name}:{user_id}") % 100
        else:
            hash_value = hash(str(datetime.utcnow().timestamp())) % 100
        
        if hash_value < test_config.traffic_split:
            return test_config.challenger_version
        else:
            return test_config.champion_version
    
    def get_model_comparison(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        if model_name not in self.model_registry:
            return {}
        
        versions = self.model_registry[model_name]
        if version1 not in versions or version2 not in versions:
            return {}
        
        v1 = versions[version1]
        v2 = versions[version2]
        
        comparison = {
            'model_name': model_name,
            'version1': {
                'version': version1,
                'stage': v1.stage.value,
                'created_at': v1.created_at.isoformat(),
                'metrics': v1.metrics,
                'benchmark': v1.performance_benchmark,
                'size_mb': v1.model_size / (1024 * 1024)
            },
            'version2': {
                'version': version2,
                'stage': v2.stage.value,
                'created_at': v2.created_at.isoformat(),
                'metrics': v2.metrics,
                'benchmark': v2.performance_benchmark,
                'size_mb': v2.model_size / (1024 * 1024)
            },
            'differences': {}
        }
        
        # Compare metrics
        for metric in set(v1.metrics.keys()) | set(v2.metrics.keys()):
            val1 = v1.metrics.get(metric, 0)
            val2 = v2.metrics.get(metric, 0)
            comparison['differences'][metric] = {
                'absolute_diff': val2 - val1,
                'relative_diff': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            }
        
        return comparison
    
    def cleanup_old_versions(self, model_name: str, keep_count: int = 5) -> int:
        """Clean up old model versions, keeping specified count"""
        if model_name not in self.model_registry:
            return 0
        
        versions = self.model_registry[model_name]
        
        # Never delete production or champion models
        protected_versions = set()
        for version, version_obj in versions.items():
            if (version_obj.stage in [ModelStage.PRODUCTION, ModelStage.STAGING] or 
                version_obj.is_champion):
                protected_versions.add(version)
        
        # Sort versions by creation date
        version_items = list(versions.items())
        version_items.sort(key=lambda x: x[1].created_at, reverse=True)
        
        # Determine versions to delete
        to_delete = []
        kept = 0
        
        for version, version_obj in version_items:
            if version in protected_versions:
                continue
            
            if kept >= keep_count:
                to_delete.append(version)
            else:
                kept += 1
        
        # Delete old versions
        deleted_count = 0
        for version in to_delete:
            try:
                version_obj = versions[version]
                # Delete files
                version_dir = Path(version_obj.model_path).parent
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                
                # Remove from registry
                del self.model_registry[model_name][version]
                deleted_count += 1
                
            except Exception as e:
                logger.error(f"Error deleting version {version}: {e}")
        
        if deleted_count > 0:
            self._save_registry()
            logger.info(f"Cleaned up {deleted_count} old versions of {model_name}")
        
        return deleted_count
    
    def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model lineage and ancestry"""
        if model_name not in self.model_registry:
            return {}
        
        if version not in self.model_registry[model_name]:
            return {}
        
        current = self.model_registry[model_name][version]
        lineage = {
            'current': {
                'version': version,
                'created_at': current.created_at.isoformat(),
                'created_by': current.created_by,
                'description': current.description
            },
            'ancestors': [],
            'descendants': []
        }
        
        # Find ancestors
        parent_version = current.parent_version
        while parent_version and parent_version in self.model_registry[model_name]:
            parent = self.model_registry[model_name][parent_version]
            lineage['ancestors'].append({
                'version': parent_version,
                'created_at': parent.created_at.isoformat(),
                'created_by': parent.created_by,
                'description': parent.description
            })
            parent_version = parent.parent_version
        
        # Find descendants
        for v, version_obj in self.model_registry[model_name].items():
            if version_obj.parent_version == version:
                lineage['descendants'].append({
                    'version': v,
                    'created_at': version_obj.created_at.isoformat(),
                    'created_by': version_obj.created_by,
                    'description': version_obj.description
                })
        
        return lineage
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        stats = {
            'total_models': len(self.model_registry),
            'total_versions': 0,
            'models_by_type': {},
            'models_by_stage': {},
            'storage_usage_mb': 0,
            'active_ab_tests': 0,
            'recent_activity': []
        }
        
        # Count versions and collect stats
        for model_name, versions in self.model_registry.items():
            stats['total_versions'] += len(versions)
            
            for version, version_obj in versions.items():
                # Count by type
                model_type = version_obj.model_type.value
                stats['models_by_type'][model_type] = stats['models_by_type'].get(model_type, 0) + 1
                
                # Count by stage
                stage = version_obj.stage.value
                stats['models_by_stage'][stage] = stats['models_by_stage'].get(stage, 0) + 1
                
                # Storage usage
                stats['storage_usage_mb'] += version_obj.model_size / (1024 * 1024)
                
                # Recent activity
                if version_obj.created_at > datetime.utcnow() - timedelta(days=7):
                    stats['recent_activity'].append({
                        'model_name': model_name,
                        'version': version,
                        'action': 'registered',
                        'timestamp': version_obj.created_at.isoformat()
                    })
        
        # Active A/B tests
        now = datetime.utcnow()
        for test in self.ab_tests.values():
            if test.start_date <= now <= test.end_date and test.status == 'active':
                stats['active_ab_tests'] += 1
        
        # Sort recent activity
        stats['recent_activity'].sort(key=lambda x: x['timestamp'], reverse=True)
        stats['recent_activity'] = stats['recent_activity'][:20]  # Last 20 activities
        
        return stats


# Global instance
_model_version_manager: Optional[ModelVersionManager] = None

def get_model_version_manager() -> ModelVersionManager:
    """Get global model version manager instance"""
    global _model_version_manager
    if _model_version_manager is None:
        _model_version_manager = ModelVersionManager()
    return _model_version_manager