"""
Model Registry - Version control and tracking for ML models
"""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import pickle
import joblib
import torch
import pandas as pd
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import mlflow
from packaging import version

logger = logging.getLogger(__name__)

Base = declarative_base()


class DeploymentStatus(Enum):
    """Model deployment status"""
    NOT_DEPLOYED = "not_deployed"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    ROLLING_BACK = "rolling_back"


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Represents a specific version of a model"""
    model_id: str
    model_name: str
    version: str
    
    # Paths
    model_path: Path
    artifacts_path: Path
    
    # Metadata
    created_at: datetime
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Training info
    training_data_hash: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Validation metrics
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Deployment info
    stage: ModelStage = ModelStage.DEVELOPMENT
    deployment_status: DeploymentStatus = DeploymentStatus.NOT_DEPLOYED
    deployed_at: Optional[datetime] = None
    deployment_endpoint: Optional[str] = None
    
    # Lineage
    parent_version: Optional[str] = None
    child_versions: List[str] = field(default_factory=list)
    
    # Performance tracking
    production_metrics: Dict[str, float] = field(default_factory=dict)
    inference_count: int = 0
    last_inference_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_path": str(self.model_path),
            "artifacts_path": str(self.artifacts_path),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "training_data_hash": self.training_data_hash,
            "training_config": self.training_config,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "stage": self.stage.value,
            "deployment_status": self.deployment_status.value,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "deployment_endpoint": self.deployment_endpoint,
            "parent_version": self.parent_version,
            "child_versions": self.child_versions,
            "production_metrics": self.production_metrics,
            "inference_count": self.inference_count,
            "last_inference_time": self.last_inference_time.isoformat() if self.last_inference_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create from dictionary"""
        # Convert string dates to datetime
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("deployed_at"):
            data["deployed_at"] = datetime.fromisoformat(data["deployed_at"])
        if data.get("last_inference_time"):
            data["last_inference_time"] = datetime.fromisoformat(data["last_inference_time"])
        
        # Convert paths
        data["model_path"] = Path(data["model_path"])
        data["artifacts_path"] = Path(data["artifacts_path"])
        
        # Convert enums
        data["stage"] = ModelStage(data["stage"])
        data["deployment_status"] = DeploymentStatus(data["deployment_status"])
        
        return cls(**data)


@dataclass
class ModelMetadata:
    """Extended metadata for models"""
    model_id: str
    
    # Documentation
    readme: str = ""
    api_documentation: str = ""
    training_documentation: str = ""
    
    # Requirements
    python_version: str = ""
    dependencies: List[str] = field(default_factory=list)
    system_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Data schema
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)
    
    # Governance
    approval_status: str = "pending"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    compliance_checks: Dict[str, bool] = field(default_factory=dict)
    
    # Cost tracking
    training_cost_usd: float = 0.0
    inference_cost_per_prediction_usd: float = 0.0
    total_inference_cost_usd: float = 0.0
    
    # SLA
    target_latency_ms: float = 100.0
    target_throughput_rps: float = 100.0
    target_availability: float = 0.99


class ModelRegistryDB(Base):
    """Database table for model registry"""
    __tablename__ = "model_registry"
    
    model_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, default="system")
    
    # Paths
    model_path = Column(String)
    artifacts_path = Column(String)
    
    # Metadata
    description = Column(Text)
    tags = Column(JSON)
    training_config = Column(JSON)
    training_metrics = Column(JSON)
    validation_metrics = Column(JSON)
    test_metrics = Column(JSON)
    
    # Deployment
    stage = Column(String, default="development")
    deployment_status = Column(String, default="not_deployed")
    deployed_at = Column(DateTime)
    deployment_endpoint = Column(String)
    
    # Lineage
    parent_version = Column(String)
    
    # Performance
    production_metrics = Column(JSON)
    inference_count = Column(Integer, default=0)
    last_inference_time = Column(DateTime)


class ModelRegistry:
    """
    Central registry for ML models
    Handles versioning, storage, and lifecycle management
    """
    
    def __init__(self, registry_path: str = "/app/ml_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.models_path = self.registry_path / "models"
        self.artifacts_path = self.registry_path / "artifacts"
        self.metadata_path = self.registry_path / "metadata"
        
        for path in [self.models_path, self.artifacts_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Registry database
        self.registry_file = self.registry_path / "registry.json"
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # {model_name: {version: ModelVersion}}
        
        # Database connection (optional)
        self.db_engine = None
        self.Session = None
        self._init_database()
        
        # MLflow integration (optional)
        self.mlflow_enabled = False
        self._init_mlflow()
        
        # Load existing registry
        self._load_registry()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            db_url = "sqlite:///" + str(self.registry_path / "registry.db")
            self.db_engine = create_engine(db_url)
            Base.metadata.create_all(self.db_engine)
            self.Session = sessionmaker(bind=self.db_engine)
            logger.info("Model registry database initialized")
        except Exception as e:
            logger.warning(f"Could not initialize database: {e}")
    
    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri(str(self.registry_path / "mlruns"))
            self.mlflow_enabled = True
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}")
    
    def _load_registry(self):
        """Load existing registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for model_name, versions in registry_data.items():
                    self.models[model_name] = {}
                    for ver, version_data in versions.items():
                        self.models[model_name][ver] = ModelVersion.from_dict(version_data)
                
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            registry_data = {}
            for model_name, versions in self.models.items():
                registry_data[model_name] = {}
                for ver, model_version in versions.items():
                    registry_data[model_name][ver] = model_version.to_dict()
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            logger.info("Registry saved to disk")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    async def register_model(
        self,
        model_artifact: Any,
        metadata: Dict[str, Any] = None
    ) -> ModelVersion:
        """Register a new model version"""
        metadata = metadata or {}
        
        # Generate model ID
        model_name = model_artifact.name
        model_version_str = model_artifact.version
        model_id = f"{model_name}_{model_version_str}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create version directory
        version_path = self.models_path / model_name / model_version_str
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Save model files
        model_file = version_path / "model.pkl"
        if hasattr(model_artifact, 'model_path'):
            shutil.copy2(model_artifact.model_path, model_file)
        
        # Create ModelVersion object
        model_version = ModelVersion(
            model_id=model_id,
            model_name=model_name,
            version=model_version_str,
            model_path=model_file,
            artifacts_path=version_path,
            created_at=datetime.utcnow(),
            created_by=metadata.get("created_by", "system"),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", []),
            training_config=model_artifact.hyperparameters,
            training_metrics=model_artifact.metrics,
            validation_metrics=model_artifact.validation_metrics,
            test_metrics=model_artifact.test_metrics,
            stage=ModelStage.DEVELOPMENT
        )
        
        # Check for parent version
        if model_name in self.models:
            # Find latest version as parent
            latest_version = self._get_latest_version(model_name)
            if latest_version:
                model_version.parent_version = latest_version.version
                latest_version.child_versions.append(model_version_str)
        
        # Add to registry
        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][model_version_str] = model_version
        
        # Save to database if available
        if self.Session:
            await self._save_to_database(model_version)
        
        # Track with MLflow if enabled
        if self.mlflow_enabled:
            await self._track_with_mlflow(model_version, model_artifact)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model: {model_name} v{model_version_str}")
        return model_version
    
    async def _save_to_database(self, model_version: ModelVersion):
        """Save model version to database"""
        try:
            session = self.Session()
            
            db_model = ModelRegistryDB(
                model_id=model_version.model_id,
                model_name=model_version.model_name,
                version=model_version.version,
                created_at=model_version.created_at,
                created_by=model_version.created_by,
                model_path=str(model_version.model_path),
                artifacts_path=str(model_version.artifacts_path),
                description=model_version.description,
                tags=model_version.tags,
                training_config=model_version.training_config,
                training_metrics=model_version.training_metrics,
                validation_metrics=model_version.validation_metrics,
                test_metrics=model_version.test_metrics,
                stage=model_version.stage.value,
                deployment_status=model_version.deployment_status.value,
                parent_version=model_version.parent_version
            )
            
            session.add(db_model)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    async def _track_with_mlflow(self, model_version: ModelVersion, model_artifact: Any):
        """Track model with MLflow"""
        try:
            with mlflow.start_run(run_name=f"{model_version.model_name}_v{model_version.version}"):
                # Log parameters
                mlflow.log_params(model_version.training_config)
                
                # Log metrics
                mlflow.log_metrics(model_version.training_metrics)
                mlflow.log_metrics({f"val_{k}": v for k, v in model_version.validation_metrics.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in model_version.test_metrics.items()})
                
                # Log model
                if hasattr(model_artifact, 'model'):
                    if isinstance(model_artifact.model, torch.nn.Module):
                        mlflow.pytorch.log_model(model_artifact.model, "model")
                    else:
                        mlflow.sklearn.log_model(model_artifact.model, "model")
                
                # Log artifacts
                mlflow.log_artifact(str(model_version.model_path))
                
        except Exception as e:
            logger.error(f"Error tracking with MLflow: {e}")
    
    def _get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        if model_name not in self.models:
            return None
        
        versions = list(self.models[model_name].keys())
        if not versions:
            return None
        
        # Sort versions using packaging.version
        sorted_versions = sorted(versions, key=lambda v: version.parse(v), reverse=True)
        return self.models[model_name][sorted_versions[0]]
    
    async def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """Get a specific model version"""
        if model_name not in self.models:
            return None
        
        if version:
            return self.models[model_name].get(version)
        
        if stage:
            # Find model in specified stage
            for ver, model_version in self.models[model_name].items():
                if model_version.stage == stage:
                    return model_version
        
        # Return latest version
        return self._get_latest_version(model_name)
    
    async def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage
    ) -> bool:
        """Promote a model to a different stage"""
        model_version = await self.get_model(model_name, version)
        if not model_version:
            logger.error(f"Model not found: {model_name} v{version}")
            return False
        
        # Validate promotion path
        valid_promotions = {
            ModelStage.DEVELOPMENT: [ModelStage.TESTING, ModelStage.STAGING],
            ModelStage.TESTING: [ModelStage.STAGING, ModelStage.DEVELOPMENT],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.TESTING],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED, ModelStage.STAGING],
            ModelStage.ARCHIVED: []
        }
        
        if target_stage not in valid_promotions.get(model_version.stage, []):
            logger.error(f"Invalid promotion from {model_version.stage} to {target_stage}")
            return False
        
        # Update stage
        old_stage = model_version.stage
        model_version.stage = target_stage
        
        # If promoting to production, demote current production model
        if target_stage == ModelStage.PRODUCTION:
            for ver, mv in self.models[model_name].items():
                if mv.stage == ModelStage.PRODUCTION and ver != version:
                    mv.stage = ModelStage.STAGING
        
        # Save changes
        self._save_registry()
        
        logger.info(f"Promoted {model_name} v{version} from {old_stage} to {target_stage}")
        return True
    
    async def deploy_model(
        self,
        model_name: str,
        version: str,
        endpoint: str,
        deployment_status: DeploymentStatus = DeploymentStatus.STAGING
    ) -> bool:
        """Mark a model as deployed"""
        model_version = await self.get_model(model_name, version)
        if not model_version:
            return False
        
        model_version.deployment_status = deployment_status
        model_version.deployed_at = datetime.utcnow()
        model_version.deployment_endpoint = endpoint
        
        # Save changes
        self._save_registry()
        
        logger.info(f"Deployed {model_name} v{version} to {endpoint}")
        return True
    
    async def get_active_models(self) -> List[Dict]:
        """Get all active (non-archived) models"""
        active_models = []
        
        for model_name, versions in self.models.items():
            for ver, model_version in versions.items():
                if model_version.stage != ModelStage.ARCHIVED:
                    active_models.append(model_version.to_dict())
        
        return active_models
    
    async def get_deployed_models(self) -> List[Dict]:
        """Get all deployed models"""
        deployed_models = []
        
        for model_name, versions in self.models.items():
            for ver, model_version in versions.items():
                if model_version.deployment_status in [
                    DeploymentStatus.STAGING,
                    DeploymentStatus.PRODUCTION
                ]:
                    deployed_models.append(model_version.to_dict())
        
        return deployed_models
    
    async def update_production_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float]
    ):
        """Update production metrics for a deployed model"""
        model_version = await self.get_model(model_name, version)
        if not model_version:
            return
        
        model_version.production_metrics.update(metrics)
        model_version.last_inference_time = datetime.utcnow()
        model_version.inference_count += metrics.get("batch_size", 1)
        
        # Save changes
        self._save_registry()
    
    async def compare_models(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict:
        """Compare two model versions"""
        mv1 = await self.get_model(model_name, version1)
        mv2 = await self.get_model(model_name, version2)
        
        if not mv1 or not mv2:
            return {"error": "One or both models not found"}
        
        comparison = {
            "model_name": model_name,
            "versions": {
                "v1": version1,
                "v2": version2
            },
            "metrics_comparison": {
                "training": self._compare_metrics(mv1.training_metrics, mv2.training_metrics),
                "validation": self._compare_metrics(mv1.validation_metrics, mv2.validation_metrics),
                "test": self._compare_metrics(mv1.test_metrics, mv2.test_metrics),
                "production": self._compare_metrics(mv1.production_metrics, mv2.production_metrics)
            },
            "deployment_status": {
                "v1": mv1.deployment_status.value,
                "v2": mv2.deployment_status.value
            },
            "stage": {
                "v1": mv1.stage.value,
                "v2": mv2.stage.value
            },
            "inference_count": {
                "v1": mv1.inference_count,
                "v2": mv2.inference_count
            }
        }
        
        return comparison
    
    def _compare_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """Compare two sets of metrics"""
        comparison = {}
        all_keys = set(metrics1.keys()) | set(metrics2.keys())
        
        for key in all_keys:
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            
            if val1 != 0:
                improvement = ((val2 - val1) / val1) * 100
            else:
                improvement = 0
            
            comparison[key] = {
                "v1": val1,
                "v2": val2,
                "improvement_pct": round(improvement, 2)
            }
        
        return comparison
    
    async def cleanup_old_models(self, days_to_keep: int = 90):
        """Archive old models not in production"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        archived_count = 0
        
        for model_name, versions in self.models.items():
            for ver, model_version in versions.items():
                if (model_version.created_at < cutoff_date and
                    model_version.stage not in [ModelStage.PRODUCTION, ModelStage.STAGING] and
                    model_version.deployment_status == DeploymentStatus.NOT_DEPLOYED):
                    
                    model_version.stage = ModelStage.ARCHIVED
                    archived_count += 1
        
        if archived_count > 0:
            self._save_registry()
            logger.info(f"Archived {archived_count} old models")
        
        return archived_count
    
    async def load_registry(self):
        """Public method to load registry"""
        self._load_registry()