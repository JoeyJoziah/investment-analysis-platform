"""
Base classes for ML Pipeline Framework
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import logging
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ModelType(Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline"""
    name: str
    version: str
    model_type: ModelType
    
    # Data configuration
    data_source: str
    feature_columns: List[str]
    target_column: str
    train_test_split: float = 0.8
    validation_split: float = 0.1
    
    # Training configuration
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Resource configuration
    max_memory_gb: float = 8.0
    max_training_time_hours: float = 24.0
    use_gpu: bool = False
    num_workers: int = 4
    
    # Cost optimization
    cost_limit_usd: float = 5.0
    spot_instances: bool = True
    
    # Feature engineering
    feature_selection_method: str = "mutual_info"
    max_features: int = 100
    scaling_method: str = "standard"
    
    # Model specific configs
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation metrics
    primary_metric: str = "accuracy"
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    
    # Output configuration
    output_path: str = "/app/ml_models"
    save_preprocessor: bool = True
    save_feature_importance: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    drift_detection: bool = True
    performance_threshold: float = 0.8
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value,
            "data_source": self.data_source,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "train_test_split": self.train_test_split,
            "validation_split": self.validation_split,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "max_memory_gb": self.max_memory_gb,
            "max_training_time_hours": self.max_training_time_hours,
            "use_gpu": self.use_gpu,
            "num_workers": self.num_workers,
            "cost_limit_usd": self.cost_limit_usd,
            "hyperparameters": self.hyperparameters,
            "primary_metric": self.primary_metric,
            "evaluation_metrics": self.evaluation_metrics,
            "output_path": self.output_path
        }
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


@dataclass
class ModelArtifact:
    """Represents a trained model artifact"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    
    # Paths
    model_path: Path
    preprocessor_path: Optional[Path] = None
    feature_columns_path: Optional[Path] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    training_duration_seconds: float = 0
    training_samples: int = 0
    feature_count: int = 0
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Configuration used
    config_hash: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Resource usage
    memory_usage_mb: float = 0
    training_cost_usd: float = 0
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Status
    is_deployed: bool = False
    deployment_endpoint: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert artifact to dictionary"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value,
            "model_path": str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path) if self.preprocessor_path else None,
            "created_at": self.created_at.isoformat(),
            "training_duration_seconds": self.training_duration_seconds,
            "training_samples": self.training_samples,
            "feature_count": self.feature_count,
            "metrics": self.metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "config_hash": self.config_hash,
            "hyperparameters": self.hyperparameters,
            "memory_usage_mb": self.memory_usage_mb,
            "training_cost_usd": self.training_cost_usd,
            "feature_importance": self.feature_importance,
            "is_deployed": self.is_deployed,
            "deployment_endpoint": self.deployment_endpoint
        }


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Artifacts produced
    model_artifact: Optional[ModelArtifact] = None
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    # Execution details
    steps_completed: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Resource usage
    total_memory_mb: float = 0
    total_compute_time_seconds: float = 0
    total_cost_usd: float = 0
    
    # Data statistics
    total_samples_processed: int = 0
    features_used: int = 0
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "model_artifact": self.model_artifact.to_dict() if self.model_artifact else None,
            "steps_completed": self.steps_completed,
            "current_step": self.current_step,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "total_memory_mb": self.total_memory_mb,
            "total_compute_time_seconds": self.total_compute_time_seconds,
            "total_cost_usd": self.total_cost_usd,
            "total_samples_processed": self.total_samples_processed,
            "features_used": self.features_used
        }


class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    async def execute(self, data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute the pipeline step
        
        Args:
            data: Input data for this step
            context: Shared context across pipeline steps
            
        Returns:
            Tuple of (output_data, updated_context)
        """
        pass
    
    async def validate_input(self, data: Any) -> bool:
        """Validate input data for this step"""
        return True
    
    async def cleanup(self):
        """Cleanup resources after execution"""
        pass


class ModelPipeline(ABC):
    """Abstract base class for ML pipelines"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps: List[PipelineStep] = []
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.result = None
        self._setup_pipeline()
    
    @abstractmethod
    def _setup_pipeline(self):
        """Setup pipeline steps"""
        pass
    
    async def execute(self) -> PipelineResult:
        """Execute the complete pipeline"""
        pipeline_id = f"{self.config.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        context = {
            "config": self.config,
            "pipeline_id": pipeline_id,
            "artifacts": {}
        }
        
        data = None
        
        try:
            for step in self.steps:
                self.result.current_step = step.name
                self.logger.info(f"Executing step: {step.name}")
                
                # Validate input
                if not await step.validate_input(data):
                    raise ValueError(f"Invalid input for step {step.name}")
                
                # Execute step
                data, context = await step.execute(data, context)
                
                # Update result
                self.result.steps_completed.append(step.name)
                
                # Check for intermediate results
                if f"{step.name}_result" in context:
                    self.result.intermediate_results[step.name] = context[f"{step.name}_result"]
            
            # Pipeline completed successfully
            self.result.status = PipelineStatus.COMPLETED
            self.result.end_time = datetime.utcnow()
            
            # Extract model artifact if available
            if "model_artifact" in context:
                self.result.model_artifact = context["model_artifact"]
            
            # Calculate total execution time
            self.result.total_compute_time_seconds = (
                self.result.end_time - self.result.start_time
            ).total_seconds()
            
            self.logger.info(f"Pipeline completed successfully in {self.result.total_compute_time_seconds:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.result.status = PipelineStatus.FAILED
            self.result.error_message = str(e)
            self.result.end_time = datetime.utcnow()
            
        finally:
            # Cleanup all steps
            for step in self.steps:
                try:
                    await step.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up step {step.name}: {e}")
        
        return self.result
    
    async def validate_config(self) -> bool:
        """Validate pipeline configuration"""
        required_fields = ["name", "version", "model_type", "data_source", "target_column"]
        
        for field in required_fields:
            if not hasattr(self.config, field) or not getattr(self.config, field):
                self.logger.error(f"Missing required configuration field: {field}")
                return False
        
        return True
    
    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline"""
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")
    
    def remove_step(self, step_name: str):
        """Remove a step from the pipeline"""
        self.steps = [s for s in self.steps if s.name != step_name]
        self.logger.info(f"Removed step: {step_name}")
    
    def get_step(self, step_name: str) -> Optional[PipelineStep]:
        """Get a specific step by name"""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None