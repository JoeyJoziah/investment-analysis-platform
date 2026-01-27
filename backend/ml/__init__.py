"""
ML Operations Package
Comprehensive ML operations system with versioning, backtesting, monitoring, and cost control
"""

# Original model manager
from .model_manager import get_model_manager, ModelManager

# New ML Operations components
from .model_versioning import (
    ModelVersionManager, 
    ModelVersion, 
    ModelStage, 
    ModelType,
    ABTestConfig,
    get_model_version_manager
)

from .backtesting import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestMetric,
    WalkForwardValidator,
    get_backtest_engine
)

from .feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureType,
    ComputeMode,
    FeatureStatus,
    FeatureDriftMetrics,
    FeatureValidator,
    FeatureDriftDetector,
    get_feature_store
)

from .model_monitoring import (
    ModelMonitor,
    ModelPerformanceTracker,
    DriftDetector,
    AlertManager,
    PerformanceMetrics,
    DriftDetectionResult,
    ModelAlert,
    ModelHealth,
    get_model_monitor
)

# Temporarily disabled due to dask/lightgbm compatibility issues
# from .online_learning import (
#     OnlineLearningManager,
#     IncrementalLearner,
#     AdaptiveEnsembleWeighter,
#     LearningMetrics,
#     EnsembleWeights,
#     get_online_learning_manager
# )

from .pipeline_optimization import (
    MLPipelineOptimizer,
    ModelArtifactManager,
    InferenceCache,
    LoadBalancer,
    InferenceMetrics,
    get_pipeline_optimizer
)

from .cost_monitoring import (
    MLCostTracker,
    MLCostOptimizer,
    ResourceType,
    CostCategory,
    ResourceUsage,
    CostAlert,
    OptimizationRecommendation,
    get_ml_cost_tracker,
    get_ml_cost_optimizer,
    track_ml_cost
)

# GPU Utilities for accelerated training
from .gpu_utils import (
    GPUConfig,
    get_gpu_config,
    get_cached_gpu_config,
    log_gpu_memory_usage,
    clear_gpu_memory,
    set_gpu_memory_fraction
)

__version__ = "1.0.0"

__all__ = [
    # Original model manager
    "get_model_manager",
    "ModelManager",
    
    # Model Versioning
    "ModelVersionManager",
    "ModelVersion", 
    "ModelStage",
    "ModelType",
    "ABTestConfig",
    "get_model_version_manager",
    
    # Backtesting
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "BacktestMetric",
    "WalkForwardValidator",
    "get_backtest_engine",
    
    # Feature Store
    "FeatureStore",
    "FeatureDefinition",
    "FeatureType",
    "ComputeMode", 
    "FeatureStatus",
    "FeatureDriftMetrics",
    "FeatureValidator",
    "FeatureDriftDetector",
    "get_feature_store",
    
    # Model Monitoring
    "ModelMonitor",
    "ModelPerformanceTracker",
    "DriftDetector",
    "AlertManager",
    "PerformanceMetrics",
    "DriftDetectionResult",
    "ModelAlert",
    "ModelHealth",
    "get_model_monitor",
    
    # Online Learning - Temporarily disabled
    # "OnlineLearningManager",
    # "IncrementalLearner", 
    # "AdaptiveEnsembleWeighter",
    # "LearningMetrics",
    # "EnsembleWeights",
    # "get_online_learning_manager",
    
    # Pipeline Optimization
    "MLPipelineOptimizer",
    "ModelArtifactManager",
    "InferenceCache",
    "LoadBalancer",
    "InferenceMetrics",
    "get_pipeline_optimizer",
    
    # Cost Monitoring
    "MLCostTracker",
    "MLCostOptimizer",
    "ResourceType",
    "CostCategory",
    "ResourceUsage",
    "CostAlert",
    "OptimizationRecommendation",
    "get_ml_cost_tracker",
    "get_ml_cost_optimizer",
    "track_ml_cost",

    # GPU Utilities
    "GPUConfig",
    "get_gpu_config",
    "get_cached_gpu_config",
    "log_gpu_memory_usage",
    "clear_gpu_memory",
    "set_gpu_memory_fraction"
]