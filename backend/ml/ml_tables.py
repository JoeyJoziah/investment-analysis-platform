"""
ML Operations Database Tables
Additional tables for comprehensive ML operations tracking
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean, 
    ForeignKey, Text, JSON, Numeric, Index, UniqueConstraint,
    CheckConstraint, Enum as SQLEnum, DECIMAL, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
import enum

# Import base from main tables module
from ..models.tables import Base, ModelTypeEnum, ModelStageEnum, FeatureTypeEnum, ComputeModeEnum, FeatureStatusEnum, DriftTypeEnum, AlertSeverityEnum, ModelHealthEnum


# ===== ML Operations Tables =====

class MLModel(Base):
    """ML Model registry and versioning"""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    model_type = Column(SQLEnum(ModelTypeEnum), nullable=False)
    stage = Column(SQLEnum(ModelStageEnum), default=ModelStageEnum.DEVELOPMENT, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(100), nullable=False)
    description = Column(Text)
    tags = Column(JSON, default=list)
    parameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    model_size_bytes = Column(BigInteger)
    training_data_hash = Column(String(64))
    feature_names = Column(JSON, default=list)
    model_path = Column(Text, nullable=False)
    metadata_path = Column(Text)
    performance_benchmark = Column(JSON, default=dict)
    dependencies = Column(JSON, default=dict)
    git_commit = Column(String(40))
    parent_version = Column(String(20))
    is_champion = Column(Boolean, default=False)
    
    # Relationships
    performance_metrics = relationship("ModelPerformanceMetric", back_populates="model", cascade="all, delete-orphan")
    drift_detections = relationship("ModelDriftDetection", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("ModelAlert", back_populates="model", cascade="all, delete-orphan")
    backtests = relationship("ModelBacktest", back_populates="model", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
        Index('idx_ml_model_name_stage', 'name', 'stage'),
        Index('idx_ml_model_type', 'model_type'),
        Index('idx_ml_model_created', 'created_at'),
        Index('idx_ml_model_champion', 'is_champion'),
    )


class FeatureDefinition(Base):
    """Feature store definition and metadata"""
    __tablename__ = "feature_definitions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)
    feature_type = Column(SQLEnum(FeatureTypeEnum), nullable=False)
    compute_mode = Column(SQLEnum(ComputeModeEnum), nullable=False)
    status = Column(SQLEnum(FeatureStatusEnum), default=FeatureStatusEnum.DEVELOPMENT, nullable=False)
    version = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(100), nullable=False)
    dependencies = Column(JSON, default=list)
    source_tables = Column(JSON, default=list)
    computation_logic = Column(Text)
    validation_rules = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    business_context = Column(Text)
    sla_hours = Column(Float)
    monitoring_config = Column(JSON, default=dict)
    
    # Relationships
    feature_values = relationship("FeatureValue", back_populates="feature", cascade="all, delete-orphan")
    drift_metrics = relationship("FeatureDriftMetric", back_populates="feature", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_feature_def_name_version', 'name', 'version'),
        Index('idx_feature_def_type_status', 'feature_type', 'status'),
        Index('idx_feature_def_compute_mode', 'compute_mode'),
        Index('idx_feature_def_created', 'created_at'),
    )


class FeatureValue(Base):
    """Individual feature values with metadata"""
    __tablename__ = "feature_values"
    
    id = Column(Integer, primary_key=True, index=True)
    feature_id = Column(Integer, ForeignKey("feature_definitions.id", ondelete="CASCADE"), nullable=False)
    entity_id = Column(String(50), nullable=False, index=True)  # e.g., ticker symbol
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Text)  # JSON-encoded value
    version = Column(String(20), nullable=False)
    quality_score = Column(Float, default=1.0)
    is_valid = Column(Boolean, default=True)
    validation_errors = Column(JSON, default=list)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    feature = relationship("FeatureDefinition", back_populates="feature_values")
    
    __table_args__ = (
        Index('idx_feature_value_entity_timestamp', 'entity_id', 'timestamp'),
        Index('idx_feature_value_feature_entity', 'feature_id', 'entity_id'),
        Index('idx_feature_value_timestamp', 'timestamp'),
        Index('idx_feature_value_quality', 'quality_score'),
    )


class FeatureDriftMetric(Base):
    """Feature drift detection metrics"""
    __tablename__ = "feature_drift_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    feature_id = Column(Integer, ForeignKey("feature_definitions.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    population_stability_index = Column(Float, nullable=False)
    kolmogorov_smirnov_statistic = Column(Float, nullable=False)
    jensen_shannon_distance = Column(Float, nullable=False)
    mean_shift = Column(Float, nullable=False)
    std_shift = Column(Float, nullable=False)
    distribution_shift_detected = Column(Boolean, nullable=False)
    drift_score = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    reference_period_start = Column(DateTime, nullable=False)
    reference_period_end = Column(DateTime, nullable=False)
    detection_period_start = Column(DateTime, nullable=False)
    detection_period_end = Column(DateTime, nullable=False)
    
    # Relationships
    feature = relationship("FeatureDefinition", back_populates="drift_metrics")
    
    __table_args__ = (
        Index('idx_feature_drift_feature_timestamp', 'feature_id', 'timestamp'),
        Index('idx_feature_drift_score', 'drift_score'),
        Index('idx_feature_drift_detected', 'distribution_shift_detected'),
    )


class ModelPerformanceMetric(Base):
    """Model performance metrics snapshots"""
    __tablename__ = "model_performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    mse = Column(Float)
    mae = Column(Float)
    r2_score = Column(Float)
    auc_roc = Column(Float)
    directional_accuracy = Column(Float)
    sharpe_ratio = Column(Float)
    sample_size = Column(Integer, nullable=False)
    prediction_latency_ms = Column(Float)
    memory_usage_mb = Column(Float)
    cpu_usage_percent = Column(Float)
    gpu_usage_percent = Column(Float)
    throughput_samples_per_sec = Column(Float)
    cache_hit_ratio = Column(Float)
    additional_metrics = Column(JSON, default=dict)
    
    # Relationships
    model = relationship("MLModel", back_populates="performance_metrics")
    
    __table_args__ = (
        Index('idx_model_perf_model_timestamp', 'model_id', 'timestamp'),
        Index('idx_model_perf_accuracy', 'accuracy'),
        Index('idx_model_perf_f1', 'f1_score'),
        Index('idx_model_perf_timestamp', 'timestamp'),
    )


class ModelDriftDetection(Base):
    """Model drift detection results"""
    __tablename__ = "model_drift_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    drift_type = Column(SQLEnum(DriftTypeEnum), nullable=False)
    drift_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_drift_detected = Column(Boolean, nullable=False)
    feature_drifts = Column(JSON, default=dict)
    statistical_test_results = Column(JSON, default=dict)
    confidence_level = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    reference_period = Column(String(100))
    detection_period = Column(String(100))
    
    # Relationships
    model = relationship("MLModel", back_populates="drift_detections")
    
    __table_args__ = (
        Index('idx_model_drift_model_timestamp', 'model_id', 'timestamp'),
        Index('idx_model_drift_type', 'drift_type'),
        Index('idx_model_drift_detected', 'is_drift_detected'),
        Index('idx_model_drift_score', 'drift_score'),
    )


class ModelAlert(Base):
    """Model monitoring alerts"""
    __tablename__ = "model_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    alert_type = Column(String(50), nullable=False)
    severity = Column(SQLEnum(AlertSeverityEnum), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, default=dict)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    resolved_by = Column(String(100))
    
    # Relationships
    model = relationship("MLModel", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_model_alert_model_timestamp', 'model_id', 'timestamp'),
        Index('idx_model_alert_type_severity', 'alert_type', 'severity'),
        Index('idx_model_alert_resolved', 'is_resolved'),
        Index('idx_model_alert_timestamp', 'timestamp'),
    )


class ModelBacktest(Base):
    """Model backtesting results"""
    __tablename__ = "model_backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    backtest_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(DECIMAL(15, 2), nullable=False)
    final_capital = Column(DECIMAL(15, 2), nullable=False)
    total_return = Column(Float, nullable=False)
    annualized_return = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    directional_accuracy = Column(Float)
    benchmark_return = Column(Float)
    alpha = Column(Float)
    beta = Column(Float)
    information_ratio = Column(Float)
    tracking_error = Column(Float)
    detailed_metrics = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    model = relationship("MLModel", back_populates="backtests")
    
    __table_args__ = (
        Index('idx_model_backtest_model_created', 'model_id', 'created_at'),
        Index('idx_model_backtest_returns', 'total_return'),
        Index('idx_model_backtest_sharpe', 'sharpe_ratio'),
        Index('idx_model_backtest_dates', 'start_date', 'end_date'),
    )


class OnlineLearningMetric(Base):
    """Online learning performance tracking"""
    __tablename__ = "online_learning_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    update_type = Column(String(50), nullable=False)  # incremental, ensemble_weighting, etc.
    samples_processed = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    performance_before = Column(Float, nullable=False)
    performance_after = Column(Float, nullable=False)
    improvement = Column(Float, nullable=False)
    computational_cost_ms = Column(Float, nullable=False)
    memory_usage_mb = Column(Float, nullable=False)
    convergence_score = Column(Float, nullable=False)
    stability_score = Column(Float, nullable=False)
    additional_meta_data = Column(JSON, default=dict)  # Renamed to avoid SQLAlchemy conflict
    
    __table_args__ = (
        Index('idx_online_learning_model_timestamp', 'model_name', 'timestamp'),
        Index('idx_online_learning_update_type', 'update_type'),
        Index('idx_online_learning_improvement', 'improvement'),
        Index('idx_online_learning_timestamp', 'timestamp'),
    )


class InferenceMetric(Base):
    """Model inference performance metrics"""
    __tablename__ = "inference_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    batch_size = Column(Integer, nullable=False)
    inference_time_ms = Column(Float, nullable=False)
    preprocessing_time_ms = Column(Float, nullable=False)
    postprocessing_time_ms = Column(Float, nullable=False)
    total_time_ms = Column(Float, nullable=False)
    memory_usage_mb = Column(Float, nullable=False)
    cpu_usage_percent = Column(Float, nullable=False)
    gpu_usage_percent = Column(Float)
    throughput_samples_per_sec = Column(Float, nullable=False)
    cache_hit_ratio = Column(Float, nullable=False)
    optimization_used = Column(String(50))
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text)
    
    __table_args__ = (
        Index('idx_inference_model_timestamp', 'model_name', 'timestamp'),
        Index('idx_inference_total_time', 'total_time_ms'),
        Index('idx_inference_throughput', 'throughput_samples_per_sec'),
        Index('idx_inference_cache_hit', 'cache_hit_ratio'),
        Index('idx_inference_timestamp', 'timestamp'),
    )


class EnsembleWeight(Base):
    """Ensemble model weights tracking"""
    __tablename__ = "ensemble_weights"
    
    id = Column(Integer, primary_key=True, index=True)
    ensemble_name = Column(String(100), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    weight = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    performance_history = Column(JSON, default=list)
    last_updated = Column(DateTime, default=func.now(), nullable=False)
    update_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_ensemble_weight_ensemble_model', 'ensemble_name', 'model_name'),
        Index('idx_ensemble_weight_updated', 'last_updated'),
        Index('idx_ensemble_weight_active', 'is_active'),
        UniqueConstraint('ensemble_name', 'model_name', name='uq_ensemble_model'),
    )


class ABTest(Base):
    """A/B testing for model versions"""
    __tablename__ = "ab_tests"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    champion_model = Column(String(100), nullable=False)
    challenger_model = Column(String(100), nullable=False)
    traffic_split = Column(Float, nullable=False)  # Percentage for challenger
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    success_metrics = Column(JSON, default=list)
    minimum_sample_size = Column(Integer, nullable=False)
    statistical_significance = Column(Float, default=0.95)
    status = Column(String(20), default="active")
    results = Column(JSON, default=dict)
    winner = Column(String(100))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    created_by = Column(String(100), nullable=False)
    
    __table_args__ = (
        Index('idx_ab_test_status_dates', 'status', 'start_date', 'end_date'),
        Index('idx_ab_test_models', 'champion_model', 'challenger_model'),
        Index('idx_ab_test_created', 'created_at'),
    )


class ModelArtifact(Base):
    """Model artifacts and optimized versions"""
    __tablename__ = "model_artifacts"
    
    id = Column(Integer, primary_key=True, index=True)
    artifact_id = Column(String(200), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    format = Column(String(50), nullable=False)  # pytorch, onnx, sklearn_joblib, etc.
    created_at = Column(DateTime, default=func.now(), nullable=False)
    original_path = Column(Text, nullable=False)
    original_size_mb = Column(Float, nullable=False)
    optimized_versions = Column(JSON, default=dict)  # optimization_type -> path mapping
    meta_data = Column(JSON, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    is_active = Column(Boolean, default=True)
    checksum = Column(String(64))
    
    __table_args__ = (
        Index('idx_model_artifact_model_version', 'model_name', 'model_version'),
        Index('idx_model_artifact_format', 'format'),
        Index('idx_model_artifact_created', 'created_at'),
        Index('idx_model_artifact_active', 'is_active'),
    )