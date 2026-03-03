"""
Feature Types
Enums and dataclasses for the feature store domain model.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class FeatureType(Enum):
    """Feature data types"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"


class ComputeMode(Enum):
    """Feature computation modes"""
    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"


class FeatureStatus(Enum):
    """Feature lifecycle status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass
class FeatureDefinition:
    """Feature definition and metadata"""
    name: str
    description: str
    feature_type: FeatureType
    compute_mode: ComputeMode
    status: FeatureStatus
    version: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    dependencies: List[str]  # Other features this depends on
    source_tables: List[str]
    computation_logic: str  # SQL or Python code
    validation_rules: Dict[str, Any]
    tags: List[str]
    business_context: str
    sla_hours: Optional[float] = None  # Max staleness allowed
    monitoring_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['feature_type'] = self.feature_type.value
        data['compute_mode'] = self.compute_mode.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class FeatureValue:
    """Individual feature value with metadata"""
    feature_name: str
    entity_id: str  # e.g., ticker symbol
    timestamp: datetime
    value: Any
    version: str
    quality_score: float = 1.0
    is_valid: bool = True
    validation_errors: List[str] = None


@dataclass
class FeatureDriftMetrics:
    """Feature drift detection metrics"""
    feature_name: str
    timestamp: datetime
    population_stability_index: float
    kolmogorov_smirnov_statistic: float
    jensen_shannon_distance: float
    mean_shift: float
    std_shift: float
    distribution_shift_detected: bool
    drift_score: float  # 0-1, higher = more drift
