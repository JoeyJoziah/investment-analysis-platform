"""
Feature Store Implementation
Provides centralized feature management, versioning, and quality monitoring
"""

import os
import json
import hashlib
# SECURITY: Removed pickle import - using JSON/joblib to prevent code execution
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


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


class FeatureValidator:
    """Feature validation and quality scoring"""
    
    def __init__(self):
        self.validation_rules = {
            'numerical': self._validate_numerical,
            'categorical': self._validate_categorical, 
            'boolean': self._validate_boolean,
            'datetime': self._validate_datetime
        }
    
    def validate_feature(self, feature_def: FeatureDefinition, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate feature values and return quality scores"""
        errors = []
        quality_scores = pd.Series(1.0, index=values.index)
        
        # Type validation
        if feature_def.feature_type in self.validation_rules:
            type_valid, type_errors = self.validation_rules[feature_def.feature_type](values)
            errors.extend(type_errors)
            quality_scores = quality_scores * type_valid.astype(float)
        
        # Custom validation rules
        for rule_name, rule_config in feature_def.validation_rules.items():
            rule_valid, rule_errors = self._apply_validation_rule(rule_name, rule_config, values)
            errors.extend(rule_errors)
            quality_scores = quality_scores * rule_valid.astype(float)
        
        return quality_scores, errors
    
    def _validate_numerical(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate numerical features"""
        errors = []
        valid = pd.Series(True, index=values.index)
        
        # Check for non-numeric values
        non_numeric = ~pd.to_numeric(values, errors='coerce').notna()
        if non_numeric.any():
            errors.append(f"Found {non_numeric.sum()} non-numeric values")
            valid = valid & ~non_numeric
        
        # Check for infinite values
        numeric_values = pd.to_numeric(values, errors='coerce')
        infinite_mask = np.isinf(numeric_values)
        if infinite_mask.any():
            errors.append(f"Found {infinite_mask.sum()} infinite values")
            valid = valid & ~infinite_mask
        
        return valid, errors
    
    def _validate_categorical(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate categorical features"""
        errors = []
        valid = pd.Series(True, index=values.index)
        
        # Check for null values
        null_mask = values.isna()
        if null_mask.any():
            errors.append(f"Found {null_mask.sum()} null values")
            valid = valid & ~null_mask
        
        return valid, errors
    
    def _validate_boolean(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate boolean features"""
        errors = []
        valid = pd.Series(True, index=values.index)
        
        # Check for valid boolean values
        boolean_values = values.isin([True, False, 0, 1, 'true', 'false', 'True', 'False'])
        if not boolean_values.all():
            errors.append(f"Found {(~boolean_values).sum()} invalid boolean values")
            valid = valid & boolean_values
        
        return valid, errors
    
    def _validate_datetime(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate datetime features"""
        errors = []
        valid = pd.Series(True, index=values.index)
        
        try:
            pd.to_datetime(values)
        except:
            errors.append("Invalid datetime format detected")
            valid = pd.Series(False, index=values.index)
        
        return valid, errors
    
    def _apply_validation_rule(self, rule_name: str, rule_config: Dict[str, Any], 
                             values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Apply custom validation rule"""
        errors = []
        valid = pd.Series(True, index=values.index)
        
        if rule_name == "range":
            min_val = rule_config.get('min')
            max_val = rule_config.get('max')
            
            if min_val is not None:
                below_min = values < min_val
                if below_min.any():
                    errors.append(f"Found {below_min.sum()} values below minimum {min_val}")
                    valid = valid & ~below_min
            
            if max_val is not None:
                above_max = values > max_val
                if above_max.any():
                    errors.append(f"Found {above_max.sum()} values above maximum {max_val}")
                    valid = valid & ~above_max
        
        elif rule_name == "allowed_values":
            allowed = rule_config.get('values', [])
            not_allowed = ~values.isin(allowed)
            if not_allowed.any():
                errors.append(f"Found {not_allowed.sum()} values not in allowed set")
                valid = valid & ~not_allowed
        
        elif rule_name == "not_null":
            if rule_config.get('required', True):
                null_mask = values.isna()
                if null_mask.any():
                    errors.append(f"Found {null_mask.sum()} null values")
                    valid = valid & ~null_mask
        
        return valid, errors


class FeatureDriftDetector:
    """Feature drift detection and monitoring"""
    
    def __init__(self, reference_window_days: int = 30, detection_window_days: int = 7):
        self.reference_window_days = reference_window_days
        self.detection_window_days = detection_window_days
    
    def detect_drift(self, feature_name: str, 
                    reference_data: pd.Series, 
                    current_data: pd.Series) -> FeatureDriftMetrics:
        """Detect drift between reference and current data"""
        
        # Population Stability Index (PSI)
        psi = self._calculate_psi(reference_data, current_data)
        
        # Kolmogorov-Smirnov test
        ks_stat = self._calculate_ks_statistic(reference_data, current_data)
        
        # Jensen-Shannon distance
        js_distance = self._calculate_js_distance(reference_data, current_data)
        
        # Mean and std shift
        mean_shift = abs(current_data.mean() - reference_data.mean()) / reference_data.std() if reference_data.std() > 0 else 0
        std_shift = abs(current_data.std() - reference_data.std()) / reference_data.std() if reference_data.std() > 0 else 0
        
        # Overall drift score (weighted combination)
        drift_score = 0.4 * psi + 0.3 * ks_stat + 0.2 * js_distance + 0.1 * (mean_shift + std_shift)
        
        # Drift detection thresholds
        drift_threshold = 0.25
        distribution_shift_detected = drift_score > drift_threshold
        
        return FeatureDriftMetrics(
            feature_name=feature_name,
            timestamp=datetime.utcnow(),
            population_stability_index=psi,
            kolmogorov_smirnov_statistic=ks_stat,
            jensen_shannon_distance=js_distance,
            mean_shift=mean_shift,
            std_shift=std_shift,
            distribution_shift_detected=distribution_shift_detected,
            drift_score=drift_score
        )
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Handle edge cases
            if len(reference) == 0 or len(current) == 0:
                return 1.0
            
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference.dropna(), bins=bins)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate frequencies
            ref_freq = pd.cut(reference, bins=bin_edges).value_counts().values
            cur_freq = pd.cut(current, bins=bin_edges).value_counts().values
            
            # Convert to percentages
            ref_pct = ref_freq / ref_freq.sum()
            cur_pct = cur_freq / cur_freq.sum()
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
            
            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            return min(psi, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def _calculate_ks_statistic(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        try:
            from scipy import stats
            
            ref_clean = reference.dropna()
            cur_clean = current.dropna()
            
            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return 0.0
            
            ks_stat, _ = stats.ks_2samp(ref_clean, cur_clean)
            return ks_stat
            
        except Exception as e:
            logger.error(f"Error calculating KS statistic: {e}")
            return 0.0
    
    def _calculate_js_distance(self, reference: pd.Series, current: pd.Series, bins: int = 50) -> float:
        """Calculate Jensen-Shannon distance"""
        try:
            # Create histograms
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            ref_hist, _ = np.histogram(reference.dropna(), bins=bins, range=(min_val, max_val), density=True)
            cur_hist, _ = np.histogram(current.dropna(), bins=bins, range=(min_val, max_val), density=True)
            
            # Normalize to probabilities
            ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_prob = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
            
            # Calculate JS distance
            m = 0.5 * (ref_prob + cur_prob)
            
            def kl_divergence(p, q):
                return np.sum(p * np.log(p / q + 1e-10))
            
            js_distance = 0.5 * kl_divergence(ref_prob, m) + 0.5 * kl_divergence(cur_prob, m)
            return min(js_distance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating JS distance: {e}")
            return 0.0


class FeatureStore:
    """
    Centralized feature store with versioning and monitoring
    """
    
    def __init__(self, 
                 storage_path: str = "/app/feature_store",
                 redis_url: str = "redis://localhost:6379",
                 db_url: str = None,
                 enable_caching: bool = True):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = FeatureValidator()
        self.drift_detector = FeatureDriftDetector()
        
        # Feature registry
        self.registry_path = self.storage_path / "registry.json"
        self.feature_registry: Dict[str, FeatureDefinition] = self._load_registry()
        
        # Caching
        self.enable_caching = enable_caching
        self.cache = None
        if enable_caching:
            try:
                import redis
                self.cache = redis.from_url(redis_url)
                self.cache.ping()  # Test connection
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
                self.cache = None
        
        # Database for metadata (optional)
        self.db_engine = None
        if db_url:
            try:
                self.db_engine = create_engine(db_url)
                logger.info("Database connection initialized")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Feature computation cache
        self.computation_cache = {}
        
        logger.info(f"Feature store initialized with {len(self.feature_registry)} registered features")
    
    def _load_registry(self) -> Dict[str, FeatureDefinition]:
        """Load feature registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                registry = {}
                for name, feature_data in data.items():
                    feature_data['feature_type'] = FeatureType(feature_data['feature_type'])
                    feature_data['compute_mode'] = ComputeMode(feature_data['compute_mode'])
                    feature_data['status'] = FeatureStatus(feature_data['status'])
                    feature_data['created_at'] = datetime.fromisoformat(feature_data['created_at'])
                    feature_data['updated_at'] = datetime.fromisoformat(feature_data['updated_at'])
                    
                    registry[name] = FeatureDefinition(**feature_data)
                
                return registry
                
            except Exception as e:
                logger.error(f"Error loading feature registry: {e}")
                return {}
        
        return {}
    
    def _save_registry(self):
        """Save feature registry to disk"""
        try:
            data = {name: feature.to_dict() for name, feature in self.feature_registry.items()}
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving feature registry: {e}")
    
    def register_feature(self,
                        name: str,
                        description: str,
                        feature_type: FeatureType,
                        compute_mode: ComputeMode,
                        computation_logic: str,
                        dependencies: List[str] = None,
                        validation_rules: Dict[str, Any] = None,
                        tags: List[str] = None,
                        created_by: str = "system",
                        business_context: str = "",
                        sla_hours: float = None) -> bool:
        """Register a new feature"""
        
        with self.lock:
            try:
                if name in self.feature_registry:
                    # Update existing feature (create new version)
                    existing = self.feature_registry[name]
                    version_parts = existing.version.split('.')
                    new_patch = int(version_parts[2]) + 1
                    new_version = f"{version_parts[0]}.{version_parts[1]}.{new_patch}"
                else:
                    new_version = "1.0.0"
                
                feature_def = FeatureDefinition(
                    name=name,
                    description=description,
                    feature_type=feature_type,
                    compute_mode=compute_mode,
                    status=FeatureStatus.DEVELOPMENT,
                    version=new_version,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    created_by=created_by,
                    dependencies=dependencies or [],
                    source_tables=[],
                    computation_logic=computation_logic,
                    validation_rules=validation_rules or {},
                    tags=tags or [],
                    business_context=business_context,
                    sla_hours=sla_hours,
                    monitoring_config={}
                )
                
                self.feature_registry[name] = feature_def
                self._save_registry()
                
                logger.info(f"Registered feature {name} version {new_version}")
                return True
                
            except Exception as e:
                logger.error(f"Error registering feature {name}: {e}")
                return False
    
    def compute_features(self,
                        feature_names: List[str],
                        entity_ids: List[str],
                        timestamp: datetime = None,
                        data_sources: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute features for given entities
        
        Args:
            feature_names: List of feature names to compute
            entity_ids: List of entity IDs (e.g., ticker symbols)
            timestamp: Point in time for feature computation
            data_sources: External data sources for computation
            
        Returns:
            DataFrame with features as columns and entity_ids as index
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        logger.info(f"Computing {len(feature_names)} features for {len(entity_ids)} entities")
        
        # Check cache first
        cache_key = self._generate_cache_key(feature_names, entity_ids, timestamp)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info("Retrieved features from cache")
            return cached_result
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(index=entity_ids)
        
        # Sort features by dependencies
        sorted_features = self._sort_features_by_dependencies(feature_names)
        
        # Compute features
        for feature_name in sorted_features:
            try:
                if feature_name not in self.feature_registry:
                    logger.warning(f"Feature {feature_name} not registered")
                    continue
                
                feature_def = self.feature_registry[feature_name]
                
                # Compute feature values
                feature_values = self._compute_single_feature(
                    feature_def, entity_ids, timestamp, data_sources, result_df
                )
                
                # Validate feature values
                quality_scores, validation_errors = self.validator.validate_feature(
                    feature_def, feature_values
                )
                
                if validation_errors:
                    logger.warning(f"Validation errors for feature {feature_name}: {validation_errors}")
                
                # Store feature values
                result_df[feature_name] = feature_values
                result_df[f"{feature_name}_quality"] = quality_scores
                
            except Exception as e:
                logger.error(f"Error computing feature {feature_name}: {e}")
                # Set null values for failed feature
                result_df[feature_name] = np.nan
                result_df[f"{feature_name}_quality"] = 0.0
        
        # Cache result
        self._save_to_cache(cache_key, result_df)
        
        logger.info(f"Successfully computed {len(result_df.columns)} features")
        return result_df
    
    def _sort_features_by_dependencies(self, feature_names: List[str]) -> List[str]:
        """Sort features by their dependencies using topological sort"""
        
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        for name in feature_names:
            if name in self.feature_registry:
                dependencies = self.feature_registry[name].dependencies
                graph[name] = [dep for dep in dependencies if dep in feature_names]
                in_degree[name] = 0
        
        # Calculate in-degrees
        for name in feature_names:
            for dep in graph.get(name, []):
                in_degree[dep] = in_degree.get(dep, 0) + 1
        
        # Topological sort
        queue = [name for name in feature_names if in_degree.get(name, 0) == 0]
        sorted_features = []
        
        while queue:
            current = queue.pop(0)
            sorted_features.append(current)
            
            for neighbor in graph.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return sorted_features
    
    def _compute_single_feature(self,
                               feature_def: FeatureDefinition,
                               entity_ids: List[str],
                               timestamp: datetime,
                               data_sources: Dict[str, pd.DataFrame],
                               computed_features: pd.DataFrame) -> pd.Series:
        """Compute a single feature for all entities"""
        
        # Check if feature is already computed
        if feature_def.name in computed_features.columns:
            return computed_features[feature_def.name]
        
        # Execute computation logic
        computation_func = self._get_computation_function(feature_def)
        
        if computation_func is None:
            # Fallback: try to execute as Python code
            return self._execute_python_computation(
                feature_def, entity_ids, timestamp, data_sources, computed_features
            )
        
        return computation_func(entity_ids, timestamp, data_sources, computed_features)
    
    def _get_computation_function(self, feature_def: FeatureDefinition) -> Optional[Callable]:
        """Get computation function for a feature"""
        
        # Check if we have a pre-registered computation function
        if feature_def.name in self.computation_cache:
            return self.computation_cache[feature_def.name]
        
        # Built-in feature computations
        builtin_features = {
            'price_return_1d': self._compute_price_return_1d,
            'price_return_5d': self._compute_price_return_5d,
            'price_volatility_20d': self._compute_price_volatility_20d,
            'volume_ratio_20d': self._compute_volume_ratio_20d,
            'rsi_14d': self._compute_rsi_14d,
            'sma_20d': self._compute_sma_20d,
            'ema_20d': self._compute_ema_20d,
            'pe_ratio': self._compute_pe_ratio,
            'market_cap': self._compute_market_cap
        }
        
        return builtin_features.get(feature_def.name)
    
    def _execute_python_computation(self,
                                   feature_def: FeatureDefinition,
                                   entity_ids: List[str],
                                   timestamp: datetime,
                                   data_sources: Dict[str, pd.DataFrame],
                                   computed_features: pd.DataFrame) -> pd.Series:
        """
        Handle unregistered feature computation requests.

        SECURITY NOTE: Previously used exec() to run arbitrary code from feature_def.computation_logic.
        This was a critical security vulnerability (arbitrary code execution).
        Now only pre-registered computation functions are allowed.

        To add a new feature computation:
        1. Add a method like _compute_<feature_name>() to this class
        2. Register it in _get_computation_function() builtin_features dict
        3. Or use register_computation() to register a callable at runtime
        """
        # SECURITY: Do NOT use exec() - it allows arbitrary code execution
        # Instead, require all computations to be pre-registered functions

        logger.error(
            f"SECURITY: Refusing to execute unregistered computation for feature '{feature_def.name}'. "
            f"Computation logic must be registered as a named function. "
            f"Use register_computation('{feature_def.name}', callable) or add to builtin_features."
        )

        # Return NaN values instead of executing arbitrary code
        return pd.Series(np.nan, index=entity_ids)

    def register_computation(self, feature_name: str, computation_func: Callable) -> None:
        """
        Register a safe computation function for a feature.

        This is the secure way to add custom feature computations.

        Args:
            feature_name: Name of the feature
            computation_func: Callable that takes (entity_ids, timestamp, data_sources, computed_features)
                            and returns a pd.Series
        """
        if not callable(computation_func):
            raise ValueError(f"computation_func must be callable, got {type(computation_func)}")

        self.computation_cache[feature_name] = computation_func
        logger.info(f"Registered computation function for feature: {feature_name}")
    
    # Built-in feature computation functions
    def _compute_price_return_1d(self, entity_ids: List[str], timestamp: datetime, 
                               data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute 1-day price return"""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        returns = []
        
        for entity_id in entity_ids:
            try:
                entity_prices = price_data[price_data['ticker'] == entity_id].sort_values('date')
                if len(entity_prices) >= 2:
                    latest_price = entity_prices['close'].iloc[-1]
                    prev_price = entity_prices['close'].iloc[-2]
                    return_1d = (latest_price - prev_price) / prev_price
                    returns.append(return_1d)
                else:
                    returns.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing 1d return for {entity_id}: {e}")
                returns.append(np.nan)
        
        return pd.Series(returns, index=entity_ids)
    
    def _compute_price_return_5d(self, entity_ids: List[str], timestamp: datetime,
                               data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute 5-day price return"""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        returns = []
        
        for entity_id in entity_ids:
            try:
                entity_prices = price_data[price_data['ticker'] == entity_id].sort_values('date')
                if len(entity_prices) >= 6:
                    latest_price = entity_prices['close'].iloc[-1]
                    price_5d_ago = entity_prices['close'].iloc[-6]
                    return_5d = (latest_price - price_5d_ago) / price_5d_ago
                    returns.append(return_5d)
                else:
                    returns.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing 5d return for {entity_id}: {e}")
                returns.append(np.nan)
        
        return pd.Series(returns, index=entity_ids)
    
    def _compute_price_volatility_20d(self, entity_ids: List[str], timestamp: datetime,
                                    data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute 20-day price volatility"""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        volatilities = []
        
        for entity_id in entity_ids:
            try:
                entity_prices = price_data[price_data['ticker'] == entity_id].sort_values('date')
                if len(entity_prices) >= 21:
                    prices = entity_prices['close'].tail(21)
                    returns = prices.pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    volatilities.append(volatility)
                else:
                    volatilities.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing volatility for {entity_id}: {e}")
                volatilities.append(np.nan)
        
        return pd.Series(volatilities, index=entity_ids)
    
    def _compute_volume_ratio_20d(self, entity_ids: List[str], timestamp: datetime,
                                data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute volume ratio vs 20-day average"""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        ratios = []
        
        for entity_id in entity_ids:
            try:
                entity_data = price_data[price_data['ticker'] == entity_id].sort_values('date')
                if len(entity_data) >= 21:
                    current_volume = entity_data['volume'].iloc[-1]
                    avg_volume_20d = entity_data['volume'].tail(20).mean()
                    ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else np.nan
                    ratios.append(ratio)
                else:
                    ratios.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing volume ratio for {entity_id}: {e}")
                ratios.append(np.nan)
        
        return pd.Series(ratios, index=entity_ids)
    
    def _compute_rsi_14d(self, entity_ids: List[str], timestamp: datetime,
                       data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute 14-day RSI"""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        rsi_values = []
        
        for entity_id in entity_ids:
            try:
                entity_prices = price_data[price_data['ticker'] == entity_id].sort_values('date')
                if len(entity_prices) >= 15:
                    prices = entity_prices['close'].tail(15)
                    delta = prices.diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_values.append(rsi.iloc[-1])
                else:
                    rsi_values.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing RSI for {entity_id}: {e}")
                rsi_values.append(np.nan)
        
        return pd.Series(rsi_values, index=entity_ids)
    
    def _compute_sma_20d(self, entity_ids: List[str], timestamp: datetime,
                       data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute 20-day Simple Moving Average"""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        sma_values = []
        
        for entity_id in entity_ids:
            try:
                entity_prices = price_data[price_data['ticker'] == entity_id].sort_values('date')
                if len(entity_prices) >= 20:
                    sma = entity_prices['close'].tail(20).mean()
                    sma_values.append(sma)
                else:
                    sma_values.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing SMA for {entity_id}: {e}")
                sma_values.append(np.nan)
        
        return pd.Series(sma_values, index=entity_ids)
    
    def _compute_ema_20d(self, entity_ids: List[str], timestamp: datetime,
                       data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute 20-day Exponential Moving Average"""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        ema_values = []
        
        for entity_id in entity_ids:
            try:
                entity_prices = price_data[price_data['ticker'] == entity_id].sort_values('date')
                if len(entity_prices) >= 20:
                    ema = entity_prices['close'].tail(40).ewm(span=20).mean().iloc[-1]
                    ema_values.append(ema)
                else:
                    ema_values.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing EMA for {entity_id}: {e}")
                ema_values.append(np.nan)
        
        return pd.Series(ema_values, index=entity_ids)
    
    def _compute_pe_ratio(self, entity_ids: List[str], timestamp: datetime,
                        data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute P/E ratio"""
        if 'price_data' not in data_sources or 'fundamental_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        fundamental_data = data_sources['fundamental_data']
        pe_ratios = []
        
        for entity_id in entity_ids:
            try:
                # Get latest price
                entity_prices = price_data[price_data['ticker'] == entity_id]
                if len(entity_prices) == 0:
                    pe_ratios.append(np.nan)
                    continue
                
                latest_price = entity_prices['close'].iloc[-1]
                
                # Get EPS
                entity_fundamentals = fundamental_data[fundamental_data['ticker'] == entity_id]
                if len(entity_fundamentals) == 0:
                    pe_ratios.append(np.nan)
                    continue
                
                eps = entity_fundamentals['eps'].iloc[-1]
                pe_ratio = latest_price / eps if eps > 0 else np.nan
                pe_ratios.append(pe_ratio)
                
            except Exception as e:
                logger.error(f"Error computing P/E ratio for {entity_id}: {e}")
                pe_ratios.append(np.nan)
        
        return pd.Series(pe_ratios, index=entity_ids)
    
    def _compute_market_cap(self, entity_ids: List[str], timestamp: datetime,
                          data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """Compute market capitalization"""
        if 'price_data' not in data_sources or 'fundamental_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)
        
        price_data = data_sources['price_data']
        fundamental_data = data_sources['fundamental_data']
        market_caps = []
        
        for entity_id in entity_ids:
            try:
                # Get latest price
                entity_prices = price_data[price_data['ticker'] == entity_id]
                if len(entity_prices) == 0:
                    market_caps.append(np.nan)
                    continue
                
                latest_price = entity_prices['close'].iloc[-1]
                
                # Get shares outstanding
                entity_fundamentals = fundamental_data[fundamental_data['ticker'] == entity_id]
                if len(entity_fundamentals) == 0:
                    market_caps.append(np.nan)
                    continue
                
                shares_outstanding = entity_fundamentals['shares_outstanding'].iloc[-1]
                market_cap = latest_price * shares_outstanding
                market_caps.append(market_cap)
                
            except Exception as e:
                logger.error(f"Error computing market cap for {entity_id}: {e}")
                market_caps.append(np.nan)
        
        return pd.Series(market_caps, index=entity_ids)
    
    def monitor_feature_drift(self,
                            feature_name: str,
                            reference_period_days: int = 30,
                            current_period_days: int = 7) -> Optional[FeatureDriftMetrics]:
        """Monitor feature drift over time"""
        
        if feature_name not in self.feature_registry:
            logger.error(f"Feature {feature_name} not registered")
            return None
        
        try:
            # Get historical feature values
            end_date = datetime.utcnow()
            reference_start = end_date - timedelta(days=reference_period_days + current_period_days)
            reference_end = end_date - timedelta(days=current_period_days)
            current_start = reference_end
            
            # Mock data for demonstration - in real implementation, query feature store
            reference_data = pd.Series(np.random.normal(0, 1, 1000))
            current_data = pd.Series(np.random.normal(0.2, 1.2, 200))  # Simulated drift
            
            drift_metrics = self.drift_detector.detect_drift(
                feature_name, reference_data, current_data
            )
            
            # Store drift metrics
            self._save_drift_metrics(drift_metrics)
            
            # Alert if significant drift detected
            if drift_metrics.distribution_shift_detected:
                logger.warning(f"Feature drift detected for {feature_name}: "
                             f"drift_score={drift_metrics.drift_score:.3f}")
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring drift for feature {feature_name}: {e}")
            return None
    
    def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get feature lineage and dependencies"""
        
        if feature_name not in self.feature_registry:
            return {}
        
        feature_def = self.feature_registry[feature_name]
        
        # Build dependency tree
        lineage = {
            'feature': feature_name,
            'version': feature_def.version,
            'direct_dependencies': feature_def.dependencies,
            'upstream_dependencies': [],
            'downstream_dependencies': []
        }
        
        # Find upstream dependencies (recursive)
        def get_upstream(fname, visited=None):
            if visited is None:
                visited = set()
            
            if fname in visited:
                return []
            
            visited.add(fname)
            upstream = []
            
            if fname in self.feature_registry:
                for dep in self.feature_registry[fname].dependencies:
                    upstream.append(dep)
                    upstream.extend(get_upstream(dep, visited))
            
            return list(set(upstream))
        
        lineage['upstream_dependencies'] = get_upstream(feature_name)
        
        # Find downstream dependencies
        for fname, fdef in self.feature_registry.items():
            if feature_name in fdef.dependencies:
                lineage['downstream_dependencies'].append(fname)
        
        return lineage
    
    def get_feature_statistics(self, feature_names: List[str], 
                             days_back: int = 30) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive feature statistics"""
        
        stats = {}
        
        for feature_name in feature_names:
            if feature_name not in self.feature_registry:
                continue
            
            try:
                # Mock statistics - in real implementation, query feature store
                feature_stats = {
                    'count': np.random.randint(1000, 10000),
                    'mean': np.random.normal(0, 1),
                    'std': np.random.uniform(0.5, 2.0),
                    'min': np.random.normal(-3, 0.5),
                    'max': np.random.normal(3, 0.5),
                    'null_percentage': np.random.uniform(0, 0.05),
                    'unique_values': np.random.randint(100, 1000),
                    'quality_score': np.random.uniform(0.8, 1.0),
                    'last_updated': datetime.utcnow().isoformat(),
                    'freshness_hours': np.random.uniform(0, 24)
                }
                
                stats[feature_name] = feature_stats
                
            except Exception as e:
                logger.error(f"Error getting statistics for feature {feature_name}: {e}")
        
        return stats
    
    def _generate_cache_key(self, feature_names: List[str], entity_ids: List[str], timestamp: datetime) -> str:
        """Generate cache key for feature computation"""
        key_components = [
            ','.join(sorted(feature_names)),
            ','.join(sorted(entity_ids)),
            timestamp.strftime('%Y-%m-%d_%H')  # Cache by hour
        ]
        return hashlib.md5('|'.join(key_components).encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached result"""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(f"features:{cache_key}")
            if cached_data:
                return pd.read_json(cached_data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame, ttl_hours: int = 24):
        """Save result to cache"""
        if not self.cache:
            return
        
        try:
            json_data = data.to_json()
            self.cache.setex(f"features:{cache_key}", ttl_hours * 3600, json_data)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _save_drift_metrics(self, drift_metrics: FeatureDriftMetrics):
        """Save drift metrics to storage"""
        try:
            drift_file = self.storage_path / f"drift_{drift_metrics.feature_name}_{datetime.utcnow().strftime('%Y%m%d')}.json"
            
            with open(drift_file, 'w') as f:
                json.dump(asdict(drift_metrics), f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving drift metrics: {e}")
    
    def cleanup_old_features(self, days_to_keep: int = 90) -> int:
        """Clean up old feature data"""
        cleanup_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        try:
            # Clean up drift metrics files
            drift_files = list(self.storage_path.glob("drift_*.json"))
            
            for drift_file in drift_files:
                if drift_file.stat().st_mtime < cleanup_date.timestamp():
                    drift_file.unlink()
                    cleaned_count += 1
            
            # Clean up cache
            if self.cache:
                # This would need implementation based on Redis pattern matching
                pass
            
            logger.info(f"Cleaned up {cleaned_count} old feature files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0


# Global feature store instance
_feature_store: Optional[FeatureStore] = None

def get_feature_store() -> FeatureStore:
    """Get global feature store instance"""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store