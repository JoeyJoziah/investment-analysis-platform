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
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


# Import types from extracted sub-module
try:
    from backend.ml.feature_types import (
        FeatureType, ComputeMode, FeatureStatus,
        FeatureDefinition, FeatureValue, FeatureDriftMetrics,
    )
except ImportError:
    from .feature_types import (
        FeatureType, ComputeMode, FeatureStatus,
        FeatureDefinition, FeatureValue, FeatureDriftMetrics,
    )

# Import validator from extracted sub-module
try:
    from backend.ml.feature_validation import FeatureValidator as _FeatureValidator
except ImportError:
    from .feature_validation import FeatureValidator as _FeatureValidator

# Import drift detector from extracted sub-module
try:
    from backend.ml.feature_drift import FeatureDriftDetector as _FeatureDriftDetector
except ImportError:
    from .feature_drift import FeatureDriftDetector as _FeatureDriftDetector

# Re-export for backward compatibility
FeatureValidator = _FeatureValidator
FeatureDriftDetector = _FeatureDriftDetector


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
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
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
            timestamp = datetime.now(timezone.utc)
        
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
            # F-03-008: ML_PIPELINE_DOCUMENTATION.md advertised MACD and
            # Bollinger Bands as builtin features but the computations
            # were never implemented. Adds them so the documented
            # surface and the runtime surface agree.
            'macd': self._compute_macd,
            'macd_signal': self._compute_macd_signal,
            'bollinger_upper_20d': self._compute_bollinger_upper_20d,
            'bollinger_lower_20d': self._compute_bollinger_lower_20d,
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
    
    # F-03-008: MACD and Bollinger Bands implementations follow.
    # All four share a small helper that returns the per-entity close
    # series, sorted by date.

    def _entity_close_series(
        self,
        entity_id: str,
        price_data: pd.DataFrame,
    ) -> pd.Series:
        rows = price_data[price_data['ticker'] == entity_id].sort_values('date')
        return rows['close'].astype(float)

    def _compute_macd(self, entity_ids: List[str], timestamp: datetime,
                     data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """MACD = EMA(12) - EMA(26) on closing prices."""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)

        price_data = data_sources['price_data']
        values: List[float] = []
        for entity_id in entity_ids:
            try:
                closes = self._entity_close_series(entity_id, price_data)
                if len(closes) >= 26:
                    ema_fast = closes.ewm(span=12, adjust=False).mean().iloc[-1]
                    ema_slow = closes.ewm(span=26, adjust=False).mean().iloc[-1]
                    values.append(float(ema_fast - ema_slow))
                else:
                    values.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing MACD for {entity_id}: {e}")
                values.append(np.nan)
        return pd.Series(values, index=entity_ids)

    def _compute_macd_signal(self, entity_ids: List[str], timestamp: datetime,
                            data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """MACD signal line = EMA(9) of the MACD series."""
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)

        price_data = data_sources['price_data']
        values: List[float] = []
        for entity_id in entity_ids:
            try:
                closes = self._entity_close_series(entity_id, price_data)
                if len(closes) >= 26 + 9:
                    macd_series = (
                        closes.ewm(span=12, adjust=False).mean()
                        - closes.ewm(span=26, adjust=False).mean()
                    )
                    signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
                    values.append(float(signal))
                else:
                    values.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing MACD signal for {entity_id}: {e}")
                values.append(np.nan)
        return pd.Series(values, index=entity_ids)

    def _compute_bollinger_upper_20d(self, entity_ids: List[str], timestamp: datetime,
                                     data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """20-day Bollinger upper band = SMA(20) + 2 * std(20)."""
        return self._compute_bollinger_band(entity_ids, data_sources, k=2.0)

    def _compute_bollinger_lower_20d(self, entity_ids: List[str], timestamp: datetime,
                                     data_sources: Dict[str, pd.DataFrame], computed_features: pd.DataFrame) -> pd.Series:
        """20-day Bollinger lower band = SMA(20) - 2 * std(20)."""
        return self._compute_bollinger_band(entity_ids, data_sources, k=-2.0)

    def _compute_bollinger_band(
        self,
        entity_ids: List[str],
        data_sources: Dict[str, pd.DataFrame],
        k: float,
    ) -> pd.Series:
        if 'price_data' not in data_sources:
            return pd.Series(np.nan, index=entity_ids)

        price_data = data_sources['price_data']
        values: List[float] = []
        for entity_id in entity_ids:
            try:
                closes = self._entity_close_series(entity_id, price_data)
                if len(closes) >= 20:
                    window = closes.tail(20)
                    mean = float(window.mean())
                    std = float(window.std(ddof=0))
                    values.append(mean + k * std)
                else:
                    values.append(np.nan)
            except Exception as e:
                logger.error(f"Error computing Bollinger band for {entity_id}: {e}")
                values.append(np.nan)
        return pd.Series(values, index=entity_ids)

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
        """Monitor feature drift over time using real feature_values rows.

        Per PRD audit 2026-04 F-03-005 / Q4 default: this method previously
        fabricated reference + current windows from ``np.random.normal()``
        which caused drift alerts to fire on noise. We now query the
        ``feature_values`` table directly and raise ``InsufficientDataError``
        when no real data is available — callers surface that as HTTP 503
        ``model_unavailable``.
        """
        from backend.exceptions import InsufficientDataError

        if feature_name not in self.feature_registry:
            logger.error(f"Feature {feature_name} not registered")
            return None

        end_date = datetime.now(timezone.utc)
        reference_start = end_date - timedelta(
            days=reference_period_days + current_period_days
        )
        reference_end = end_date - timedelta(days=current_period_days)
        current_start = reference_end

        reference_values = self._load_feature_window(
            feature_name, reference_start, reference_end,
        )
        current_values = self._load_feature_window(
            feature_name, current_start, end_date,
        )

        if not reference_values or not current_values:
            raise InsufficientDataError(
                reason="insufficient_data",
                details={
                    "feature": feature_name,
                    "reference_count": len(reference_values),
                    "current_count": len(current_values),
                    "minimum_required": 1,
                },
            )

        reference_data = pd.Series(reference_values)
        current_data = pd.Series(current_values)

        try:
            drift_metrics = self.drift_detector.detect_drift(
                feature_name, reference_data, current_data
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error monitoring drift for feature {feature_name}: {exc}")
            return None

        self._save_drift_metrics(drift_metrics)

        if drift_metrics.distribution_shift_detected:
            logger.warning(
                f"Feature drift detected for {feature_name}: "
                f"drift_score={drift_metrics.drift_score:.3f}"
            )

        return drift_metrics

    def _load_feature_window(
        self,
        feature_name: str,
        start: datetime,
        end: datetime,
    ) -> List[float]:
        """Load feature_values rows for ``feature_name`` between ``start`` and ``end``.

        Returns an empty list when ``self.db_engine`` is not configured or
        when the query yields zero rows. Callers must treat empty as
        ``InsufficientDataError`` rather than fabricate values (F-03-005).
        """
        if self.db_engine is None:
            return []
        try:
            from sqlalchemy import text  # local import to keep top-level light

            sql = text(
                """
                SELECT fv.value
                  FROM feature_values fv
                  JOIN feature_definitions fd ON fd.id = fv.feature_id
                 WHERE fd.name = :name
                   AND fv.computed_at >= :start
                   AND fv.computed_at < :end
                """
            )
            with self.db_engine.connect() as conn:
                rows = conn.execute(
                    sql, {"name": feature_name, "start": start, "end": end}
                ).fetchall()
            values: List[float] = []
            for row in rows:
                try:
                    values.append(float(row[0]))
                except (TypeError, ValueError):
                    continue
            return values
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"_load_feature_window query failed for {feature_name}: {exc}"
            )
            return []
    
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
        """Get comprehensive feature statistics from the feature store.

        Per PRD audit 2026-04 F-03-005 / Q4 default: previously returned
        ``{'count': np.random.randint(1000, 10000), 'mean': np.random.normal(...)}``
        which broke alerting / capacity-planning telemetry. We now compute
        real summary statistics from ``feature_values`` rows, or raise
        ``InsufficientDataError`` when no real rows exist.
        """
        from backend.exceptions import InsufficientDataError

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)

        stats: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []

        for feature_name in feature_names:
            if feature_name not in self.feature_registry:
                continue

            values = self._load_feature_window(feature_name, start, end)
            if not values:
                missing.append(feature_name)
                continue

            arr = np.asarray(values, dtype=float)
            non_null = arr[~np.isnan(arr)]
            stats[feature_name] = {
                'count': int(arr.size),
                'mean': float(non_null.mean()) if non_null.size else None,
                'std': float(non_null.std(ddof=0)) if non_null.size else None,
                'min': float(non_null.min()) if non_null.size else None,
                'max': float(non_null.max()) if non_null.size else None,
                'null_percentage': (
                    float((arr.size - non_null.size) / arr.size)
                    if arr.size else 0.0
                ),
                'unique_values': int(np.unique(non_null).size) if non_null.size else 0,
                'last_updated': end.isoformat(),
            }

        if missing and not stats:
            # Nothing real for any requested feature — surface as 503.
            raise InsufficientDataError(
                reason="insufficient_data",
                details={"missing_features": missing, "days_back": days_back},
            )

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
            drift_file = self.storage_path / f"drift_{drift_metrics.feature_name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
            
            with open(drift_file, 'w') as f:
                json.dump(asdict(drift_metrics), f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving drift metrics: {e}")
    
    def cleanup_old_features(self, days_to_keep: int = 90) -> int:
        """Clean up old feature data"""
        cleanup_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
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