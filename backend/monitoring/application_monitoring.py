"""
Application-Specific Monitoring
Monitors stock processing, recommendation engines, and business logic performance.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum
)

from backend.config.monitoring_config import monitoring_config
from backend.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Stock Processing Metrics
stock_processing_pipeline_duration = Histogram(
    'stock_processing_pipeline_seconds',
    'Stock processing pipeline duration',
    ['stage', 'tier', 'data_source']
)

stock_data_quality_score = Gauge(
    'stock_data_quality_score',
    'Stock data quality score (0-100)',
    ['ticker', 'data_type', 'source']
)

stock_tier_distribution = Gauge(
    'stock_tier_distribution',
    'Number of stocks per tier',
    ['tier', 'status']
)

stock_processing_errors = Counter(
    'stock_processing_errors_total',
    'Stock processing errors by type',
    ['error_type', 'tier', 'stage', 'data_source']
)

stock_api_calls_per_stock = Histogram(
    'stock_api_calls_per_processing',
    'API calls made per stock processing',
    ['tier', 'data_source']
)

stock_cache_efficiency = Gauge(
    'stock_cache_hit_ratio',
    'Stock data cache hit ratio',
    ['tier', 'data_type']
)

# Recommendation Engine Metrics
recommendation_generation_duration = Histogram(
    'recommendation_generation_seconds',
    'Time to generate recommendations',
    ['model_type', 'complexity']
)

recommendation_accuracy_tracking = Gauge(
    'recommendation_accuracy_score',
    'Recommendation accuracy score',
    ['model', 'time_horizon', 'tier']
)

model_feature_importance = Gauge(
    'model_feature_importance',
    'Feature importance in models',
    ['model_name', 'feature', 'feature_category']
)

recommendation_confidence_distribution = Histogram(
    'recommendation_confidence_score',
    'Distribution of recommendation confidence scores',
    ['model', 'recommendation_type']
)

# Analysis Engine Metrics
technical_analysis_duration = Histogram(
    'technical_analysis_seconds',
    'Technical analysis processing time',
    ['indicator_type', 'complexity']
)

fundamental_analysis_duration = Histogram(
    'fundamental_analysis_seconds',
    'Fundamental analysis processing time',
    ['analysis_type', 'data_source']
)

sentiment_analysis_duration = Histogram(
    'sentiment_analysis_seconds',
    'Sentiment analysis processing time',
    ['source_type', 'analysis_method']
)

# Business Logic Metrics
portfolio_calculation_duration = Histogram(
    'portfolio_calculation_seconds',
    'Portfolio calculation time',
    ['calculation_type', 'portfolio_size']
)

risk_calculation_duration = Histogram(
    'risk_calculation_seconds',
    'Risk calculation time',
    ['risk_type', 'method']
)

backtest_execution_duration = Histogram(
    'backtest_execution_seconds',
    'Backtest execution time',
    ['strategy', 'timeframe', 'complexity']
)

# ML Model Performance
model_training_duration = Histogram(
    'model_training_seconds',
    'Model training duration',
    ['model_type', 'data_size']
)

model_inference_batch_size = Histogram(
    'model_inference_batch_size',
    'Batch size for model inference',
    ['model_type']
)

model_memory_usage = Gauge(
    'model_memory_usage_bytes',
    'Memory usage by ML models',
    ['model_name', 'model_type']
)

model_prediction_drift = Gauge(
    'model_prediction_drift_score',
    'Model prediction drift detection score',
    ['model_name', 'drift_type']
)

# Data Pipeline Metrics
data_freshness = Gauge(
    'data_freshness_seconds',
    'Age of data in seconds',
    ['data_type', 'source', 'tier']
)

data_validation_failures = Counter(
    'data_validation_failures_total',
    'Data validation failures',
    ['validation_type', 'data_type', 'source']
)

data_transformation_duration = Histogram(
    'data_transformation_seconds',
    'Data transformation processing time',
    ['transformation_type', 'data_volume']
)


@dataclass
class StockProcessingMetrics:
    """Stock processing metrics tracking."""
    ticker: str
    tier: str
    start_time: datetime
    stages_completed: List[str]
    api_calls_made: int
    cache_hits: int
    cache_misses: int
    data_quality_scores: Dict[str, float]
    errors: List[Dict[str, Any]]


@dataclass
class RecommendationMetrics:
    """Recommendation generation metrics."""
    model_name: str
    generation_time: float
    input_features: int
    confidence_score: float
    recommendation_type: str
    complexity_score: float


class ApplicationMonitor:
    """
    Comprehensive application-specific monitoring.
    """
    
    def __init__(self):
        self.stock_processing_sessions: Dict[str, StockProcessingMetrics] = {}
        self.recommendation_history: deque = deque(maxlen=10000)
        self.model_performance_cache: Dict[str, Dict] = {}
        self.active_processing_count = 0
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.processing_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_patterns: Dict[str, int] = defaultdict(int)
        
    async def start_monitoring(self) -> None:
        """Start application monitoring."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started application-specific monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop application monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped application-specific monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await self._collect_application_metrics()
                await self._analyze_performance_trends()
                await self._check_data_freshness()
                await self._monitor_model_drift()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in application monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_application_metrics(self) -> None:
        """Collect application-specific metrics."""
        try:
            # Stock tier distribution
            tier_counts = await self._get_stock_tier_distribution()
            for tier, counts in tier_counts.items():
                for status, count in counts.items():
                    stock_tier_distribution.labels(tier=tier, status=status).set(count)
            
            # Cache efficiency by tier
            cache_stats = await self._get_cache_efficiency_stats()
            for tier, data_types in cache_stats.items():
                for data_type, hit_ratio in data_types.items():
                    stock_cache_efficiency.labels(tier=tier, data_type=data_type).set(hit_ratio)
            
            # Processing rates
            processing_rates = await self._get_processing_rates()
            for tier, rate in processing_rates.items():
                self.processing_rates[tier].append(rate)
        
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and detect anomalies."""
        try:
            for tier, rates in self.processing_rates.items():
                if len(rates) >= 10:  # Need enough data points
                    recent_avg = sum(list(rates)[-5:]) / 5
                    historical_avg = sum(rates) / len(rates)
                    
                    # Detect significant performance degradation
                    if recent_avg < historical_avg * 0.7:  # 30% degradation
                        logger.warning(
                            f"Performance degradation detected for tier {tier}",
                            extra={
                                "tier": tier,
                                "recent_avg": recent_avg,
                                "historical_avg": historical_avg,
                                "degradation_pct": (1 - recent_avg/historical_avg) * 100
                            }
                        )
        
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
    
    async def _check_data_freshness(self) -> None:
        """Check data freshness and update metrics."""
        try:
            from backend.repositories.stock_repository import StockRepository
            
            stock_repo = StockRepository()
            freshness_data = await stock_repo.get_data_freshness_stats()
            
            for data_type, sources in freshness_data.items():
                for source, tiers in sources.items():
                    for tier, age_seconds in tiers.items():
                        data_freshness.labels(
                            data_type=data_type,
                            source=source,
                            tier=tier
                        ).set(age_seconds)
        
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
    
    async def _monitor_model_drift(self) -> None:
        """Monitor ML model prediction drift."""
        try:
            from backend.ml.model_manager import ModelManager
            
            model_manager = ModelManager()
            drift_scores = await model_manager.get_drift_scores()
            
            for model_name, drift_data in drift_scores.items():
                for drift_type, score in drift_data.items():
                    model_prediction_drift.labels(
                        model_name=model_name,
                        drift_type=drift_type
                    ).set(score)
        
        except Exception as e:
            logger.error(f"Error monitoring model drift: {e}")
    
    async def _get_stock_tier_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get stock distribution across tiers."""
        try:
            from backend.repositories.stock_repository import StockRepository
            
            stock_repo = StockRepository()
            return await stock_repo.get_tier_distribution()
        except Exception as e:
            logger.error(f"Error getting tier distribution: {e}")
            return {}
    
    async def _get_cache_efficiency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get cache efficiency statistics."""
        try:
            from backend.utils.advanced_cache import cache_manager
            
            return await cache_manager.get_efficiency_stats_by_tier()
        except Exception as e:
            logger.error(f"Error getting cache efficiency stats: {e}")
            return {}
    
    async def _get_processing_rates(self) -> Dict[str, float]:
        """Get current processing rates by tier."""
        try:
            from backend.repositories.stock_repository import StockRepository
            
            stock_repo = StockRepository()
            return await stock_repo.get_processing_rates()
        except Exception as e:
            logger.error(f"Error getting processing rates: {e}")
            return {}
    
    # Context managers for tracking specific operations
    
    def track_stock_processing(self, ticker: str, tier: str) -> 'StockProcessingContext':
        """Track stock processing operation."""
        return StockProcessingContext(self, ticker, tier)
    
    def track_recommendation_generation(self, model_name: str) -> 'RecommendationContext':
        """Track recommendation generation."""
        return RecommendationContext(self, model_name)
    
    def track_analysis_operation(self, analysis_type: str, complexity: str = "medium") -> 'AnalysisContext':
        """Track analysis operation."""
        return AnalysisContext(self, analysis_type, complexity)
    
    def track_model_operation(self, model_name: str, operation: str) -> 'ModelOperationContext':
        """Track ML model operation."""
        return ModelOperationContext(self, model_name, operation)
    
    # Recording methods
    
    def record_stock_processing_complete(self, metrics: StockProcessingMetrics) -> None:
        """Record completed stock processing metrics."""
        duration = (datetime.now() - metrics.start_time).total_seconds()
        
        # Record duration for each completed stage
        for stage in metrics.stages_completed:
            stock_processing_pipeline_duration.labels(
                stage=stage,
                tier=metrics.tier,
                data_source="combined"
            ).observe(duration / len(metrics.stages_completed))
        
        # Record API calls
        stock_api_calls_per_stock.labels(
            tier=metrics.tier,
            data_source="combined"
        ).observe(metrics.api_calls_made)
        
        # Record data quality scores
        for data_type, score in metrics.data_quality_scores.items():
            stock_data_quality_score.labels(
                ticker=metrics.ticker,
                data_type=data_type,
                source="combined"
            ).set(score)
        
        # Record errors
        for error in metrics.errors:
            stock_processing_errors.labels(
                error_type=error.get('type', 'unknown'),
                tier=metrics.tier,
                stage=error.get('stage', 'unknown'),
                data_source=error.get('source', 'unknown')
            ).inc()
    
    def record_recommendation_metrics(self, metrics: RecommendationMetrics) -> None:
        """Record recommendation generation metrics."""
        recommendation_generation_duration.labels(
            model_type=metrics.model_name,
            complexity="high" if metrics.complexity_score > 0.7 else "medium" if metrics.complexity_score > 0.3 else "low"
        ).observe(metrics.generation_time)
        
        recommendation_confidence_distribution.labels(
            model=metrics.model_name,
            recommendation_type=metrics.recommendation_type
        ).observe(metrics.confidence_score)
        
        # Store for trend analysis
        self.recommendation_history.append({
            "timestamp": datetime.now(),
            "model": metrics.model_name,
            "confidence": metrics.confidence_score,
            "generation_time": metrics.generation_time
        })
    
    def record_data_validation_failure(self, validation_type: str, data_type: str, source: str) -> None:
        """Record data validation failure."""
        data_validation_failures.labels(
            validation_type=validation_type,
            data_type=data_type,
            source=source
        ).inc()
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            recent_recommendations = list(self.recommendation_history)[-100:]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "active_processing": len(self.stock_processing_sessions),
                "recent_recommendations": len(recent_recommendations),
                "average_confidence": sum(r['confidence'] for r in recent_recommendations) / len(recent_recommendations) if recent_recommendations else 0,
                "error_patterns": dict(self.error_patterns),
                "processing_rates": {tier: list(rates)[-5:] for tier, rates in self.processing_rates.items()}
            }
        
        except Exception as e:
            logger.error(f"Error generating monitoring summary: {e}")
            return {"error": str(e)}


class StockProcessingContext:
    """Context manager for stock processing tracking."""
    
    def __init__(self, monitor: ApplicationMonitor, ticker: str, tier: str):
        self.monitor = monitor
        self.ticker = ticker
        self.tier = tier
        self.metrics: Optional[StockProcessingMetrics] = None
    
    async def __aenter__(self):
        self.metrics = StockProcessingMetrics(
            ticker=self.ticker,
            tier=self.tier,
            start_time=datetime.now(),
            stages_completed=[],
            api_calls_made=0,
            cache_hits=0,
            cache_misses=0,
            data_quality_scores={},
            errors=[]
        )
        
        self.monitor.stock_processing_sessions[self.ticker] = self.metrics
        self.monitor.active_processing_count += 1
        
        return self.metrics
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.metrics:
            # Record completion
            self.monitor.record_stock_processing_complete(self.metrics)
            
            # Clean up
            if self.ticker in self.monitor.stock_processing_sessions:
                del self.monitor.stock_processing_sessions[self.ticker]
            
            self.monitor.active_processing_count = max(0, self.monitor.active_processing_count - 1)
            
            # Log any exceptions
            if exc_type:
                self.metrics.errors.append({
                    "type": exc_type.__name__,
                    "message": str(exc_val),
                    "stage": "processing",
                    "source": "system"
                })


class RecommendationContext:
    """Context manager for recommendation generation tracking."""
    
    def __init__(self, monitor: ApplicationMonitor, model_name: str):
        self.monitor = monitor
        self.model_name = model_name
        self.start_time: Optional[float] = None
        self.metrics: Optional[RecommendationMetrics] = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and self.metrics:
            self.metrics.generation_time = time.time() - self.start_time
            self.monitor.record_recommendation_metrics(self.metrics)
    
    def set_metrics(self, **kwargs):
        """Set recommendation metrics."""
        self.metrics = RecommendationMetrics(
            model_name=self.model_name,
            generation_time=0,  # Will be set in __aexit__
            **kwargs
        )


class AnalysisContext:
    """Context manager for analysis operation tracking."""
    
    def __init__(self, monitor: ApplicationMonitor, analysis_type: str, complexity: str):
        self.monitor = monitor
        self.analysis_type = analysis_type
        self.complexity = complexity
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            if "technical" in self.analysis_type.lower():
                technical_analysis_duration.labels(
                    indicator_type=self.analysis_type,
                    complexity=self.complexity
                ).observe(duration)
            elif "fundamental" in self.analysis_type.lower():
                fundamental_analysis_duration.labels(
                    analysis_type=self.analysis_type,
                    data_source="combined"
                ).observe(duration)
            elif "sentiment" in self.analysis_type.lower():
                sentiment_analysis_duration.labels(
                    source_type="combined",
                    analysis_method=self.analysis_type
                ).observe(duration)


class ModelOperationContext:
    """Context manager for ML model operation tracking."""
    
    def __init__(self, monitor: ApplicationMonitor, model_name: str, operation: str):
        self.monitor = monitor
        self.model_name = model_name
        self.operation = operation
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            if self.operation == "training":
                model_training_duration.labels(
                    model_type=self.model_name,
                    data_size="unknown"
                ).observe(duration)
            elif self.operation == "inference":
                from backend.monitoring.metrics_collector import model_inference_time
                model_inference_time.labels(
                    model_name=self.model_name
                ).observe(duration)


# Global application monitor
app_monitor = ApplicationMonitor()


# Setup function
async def setup_application_monitoring():
    """Setup application monitoring."""
    await app_monitor.start_monitoring()
    logger.info("Application monitoring setup completed")


# Utility functions for easy integration
def track_stock_processing(ticker: str, tier: str):
    """Convenience function to track stock processing."""
    return app_monitor.track_stock_processing(ticker, tier)


def track_recommendation_generation(model_name: str):
    """Convenience function to track recommendation generation."""
    return app_monitor.track_recommendation_generation(model_name)


def track_analysis(analysis_type: str, complexity: str = "medium"):
    """Convenience function to track analysis operations."""
    return app_monitor.track_analysis_operation(analysis_type, complexity)


def track_model_operation(model_name: str, operation: str):
    """Convenience function to track model operations."""
    return app_monitor.track_model_operation(model_name, operation)