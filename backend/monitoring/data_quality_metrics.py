"""
Data quality metrics integration with Prometheus.
Exports data quality checks, validation results, and anomaly detection metrics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest
)

from backend.utils.data_quality import (
    DataQualityChecker,
    DataQualitySeverity
)
from backend.utils.monitoring import metrics as base_metrics


# Create separate registry for data quality metrics
dq_registry = CollectorRegistry()

# Data quality score metrics
data_quality_score = Gauge(
    'data_quality_score',
    'Overall data quality score (0-100)',
    ['symbol', 'data_type'],
    registry=dq_registry
)

data_quality_checks_total = Counter(
    'data_quality_checks_total',
    'Total number of data quality checks performed',
    ['data_type', 'check_type', 'result'],
    registry=dq_registry
)

data_quality_issues = Counter(
    'data_quality_issues_total',
    'Total number of data quality issues detected',
    ['severity', 'issue_type', 'data_type'],
    registry=dq_registry
)

data_validation_failures = Counter(
    'data_validation_failures_total',
    'Total number of data validation failures',
    ['validation_type', 'field', 'reason'],
    registry=dq_registry
)

# Data freshness metrics
data_staleness_seconds = Gauge(
    'data_staleness_seconds',
    'Age of data in seconds',
    ['symbol', 'data_type'],
    registry=dq_registry
)

missing_data_points = Gauge(
    'missing_data_points',
    'Number of missing data points',
    ['symbol', 'data_type', 'period'],
    registry=dq_registry
)

# Anomaly detection metrics
anomalies_detected = Counter(
    'anomalies_detected_total',
    'Total number of anomalies detected',
    ['anomaly_type', 'symbol', 'severity'],
    registry=dq_registry
)

anomaly_score = Gauge(
    'anomaly_score',
    'Anomaly score for data (0-1, higher is more anomalous)',
    ['symbol', 'data_type'],
    registry=dq_registry
)

# Price data specific metrics
price_consistency_violations = Counter(
    'price_consistency_violations_total',
    'Price consistency violations (e.g., high < low)',
    ['violation_type', 'symbol'],
    registry=dq_registry
)

price_gap_percentage = Histogram(
    'price_gap_percentage',
    'Price gap percentage between periods',
    ['symbol'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    registry=dq_registry
)

volume_outliers = Counter(
    'volume_outliers_total',
    'Volume outliers detected',
    ['symbol', 'outlier_type'],
    registry=dq_registry
)

# Fundamental data metrics
fundamental_data_completeness = Gauge(
    'fundamental_data_completeness',
    'Percentage of fundamental data fields populated',
    ['symbol', 'report_type'],
    registry=dq_registry
)

fundamental_ratio_violations = Counter(
    'fundamental_ratio_violations_total',
    'Fundamental ratio violations (e.g., negative P/E)',
    ['ratio_type', 'symbol'],
    registry=dq_registry
)

# Technical indicator metrics
technical_indicator_validity = Gauge(
    'technical_indicator_validity',
    'Percentage of valid technical indicators',
    ['indicator_type', 'symbol'],
    registry=dq_registry
)

indicator_calculation_errors = Counter(
    'indicator_calculation_errors_total',
    'Technical indicator calculation errors',
    ['indicator_type', 'error_type'],
    registry=dq_registry
)

# Data pipeline metrics
pipeline_data_quality = Gauge(
    'pipeline_data_quality_score',
    'Data quality score at pipeline stage',
    ['pipeline_stage', 'data_source'],
    registry=dq_registry
)

data_transformation_errors = Counter(
    'data_transformation_errors_total',
    'Data transformation errors in pipeline',
    ['transformation_type', 'error_type'],
    registry=dq_registry
)

# Timing metrics
data_quality_check_duration = Histogram(
    'data_quality_check_duration_seconds',
    'Time taken for data quality checks',
    ['check_type', 'data_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=dq_registry
)

# Compliance metrics
data_compliance_violations = Counter(
    'data_compliance_violations_total',
    'Data compliance violations (SEC, GDPR)',
    ['compliance_type', 'violation_type'],
    registry=dq_registry
)

data_retention_violations = Counter(
    'data_retention_violations_total',
    'Data retention policy violations',
    ['data_type', 'policy_type'],
    registry=dq_registry
)


class DataQualityMetricsCollector:
    """
    Collector for data quality metrics integrated with Prometheus.
    """
    
    def __init__(self, quality_checker: Optional[DataQualityChecker] = None):
        """
        Initialize metrics collector.
        
        Args:
            quality_checker: DataQualityChecker instance
        """
        self.quality_checker = quality_checker or DataQualityChecker()
        self.last_check_times = {}
        self.quality_history = {}
        
    def record_quality_check(
        self,
        symbol: str,
        data_type: str,
        quality_result: Dict[str, Any]
    ):
        """
        Record results of a data quality check.
        
        Args:
            symbol: Stock symbol
            data_type: Type of data checked
            quality_result: Result from DataQualityChecker
        """
        # Record overall quality score
        score = quality_result.get('quality_score', 0)
        data_quality_score.labels(
            symbol=symbol,
            data_type=data_type
        ).set(score)
        
        # Record check completion
        data_quality_checks_total.labels(
            data_type=data_type,
            check_type='comprehensive',
            result='pass' if quality_result.get('valid') else 'fail'
        ).inc()
        
        # Record issues by severity
        issues = quality_result.get('issues', [])
        for issue in issues:
            severity = issue.get('severity', 'unknown')
            issue_type = issue.get('type', 'unknown')
            
            data_quality_issues.labels(
                severity=severity,
                issue_type=issue_type,
                data_type=data_type
            ).inc()
            
            # Record specific violation types
            if issue_type == 'high_less_than_low':
                price_consistency_violations.labels(
                    violation_type='high_low',
                    symbol=symbol
                ).inc()
            elif issue_type == 'close_outside_range':
                price_consistency_violations.labels(
                    violation_type='close_range',
                    symbol=symbol
                ).inc()
        
        # Store for history tracking
        if symbol not in self.quality_history:
            self.quality_history[symbol] = []
        
        self.quality_history[symbol].append({
            'timestamp': datetime.utcnow(),
            'score': score,
            'issues': len(issues)
        })
        
        # Keep only last 100 entries per symbol
        self.quality_history[symbol] = self.quality_history[symbol][-100:]
    
    def record_data_staleness(
        self,
        symbol: str,
        data_type: str,
        last_update: datetime
    ):
        """Record data staleness metrics."""
        staleness = (datetime.utcnow() - last_update).total_seconds()
        
        data_staleness_seconds.labels(
            symbol=symbol,
            data_type=data_type
        ).set(staleness)
        
        # Flag if data is stale (>1 day for daily data)
        if staleness > 86400:
            data_quality_issues.labels(
                severity='medium',
                issue_type='stale_data',
                data_type=data_type
            ).inc()
    
    def record_missing_data(
        self,
        symbol: str,
        data_type: str,
        period: str,
        missing_count: int
    ):
        """Record missing data points."""
        missing_data_points.labels(
            symbol=symbol,
            data_type=data_type,
            period=period
        ).set(missing_count)
        
        if missing_count > 0:
            data_quality_issues.labels(
                severity='low' if missing_count < 5 else 'medium',
                issue_type='missing_data',
                data_type=data_type
            ).inc()
    
    def record_anomaly(
        self,
        symbol: str,
        data_type: str,
        anomaly_type: str,
        score: float,
        severity: str = 'medium'
    ):
        """Record detected anomaly."""
        anomalies_detected.labels(
            anomaly_type=anomaly_type,
            symbol=symbol,
            severity=severity
        ).inc()
        
        anomaly_score.labels(
            symbol=symbol,
            data_type=data_type
        ).set(score)
    
    def record_price_gap(
        self,
        symbol: str,
        gap_percentage: float
    ):
        """Record price gap metrics."""
        price_gap_percentage.labels(symbol=symbol).observe(abs(gap_percentage))
        
        # Flag large gaps as anomalies
        if abs(gap_percentage) > 0.2:  # 20% gap
            self.record_anomaly(
                symbol=symbol,
                data_type='price',
                anomaly_type='large_gap',
                score=min(1.0, abs(gap_percentage)),
                severity='high' if abs(gap_percentage) > 0.3 else 'medium'
            )
    
    def record_volume_outlier(
        self,
        symbol: str,
        volume: int,
        avg_volume: float,
        std_dev: float
    ):
        """Record volume outlier."""
        z_score = abs((volume - avg_volume) / std_dev) if std_dev > 0 else 0
        
        if z_score > 3:
            outlier_type = 'extreme' if z_score > 5 else 'moderate'
            volume_outliers.labels(
                symbol=symbol,
                outlier_type=outlier_type
            ).inc()
            
            self.record_anomaly(
                symbol=symbol,
                data_type='volume',
                anomaly_type='outlier',
                score=min(1.0, z_score / 10),
                severity='high' if z_score > 5 else 'medium'
            )
    
    def record_fundamental_completeness(
        self,
        symbol: str,
        report_type: str,
        total_fields: int,
        populated_fields: int
    ):
        """Record fundamental data completeness."""
        completeness = (populated_fields / total_fields * 100) if total_fields > 0 else 0
        
        fundamental_data_completeness.labels(
            symbol=symbol,
            report_type=report_type
        ).set(completeness)
        
        if completeness < 70:
            data_quality_issues.labels(
                severity='medium' if completeness < 50 else 'low',
                issue_type='incomplete_fundamentals',
                data_type='fundamental'
            ).inc()
    
    def record_validation_failure(
        self,
        validation_type: str,
        field: str,
        reason: str,
        data_type: str = 'unknown'
    ):
        """Record data validation failure."""
        data_validation_failures.labels(
            validation_type=validation_type,
            field=field,
            reason=reason
        ).inc()
        
        data_quality_issues.labels(
            severity='medium',
            issue_type='validation_failure',
            data_type=data_type
        ).inc()
    
    def record_compliance_violation(
        self,
        compliance_type: str,
        violation_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record compliance violation."""
        data_compliance_violations.labels(
            compliance_type=compliance_type,
            violation_type=violation_type
        ).inc()
        
        # Log critical compliance issues
        if compliance_type in ['SEC', 'GDPR']:
            data_quality_issues.labels(
                severity='critical',
                issue_type='compliance_violation',
                data_type='compliance'
            ).inc()
    
    def record_pipeline_quality(
        self,
        pipeline_stage: str,
        data_source: str,
        quality_score: float
    ):
        """Record data quality at pipeline stage."""
        pipeline_data_quality.labels(
            pipeline_stage=pipeline_stage,
            data_source=data_source
        ).set(quality_score)
        
        # Flag poor quality data in pipeline
        if quality_score < 70:
            data_quality_issues.labels(
                severity='high' if quality_score < 50 else 'medium',
                issue_type='pipeline_quality',
                data_type='pipeline'
            ).inc()
    
    async def perform_quality_check_with_metrics(
        self,
        df,
        symbol: str,
        data_type: str = 'price'
    ) -> Dict[str, Any]:
        """
        Perform quality check and record metrics.
        
        Args:
            df: DataFrame to check
            symbol: Stock symbol
            data_type: Type of data
            
        Returns:
            Quality check results
        """
        import time
        
        # Time the quality check
        start_time = time.time()
        
        try:
            # Perform quality check
            result = self.quality_checker.validate_price_data(df, symbol)
            
            # Record metrics
            self.record_quality_check(symbol, data_type, result)
            
            # Record timing
            duration = time.time() - start_time
            data_quality_check_duration.labels(
                check_type='comprehensive',
                data_type=data_type
            ).observe(duration)
            
            return result
            
        except Exception as e:
            # Record error
            data_quality_issues.labels(
                severity='critical',
                issue_type='check_failure',
                data_type=data_type
            ).inc()
            
            raise
    
    def get_quality_trends(
        self,
        symbol: str,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get quality trends for a symbol.
        
        Args:
            symbol: Stock symbol
            window_hours: Time window in hours
            
        Returns:
            Quality trend analysis
        """
        if symbol not in self.quality_history:
            return {'status': 'no_data'}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        recent_checks = [
            check for check in self.quality_history[symbol]
            if check['timestamp'] > cutoff_time
        ]
        
        if not recent_checks:
            return {'status': 'no_recent_data'}
        
        scores = [check['score'] for check in recent_checks]
        issues = [check['issues'] for check in recent_checks]
        
        return {
            'status': 'ok',
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'total_issues': sum(issues),
            'check_count': len(recent_checks),
            'trend': 'improving' if scores[-1] > scores[0] else 'declining'
        }
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        return generate_latest(dq_registry)


# Global metrics collector instance
dq_metrics = DataQualityMetricsCollector()


# Integration function for existing data quality checks
async def check_data_quality_with_metrics(
    df,
    symbol: str,
    data_type: str = 'price'
) -> Dict[str, Any]:
    """
    Wrapper function to perform data quality checks with metrics.
    
    Args:
        df: DataFrame to check
        symbol: Stock symbol
        data_type: Type of data
        
    Returns:
        Quality check results
    """
    return await dq_metrics.perform_quality_check_with_metrics(df, symbol, data_type)