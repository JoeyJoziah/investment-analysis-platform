"""
SLA Tracking and Monitoring System
Tracks service level agreements for different stock tiers
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import json
import numpy as np

from backend.utils.cache import get_redis
from backend.utils.enhanced_cost_monitor import StockPriority
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class SLAMetric(Enum):
    """Types of SLA metrics"""
    DATA_FRESHNESS = "data_freshness"
    UPDATE_FREQUENCY = "update_frequency"
    API_LATENCY = "api_latency"
    DATA_COMPLETENESS = "data_completeness"
    QUALITY_SCORE = "quality_score"
    AVAILABILITY = "availability"
    ERROR_RATE = "error_rate"


@dataclass
class SLATarget:
    """SLA target definition"""
    metric: SLAMetric
    target_value: float
    unit: str
    measurement_window: timedelta
    critical_threshold: float  # Below this is SLA violation
    warning_threshold: float   # Below this triggers warning


@dataclass
class SLAMeasurement:
    """Single SLA measurement"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric: SLAMetric = SLAMetric.DATA_FRESHNESS
    tier: StockPriority = StockPriority.MEDIUM
    ticker: Optional[str] = None
    value: float = 0.0
    unit: str = ""
    meets_sla: bool = True
    warning: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLATracker:
    """
    Comprehensive SLA tracking system for tiered stock data
    """
    
    def __init__(self):
        self.redis = None
        self.sla_targets = self._define_sla_targets()
        self.measurements = defaultdict(lambda: deque(maxlen=10000))
        self.violations = defaultdict(list)
        self.performance_history = defaultdict(lambda: deque(maxlen=168))  # 7 days hourly
        
    def _define_sla_targets(self) -> Dict[StockPriority, Dict[SLAMetric, SLATarget]]:
        """Define SLA targets for each tier"""
        return {
            StockPriority.CRITICAL: {
                SLAMetric.DATA_FRESHNESS: SLATarget(
                    metric=SLAMetric.DATA_FRESHNESS,
                    target_value=5,  # 5 minutes
                    unit="minutes",
                    measurement_window=timedelta(hours=1),
                    critical_threshold=15,  # 15 minutes
                    warning_threshold=10    # 10 minutes
                ),
                SLAMetric.UPDATE_FREQUENCY: SLATarget(
                    metric=SLAMetric.UPDATE_FREQUENCY,
                    target_value=60,  # Updates per hour
                    unit="updates/hour",
                    measurement_window=timedelta(hours=1),
                    critical_threshold=30,
                    warning_threshold=45
                ),
                SLAMetric.API_LATENCY: SLATarget(
                    metric=SLAMetric.API_LATENCY,
                    target_value=100,  # 100ms
                    unit="milliseconds",
                    measurement_window=timedelta(minutes=5),
                    critical_threshold=500,
                    warning_threshold=300
                ),
                SLAMetric.DATA_COMPLETENESS: SLATarget(
                    metric=SLAMetric.DATA_COMPLETENESS,
                    target_value=99.9,  # 99.9%
                    unit="percent",
                    measurement_window=timedelta(hours=24),
                    critical_threshold=95,
                    warning_threshold=98
                ),
                SLAMetric.AVAILABILITY: SLATarget(
                    metric=SLAMetric.AVAILABILITY,
                    target_value=99.9,  # 99.9%
                    unit="percent",
                    measurement_window=timedelta(hours=24),
                    critical_threshold=95,
                    warning_threshold=98
                ),
                SLAMetric.ERROR_RATE: SLATarget(
                    metric=SLAMetric.ERROR_RATE,
                    target_value=0.1,  # 0.1%
                    unit="percent",
                    measurement_window=timedelta(hours=1),
                    critical_threshold=5,
                    warning_threshold=2
                )
            },
            StockPriority.HIGH: {
                SLAMetric.DATA_FRESHNESS: SLATarget(
                    metric=SLAMetric.DATA_FRESHNESS,
                    target_value=15,  # 15 minutes
                    unit="minutes",
                    measurement_window=timedelta(hours=4),
                    critical_threshold=60,
                    warning_threshold=30
                ),
                SLAMetric.UPDATE_FREQUENCY: SLATarget(
                    metric=SLAMetric.UPDATE_FREQUENCY,
                    target_value=15,  # Updates per hour
                    unit="updates/hour",
                    measurement_window=timedelta(hours=4),
                    critical_threshold=4,
                    warning_threshold=8
                ),
                SLAMetric.API_LATENCY: SLATarget(
                    metric=SLAMetric.API_LATENCY,
                    target_value=200,
                    unit="milliseconds",
                    measurement_window=timedelta(minutes=15),
                    critical_threshold=1000,
                    warning_threshold=500
                ),
                SLAMetric.DATA_COMPLETENESS: SLATarget(
                    metric=SLAMetric.DATA_COMPLETENESS,
                    target_value=99,
                    unit="percent",
                    measurement_window=timedelta(hours=24),
                    critical_threshold=90,
                    warning_threshold=95
                ),
                SLAMetric.AVAILABILITY: SLATarget(
                    metric=SLAMetric.AVAILABILITY,
                    target_value=99,
                    unit="percent",
                    measurement_window=timedelta(hours=24),
                    critical_threshold=90,
                    warning_threshold=95
                )
            },
            StockPriority.MEDIUM: {
                SLAMetric.DATA_FRESHNESS: SLATarget(
                    metric=SLAMetric.DATA_FRESHNESS,
                    target_value=60,  # 1 hour
                    unit="minutes",
                    measurement_window=timedelta(hours=8),
                    critical_threshold=240,  # 4 hours
                    warning_threshold=120    # 2 hours
                ),
                SLAMetric.UPDATE_FREQUENCY: SLATarget(
                    metric=SLAMetric.UPDATE_FREQUENCY,
                    target_value=3,  # Updates per hour
                    unit="updates/hour",
                    measurement_window=timedelta(hours=8),
                    critical_threshold=1,
                    warning_threshold=2
                ),
                SLAMetric.DATA_COMPLETENESS: SLATarget(
                    metric=SLAMetric.DATA_COMPLETENESS,
                    target_value=95,
                    unit="percent",
                    measurement_window=timedelta(hours=24),
                    critical_threshold=85,
                    warning_threshold=90
                )
            },
            StockPriority.LOW: {
                SLAMetric.DATA_FRESHNESS: SLATarget(
                    metric=SLAMetric.DATA_FRESHNESS,
                    target_value=480,  # 8 hours
                    unit="minutes",
                    measurement_window=timedelta(hours=24),
                    critical_threshold=1440,  # 24 hours
                    warning_threshold=720     # 12 hours
                ),
                SLAMetric.UPDATE_FREQUENCY: SLATarget(
                    metric=SLAMetric.UPDATE_FREQUENCY,
                    target_value=0.125,  # Updates per hour (1 per 8 hours)
                    unit="updates/hour",
                    measurement_window=timedelta(hours=24),
                    critical_threshold=0.042,  # 1 per day
                    warning_threshold=0.083   # 2 per day
                )
            },
            StockPriority.MINIMAL: {
                SLAMetric.DATA_FRESHNESS: SLATarget(
                    metric=SLAMetric.DATA_FRESHNESS,
                    target_value=10080,  # 7 days
                    unit="minutes",
                    measurement_window=timedelta(days=7),
                    critical_threshold=20160,  # 14 days
                    warning_threshold=15120    # 10.5 days
                )
            }
        }
    
    async def initialize(self):
        """Initialize the SLA tracker"""
        self.redis = await get_redis()
        await self._load_historical_data()
        logger.info("SLA tracker initialized")
    
    async def record_measurement(
        self,
        metric: SLAMetric,
        tier: StockPriority,
        value: float,
        ticker: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> SLAMeasurement:
        """
        Record an SLA measurement
        
        Args:
            metric: Type of metric
            tier: Stock tier
            value: Measured value
            ticker: Optional specific ticker
            metadata: Additional metadata
            
        Returns:
            SLA measurement with evaluation
        """
        # Get SLA target
        tier_slas = self.sla_targets.get(tier, {})
        sla_target = tier_slas.get(metric)
        
        # Create measurement
        measurement = SLAMeasurement(
            metric=metric,
            tier=tier,
            ticker=ticker,
            value=value,
            unit=sla_target.unit if sla_target else "",
            metadata=metadata or {}
        )
        
        # Evaluate against SLA
        if sla_target:
            if metric in [SLAMetric.DATA_FRESHNESS, SLAMetric.API_LATENCY, SLAMetric.ERROR_RATE]:
                # Lower is better
                measurement.meets_sla = value <= sla_target.critical_threshold
                measurement.warning = (
                    value > sla_target.warning_threshold and
                    value <= sla_target.critical_threshold
                )
            else:
                # Higher is better
                measurement.meets_sla = value >= sla_target.critical_threshold
                measurement.warning = (
                    value < sla_target.warning_threshold and
                    value >= sla_target.critical_threshold
                )
        
        # Store measurement
        key = f"{tier.value}:{metric.value}"
        self.measurements[key].append(measurement)
        
        # Track violations
        if not measurement.meets_sla:
            await self._record_violation(measurement, sla_target)
        
        # Update performance history
        await self._update_performance_history(tier, metric, value)
        
        # Persist to Redis
        await self._persist_measurement(measurement)
        
        return measurement
    
    async def get_sla_status(
        self,
        tier: Optional[StockPriority] = None,
        metric: Optional[SLAMetric] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get current SLA status
        
        Args:
            tier: Filter by tier
            metric: Filter by metric
            time_window: Time window for calculation
            
        Returns:
            SLA status report
        """
        if not time_window:
            time_window = timedelta(hours=24)
        
        cutoff_time = datetime.utcnow() - time_window
        status_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'time_window': str(time_window),
            'overall_compliance': 0.0,
            'tiers': {}
        }
        
        # Calculate for each tier
        tiers_to_check = [tier] if tier else list(StockPriority)
        
        for check_tier in tiers_to_check:
            tier_status = {
                'compliance_rate': 0.0,
                'metrics': {},
                'violations': 0,
                'warnings': 0
            }
            
            # Check each metric
            metrics_to_check = [metric] if metric else list(SLAMetric)
            tier_slas = self.sla_targets.get(check_tier, {})
            
            for check_metric in metrics_to_check:
                if check_metric not in tier_slas:
                    continue
                
                key = f"{check_tier.value}:{check_metric.value}"
                measurements = [
                    m for m in self.measurements[key]
                    if m.timestamp >= cutoff_time
                ]
                
                if measurements:
                    compliant = sum(1 for m in measurements if m.meets_sla)
                    warnings = sum(1 for m in measurements if m.warning)
                    total = len(measurements)
                    
                    metric_status = {
                        'compliance_rate': (compliant / total) * 100,
                        'total_measurements': total,
                        'compliant': compliant,
                        'violations': total - compliant,
                        'warnings': warnings,
                        'average_value': np.mean([m.value for m in measurements]),
                        'p95_value': np.percentile([m.value for m in measurements], 95),
                        'target': tier_slas[check_metric].target_value,
                        'unit': tier_slas[check_metric].unit
                    }
                    
                    tier_status['metrics'][check_metric.value] = metric_status
                    tier_status['violations'] += metric_status['violations']
                    tier_status['warnings'] += warnings
            
            # Calculate tier compliance
            if tier_status['metrics']:
                tier_status['compliance_rate'] = np.mean([
                    m['compliance_rate'] for m in tier_status['metrics'].values()
                ])
            
            status_report['tiers'][check_tier.value] = tier_status
        
        # Calculate overall compliance
        if status_report['tiers']:
            status_report['overall_compliance'] = np.mean([
                t['compliance_rate'] for t in status_report['tiers'].values()
            ])
        
        return status_report
    
    async def get_tier_performance(
        self,
        tier: StockPriority,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get detailed performance metrics for a tier
        
        Args:
            tier: Stock tier
            hours: Hours of history to analyze
            
        Returns:
            Performance report
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        performance = {
            'tier': tier.value,
            'period_hours': hours,
            'metrics': {},
            'trending': {},
            'alerts': []
        }
        
        tier_slas = self.sla_targets.get(tier, {})
        
        for metric, sla_target in tier_slas.items():
            key = f"{tier.value}:{metric.value}"
            measurements = [
                m for m in self.measurements[key]
                if m.timestamp >= cutoff_time
            ]
            
            if not measurements:
                continue
            
            values = [m.value for m in measurements]
            timestamps = [m.timestamp for m in measurements]
            
            # Calculate statistics
            metric_perf = {
                'current_value': values[-1] if values else 0,
                'average': np.mean(values),
                'median': np.median(values),
                'std_dev': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'target': sla_target.target_value,
                'compliance_rate': sum(1 for m in measurements if m.meets_sla) / len(measurements) * 100
            }
            
            # Calculate trend
            if len(values) > 10:
                recent_avg = np.mean(values[-5:])
                older_avg = np.mean(values[-10:-5])
                trend = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0
                metric_perf['trend'] = trend
                
                # Trending analysis
                if abs(trend) > 20:
                    performance['trending'][metric.value] = {
                        'direction': 'improving' if trend < 0 else 'degrading',
                        'change_percent': abs(trend)
                    }
            
            performance['metrics'][metric.value] = metric_perf
            
            # Generate alerts
            if metric_perf['compliance_rate'] < 90:
                performance['alerts'].append({
                    'severity': 'high' if metric_perf['compliance_rate'] < 80 else 'medium',
                    'metric': metric.value,
                    'message': f"{metric.value} compliance at {metric_perf['compliance_rate']:.1f}%"
                })
        
        return performance
    
    async def get_violation_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tier: Optional[StockPriority] = None
    ) -> Dict[str, Any]:
        """
        Get SLA violation report
        
        Args:
            start_date: Start of report period
            end_date: End of report period
            tier: Filter by tier
            
        Returns:
            Violation report
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
        
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_violations': 0,
            'violations_by_tier': {},
            'violations_by_metric': {},
            'top_violators': [],
            'violation_patterns': []
        }
        
        # Aggregate violations
        for key, violations_list in self.violations.items():
            tier_str, metric_str = key.split(':')
            
            if tier and tier.value != tier_str:
                continue
            
            period_violations = [
                v for v in violations_list
                if start_date <= v['timestamp'] <= end_date
            ]
            
            if not period_violations:
                continue
            
            # Count by tier
            if tier_str not in report['violations_by_tier']:
                report['violations_by_tier'][tier_str] = 0
            report['violations_by_tier'][tier_str] += len(period_violations)
            
            # Count by metric
            if metric_str not in report['violations_by_metric']:
                report['violations_by_metric'][metric_str] = 0
            report['violations_by_metric'][metric_str] += len(period_violations)
            
            report['total_violations'] += len(period_violations)
            
            # Track top violating tickers
            ticker_violations = defaultdict(int)
            for v in period_violations:
                if v.get('ticker'):
                    ticker_violations[v['ticker']] += 1
            
            for ticker, count in ticker_violations.items():
                report['top_violators'].append({
                    'ticker': ticker,
                    'tier': tier_str,
                    'metric': metric_str,
                    'violation_count': count
                })
        
        # Sort top violators
        report['top_violators'].sort(key=lambda x: x['violation_count'], reverse=True)
        report['top_violators'] = report['top_violators'][:20]
        
        # Detect patterns
        report['violation_patterns'] = await self._detect_violation_patterns(
            start_date,
            end_date
        )
        
        return report
    
    async def calculate_sla_credits(
        self,
        tier: StockPriority,
        period: timedelta
    ) -> Dict[str, Any]:
        """
        Calculate SLA credits/penalties based on violations
        
        Args:
            tier: Stock tier
            period: Period to calculate
            
        Returns:
            Credit calculation
        """
        # Define credit model (simplified)
        credit_model = {
            StockPriority.CRITICAL: {
                'base_value': 1000,  # Base value units per period
                'violation_penalty': 50,  # Units per violation
                'extended_outage_multiplier': 2  # For violations > 1 hour
            },
            StockPriority.HIGH: {
                'base_value': 500,
                'violation_penalty': 20,
                'extended_outage_multiplier': 1.5
            },
            StockPriority.MEDIUM: {
                'base_value': 200,
                'violation_penalty': 5,
                'extended_outage_multiplier': 1.2
            }
        }
        
        model = credit_model.get(tier, {'base_value': 100, 'violation_penalty': 2})
        
        # Get violations in period
        cutoff = datetime.utcnow() - period
        violations = []
        
        for metric in self.sla_targets.get(tier, {}).keys():
            key = f"{tier.value}:{metric.value}"
            violations.extend([
                v for v in self.violations.get(key, [])
                if v['timestamp'] >= cutoff
            ])
        
        # Calculate credits
        total_penalty = len(violations) * model['violation_penalty']
        
        # Check for extended outages
        extended_outages = [
            v for v in violations
            if v.get('duration_minutes', 0) > 60
        ]
        
        if extended_outages:
            total_penalty *= model.get('extended_outage_multiplier', 1)
        
        credits = {
            'tier': tier.value,
            'period': str(period),
            'base_value': model['base_value'],
            'violations': len(violations),
            'penalty': total_penalty,
            'credit_amount': max(0, model['base_value'] - total_penalty),
            'credit_percentage': max(0, (1 - total_penalty / model['base_value']) * 100)
        }
        
        return credits
    
    async def generate_sla_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for SLA dashboard visualization
        
        Returns:
            Dashboard data
        """
        dashboard = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {},
            'tier_breakdown': {},
            'metric_trends': {},
            'recent_violations': [],
            'alerts': []
        }
        
        # Overall summary
        status_24h = await self.get_sla_status(time_window=timedelta(hours=24))
        dashboard['summary'] = {
            'overall_compliance': status_24h['overall_compliance'],
            'total_violations_24h': sum(
                t.get('violations', 0) for t in status_24h['tiers'].values()
            ),
            'active_warnings': sum(
                t.get('warnings', 0) for t in status_24h['tiers'].values()
            )
        }
        
        # Tier breakdown
        for tier in StockPriority:
            tier_perf = await self.get_tier_performance(tier, hours=24)
            dashboard['tier_breakdown'][tier.value] = {
                'compliance': tier_perf.get('metrics', {}).get(
                    SLAMetric.DATA_FRESHNESS.value, {}
                ).get('compliance_rate', 100),
                'alerts': len(tier_perf.get('alerts', [])),
                'trending': tier_perf.get('trending', {})
            }
        
        # Metric trends (last 7 days, hourly)
        for metric in [SLAMetric.DATA_FRESHNESS, SLAMetric.API_LATENCY]:
            trend_data = []
            for hour_offset in range(168):  # 7 days
                timestamp = datetime.utcnow() - timedelta(hours=hour_offset)
                hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                
                # Aggregate across all tiers
                values = []
                for tier in StockPriority:
                    key = f"{tier.value}:{metric.value}"
                    hour_measurements = [
                        m.value for m in self.measurements[key]
                        if m.timestamp.strftime('%Y-%m-%d %H:00') == hour_key
                    ]
                    if hour_measurements:
                        values.extend(hour_measurements)
                
                if values:
                    trend_data.append({
                        'timestamp': hour_key,
                        'average': np.mean(values),
                        'p95': np.percentile(values, 95)
                    })
            
            dashboard['metric_trends'][metric.value] = trend_data
        
        # Recent violations
        all_violations = []
        for violations_list in self.violations.values():
            all_violations.extend(violations_list[-10:])  # Last 10 per category
        
        all_violations.sort(key=lambda x: x['timestamp'], reverse=True)
        dashboard['recent_violations'] = all_violations[:20]
        
        # Active alerts
        for tier in StockPriority:
            tier_perf = await self.get_tier_performance(tier, hours=1)
            for alert in tier_perf.get('alerts', []):
                alert['tier'] = tier.value
                dashboard['alerts'].append(alert)
        
        return dashboard
    
    # Helper methods
    
    async def _record_violation(
        self,
        measurement: SLAMeasurement,
        sla_target: SLATarget
    ):
        """Record an SLA violation"""
        violation = {
            'timestamp': measurement.timestamp,
            'metric': measurement.metric.value,
            'tier': measurement.tier.value,
            'ticker': measurement.ticker,
            'value': measurement.value,
            'target': sla_target.target_value,
            'threshold': sla_target.critical_threshold,
            'severity': self._calculate_severity(
                measurement.value,
                sla_target
            )
        }
        
        key = f"{measurement.tier.value}:{measurement.metric.value}"
        self.violations[key].append(violation)
        
        # Persist to Redis
        await self._persist_violation(violation)
        
        # Log violation
        logger.warning(
            f"SLA violation: {measurement.tier.value}/{measurement.metric.value} "
            f"value={measurement.value} (target={sla_target.target_value})"
        )
    
    def _calculate_severity(
        self,
        value: float,
        sla_target: SLATarget
    ) -> str:
        """Calculate violation severity"""
        if sla_target.metric in [SLAMetric.DATA_FRESHNESS, SLAMetric.API_LATENCY]:
            # Lower is better
            deviation = (value - sla_target.target_value) / sla_target.target_value
        else:
            # Higher is better
            deviation = (sla_target.target_value - value) / sla_target.target_value
        
        if abs(deviation) > 2:
            return 'critical'
        elif abs(deviation) > 1:
            return 'high'
        elif abs(deviation) > 0.5:
            return 'medium'
        else:
            return 'low'
    
    async def _update_performance_history(
        self,
        tier: StockPriority,
        metric: SLAMetric,
        value: float
    ):
        """Update performance history"""
        hour_key = datetime.utcnow().strftime('%Y-%m-%d %H:00')
        history_key = f"{tier.value}:{metric.value}:{hour_key}"
        
        # Add to hourly aggregation
        if history_key not in self.performance_history:
            self.performance_history[history_key] = []
        
        self.performance_history[history_key].append(value)
    
    async def _detect_violation_patterns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Detect patterns in violations"""
        patterns = []
        
        # Time-based patterns
        hourly_violations = defaultdict(int)
        daily_violations = defaultdict(int)
        
        for violations_list in self.violations.values():
            for v in violations_list:
                if start_date <= v['timestamp'] <= end_date:
                    hour = v['timestamp'].hour
                    day = v['timestamp'].weekday()
                    hourly_violations[hour] += 1
                    daily_violations[day] += 1
        
        # Peak violation hours
        if hourly_violations:
            peak_hour = max(hourly_violations, key=hourly_violations.get)
            patterns.append({
                'type': 'peak_hour',
                'description': f"Most violations occur at {peak_hour}:00",
                'data': dict(hourly_violations)
            })
        
        # Peak violation days
        if daily_violations:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            peak_day = max(daily_violations, key=daily_violations.get)
            patterns.append({
                'type': 'peak_day',
                'description': f"Most violations occur on {days[peak_day]}",
                'data': {days[k]: v for k, v in daily_violations.items()}
            })
        
        return patterns
    
    async def _persist_measurement(self, measurement: SLAMeasurement):
        """Persist measurement to Redis"""
        key = f"sla:measurement:{measurement.tier.value}:{measurement.metric.value}:{measurement.timestamp.timestamp()}"
        await self.redis.set(
            key,
            json.dumps({
                'timestamp': measurement.timestamp.isoformat(),
                'metric': measurement.metric.value,
                'tier': measurement.tier.value,
                'ticker': measurement.ticker,
                'value': measurement.value,
                'meets_sla': measurement.meets_sla,
                'warning': measurement.warning
            }),
            ex=86400 * 7  # 7 days TTL
        )
    
    async def _persist_violation(self, violation: Dict):
        """Persist violation to Redis"""
        key = f"sla:violation:{violation['tier']}:{violation['metric']}:{violation['timestamp'].timestamp()}"
        await self.redis.set(
            key,
            json.dumps(violation, default=str),
            ex=86400 * 30  # 30 days TTL
        )
    
    async def _load_historical_data(self):
        """Load historical SLA data from Redis"""
        # In production, implement loading of historical measurements
        pass


# Global instance
_sla_tracker: Optional[SLATracker] = None


async def get_sla_tracker() -> SLATracker:
    """Get or create the global SLA tracker"""
    global _sla_tracker
    if _sla_tracker is None:
        _sla_tracker = SLATracker()
        await _sla_tracker.initialize()
    return _sla_tracker