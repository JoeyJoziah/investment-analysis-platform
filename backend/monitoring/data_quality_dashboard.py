"""
Data Quality Monitoring Dashboard

Comprehensive data quality monitoring including:
- Data freshness and staleness detection
- Missing data analysis and reporting
- Data drift and anomaly detection
- Source reliability tracking
- Data lineage and dependency mapping
- Quality score calculation and trending
- Automated data quality alerts
- Data validation rules and compliance

Ensures high-quality data for investment decisions
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from scipy import stats

from backend.utils.cache import CacheManager
from backend.models.database import SessionLocal
from backend.monitoring.real_time_alerts import RealTimeAlertManager, AlertSeverity, AlertCategory

logger = logging.getLogger(__name__)

class DataQualityStatus(Enum):
    EXCELLENT = "excellent"     # 95-100%
    GOOD = "good"              # 85-94%
    FAIR = "fair"              # 70-84%
    POOR = "poor"              # 50-69%
    CRITICAL = "critical"      # <50%

class DataSourceType(Enum):
    PRICE_DATA = "price_data"
    VOLUME_DATA = "volume_data" 
    FUNDAMENTAL_DATA = "fundamental_data"
    NEWS_DATA = "news_data"
    SOCIAL_DATA = "social_data"
    OPTIONS_DATA = "options_data"
    INSIDER_DATA = "insider_data"
    ECONOMIC_DATA = "economic_data"

@dataclass
class DataQualityMetric:
    """Individual data quality metric"""
    metric_name: str
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    status: DataQualityStatus
    last_updated: datetime
    trend_7d: Optional[float] = None
    trend_30d: Optional[float] = None

@dataclass
class DataSourceQuality:
    """Data quality assessment for a source"""
    source_name: str
    source_type: DataSourceType
    overall_score: float
    status: DataQualityStatus
    metrics: Dict[str, DataQualityMetric]
    last_update: datetime
    issues: List[str]
    recommendations: List[str]

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system
    
    Features:
    - Real-time quality monitoring
    - Freshness tracking
    - Anomaly detection
    - Missing data analysis
    - Quality scoring and trending
    - Automated alerting
    - Data lineage tracking
    """
    
    def __init__(self, config: Dict, alert_manager: Optional[RealTimeAlertManager] = None):
        self.config = config
        self.cache = CacheManager()
        self.alert_manager = alert_manager
        
        # Quality thresholds
        self.thresholds = self.config.get('quality_thresholds', {
            'excellent': 0.95,
            'good': 0.85,
            'fair': 0.70,
            'poor': 0.50
        })
        
        # Data sources to monitor
        self.data_sources = self.config.get('data_sources', {})
        
        # Quality metrics cache
        self.quality_cache: Dict[str, DataSourceQuality] = {}
        
        # Historical quality data
        self.quality_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    async def assess_data_quality(self, source_name: str) -> DataSourceQuality:
        """Assess data quality for a specific source"""
        try:
            source_config = self.data_sources.get(source_name)
            if not source_config:
                raise ValueError(f"Unknown data source: {source_name}")
            
            source_type = DataSourceType(source_config['type'])
            
            # Calculate individual metrics
            metrics = {}
            
            # Freshness metric
            freshness_metric = await self._assess_freshness(source_name, source_config)
            metrics['freshness'] = freshness_metric
            
            # Completeness metric
            completeness_metric = await self._assess_completeness(source_name, source_config)
            metrics['completeness'] = completeness_metric
            
            # Accuracy metric
            accuracy_metric = await self._assess_accuracy(source_name, source_config)
            metrics['accuracy'] = accuracy_metric
            
            # Consistency metric
            consistency_metric = await self._assess_consistency(source_name, source_config)
            metrics['consistency'] = consistency_metric
            
            # Reliability metric
            reliability_metric = await self._assess_reliability(source_name, source_config)
            metrics['reliability'] = reliability_metric
            
            # Calculate overall score (weighted average)
            weights = source_config.get('metric_weights', {
                'freshness': 0.25,
                'completeness': 0.25,
                'accuracy': 0.20,
                'consistency': 0.15,
                'reliability': 0.15
            })
            
            overall_score = sum(metrics[metric].current_value * weights[metric] 
                              for metric in metrics.keys())
            
            # Determine overall status
            status = self._calculate_status(overall_score)
            
            # Identify issues and recommendations
            issues = self._identify_issues(metrics)
            recommendations = self._generate_recommendations(metrics, source_type)
            
            quality_assessment = DataSourceQuality(
                source_name=source_name,
                source_type=source_type,
                overall_score=overall_score,
                status=status,
                metrics=metrics,
                last_update=datetime.now(),
                issues=issues,
                recommendations=recommendations
            )
            
            # Cache assessment
            self.quality_cache[source_name] = quality_assessment
            
            # Update historical data
            self._update_quality_history(source_name, overall_score)
            
            # Check for alerts
            await self._check_quality_alerts(quality_assessment)
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error assessing data quality for {source_name}: {e}")
            raise e
    
    async def _assess_freshness(self, source_name: str, source_config: Dict) -> DataQualityMetric:
        """Assess data freshness"""
        try:
            table_name = source_config['table_name']
            timestamp_column = source_config.get('timestamp_column', 'created_at')
            expected_frequency = source_config.get('expected_frequency_hours', 24)
            
            # Query latest data timestamp
            query = f"""
                SELECT MAX({timestamp_column}) as latest_timestamp
                FROM {table_name}
                WHERE {timestamp_column} > NOW() - INTERVAL '7 days'
            """
            
            with SessionLocal() as session:
                result = session.execute(text(query)).fetchone()
                
                if result and result.latest_timestamp:
                    latest_timestamp = result.latest_timestamp
                    
                    # Calculate hours since last update
                    hours_since_update = (datetime.now() - latest_timestamp).total_seconds() / 3600
                    
                    # Calculate freshness score (1.0 = fresh, 0.0 = very stale)
                    if hours_since_update <= expected_frequency:
                        freshness_score = 1.0
                    elif hours_since_update <= expected_frequency * 2:
                        # Linear decay from 1.0 to 0.5
                        freshness_score = 1.0 - 0.5 * (hours_since_update - expected_frequency) / expected_frequency
                    else:
                        # Exponential decay after 2x expected frequency
                        excess_hours = hours_since_update - expected_frequency * 2
                        freshness_score = 0.5 * np.exp(-excess_hours / expected_frequency)
                else:
                    freshness_score = 0.0
                    hours_since_update = float('inf')
            
            status = self._calculate_status(freshness_score)
            
            return DataQualityMetric(
                metric_name='freshness',
                current_value=freshness_score,
                target_value=1.0,
                threshold_warning=0.8,
                threshold_critical=0.5,
                status=status,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing freshness for {source_name}: {e}")
            # Return default metric on error
            return DataQualityMetric(
                metric_name='freshness',
                current_value=0.0,
                target_value=1.0,
                threshold_warning=0.8,
                threshold_critical=0.5,
                status=DataQualityStatus.CRITICAL,
                last_updated=datetime.now()
            )
    
    async def _assess_completeness(self, source_name: str, source_config: Dict) -> DataQualityMetric:
        """Assess data completeness (missing values)"""
        try:
            table_name = source_config['table_name']
            required_columns = source_config.get('required_columns', [])
            
            if not required_columns:
                # No specific columns defined, use a general completeness check
                return DataQualityMetric(
                    metric_name='completeness',
                    current_value=1.0,
                    target_value=1.0,
                    threshold_warning=0.9,
                    threshold_critical=0.7,
                    status=DataQualityStatus.EXCELLENT,
                    last_updated=datetime.now()
                )
            
            # Calculate completeness for each required column
            completeness_scores = []
            
            for column in required_columns:
                query = f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT({column}) as non_null_rows
                    FROM {table_name}
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """
                
                with SessionLocal() as session:
                    result = session.execute(text(query)).fetchone()
                    
                    if result and result.total_rows > 0:
                        completeness = result.non_null_rows / result.total_rows
                    else:
                        completeness = 0.0
                    
                    completeness_scores.append(completeness)
            
            # Overall completeness is the minimum of all required columns
            overall_completeness = min(completeness_scores) if completeness_scores else 0.0
            status = self._calculate_status(overall_completeness)
            
            return DataQualityMetric(
                metric_name='completeness',
                current_value=overall_completeness,
                target_value=1.0,
                threshold_warning=0.9,
                threshold_critical=0.7,
                status=status,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing completeness for {source_name}: {e}")
            return DataQualityMetric(
                metric_name='completeness',
                current_value=0.0,
                target_value=1.0,
                threshold_warning=0.9,
                threshold_critical=0.7,
                status=DataQualityStatus.CRITICAL,
                last_updated=datetime.now()
            )
    
    async def _assess_accuracy(self, source_name: str, source_config: Dict) -> DataQualityMetric:
        """Assess data accuracy using validation rules"""
        try:
            table_name = source_config['table_name']
            validation_rules = source_config.get('validation_rules', [])
            
            if not validation_rules:
                # No validation rules defined
                return DataQualityMetric(
                    metric_name='accuracy',
                    current_value=1.0,
                    target_value=1.0,
                    threshold_warning=0.95,
                    threshold_critical=0.85,
                    status=DataQualityStatus.EXCELLENT,
                    last_updated=datetime.now()
                )
            
            total_violations = 0
            total_checks = 0
            
            with SessionLocal() as session:
                # Get total row count for recent data
                count_query = f"""
                    SELECT COUNT(*) as total_rows
                    FROM {table_name}
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """
                
                result = session.execute(text(count_query)).fetchone()
                total_rows = result.total_rows if result else 0
                
                if total_rows == 0:
                    accuracy_score = 0.0
                else:
                    # Check each validation rule
                    for rule in validation_rules:
                        rule_query = f"""
                            SELECT COUNT(*) as violation_count
                            FROM {table_name}
                            WHERE created_at > NOW() - INTERVAL '7 days'
                            AND NOT ({rule})
                        """
                        
                        result = session.execute(text(rule_query)).fetchone()
                        violations = result.violation_count if result else 0
                        
                        total_violations += violations
                        total_checks += total_rows
                    
                    # Calculate accuracy score
                    accuracy_score = 1.0 - (total_violations / total_checks) if total_checks > 0 else 0.0
            
            status = self._calculate_status(accuracy_score)
            
            return DataQualityMetric(
                metric_name='accuracy',
                current_value=accuracy_score,
                target_value=1.0,
                threshold_warning=0.95,
                threshold_critical=0.85,
                status=status,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing accuracy for {source_name}: {e}")
            return DataQualityMetric(
                metric_name='accuracy',
                current_value=0.0,
                target_value=1.0,
                threshold_warning=0.95,
                threshold_critical=0.85,
                status=DataQualityStatus.CRITICAL,
                last_updated=datetime.now()
            )
    
    async def _assess_consistency(self, source_name: str, source_config: Dict) -> DataQualityMetric:
        """Assess data consistency over time"""
        try:
            table_name = source_config['table_name']
            consistency_checks = source_config.get('consistency_checks', [])
            
            consistency_score = 1.0  # Default to perfect consistency
            
            if consistency_checks:
                with SessionLocal() as session:
                    for check in consistency_checks:
                        check_query = f"""
                            {check}
                        """
                        
                        result = session.execute(text(check_query)).fetchone()
                        
                        # Each check should return a score between 0 and 1
                        if result:
                            check_score = float(result[0])
                            consistency_score = min(consistency_score, check_score)
            
            status = self._calculate_status(consistency_score)
            
            return DataQualityMetric(
                metric_name='consistency',
                current_value=consistency_score,
                target_value=1.0,
                threshold_warning=0.9,
                threshold_critical=0.8,
                status=status,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing consistency for {source_name}: {e}")
            return DataQualityMetric(
                metric_name='consistency',
                current_value=0.8,  # Default reasonable score
                target_value=1.0,
                threshold_warning=0.9,
                threshold_critical=0.8,
                status=DataQualityStatus.GOOD,
                last_updated=datetime.now()
            )
    
    async def _assess_reliability(self, source_name: str, source_config: Dict) -> DataQualityMetric:
        """Assess data source reliability based on uptime and error rates"""
        try:
            # Check recent API usage statistics
            cache_key = f"api_reliability:{source_name}"
            reliability_data = await self.cache.get(cache_key)
            
            if reliability_data:
                success_rate = reliability_data.get('success_rate', 1.0)
                uptime = reliability_data.get('uptime', 1.0)
                
                # Combine success rate and uptime
                reliability_score = (success_rate + uptime) / 2
            else:
                # Default to good reliability if no data available
                reliability_score = 0.9
            
            status = self._calculate_status(reliability_score)
            
            return DataQualityMetric(
                metric_name='reliability',
                current_value=reliability_score,
                target_value=1.0,
                threshold_warning=0.95,
                threshold_critical=0.9,
                status=status,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing reliability for {source_name}: {e}")
            return DataQualityMetric(
                metric_name='reliability',
                current_value=0.9,
                target_value=1.0,
                threshold_warning=0.95,
                threshold_critical=0.9,
                status=DataQualityStatus.GOOD,
                last_updated=datetime.now()
            )
    
    def _calculate_status(self, score: float) -> DataQualityStatus:
        """Calculate quality status based on score"""
        if score >= self.thresholds['excellent']:
            return DataQualityStatus.EXCELLENT
        elif score >= self.thresholds['good']:
            return DataQualityStatus.GOOD
        elif score >= self.thresholds['fair']:
            return DataQualityStatus.FAIR
        elif score >= self.thresholds['poor']:
            return DataQualityStatus.POOR
        else:
            return DataQualityStatus.CRITICAL
    
    def _identify_issues(self, metrics: Dict[str, DataQualityMetric]) -> List[str]:
        """Identify data quality issues based on metrics"""
        issues = []
        
        for metric_name, metric in metrics.items():
            if metric.status == DataQualityStatus.CRITICAL:
                issues.append(f"Critical {metric_name} issue: score {metric.current_value:.2f}")
            elif metric.status == DataQualityStatus.POOR:
                issues.append(f"Poor {metric_name}: score {metric.current_value:.2f}")
            elif metric.current_value < metric.threshold_warning:
                issues.append(f"{metric_name.title()} below warning threshold: {metric.current_value:.2f}")
        
        return issues
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, DataQualityMetric],
        source_type: DataSourceType
    ) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        for metric_name, metric in metrics.items():
            if metric.current_value < metric.threshold_warning:
                if metric_name == 'freshness':
                    recommendations.append("Increase data collection frequency or check data pipeline")
                elif metric_name == 'completeness':
                    recommendations.append("Review data sources for missing values and improve data validation")
                elif metric_name == 'accuracy':
                    recommendations.append("Review data validation rules and source data quality")
                elif metric_name == 'consistency':
                    recommendations.append("Implement additional consistency checks and data normalization")
                elif metric_name == 'reliability':
                    recommendations.append("Review API reliability and implement backup data sources")
        
        # Source-specific recommendations
        if source_type == DataSourceType.PRICE_DATA:
            if any(m.current_value < 0.9 for m in metrics.values()):
                recommendations.append("Consider using multiple price data providers for redundancy")
        
        elif source_type == DataSourceType.NEWS_DATA:
            if metrics.get('freshness', DataQualityMetric('', 1.0, 1.0, 0.8, 0.5, DataQualityStatus.EXCELLENT, datetime.now())).current_value < 0.8:
                recommendations.append("Increase news scraping frequency during market hours")
        
        return recommendations
    
    def _update_quality_history(self, source_name: str, score: float):
        """Update historical quality data"""
        if source_name not in self.quality_history:
            self.quality_history[source_name] = []
        
        self.quality_history[source_name].append((datetime.now(), score))
        
        # Keep only last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        self.quality_history[source_name] = [
            (timestamp, score) for timestamp, score in self.quality_history[source_name]
            if timestamp > cutoff_date
        ]
    
    async def _check_quality_alerts(self, quality_assessment: DataSourceQuality):
        """Check if alerts should be triggered based on quality assessment"""
        if not self.alert_manager:
            return
        
        try:
            # Critical quality alert
            if quality_assessment.status == DataQualityStatus.CRITICAL:
                await self.alert_manager.trigger_alert(
                    rule_id='data_quality_critical',
                    symbol=None,
                    triggered_value=quality_assessment.overall_score,
                    custom_data={
                        'source_name': quality_assessment.source_name,
                        'source_type': quality_assessment.source_type.value,
                        'issues': quality_assessment.issues
                    }
                )
            
            # Poor quality warning
            elif quality_assessment.status == DataQualityStatus.POOR:
                await self.alert_manager.trigger_alert(
                    rule_id='data_quality_poor',
                    symbol=None,
                    triggered_value=quality_assessment.overall_score,
                    custom_data={
                        'source_name': quality_assessment.source_name,
                        'source_type': quality_assessment.source_type.value,
                        'issues': quality_assessment.issues
                    }
                )
            
            # Metric-specific alerts
            for metric_name, metric in quality_assessment.metrics.items():
                if metric.current_value < metric.threshold_critical:
                    await self.alert_manager.trigger_alert(
                        rule_id=f'data_quality_metric_{metric_name}',
                        symbol=None,
                        triggered_value=metric.current_value,
                        custom_data={
                            'source_name': quality_assessment.source_name,
                            'metric_name': metric_name,
                            'threshold': metric.threshold_critical
                        }
                    )
        
        except Exception as e:
            logger.error(f"Error checking quality alerts: {e}")
    
    async def get_quality_dashboard_data(self) -> Dict:
        """Get comprehensive data quality dashboard data"""
        try:
            dashboard_data = {
                'overview': {
                    'total_sources': len(self.data_sources),
                    'healthy_sources': 0,
                    'warning_sources': 0,
                    'critical_sources': 0,
                    'last_updated': datetime.now()
                },
                'sources': {},
                'trends': {},
                'alerts': []
            }
            
            # Assess all data sources
            for source_name in self.data_sources.keys():
                try:
                    quality_assessment = await self.assess_data_quality(source_name)
                    dashboard_data['sources'][source_name] = {
                        'overall_score': quality_assessment.overall_score,
                        'status': quality_assessment.status.value,
                        'metrics': {
                            name: {
                                'score': metric.current_value,
                                'status': metric.status.value,
                                'trend_7d': metric.trend_7d
                            } for name, metric in quality_assessment.metrics.items()
                        },
                        'issues': quality_assessment.issues,
                        'recommendations': quality_assessment.recommendations,
                        'last_update': quality_assessment.last_update
                    }
                    
                    # Update overview counts
                    if quality_assessment.status in [DataQualityStatus.EXCELLENT, DataQualityStatus.GOOD]:
                        dashboard_data['overview']['healthy_sources'] += 1
                    elif quality_assessment.status in [DataQualityStatus.FAIR, DataQualityStatus.POOR]:
                        dashboard_data['overview']['warning_sources'] += 1
                    else:
                        dashboard_data['overview']['critical_sources'] += 1
                        
                except Exception as e:
                    logger.error(f"Error assessing {source_name}: {e}")
                    dashboard_data['overview']['critical_sources'] += 1
            
            # Calculate trends
            dashboard_data['trends'] = self._calculate_quality_trends()
            
            # Get recent quality alerts
            if self.alert_manager:
                recent_alerts = self.alert_manager.get_active_alerts(category=AlertCategory.DATA_QUALITY)
                dashboard_data['alerts'] = [
                    {
                        'id': alert.alert_id,
                        'title': alert.title,
                        'severity': alert.severity.value,
                        'created_at': alert.created_at,
                        'acknowledged': alert.acknowledged
                    } for alert in recent_alerts[:10]  # Last 10 alerts
                ]
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _calculate_quality_trends(self) -> Dict:
        """Calculate quality trends for dashboard"""
        trends = {}
        
        for source_name, history in self.quality_history.items():
            if len(history) < 2:
                continue
            
            # Calculate 7-day and 30-day trends
            now = datetime.now()
            
            # 7-day trend
            week_ago = now - timedelta(days=7)
            recent_scores = [score for timestamp, score in history if timestamp > week_ago]
            
            if len(recent_scores) >= 2:
                trend_7d = (recent_scores[-1] - recent_scores[0]) / recent_scores[0] * 100
            else:
                trend_7d = 0
            
            # 30-day trend
            month_ago = now - timedelta(days=30)
            monthly_scores = [score for timestamp, score in history if timestamp > month_ago]
            
            if len(monthly_scores) >= 2:
                trend_30d = (monthly_scores[-1] - monthly_scores[0]) / monthly_scores[0] * 100
            else:
                trend_30d = 0
            
            trends[source_name] = {
                'trend_7d': trend_7d,
                'trend_30d': trend_30d,
                'current_score': history[-1][1] if history else 0,
                'data_points': len(history)
            }
        
        return trends
    
    async def generate_quality_report(self, days: int = 7) -> Dict:
        """Generate comprehensive data quality report"""
        try:
            report = {
                'report_date': datetime.now(),
                'period_days': days,
                'executive_summary': {},
                'source_details': {},
                'recommendations': [],
                'trends': {},
                'sla_compliance': {}
            }
            
            all_scores = []
            critical_issues = []
            
            # Analyze each data source
            for source_name in self.data_sources.keys():
                quality_assessment = await self.assess_data_quality(source_name)
                
                report['source_details'][source_name] = {
                    'overall_score': quality_assessment.overall_score,
                    'status': quality_assessment.status.value,
                    'metrics_breakdown': {
                        name: {
                            'score': metric.current_value,
                            'status': metric.status.value,
                            'target': metric.target_value
                        } for name, metric in quality_assessment.metrics.items()
                    },
                    'issues': quality_assessment.issues,
                    'recommendations': quality_assessment.recommendations
                }
                
                all_scores.append(quality_assessment.overall_score)
                
                if quality_assessment.status == DataQualityStatus.CRITICAL:
                    critical_issues.extend([
                        f"{source_name}: {issue}" for issue in quality_assessment.issues
                    ])
            
            # Executive summary
            if all_scores:
                report['executive_summary'] = {
                    'average_quality_score': np.mean(all_scores),
                    'sources_above_threshold': len([s for s in all_scores if s >= self.thresholds['good']]),
                    'sources_critical': len([s for s in all_scores if s < self.thresholds['poor']]),
                    'critical_issues_count': len(critical_issues),
                    'overall_health': self._calculate_overall_health(all_scores)
                }
            
            # Top recommendations
            all_recommendations = []
            for source_data in report['source_details'].values():
                all_recommendations.extend(source_data['recommendations'])
            
            # Count and prioritize recommendations
            recommendation_counts = {}
            for rec in all_recommendations:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
            
            report['recommendations'] = sorted(
                recommendation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {}
    
    def _calculate_overall_health(self, scores: List[float]) -> str:
        """Calculate overall system health based on all scores"""
        if not scores:
            return "unknown"
        
        avg_score = np.mean(scores)
        
        if avg_score >= self.thresholds['excellent']:
            return "excellent"
        elif avg_score >= self.thresholds['good']:
            return "good"
        elif avg_score >= self.thresholds['fair']:
            return "fair"
        else:
            return "poor"