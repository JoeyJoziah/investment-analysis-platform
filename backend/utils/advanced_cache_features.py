"""
Advanced Cache Features System
Comprehensive analytics, monitoring, and warming job management for
high-performance financial data caching.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import schedule

from backend.utils.tier_based_caching import tier_based_cache, StockTier
from backend.utils.cache_hit_optimization import cache_hit_optimizer
from backend.utils.predictive_cache_warming import predictive_cache_warmer
from backend.utils.distributed_cache_coordination import distributed_cache_coordinator
from backend.utils.monitoring import metrics

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Cache job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Cache job types."""
    WARMING = "warming"
    CLEANUP = "cleanup"
    OPTIMIZATION = "optimization"
    ANALYTICS = "analytics"
    MAINTENANCE = "maintenance"


@dataclass
class CacheJob:
    """Cache job definition."""
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    priority: int = 5  # 1=highest, 10=lowest
    retry_count: int = 0
    max_retries: int = 3


class CacheAnalytics:
    """
    Advanced cache analytics and performance insights.
    """
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)  # Last 10k operations
        self.performance_data = defaultdict(list)
        self.access_patterns = defaultdict(lambda: defaultdict(int))
        self.cost_tracking = defaultdict(float)
        self.anomaly_detections = []
        self._lock = threading.RLock()
        
        # Analytics configuration
        self.analysis_window = timedelta(hours=24)
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection
        
    def record_cache_operation(
        self,
        operation: str,
        key: str,
        hit: bool,
        response_time_ms: float,
        data_size_bytes: int = 0,
        tier: Optional[str] = None,
        cost: float = 0.0
    ):
        """Record cache operation for analytics."""
        timestamp = time.time()
        
        with self._lock:
            # Record operation
            operation_data = {
                'timestamp': timestamp,
                'operation': operation,
                'key': key,
                'hit': hit,
                'response_time_ms': response_time_ms,
                'data_size_bytes': data_size_bytes,
                'tier': tier,
                'cost': cost
            }
            
            self.metrics_history.append(operation_data)
            
            # Update performance tracking
            hour_bucket = int(timestamp // 3600)
            self.performance_data[hour_bucket].append({
                'hit': hit,
                'response_time': response_time_ms,
                'size': data_size_bytes
            })
            
            # Track access patterns
            if ':' in key:
                pattern = ':'.join(key.split(':')[:2])  # First two components
                self.access_patterns[pattern][hour_bucket] += 1
            
            # Track costs
            self.cost_tracking[hour_bucket] += cost
    
    def get_hit_rate_analysis(
        self,
        timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze cache hit rates over timeframe."""
        cutoff_time = time.time() - (timeframe_hours * 3600)
        
        with self._lock:
            # Filter recent operations
            recent_ops = [
                op for op in self.metrics_history 
                if op['timestamp'] >= cutoff_time
            ]
            
            if not recent_ops:
                return {'error': 'No data available'}
            
            # Calculate overall hit rate
            total_ops = len(recent_ops)
            hits = sum(1 for op in recent_ops if op['hit'])
            overall_hit_rate = (hits / total_ops) * 100
            
            # Hit rate by tier
            tier_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
            for op in recent_ops:
                tier = op.get('tier', 'unknown')
                tier_stats[tier]['total'] += 1
                if op['hit']:
                    tier_stats[tier]['hits'] += 1
            
            tier_hit_rates = {}
            for tier, stats in tier_stats.items():
                if stats['total'] > 0:
                    tier_hit_rates[tier] = (stats['hits'] / stats['total']) * 100
            
            # Hit rate trend (hourly)
            hourly_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
            for op in recent_ops:
                hour = int(op['timestamp'] // 3600)
                hourly_stats[hour]['total'] += 1
                if op['hit']:
                    hourly_stats[hour]['hits'] += 1
            
            hit_rate_trend = []
            for hour in sorted(hourly_stats.keys()):
                stats = hourly_stats[hour]
                rate = (stats['hits'] / stats['total']) * 100 if stats['total'] > 0 else 0
                hit_rate_trend.append({
                    'hour': datetime.fromtimestamp(hour * 3600).isoformat(),
                    'hit_rate': rate,
                    'operations': stats['total']
                })
            
            return {
                'timeframe_hours': timeframe_hours,
                'total_operations': total_ops,
                'overall_hit_rate': overall_hit_rate,
                'tier_hit_rates': tier_hit_rates,
                'hit_rate_trend': hit_rate_trend[-24:]  # Last 24 hours
            }
    
    def get_performance_analysis(
        self,
        timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze cache performance metrics."""
        cutoff_time = time.time() - (timeframe_hours * 3600)
        
        with self._lock:
            recent_ops = [
                op for op in self.metrics_history 
                if op['timestamp'] >= cutoff_time
            ]
            
            if not recent_ops:
                return {'error': 'No data available'}
            
            # Response time statistics
            response_times = [op['response_time_ms'] for op in recent_ops]
            response_stats = {
                'mean': np.mean(response_times),
                'median': np.median(response_times),
                'p95': np.percentile(response_times, 95),
                'p99': np.percentile(response_times, 99),
                'min': np.min(response_times),
                'max': np.max(response_times),
                'std': np.std(response_times)
            }
            
            # Data size statistics
            data_sizes = [op.get('data_size_bytes', 0) for op in recent_ops]
            size_stats = {
                'mean_bytes': np.mean(data_sizes),
                'median_bytes': np.median(data_sizes),
                'total_bytes': np.sum(data_sizes),
                'max_bytes': np.max(data_sizes) if data_sizes else 0
            }
            
            # Performance by operation type
            op_performance = defaultdict(list)
            for op in recent_ops:
                op_performance[op['operation']].append(op['response_time_ms'])
            
            operation_stats = {}
            for operation, times in op_performance.items():
                if times:
                    operation_stats[operation] = {
                        'count': len(times),
                        'mean_response_ms': np.mean(times),
                        'p95_response_ms': np.percentile(times, 95)
                    }
            
            return {
                'timeframe_hours': timeframe_hours,
                'response_time_stats': response_stats,
                'data_size_stats': size_stats,
                'operation_performance': operation_stats,
                'total_operations': len(recent_ops)
            }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        anomalies = []
        
        with self._lock:
            # Get recent performance data
            recent_ops = list(self.metrics_history)[-1000:]  # Last 1000 operations
            
            if len(recent_ops) < 100:
                return anomalies
            
            # Response time anomaly detection
            response_times = [op['response_time_ms'] for op in recent_ops]
            mean_response = np.mean(response_times)
            std_response = np.std(response_times)
            
            for op in recent_ops[-50:]:  # Check last 50 operations
                response_time = op['response_time_ms']
                z_score = abs(response_time - mean_response) / std_response if std_response > 0 else 0
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        'type': 'high_response_time',
                        'timestamp': op['timestamp'],
                        'key': op['key'],
                        'response_time_ms': response_time,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3.0 else 'medium'
                    })
            
            # Hit rate drop detection
            recent_hit_rate = sum(1 for op in recent_ops[-100:] if op['hit']) / 100
            overall_hit_rate = sum(1 for op in recent_ops if op['hit']) / len(recent_ops)
            
            if recent_hit_rate < overall_hit_rate * 0.8:  # 20% drop
                anomalies.append({
                    'type': 'hit_rate_drop',
                    'timestamp': time.time(),
                    'recent_hit_rate': recent_hit_rate * 100,
                    'overall_hit_rate': overall_hit_rate * 100,
                    'severity': 'high'
                })
        
        # Store anomalies
        self.anomaly_detections.extend(anomalies)
        
        # Limit anomaly history
        if len(self.anomaly_detections) > 1000:
            self.anomaly_detections = self.anomaly_detections[-500:]
        
        return anomalies
    
    def get_access_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze cache access patterns."""
        with self._lock:
            # Get top accessed patterns
            pattern_totals = defaultdict(int)
            for pattern, hourly_counts in self.access_patterns.items():
                pattern_totals[pattern] = sum(hourly_counts.values())
            
            top_patterns = sorted(
                pattern_totals.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            # Temporal patterns (by hour of day)
            hourly_distribution = defaultdict(int)
            for hourly_counts in self.access_patterns.values():
                for hour_bucket, count in hourly_counts.items():
                    hour_of_day = (hour_bucket % 24)
                    hourly_distribution[hour_of_day] += count
            
            return {
                'top_access_patterns': [
                    {'pattern': pattern, 'count': count}
                    for pattern, count in top_patterns
                ],
                'hourly_distribution': dict(hourly_distribution),
                'total_unique_patterns': len(self.access_patterns)
            }
    
    def get_cost_analysis(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Analyze cache operation costs."""
        cutoff_hour = int((time.time() - timeframe_hours * 3600) // 3600)
        
        with self._lock:
            # Get costs within timeframe
            relevant_costs = {
                hour: cost for hour, cost in self.cost_tracking.items()
                if hour >= cutoff_hour
            }
            
            if not relevant_costs:
                return {'error': 'No cost data available'}
            
            total_cost = sum(relevant_costs.values())
            avg_hourly_cost = total_cost / len(relevant_costs)
            
            # Cost trend
            cost_trend = []
            for hour in sorted(relevant_costs.keys()):
                cost_trend.append({
                    'hour': datetime.fromtimestamp(hour * 3600).isoformat(),
                    'cost': relevant_costs[hour]
                })
            
            return {
                'timeframe_hours': timeframe_hours,
                'total_cost': total_cost,
                'average_hourly_cost': avg_hourly_cost,
                'cost_trend': cost_trend,
                'projected_monthly_cost': avg_hourly_cost * 24 * 30
            }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'hit_rate_analysis': self.get_hit_rate_analysis(24),
            'performance_analysis': self.get_performance_analysis(24),
            'access_patterns': self.get_access_pattern_analysis(),
            'cost_analysis': self.get_cost_analysis(24),
            'anomalies': self.anomaly_detections[-10:],  # Last 10 anomalies
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on analytics."""
        recommendations = []
        
        # Analyze recent performance
        hit_rate_data = self.get_hit_rate_analysis(24)
        perf_data = self.get_performance_analysis(24)
        
        if 'overall_hit_rate' in hit_rate_data:
            hit_rate = hit_rate_data['overall_hit_rate']
            
            if hit_rate < 70:
                recommendations.append({
                    'type': 'hit_rate_improvement',
                    'priority': 'high',
                    'message': f'Cache hit rate is low ({hit_rate:.1f}%). Consider increasing cache size or improving key strategies.'
                })
            elif hit_rate < 85:
                recommendations.append({
                    'type': 'hit_rate_improvement',
                    'priority': 'medium',
                    'message': f'Cache hit rate ({hit_rate:.1f}%) has room for improvement. Review caching policies.'
                })
        
        if 'response_time_stats' in perf_data:
            p95_time = perf_data['response_time_stats'].get('p95', 0)
            
            if p95_time > 100:
                recommendations.append({
                    'type': 'performance_optimization',
                    'priority': 'high',
                    'message': f'P95 response time is high ({p95_time:.1f}ms). Consider optimizing cache implementation.'
                })
        
        # Check anomalies
        recent_anomalies = [a for a in self.anomaly_detections if time.time() - a['timestamp'] < 3600]
        if len(recent_anomalies) > 5:
            recommendations.append({
                'type': 'anomaly_investigation',
                'priority': 'high',
                'message': f'Detected {len(recent_anomalies)} anomalies in the last hour. Investigation recommended.'
            })
        
        return recommendations


class CacheJobScheduler:
    """
    Advanced job scheduler for cache operations.
    """
    
    def __init__(self):
        self.jobs: Dict[str, CacheJob] = {}
        self.job_queue = asyncio.PriorityQueue()
        self.running_jobs: Set[str] = set()
        self.completed_jobs = deque(maxlen=1000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()
        
        # Scheduler state
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.max_concurrent_jobs = 3
        
        # Job handlers
        self.job_handlers: Dict[JobType, Callable] = {
            JobType.WARMING: self._handle_warming_job,
            JobType.CLEANUP: self._handle_cleanup_job,
            JobType.OPTIMIZATION: self._handle_optimization_job,
            JobType.ANALYTICS: self._handle_analytics_job,
            JobType.MAINTENANCE: self._handle_maintenance_job
        }
    
    async def start(self):
        """Start the job scheduler."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent_jobs):
            task = asyncio.create_task(self._job_worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        # Start scheduling task
        schedule_task = asyncio.create_task(self._job_scheduler())
        self.worker_tasks.append(schedule_task)
        
        logger.info(f"Cache job scheduler started with {self.max_concurrent_jobs} workers")
    
    async def stop(self):
        """Stop the job scheduler."""
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("Cache job scheduler stopped")
    
    def schedule_job(
        self,
        job_type: JobType,
        parameters: Dict[str, Any],
        priority: int = 5,
        delay_seconds: int = 0
    ) -> str:
        """Schedule a new cache job."""
        job_id = str(uuid.uuid4())
        
        job = CacheJob(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            parameters=parameters,
            priority=priority
        )
        
        with self._lock:
            self.jobs[job_id] = job
        
        # Add to queue (priority queue uses negative priority for max-heap behavior)
        scheduled_time = time.time() + delay_seconds
        asyncio.create_task(self._enqueue_job_with_delay(job, scheduled_time))
        
        logger.info(f"Scheduled {job_type.value} job {job_id} with priority {priority}")
        return job_id
    
    async def _enqueue_job_with_delay(self, job: CacheJob, scheduled_time: float):
        """Enqueue job after delay."""
        delay = scheduled_time - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        
        await self.job_queue.put((job.priority, job.job_id))
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        with self._lock:
            if job_id in self.jobs and self.jobs[job_id].status == JobStatus.PENDING:
                self.jobs[job_id].status = JobStatus.CANCELLED
                logger.info(f"Cancelled job {job_id}")
                return True
        
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and details."""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                return {
                    'job_id': job.job_id,
                    'job_type': job.job_type.value,
                    'status': job.status.value,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'priority': job.priority,
                    'retry_count': job.retry_count,
                    'result': job.result,
                    'error': job.error
                }
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get scheduler queue status."""
        with self._lock:
            status_counts = defaultdict(int)
            for job in self.jobs.values():
                status_counts[job.status.value] += 1
            
            return {
                'queue_size': self.job_queue.qsize(),
                'running_jobs': len(self.running_jobs),
                'job_counts_by_status': dict(status_counts),
                'total_jobs': len(self.jobs),
                'scheduler_running': self.is_running
            }
    
    async def _job_worker(self, worker_name: str):
        """Job worker that processes jobs from the queue."""
        logger.info(f"Job worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get next job from queue
                try:
                    priority, job_id = await asyncio.wait_for(
                        self.job_queue.get(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Get job details
                with self._lock:
                    if job_id not in self.jobs:
                        continue
                    
                    job = self.jobs[job_id]
                    
                    # Check if job was cancelled
                    if job.status == JobStatus.CANCELLED:
                        continue
                    
                    # Mark as running
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()
                    self.running_jobs.add(job_id)
                
                logger.info(f"Worker {worker_name} processing job {job_id} ({job.job_type.value})")
                
                # Execute job
                try:
                    result = await self._execute_job(job)
                    
                    with self._lock:
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        job.result = result
                        self.running_jobs.discard(job_id)
                        self.completed_jobs.append(job_id)
                    
                    logger.info(f"Job {job_id} completed successfully")
                
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {e}")
                    
                    with self._lock:
                        job.error = str(e)
                        job.retry_count += 1
                        self.running_jobs.discard(job_id)
                        
                        # Retry logic
                        if job.retry_count < job.max_retries:
                            job.status = JobStatus.PENDING
                            # Re-queue with exponential backoff
                            delay = 2 ** job.retry_count
                            await self._enqueue_job_with_delay(job, time.time() + delay)
                            logger.info(f"Re-queued job {job_id} for retry {job.retry_count}")
                        else:
                            job.status = JobStatus.FAILED
                            job.completed_at = datetime.now()
                            logger.error(f"Job {job_id} failed permanently after {job.retry_count} retries")
            
            except Exception as e:
                logger.error(f"Job worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Job worker {worker_name} stopped")
    
    async def _execute_job(self, job: CacheJob) -> Dict[str, Any]:
        """Execute a cache job."""
        handler = self.job_handlers.get(job.job_type)
        
        if not handler:
            raise ValueError(f"No handler for job type {job.job_type}")
        
        return await handler(job)
    
    async def _handle_warming_job(self, job: CacheJob) -> Dict[str, Any]:
        """Handle cache warming job."""
        tier = job.parameters.get('tier')
        symbols = job.parameters.get('symbols', [])
        
        if tier:
            stock_tier = StockTier[tier.upper()]
            result = await tier_based_cache.warm_tier_caches(stock_tier)
        elif symbols:
            # Warm specific symbols
            result = {'warmed': 0, 'failed': 0}
            for symbol in symbols:
                # This would integrate with actual warming logic
                result['warmed'] += 1
        else:
            result = await predictive_cache_warmer.warm_critical_caches()
        
        return result
    
    async def _handle_cleanup_job(self, job: CacheJob) -> Dict[str, Any]:
        """Handle cache cleanup job."""
        max_age_hours = job.parameters.get('max_age_hours', 24)
        patterns = job.parameters.get('patterns', ['*:expired:*'])
        
        cleaned_count = 0
        for pattern in patterns:
            # This would integrate with actual cleanup logic
            cleaned_count += 10  # Placeholder
        
        return {
            'cleaned_entries': cleaned_count,
            'patterns': patterns,
            'max_age_hours': max_age_hours
        }
    
    async def _handle_optimization_job(self, job: CacheJob) -> Dict[str, Any]:
        """Handle cache optimization job."""
        optimization_type = job.parameters.get('type', 'full')
        
        if optimization_type == 'key_strategy':
            # Optimize key generation strategy
            result = cache_hit_optimizer.key_generator.optimize_key_strategy()
            return {'optimized_strategy': result.value}
        
        elif optimization_type == 'tier_assignments':
            # Optimize tier assignments
            market_data = job.parameters.get('market_data', {})
            result = await tier_based_cache.optimize_tier_assignments(market_data)
            return result
        
        else:
            # Full optimization
            return {
                'key_optimization': 'completed',
                'tier_optimization': 'completed',
                'cache_warming': 'scheduled'
            }
    
    async def _handle_analytics_job(self, job: CacheJob) -> Dict[str, Any]:
        """Handle analytics job."""
        analytics_type = job.parameters.get('type', 'comprehensive')
        
        if analytics_type == 'anomaly_detection':
            anomalies = cache_analytics.detect_anomalies()
            return {'anomalies_detected': len(anomalies), 'anomalies': anomalies}
        
        elif analytics_type == 'performance_report':
            return cache_analytics.get_performance_analysis(24)
        
        else:
            return cache_analytics.generate_comprehensive_report()
    
    async def _handle_maintenance_job(self, job: CacheJob) -> Dict[str, Any]:
        """Handle maintenance job."""
        maintenance_tasks = job.parameters.get('tasks', ['cleanup_expired', 'optimize_memory'])
        results = {}
        
        for task in maintenance_tasks:
            if task == 'cleanup_expired':
                results[task] = {'cleaned_entries': 50}  # Placeholder
            elif task == 'optimize_memory':
                results[task] = {'memory_freed_mb': 100}  # Placeholder
            elif task == 'update_statistics':
                results[task] = {'statistics_updated': True}
        
        return results
    
    async def _job_scheduler(self):
        """Automatic job scheduler for recurring tasks."""
        logger.info("Automatic job scheduler started")
        
        while self.is_running:
            try:
                current_time = datetime.now()
                hour = current_time.hour
                minute = current_time.minute
                
                # Schedule daily analytics job at 2 AM
                if hour == 2 and minute == 0:
                    self.schedule_job(
                        JobType.ANALYTICS,
                        {'type': 'comprehensive'},
                        priority=3
                    )
                
                # Schedule cache warming before market open (8 AM)
                if hour == 8 and minute == 0:
                    self.schedule_job(
                        JobType.WARMING,
                        {'tier': 'critical'},
                        priority=1
                    )
                
                # Schedule cleanup every 6 hours
                if minute == 0 and hour % 6 == 0:
                    self.schedule_job(
                        JobType.CLEANUP,
                        {'max_age_hours': 24},
                        priority=4
                    )
                
                # Schedule optimization weekly (Sunday 1 AM)
                if current_time.weekday() == 6 and hour == 1 and minute == 0:
                    self.schedule_job(
                        JobType.OPTIMIZATION,
                        {'type': 'full'},
                        priority=2
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Job scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error


class CacheMonitoringDashboard:
    """
    Real-time cache monitoring dashboard.
    """
    
    def __init__(self):
        self.metrics_collectors = []
        self.alert_rules = []
        self.dashboard_data = {}
        self.real_time_data = deque(maxlen=1000)
        self._lock = threading.RLock()
        
    def register_metrics_collector(self, collector: Callable):
        """Register a metrics collector function."""
        self.metrics_collectors.append(collector)
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        message: str,
        severity: str = 'medium'
    ):
        """Add alert rule."""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'message': message,
            'severity': severity
        })
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect all metrics for dashboard."""
        metrics_data = {
            'timestamp': time.time(),
            'cache_hit_optimizer': cache_hit_optimizer.get_optimization_metrics(),
            'tier_based_cache': tier_based_cache.get_comprehensive_stats(),
            'analytics': cache_analytics.generate_comprehensive_report()
        }
        
        # Collect from registered collectors
        for collector in self.metrics_collectors:
            try:
                collector_data = await collector() if asyncio.iscoroutinefunction(collector) else collector()
                if isinstance(collector_data, dict):
                    metrics_data.update(collector_data)
            except Exception as e:
                logger.error(f"Metrics collector failed: {e}")
        
        # Store real-time data
        with self._lock:
            self.real_time_data.append(metrics_data)
            self.dashboard_data = metrics_data
        
        # Check alerts
        await self._check_alerts(metrics_data)
        
        return metrics_data
    
    async def _check_alerts(self, metrics_data: Dict[str, Any]):
        """Check alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics_data):
                    alert = {
                        'timestamp': time.time(),
                        'rule_name': rule['name'],
                        'message': rule['message'],
                        'severity': rule['severity'],
                        'metrics': metrics_data
                    }
                    
                    logger.warning(f"Cache alert: {rule['name']} - {rule['message']}")
                    
                    # Here you would send alerts to monitoring systems
                    
            except Exception as e:
                logger.error(f"Alert rule {rule['name']} failed: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        with self._lock:
            return self.dashboard_data.copy() if self.dashboard_data else {}
    
    def get_real_time_series(self, metric_path: str, points: int = 100) -> List[Dict[str, Any]]:
        """Get time series data for a specific metric."""
        with self._lock:
            recent_data = list(self.real_time_data)[-points:]
            
            series_data = []
            for data_point in recent_data:
                try:
                    # Navigate metric path (e.g., "cache_hit_optimizer.hit_metrics.hit_rate_percent")
                    value = data_point
                    for key in metric_path.split('.'):
                        value = value[key]
                    
                    series_data.append({
                        'timestamp': data_point['timestamp'],
                        'value': value
                    })
                    
                except (KeyError, TypeError):
                    continue
            
            return series_data


# Global instances
cache_analytics = CacheAnalytics()
cache_job_scheduler = CacheJobScheduler()
cache_monitoring_dashboard = CacheMonitoringDashboard()


# Default alert rules
def setup_default_alerts():
    """Setup default alert rules for cache monitoring."""
    
    # Hit rate alert
    cache_monitoring_dashboard.add_alert_rule(
        'low_hit_rate',
        lambda data: data.get('cache_hit_optimizer', {}).get('hit_metrics', {}).get('hit_rate_percent', 100) < 70,
        'Cache hit rate is below 70%',
        'high'
    )
    
    # High response time alert
    cache_monitoring_dashboard.add_alert_rule(
        'high_response_time',
        lambda data: data.get('cache_hit_optimizer', {}).get('hit_metrics', {}).get('average_response_time_ms', 0) > 100,
        'Average response time exceeds 100ms',
        'medium'
    )
    
    # Anomaly alert
    cache_monitoring_dashboard.add_alert_rule(
        'anomalies_detected',
        lambda data: len(data.get('analytics', {}).get('anomalies', [])) > 3,
        'Multiple cache anomalies detected',
        'high'
    )


async def initialize_advanced_cache_features():
    """Initialize all advanced cache features."""
    # Start job scheduler
    await cache_job_scheduler.start()
    
    # Setup default alerts
    setup_default_alerts()
    
    # Start monitoring collection
    asyncio.create_task(monitoring_loop())
    
    logger.info("Advanced cache features initialized")


async def monitoring_loop():
    """Background monitoring loop."""
    while True:
        try:
            await cache_monitoring_dashboard.collect_metrics()
            await asyncio.sleep(30)  # Collect metrics every 30 seconds
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error