"""
Chaos Engineering Features for Resilience Testing
Automated failure injection and resilience validation for the investment analysis system
"""

import asyncio
import time
import random
import json
import psutil
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import logging
from pathlib import Path
import aiofiles
import subprocess
import signal
import socket
import httpx
from contextlib import asynccontextmanager

from .enhanced_error_handling import with_error_handling, error_handler
from .service_health_manager import ServiceHealthManager, HealthStatus
from .advanced_circuit_breaker import EnhancedCircuitBreaker
from .exceptions import *

logger = logging.getLogger(__name__)


class ChaosExperimentType(Enum):
    """Types of chaos experiments"""
    LATENCY_INJECTION = "latency_injection"
    NETWORK_PARTITION = "network_partition"
    SERVICE_FAILURE = "service_failure"
    DATABASE_FAILURE = "database_failure"
    CACHE_FAILURE = "cache_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DISK_FILL = "disk_fill"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    DEPENDENCY_FAILURE = "dependency_failure"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION_DRIFT = "configuration_drift"
    TIME_DRIFT = "time_drift"


class ExperimentStatus(Enum):
    """Experiment execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    PAUSED = "paused"


class ImpactScope(Enum):
    """Scope of chaos experiment impact"""
    SINGLE_SERVICE = "single_service"
    SERVICE_GROUP = "service_group"
    DEPENDENCY = "dependency"
    INFRASTRUCTURE = "infrastructure"
    NETWORK = "network"
    DATA_LAYER = "data_layer"


@dataclass
class ChaosExperiment:
    """Configuration for a chaos experiment"""
    experiment_id: str
    experiment_type: ChaosExperimentType
    name: str
    description: str
    impact_scope: ImpactScope
    target_services: List[str]
    duration_seconds: int
    intensity: float  # 0.0-1.0 scale
    parameters: Dict[str, Any]
    safety_checks: List[str]
    success_criteria: List[str]
    rollback_strategy: str
    created_at: datetime
    created_by: str
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ExperimentResult:
    """Results of a chaos experiment execution"""
    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    impact_metrics: Dict[str, Any]
    system_health_before: Dict[str, Any]
    system_health_after: Dict[str, Any]
    observations: List[str]
    failures_detected: List[str]
    recovery_time_seconds: float
    success_criteria_met: bool
    lessons_learned: List[str]
    recommendations: List[str]


class SafetyValidator:
    """Validates safety conditions before and during chaos experiments"""
    
    def __init__(self, health_manager: ServiceHealthManager):
        self.health_manager = health_manager
        self.safety_rules: Dict[str, Callable] = {}
        self.emergency_stop_conditions: List[Callable] = []
        
        # Register default safety rules
        self._register_default_safety_rules()
    
    def _register_default_safety_rules(self):
        """Register default safety validation rules"""
        
        self.safety_rules.update({
            'system_healthy': self._check_system_healthy,
            'no_ongoing_incidents': self._check_no_ongoing_incidents,
            'low_error_rate': self._check_low_error_rate,
            'adequate_resources': self._check_adequate_resources,
            'business_hours_only': self._check_business_hours,
            'no_critical_operations': self._check_no_critical_operations,
            'backup_systems_ready': self._check_backup_systems_ready
        })
        
        # Emergency stop conditions
        self.emergency_stop_conditions.extend([
            self._check_critical_service_down,
            self._check_excessive_error_rate,
            self._check_data_corruption_detected,
            self._check_security_breach
        ])
    
    async def validate_safety_preconditions(self, experiment: ChaosExperiment) -> tuple[bool, List[str]]:
        """Validate safety preconditions before starting experiment"""
        violations = []
        
        for safety_check in experiment.safety_checks:
            if safety_check in self.safety_rules:
                try:
                    is_safe = await self.safety_rules[safety_check](experiment)
                    if not is_safe:
                        violations.append(f"Safety check failed: {safety_check}")
                except Exception as e:
                    violations.append(f"Safety check error for {safety_check}: {e}")
            else:
                violations.append(f"Unknown safety check: {safety_check}")
        
        return len(violations) == 0, violations
    
    async def check_emergency_stop_conditions(self, experiment: ChaosExperiment) -> tuple[bool, List[str]]:
        """Check if emergency stop is required"""
        stop_reasons = []
        
        for condition_check in self.emergency_stop_conditions:
            try:
                should_stop, reason = await condition_check(experiment)
                if should_stop:
                    stop_reasons.append(reason)
            except Exception as e:
                stop_reasons.append(f"Emergency check error: {e}")
        
        return len(stop_reasons) > 0, stop_reasons
    
    # Default safety rule implementations
    async def _check_system_healthy(self, experiment: ChaosExperiment) -> bool:
        """Check if system is in healthy state"""
        health_status = self.health_manager.get_health_status()
        return health_status['overall_status'] in ['healthy', 'degraded']
    
    async def _check_no_ongoing_incidents(self, experiment: ChaosExperiment) -> bool:
        """Check if there are no ongoing incidents"""
        # This would integrate with incident management system
        return True  # Simplified for demo
    
    async def _check_low_error_rate(self, experiment: ChaosExperiment) -> bool:
        """Check if current error rate is acceptably low"""
        health_status = self.health_manager.get_health_status()
        current_metrics = health_status.get('current_metrics')
        
        if current_metrics:
            error_rate = current_metrics.get('error_rate', 0)
            return error_rate < 0.05  # Less than 5% error rate
        
        return True
    
    async def _check_adequate_resources(self, experiment: ChaosExperiment) -> bool:
        """Check if system has adequate resources"""
        health_status = self.health_manager.get_health_status()
        resource_metrics = health_status.get('resource_metrics')
        
        if resource_metrics:
            cpu_usage = resource_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = resource_metrics.get('memory', {}).get('percent', 0)
            
            return cpu_usage < 70 and memory_usage < 80
        
        return True
    
    async def _check_business_hours(self, experiment: ChaosExperiment) -> bool:
        """Check if experiment is running during safe hours"""
        current_hour = datetime.now().hour
        # Only allow experiments during off-peak hours (2 AM - 6 AM)
        return 2 <= current_hour <= 6
    
    async def _check_no_critical_operations(self, experiment: ChaosExperiment) -> bool:
        """Check if no critical operations are running"""
        # This would check for critical batch jobs, deployments, etc.
        return True  # Simplified for demo
    
    async def _check_backup_systems_ready(self, experiment: ChaosExperiment) -> bool:
        """Check if backup systems are ready for failover"""
        # This would verify backup systems are operational
        return True  # Simplified for demo
    
    # Emergency stop condition implementations
    async def _check_critical_service_down(self, experiment: ChaosExperiment) -> tuple[bool, str]:
        """Check if critical services are down"""
        health_status = self.health_manager.get_health_status()
        
        if health_status['overall_status'] == 'critical':
            return True, "Critical service failure detected"
        
        return False, ""
    
    async def _check_excessive_error_rate(self, experiment: ChaosExperiment) -> tuple[bool, str]:
        """Check for excessive error rate"""
        health_status = self.health_manager.get_health_status()
        current_metrics = health_status.get('current_metrics')
        
        if current_metrics:
            error_rate = current_metrics.get('error_rate', 0)
            if error_rate > 0.2:  # More than 20% error rate
                return True, f"Excessive error rate: {error_rate:.1%}"
        
        return False, ""
    
    async def _check_data_corruption_detected(self, experiment: ChaosExperiment) -> tuple[bool, str]:
        """Check for data corruption"""
        # This would integrate with data quality monitoring
        return False, ""  # Simplified for demo
    
    async def _check_security_breach(self, experiment: ChaosExperiment) -> tuple[bool, str]:
        """Check for security breaches"""
        # This would integrate with security monitoring
        return False, ""  # Simplified for demo


class FaultInjector:
    """Injects various types of faults into the system"""
    
    def __init__(self):
        self.active_injections: Dict[str, Dict] = {}
        self.injection_handlers: Dict[ChaosExperimentType, Callable] = {}
        
        # Register fault injection handlers
        self._register_injection_handlers()
    
    def _register_injection_handlers(self):
        """Register fault injection handlers for different experiment types"""
        self.injection_handlers.update({
            ChaosExperimentType.LATENCY_INJECTION: self._inject_latency,
            ChaosExperimentType.NETWORK_PARTITION: self._inject_network_partition,
            ChaosExperimentType.SERVICE_FAILURE: self._inject_service_failure,
            ChaosExperimentType.DATABASE_FAILURE: self._inject_database_failure,
            ChaosExperimentType.CACHE_FAILURE: self._inject_cache_failure,
            ChaosExperimentType.RESOURCE_EXHAUSTION: self._inject_resource_exhaustion,
            ChaosExperimentType.DISK_FILL: self._inject_disk_fill,
            ChaosExperimentType.MEMORY_PRESSURE: self._inject_memory_pressure,
            ChaosExperimentType.CPU_SPIKE: self._inject_cpu_spike,
            ChaosExperimentType.DEPENDENCY_FAILURE: self._inject_dependency_failure,
            ChaosExperimentType.DATA_CORRUPTION: self._inject_data_corruption
        })
    
    async def inject_fault(self, experiment: ChaosExperiment) -> str:
        """Inject fault based on experiment configuration"""
        injection_id = f"injection_{experiment.experiment_id}_{int(time.time())}"
        
        if experiment.experiment_type not in self.injection_handlers:
            raise ValueError(f"No handler for experiment type: {experiment.experiment_type}")
        
        handler = self.injection_handlers[experiment.experiment_type]
        
        try:
            injection_context = await handler(experiment)
            
            self.active_injections[injection_id] = {
                'experiment_id': experiment.experiment_id,
                'experiment_type': experiment.experiment_type,
                'context': injection_context,
                'start_time': datetime.now(),
                'target_services': experiment.target_services
            }
            
            logger.warning(f"Fault injected: {injection_id} for {experiment.experiment_type.value}")
            return injection_id
            
        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
            raise
    
    async def remove_fault(self, injection_id: str):
        """Remove injected fault"""
        if injection_id not in self.active_injections:
            logger.warning(f"Injection not found: {injection_id}")
            return
        
        injection = self.active_injections[injection_id]
        experiment_type = injection['experiment_type']
        context = injection['context']
        
        try:
            # Call cleanup function based on experiment type
            cleanup_method = f"_cleanup_{experiment_type.value}"
            if hasattr(self, cleanup_method):
                cleanup_func = getattr(self, cleanup_method)
                await cleanup_func(context)
            
            del self.active_injections[injection_id]
            logger.info(f"Fault removed: {injection_id}")
            
        except Exception as e:
            logger.error(f"Fault cleanup failed for {injection_id}: {e}")
            raise
    
    # Fault injection implementations
    async def _inject_latency(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject network latency"""
        delay_ms = experiment.parameters.get('delay_ms', 1000)
        target_hosts = experiment.parameters.get('target_hosts', [])
        
        logger.info(f"Injecting {delay_ms}ms latency to {target_hosts}")
        
        # This would use tc (traffic control) on Linux to inject latency
        # For demo purposes, we'll simulate the injection
        context = {
            'type': 'latency',
            'delay_ms': delay_ms,
            'target_hosts': target_hosts,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_network_partition(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Create network partition"""
        partition_groups = experiment.parameters.get('partition_groups', [])
        
        logger.warning(f"Creating network partition between groups: {partition_groups}")
        
        # This would use iptables or similar to block network traffic
        context = {
            'type': 'network_partition',
            'partition_groups': partition_groups,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_service_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate service failure"""
        failure_mode = experiment.parameters.get('failure_mode', 'stop')
        target_services = experiment.target_services
        
        logger.warning(f"Injecting service failure ({failure_mode}) for: {target_services}")
        
        if failure_mode == 'stop':
            # Simulate stopping services
            for service in target_services:
                logger.info(f"Simulating stop of service: {service}")
        elif failure_mode == 'kill':
            # Simulate killing processes
            for service in target_services:
                logger.info(f"Simulating kill of service: {service}")
        
        context = {
            'type': 'service_failure',
            'failure_mode': failure_mode,
            'target_services': target_services,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_database_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate database failure"""
        failure_type = experiment.parameters.get('failure_type', 'connection_drop')
        
        logger.warning(f"Injecting database failure: {failure_type}")
        
        # This would simulate various database failures
        context = {
            'type': 'database_failure',
            'failure_type': failure_type,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_cache_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate cache failure"""
        failure_type = experiment.parameters.get('failure_type', 'connection_drop')
        
        logger.warning(f"Injecting cache failure: {failure_type}")
        
        context = {
            'type': 'cache_failure',
            'failure_type': failure_type,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_resource_exhaustion(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate resource exhaustion"""
        resource_type = experiment.parameters.get('resource_type', 'memory')
        intensity = experiment.intensity
        
        logger.warning(f"Injecting {resource_type} exhaustion at {intensity:.1%} intensity")
        
        if resource_type == 'memory':
            # Allocate memory to simulate pressure
            allocated_mb = int(1000 * intensity)  # Up to 1GB
            memory_hog = bytearray(allocated_mb * 1024 * 1024)
        
        context = {
            'type': 'resource_exhaustion',
            'resource_type': resource_type,
            'intensity': intensity,
            'injected_at': datetime.now(),
            'allocated_memory': memory_hog if resource_type == 'memory' else None
        }
        
        return context
    
    async def _inject_disk_fill(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Fill disk space to simulate disk full"""
        target_path = experiment.parameters.get('target_path', '/tmp')
        fill_percentage = experiment.parameters.get('fill_percentage', 90)
        
        logger.warning(f"Filling disk at {target_path} to {fill_percentage}%")
        
        # Calculate how much space to fill
        disk_usage = psutil.disk_usage(target_path)
        target_used = disk_usage.total * (fill_percentage / 100)
        additional_space = max(0, target_used - disk_usage.used)
        
        # Create temporary file to fill space
        temp_file = Path(target_path) / f"chaos_disk_fill_{int(time.time())}.tmp"
        
        try:
            with open(temp_file, 'wb') as f:
                # Write in chunks to avoid memory issues
                chunk_size = 1024 * 1024  # 1MB chunks
                written = 0
                while written < additional_space:
                    remaining = min(chunk_size, additional_space - written)
                    f.write(b'0' * remaining)
                    written += remaining
        except Exception as e:
            logger.error(f"Disk fill injection failed: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
        
        context = {
            'type': 'disk_fill',
            'temp_file': str(temp_file),
            'filled_bytes': additional_space,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_memory_pressure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Create memory pressure"""
        pressure_mb = experiment.parameters.get('pressure_mb', 512)
        
        logger.warning(f"Creating memory pressure: {pressure_mb}MB")
        
        # Allocate memory to create pressure
        memory_hog = bytearray(pressure_mb * 1024 * 1024)
        
        context = {
            'type': 'memory_pressure',
            'pressure_mb': pressure_mb,
            'memory_hog': memory_hog,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_cpu_spike(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Create CPU spike"""
        cpu_percentage = experiment.parameters.get('cpu_percentage', 80)
        duration = experiment.duration_seconds
        
        logger.warning(f"Creating CPU spike: {cpu_percentage}% for {duration}s")
        
        # Start CPU intensive task
        cpu_task = asyncio.create_task(self._cpu_intensive_task(cpu_percentage, duration))
        
        context = {
            'type': 'cpu_spike',
            'cpu_percentage': cpu_percentage,
            'cpu_task': cpu_task,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_dependency_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate external dependency failure"""
        dependency_name = experiment.parameters.get('dependency_name')
        failure_rate = experiment.parameters.get('failure_rate', 1.0)  # 100% failure
        
        logger.warning(f"Injecting dependency failure for {dependency_name}: {failure_rate:.1%} failure rate")
        
        # This would integrate with circuit breakers to simulate failures
        context = {
            'type': 'dependency_failure',
            'dependency_name': dependency_name,
            'failure_rate': failure_rate,
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _inject_data_corruption(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate data corruption"""
        corruption_type = experiment.parameters.get('corruption_type', 'bit_flip')
        target_files = experiment.parameters.get('target_files', [])
        
        logger.warning(f"Injecting data corruption ({corruption_type}) in: {target_files}")
        
        # This would corrupt specific data files or database records
        context = {
            'type': 'data_corruption',
            'corruption_type': corruption_type,
            'target_files': target_files,
            'backup_files': [],  # Would store backups for cleanup
            'injected_at': datetime.now()
        }
        
        return context
    
    async def _cpu_intensive_task(self, cpu_percentage: int, duration: int):
        """CPU intensive task for CPU spike simulation"""
        end_time = time.time() + duration
        target_load = cpu_percentage / 100.0
        
        while time.time() < end_time:
            start = time.time()
            
            # CPU intensive operation
            count = 0
            while count < 100000:
                count += 1
            
            elapsed = time.time() - start
            sleep_time = elapsed * (1 - target_load) / target_load
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    # Cleanup methods
    async def _cleanup_latency_injection(self, context: Dict[str, Any]):
        """Cleanup latency injection"""
        logger.info("Cleaning up latency injection")
        # Remove tc rules
    
    async def _cleanup_network_partition(self, context: Dict[str, Any]):
        """Cleanup network partition"""
        logger.info("Cleaning up network partition")
        # Remove iptables rules
    
    async def _cleanup_service_failure(self, context: Dict[str, Any]):
        """Cleanup service failure"""
        logger.info("Cleaning up service failure")
        # Restart stopped services
    
    async def _cleanup_disk_fill(self, context: Dict[str, Any]):
        """Cleanup disk fill"""
        temp_file = Path(context['temp_file'])
        if temp_file.exists():
            temp_file.unlink()
            logger.info(f"Cleaned up disk fill file: {temp_file}")
    
    async def _cleanup_memory_pressure(self, context: Dict[str, Any]):
        """Cleanup memory pressure"""
        logger.info("Cleaning up memory pressure")
        # Memory will be freed when context is deleted
    
    async def _cleanup_cpu_spike(self, context: Dict[str, Any]):
        """Cleanup CPU spike"""
        cpu_task = context.get('cpu_task')
        if cpu_task and not cpu_task.done():
            cpu_task.cancel()
            try:
                await cpu_task
            except asyncio.CancelledError:
                pass
        logger.info("Cleaned up CPU spike")


class ExperimentObserver:
    """Observes system behavior during chaos experiments"""
    
    def __init__(self, health_manager: ServiceHealthManager):
        self.health_manager = health_manager
        self.observations: deque = deque(maxlen=10000)
        self.metrics_collectors: List[Callable] = []
        
        # Register default metrics collectors
        self._register_default_collectors()
    
    def _register_default_collectors(self):
        """Register default metrics collection functions"""
        self.metrics_collectors.extend([
            self._collect_health_metrics,
            self._collect_performance_metrics,
            self._collect_error_metrics,
            self._collect_resource_metrics
        ])
    
    async def start_observation(self, experiment: ChaosExperiment) -> str:
        """Start observing system during experiment"""
        observation_id = f"obs_{experiment.experiment_id}"
        
        # Collect baseline metrics
        baseline_metrics = await self._collect_all_metrics()
        
        observation_context = {
            'observation_id': observation_id,
            'experiment_id': experiment.experiment_id,
            'start_time': datetime.now(),
            'baseline_metrics': baseline_metrics,
            'timeline_metrics': [],
            'anomalies_detected': [],
            'recovery_events': []
        }
        
        # Start continuous monitoring
        monitor_task = asyncio.create_task(
            self._continuous_monitoring(observation_context, experiment.duration_seconds)
        )
        
        observation_context['monitor_task'] = monitor_task
        self.observations.append(observation_context)
        
        logger.info(f"Started observation: {observation_id}")
        return observation_id
    
    async def stop_observation(self, observation_id: str) -> Dict[str, Any]:
        """Stop observation and return results"""
        observation = None
        for obs in self.observations:
            if obs['observation_id'] == observation_id:
                observation = obs
                break
        
        if not observation:
            raise ValueError(f"Observation not found: {observation_id}")
        
        # Stop monitoring task
        monitor_task = observation.get('monitor_task')
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Collect final metrics
        final_metrics = await self._collect_all_metrics()
        observation['end_time'] = datetime.now()
        observation['final_metrics'] = final_metrics
        
        # Analyze results
        analysis = await self._analyze_observation_results(observation)
        observation['analysis'] = analysis
        
        logger.info(f"Stopped observation: {observation_id}")
        return observation
    
    async def _continuous_monitoring(self, observation_context: Dict, duration: int):
        """Continuously monitor system during experiment"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                # Collect current metrics
                current_metrics = await self._collect_all_metrics()
                
                # Store timestamped metrics
                observation_context['timeline_metrics'].append({
                    'timestamp': datetime.now(),
                    'metrics': current_metrics
                })
                
                # Detect anomalies
                anomalies = await self._detect_anomalies(
                    observation_context['baseline_metrics'],
                    current_metrics
                )
                
                if anomalies:
                    observation_context['anomalies_detected'].extend(anomalies)
                
                # Check for recovery events
                recovery_events = await self._detect_recovery_events(current_metrics)
                if recovery_events:
                    observation_context['recovery_events'].extend(recovery_events)
                
                await asyncio.sleep(5)  # Collect metrics every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics"""
        all_metrics = {}
        
        for collector in self.metrics_collectors:
            try:
                metrics = await collector()
                all_metrics.update(metrics)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
        
        return all_metrics
    
    async def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect system health metrics"""
        health_status = self.health_manager.get_health_status()
        
        return {
            'health_overall_status': health_status['overall_status'],
            'health_uptime': health_status['uptime_seconds'],
            'health_dependency_count': len(health_status['dependency_health']),
            'health_unhealthy_dependencies': len([
                dep for dep in health_status['dependency_health'].values()
                if dep['status'] in ['unhealthy', 'critical']
            ])
        }
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        health_status = self.health_manager.get_health_status()
        current_metrics = health_status.get('current_metrics', {})
        
        return {
            'perf_response_time_ms': current_metrics.get('response_time_ms', 0),
            'perf_throughput_per_sec': current_metrics.get('throughput_per_second', 0),
            'perf_active_connections': current_metrics.get('active_connections', 0)
        }
    
    async def _collect_error_metrics(self) -> Dict[str, Any]:
        """Collect error metrics"""
        health_status = self.health_manager.get_health_status()
        current_metrics = health_status.get('current_metrics', {})
        
        return {
            'error_rate': current_metrics.get('error_rate', 0),
            'error_total_recent': 0,  # Would get from error tracking system
        }
    
    async def _collect_resource_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics"""
        health_status = self.health_manager.get_health_status()
        resource_metrics = health_status.get('resource_metrics', {})
        
        cpu_metrics = resource_metrics.get('cpu', {})
        memory_metrics = resource_metrics.get('memory', {})
        disk_metrics = resource_metrics.get('disk', {})
        
        return {
            'resource_cpu_percent': cpu_metrics.get('percent', 0),
            'resource_memory_percent': memory_metrics.get('percent', 0),
            'resource_disk_percent': disk_metrics.get('percent', 0),
            'resource_load_avg_1m': cpu_metrics.get('load_avg_1m', 0)
        }
    
    async def _detect_anomalies(
        self,
        baseline_metrics: Dict[str, Any],
        current_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies compared to baseline"""
        anomalies = []
        
        # Define anomaly thresholds
        thresholds = {
            'perf_response_time_ms': 2.0,  # 2x increase
            'error_rate': 0.1,  # 10% absolute increase
            'resource_cpu_percent': 30,  # 30% absolute increase
            'resource_memory_percent': 20,  # 20% absolute increase
        }
        
        for metric, threshold in thresholds.items():
            baseline_value = baseline_metrics.get(metric, 0)
            current_value = current_metrics.get(metric, 0)
            
            if metric in ['perf_response_time_ms']:
                # Multiplicative threshold
                if baseline_value > 0 and current_value > baseline_value * threshold:
                    anomalies.append({
                        'type': 'performance_degradation',
                        'metric': metric,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'severity': 'high' if current_value > baseline_value * 3 else 'medium',
                        'detected_at': datetime.now()
                    })
            else:
                # Additive threshold
                if current_value > baseline_value + threshold:
                    anomalies.append({
                        'type': 'resource_spike' if 'resource' in metric else 'error_increase',
                        'metric': metric,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'severity': 'high' if current_value > baseline_value + threshold * 2 else 'medium',
                        'detected_at': datetime.now()
                    })
        
        return anomalies
    
    async def _detect_recovery_events(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect system recovery events"""
        recovery_events = []
        
        # Check if circuit breakers are closing (indicating recovery)
        # Check if error rates are decreasing
        # Check if response times are improving
        
        # Simplified recovery detection
        if (current_metrics.get('error_rate', 0) < 0.01 and 
            current_metrics.get('perf_response_time_ms', 0) < 1000):
            recovery_events.append({
                'type': 'service_recovery',
                'detected_at': datetime.now(),
                'metrics': current_metrics
            })
        
        return recovery_events
    
    async def _analyze_observation_results(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze observation results and generate insights"""
        timeline_metrics = observation['timeline_metrics']
        anomalies = observation['anomalies_detected']
        recovery_events = observation['recovery_events']
        
        if not timeline_metrics:
            return {'analysis': 'No timeline data available'}
        
        # Calculate key statistics
        start_time = observation['start_time']
        end_time = observation['end_time']
        duration = (end_time - start_time).total_seconds()
        
        # Performance impact analysis
        response_times = [m['metrics'].get('perf_response_time_ms', 0) for m in timeline_metrics]
        error_rates = [m['metrics'].get('error_rate', 0) for m in timeline_metrics]
        
        max_response_time = max(response_times) if response_times else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_error_rate = max(error_rates) if error_rates else 0
        
        # Recovery analysis
        recovery_time = None
        if recovery_events:
            first_recovery = min(recovery_events, key=lambda x: x['detected_at'])
            recovery_time = (first_recovery['detected_at'] - start_time).total_seconds()
        
        return {
            'duration_seconds': duration,
            'anomalies_count': len(anomalies),
            'recovery_events_count': len(recovery_events),
            'performance_impact': {
                'max_response_time_ms': max_response_time,
                'avg_response_time_ms': avg_response_time,
                'max_error_rate': max_error_rate
            },
            'recovery_time_seconds': recovery_time,
            'severity_assessment': self._assess_severity(anomalies),
            'recommendations': self._generate_recommendations(observation)
        }
    
    def _assess_severity(self, anomalies: List[Dict[str, Any]]) -> str:
        """Assess overall severity of experiment impact"""
        if not anomalies:
            return 'none'
        
        high_severity_count = len([a for a in anomalies if a['severity'] == 'high'])
        
        if high_severity_count > 0:
            return 'high'
        elif len(anomalies) > 5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, observation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on observation results"""
        recommendations = []
        
        anomalies = observation['anomalies_detected']
        analysis = observation.get('analysis', {})
        
        # Performance recommendations
        if analysis.get('performance_impact', {}).get('max_response_time_ms', 0) > 5000:
            recommendations.append("Consider implementing request timeout handling")
            recommendations.append("Review application performance under stress")
        
        # Error handling recommendations
        if analysis.get('performance_impact', {}).get('max_error_rate', 0) > 0.1:
            recommendations.append("Improve error handling and retry mechanisms")
            recommendations.append("Consider implementing circuit breaker pattern")
        
        # Recovery recommendations
        if analysis.get('recovery_time_seconds', 0) > 300:  # 5 minutes
            recommendations.append("Optimize recovery procedures for faster restoration")
            recommendations.append("Consider implementing automated failover mechanisms")
        
        # Anomaly-specific recommendations
        performance_anomalies = [a for a in anomalies if 'performance' in a['type']]
        if performance_anomalies:
            recommendations.append("Implement performance monitoring and alerting")
        
        resource_anomalies = [a for a in anomalies if 'resource' in a['type']]
        if resource_anomalies:
            recommendations.append("Review resource allocation and scaling policies")
        
        return recommendations


class ChaosEngineeringOrchestrator:
    """
    Main orchestrator for chaos engineering experiments
    Manages experiment lifecycle, safety, and reporting
    """
    
    def __init__(self, health_manager: ServiceHealthManager):
        self.health_manager = health_manager
        self.safety_validator = SafetyValidator(health_manager)
        self.fault_injector = FaultInjector()
        self.observer = ExperimentObserver(health_manager)
        
        # Experiment management
        self.experiment_catalog: Dict[str, ChaosExperiment] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.active_experiments: Dict[str, Dict] = {}
        
        # Configuration
        self.experiment_enabled = True
        self.max_concurrent_experiments = 1
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def register_experiment(self, experiment: ChaosExperiment):
        """Register a chaos experiment in the catalog"""
        async with self._lock:
            self.experiment_catalog[experiment.experiment_id] = experiment
        
        logger.info(f"Registered experiment: {experiment.experiment_id}")
    
    async def run_experiment(self, experiment_id: str, force: bool = False) -> str:
        """Execute a chaos experiment"""
        if not self.experiment_enabled and not force:
            raise ValueError("Chaos engineering is disabled")
        
        if experiment_id not in self.experiment_catalog:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        async with self._lock:
            if len(self.active_experiments) >= self.max_concurrent_experiments:
                raise ValueError("Maximum concurrent experiments reached")
        
        experiment = self.experiment_catalog[experiment_id]
        
        # Validate safety preconditions
        if not force:
            safety_ok, violations = await self.safety_validator.validate_safety_preconditions(experiment)
            if not safety_ok:
                raise ValueError(f"Safety preconditions not met: {violations}")
        
        # Start experiment execution
        execution_id = f"exec_{experiment_id}_{int(time.time())}"
        
        execution_context = {
            'execution_id': execution_id,
            'experiment': experiment,
            'status': ExperimentStatus.RUNNING,
            'start_time': datetime.now(),
            'injection_id': None,
            'observation_id': None,
            'safety_checks_passed': safety_ok,
            'emergency_stopped': False
        }
        
        async with self._lock:
            self.active_experiments[execution_id] = execution_context
        
        # Execute experiment in background
        asyncio.create_task(self._execute_experiment(execution_id))
        
        logger.info(f"Started chaos experiment: {execution_id}")
        return execution_id
    
    async def _execute_experiment(self, execution_id: str):
        """Execute chaos experiment with full lifecycle management"""
        try:
            execution = self.active_experiments[execution_id]
            experiment = execution['experiment']
            
            logger.info(f"Executing chaos experiment: {execution_id}")
            
            # Capture baseline system state
            baseline_health = self.health_manager.get_health_status()
            
            # Start observation
            observation_id = await self.observer.start_observation(experiment)
            execution['observation_id'] = observation_id
            
            # Inject fault
            injection_id = await self.fault_injector.inject_fault(experiment)
            execution['injection_id'] = injection_id
            
            # Monitor during experiment with safety checks
            await self._monitor_experiment_safety(execution_id, experiment.duration_seconds)
            
            # Remove fault
            await self.fault_injector.remove_fault(injection_id)
            
            # Stop observation and collect results
            observation_results = await self.observer.stop_observation(observation_id)
            
            # Capture final system state
            final_health = self.health_manager.get_health_status()
            
            # Create experiment result
            result = await self._create_experiment_result(
                execution, observation_results, baseline_health, final_health
            )
            
            # Store result and cleanup
            await self._complete_experiment(execution_id, result)
            
        except Exception as e:
            logger.error(f"Experiment execution failed: {execution_id}: {e}")
            await self._fail_experiment(execution_id, str(e))
    
    async def _monitor_experiment_safety(self, execution_id: str, duration: int):
        """Monitor experiment for safety violations"""
        execution = self.active_experiments[execution_id]
        experiment = execution['experiment']
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                # Check emergency stop conditions
                should_stop, reasons = await self.safety_validator.check_emergency_stop_conditions(experiment)
                
                if should_stop:
                    logger.critical(f"Emergency stop triggered for {execution_id}: {reasons}")
                    await self._emergency_stop_experiment(execution_id, reasons)
                    break
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _emergency_stop_experiment(self, execution_id: str, reasons: List[str]):
        """Emergency stop experiment execution"""
        execution = self.active_experiments[execution_id]
        execution['emergency_stopped'] = True
        execution['emergency_reasons'] = reasons
        
        # Immediately remove fault
        injection_id = execution.get('injection_id')
        if injection_id:
            try:
                await self.fault_injector.remove_fault(injection_id)
            except Exception as e:
                logger.error(f"Emergency fault cleanup failed: {e}")
        
        logger.critical(f"Emergency stopped experiment: {execution_id}")
    
    async def _create_experiment_result(
        self,
        execution: Dict[str, Any],
        observation_results: Dict[str, Any],
        baseline_health: Dict[str, Any],
        final_health: Dict[str, Any]
    ) -> ExperimentResult:
        """Create comprehensive experiment result"""
        
        experiment = execution['experiment']
        start_time = execution['start_time']
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Evaluate success criteria
        success_criteria_met = await self._evaluate_success_criteria(
            experiment, observation_results
        )
        
        # Calculate recovery time
        recovery_time = observation_results.get('analysis', {}).get('recovery_time_seconds', 0)
        
        # Extract observations and lessons
        observations = self._extract_observations(observation_results)
        lessons_learned = self._extract_lessons_learned(observation_results)
        recommendations = observation_results.get('analysis', {}).get('recommendations', [])
        
        # Detect failures
        failures_detected = []
        anomalies = observation_results.get('anomalies_detected', [])
        for anomaly in anomalies:
            if anomaly['severity'] == 'high':
                failures_detected.append(f"{anomaly['type']}: {anomaly['metric']}")
        
        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            status=ExperimentStatus.COMPLETED if not execution.get('emergency_stopped') else ExperimentStatus.ABORTED,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            impact_metrics=observation_results.get('analysis', {}),
            system_health_before=baseline_health,
            system_health_after=final_health,
            observations=observations,
            failures_detected=failures_detected,
            recovery_time_seconds=recovery_time,
            success_criteria_met=success_criteria_met,
            lessons_learned=lessons_learned,
            recommendations=recommendations
        )
        
        return result
    
    async def _evaluate_success_criteria(
        self,
        experiment: ChaosExperiment,
        observation_results: Dict[str, Any]
    ) -> bool:
        """Evaluate if experiment met success criteria"""
        
        success_criteria = experiment.success_criteria
        analysis = observation_results.get('analysis', {})
        
        for criterion in success_criteria:
            if criterion == "system_recovers_within_5_minutes":
                recovery_time = analysis.get('recovery_time_seconds', float('inf'))
                if recovery_time > 300:  # 5 minutes
                    return False
            
            elif criterion == "error_rate_below_10_percent":
                max_error_rate = analysis.get('performance_impact', {}).get('max_error_rate', 1.0)
                if max_error_rate > 0.1:
                    return False
            
            elif criterion == "no_data_loss":
                # This would check for data integrity issues
                # Simplified for demo
                data_loss_detected = False  # Would integrate with data validation
                if data_loss_detected:
                    return False
        
        return True
    
    def _extract_observations(self, observation_results: Dict[str, Any]) -> List[str]:
        """Extract key observations from experiment"""
        observations = []
        
        anomalies = observation_results.get('anomalies_detected', [])
        analysis = observation_results.get('analysis', {})
        
        # System behavior observations
        if anomalies:
            observations.append(f"Detected {len(anomalies)} system anomalies during experiment")
        
        recovery_time = analysis.get('recovery_time_seconds')
        if recovery_time:
            observations.append(f"System recovered after {recovery_time:.1f} seconds")
        
        max_response_time = analysis.get('performance_impact', {}).get('max_response_time_ms', 0)
        if max_response_time > 0:
            observations.append(f"Maximum response time reached {max_response_time:.1f}ms")
        
        return observations
    
    def _extract_lessons_learned(self, observation_results: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from experiment"""
        lessons = []
        
        analysis = observation_results.get('analysis', {})
        severity = analysis.get('severity_assessment', 'none')
        
        if severity == 'high':
            lessons.append("System showed high sensitivity to the injected fault")
            lessons.append("Consider improving fault tolerance mechanisms")
        elif severity == 'medium':
            lessons.append("System degraded gracefully under fault conditions")
        else:
            lessons.append("System demonstrated good resilience to the injected fault")
        
        recovery_time = analysis.get('recovery_time_seconds', 0)
        if recovery_time > 300:
            lessons.append("Recovery time exceeded acceptable thresholds")
        elif recovery_time > 0:
            lessons.append("System demonstrated automatic recovery capabilities")
        
        return lessons
    
    async def _complete_experiment(self, execution_id: str, result: ExperimentResult):
        """Complete experiment and store results"""
        async with self._lock:
            self.experiment_results[execution_id] = result
            del self.active_experiments[execution_id]
        
        # Save results to file
        await self._save_experiment_result(result)
        
        logger.info(f"Completed chaos experiment: {execution_id}")
    
    async def _fail_experiment(self, execution_id: str, error: str):
        """Mark experiment as failed"""
        execution = self.active_experiments.get(execution_id)
        if not execution:
            return
        
        # Cleanup any active injection
        injection_id = execution.get('injection_id')
        if injection_id:
            try:
                await self.fault_injector.remove_fault(injection_id)
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
        
        # Stop observation
        observation_id = execution.get('observation_id')
        if observation_id:
            try:
                await self.observer.stop_observation(observation_id)
            except Exception as e:
                logger.error(f"Stop observation failed: {e}")
        
        # Create failed result
        result = ExperimentResult(
            experiment_id=execution['experiment'].experiment_id,
            status=ExperimentStatus.FAILED,
            start_time=execution['start_time'],
            end_time=datetime.now(),
            duration_seconds=(datetime.now() - execution['start_time']).total_seconds(),
            impact_metrics={},
            system_health_before={},
            system_health_after={},
            observations=[f"Experiment failed: {error}"],
            failures_detected=[error],
            recovery_time_seconds=0,
            success_criteria_met=False,
            lessons_learned=[f"Experiment infrastructure needs improvement: {error}"],
            recommendations=["Review experiment setup and safety procedures"]
        )
        
        async with self._lock:
            self.experiment_results[execution_id] = result
            del self.active_experiments[execution_id]
        
        logger.error(f"Failed chaos experiment: {execution_id}: {error}")
    
    async def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to persistent storage"""
        try:
            results_dir = Path("data/chaos_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = results_dir / f"{result.experiment_id}_{int(result.start_time.timestamp())}.json"
            
            result_data = asdict(result)
            result_data['start_time'] = result.start_time.isoformat()
            result_data['end_time'] = result.end_time.isoformat() if result.end_time else None
            result_data['status'] = result.status.value
            
            async with aiofiles.open(result_file, 'w') as f:
                await f.write(json.dumps(result_data, indent=2, default=str))
            
        except Exception as e:
            logger.error(f"Failed to save experiment result: {e}")
    
    def get_experiment_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of running experiment"""
        execution = self.active_experiments.get(execution_id)
        if not execution:
            return None
        
        return {
            'execution_id': execution_id,
            'experiment_id': execution['experiment'].experiment_id,
            'status': execution['status'].value,
            'start_time': execution['start_time'].isoformat(),
            'duration_seconds': (datetime.now() - execution['start_time']).total_seconds(),
            'safety_checks_passed': execution['safety_checks_passed'],
            'emergency_stopped': execution.get('emergency_stopped', False)
        }
    
    def get_all_experiment_results(self) -> Dict[str, Any]:
        """Get all experiment results summary"""
        total_experiments = len(self.experiment_results)
        successful_experiments = len([
            r for r in self.experiment_results.values()
            if r.status == ExperimentStatus.COMPLETED and r.success_criteria_met
        ])
        
        return {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0,
            'active_experiments': len(self.active_experiments),
            'experiment_catalog_size': len(self.experiment_catalog),
            'chaos_engineering_enabled': self.experiment_enabled,
            'recent_results': [
                {
                    'experiment_id': r.experiment_id,
                    'status': r.status.value,
                    'success': r.success_criteria_met,
                    'duration': r.duration_seconds,
                    'start_time': r.start_time.isoformat()
                }
                for r in sorted(
                    self.experiment_results.values(),
                    key=lambda x: x.start_time,
                    reverse=True
                )[:10]
            ]
        }


# Global chaos engineering orchestrator
chaos_orchestrator: Optional[ChaosEngineeringOrchestrator] = None


def initialize_chaos_engineering(health_manager: ServiceHealthManager) -> ChaosEngineeringOrchestrator:
    """Initialize chaos engineering system"""
    global chaos_orchestrator
    
    chaos_orchestrator = ChaosEngineeringOrchestrator(health_manager)
    
    # Register sample experiments
    asyncio.create_task(_register_sample_experiments())
    
    logger.info("Chaos engineering system initialized")
    return chaos_orchestrator


async def _register_sample_experiments():
    """Register sample chaos experiments"""
    if not chaos_orchestrator:
        return
    
    # Database failure experiment
    db_failure_experiment = ChaosExperiment(
        experiment_id="db_failure_test",
        experiment_type=ChaosExperimentType.DATABASE_FAILURE,
        name="Database Connection Failure",
        description="Test system resilience when database connections fail",
        impact_scope=ImpactScope.DATA_LAYER,
        target_services=["database"],
        duration_seconds=300,  # 5 minutes
        intensity=1.0,
        parameters={
            'failure_type': 'connection_drop'
        },
        safety_checks=['system_healthy', 'business_hours_only', 'backup_systems_ready'],
        success_criteria=['system_recovers_within_5_minutes', 'error_rate_below_10_percent'],
        rollback_strategy='automatic',
        created_at=datetime.now(),
        created_by='chaos_engineer',
        tags=['database', 'resilience', 'automated']
    )
    
    # Cache failure experiment
    cache_failure_experiment = ChaosExperiment(
        experiment_id="cache_failure_test",
        experiment_type=ChaosExperimentType.CACHE_FAILURE,
        name="Cache Service Failure",
        description="Test system performance when cache service fails",
        impact_scope=ImpactScope.DEPENDENCY,
        target_services=["redis"],
        duration_seconds=180,  # 3 minutes
        intensity=1.0,
        parameters={
            'failure_type': 'service_stop'
        },
        safety_checks=['system_healthy', 'low_error_rate'],
        success_criteria=['system_recovers_within_5_minutes'],
        rollback_strategy='automatic',
        created_at=datetime.now(),
        created_by='chaos_engineer',
        tags=['cache', 'performance', 'automated']
    )
    
    # Latency injection experiment
    latency_experiment = ChaosExperiment(
        experiment_id="api_latency_test",
        experiment_type=ChaosExperimentType.LATENCY_INJECTION,
        name="API Latency Injection",
        description="Test system behavior under high API latency",
        impact_scope=ImpactScope.NETWORK,
        target_services=["external_apis"],
        duration_seconds=240,  # 4 minutes
        intensity=0.7,
        parameters={
            'delay_ms': 2000,
            'target_hosts': ['api.finnhub.io', 'api.alphavantage.co']
        },
        safety_checks=['system_healthy', 'adequate_resources'],
        success_criteria=['error_rate_below_10_percent'],
        rollback_strategy='automatic',
        created_at=datetime.now(),
        created_by='chaos_engineer',
        tags=['network', 'latency', 'apis']
    )
    
    await chaos_orchestrator.register_experiment(db_failure_experiment)
    await chaos_orchestrator.register_experiment(cache_failure_experiment)
    await chaos_orchestrator.register_experiment(latency_experiment)