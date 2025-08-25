"""
Production Auto-Scaling System for Investment Analysis Platform
Cost-optimized auto-scaling with intelligent resource management
Designed to maintain <$50/month operational costs
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import docker
import psutil
from prometheus_client import Gauge, Counter, Histogram
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Prometheus metrics
scaling_events = Counter('investment_scaling_events_total', 'Total scaling events', ['action', 'service'])
resource_utilization = Gauge('investment_resource_utilization', 'Resource utilization', ['resource', 'service'])
cost_metrics = Gauge('investment_cost_metrics', 'Cost tracking metrics', ['metric_type'])
scaling_latency = Histogram('investment_scaling_latency_seconds', 'Time taken to scale services')


class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration"""
    service_name: str
    min_replicas: int = 1
    max_replicas: int = 3
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_response_time_ms: float = 1000.0
    scale_up_threshold: float = 0.8    # 80% of target triggers scale up
    scale_down_threshold: float = 0.3  # 30% of target triggers scale down
    cooldown_seconds: int = 300        # 5 minutes between scaling actions
    
    # Cost optimization parameters
    cost_weight: float = 1.0           # Higher weight = more expensive to run
    peak_hours: List[int] = field(default_factory=lambda: list(range(9, 17)))  # 9 AM - 5 PM
    weekend_scale_factor: float = 0.5  # Scale down on weekends
    
    # State tracking
    last_scale_action: Optional[datetime] = None
    current_replicas: int = 1
    consecutive_violations: int = 0


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    response_time_ms: float
    error_rate_percent: float
    active_connections: int
    queue_length: int
    timestamp: datetime


class CostOptimizedAutoScaler:
    """
    Production auto-scaler with intelligent cost optimization
    Balances performance with cost constraints (<$50/month)
    """
    
    def __init__(self, docker_client: docker.DockerClient, db_session: AsyncSession):
        self.docker_client = docker_client
        self.db_session = db_session
        
        # Scaling configuration
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.daily_budget_usd = 1.67  # $50/month = ~$1.67/day
        self.current_daily_cost = 0.0
        
        # Resource monitoring
        self.metrics_cache: Dict[str, ResourceMetrics] = {}
        self.cache_ttl = 30  # seconds
        
        # Cost tracking
        self.cost_per_replica_hour = {
            'backend': 0.02,      # $0.02/hour per replica
            'frontend': 0.01,     # $0.01/hour per replica
            'worker': 0.015,      # $0.015/hour per replica
            'database': 0.05,     # $0.05/hour (fixed cost)
            'redis': 0.01         # $0.01/hour (fixed cost)
        }
        
        # Initialize scaling rules
        self._setup_default_scaling_rules()
        
        # Start monitoring
        self._monitor_task = None
        self.start_monitoring()
    
    def _setup_default_scaling_rules(self):
        """Setup default auto-scaling rules for each service"""
        
        # Backend API - Most critical service
        self.scaling_rules['backend'] = ScalingRule(
            service_name='investment_api_prod',
            min_replicas=1,
            max_replicas=4,
            target_cpu_percent=70.0,
            target_memory_percent=75.0,
            target_response_time_ms=1000.0,
            scale_up_threshold=0.8,
            scale_down_threshold=0.4,
            cooldown_seconds=180,  # 3 minutes
            cost_weight=3.0
        )
        
        # Frontend - Can scale aggressively as it's lightweight
        self.scaling_rules['frontend'] = ScalingRule(
            service_name='investment_web_prod',
            min_replicas=1,
            max_replicas=3,
            target_cpu_percent=60.0,
            target_memory_percent=70.0,
            target_response_time_ms=500.0,
            scale_up_threshold=0.7,
            scale_down_threshold=0.3,
            cooldown_seconds=120,  # 2 minutes
            cost_weight=1.0
        )
        
        # Celery Workers - Scale based on queue length
        self.scaling_rules['worker'] = ScalingRule(
            service_name='investment_worker_prod',
            min_replicas=1,
            max_replicas=3,
            target_cpu_percent=80.0,
            target_memory_percent=85.0,
            target_response_time_ms=5000.0,
            scale_up_threshold=0.9,
            scale_down_threshold=0.2,
            cooldown_seconds=300,  # 5 minutes
            cost_weight=2.0
        )
    
    async def get_service_metrics(self, service_name: str) -> Optional[ResourceMetrics]:
        """Get current resource metrics for a service"""
        try:
            # Check cache first
            now = datetime.utcnow()
            if service_name in self.metrics_cache:
                cached_metrics = self.metrics_cache[service_name]
                if (now - cached_metrics.timestamp).total_seconds() < self.cache_ttl:
                    return cached_metrics
            
            # Get container stats
            containers = self.docker_client.containers.list(
                filters={'name': service_name, 'status': 'running'}
            )
            
            if not containers:
                logger.warning(f"No running containers found for service: {service_name}")
                return None
            
            # Aggregate metrics across all replicas
            total_cpu = 0.0
            total_memory = 0.0
            total_connections = 0
            response_times = []
            error_rates = []
            
            for container in containers:
                stats = container.stats(stream=False, decode=True)
                
                # CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                
                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * 100.0
                    total_cpu += cpu_percent
                
                # Memory usage
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0
                total_memory += memory_percent
                
                # Get application-specific metrics via HTTP
                try:
                    container_ip = container.attrs['NetworkSettings']['Networks']['investment_network']['IPAddress']
                    
                    if 'backend' in service_name or 'api' in service_name:
                        # Get API metrics
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            response = await client.get(f"http://{container_ip}:8000/api/metrics")
                            if response.status_code == 200:
                                metrics_data = response.json()
                                response_times.append(metrics_data.get('avg_response_time', 0))
                                error_rates.append(metrics_data.get('error_rate', 0))
                                total_connections += metrics_data.get('active_connections', 0)
                
                except Exception as e:
                    logger.debug(f"Could not get application metrics for {container.name}: {e}")
            
            # Calculate averages
            num_replicas = len(containers)
            avg_cpu = total_cpu / num_replicas if num_replicas > 0 else 0
            avg_memory = total_memory / num_replicas if num_replicas > 0 else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0
            
            # Get disk usage (system-wide)
            disk_usage = psutil.disk_usage('/').percent
            
            # Get queue length for workers
            queue_length = await self._get_queue_length(service_name)
            
            metrics = ResourceMetrics(
                cpu_percent=avg_cpu,
                memory_percent=avg_memory,
                disk_percent=disk_usage,
                response_time_ms=avg_response_time,
                error_rate_percent=avg_error_rate,
                active_connections=total_connections,
                queue_length=queue_length,
                timestamp=now
            )
            
            # Cache metrics
            self.metrics_cache[service_name] = metrics
            
            # Update Prometheus metrics
            resource_utilization.labels(resource='cpu', service=service_name).set(avg_cpu)
            resource_utilization.labels(resource='memory', service=service_name).set(avg_memory)
            resource_utilization.labels(resource='disk', service=service_name).set(disk_usage)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics for {service_name}: {e}")
            return None
    
    async def _get_queue_length(self, service_name: str) -> int:
        """Get Celery queue length for worker services"""
        if 'worker' not in service_name:
            return 0
        
        try:
            # Query Redis for queue length
            import redis.asyncio as redis
            redis_client = redis.Redis(host='redis', port=6379, db=1, decode_responses=True)
            queue_length = await redis_client.llen('celery')
            await redis_client.close()
            return queue_length
        except Exception as e:
            logger.debug(f"Could not get queue length: {e}")
            return 0
    
    def _is_peak_hour(self) -> bool:
        """Check if current time is during peak hours"""
        now = datetime.now()
        hour = now.hour
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
        
        # Not peak if weekend
        if is_weekend:
            return False
        
        # Check if within peak hours (typically business hours)
        return hour in range(9, 17)  # 9 AM - 5 PM
    
    def _calculate_scaling_decision(self, rule: ScalingRule, metrics: ResourceMetrics) -> Tuple[ScalingAction, int]:
        """Determine scaling action based on metrics and rules"""
        
        # Calculate utilization scores
        cpu_score = metrics.cpu_percent / rule.target_cpu_percent
        memory_score = metrics.memory_percent / rule.target_memory_percent
        response_time_score = metrics.response_time_ms / rule.target_response_time_ms
        
        # For workers, consider queue length
        queue_score = 0
        if 'worker' in rule.service_name and metrics.queue_length > 0:
            # Each replica should handle ~10 tasks, scale up if queue is longer
            ideal_replicas_for_queue = max(1, metrics.queue_length // 10)
            queue_score = ideal_replicas_for_queue / rule.current_replicas
        
        # Calculate overall pressure score
        pressure_score = max(cpu_score, memory_score, response_time_score, queue_score)
        
        # Apply time-based scaling factors
        time_factor = 1.0
        if not self._is_peak_hour():
            time_factor = 0.7  # Reduce scaling during off-peak
        
        if datetime.now().weekday() >= 5:  # Weekend
            time_factor *= rule.weekend_scale_factor
        
        adjusted_pressure = pressure_score * time_factor
        
        # Check cost constraints
        daily_cost_projection = self._calculate_daily_cost_projection()
        cost_constraint_factor = 1.0
        
        if daily_cost_projection > self.daily_budget_usd * 0.8:  # Near budget limit
            cost_constraint_factor = 0.5  # Reduce scaling aggressiveness
            logger.warning(f"Near daily budget limit: ${daily_cost_projection:.2f} / ${self.daily_budget_usd:.2f}")
        elif daily_cost_projection > self.daily_budget_usd:  # Over budget
            cost_constraint_factor = 0.1  # Severely restrict scaling
            logger.critical(f"Over daily budget: ${daily_cost_projection:.2f} / ${self.daily_budget_usd:.2f}")
        
        # Determine scaling action
        if adjusted_pressure >= rule.scale_up_threshold and rule.current_replicas < rule.max_replicas:
            if cost_constraint_factor > 0.5:  # Only scale up if cost allows
                target_replicas = min(rule.current_replicas + 1, rule.max_replicas)
                return ScalingAction.SCALE_UP, target_replicas
        
        elif adjusted_pressure <= rule.scale_down_threshold and rule.current_replicas > rule.min_replicas:
            target_replicas = max(rule.current_replicas - 1, rule.min_replicas)
            return ScalingAction.SCALE_DOWN, target_replicas
        
        return ScalingAction.MAINTAIN, rule.current_replicas
    
    def _calculate_daily_cost_projection(self) -> float:
        """Calculate projected daily cost based on current scaling"""
        total_cost = 0.0
        
        for service_name, rule in self.scaling_rules.items():
            service_key = service_name.split('_')[0]  # Extract service type
            cost_per_hour = self.cost_per_replica_hour.get(service_key, 0.02)
            daily_service_cost = cost_per_hour * rule.current_replicas * 24
            total_cost += daily_service_cost
        
        # Add fixed costs
        total_cost += self.cost_per_replica_hour.get('database', 0.05) * 24
        total_cost += self.cost_per_replica_hour.get('redis', 0.01) * 24
        
        return total_cost
    
    async def _execute_scaling_action(self, rule: ScalingRule, action: ScalingAction, target_replicas: int) -> bool:
        """Execute the scaling action"""
        try:
            start_time = time.time()
            
            if action == ScalingAction.MAINTAIN:
                return True
            
            # Check cooldown period
            if rule.last_scale_action:
                cooldown_elapsed = (datetime.utcnow() - rule.last_scale_action).total_seconds()
                if cooldown_elapsed < rule.cooldown_seconds:
                    logger.debug(f"Scaling action for {rule.service_name} is in cooldown. "
                               f"Remaining: {rule.cooldown_seconds - cooldown_elapsed:.0f}s")
                    return False
            
            logger.info(f"Executing {action.value} for {rule.service_name}: "
                       f"{rule.current_replicas} -> {target_replicas} replicas")
            
            # For Docker Compose, we need to scale the service
            # This is a simplified approach - in production, you'd use Docker Swarm or Kubernetes
            service_containers = self.docker_client.containers.list(
                filters={'name': rule.service_name}
            )
            
            if action == ScalingAction.SCALE_UP:
                # Create new container(s)
                if service_containers:
                    base_container = service_containers[0]
                    container_config = base_container.attrs['Config']
                    host_config = base_container.attrs['HostConfig']
                    
                    for i in range(rule.current_replicas, target_replicas):
                        new_name = f"{rule.service_name}_{i+1}"
                        
                        # Create new container with same configuration
                        new_container = self.docker_client.containers.run(
                            container_config['Image'],
                            name=new_name,
                            detach=True,
                            **host_config
                        )
                        logger.info(f"Created new container: {new_name}")
            
            elif action == ScalingAction.SCALE_DOWN:
                # Remove excess containers
                containers_to_remove = service_containers[target_replicas:]
                
                for container in containers_to_remove:
                    logger.info(f"Stopping and removing container: {container.name}")
                    container.stop(timeout=30)
                    container.remove()
            
            # Update rule state
            rule.current_replicas = target_replicas
            rule.last_scale_action = datetime.utcnow()
            rule.consecutive_violations = 0
            
            # Update cost tracking
            service_key = rule.service_name.split('_')[0]
            cost_per_hour = self.cost_per_replica_hour.get(service_key, 0.02)
            
            if action == ScalingAction.SCALE_UP:
                self.current_daily_cost += cost_per_hour * 24
                cost_metrics.labels(metric_type='scaling_up_cost').inc(cost_per_hour * 24)
            elif action == ScalingAction.SCALE_DOWN:
                self.current_daily_cost -= cost_per_hour * 24
                cost_metrics.labels(metric_type='scaling_down_savings').inc(cost_per_hour * 24)
            
            # Record metrics
            scaling_events.labels(action=action.value, service=rule.service_name).inc()
            scaling_latency.observe(time.time() - start_time)
            
            # Log cost impact
            daily_cost_projection = self._calculate_daily_cost_projection()
            logger.info(f"Scaling completed. Daily cost projection: ${daily_cost_projection:.2f} / ${self.daily_budget_usd:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action {action.value} for {rule.service_name}: {e}")
            return False
    
    async def evaluate_scaling(self) -> Dict[str, Dict]:
        """Evaluate and execute scaling decisions for all services"""
        results = {}
        
        for service_name, rule in self.scaling_rules.items():
            try:
                # Get current metrics
                metrics = await self.get_service_metrics(rule.service_name)
                
                if metrics is None:
                    results[service_name] = {
                        'status': 'error',
                        'message': 'Failed to get metrics'
                    }
                    continue
                
                # Determine scaling action
                action, target_replicas = self._calculate_scaling_decision(rule, metrics)
                
                # Execute scaling if needed
                if action != ScalingAction.MAINTAIN:
                    success = await self._execute_scaling_action(rule, action, target_replicas)
                    
                    results[service_name] = {
                        'status': 'success' if success else 'failed',
                        'action': action.value,
                        'current_replicas': rule.current_replicas,
                        'target_replicas': target_replicas,
                        'metrics': {
                            'cpu_percent': metrics.cpu_percent,
                            'memory_percent': metrics.memory_percent,
                            'response_time_ms': metrics.response_time_ms,
                            'queue_length': metrics.queue_length
                        }
                    }
                else:
                    results[service_name] = {
                        'status': 'maintained',
                        'action': 'none',
                        'current_replicas': rule.current_replicas,
                        'metrics': {
                            'cpu_percent': metrics.cpu_percent,
                            'memory_percent': metrics.memory_percent,
                            'response_time_ms': metrics.response_time_ms,
                            'queue_length': metrics.queue_length
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error evaluating scaling for {service_name}: {e}")
                results[service_name] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        return results
    
    async def _monitoring_loop(self):
        """Main monitoring and scaling loop"""
        while True:
            try:
                logger.debug("Running auto-scaling evaluation...")
                
                # Evaluate scaling for all services
                results = await self.evaluate_scaling()
                
                # Log summary
                actions_taken = sum(1 for r in results.values() 
                                  if r.get('status') == 'success' and r.get('action') != 'none')
                
                if actions_taken > 0:
                    daily_cost = self._calculate_daily_cost_projection()
                    logger.info(f"Auto-scaling cycle completed. {actions_taken} actions taken. "
                               f"Daily cost projection: ${daily_cost:.2f}")
                
                # Update cost metrics
                cost_metrics.labels(metric_type='daily_cost_projection').set(
                    self._calculate_daily_cost_projection()
                )
                cost_metrics.labels(metric_type='budget_utilization').set(
                    (self._calculate_daily_cost_projection() / self.daily_budget_usd) * 100
                )
                
                # Wait before next evaluation
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error
    
    def start_monitoring(self):
        """Start the auto-scaling monitoring loop"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started auto-scaling monitoring")
    
    def stop_monitoring(self):
        """Stop the auto-scaling monitoring loop"""
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
            logger.info("Stopped auto-scaling monitoring")
    
    def get_scaling_status(self) -> Dict:
        """Get current scaling status and statistics"""
        status = {
            'daily_cost_projection': self._calculate_daily_cost_projection(),
            'daily_budget': self.daily_budget_usd,
            'budget_utilization_percent': (self._calculate_daily_cost_projection() / self.daily_budget_usd) * 100,
            'services': {}
        }
        
        for service_name, rule in self.scaling_rules.items():
            status['services'][service_name] = {
                'current_replicas': rule.current_replicas,
                'min_replicas': rule.min_replicas,
                'max_replicas': rule.max_replicas,
                'last_scale_action': rule.last_scale_action.isoformat() if rule.last_scale_action else None,
                'consecutive_violations': rule.consecutive_violations
            }
        
        return status


# Global auto-scaler instance
auto_scaler = None

def get_auto_scaler() -> CostOptimizedAutoScaler:
    """Get global auto-scaler instance"""
    global auto_scaler
    if auto_scaler is None:
        docker_client = docker.from_env()
        # db_session would be injected in real implementation
        auto_scaler = CostOptimizedAutoScaler(docker_client, None)
    return auto_scaler