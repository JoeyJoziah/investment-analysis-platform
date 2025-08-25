"""
Model Deployment Pipeline - Handles model deployment, rollback, and A/B testing
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import json
import random
import numpy as np
import aiohttp
import docker
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .registry import ModelRegistry, ModelVersion, DeploymentStatus, ModelStage
from .monitoring import ModelMonitor, PerformanceMetrics, AlertSeverity

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "a_b_test"
    SHADOW = "shadow"


class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    model_version: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    
    # Endpoint configuration
    endpoint_url: str
    port: int = 8080
    replicas: int = 2
    
    # Resource limits
    cpu_limit: float = 1.0  # CPU cores
    memory_limit_mb: int = 512
    
    # Health checks
    health_check_path: str = "/health"
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    startup_time_seconds: int = 60
    
    # Rollout configuration
    canary_percentage: float = 10.0
    rollout_duration_minutes: int = 60
    
    # Auto-scaling
    auto_scale: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 70
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_path: str = "/metrics"
    
    # Rollback
    auto_rollback: bool = True
    rollback_threshold_error_rate: float = 0.1
    rollback_threshold_latency_ms: float = 1000


@dataclass
class DeploymentStatus:
    """Status of a deployment"""
    deployment_id: str
    model_name: str
    model_version: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    
    # Status
    status: str  # "pending", "deploying", "active", "failed", "rolled_back"
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Endpoints
    endpoints: List[str] = field(default_factory=list)
    load_balancer_url: Optional[str] = None
    
    # Health
    healthy_replicas: int = 0
    total_replicas: int = 0
    
    # Metrics
    deployment_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Errors
    errors: List[str] = field(default_factory=list)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_name: str
    model_a_name: str
    model_a_version: str
    model_b_name: str
    model_b_version: str
    
    # Traffic split
    traffic_percentage_a: float = 50.0
    
    # Test duration
    start_time: datetime = field(default_factory=datetime.utcnow)
    duration_hours: int = 24
    
    # Success criteria
    primary_metric: str = "accuracy"
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    
    # User segmentation
    user_segments: List[str] = field(default_factory=list)
    segment_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Results from A/B test"""
    test_name: str
    winner: Optional[str] = None
    
    # Metrics
    model_a_metrics: Dict[str, float] = field(default_factory=dict)
    model_b_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Statistical significance
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    is_significant: bool = False
    
    # Sample sizes
    model_a_samples: int = 0
    model_b_samples: int = 0
    
    # Recommendations
    recommendation: str = ""


class ModelDeployer:
    """Handles model deployment operations"""
    
    def __init__(self, registry: ModelRegistry, monitor: ModelMonitor):
        self.registry = registry
        self.monitor = monitor
        self.docker_client = None
        self.deployments: Dict[str, DeploymentStatus] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Docker client: {e}")
    
    async def deploy(self, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy a model using specified strategy"""
        deployment_id = f"deploy_{config.model_name}_{config.model_version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        deployment_status = DeploymentStatus(
            deployment_id=deployment_id,
            model_name=config.model_name,
            model_version=config.model_version,
            environment=config.environment,
            strategy=config.strategy,
            status="pending",
            started_at=datetime.utcnow()
        )
        
        self.deployments[deployment_id] = deployment_status
        
        try:
            # Get model from registry
            model_version = await self.registry.get_model(config.model_name, config.model_version)
            if not model_version:
                raise ValueError(f"Model not found: {config.model_name} v{config.model_version}")
            
            # Deploy based on strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._deploy_blue_green(config, deployment_status, model_version)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._deploy_canary(config, deployment_status, model_version)
            elif config.strategy == DeploymentStrategy.ROLLING:
                await self._deploy_rolling(config, deployment_status, model_version)
            elif config.strategy == DeploymentStrategy.A_B_TEST:
                await self._deploy_ab_test(config, deployment_status, model_version)
            elif config.strategy == DeploymentStrategy.SHADOW:
                await self._deploy_shadow(config, deployment_status, model_version)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy}")
            
            # Update deployment status
            deployment_status.status = "active"
            deployment_status.completed_at = datetime.utcnow()
            
            # Update model registry
            await self.registry.deploy_model(
                config.model_name,
                config.model_version,
                deployment_status.load_balancer_url or deployment_status.endpoints[0],
                DeploymentStatus.PRODUCTION if config.environment == DeploymentEnvironment.PRODUCTION else DeploymentStatus.STAGING
            )
            
            # Register with monitor
            if config.enable_monitoring:
                await self.monitor.register_model(
                    config.model_name,
                    config.model_version,
                    deployment_status.load_balancer_url or deployment_status.endpoints[0]
                )
            
            logger.info(f"Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            deployment_status.status = "failed"
            deployment_status.errors.append(str(e))
            deployment_status.completed_at = datetime.utcnow()
            
            # Attempt rollback if configured
            if config.auto_rollback:
                await self.rollback(deployment_id)
        
        return deployment_status
    
    async def _deploy_blue_green(
        self,
        config: DeploymentConfig,
        status: DeploymentStatus,
        model_version: ModelVersion
    ):
        """Blue-green deployment strategy"""
        logger.info(f"Starting blue-green deployment for {config.model_name}")
        
        status.status = "deploying"
        
        # Deploy to green environment
        green_endpoints = await self._create_deployment(
            config,
            model_version,
            f"{config.model_name}-green"
        )
        
        status.endpoints.extend(green_endpoints)
        
        # Health check green environment
        await self._wait_for_health(green_endpoints, config)
        
        # Switch traffic from blue to green
        load_balancer_url = await self._update_load_balancer(
            config.model_name,
            green_endpoints
        )
        
        status.load_balancer_url = load_balancer_url
        
        # Monitor for issues
        await asyncio.sleep(config.startup_time_seconds)
        
        # If stable, remove blue environment
        await self._remove_deployment(f"{config.model_name}-blue")
        
        logger.info("Blue-green deployment completed")
    
    async def _deploy_canary(
        self,
        config: DeploymentConfig,
        status: DeploymentStatus,
        model_version: ModelVersion
    ):
        """Canary deployment strategy"""
        logger.info(f"Starting canary deployment for {config.model_name}")
        
        status.status = "deploying"
        
        # Deploy canary instance
        canary_endpoints = await self._create_deployment(
            config,
            model_version,
            f"{config.model_name}-canary",
            replicas=1
        )
        
        status.endpoints.extend(canary_endpoints)
        
        # Get existing endpoints
        existing_endpoints = await self._get_existing_endpoints(config.model_name)
        
        # Configure traffic split
        all_endpoints = existing_endpoints + canary_endpoints
        weights = self._calculate_canary_weights(
            len(existing_endpoints),
            len(canary_endpoints),
            config.canary_percentage
        )
        
        load_balancer_url = await self._update_load_balancer(
            config.model_name,
            all_endpoints,
            weights
        )
        
        status.load_balancer_url = load_balancer_url
        
        # Monitor canary
        rollout_duration = timedelta(minutes=config.rollout_duration_minutes)
        start_time = datetime.utcnow()
        
        while datetime.utcnow() - start_time < rollout_duration:
            # Check metrics
            metrics = await self.monitor.collect_metrics(
                config.model_name,
                canary_endpoints[0]
            )
            
            # Check for rollback conditions
            if await self._should_rollback(metrics, config):
                logger.warning("Canary metrics below threshold, rolling back")
                await self._remove_deployment(f"{config.model_name}-canary")
                return
            
            # Gradually increase traffic
            current_percentage = config.canary_percentage * (
                (datetime.utcnow() - start_time).total_seconds() /
                rollout_duration.total_seconds()
            )
            
            weights = self._calculate_canary_weights(
                len(existing_endpoints),
                len(canary_endpoints),
                min(current_percentage, 100)
            )
            
            await self._update_load_balancer(
                config.model_name,
                all_endpoints,
                weights
            )
            
            await asyncio.sleep(60)  # Check every minute
        
        # Full rollout
        await self._remove_deployment(f"{config.model_name}-old")
        
        logger.info("Canary deployment completed")
    
    async def _deploy_rolling(
        self,
        config: DeploymentConfig,
        status: DeploymentStatus,
        model_version: ModelVersion
    ):
        """Rolling deployment strategy"""
        logger.info(f"Starting rolling deployment for {config.model_name}")
        
        status.status = "deploying"
        
        # Get existing replicas
        existing_replicas = await self._get_existing_replicas(config.model_name)
        
        # Deploy new replicas one by one
        new_endpoints = []
        for i in range(config.replicas):
            # Deploy new replica
            endpoint = await self._create_single_deployment(
                config,
                model_version,
                f"{config.model_name}-v{config.model_version}-{i}"
            )
            
            new_endpoints.append(endpoint)
            status.endpoints.append(endpoint)
            
            # Wait for health
            await self._wait_for_health([endpoint], config)
            
            # Remove old replica if exists
            if i < len(existing_replicas):
                await self._remove_single_deployment(existing_replicas[i])
            
            # Update load balancer
            current_endpoints = new_endpoints + existing_replicas[i+1:]
            await self._update_load_balancer(config.model_name, current_endpoints)
            
            status.healthy_replicas = len(new_endpoints)
            status.total_replicas = config.replicas
        
        logger.info("Rolling deployment completed")
    
    async def _deploy_ab_test(
        self,
        config: DeploymentConfig,
        status: DeploymentStatus,
        model_version: ModelVersion
    ):
        """A/B test deployment (handled by ABTestManager)"""
        logger.info(f"A/B test deployment for {config.model_name}")
        
        # Deploy model alongside existing
        endpoints = await self._create_deployment(
            config,
            model_version,
            f"{config.model_name}-test"
        )
        
        status.endpoints.extend(endpoints)
        
        # A/B test configuration will be handled by ABTestManager
        status.deployment_metrics["ab_test"] = True
    
    async def _deploy_shadow(
        self,
        config: DeploymentConfig,
        status: DeploymentStatus,
        model_version: ModelVersion
    ):
        """Shadow deployment for testing without affecting production"""
        logger.info(f"Starting shadow deployment for {config.model_name}")
        
        # Deploy shadow instance
        shadow_endpoints = await self._create_deployment(
            config,
            model_version,
            f"{config.model_name}-shadow"
        )
        
        status.endpoints.extend(shadow_endpoints)
        
        # Configure to receive duplicate traffic without affecting responses
        # This would typically be done at the proxy/load balancer level
        status.deployment_metrics["shadow_mode"] = True
        status.deployment_metrics["shadow_endpoints"] = shadow_endpoints
    
    async def _create_deployment(
        self,
        config: DeploymentConfig,
        model_version: ModelVersion,
        deployment_name: str,
        replicas: Optional[int] = None
    ) -> List[str]:
        """Create deployment containers/pods"""
        endpoints = []
        replicas = replicas or config.replicas
        
        for i in range(replicas):
            endpoint = await self._create_single_deployment(
                config,
                model_version,
                f"{deployment_name}-{i}"
            )
            endpoints.append(endpoint)
        
        return endpoints
    
    async def _create_single_deployment(
        self,
        config: DeploymentConfig,
        model_version: ModelVersion,
        container_name: str
    ) -> str:
        """Create a single deployment container"""
        if not self.docker_client:
            # Fallback to mock deployment
            port = config.port + random.randint(0, 100)
            return f"http://localhost:{port}"
        
        try:
            # Build Docker image if needed
            image_name = f"{config.model_name}:{config.model_version}"
            
            # Run container
            container = self.docker_client.containers.run(
                image_name,
                name=container_name,
                ports={f"{config.port}/tcp": None},
                environment={
                    "MODEL_NAME": config.model_name,
                    "MODEL_VERSION": config.model_version,
                    "MODEL_PATH": str(model_version.model_path)
                },
                mem_limit=f"{config.memory_limit_mb}m",
                cpu_quota=int(config.cpu_limit * 100000),
                detach=True,
                auto_remove=False
            )
            
            # Get assigned port
            container.reload()
            port_mapping = container.ports.get(f"{config.port}/tcp")
            if port_mapping:
                host_port = port_mapping[0]["HostPort"]
                return f"http://localhost:{host_port}"
            
        except Exception as e:
            logger.error(f"Error creating container: {e}")
        
        # Fallback
        return f"http://localhost:{config.port}"
    
    async def _wait_for_health(self, endpoints: List[str], config: DeploymentConfig):
        """Wait for endpoints to become healthy"""
        max_attempts = config.startup_time_seconds // config.health_check_interval_seconds
        
        for endpoint in endpoints:
            healthy = False
            attempts = 0
            
            while not healthy and attempts < max_attempts:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{endpoint}{config.health_check_path}",
                            timeout=config.health_check_timeout_seconds
                        ) as response:
                            if response.status == 200:
                                healthy = True
                                logger.info(f"Endpoint {endpoint} is healthy")
                except:
                    pass
                
                if not healthy:
                    await asyncio.sleep(config.health_check_interval_seconds)
                    attempts += 1
            
            if not healthy:
                raise Exception(f"Endpoint {endpoint} failed health check")
    
    async def _update_load_balancer(
        self,
        model_name: str,
        endpoints: List[str],
        weights: Optional[List[float]] = None
    ) -> str:
        """Update load balancer configuration"""
        # In production, this would update actual load balancer
        # For now, return a mock URL
        load_balancer_url = f"http://lb.{model_name}.local"
        
        logger.info(f"Updated load balancer for {model_name} with {len(endpoints)} endpoints")
        
        return load_balancer_url
    
    def _calculate_canary_weights(
        self,
        existing_count: int,
        canary_count: int,
        canary_percentage: float
    ) -> List[float]:
        """Calculate traffic weights for canary deployment"""
        total = existing_count + canary_count
        canary_weight = canary_percentage / 100
        existing_weight = 1 - canary_weight
        
        weights = []
        
        # Existing endpoints
        for _ in range(existing_count):
            weights.append(existing_weight / existing_count if existing_count > 0 else 0)
        
        # Canary endpoints
        for _ in range(canary_count):
            weights.append(canary_weight / canary_count if canary_count > 0 else 0)
        
        return weights
    
    async def _should_rollback(
        self,
        metrics: PerformanceMetrics,
        config: DeploymentConfig
    ) -> bool:
        """Check if rollback conditions are met"""
        if not config.auto_rollback:
            return False
        
        if metrics.error_rate and metrics.error_rate > config.rollback_threshold_error_rate:
            return True
        
        if metrics.latency_ms and metrics.latency_ms > config.rollback_threshold_latency_ms:
            return True
        
        return False
    
    async def _get_existing_endpoints(self, model_name: str) -> List[str]:
        """Get existing deployment endpoints"""
        # In production, query actual deployment system
        return []
    
    async def _get_existing_replicas(self, model_name: str) -> List[str]:
        """Get existing deployment replicas"""
        # In production, query actual deployment system
        return []
    
    async def _remove_deployment(self, deployment_name: str):
        """Remove a deployment"""
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list(
                    filters={"name": deployment_name}
                )
                for container in containers:
                    container.stop()
                    container.remove()
            except Exception as e:
                logger.error(f"Error removing deployment: {e}")
    
    async def _remove_single_deployment(self, endpoint: str):
        """Remove a single deployment"""
        # Extract container name from endpoint and remove
        pass
    
    async def rollback(self, deployment_id: str) -> bool:
        """Rollback a deployment"""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        logger.info(f"Rolling back deployment {deployment_id}")
        
        # Remove new deployment
        for endpoint in deployment.endpoints:
            await self._remove_single_deployment(endpoint)
        
        deployment.status = "rolled_back"
        deployment.completed_at = datetime.utcnow()
        
        return True


class RollbackManager:
    """Manages deployment rollbacks"""
    
    def __init__(self, deployer: ModelDeployer, registry: ModelRegistry):
        self.deployer = deployer
        self.registry = registry
        self.rollback_history: List[Dict] = []
    
    async def rollback_to_version(
        self,
        model_name: str,
        target_version: str,
        reason: str = ""
    ) -> bool:
        """Rollback to a specific model version"""
        # Get target model version
        target_model = await self.registry.get_model(model_name, target_version)
        if not target_model:
            logger.error(f"Target version not found: {model_name} v{target_version}")
            return False
        
        # Create deployment config for rollback
        config = DeploymentConfig(
            model_name=model_name,
            model_version=target_version,
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=DeploymentStrategy.BLUE_GREEN,
            endpoint_url=f"http://{model_name}.api",
            auto_rollback=False  # Prevent recursive rollbacks
        )
        
        # Deploy target version
        deployment_status = await self.deployer.deploy(config)
        
        # Record rollback
        self.rollback_history.append({
            "timestamp": datetime.utcnow(),
            "model_name": model_name,
            "rolled_back_to": target_version,
            "reason": reason,
            "status": deployment_status.status
        })
        
        return deployment_status.status == "active"
    
    async def auto_rollback(
        self,
        model_name: str,
        current_metrics: PerformanceMetrics,
        thresholds: Dict[str, float]
    ) -> bool:
        """Automatically rollback if metrics fall below thresholds"""
        # Check if rollback is needed
        should_rollback = False
        reasons = []
        
        if current_metrics.accuracy and current_metrics.accuracy < thresholds.get("min_accuracy", 0):
            should_rollback = True
            reasons.append(f"Accuracy {current_metrics.accuracy} < {thresholds['min_accuracy']}")
        
        if current_metrics.error_rate and current_metrics.error_rate > thresholds.get("max_error_rate", 1):
            should_rollback = True
            reasons.append(f"Error rate {current_metrics.error_rate} > {thresholds['max_error_rate']}")
        
        if not should_rollback:
            return False
        
        # Get previous stable version
        models = await self.registry.get_active_models()
        previous_version = None
        
        for model in models:
            if (model["model_name"] == model_name and
                model["stage"] == "production" and
                model["deployment_status"] == "production"):
                previous_version = model["version"]
                break
        
        if not previous_version:
            logger.error(f"No previous stable version found for {model_name}")
            return False
        
        # Perform rollback
        return await self.rollback_to_version(
            model_name,
            previous_version,
            reason="; ".join(reasons)
        )


class ABTestManager:
    """Manages A/B testing for models"""
    
    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, ABTestResult] = {}
        self.request_routing: Dict[str, str] = {}  # user_id -> model_version
    
    async def start_test(self, config: ABTestConfig) -> str:
        """Start an A/B test"""
        test_id = f"ab_test_{config.test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_tests[test_id] = config
        
        # Initialize result tracking
        self.test_results[test_id] = ABTestResult(
            test_name=config.test_name,
            model_a_metrics={},
            model_b_metrics={}
        )
        
        logger.info(f"Started A/B test {test_id}: {config.model_a_name} vs {config.model_b_name}")
        
        # Start monitoring
        asyncio.create_task(self._monitor_test(test_id))
        
        return test_id
    
    async def route_request(self, test_id: str, user_id: str) -> str:
        """Route a request to appropriate model version"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        config = self.active_tests[test_id]
        
        # Check if user already assigned
        routing_key = f"{test_id}_{user_id}"
        if routing_key in self.request_routing:
            return self.request_routing[routing_key]
        
        # Assign based on traffic split
        random_value = random.random() * 100
        
        if random_value < config.traffic_percentage_a:
            model_version = config.model_a_version
            model_name = config.model_a_name
        else:
            model_version = config.model_b_version
            model_name = config.model_b_name
        
        # Store assignment for consistency
        self.request_routing[routing_key] = model_version
        
        return model_version
    
    async def record_result(
        self,
        test_id: str,
        model_version: str,
        metrics: Dict[str, float]
    ):
        """Record results for A/B test"""
        if test_id not in self.test_results:
            return
        
        result = self.test_results[test_id]
        config = self.active_tests[test_id]
        
        # Update metrics
        if model_version == config.model_a_version:
            for key, value in metrics.items():
                if key not in result.model_a_metrics:
                    result.model_a_metrics[key] = []
                result.model_a_metrics[key].append(value)
            result.model_a_samples += 1
        else:
            for key, value in metrics.items():
                if key not in result.model_b_metrics:
                    result.model_b_metrics[key] = []
                result.model_b_metrics[key].append(value)
            result.model_b_samples += 1
    
    async def _monitor_test(self, test_id: str):
        """Monitor A/B test progress"""
        config = self.active_tests[test_id]
        end_time = config.start_time + timedelta(hours=config.duration_hours)
        
        while datetime.utcnow() < end_time:
            # Check if minimum samples reached
            result = self.test_results[test_id]
            
            if (result.model_a_samples >= config.minimum_sample_size and
                result.model_b_samples >= config.minimum_sample_size):
                
                # Calculate statistics
                await self._calculate_statistics(test_id)
                
                # Check for early stopping if clear winner
                if result.is_significant and result.p_value < 0.01:
                    logger.info(f"A/B test {test_id} stopped early due to clear winner")
                    break
            
            await asyncio.sleep(300)  # Check every 5 minutes
        
        # Final analysis
        await self._finalize_test(test_id)
    
    async def _calculate_statistics(self, test_id: str):
        """Calculate statistical significance"""
        result = self.test_results[test_id]
        config = self.active_tests[test_id]
        
        # Get primary metric values
        metric_a = result.model_a_metrics.get(config.primary_metric, [])
        metric_b = result.model_b_metrics.get(config.primary_metric, [])
        
        if not metric_a or not metric_b:
            return
        
        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(metric_a, metric_b)
        
        result.p_value = p_value
        result.is_significant = p_value < (1 - config.confidence_level)
        
        # Calculate confidence interval
        mean_diff = np.mean(metric_a) - np.mean(metric_b)
        se = np.sqrt(np.var(metric_a)/len(metric_a) + np.var(metric_b)/len(metric_b))
        ci = 1.96 * se  # 95% confidence interval
        
        result.confidence_interval = (mean_diff - ci, mean_diff + ci)
        
        # Determine winner
        if result.is_significant:
            if np.mean(metric_a) > np.mean(metric_b):
                result.winner = config.model_a_name
            else:
                result.winner = config.model_b_name
    
    async def _finalize_test(self, test_id: str):
        """Finalize A/B test and generate recommendations"""
        result = self.test_results[test_id]
        config = self.active_tests[test_id]
        
        # Final statistical analysis
        await self._calculate_statistics(test_id)
        
        # Generate recommendation
        if result.winner:
            result.recommendation = f"Deploy {result.winner} to production"
            if result.p_value < 0.01:
                result.recommendation += " (strong statistical significance)"
            elif result.p_value < 0.05:
                result.recommendation += " (moderate statistical significance)"
        else:
            result.recommendation = "No significant difference between models"
            if result.model_a_samples < config.minimum_sample_size:
                result.recommendation += f" (insufficient samples: {result.model_a_samples}/{config.minimum_sample_size})"
        
        # Clean up
        del self.active_tests[test_id]
        
        # Clean up routing entries
        keys_to_remove = [k for k in self.request_routing if k.startswith(test_id)]
        for key in keys_to_remove:
            del self.request_routing[key]
        
        logger.info(f"A/B test {test_id} completed. Winner: {result.winner}")
    
    def get_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """Get results for a specific test"""
        return self.test_results.get(test_id)