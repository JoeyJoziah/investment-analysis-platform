"""
Resilience Integration Module
Comprehensive integration of all error handling and resilience features
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging
from pathlib import Path

from .advanced_circuit_breaker import EnhancedCircuitBreaker, AdaptiveThresholds, CascadingFailurePreventor
from .enhanced_error_handling import ErrorHandlingManager, ErrorClassifier, ErrorCorrelationEngine
from .resilient_pipeline import ResilientPipeline, TaskExecutor, DataQualityValidator
from .service_health_manager import ServiceHealthManager, ServiceConfig, DependencyConfig, DependencyType, bulkhead_manager
from .disaster_recovery import DisasterRecoverySystem, BackupManager, DisasterRecoveryOrchestrator
from .chaos_engineering import ChaosEngineeringOrchestrator, initialize_chaos_engineering
from .enhanced_logging import LoggingSystem, initialize_logging_system, get_logger, LogLevel
from .exceptions import *

logger = logging.getLogger(__name__)


class ResilienceConfiguration:
    """Configuration for the complete resilience system"""
    
    def __init__(
        self,
        # Circuit breaker settings
        enable_circuit_breakers: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: int = 60,
        
        # Error handling settings
        enable_error_correlation: bool = True,
        error_correlation_time_window: int = 10,
        
        # Pipeline settings
        enable_resilient_pipelines: bool = True,
        pipeline_max_concurrent_tasks: int = 10,
        pipeline_enable_checkpointing: bool = True,
        
        # Health monitoring settings
        enable_health_monitoring: bool = True,
        health_check_interval: int = 30,
        
        # Disaster recovery settings
        enable_disaster_recovery: bool = True,
        backup_retention_days: int = 30,
        enable_auto_recovery: bool = True,
        
        # Chaos engineering settings
        enable_chaos_engineering: bool = False,
        chaos_experiments_enabled: bool = False,
        
        # Logging settings
        log_level: LogLevel = LogLevel.INFO,
        enable_elasticsearch_logging: bool = False,
        elasticsearch_hosts: Optional[List[str]] = None
    ):
        self.enable_circuit_breakers = enable_circuit_breakers
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_recovery_timeout = circuit_breaker_recovery_timeout
        
        self.enable_error_correlation = enable_error_correlation
        self.error_correlation_time_window = error_correlation_time_window
        
        self.enable_resilient_pipelines = enable_resilient_pipelines
        self.pipeline_max_concurrent_tasks = pipeline_max_concurrent_tasks
        self.pipeline_enable_checkpointing = pipeline_enable_checkpointing
        
        self.enable_health_monitoring = enable_health_monitoring
        self.health_check_interval = health_check_interval
        
        self.enable_disaster_recovery = enable_disaster_recovery
        self.backup_retention_days = backup_retention_days
        self.enable_auto_recovery = enable_auto_recovery
        
        self.enable_chaos_engineering = enable_chaos_engineering
        self.chaos_experiments_enabled = chaos_experiments_enabled
        
        self.log_level = log_level
        self.enable_elasticsearch_logging = enable_elasticsearch_logging
        self.elasticsearch_hosts = elasticsearch_hosts or []


class InvestmentAnalysisResilienceSystem:
    """
    Complete resilience system for the investment analysis application
    Integrates all error handling and resilience components
    """
    
    def __init__(self, config: ResilienceConfiguration):
        self.config = config
        
        # Core components
        self.logging_system: Optional[LoggingSystem] = None
        self.error_handler: Optional[ErrorHandlingManager] = None
        self.health_manager: Optional[ServiceHealthManager] = None
        self.disaster_recovery: Optional[DisasterRecoverySystem] = None
        self.chaos_orchestrator: Optional[ChaosEngineeringOrchestrator] = None
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.cascading_failure_preventor: Optional[CascadingFailurePreventor] = None
        
        # Resilient pipelines
        self.pipelines: Dict[str, ResilientPipeline] = {}
        
        # System state
        self._initialized = False
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Metrics
        self._system_metrics = {
            'initialization_time': None,
            'uptime_start': None,
            'total_errors_handled': 0,
            'total_recoveries_performed': 0,
            'circuit_breakers_activated': 0,
            'chaos_experiments_run': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the complete resilience system"""
        if self._initialized:
            logger.warning("Resilience system already initialized")
            return
        
        start_time = time.time()
        
        try:
            logger.info("Initializing Investment Analysis Resilience System...")
            
            # 1. Initialize logging system first
            await self._initialize_logging()
            
            # 2. Initialize error handling
            await self._initialize_error_handling()
            
            # 3. Initialize health monitoring
            await self._initialize_health_monitoring()
            
            # 4. Initialize circuit breakers
            await self._initialize_circuit_breakers()
            
            # 5. Initialize resilient pipelines
            await self._initialize_resilient_pipelines()
            
            # 6. Initialize disaster recovery
            await self._initialize_disaster_recovery()
            
            # 7. Initialize chaos engineering (if enabled)
            if self.config.enable_chaos_engineering:
                await self._initialize_chaos_engineering()
            
            # 8. Set up integrations between components
            await self._setup_integrations()
            
            # Mark as initialized
            self._initialized = True
            self._system_metrics['initialization_time'] = time.time() - start_time
            
            logger.info(
                f"Resilience system initialized successfully in "
                f"{self._system_metrics['initialization_time']:.2f} seconds"
            )
            
        except Exception as e:
            logger.critical(f"Failed to initialize resilience system: {e}")
            raise ResilienceSystemError(f"Initialization failed: {e}")
    
    async def start(self) -> None:
        """Start all resilience components"""
        if not self._initialized:
            raise ResilienceSystemError("System not initialized. Call initialize() first.")
        
        if self._running:
            logger.warning("Resilience system already running")
            return
        
        try:
            logger.info("Starting resilience system components...")
            
            # Start health monitoring
            if self.health_manager and self.config.enable_health_monitoring:
                await self.health_manager.start_monitoring()
            
            # Start disaster recovery monitoring
            if self.disaster_recovery:
                await self.disaster_recovery.initialize()
            
            # Start logging system monitoring
            if self.logging_system:
                self.logging_system.start_real_time_monitoring()
            
            # Start background tasks
            self._background_tasks = [
                asyncio.create_task(self._system_monitoring_loop()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            self._running = True
            self._system_metrics['uptime_start'] = datetime.now()
            
            logger.info("Resilience system started successfully")
            
        except Exception as e:
            logger.critical(f"Failed to start resilience system: {e}")
            raise ResilienceSystemError(f"Startup failed: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the resilience system"""
        if not self._running:
            return
        
        logger.info("Shutting down resilience system...")
        
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            
            # Stop health monitoring
            if self.health_manager:
                await self.health_manager.stop_monitoring()
            
            # Stop disaster recovery
            if self.disaster_recovery:
                await self.disaster_recovery.shutdown()
            
            # Stop logging monitoring
            if self.logging_system:
                await self.logging_system.stop_real_time_monitoring()
            
            # Stop pipelines
            for pipeline in self.pipelines.values():
                await pipeline.stop()
            
            self._running = False
            
            logger.info("Resilience system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _initialize_logging(self):
        """Initialize enhanced logging system"""
        logger.info("Initializing logging system...")
        
        self.logging_system = initialize_logging_system(
            log_level=self.config.log_level,
            elasticsearch_hosts=self.config.elasticsearch_hosts if self.config.enable_elasticsearch_logging else None,
            log_directory="logs",
            enable_file_rotation=True,
            enable_console_output=True
        )
        
        # Register custom alert callbacks
        def resilience_alert_handler(alert_data):
            logger.critical(f"RESILIENCE ALERT: {alert_data['pattern_name']}")
            # Could trigger additional recovery actions here
        
        self.logging_system.register_alert_callback(resilience_alert_handler)
    
    async def _initialize_error_handling(self):
        """Initialize error handling and correlation"""
        logger.info("Initializing error handling system...")
        
        from .enhanced_error_handling import error_handler
        self.error_handler = error_handler
        
        if self.config.enable_error_correlation:
            # Error correlation is already integrated in the error handler
            pass
    
    async def _initialize_health_monitoring(self):
        """Initialize service health monitoring"""
        if not self.config.enable_health_monitoring:
            return
        
        logger.info("Initializing health monitoring...")
        
        # Create service configuration for the investment analysis app
        service_config = ServiceConfig(
            name="investment_analysis_service",
            check_interval_seconds=self.config.health_check_interval,
            dependencies=[
                DependencyConfig(
                    name="postgres_database",
                    dependency_type=DependencyType.DATABASE,
                    endpoint="postgresql://localhost:5432/investment_db",
                    timeout_seconds=5.0,
                    check_interval_seconds=30,
                    critical=True,
                    recovery_actions=["restart_service", "switch_to_replica"]
                ),
                DependencyConfig(
                    name="redis_cache",
                    dependency_type=DependencyType.CACHE,
                    endpoint="redis://localhost:6379",
                    timeout_seconds=3.0,
                    check_interval_seconds=15,
                    critical=False,
                    recovery_actions=["clear_cache", "restart_cache"]
                ),
                DependencyConfig(
                    name="finnhub_api",
                    dependency_type=DependencyType.EXTERNAL_API,
                    endpoint="https://finnhub.io/api/v1/status",
                    timeout_seconds=10.0,
                    check_interval_seconds=60,
                    critical=True,
                    recovery_actions=["circuit_break", "switch_provider"]
                ),
                DependencyConfig(
                    name="alpha_vantage_api",
                    dependency_type=DependencyType.EXTERNAL_API,
                    endpoint="https://www.alphavantage.co",
                    timeout_seconds=15.0,
                    check_interval_seconds=60,
                    critical=False,
                    recovery_actions=["circuit_break", "use_cache"]
                )
            ],
            resource_thresholds={
                'cpu_critical': 85.0,
                'memory_critical': 90.0,
                'disk_critical': 95.0
            },
            bulkhead_enabled=True
        )
        
        self.health_manager = ServiceHealthManager(service_config)
        
        # Set up bulkhead isolation groups
        await bulkhead_manager.create_isolation_group(
            "external_apis",
            ["finnhub_api", "alpha_vantage_api", "polygon_api"],
            max_concurrent_requests=50,
            timeout_seconds=30
        )
        
        await bulkhead_manager.create_isolation_group(
            "data_processing",
            ["recommendation_engine", "analysis_pipeline"],
            max_concurrent_requests=20,
            timeout_seconds=60
        )
    
    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical services"""
        if not self.config.enable_circuit_breakers:
            return
        
        logger.info("Initializing circuit breakers...")
        
        # Circuit breaker configurations for different services
        cb_configs = {
            'finnhub_api': AdaptiveThresholds(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2,
                rate_limit_threshold=2,
                timeout_threshold=3,
                error_rate_threshold=0.3
            ),
            'alpha_vantage_api': AdaptiveThresholds(
                failure_threshold=2,
                recovery_timeout=60,
                success_threshold=1,
                rate_limit_threshold=1,
                timeout_threshold=2,
                error_rate_threshold=0.5
            ),
            'database': AdaptiveThresholds(
                failure_threshold=5,
                recovery_timeout=15,
                success_threshold=3,
                rate_limit_threshold=10,
                timeout_threshold=5,
                error_rate_threshold=0.2
            ),
            'cache': AdaptiveThresholds(
                failure_threshold=3,
                recovery_timeout=10,
                success_threshold=2,
                rate_limit_threshold=5,
                timeout_threshold=3,
                error_rate_threshold=0.4
            )
        }
        
        # Create circuit breakers
        for service_name, thresholds in cb_configs.items():
            circuit_breaker = EnhancedCircuitBreaker(
                name=service_name,
                base_thresholds=thresholds,
                provider_name=service_name,
                on_state_change=self._on_circuit_breaker_state_change
            )
            
            self.circuit_breakers[service_name] = circuit_breaker
        
        # Initialize cascading failure prevention
        self.cascading_failure_preventor = CascadingFailurePreventor()
        
        # Register circuit breakers with cascading failure preventor
        for name, breaker in self.circuit_breakers.items():
            self.cascading_failure_preventor.register_circuit_breaker(name, breaker)
        
        # Set up service dependencies
        self.cascading_failure_preventor.add_dependency("recommendation_engine", "database")
        self.cascading_failure_preventor.add_dependency("recommendation_engine", "cache")
        self.cascading_failure_preventor.add_dependency("data_ingestion", "finnhub_api")
        self.cascading_failure_preventor.add_dependency("data_ingestion", "alpha_vantage_api")
    
    async def _initialize_resilient_pipelines(self):
        """Initialize resilient data processing pipelines"""
        if not self.config.enable_resilient_pipelines:
            return
        
        logger.info("Initializing resilient pipelines...")
        
        # Main data ingestion pipeline
        data_ingestion_pipeline = ResilientPipeline(
            name="data_ingestion_pipeline",
            max_concurrent_tasks=self.config.pipeline_max_concurrent_tasks,
            enable_checkpointing=self.config.pipeline_enable_checkpointing,
            enable_caching=True,
            cache_ttl=3600
        )
        
        # Register data quality validators
        data_ingestion_pipeline.register_data_validator(
            "stock_price_data",
            self._validate_stock_price_data
        )
        
        data_ingestion_pipeline.register_data_validator(
            "financial_metrics",
            self._validate_financial_metrics
        )
        
        # Register executors with circuit breaker protection
        data_ingestion_pipeline.register_executor(
            "fetch_stock_data",
            self._fetch_stock_data_with_cb,
            max_retries=3,
            circuit_breaker_config={
                'failure_threshold': 5,
                'recovery_timeout': 30,
                'success_threshold': 2,
                'rate_limit_threshold': 3,
                'timeout_threshold': 3,
                'error_rate_threshold': 0.4
            }
        )
        
        data_ingestion_pipeline.register_executor(
            "process_stock_analysis",
            self._process_stock_analysis,
            max_retries=2
        )
        
        self.pipelines["data_ingestion"] = data_ingestion_pipeline
        
        # Analysis pipeline
        analysis_pipeline = ResilientPipeline(
            name="analysis_pipeline",
            max_concurrent_tasks=self.config.pipeline_max_concurrent_tasks // 2,
            enable_checkpointing=True,
            enable_caching=True
        )
        
        analysis_pipeline.register_executor(
            "technical_analysis",
            self._perform_technical_analysis,
            max_retries=2
        )
        
        analysis_pipeline.register_executor(
            "fundamental_analysis",
            self._perform_fundamental_analysis,
            max_retries=2
        )
        
        self.pipelines["analysis"] = analysis_pipeline
    
    async def _initialize_disaster_recovery(self):
        """Initialize disaster recovery system"""
        if not self.config.enable_disaster_recovery:
            return
        
        logger.info("Initializing disaster recovery...")
        
        self.disaster_recovery = DisasterRecoverySystem({
            'backup_root': 'data/backups',
            'retention_days': self.config.backup_retention_days,
            'disaster_monitoring': self.config.enable_auto_recovery
        })
        
        # Set health manager for disaster detection
        if self.health_manager:
            self.disaster_recovery.set_health_manager(self.health_manager)
        
        # Register backup sources
        backup_manager = self.disaster_recovery.backup_manager
        
        await backup_manager.register_backup_source(
            "database_backup",
            "/var/lib/postgresql/data",
            from .disaster_recovery import BackupType.FULL,
            "daily",
            retention_days=7
        )
        
        await backup_manager.register_backup_source(
            "application_config",
            "config/",
            BackupType.CONFIGURATION,
            "daily",
            retention_days=30
        )
        
        await backup_manager.register_backup_source(
            "ml_models",
            "models/",
            BackupType.FULL,
            "weekly",
            retention_days=60
        )
    
    async def _initialize_chaos_engineering(self):
        """Initialize chaos engineering system"""
        logger.info("Initializing chaos engineering...")
        
        if not self.health_manager:
            logger.warning("Health manager not available for chaos engineering")
            return
        
        self.chaos_orchestrator = initialize_chaos_engineering(self.health_manager)
        self.chaos_orchestrator.experiment_enabled = self.config.chaos_experiments_enabled
    
    async def _setup_integrations(self):
        """Set up integrations between components"""
        logger.info("Setting up component integrations...")
        
        # Integrate error handler with circuit breakers
        if self.error_handler and self.circuit_breakers:
            for service_name, circuit_breaker in self.circuit_breakers.items():
                # This would set up automatic circuit breaking based on error patterns
                pass
        
        # Integrate health manager with disaster recovery
        if self.health_manager and self.disaster_recovery:
            self.disaster_recovery.set_health_manager(self.health_manager)
    
    def _on_circuit_breaker_state_change(self, state):
        """Handle circuit breaker state changes"""
        self._system_metrics['circuit_breakers_activated'] += 1
        
        resilience_logger = get_logger("resilience_system")
        resilience_logger.warning(f"Circuit breaker state changed: {state}")
        
        # Could trigger additional recovery actions here
    
    async def _system_monitoring_loop(self):
        """Background task for system monitoring"""
        while self._running:
            try:
                await self._collect_system_health()
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Background task for metrics collection"""
        while self._running:
            try:
                await self._collect_resilience_metrics()
                await asyncio.sleep(300)  # Collect metrics every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(300)
    
    async def _health_check_loop(self):
        """Background task for additional health checks"""
        while self._running:
            try:
                await self._perform_system_health_check()
                await asyncio.sleep(120)  # Health check every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(120)
    
    async def _collect_system_health(self):
        """Collect overall system health metrics"""
        if self.health_manager:
            health_status = self.health_manager.get_health_status()
            
            if health_status['overall_status'] == 'critical':
                logger.critical("System health critical - initiating recovery procedures")
                
                # Could trigger automatic recovery here
                if self.disaster_recovery and self.config.enable_auto_recovery:
                    from .disaster_recovery import DisasterType
                    await self.disaster_recovery.orchestrator.initiate_recovery(
                        DisasterType.SERVICE_FAILURE
                    )
    
    async def _collect_resilience_metrics(self):
        """Collect resilience system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_uptime': (
                datetime.now() - self._system_metrics['uptime_start']
            ).total_seconds() if self._system_metrics['uptime_start'] else 0,
            'circuit_breakers': {
                name: cb.get_comprehensive_metrics()
                for name, cb in self.circuit_breakers.items()
            },
            'pipelines': {
                name: pipeline.get_health_status()
                for name, pipeline in self.pipelines.items()
            },
            'error_handling': self.error_handler.get_error_analytics() if self.error_handler else {},
            'disaster_recovery': (
                self.disaster_recovery.get_system_status()
                if self.disaster_recovery else {}
            )
        }
        
        # Log metrics
        resilience_logger = get_logger("resilience_metrics")
        resilience_logger.info("Resilience system metrics collected", custom_fields=metrics)
    
    async def _perform_system_health_check(self):
        """Perform comprehensive system health check"""
        issues_found = []
        
        # Check circuit breaker states
        for name, cb in self.circuit_breakers.items():
            if cb.state.value == 'open':
                issues_found.append(f"Circuit breaker {name} is open")
        
        # Check pipeline health
        for name, pipeline in self.pipelines.items():
            health_status = pipeline.get_health_status()
            if health_status['status'] != 'healthy':
                issues_found.append(f"Pipeline {name} status: {health_status['status']}")
        
        if issues_found:
            logger.warning(f"System health issues detected: {issues_found}")
    
    # Pipeline executor implementations
    async def _validate_stock_price_data(self, data):
        """Validate stock price data quality"""
        if not isinstance(data, dict):
            return {'is_valid': False, 'errors': ['Data must be a dictionary']}
        
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {'is_valid': False, 'errors': [f'Missing fields: {missing_fields}']}
        
        # Validate price consistency
        if data['high'] < data['low']:
            return {'is_valid': False, 'errors': ['High price less than low price']}
        
        if not (data['low'] <= data['open'] <= data['high']):
            return {'is_valid': False, 'errors': ['Open price outside high-low range']}
        
        if not (data['low'] <= data['close'] <= data['high']):
            return {'is_valid': False, 'errors': ['Close price outside high-low range']}
        
        return {'is_valid': True, 'quality_score': 1.0}
    
    async def _validate_financial_metrics(self, data):
        """Validate financial metrics data"""
        # Simplified validation
        return {'is_valid': True, 'quality_score': 0.9}
    
    async def _fetch_stock_data_with_cb(self, ticker):
        """Fetch stock data with circuit breaker protection"""
        # This would use the appropriate circuit breaker
        circuit_breaker = self.circuit_breakers.get('finnhub_api')
        
        if circuit_breaker:
            return await circuit_breaker.call(self._fetch_stock_data, ticker)
        else:
            return await self._fetch_stock_data(ticker)
    
    async def _fetch_stock_data(self, ticker):
        """Actual stock data fetching implementation"""
        # This would integrate with your actual data fetching logic
        await asyncio.sleep(0.1)  # Simulate API call
        return {'ticker': ticker, 'price': 100.0, 'volume': 1000}
    
    async def _process_stock_analysis(self, stock_data):
        """Process stock analysis"""
        # This would integrate with your analysis logic
        await asyncio.sleep(0.1)  # Simulate processing
        return {'analysis': 'completed', 'recommendation': 'hold'}
    
    async def _perform_technical_analysis(self, data):
        """Perform technical analysis"""
        await asyncio.sleep(0.1)
        return {'technical_score': 0.7}
    
    async def _perform_fundamental_analysis(self, data):
        """Perform fundamental analysis"""
        await asyncio.sleep(0.1)
        return {'fundamental_score': 0.8}
    
    def get_circuit_breaker(self, service_name: str) -> Optional[EnhancedCircuitBreaker]:
        """Get circuit breaker for a service"""
        return self.circuit_breakers.get(service_name)
    
    def get_pipeline(self, pipeline_name: str) -> Optional[ResilientPipeline]:
        """Get resilient pipeline by name"""
        return self.pipelines.get(pipeline_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'initialized': self._initialized,
            'running': self._running,
            'uptime_seconds': (
                (datetime.now() - self._system_metrics['uptime_start']).total_seconds()
                if self._system_metrics['uptime_start'] else 0
            ),
            'metrics': self._system_metrics,
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb._failure_count,
                    'reliability_score': cb.reliability_tracker.reliability_score
                }
                for name, cb in self.circuit_breakers.items()
            },
            'pipelines': {
                name: pipeline.get_health_status()
                for name, pipeline in self.pipelines.items()
            },
            'health_manager': (
                self.health_manager.get_health_status()
                if self.health_manager else None
            ),
            'disaster_recovery': (
                self.disaster_recovery.get_system_status()
                if self.disaster_recovery else None
            ),
            'chaos_engineering': (
                self.chaos_orchestrator.get_all_experiment_results()
                if self.chaos_orchestrator else None
            ),
            'logging_system': (
                self.logging_system.get_system_metrics()
                if self.logging_system else None
            )
        }


class ResilienceSystemError(InvestmentAnalysisException):
    """Exception raised by the resilience system"""
    pass


# Global resilience system instance
_resilience_system: Optional[InvestmentAnalysisResilienceSystem] = None


async def initialize_resilience_system(config: ResilienceConfiguration = None) -> InvestmentAnalysisResilienceSystem:
    """Initialize the complete resilience system"""
    global _resilience_system
    
    if config is None:
        config = ResilienceConfiguration()
    
    _resilience_system = InvestmentAnalysisResilienceSystem(config)
    await _resilience_system.initialize()
    
    return _resilience_system


async def start_resilience_system():
    """Start the resilience system"""
    if not _resilience_system:
        raise ResilienceSystemError("Resilience system not initialized")
    
    await _resilience_system.start()


async def shutdown_resilience_system():
    """Shutdown the resilience system"""
    if _resilience_system:
        await _resilience_system.shutdown()


def get_resilience_system() -> Optional[InvestmentAnalysisResilienceSystem]:
    """Get the global resilience system instance"""
    return _resilience_system


def get_circuit_breaker(service_name: str) -> Optional[EnhancedCircuitBreaker]:
    """Get circuit breaker for a service"""
    if _resilience_system:
        return _resilience_system.get_circuit_breaker(service_name)
    return None


def get_resilient_pipeline(pipeline_name: str) -> Optional[ResilientPipeline]:
    """Get resilient pipeline by name"""
    if _resilience_system:
        return _resilience_system.get_pipeline(pipeline_name)
    return None


# Example usage and configuration
async def setup_investment_app_resilience():
    """Set up resilience for the investment analysis application"""
    
    # Create configuration
    config = ResilienceConfiguration(
        # Enable all resilience features
        enable_circuit_breakers=True,
        enable_error_correlation=True,
        enable_resilient_pipelines=True,
        enable_health_monitoring=True,
        enable_disaster_recovery=True,
        
        # Chaos engineering disabled by default in production
        enable_chaos_engineering=False,
        chaos_experiments_enabled=False,
        
        # Logging configuration
        log_level=LogLevel.INFO,
        enable_elasticsearch_logging=True,
        elasticsearch_hosts=['localhost:9200'],
        
        # Pipeline settings for handling 6000+ stocks
        pipeline_max_concurrent_tasks=20,
        pipeline_enable_checkpointing=True,
        
        # Health monitoring
        health_check_interval=30,
        
        # Disaster recovery
        backup_retention_days=30,
        enable_auto_recovery=True
    )
    
    # Initialize and start resilience system
    resilience_system = await initialize_resilience_system(config)
    await start_resilience_system()
    
    logger.info("Investment Analysis Resilience System fully operational")
    
    return resilience_system