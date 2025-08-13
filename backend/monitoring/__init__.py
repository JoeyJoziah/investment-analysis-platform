"""
Comprehensive Monitoring System Integration
Initializes and coordinates all monitoring components.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from backend.config.monitoring_config import monitoring_config, initialize_monitoring
from backend.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class MonitoringSystem:
    """
    Main monitoring system coordinator.
    """
    
    def __init__(self):
        self.initialized = False
        self._components = {}
    
    async def initialize(self) -> bool:
        """Initialize all monitoring components."""
        try:
            if self.initialized:
                logger.info("Monitoring system already initialized")
                return True
            
            logger.info("Initializing comprehensive monitoring system...")
            
            # Initialize configuration
            config = initialize_monitoring()
            logger.info(f"Monitoring configuration loaded: {config.logging.service_name}")
            
            # Initialize metrics collection
            from backend.monitoring.metrics_collector import metrics_collector, setup_metrics_endpoint
            await metrics_collector.start_collection()
            self._components['metrics_collector'] = metrics_collector
            logger.info("✓ Metrics collection initialized")
            
            # Initialize database performance monitoring
            if monitoring_config.enable_monitoring:
                from backend.monitoring.database_performance import setup_database_monitoring
                await setup_database_monitoring()
                logger.info("✓ Database performance monitoring initialized")
            
            # Initialize application-specific monitoring
            from backend.monitoring.application_monitoring import setup_application_monitoring
            await setup_application_monitoring()
            self._components['application_monitor'] = 'initialized'
            logger.info("✓ Application monitoring initialized")
            
            # Initialize financial monitoring
            from backend.monitoring.financial_monitoring import setup_financial_monitoring
            await setup_financial_monitoring()
            self._components['financial_monitor'] = 'initialized'
            logger.info("✓ Financial monitoring initialized")
            
            # Initialize health monitoring
            from backend.monitoring.health_checks import setup_health_monitoring
            await setup_health_monitoring()
            self._components['health_monitor'] = 'initialized'
            logger.info("✓ Health monitoring initialized")
            
            # Initialize alerting system
            from backend.monitoring.alerting_system import setup_alerting_system
            await setup_alerting_system()
            self._components['alerting_system'] = 'initialized'
            logger.info("✓ Alerting system initialized")
            
            # Initialize log analysis
            from backend.monitoring.log_analysis import setup_log_analysis
            await setup_log_analysis()
            self._components['log_analysis'] = 'initialized'
            logger.info("✓ Log analysis initialized")
            
            self.initialized = True
            logger.info("🎉 Comprehensive monitoring system initialized successfully!")
            
            # Log system summary
            await self._log_system_summary()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}", exc_info=True)
            return False
    
    async def _log_system_summary(self):
        """Log monitoring system summary."""
        try:
            summary = {
                'monitoring_enabled': monitoring_config.enable_monitoring,
                'components_initialized': len(self._components),
                'metrics_collection_interval': monitoring_config.metrics_collection_interval,
                'health_check_interval': monitoring_config.health_check_interval,
                'cost_monitoring_enabled': monitoring_config.enable_cost_monitoring,
                'compliance_monitoring_enabled': monitoring_config.enable_compliance_monitoring,
                'monthly_budget': monitoring_config.monthly_budget,
                'emergency_threshold': monitoring_config.emergency_mode_threshold
            }
            
            logger.info("Monitoring system summary", extra=summary)
        
        except Exception as e:
            logger.error(f"Error logging system summary: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown all monitoring components."""
        try:
            logger.info("Shutting down monitoring system...")
            
            # Stop metrics collection
            if 'metrics_collector' in self._components:
                await self._components['metrics_collector'].stop_collection()
                logger.info("✓ Metrics collection stopped")
            
            # Stop health monitoring
            try:
                from backend.monitoring.health_checks import health_monitor
                await health_monitor.stop_monitoring()
                logger.info("✓ Health monitoring stopped")
            except Exception as e:
                logger.warning(f"Error stopping health monitoring: {e}")
            
            # Stop database monitoring
            try:
                from backend.monitoring.database_performance import db_performance_monitor
                await db_performance_monitor.stop_monitoring()
                logger.info("✓ Database monitoring stopped")
            except Exception as e:
                logger.warning(f"Error stopping database monitoring: {e}")
            
            # Stop alerting system
            try:
                from backend.monitoring.alerting_system import alert_manager
                await alert_manager.stop()
                logger.info("✓ Alerting system stopped")
            except Exception as e:
                logger.warning(f"Error stopping alerting system: {e}")
            
            # Stop log analysis
            try:
                from backend.monitoring.log_analysis import log_analysis_system
                await log_analysis_system.stop_analysis()
                logger.info("✓ Log analysis stopped")
            except Exception as e:
                logger.warning(f"Error stopping log analysis: {e}")
            
            self.initialized = False
            self._components.clear()
            
            logger.info("🛑 Monitoring system shutdown completed")
        
        except Exception as e:
            logger.error(f"Error during monitoring system shutdown: {e}")
    
    def get_status(self) -> dict:
        """Get monitoring system status."""
        return {
            'initialized': self.initialized,
            'components': list(self._components.keys()),
            'component_count': len(self._components),
            'config_loaded': monitoring_config is not None,
            'timestamp': datetime.now().isoformat()
        }


# Global monitoring system instance
monitoring_system = MonitoringSystem()


# Convenience functions for FastAPI integration
async def setup_monitoring_for_app(app):
    """Setup monitoring for FastAPI application."""
    try:
        # Initialize monitoring system
        success = await monitoring_system.initialize()
        if not success:
            logger.error("Failed to initialize monitoring system")
            return False
        
        # Setup metrics endpoint
        from backend.monitoring.metrics_collector import setup_metrics_endpoint
        setup_metrics_endpoint(app)
        
        # Setup API performance middleware
        from backend.monitoring.api_performance import setup_api_monitoring
        setup_api_monitoring(app)
        
        # Add health check endpoint
        @app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            try:
                from backend.monitoring.health_checks import get_system_health
                health = await get_system_health()
                return health
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        @app.get("/api/monitoring/status")
        async def monitoring_status():
            """Get monitoring system status."""
            return monitoring_system.get_status()
        
        # Setup startup and shutdown events
        @app.on_event("startup")
        async def startup_monitoring():
            """Ensure monitoring is started."""
            if not monitoring_system.initialized:
                await monitoring_system.initialize()
        
        @app.on_event("shutdown")
        async def shutdown_monitoring():
            """Gracefully shutdown monitoring."""
            await monitoring_system.shutdown()
        
        logger.info("✅ Monitoring system integrated with FastAPI application")
        return True
    
    except Exception as e:
        logger.error(f"Failed to setup monitoring for app: {e}")
        return False


__version__ = "2.0.0"
__all__ = [
    'MonitoringSystem',
    'monitoring_system',
    'setup_monitoring_for_app'
]