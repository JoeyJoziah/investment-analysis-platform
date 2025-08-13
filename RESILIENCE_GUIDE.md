# Investment Analysis App - Comprehensive Error Handling & Resilience Guide

## Overview

This guide covers the comprehensive error handling and resilience features implemented for the investment analysis application. The system is designed to handle 6,000+ stocks daily while maintaining stability, respecting budget constraints, and providing robust recovery capabilities.

## Architecture Overview

The resilience system consists of seven integrated components:

1. **Advanced Circuit Breaker System** - Prevents cascading failures with adaptive thresholds
2. **Comprehensive Error Classification** - Intelligent error categorization and correlation
3. **Resilient Data Pipeline** - Fault-tolerant data processing with recovery
4. **Service Health Management** - Comprehensive health monitoring and auto-recovery
5. **Disaster Recovery** - Automated backup, validation, and restoration
6. **Chaos Engineering** - Resilience testing and validation
7. **Enhanced Logging** - Structured logging with correlation and analysis

## Quick Start

### 1. Initialize the Complete Resilience System

```python
from backend.utils.resilience_integration import (
    ResilienceConfiguration,
    setup_investment_app_resilience
)

# Set up resilience system with default production settings
resilience_system = await setup_investment_app_resilience()

# Or with custom configuration
config = ResilienceConfiguration(
    enable_circuit_breakers=True,
    enable_error_correlation=True,
    enable_resilient_pipelines=True,
    enable_health_monitoring=True,
    enable_disaster_recovery=True,
    log_level=LogLevel.INFO,
    pipeline_max_concurrent_tasks=20
)

from backend.utils.resilience_integration import initialize_resilience_system, start_resilience_system
resilience_system = await initialize_resilience_system(config)
await start_resilience_system()
```

### 2. Basic Usage in Application Code

```python
from backend.utils.resilience_integration import get_circuit_breaker, get_resilient_pipeline
from backend.utils.enhanced_logging import get_logger, correlation_context
from backend.utils.enhanced_error_handling import with_error_handling

# Get components
logger = get_logger(__name__)
circuit_breaker = get_circuit_breaker('finnhub_api')
pipeline = get_resilient_pipeline('data_ingestion')

@with_error_handling(service="data_ingestion", critical_path=True)
async def fetch_stock_data(ticker: str):
    # Use correlation context for request tracing
    with correlation_context(f"fetch_stock_data_{ticker}", user_id="system"):
        
        # Use circuit breaker for external API calls
        if circuit_breaker:
            return await circuit_breaker.call(api_client.get_stock_data, ticker)
        else:
            return await api_client.get_stock_data(ticker)
```

## Component Details

### 1. Advanced Circuit Breaker System

#### Features
- **Adaptive Thresholds**: Automatically adjusts failure thresholds based on provider reliability
- **Multiple Failure Types**: Handles timeouts, rate limits, server errors, network issues
- **Cascading Failure Prevention**: Isolates services to prevent system-wide failures
- **Intelligent Recovery**: Uses jitter and exponential backoff for recovery attempts

#### Usage

```python
from backend.utils.advanced_circuit_breaker import EnhancedCircuitBreaker, AdaptiveThresholds

# Create circuit breaker for external API
thresholds = AdaptiveThresholds(
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=2,
    rate_limit_threshold=3,
    timeout_threshold=3,
    error_rate_threshold=0.3
)

circuit_breaker = EnhancedCircuitBreaker(
    name="finnhub_api",
    base_thresholds=thresholds,
    fallback_func=use_cached_data
)

# Use in API calls
try:
    data = await circuit_breaker.call(fetch_from_finnhub, ticker)
except CircuitBreakerError:
    # Circuit is open, use fallback
    data = await use_cached_data(ticker)
```

#### Monitoring

```python
# Get circuit breaker metrics
metrics = circuit_breaker.get_comprehensive_metrics()
print(f"State: {metrics['state']}")
print(f"Reliability Score: {metrics['reliability_score']}")
print(f"Recent Error Rate: {metrics['metrics']['recent_error_rate_5min']}")
```

### 2. Comprehensive Error Classification

#### Features
- **Intelligent Categorization**: Automatically classifies errors by type and severity
- **Error Correlation**: Links related errors across services using correlation IDs
- **Pattern Recognition**: Detects error patterns and provides recovery recommendations
- **Root Cause Analysis**: Identifies likely root causes of cascading failures

#### Usage

```python
from backend.utils.enhanced_error_handling import with_error_handling, error_handler

@with_error_handling(service="recommendation_engine", operation="generate_recommendations")
async def generate_recommendations(user_id: str):
    try:
        # Your business logic here
        recommendations = await compute_recommendations(user_id)
        return recommendations
    except Exception as e:
        # Error is automatically classified and correlated
        # Recovery strategies are suggested
        raise

# Get error analytics
analytics = error_handler.get_error_analytics(time_window_hours=24)
print(f"Total errors: {analytics['total_errors']}")
print(f"Top error categories: {analytics['top_error_categories']}")
```

### 3. Resilient Data Pipeline

#### Features
- **Fault Tolerance**: Handles partial failures with graceful degradation
- **Intelligent Retry**: Uses exponential backoff with jitter
- **Data Quality Validation**: Validates data integrity throughout processing
- **Checkpointing**: Saves progress to enable resume after failures
- **Cache Integration**: Uses multi-tier caching for efficiency

#### Usage

```python
from backend.utils.resilient_pipeline import ResilientPipeline, TaskExecutor

# Create pipeline for stock data processing
pipeline = ResilientPipeline(
    name="stock_analysis_pipeline",
    max_concurrent_tasks=10,
    enable_checkpointing=True,
    enable_caching=True
)

# Register executors
pipeline.register_executor(
    "fetch_data",
    fetch_stock_data_executor,
    max_retries=3,
    circuit_breaker_config={
        'failure_threshold': 5,
        'recovery_timeout': 30
    }
)

# Add tasks to pipeline
task_id = await pipeline.add_task(
    task_id="AAPL_analysis",
    data={"ticker": "AAPL"},
    stage_name="fetch_data",
    priority=1
)

# Start pipeline
await pipeline.start()

# Monitor progress
status = pipeline.get_health_status()
print(f"Success rate: {status['success_rate']}")
print(f"Throughput: {status['throughput_per_minute']} tasks/min")
```

### 4. Service Health Management

#### Features
- **Dependency Tracking**: Monitors all external dependencies
- **Resource Monitoring**: Tracks CPU, memory, disk, network usage
- **Automatic Recovery**: Executes recovery actions based on health status
- **Bulkhead Pattern**: Isolates services to prevent cascading failures
- **Health Scoring**: Provides comprehensive health scores

#### Usage

```python
from backend.utils.service_health_manager import ServiceHealthManager, ServiceConfig, DependencyConfig

# Configure service health monitoring
service_config = ServiceConfig(
    name="investment_analysis_service",
    check_interval_seconds=30,
    dependencies=[
        DependencyConfig(
            name="postgres_database",
            dependency_type=DependencyType.DATABASE,
            endpoint="postgresql://localhost:5432/investment_db",
            timeout_seconds=5.0,
            critical=True
        )
    ],
    resource_thresholds={
        'cpu_critical': 85.0,
        'memory_critical': 90.0
    }
)

health_manager = ServiceHealthManager(service_config)
await health_manager.start_monitoring()

# Get health status
health = health_manager.get_health_status()
print(f"Overall status: {health['overall_status']}")
print(f"Uptime: {health['uptime_seconds']} seconds")
```

### 5. Disaster Recovery

#### Features
- **Automated Backups**: Scheduled backups with retention policies
- **Data Validation**: Verifies backup integrity
- **Recovery Orchestration**: Automated recovery procedures
- **Multiple Recovery Strategies**: Full, incremental, and point-in-time recovery
- **Cloud Integration**: S3 backup support

#### Usage

```python
from backend.utils.disaster_recovery import DisasterRecoverySystem, BackupType

# Initialize disaster recovery
dr_system = DisasterRecoverySystem({
    'backup_root': 'data/backups',
    'retention_days': 30,
    's3_bucket': 'investment-app-backups'
})

await dr_system.initialize()

# Register backup sources
await dr_system.backup_manager.register_backup_source(
    "database_backup",
    "/var/lib/postgresql/data",
    BackupType.FULL,
    "daily",
    retention_days=7
)

# Manual backup
backup_metadata = await dr_system.backup_manager.create_manual_backup(
    "emergency_backup",
    "/path/to/critical/data",
    BackupType.FULL
)

# Initiate recovery if needed
from backend.utils.disaster_recovery import DisasterType
recovery_id = await dr_system.orchestrator.initiate_recovery(
    DisasterType.DATABASE_FAILURE
)
```

### 6. Chaos Engineering

#### Features
- **Fault Injection**: Simulates various failure scenarios
- **Safety Validation**: Ensures experiments don't cause real damage
- **Automated Testing**: Validates system resilience
- **Comprehensive Monitoring**: Tracks system behavior during experiments
- **Recovery Validation**: Ensures proper recovery procedures

#### Usage

```python
from backend.utils.chaos_engineering import (
    ChaosExperiment, ChaosExperimentType, ImpactScope
)

# Create chaos experiment
experiment = ChaosExperiment(
    experiment_id="db_failure_test",
    experiment_type=ChaosExperimentType.DATABASE_FAILURE,
    name="Database Connection Failure Test",
    description="Test system resilience when database connections fail",
    impact_scope=ImpactScope.DATA_LAYER,
    target_services=["database"],
    duration_seconds=300,
    intensity=1.0,
    parameters={'failure_type': 'connection_drop'},
    safety_checks=['system_healthy', 'business_hours_only'],
    success_criteria=['system_recovers_within_5_minutes']
)

# Register and run experiment
from backend.utils.chaos_engineering import chaos_orchestrator
await chaos_orchestrator.register_experiment(experiment)
execution_id = await chaos_orchestrator.run_experiment("db_failure_test")

# Monitor experiment
status = chaos_orchestrator.get_experiment_status(execution_id)
```

### 7. Enhanced Logging

#### Features
- **Structured Logging**: JSON format with rich metadata
- **Correlation Tracking**: Traces requests across service boundaries
- **Pattern Analysis**: Detects log patterns and generates alerts
- **Real-time Monitoring**: Live log analysis and alerting
- **Elasticsearch Integration**: Centralized log storage and search

#### Usage

```python
from backend.utils.enhanced_logging import get_logger, correlation_context

logger = get_logger(__name__)

# Use correlation context for request tracing
with correlation_context("process_user_request", user_id="12345"):
    logger.info("Processing user request", 
                custom_fields={'request_type': 'recommendation'},
                tags=['user_request'])
    
    try:
        result = await process_request()
        logger.info("Request processed successfully",
                   custom_fields={'result_count': len(result)})
    except Exception as e:
        logger.error("Request processing failed", error=e)
        raise

# Performance logging
operation_id = logger.start_operation("complex_calculation")
# ... do work ...
logger.end_operation(operation_id, "complex_calculation", success=True)

# Business event logging
logger.business("New user registered",
               custom_fields={'user_id': '12345', 'plan': 'premium'})

# Security logging
logger.security("Failed login attempt",
                custom_fields={'ip_address': '192.168.1.1', 'username': 'admin'})
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_db
DB_USER=postgres
DB_PASSWORD=your_password

# Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# API Keys (for external services)
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_av_key
POLYGON_API_KEY=your_polygon_key

# Logging
LOG_LEVEL=INFO
ELASTICSEARCH_HOSTS=localhost:9200

# Backup
BACKUP_S3_BUCKET=investment-app-backups
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret

# Resilience Settings
CIRCUIT_BREAKER_ENABLED=true
DISASTER_RECOVERY_ENABLED=true
CHAOS_ENGINEERING_ENABLED=false
```

### Configuration Files

#### `config/resilience_config.json`
```json
{
  "circuit_breakers": {
    "finnhub_api": {
      "failure_threshold": 5,
      "recovery_timeout": 60,
      "rate_limit_threshold": 3
    },
    "alpha_vantage_api": {
      "failure_threshold": 3,
      "recovery_timeout": 120,
      "rate_limit_threshold": 2
    }
  },
  "pipelines": {
    "data_ingestion": {
      "max_concurrent_tasks": 20,
      "enable_checkpointing": true,
      "checkpoint_interval": 100
    }
  },
  "health_monitoring": {
    "check_interval_seconds": 30,
    "resource_thresholds": {
      "cpu_critical": 85.0,
      "memory_critical": 90.0,
      "disk_critical": 95.0
    }
  }
}
```

## Monitoring and Alerting

### Health Dashboards

The system provides comprehensive health dashboards showing:

- **Circuit Breaker Status**: Real-time state of all circuit breakers
- **Pipeline Health**: Processing rates, error rates, queue sizes
- **Resource Usage**: CPU, memory, disk, network utilization
- **Error Analytics**: Error patterns, correlation analysis, trending
- **Recovery Operations**: Active and historical recovery operations

### Alert Configuration

```python
from backend.utils.enhanced_logging import get_logging_system

# Register custom alert handlers
def slack_alert_handler(alert_data):
    # Send alert to Slack
    send_slack_message(f"🚨 {alert_data['pattern_name']}: {alert_data['match_count']} occurrences")

def pagerduty_alert_handler(alert_data):
    if alert_data['severity'] in ['ERROR', 'CRITICAL']:
        # Trigger PagerDuty incident
        trigger_pagerduty_incident(alert_data)

logging_system = get_logging_system()
logging_system.register_alert_callback(slack_alert_handler)
logging_system.register_alert_callback(pagerduty_alert_handler)
```

## Best Practices

### 1. Error Handling

```python
# Always use error handling decorators for critical functions
@with_error_handling(service="data_ingestion", operation="fetch_stock_data", critical_path=True)
async def fetch_stock_data(ticker: str):
    # Use circuit breakers for external calls
    circuit_breaker = get_circuit_breaker('finnhub_api')
    return await circuit_breaker.call(external_api_call, ticker)

# Provide meaningful error context
try:
    result = await risky_operation()
except Exception as e:
    logger.error("Operation failed", 
                error=e,
                custom_fields={
                    'operation': 'risky_operation',
                    'input_data': input_data,
                    'attempt_number': retry_count
                })
    raise
```

### 2. Circuit Breaker Usage

```python
# Always check circuit breaker state before expensive operations
circuit_breaker = get_circuit_breaker('external_service')
if circuit_breaker.is_open:
    # Use cached data or alternative service
    return await get_cached_data()

# Use with context manager for automatic cleanup
async with circuit_breaker:
    result = await external_service_call()
    return result
```

### 3. Pipeline Design

```python
# Design pipeline tasks to be idempotent
async def process_stock_data(stock_data):
    # Check if already processed
    if await is_already_processed(stock_data['ticker'], stock_data['date']):
        return await get_processed_result(stock_data['ticker'], stock_data['date'])
    
    # Process data
    result = await perform_analysis(stock_data)
    
    # Store result
    await store_result(result)
    return result

# Use appropriate retry strategies
pipeline.register_executor(
    "api_call_task",
    api_executor,
    max_retries=3,  # Retry for transient failures
    circuit_breaker_config=cb_config
)

pipeline.register_executor(
    "data_processing_task", 
    processing_executor,
    max_retries=1  # Don't retry processing failures
)
```

### 4. Logging Best Practices

```python
# Use correlation contexts for request tracing
with correlation_context("user_portfolio_analysis", user_id=user_id):
    logger.info("Starting portfolio analysis")
    
    # All logs within this context will have the same correlation_id
    result = await analyze_portfolio(user_id)
    
    logger.info("Portfolio analysis completed",
               custom_fields={'portfolio_value': result.total_value})

# Log performance metrics
start_time = time.time()
result = await expensive_operation()
duration = time.time() - start_time

logger.performance("Expensive operation completed",
                  duration_ms=duration * 1000,
                  custom_fields={'result_size': len(result)})
```

## Troubleshooting

### Common Issues

1. **Circuit Breakers Frequently Opening**
   - Check external service reliability
   - Adjust failure thresholds
   - Review retry strategies
   - Monitor for rate limiting

2. **Pipeline Bottlenecks**
   - Increase concurrent task limits
   - Optimize task execution time
   - Check for resource constraints
   - Review data quality issues

3. **High Error Rates**
   - Review error classification patterns
   - Check correlation analysis
   - Validate data sources
   - Review retry configurations

4. **Recovery Failures**
   - Check backup integrity
   - Validate recovery procedures
   - Review dependency health
   - Check resource availability

### Debugging Tools

```python
# Get comprehensive system status
from backend.utils.resilience_integration import get_resilience_system

resilience_system = get_resilience_system()
status = resilience_system.get_system_status()

print("System Status:", json.dumps(status, indent=2, default=str))

# Analyze error patterns
error_summary = resilience_system.error_handler.get_error_analytics(hours=24)
print("Error Analysis:", json.dumps(error_summary, indent=2, default=str))

# Check circuit breaker states
for name, cb in resilience_system.circuit_breakers.items():
    metrics = cb.get_comprehensive_metrics()
    print(f"{name}: {metrics['state']} (reliability: {metrics['reliability_score']:.3f})")
```

## Performance Considerations

### Resource Usage

The resilience system is designed to have minimal performance impact:

- **Memory**: < 100MB additional memory usage
- **CPU**: < 5% additional CPU overhead
- **Storage**: Log rotation prevents disk space issues
- **Network**: Minimal additional network traffic

### Scaling

The system scales with your application:

- **Circuit Breakers**: O(1) per service call
- **Pipelines**: Configurable concurrency limits
- **Health Monitoring**: Configurable check intervals
- **Logging**: Asynchronous, buffered output

### Cost Optimization

The system helps optimize costs by:

- **Preventing API rate limit overages**
- **Reducing retry storms through intelligent backoff**
- **Caching successful responses**
- **Early failure detection to prevent resource waste**

## Integration with Existing Code

### Gradual Adoption

You can adopt the resilience features gradually:

1. **Start with Logging**: Initialize enhanced logging first
2. **Add Circuit Breakers**: Protect external API calls
3. **Implement Health Monitoring**: Monitor critical dependencies
4. **Add Error Handling**: Use decorators for error classification
5. **Introduce Pipelines**: Migrate batch processing to resilient pipelines
6. **Enable Disaster Recovery**: Set up backup and recovery procedures
7. **Add Chaos Engineering**: Validate system resilience

### Migration Guide

```python
# Before: Simple API call
async def get_stock_data(ticker):
    response = await httpx.get(f"https://api.finnhub.io/api/v1/quote?symbol={ticker}")
    return response.json()

# After: Resilient API call
@with_error_handling(service="data_ingestion", operation="get_stock_data")
async def get_stock_data(ticker):
    with correlation_context(f"get_stock_data_{ticker}"):
        circuit_breaker = get_circuit_breaker('finnhub_api')
        
        try:
            response = await circuit_breaker.call(
                httpx.get, 
                f"https://api.finnhub.io/api/v1/quote?symbol={ticker}"
            )
            return response.json()
        except CircuitBreakerError:
            # Fallback to cached data
            return await get_cached_stock_data(ticker)
```

This comprehensive resilience system ensures your investment analysis application can handle the challenges of processing 6,000+ stocks daily while maintaining high availability, performance, and cost efficiency.