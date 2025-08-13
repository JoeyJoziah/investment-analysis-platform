# ADR-011: Circuit Breaker Pattern Implementation

**Status**: Accepted  
**Date**: 2025-01-08  
**Deciders**: Development Team, Reliability Team

## Context

The Investment Analysis Application relies heavily on external APIs (Alpha Vantage, Finnhub, Polygon) for real-time financial data. These external dependencies can become unavailable or slow, potentially causing cascading failures throughout the system. Without proper fault tolerance mechanisms:

- API failures can block the entire data ingestion pipeline
- Slow responses can exhaust connection pools and memory
- Repeated failed requests waste API quota and increase costs
- Users experience poor performance even when cached data is available

Traditional retry mechanisms alone are insufficient for handling sustained outages or degraded performance from external services.

## Decision

We will implement the Circuit Breaker pattern across all external API clients and critical internal services. The circuit breaker will:

### 1. Monitor Service Health
- Track failure rates and response times
- Differentiate between different types of failures
- Maintain rolling windows of recent performance metrics

### 2. Implement Three States
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failing fast, requests are immediately rejected
- **HALF-OPEN**: Testing recovery, limited requests allowed

### 3. Provide Fallback Mechanisms
- Serve stale data from cache when circuit is open
- Switch to alternative data providers
- Gracefully degrade functionality rather than complete failure

### 4. Enable Observability
- Emit metrics for monitoring circuit breaker state changes
- Log transitions with context for troubleshooting
- Provide health endpoints for operational visibility

## Implementation Details

### Circuit Breaker Configuration
```python
API_CIRCUIT_BREAKER_CONFIG = {
    'alpha_vantage': {
        'failure_threshold': 5,      # Open after 5 consecutive failures
        'recovery_timeout': 60,      # Test recovery after 60 seconds
        'success_threshold': 2,      # Close after 2 consecutive successes
        'timeout': 30,               # Request timeout in seconds
        'expected_exceptions': (aiohttp.ClientError, asyncio.TimeoutError)
    },
    'finnhub': {
        'failure_threshold': 3,      # More sensitive due to higher usage
        'recovery_timeout': 30,      # Faster recovery testing
        'success_threshold': 2,
        'timeout': 15,
        'expected_exceptions': (aiohttp.ClientError, asyncio.TimeoutError)
    },
    'polygon': {
        'failure_threshold': 5,
        'recovery_timeout': 120,     # Longer recovery for free tier
        'success_threshold': 3,      # More conservative recovery
        'timeout': 45,
        'expected_exceptions': (aiohttp.ClientError, asyncio.TimeoutError)
    }
}
```

### Base Client Integration
All API clients inherit from `BaseAPIClient` which includes circuit breaker functionality:

```python
class BaseAPIClient:
    def __init__(self, provider_name: str):
        self.circuit_breaker = CircuitBreaker(
            name=f"{provider_name}_circuit_breaker",
            **API_CIRCUIT_BREAKER_CONFIG[provider_name]
        )
    
    async def _make_request(self, endpoint: str, params: dict = None):
        try:
            return await self.circuit_breaker.call(
                self._make_request_internal(endpoint, params)
            )
        except CircuitBreakerError:
            # Fallback to stale cache data
            return await self._get_fallback_data(endpoint, params)
```

### Fallback Strategies
1. **Stale Cache Data**: Serve cached data beyond normal TTL
2. **Alternative Providers**: Switch to backup data sources
3. **Graceful Degradation**: Return partial data or indicators of unavailability
4. **Queue for Later**: Queue non-critical requests for when service recovers

### Monitoring Integration
Circuit breaker state changes are emitted as metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

circuit_breaker_state_changes = Counter(
    'circuit_breaker_state_changes_total',
    'Total circuit breaker state changes',
    ['service', 'from_state', 'to_state']
)

circuit_breaker_fallback_requests = Counter(
    'circuit_breaker_fallback_requests_total',
    'Requests served via fallback mechanisms',
    ['service', 'fallback_type']
)

circuit_breaker_current_state = Gauge(
    'circuit_breaker_state',
    'Current circuit breaker state (0=closed, 1=half-open, 2=open)',
    ['service']
)
```

## Consequences

### Positive
- **Improved Resilience**: System continues operating during external service outages
- **Cost Control**: Prevents wasted API calls to failing services
- **Better User Experience**: Fast failures instead of hanging requests
- **Operational Visibility**: Clear metrics and alerts for service health
- **Automatic Recovery**: Self-healing when services recover
- **Resource Protection**: Prevents resource exhaustion from hanging connections

### Negative
- **Increased Complexity**: Additional logic and state management required
- **Potential Data Staleness**: May serve older data during outages
- **Configuration Overhead**: Requires tuning thresholds for each service
- **False Positives**: Transient network issues might trigger unnecessary circuit opening
- **Development Time**: Additional testing and monitoring setup required

### Risks
- **Aggressive Thresholds**: Too sensitive settings may cause unnecessary circuit opening
- **Conservative Thresholds**: Too lenient settings may not protect against real failures  
- **Fallback Data Quality**: Stale data might lead to poor investment decisions
- **Monitoring Blind Spots**: Circuit breaker issues might not be immediately visible

## Mitigation Strategies

### Threshold Tuning
- Start with conservative thresholds and adjust based on observed behavior
- Use different thresholds for different types of requests (critical vs. nice-to-have)
- Implement A/B testing for threshold optimization

### Fallback Data Quality
- Implement data freshness indicators in API responses
- Set maximum age limits for stale data serving
- Provide clear UI indicators when data is from fallback sources

### Monitoring and Alerting
```python
# Alert when circuit breaker opens
circuit_breaker_state_changes.labels(
    service='alpha_vantage', 
    from_state='closed', 
    to_state='open'
).inc()

# Alert when fallback usage is high
if fallback_usage_rate > 0.1:  # >10% of requests using fallback
    send_alert("High fallback usage detected")
```

## Testing Strategy

### Unit Tests
- Test all three circuit breaker states
- Verify fallback mechanisms activate correctly
- Test recovery behavior after failures

### Integration Tests
- Simulate API outages and verify system behavior
- Test cascading failure scenarios
- Verify metrics and logging work correctly

### Load Tests
- Test circuit breaker behavior under high load
- Verify resource protection during failures
- Test recovery patterns with realistic traffic

## Operational Procedures

### Circuit Breaker Management
```bash
# Check circuit breaker status
curl http://localhost:8000/api/health/circuit-breakers

# Manual circuit breaker control
curl -X POST http://localhost:8000/api/admin/circuit-breaker/alpha_vantage/open
curl -X POST http://localhost:8000/api/admin/circuit-breaker/alpha_vantage/close

# View circuit breaker metrics
curl http://localhost:8000/metrics | grep circuit_breaker
```

### Troubleshooting Runbook
1. **Circuit Opens Unexpectedly**: Check service logs, verify external service status
2. **Circuit Won't Close**: Verify success threshold settings, check if service actually recovered
3. **High Fallback Usage**: Investigate root cause, consider adjusting thresholds
4. **Stale Data Issues**: Review fallback data age limits, consider alternative providers

## Related ADRs
- [ADR-009: Cost Optimization Strategy](./009-cost-optimization-strategy.md)
- [ADR-003: Caching Strategy](./003-caching-strategy.md)
- [ADR-015: Error Handling Standards](./015-error-handling-standards.md)
- [ADR-013: Monitoring and Alerting](./013-monitoring-alerting.md)

## Review Schedule
Circuit breaker configurations and thresholds should be reviewed quarterly based on:
- Service reliability patterns
- False positive/negative rates  
- User impact during outages
- Cost implications of fallback strategies