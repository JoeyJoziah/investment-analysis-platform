# API Rate Limit Exceeded Recovery Runbook

## Overview
This runbook provides procedures for handling external API rate limit exhaustion in the Investment Analysis Application. Use this when API providers (Alpha Vantage, Finnhub, Polygon) return rate limit errors or when internal cost monitoring triggers emergency mode.

## Prerequisites
- Access to application logs and monitoring
- Understanding of API tier limits and costs
- Access to Redis cache for fallback data

## Immediate Detection

### 1. Identifying Rate Limit Issues
```bash
# Check for rate limit errors in logs
make logs | grep -i "rate.*limit\|quota.*exceeded\|429"

# Check cost monitoring status
curl -s http://localhost:8000/api/admin/cost-status | jq .

# Check API call counts for today
docker-compose exec redis redis-cli GET "api_calls:alpha_vantage:$(date +%Y%m%d)"
docker-compose exec redis redis-cli GET "api_calls:finnhub:$(date +%Y%m%d)"
docker-compose exec redis redis-cli GET "api_calls:polygon:$(date +%Y%m%d)"
```

### 2. Check Emergency Mode Status
```bash
# Check if system is in emergency mode
docker-compose exec backend python -c "
from backend.utils.enhanced_cost_monitor import enhanced_cost_monitor
print(f'Emergency mode: {enhanced_cost_monitor.emergency_mode}')
print(f'Monthly spend: ${enhanced_cost_monitor.get_monthly_cost():.2f}')
print(f'API limits: {enhanced_cost_monitor.get_api_usage_summary()}')
"
```

## Recovery Procedures by Provider

### Alpha Vantage (25 calls/day, 5 calls/minute)

#### When Daily Limit Exceeded
```bash
# Check current usage
docker-compose exec redis redis-cli GET "av_daily_calls:$(date +%Y%m%d)"

# Enable extended cache mode
docker-compose exec backend python -c "
from backend.utils.enhanced_cost_monitor import enhanced_cost_monitor
enhanced_cost_monitor.enable_extended_cache()
print('Extended cache mode enabled')
"

# Switch to fallback providers for urgent data
docker-compose exec backend python -c "
from backend.data_ingestion.provider_manager import provider_manager
provider_manager.set_fallback_provider('finnhub')
print('Switched to Finnhub as primary provider')
"

# Verify stale cache availability
docker-compose exec redis redis-cli KEYS "av:*:stale" | head -10
```

#### When Minute Limit Exceeded
```bash
# System should automatically handle this with 12-second delays
# Check if delays are working properly
make logs | grep -i "alpha.*rate.*pause" | tail -5

# If still getting errors, extend the delay
docker-compose exec backend python -c "
import json
import asyncio
from backend.utils.cache import get_redis

async def set_extended_delay():
    redis = await get_redis()
    await redis.set('av_delay_override', '15', ex=3600)  # 15 seconds for 1 hour
    print('Extended Alpha Vantage delay to 15 seconds')

asyncio.run(set_extended_delay())
"
```

### Finnhub (60 calls/minute free tier)

#### When Rate Limited
```bash
# Check current minute usage
docker-compose exec redis redis-cli GET "fh_minute_calls:$(date +%Y%m%d%H%M)"

# Switch to batch processing mode
docker-compose exec backend python -c "
from backend.utils.integration import unified_ingestion
unified_ingestion.enable_batch_mode(delay=2)  # 2-second delays between calls
print('Enabled batch mode for Finnhub')
"

# Use Alpha Vantage for essential data
docker-compose exec backend python -c "
from backend.data_ingestion.provider_manager import provider_manager
provider_manager.prioritize_provider('alpha_vantage', ['quote', 'overview'])
print('Prioritized Alpha Vantage for essential data')
"
```

### Polygon (5 calls/minute free tier)

#### When Rate Limited
```bash
# This provider should already have conservative limits
# Check if rate limiting is working
make logs | grep -i "polygon.*rate.*limit" | tail -5

# Disable non-essential Polygon calls
docker-compose exec backend python -c "
from backend.data_ingestion.polygon_client import PolygonClient
# Temporarily disable historical data fetching
import json
from backend.utils.cache import get_redis
import asyncio

async def disable_polygon_historical():
    redis = await get_redis()
    await redis.set('polygon_disable_historical', '1', ex=7200)  # 2 hours
    print('Disabled Polygon historical data fetching')

asyncio.run(disable_polygon_historical())
"
```

## Emergency Cost Management

### When Monthly Budget Exceeded

#### Activate Emergency Mode
```bash
# Check current monthly spend
docker-compose exec backend python -c "
from backend.utils.enhanced_cost_monitor import enhanced_cost_monitor
cost = enhanced_cost_monitor.get_monthly_cost()
print(f'Current monthly cost: ${cost:.2f}')
if cost > 45:  # $45 threshold
    enhanced_cost_monitor.activate_emergency_mode()
    print('Emergency mode activated')
"

# Switch to cache-only mode
docker-compose exec backend python -c "
from backend.utils.cache import cache_manager
cache_manager.enable_cache_only_mode()
print('Cache-only mode enabled - no external API calls')
"

# Verify cache coverage
docker-compose exec redis redis-cli INFO keyspace
docker-compose exec redis redis-cli KEYS "*:stale" | wc -l
```

#### Prioritize Critical Data
```bash
# Configure tiered data access
docker-compose exec backend python -c "
from backend.data_ingestion.provider_manager import provider_manager
from backend.utils.enhanced_cost_monitor import enhanced_cost_monitor

# Only fetch Tier 1 stocks (S&P 500)
enhanced_cost_monitor.set_emergency_tiers(['1'])

# Limit to essential data types
provider_manager.set_essential_only(['quote', 'overview'])

print('Emergency tiering activated')
"

# Extend cache TTLs to maximum
docker-compose exec backend python -c "
from backend.utils.advanced_cache import advanced_cache
advanced_cache.extend_all_ttls(multiplier=10)
print('Extended all cache TTLs by 10x')
"
```

## Fallback Data Strategies

### 1. Use Stale Cache Data
```bash
# Check available stale data
docker-compose exec redis redis-cli KEYS "*:stale" | head -20

# Enable stale data serving
docker-compose exec backend python -c "
from backend.utils.advanced_cache import advanced_cache
advanced_cache.enable_stale_data_serving()
print('Stale data serving enabled')
"

# Verify stale data is being served
curl -s http://localhost:8000/api/stocks/AAPL | jq '.metadata.cache_status'
```

### 2. Switch to Alternative Data Sources
```bash
# Enable free data sources
docker-compose exec backend python -c "
from backend.data_ingestion.provider_manager import provider_manager

# Configure free-tier optimized settings
provider_manager.configure_free_tier_mode()
print('Configured for free-tier operation')
"

# Use Yahoo Finance as backup (if implemented)
docker-compose exec backend python -c "
# This would require implementing a YahooFinanceClient
print('Consider implementing Yahoo Finance fallback')
"
```

### 3. Reduce Update Frequency
```bash
# Change update intervals
docker-compose exec backend python -c "
from backend.tasks.scheduler import scheduler

# Reduce update frequency
scheduler.set_emergency_schedule({
    'tier_1': '4h',    # Every 4 hours instead of hourly
    'tier_2': '8h',    # Every 8 hours instead of 4h
    'tier_3': '24h',   # Daily instead of 8h
    'tier_4': '72h',   # Every 3 days
    'tier_5': '168h'   # Weekly
})
print('Emergency schedule activated')
"
```

## Monitoring and Verification

### 1. Verify Rate Limit Compliance
```bash
# Check recent API call patterns
docker-compose exec backend python -c "
from backend.utils.enhanced_cost_monitor import enhanced_cost_monitor
usage = enhanced_cost_monitor.get_hourly_api_usage()
for hour, calls in usage.items():
    print(f'{hour}: {calls} calls')
"

# Monitor response codes
make logs | grep -E "HTTP [4-5][0-9][0-9]" | tail -10
```

### 2. Verify Cache Performance
```bash
# Check cache hit rates
docker-compose exec backend python -c "
from backend.utils.advanced_cache import advanced_cache
stats = advanced_cache.get_cache_stats()
print(f'Hit rate: {stats.get(\"hit_rate\", 0):.2f}%')
print(f'Total requests: {stats.get(\"total_requests\", 0)}')
"

# Check stale cache usage
docker-compose exec redis redis-cli INFO stats | grep keyspace_hits
```

### 3. Data Quality Assessment
```bash
# Check data freshness
docker-compose exec backend python -c "
from backend.utils.data_quality import DataQualityChecker
checker = DataQualityChecker()
freshness = checker.check_data_freshness()
print(f'Average data age: {freshness.get(\"avg_age_hours\", 0):.1f} hours')
print(f'Stale data percentage: {freshness.get(\"stale_percentage\", 0):.1f}%')
"

# Verify essential data availability
curl -s http://localhost:8000/api/stocks?tier=1 | jq '.[] | {symbol, last_updated}'
```

## Recovery Actions

### 1. Gradual Service Restoration
```bash
# After rate limits reset (next day/hour), gradually restore
docker-compose exec backend python -c "
from backend.utils.enhanced_cost_monitor import enhanced_cost_monitor
if enhanced_cost_monitor.check_rate_limit_reset():
    enhanced_cost_monitor.disable_emergency_mode()
    print('Rate limits reset - emergency mode disabled')
else:
    print('Rate limits still active')
"

# Re-enable normal update schedules
docker-compose exec backend python -c "
from backend.tasks.scheduler import scheduler
scheduler.restore_normal_schedule()
print('Normal update schedule restored')
"
```

### 2. Cache Warming
```bash
# Warm cache with essential data
docker-compose exec backend python -c "
from backend.utils.cache_warming import cache_warmer
import asyncio

async def warm_essential_cache():
    # Warm cache for top 100 stocks
    await cache_warmer.warm_essential_data(limit=100)
    print('Essential cache warmed')

asyncio.run(warm_essential_cache())
"
```

## Prevention Measures

### 1. Enhanced Monitoring
```bash
# Set up proactive alerts
docker-compose exec backend python -c "
from backend.utils.enhanced_cost_monitor import enhanced_cost_monitor

# Set conservative thresholds
enhanced_cost_monitor.set_alert_thresholds({
    'daily_cost': 1.50,      # $1.50/day = $45/month
    'alpha_vantage_daily': 20, # 20/25 calls
    'finnhub_minute': 50,     # 50/60 calls
    'polygon_minute': 4       # 4/5 calls
})
print('Conservative alert thresholds set')
"
```

### 2. Intelligent Caching
```bash
# Enable predictive cache warming
docker-compose exec backend python -c "
from backend.utils.advanced_cache import advanced_cache
advanced_cache.enable_predictive_warming()
print('Predictive cache warming enabled')
"

# Set up cache analytics
docker-compose exec backend python -c "
from backend.utils.cache_analytics import cache_analytics
cache_analytics.enable_usage_tracking()
print('Cache usage tracking enabled')
"
```

## Related Runbooks
- [Cost Monitoring](./cost-monitoring.md)
- [Performance Tuning](./performance-tuning.md)
- [Cache Management](./cache-management.md)
- [Service Down Recovery](./service-down-recovery.md)