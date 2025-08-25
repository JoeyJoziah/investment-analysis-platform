# Investment Analysis Platform - Comprehensive Monitoring Setup

This directory contains the complete monitoring infrastructure for the Investment Analysis Platform, designed to operate within the $50/month budget constraint.

## Overview

The monitoring system provides comprehensive observability across four key areas:

1. **Infrastructure Monitoring** - System resources, containers, databases
2. **Application Monitoring** - API performance, business metrics, user activity  
3. **Business Metrics** - Daily recommendations, cost tracking, ML model performance
4. **Alerting & Notification** - Proactive alerts for issues and budget management

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │     Grafana     │    │  AlertManager   │
│   (Metrics)     │    │  (Dashboards)   │    │  (Alerting)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Node    │  │cAdvisor │  │Postgres │  │ Redis   │  │ Custom  │
│Exporter │  │(Contain)│  │Exporter │  │Exporter │  │ App     │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

## Quick Start

1. **Setup Monitoring Stack**:
   ```bash
   ./scripts/setup-monitoring.sh
   ```

2. **Access Dashboards**:
   - Grafana: http://localhost:3001 (admin/password)
   - Prometheus: http://localhost:9090
   - AlertManager: http://localhost:9093

3. **Import Dashboards**: Dashboards are auto-imported via provisioning

## Dashboard Descriptions

### 1. System Overview Dashboard
- **Purpose**: High-level system health and performance
- **Key Metrics**: 
  - Service availability (API, Database, Cache)
  - System resource usage (CPU, Memory, Disk)
  - API request rates and response times
- **Refresh**: 5 seconds
- **Use Case**: First-line monitoring, incident detection

### 2. API Performance Dashboard  
- **Purpose**: Detailed API performance analysis
- **Key Metrics**:
  - Response time percentiles (50th, 95th, 99th)
  - Request rate by endpoint and status code
  - Error rates and types
  - Success rate gauge
- **Refresh**: 5 seconds
- **Use Case**: Performance optimization, SLA monitoring

### 3. Business Metrics Dashboard
- **Purpose**: Business KPIs and cost tracking
- **Key Metrics**:
  - Daily recommendations generated
  - Stocks analyzed count
  - Monthly cost projection vs $50 budget
  - Budget usage percentage
  - ML model accuracy trends
- **Refresh**: 30 seconds  
- **Use Case**: Business monitoring, cost management

### 4. Database Performance Dashboard
- **Purpose**: Database and cache performance
- **Key Metrics**:
  - Database connection counts
  - Query duration percentiles  
  - Cache hit/miss rates
  - Database query rates by operation
- **Refresh**: 30 seconds
- **Use Case**: Database optimization, performance tuning

### 5. External APIs Dashboard
- **Purpose**: External API usage and cost tracking
- **Key Metrics**:
  - API call rates by provider (Alpha Vantage, Finnhub, Polygon)
  - Response times and failure rates
  - Rate limit remaining counters
  - Daily API costs
  - Circuit breaker states
- **Refresh**: 30 seconds
- **Use Case**: API quota management, cost optimization

## Alerting Rules

### Critical Alerts (Immediate Response)
- **ServiceDown**: Any core service unavailable > 1 minute
- **HighAPIErrorRate**: API error rate > 5% for 5 minutes  
- **DatabaseDown**: PostgreSQL unavailable > 1 minute
- **RedisDown**: Redis cache unavailable > 1 minute
- **LowDiskSpace**: Disk usage > 90%

### Warning Alerts (30min - 2hr Response)
- **HighAPILatency**: 95th percentile > 2 seconds for 5 minutes
- **HighCPUUsage**: CPU usage > 80% for 5 minutes
- **HighMemoryUsage**: Memory usage > 85% for 5 minutes
- **LowCacheHitRate**: Cache hit rate < 80% for 10 minutes
- **SlowDatabaseQueries**: 95th percentile query time > 1 second

### Cost Alerts (Budget Management)
- **BudgetExceeded**: Monthly usage > 90% of $50 budget
- **HighDailyCost**: Daily cost > $2 (would exceed monthly budget)

### Business Alerts (Data Quality)
- **LowRecommendationGeneration**: < 10 recommendations/hour
- **NoStockDataUpdate**: No stocks analyzed in 6 hours
- **MLModelAccuracyDrop**: Model accuracy < 70%

## Cost Optimization Features

### 1. Budget Tracking
- Real-time monthly cost projection
- Daily cost breakdown by service
- Budget usage percentage with thresholds
- Cost optimization recommendations

### 2. API Usage Monitoring
- Per-provider API call tracking
- Rate limit monitoring
- Cost-per-call estimates
- Usage pattern analysis

### 3. Resource Optimization
- Container resource usage tracking
- Database connection pool monitoring
- Cache efficiency metrics
- Storage usage trends

## Configuration Files

### Prometheus Configuration (`prometheus.yml`)
- **Scrape Intervals**: Optimized for cost (10-60 seconds)
- **Retention**: 30 days for cost efficiency
- **Targets**: All application and infrastructure components
- **Alert Rules**: Comprehensive coverage for reliability and cost

### Grafana Provisioning
- **Datasources**: Auto-configured Prometheus connection
- **Dashboards**: Auto-imported via provisioning
- **Alerting**: Integrated with AlertManager

### AlertManager Configuration (`alertmanager.yml`)
- **Routing**: Severity-based alert routing
- **Receivers**: Webhook integration with application
- **Inhibition**: Prevents alert storms
- **Templates**: Custom notification formats

## Exporters

### Infrastructure Exporters
- **Node Exporter**: System metrics (CPU, memory, disk, network)
- **cAdvisor**: Container metrics (resource usage, performance)

### Application Exporters  
- **PostgreSQL Exporter**: Database performance metrics
- **Redis Exporter**: Cache performance and memory usage
- **Elasticsearch Exporter**: Search engine metrics
- **Nginx Exporter**: Web server performance

### Custom Metrics
- **Business Metrics**: Recommendations, stock analysis, costs
- **API Metrics**: Request rates, latencies, errors
- **ML Metrics**: Model accuracy, prediction times

## Security Considerations

### Access Control
- Grafana authentication required
- Prometheus metrics endpoint protected
- AlertManager webhook authentication
- Network isolation between components

### Data Privacy
- No sensitive financial data in metrics
- Anonymized user identifiers
- GDPR compliance for user metrics
- Audit logging for access

## Maintenance

### Daily Tasks
- Review budget usage dashboard
- Check for any critical alerts
- Verify data pipeline success rates

### Weekly Tasks  
- Analyze cost trends and optimization opportunities
- Review ML model accuracy trends
- Check for any unusual API usage patterns

### Monthly Tasks
- Archive old monitoring data
- Review and update alerting thresholds
- Evaluate monitoring infrastructure costs

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check container resource limits
   - Review Prometheus retention settings
   - Optimize query complexity

2. **Missing Metrics**
   - Verify exporter connectivity
   - Check Prometheus targets page
   - Review application metric emission

3. **Alert Fatigue**
   - Review alert thresholds
   - Implement alert inhibition rules
   - Use appropriate alert severity levels

### Performance Optimization

1. **Reduce Scrape Frequency**: For non-critical metrics
2. **Optimize Queries**: Use recording rules for expensive queries  
3. **Implement Caching**: For frequently accessed metrics
4. **Archive Old Data**: Maintain cost-effective retention

## Integration Points

### Application Integration
- Custom metrics emission from FastAPI
- Business metrics tracking in data pipelines
- Cost tracking in external API calls
- Performance metrics in ML model inference

### External Systems
- Webhook notifications to Slack/Teams
- Email alerts for critical issues
- Integration with incident management tools
- Cost reporting to finance systems

## Budget Breakdown

Estimated monthly costs for monitoring infrastructure:

- **Prometheus Storage**: ~$2/month (30 days retention)
- **Grafana**: $0 (open source)
- **Exporters**: $0 (open source) 
- **Compute Resources**: ~$3/month (minimal resource requirements)
- **Total Monitoring Cost**: ~$5/month (10% of total budget)

This leaves $45/month for the core application infrastructure and external API usage.

## Support

For monitoring-related issues:
1. Check the troubleshooting section above
2. Review Grafana dashboard for error patterns
3. Examine Prometheus targets for connectivity issues
4. Check AlertManager for notification delivery problems

The monitoring system is designed to be self-maintaining with minimal operational overhead while providing comprehensive visibility into the platform's performance and costs.
