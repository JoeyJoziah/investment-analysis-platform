# Production Deployment Guide
## Investment Analysis Platform - Cost-Optimized Production Deployment

This guide provides comprehensive procedures for production deployment, monitoring, and rollback of the Investment Analysis Platform, designed to operate under $50/month.

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Procedures](#deployment-procedures)
3. [Post-Deployment Verification](#post-deployment-verification)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Rollback Procedures](#rollback-procedures)
6. [Cost Management](#cost-management)
7. [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

### Infrastructure Readiness
- [ ] **Server Resources**
  - [ ] CPU: 2 cores minimum (4 cores recommended)
  - [ ] RAM: 4GB minimum (8GB recommended)
  - [ ] Storage: 20GB minimum (50GB recommended)
  - [ ] Network: 1Gbps bandwidth minimum

- [ ] **Docker Environment**
  - [ ] Docker Engine 20.10+ installed
  - [ ] Docker Compose 2.0+ installed
  - [ ] Docker daemon running and accessible
  - [ ] Sufficient disk space for images (10GB minimum)

- [ ] **Database Preparation**
  - [ ] PostgreSQL connection tested
  - [ ] TimescaleDB extension available
  - [ ] Database migration scripts ready
  - [ ] Backup procedures verified

### Configuration Verification
- [ ] **Environment Variables**
  - [ ] `.env` file populated with production values
  - [ ] API keys valid and within rate limits
  - [ ] Database credentials correct
  - [ ] Secret keys generated and secure

- [ ] **External Dependencies**
  - [ ] Alpha Vantage API key tested (25 calls/day limit)
  - [ ] Finnhub API key tested (60 calls/minute limit)
  - [ ] Polygon.io API key tested (5 calls/minute limit)
  - [ ] NewsAPI key tested and functional

- [ ] **Security Configuration**
  - [ ] SSL certificates installed
  - [ ] Security headers configured
  - [ ] Rate limiting configured
  - [ ] Firewall rules applied

### Code Quality Assurance
- [ ] **Testing**
  - [ ] Unit tests passing (>90% coverage)
  - [ ] Integration tests passing
  - [ ] Load tests completed
  - [ ] Security scans completed

- [ ] **Code Review**
  - [ ] Pull request approved
  - [ ] Security review completed
  - [ ] Performance review completed
  - [ ] Documentation updated

## Deployment Procedures

### Method 1: Blue-Green Deployment (Recommended)
```bash
# 1. Run pre-deployment checks
./scripts/deployment/pre_deployment_check.sh

# 2. Execute blue-green deployment
./scripts/deployment/blue_green_deploy.sh \
  --version v1.2.3 \
  --environment production \
  --services "backend frontend"

# 3. Monitor deployment logs
tail -f logs/deployment.log
```

### Method 2: Rolling Deployment
```bash
# 1. Build optimized images
docker build -f infrastructure/docker/backend/Dockerfile.optimized \
  -t investment-backend:v1.2.3 .

docker build -f infrastructure/docker/frontend/Dockerfile.optimized \
  -t investment-frontend:v1.2.3 ./frontend/web

# 2. Deploy with zero downtime
docker-compose -f docker-compose.yml \
  -f docker-compose.production.yml \
  up -d --force-recreate --no-deps backend

# 3. Verify backend health before frontend
curl -f http://localhost:8000/api/health

# 4. Deploy frontend
docker-compose -f docker-compose.yml \
  -f docker-compose.production.yml \
  up -d --force-recreate --no-deps frontend
```

### Database Migrations
```bash
# 1. Create backup before migration
./scripts/backup.sh

# 2. Run migrations
docker exec investment_api_prod python -m alembic upgrade head

# 3. Verify migration success
docker exec investment_api_prod python -m alembic current
```

## Post-Deployment Verification

### Health Checks
```bash
# 1. Basic health check
curl -f http://localhost/api/health

# 2. Detailed health check
curl -s http://localhost/api/health | jq .

# 3. Database connectivity
curl -f http://localhost/api/health/database

# 4. Cache functionality
curl -f http://localhost/api/health/cache

# 5. External API connectivity
curl -f http://localhost/api/health/external-apis
```

### Performance Verification
```bash
# 1. Response time check
curl -w "Total time: %{time_total}s\n" -o /dev/null -s http://localhost/api/stocks/prices?symbols=AAPL

# 2. Load test (basic)
ab -n 100 -c 10 http://localhost/api/health

# 3. Memory usage check
docker stats --no-stream

# 4. Database performance
docker exec investment_db_prod psql -U postgres -d investment_db \
  -c "SELECT pg_stat_get_db_tuples_fetched(oid) FROM pg_database WHERE datname='investment_db';"
```

### Functional Testing
```bash
# 1. API endpoints test
./scripts/testing/api_smoke_test.sh

# 2. WebSocket connectivity
./scripts/testing/websocket_test.sh

# 3. Data pipeline verification
./scripts/testing/pipeline_test.sh

# 4. User authentication test
curl -X POST http://localhost/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'
```

## Monitoring and Alerting

### Prometheus Metrics
- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Request rate, response time, error rate
- **Business Metrics**: API calls, cost tracking, cache hit rate
- **Database Metrics**: Query performance, connection pool status

### Grafana Dashboards
1. **System Overview**: Infrastructure health
2. **Application Performance**: API metrics and response times
3. **Business Metrics**: Cost tracking and API usage
4. **Database Performance**: Query metrics and optimization

### Alert Thresholds
```yaml
# CPU Usage
- alert: HighCPUUsage
  expr: (100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)) > 80
  for: 5m

# Memory Usage  
- alert: HighMemoryUsage
  expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
  for: 5m

# API Response Time
- alert: HighAPIResponseTime
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 2m

# Cost Budget Alert
- alert: CostBudgetExceeded
  expr: investment_daily_cost_projection > 1.67
  for: 1m
```

## Rollback Procedures

### Automatic Rollback
The blue-green deployment script includes automatic rollback on failure:
```bash
# Rollback is triggered automatically if:
# - Health checks fail
# - Response time exceeds threshold
# - Error rate increases significantly
```

### Manual Rollback
```bash
# 1. Quick rollback to previous version
./scripts/deployment/rollback.sh

# 2. Rollback with database restoration
./scripts/deployment/rollback.sh \
  --backup-path /path/to/backup \
  --restore-db

# 3. Rollback to specific version
./scripts/deployment/rollback.sh --version v1.2.2

# 4. Force rollback (skip confirmation)
./scripts/deployment/rollback.sh --force
```

### Rollback Verification
```bash
# 1. Verify rollback completion
curl -s http://localhost/api/health | jq '.version'

# 2. Check service status
docker ps --filter name=investment_

# 3. Verify database state
docker exec investment_db_prod psql -U postgres -d investment_db \
  -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# 4. Test critical functionality
./scripts/testing/critical_path_test.sh
```

## Cost Management

### Daily Cost Monitoring
```bash
# Check current daily cost projection
curl -s http://localhost/api/metrics/cost | jq '.daily_projection'

# View API usage by provider
curl -s http://localhost/api/metrics/api-usage | jq '.'

# Check cache efficiency
curl -s http://localhost/api/metrics/cache | jq '.hit_rate'
```

### Cost Optimization Actions
1. **Scale Down Non-Peak Hours**
   ```bash
   # Reduce replicas during low traffic (automated via cron)
   docker-compose -f docker-compose.production.yml up -d --scale backend=1
   ```

2. **Cache Warming**
   ```bash
   # Pre-populate cache to reduce API calls
   ./scripts/optimization/cache_warmer.sh
   ```

3. **Database Optimization**
   ```bash
   # Run database maintenance
   docker exec investment_db_prod psql -U postgres -d investment_db \
     -c "VACUUM ANALYZE;"
   ```

## Troubleshooting

### Common Issues and Solutions

#### High Memory Usage
```bash
# Check memory usage by service
docker stats --no-stream

# Restart memory-intensive services
docker-compose restart celery_worker

# Clear cache if necessary
docker exec investment_cache_prod redis-cli FLUSHDB
```

#### Database Connection Issues
```bash
# Check database connectivity
docker exec investment_db_prod pg_isready -U postgres

# View active connections
docker exec investment_db_prod psql -U postgres -d investment_db \
  -c "SELECT count(*) FROM pg_stat_activity;"

# Restart database if necessary
docker-compose restart postgres
```

#### API Rate Limit Exceeded
```bash
# Check API usage
grep "rate limit" logs/*.log

# Implement emergency rate limiting
docker exec investment_cache_prod redis-cli SET emergency_rate_limit 1 EX 3600

# Scale down API calls
docker-compose scale celery_worker=0
```

#### Performance Degradation
```bash
# Check system resources
top -p $(docker inspect -f '{{.State.Pid}}' investment_api_prod)

# Analyze slow queries
docker exec investment_db_prod psql -U postgres -d investment_db \
  -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Clear application cache
curl -X POST http://localhost/api/cache/clear \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Emergency Contacts
- **DevOps Team**: devops@company.com
- **Database Admin**: dba@company.com  
- **Security Team**: security@company.com

### Log Locations
- **Application Logs**: `/logs/`
- **Nginx Logs**: `/var/log/nginx/`
- **Database Logs**: `/var/log/postgresql/`
- **Docker Logs**: `docker logs <container_name>`

## Production Checklist Summary

### Before Deployment ✓
- [ ] All tests passing
- [ ] Configuration verified
- [ ] Backup completed
- [ ] Monitoring configured
- [ ] Rollback plan ready

### During Deployment ✓
- [ ] Health checks enabled
- [ ] Deployment logs monitored
- [ ] Performance metrics tracked
- [ ] Team notified

### After Deployment ✓
- [ ] Functionality verified
- [ ] Performance validated
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Success communicated

This guide ensures reliable, cost-effective production deployments while maintaining the $50/month operational budget target.