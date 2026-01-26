# Operational Runbook

This runbook provides operational procedures for the Investment Analysis Platform.

---

## Table of Contents

1. [Service Management](#service-management)
2. [Database Operations](#database-operations)
3. [ML Model Operations](#ml-model-operations)
4. [Data Pipeline Operations](#data-pipeline-operations)
5. [Monitoring & Alerts](#monitoring--alerts)
6. [Backup & Recovery](#backup--recovery)
7. [Troubleshooting](#troubleshooting)
8. [Emergency Procedures](#emergency-procedures)

---

## Service Management

### Starting Services

```bash
# Development environment
./start.sh dev

# Production environment
./start.sh prod

# Start specific service
docker-compose up -d backend

# Start with rebuild
docker-compose up -d --build backend
```

### Stopping Services

```bash
# Stop all services
./stop.sh

# Stop specific service
docker-compose stop backend

# Stop and remove containers
docker-compose down
```

### Restarting Services

```bash
# Restart all
./scripts/deployment/restart.sh

# Restart specific service
docker-compose restart backend

# Rolling restart (zero downtime)
docker-compose up -d --no-deps --build backend
```

### Viewing Logs

```bash
# All services
./logs.sh

# Specific service
./logs.sh backend

# Follow logs
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Health Checks

```bash
# Check all services
./scripts/monitoring-health-check.sh

# Check API health
curl http://localhost:8000/api/health

# Check individual services
docker-compose ps
```

---

## Database Operations

### Connection

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d investment_db

# Or using local client
psql -h localhost -p 5432 -U postgres -d investment_db
```

### Common Queries

```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('investment_db'));

-- List tables with sizes
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- Kill long-running queries (>5 minutes)
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active'
  AND query_start < NOW() - INTERVAL '5 minutes';
```

### Migrations

```bash
# Run migrations
python scripts/simple_migrate.py

# Check migration status
alembic current

# Generate new migration
alembic revision --autogenerate -m "description"

# Rollback one migration
alembic downgrade -1
```

### Schema Fixes

```bash
# Apply schema fixes
python scripts/fix_database_schema.py

# Apply optimizations
python scripts/apply_database_optimizations.py
```

---

## ML Model Operations

### Training Models

```bash
# Full training
python scripts/train_ml_models.py

# Minimal/quick training
python scripts/train_ml_models_minimal.py

# Train specific model
python scripts/train_ml_models.py --model prophet
python scripts/train_ml_models.py --model xgboost
```

### Deploying Models

```bash
# Deploy to production
./scripts/deploy_ml_production.sh

# Deploy specific model
python scripts/deploy_ml_models.py --model prophet

# Verify deployment
curl http://localhost:8001/api/models/status
```

### Model Status

```bash
# Check model status
python scripts/load_trained_models.py --status

# List available models
ls -la ml_models/

# Check model versions
cat ml_models/xgboost_training_results.json
cat ml_models/prophet/prophet_training_results.json
```

### Starting ML API

```bash
# Start ML API server
./scripts/start_ml_api.sh

# Or directly
python -m backend.ml.ml_api_server

# Check ML API health
curl http://localhost:8001/api/health
```

---

## Data Pipeline Operations

### Starting Data Pipeline

```bash
# Start ETL pipeline
python scripts/activate_etl_pipeline.py

# Start background loader
python scripts/data/background_loader_enhanced.py

# Start Airflow
./scripts/start_airflow_minimal.sh
```

### Monitoring Pipeline

```bash
# Check pipeline status
python scripts/monitoring/monitor_pipeline.py

# Check Airflow status
./scripts/check_airflow_status.sh

# View Airflow UI
open http://localhost:8080
```

### Manual Data Loading

```bash
# Load data immediately
python scripts/data/load_data_now.py

# Load historical data
python scripts/load_historical_data.py

# Load specific stock
python scripts/data/load_data_now.py --ticker AAPL
```

### API Rate Limit Status

| API | Daily Limit | Check Command |
|-----|-------------|---------------|
| Alpha Vantage | 25 | Check logs for "rate limit" |
| Finnhub | 1800 (30/min) | `curl https://finnhub.io/api/v1/quote?symbol=AAPL&token=YOUR_KEY` |
| Polygon | 150 | Check dashboard at polygon.io |
| NewsAPI | 100 | Check headers in API response |

---

## Monitoring & Alerts

### Accessing Dashboards

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Grafana | http://localhost:3001 | admin / (see .env) |
| Prometheus | http://localhost:9090 | N/A |
| Airflow | http://localhost:8080 | admin / (see .env) |
| API Docs | http://localhost:8000/docs | N/A |

### Key Metrics to Monitor

1. **API Response Times** - Should be <200ms p95
2. **Error Rate** - Should be <1%
3. **Database Connections** - Should be <80% of max
4. **Memory Usage** - Should be <80%
5. **API Rate Limit Usage** - Should be <80% of daily limits
6. **Cost Tracking** - Should be <$50/month

### Alert Response

| Alert | Severity | Response |
|-------|----------|----------|
| High Error Rate | Critical | Check logs, restart service |
| Database Connection Pool Exhausted | Critical | Restart backend, check for leaks |
| API Rate Limit Exceeded | High | Wait for reset, check caching |
| Memory Usage High | High | Restart service, increase limits |
| Cost Alert ($40+) | Medium | Review API usage, optimize queries |

---

## Backup & Recovery

### Creating Backups

```bash
# Full database backup
./scripts/backup.sh

# Backup to specific location
./scripts/backup.sh /path/to/backup/

# Backup ML models
tar -czf ml_models_backup.tar.gz ml_models/
```

### Verifying Backups

```bash
# Verify backup integrity
./scripts/verify-backup.sh

# List available backups
ls -la backups/
```

### Restoring from Backup

```bash
# Restore database
./scripts/restore-backup.sh backup_file.sql

# Restore ML models
tar -xzf ml_models_backup.tar.gz
```

### Backup Schedule

| Data | Frequency | Retention |
|------|-----------|-----------|
| Database | Daily | 30 days |
| ML Models | After each training | 5 versions |
| Configuration | On change | Unlimited (in Git) |

---

## Troubleshooting

### Common Issues

#### Backend Won't Start

```bash
# Check logs
docker-compose logs backend

# Check if ports are in use
lsof -i :8000

# Restart with clean build
docker-compose down
docker-compose build --no-cache backend
docker-compose up -d backend
```

#### Database Connection Failed

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check connection
docker-compose exec postgres pg_isready

# Restart PostgreSQL
docker-compose restart postgres
```

#### Redis Connection Issues

```bash
# Check Redis status
docker-compose exec redis redis-cli ping

# Check memory
docker-compose exec redis redis-cli info memory

# Flush cache if needed
docker-compose exec redis redis-cli FLUSHALL
```

#### ML Models Not Loading

```bash
# Check model files exist
ls -la ml_models/

# Verify model integrity
python scripts/load_trained_models.py --verify

# Retrain if corrupted
python scripts/train_ml_models_minimal.py
```

#### Data Pipeline Stuck

```bash
# Check Airflow logs
docker-compose logs airflow

# Check task status
./scripts/check_airflow_status.sh

# Clear failed tasks
# Via Airflow UI: Mark task as success/clear
```

### Debug Mode

```bash
# Start backend in debug mode
cd backend
DEBUG=true uvicorn backend.api.main:app --reload --log-level debug

# Enable verbose logging
export LOG_LEVEL=DEBUG
./start.sh dev
```

---

## Emergency Procedures

### Complete Service Outage

1. **Assess**: Check all services with `docker-compose ps`
2. **Logs**: Review logs `./logs.sh`
3. **Restart**: `docker-compose down && ./start.sh prod`
4. **Verify**: Check health endpoints
5. **Notify**: Inform stakeholders

### Database Corruption

1. **Stop services**: `./stop.sh`
2. **Backup current state**: `docker-compose exec postgres pg_dump ... > emergency_backup.sql`
3. **Restore from backup**: `./scripts/restore-backup.sh latest_backup.sql`
4. **Verify data**: Run validation queries
5. **Restart services**: `./start.sh prod`

### Security Incident

1. **Isolate**: Disconnect affected services
2. **Preserve**: Save logs and evidence
3. **Rotate**: Change all credentials
   ```bash
   ./scripts/generate_secrets.sh
   # Update .env with new secrets
   ```
4. **Investigate**: Review access logs
5. **Remediate**: Fix vulnerability
6. **Document**: Create incident report

### Cost Overrun

1. **Pause data collection**: Stop data pipeline
2. **Review API usage**: Check which APIs are consuming most
3. **Enable stricter caching**: Increase cache TTLs
4. **Reduce stock universe**: Limit to high-priority stocks
5. **Monitor**: Watch costs for remainder of billing period

### Rollback Deployment

```bash
# Quick rollback
./scripts/deployment/rollback.sh

# Manual rollback steps
git checkout previous-tag
docker-compose down
docker-compose build
docker-compose up -d
```

---

## Contact & Escalation

| Issue Type | First Response | Escalation |
|------------|---------------|------------|
| Service Outage | On-call engineer | Team lead |
| Security Incident | Security team | CTO |
| Data Issues | Data engineer | Data lead |
| Cost Issues | DevOps | Finance |

---

## Maintenance Windows

| Activity | Frequency | Duration | Impact |
|----------|-----------|----------|--------|
| Database maintenance | Weekly (Sun 3am) | 30 min | Degraded |
| Model retraining | Daily (2am) | 2 hours | None |
| Full backup | Daily (4am) | 1 hour | None |
| Security updates | Monthly | 2 hours | Downtime |

---

## Related Documentation

- [Scripts Reference](./SCRIPTS_REFERENCE.md)
- [Environment Variables](./ENVIRONMENT.md)
- [Contributing Guide](./CONTRIB.md)
- [Production Deployment Guide](../PRODUCTION_DEPLOYMENT_GUIDE.md)
