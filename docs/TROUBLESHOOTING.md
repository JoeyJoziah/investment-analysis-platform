# Troubleshooting & Support Guide

**Last Updated**: 2026-01-27
**Version**: 1.0.0
**Status**: Production Support

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Service-Specific Troubleshooting](#service-specific-troubleshooting)
4. [Database Issues](#database-issues)
5. [Performance Issues](#performance-issues)
6. [Authentication Problems](#authentication-problems)
7. [Data Pipeline Issues](#data-pipeline-issues)
8. [Monitoring & Debugging](#monitoring--debugging)
9. [Getting Help](#getting-help)

---

## Quick Diagnostics

### System Health Check

Run this to quickly assess system status:

```bash
# Complete system diagnostic
./docs/scripts/health-check.sh

# Expected output:
# ✅ Docker running
# ✅ All 12 services up
# ✅ PostgreSQL accessible
# ✅ Redis responding
# ✅ Backend healthy
# ✅ Frontend responding
```

### Docker Service Status

```bash
# Check all running services
docker-compose ps

# Detailed service status
docker-compose ps --verbose

# Check specific service
docker-compose ps investment_backend

# View service health
docker-compose exec investment_db pg_isready -h localhost
docker-compose exec investment_cache redis-cli ping
```

### API Health Endpoint

```bash
# Backend API health
curl http://localhost:8000/api/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", "timestamp": "2026-01-27T00:00:00Z"}

# If fails, check backend logs
docker-compose logs investment_backend
```

### Database Connection

```bash
# Quick database test
docker-compose exec investment_db psql -U investment_user -d investment_db -c "SELECT 1;"

# Should return: 1
# If error, check database logs
docker-compose logs investment_db
```

---

## Common Issues

### Issue 1: "Database connection refused"

**Symptoms**:
- Backend won't start
- Error: `connection refused on 5432`
- Logs: `psycopg2.OperationalError`

**Diagnosis**:

```bash
# Check if PostgreSQL is running
docker-compose ps investment_db

# Check database logs
docker-compose logs investment_db | tail -50

# Try direct connection
docker-compose exec investment_db pg_isready -h localhost
```

**Solutions** (in order):

```bash
# 1. Ensure database container is healthy
docker-compose up -d investment_db
sleep 30  # Wait for startup
docker-compose logs investment_db | grep "database system is ready"

# 2. Check credentials in .env
grep DB_ .env

# 3. Verify database user exists
docker-compose exec investment_db psql -U postgres -c "\du"

# 4. Check port availability
lsof -i :5432

# 5. Rebuild database
docker-compose down investment_db
docker volume rm investment-db-data  # WARNING: Deletes data
docker-compose up -d investment_db

# 6. Check network
docker network ls
docker-compose down  # Recreates network
docker-compose up -d
```

---

### Issue 2: "Backend won't start - GDPR key missing"

**Symptoms**:
- Error: `GDPR_ENCRYPTION_KEY not configured`
- Backend container exits immediately
- Logs: `AttributeError: 'NoneType' object has no attribute 'encode'`

**Solution**:

```bash
# Generate encryption key
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Add to .env
echo "GDPR_ENCRYPTION_KEY=$FERNET_KEY" >> .env

# Verify added
grep GDPR_ENCRYPTION_KEY .env

# Restart backend
docker-compose restart investment_backend

# Check logs
docker-compose logs investment_backend | grep -A5 "Starting"
```

---

### Issue 3: "Out of memory"

**Symptoms**:
- Services randomly crash
- Error: `OOMKilled` in logs
- Container exits with code 137

**Diagnosis**:

```bash
# Check memory usage
docker stats

# Check memory limits in docker-compose
grep -A3 "mem_limit" docker-compose.yml

# Monitor over time
watch docker stats
```

**Solutions**:

```bash
# 1. Increase Docker memory limit
# Edit docker-compose.yml
services:
  investment_backend:
    mem_limit: 2g      # Increase from 1g
    mem_reservation: 1g

# 2. Restart services
docker-compose down
docker-compose up -d

# 3. Monitor memory
watch -n 1 'docker stats --no-stream | head -10'

# 4. Check for memory leaks in logs
docker-compose logs investment_backend | grep -i "memory"

# 5. Clear Docker cache
docker system prune -a --volumes
```

---

### Issue 4: "Disk space full"

**Symptoms**:
- Error: `No space left on device`
- Services stop responding
- Logs won't write

**Diagnosis**:

```bash
# Check disk usage
df -h

# Check Docker disk usage
docker system df

# Check database size
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "SELECT pg_size_pretty(pg_database_size('investment_db'));"
```

**Solutions**:

```bash
# 1. Clean up Docker
docker system prune -a --volumes  # WARNING: Removes unused images/volumes

# 2. Clean logs
docker-compose logs --tail 0  # Truncate old logs

# 3. Compress or archive old data
docker-compose exec investment_db \
  psql -U investment_user -d investment_db \
  -c "DELETE FROM price_history WHERE date < NOW() - interval '1 year';"

# 4. Check what's taking space
du -sh /var/lib/docker/*
ls -lh ./data/

# 5. Expand disk (infrastructure-level)
# SSH to server and increase volume
# Then resize PostgreSQL: ALTER TABLESPACE
```

---

### Issue 5: "API responding slowly (>5s latency)"

**Symptoms**:
- API calls take 5+ seconds
- Timeout errors in logs
- Frontend feels sluggish

**Diagnosis**:

```bash
# Measure API response time
time curl http://localhost:8000/api/health

# Check backend logs for slow queries
docker-compose logs investment_backend | grep -i "slow\|took"

# Check database performance
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 5;"

# Check CPU/memory usage
docker stats investment_backend investment_db

# Measure network latency
ping -c 5 localhost
```

**Solutions**:

```bash
# 1. Check cache hit rate
redis-cli INFO stats | grep hits

# 2. Warm cache
docker-compose exec investment_backend python /app/backend/scripts/warm_cache.py

# 3. Add database indexes
docker-compose exec investment_db psql -U investment_user -d investment_db << 'EOF'
CREATE INDEX idx_stock_ticker ON stocks(ticker);
CREATE INDEX idx_price_history_stock_date ON price_history(stock_id, date);
CREATE INDEX idx_user_email ON users(email);
EOF

# 4. Analyze table
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "ANALYZE stocks; ANALYZE price_history;"

# 5. Increase backend resources
# Edit docker-compose.yml
# services:
#   investment_backend:
#     cpus: '2.0'      # Increase from 1.0
#     mem_limit: 2g    # Increase from 1g

docker-compose down && docker-compose up -d
```

---

### Issue 6: "Stock data not loading"

**Symptoms**:
- Database shows 0 stocks
- API returns empty results
- ETL logs show errors

**Diagnosis**:

```bash
# Check stock count
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "SELECT COUNT(*) FROM stocks;"

# Check ETL logs
docker-compose logs investment_airflow
docker-compose logs investment_worker

# Check API data endpoint
curl http://localhost:8000/api/stocks

# Check if stocks table exists
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "\dt stocks"
```

**Solutions**:

```bash
# 1. Verify data sources are accessible
curl -H "Authorization: Bearer $FINNHUB_API_KEY" \
  "https://finnhub.io/api/v1/quote?symbol=AAPL"

# 2. Run data loader manually
docker-compose exec investment_backend \
  python backend/scripts/load_stocks.py --symbol AAPL --verbose

# 3. Check ETL configuration
grep -A10 "STOCK_DATA" .env

# 4. Run Airflow DAG manually
docker-compose exec investment_airflow \
  airflow dags trigger daily_stock_pipeline

# 5. Monitor execution
docker-compose logs -f investment_airflow | grep daily_stock_pipeline

# 6. Check Airflow UI
# Navigate to http://localhost:8080
# Look for failed tasks in daily_stock_pipeline DAG
```

---

## Service-Specific Troubleshooting

### PostgreSQL Issues

#### PostgreSQL won't start

```bash
# Check logs
docker-compose logs investment_db

# Common fixes:
# 1. Remove corrupted data
docker-compose down investment_db
docker volume rm investment-db-data
docker-compose up -d investment_db

# 2. Check disk space
df -h /var/lib/docker

# 3. Check port conflict
lsof -i :5432

# 4. Check permissions on volume
ls -la ./data/postgresql/
sudo chmod 755 ./data/postgresql/
```

#### Connection pooling exhausted

```bash
# Check active connections
docker-compose exec investment_db psql -U postgres \
  -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
docker-compose exec investment_db psql -U postgres -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
   WHERE state = 'idle' AND state_change < now() - interval '10 minutes';"

# Increase pool size in .env
# DB_POOL_SIZE=20
# DB_POOL_MAX_OVERFLOW=10

# Restart services
docker-compose restart investment_backend investment_worker
```

### Redis Issues

#### Redis won't respond

```bash
# Test connection
docker-compose exec investment_cache redis-cli ping

# If no response:
docker-compose restart investment_cache

# Check memory
docker-compose exec investment_cache redis-cli INFO memory

# Clear cache if needed
docker-compose exec investment_cache redis-cli FLUSHDB

# Check for memory leaks
docker-compose exec investment_cache redis-cli INFO stats
```

#### Cache hit rate low

```bash
# Check hit rate
docker-compose exec investment_cache redis-cli INFO stats | grep hits

# If <80%, optimize:
# 1. Increase TTL in backend/config.py
# 2. Pre-warm cache on startup
# 3. Check cache key patterns

# Increase Redis memory
# Edit docker-compose.yml:
# services:
#   investment_cache:
#     environment:
#       REDIS_MAX_MEMORY: 512mb

docker-compose restart investment_cache
```

### Elasticsearch Issues

#### Elasticsearch cluster unhealthy

```bash
# Check cluster status
curl -X GET http://localhost:9200/_cluster/health

# Expected: "status": "green"
# If yellow/red:

# Check shards
curl -X GET http://localhost:9200/_cat/shards

# Recover unassigned shards
curl -X POST http://localhost:9200/_cluster/reroute?retry_failed=true

# Check nodes
curl -X GET http://localhost:9200/_cat/nodes

# Restart if needed
docker-compose restart investment_search
```

### Celery Worker Issues

#### Tasks not processing

```bash
# Check worker status
docker-compose exec investment_worker celery -A backend.celery_app inspect active

# If no tasks:
# 1. Check worker logs
docker-compose logs investment_worker

# 2. Verify Redis connection
docker-compose exec investment_worker redis-cli ping

# 3. Restart worker
docker-compose restart investment_worker

# 4. Clear task queue
docker-compose exec investment_cache redis-cli DEL celery

# 5. Check task configuration
grep -A10 "CELERY_" .env
```

#### High task queue

```bash
# Check queue size
docker-compose exec investment_cache redis-cli LLEN celery

# Scale workers
docker-compose up -d --scale investment_worker=3

# Monitor queue
watch "redis-cli LLEN celery"

# Purge old tasks
docker-compose exec investment_cache redis-cli DEL celery
```

---

## Database Issues

### Backup & Recovery

#### Verify backups exist

```bash
# List backups
ls -lh ./backups/

# Check backup size
du -sh ./backups/

# Verify backup integrity
pg_restore --list ./backups/backup-latest.sql > /tmp/backup-contents.txt

# Count tables in backup
grep "CREATE TABLE" ./backups/backup-latest.sql | wc -l
```

#### Restore from backup

```bash
# 1. Stop services
docker-compose down investment_backend investment_worker

# 2. Restore database
docker-compose exec investment_db \
  psql -U investment_user -d investment_db < ./backups/backup-20260127.sql

# 3. Verify restore
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "SELECT COUNT(*) FROM stocks;"

# 4. Start services
docker-compose up -d investment_backend investment_worker
```

### Query Performance

#### Find slow queries

```bash
# Enable query logging
docker-compose exec investment_db psql -U postgres << 'EOF'
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- 1 second
SELECT pg_reload_conf();
EOF

# View slow queries
docker-compose exec investment_db tail -f /var/log/postgresql/postgresql.log | grep "duration:"

# Analyze query plan
docker-compose exec investment_db psql -U investment_user -d investment_db << 'EOF'
EXPLAIN ANALYZE
SELECT * FROM price_history
WHERE stock_id = 1
ORDER BY date DESC
LIMIT 100;
EOF
```

---

## Performance Issues

### CPU Usage High

```bash
# Identify top CPU consumers
docker stats --no-stream | sort -k3 -rn

# Check specific container
docker exec investment_backend ps aux --sort -%cpu | head -10

# If backend:
docker-compose logs investment_backend | grep ERROR

# If database:
docker-compose exec investment_db \
  psql -U investment_user -d investment_db << 'EOF'
SELECT query, calls, mean_time
FROM pg_stat_statements
ORDER BY mean_time * calls DESC LIMIT 10;
EOF
```

### Memory Leak Detection

```bash
# Monitor memory over time
watch -n 5 'docker stats --no-stream investment_backend | grep investment_backend'

# Check for growing memory
docker exec investment_backend top -b -n 1 | head -20

# If leaking, restart service
docker-compose restart investment_backend

# Check application logs
docker-compose logs investment_backend | grep -i "memory\|allocation\|pool"
```

### Database Lock Issues

```bash
# Find locks
docker-compose exec investment_db psql -U investment_user -d investment_db << 'EOF'
SELECT pid, usename, application_name, query
FROM pg_stat_activity
WHERE state != 'idle';
EOF

# Kill blocking session
docker-compose exec investment_db psql -U postgres \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid = 12345;"

# Check for long-running transactions
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "SELECT query_start, query FROM pg_stat_activity WHERE state = 'active';"
```

---

## Authentication Problems

### Can't Login

**Symptoms**:
- Error: `Invalid credentials`
- Error: `401 Unauthorized`

**Diagnosis**:

```bash
# Check user exists
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "SELECT id, email FROM users WHERE email = 'user@example.com';"

# Check password hash valid
docker-compose exec investment_db psql -U investment_user -d investment_db \
  -c "SELECT id, password_hash FROM users WHERE email = 'user@example.com';"
```

**Solutions**:

```bash
# 1. Reset password manually
docker-compose exec investment_backend python << 'EOF'
from backend.utils.auth import PasswordManager
from backend.models.user import User
from backend.database import SessionLocal

db = SessionLocal()
user = db.query(User).filter_by(email='user@example.com').first()
if user:
    user.password_hash = PasswordManager.hash_password('NewPassword123!')
    db.commit()
    print("Password reset successfully")
EOF

# 2. Check JWT configuration
grep JWT .env

# 3. Test token generation
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user@example.com","password":"password"}'
```

### MFA Issues

```bash
# Disable MFA for user (if locked out)
docker-compose exec investment_backend python << 'EOF'
from backend.models.user import User
from backend.database import SessionLocal

db = SessionLocal()
user = db.query(User).filter_by(email='user@example.com').first()
if user:
    user.mfa_enabled = False
    db.commit()
    print("MFA disabled")
EOF

# User can re-enable after logging in
```

---

## Data Pipeline Issues

### Airflow DAG not triggering

```bash
# Check DAG syntax
docker-compose exec investment_airflow \
  airflow dags validate daily_stock_pipeline

# Trigger manually
docker-compose exec investment_airflow \
  airflow dags trigger daily_stock_pipeline

# Check DAG status
docker-compose exec investment_airflow \
  airflow dags list-runs --dag-id daily_stock_pipeline

# View Airflow UI
# http://localhost:8080
# Check DAG status, click to debug
```

### ETL task failures

```bash
# Check Celery task status
docker-compose exec investment_worker \
  celery -A backend.celery_app inspect active

# View task results
redis-cli HGETALL celery-task-meta:*

# Retry failed tasks
docker-compose exec investment_backend python << 'EOF'
from backend.tasks import daily_price_update
daily_price_update.delay()  # Retry task
EOF

# Check worker logs
docker-compose logs investment_worker | grep ERROR
```

---

## Monitoring & Debugging

### View Real-Time Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f investment_backend

# Last 100 lines
docker-compose logs -n 100 investment_backend

# Search logs
docker-compose logs | grep ERROR

# Timestamp + service
docker-compose logs -t investment_backend
```

### Grafana Dashboards

```
Access: http://localhost:3001
Default login: admin/admin

Dashboards to check:
1. Application Metrics (API response times)
2. Database Metrics (query performance)
3. System Metrics (CPU, memory, disk)
4. Business Metrics (trades, recommendations)
```

### Prometheus Queries

```bash
# Open Prometheus: http://localhost:9090

# Useful queries:
rate(http_requests_total[5m])        # Request rate
histogram_quantile(0.95, ...)        # 95th percentile response time
up                                   # Service uptime
pg_connections                       # Active DB connections
redis_memory_used_bytes              # Redis memory usage
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

docker-compose down
docker-compose up -d

# Check debug output
docker-compose logs investment_backend | grep DEBUG
```

---

## Getting Help

### Before Contacting Support

1. Run health check: `./health-check.sh`
2. Collect logs: `docker-compose logs > /tmp/logs.txt`
3. Check system resources: `docker stats --no-stream`
4. Review error messages carefully
5. Search this guide for similar issues

### Contact Information

```
Technical Support:
- Email: support@yourdomain.com
- Phone: [support-number]
- Slack: #support-channel

On-Call Engineer (after hours):
- Phone: [emergency-number]

Security Issues:
- Email: security@yourdomain.com
- Do NOT use public channels
```

### Provide for Support

When contacting support, include:

1. **Error Message**: Full error text
2. **Service Status**: `docker-compose ps` output
3. **System Info**: `docker stats --no-stream`
4. **Recent Logs**: `docker-compose logs --tail 50`
5. **Steps to Reproduce**: How to trigger the issue
6. **When it Started**: Approximate time issue began
7. **Environment**: Dev/Staging/Production, OS, Docker version

### Emergency Escalation

If critical service is down:

1. Call emergency number immediately
2. Follow incident response procedure
3. Be ready with system information
4. Follow on-call engineer's instructions

---

## Appendix: Useful Commands

```bash
# System Info
docker-compose version
docker version
docker info

# Container Management
docker-compose ps              # List all containers
docker-compose logs -f         # Follow all logs
docker-compose exec <svc> bash # Shell access
docker-compose restart <svc>   # Restart service
docker-compose scale <svc>=3   # Scale service

# Database
psql -U investment_user -d investment_db -c "SELECT 1;"
pg_dump -U investment_user investment_db > backup.sql

# Redis
redis-cli INFO
redis-cli KEYS "*"
redis-cli FLUSHDB

# Monitoring
docker stats --no-stream
df -h
free -h
ps aux --sort -%cpu
```

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-27*
*Maintained by: Support Team*
