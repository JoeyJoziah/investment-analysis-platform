> **ARCHIVED 2026-04-27 by 13-infra-deployment**
> Original: docs/TROUBLESHOOTING.md
> Validation summary: see ../../reports/13-infra-deployment.md §2 for per-claim status.

# Troubleshooting Guide

**Last Updated**: 2026-03-04

---

## Quick Diagnostics

### Check All Service Health

```bash
# Show all container statuses
docker compose ps

# Follow all logs
docker compose logs -f

# Per-service health check
docker compose exec postgres pg_isready -U postgres
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" ping

# Backend API health endpoint
curl http://localhost:8000/api/health
```

---

## Common Issues

### Issue 1: Database Connection Refused

**Symptoms**: Backend won't start, `psycopg2.OperationalError`, error on port 5432.

**Diagnosis**:
```bash
docker compose ps postgres
docker compose logs postgres | tail -20
```

**Solutions** (try in order):

```bash
# 1. Wait for startup and retry
docker compose up -d postgres
sleep 30
docker compose logs postgres | grep "database system is ready"

# 2. Check .env credentials match
grep DB_ .env

# 3. Verify investment_user role exists
docker compose exec postgres psql -U postgres -c "\du investment_user"

# 4. Check port conflict (nothing else on 5432)
lsof -i :5432

# 5. Full reset (WARNING: destroys data)
docker compose down postgres
docker volume rm investment-analysis-platform_postgres_data
docker compose up -d postgres
```

---

### Issue 2: Backend Exits Immediately - GDPR Key Missing

**Symptoms**: Container exits with code 1, log shows `GDPR_ENCRYPTION_KEY not configured` or `AttributeError: 'NoneType' object`.

**Solution**:

```bash
# Generate a Fernet key
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Add to .env
echo "GDPR_ENCRYPTION_KEY=$FERNET_KEY" >> .env

# Restart
docker compose restart backend

# Verify startup
docker compose logs backend | grep -E "Starting|Application startup"
```

---

### Issue 3: Out of Memory (OOMKilled)

**Symptoms**: Containers crash randomly, exit code 137, `OOMKilled` in `docker inspect`.

**Diagnosis**:
```bash
docker stats --no-stream
docker inspect <container_id> | grep -i oom
```

**Solutions**:

```bash
# 1. Increase Docker Desktop memory limit (macOS/Windows):
# Docker Desktop > Settings > Resources > Memory > 8GB+

# 2. Increase container limits in docker-compose.yml
# Under services.backend.deploy.resources.limits:
#   memory: 2g   (increase from 1g)

# 3. Monitor memory consumption
watch -n 2 'docker stats --no-stream | head -10'

# 4. Clear unused Docker objects
docker system prune -a  # WARNING: removes all unused images
```

---

### Issue 4: Disk Space Full

**Symptoms**: `No space left on device`, services stop responding, logs stop writing.

**Diagnosis**:
```bash
df -h
docker system df
docker compose exec postgres psql -U investment_user -d investment_db \
  -c "SELECT pg_size_pretty(pg_database_size('investment_db'));"
```

**Solutions**:

```bash
# Clean unused Docker resources (images, stopped containers, build cache)
docker system prune -a --volumes  # WARNING: removes unused volumes too

# Archive old price history
docker compose exec postgres psql -U investment_user -d investment_db \
  -c "DELETE FROM price_history WHERE date < NOW() - interval '1 year';"

# Truncate old container logs
# Edit daemon.json: set log-driver json-file with max-size/max-file
```

---

### Issue 5: API Slow (>2s Response Time)

**Symptoms**: API calls take multiple seconds, frontend sluggish, timeout errors.

**Diagnosis**:
```bash
time curl http://localhost:8000/api/health

# Check Redis cache hit rate
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" INFO stats | grep -E "hits|misses"

# Find slow queries
docker compose exec postgres psql -U investment_user -d investment_db \
  -c "SELECT query, mean_exec_time, calls FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 5;"

# Check resource usage
docker stats --no-stream
```

**Solutions**:

```bash
# 1. Add missing indexes
docker compose exec postgres psql -U investment_user -d investment_db << 'EOF'
CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stocks(ticker);
CREATE INDEX IF NOT EXISTS idx_price_history_stock_date ON price_history(stock_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);
ANALYZE;
EOF

# 2. Increase Redis memory if evictions are high
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" INFO stats | grep evicted_keys

# 3. Scale backend replicas (production)
docker compose -f docker-compose.production.yml up -d --scale backend=2
```

---

### Issue 6: Stock Data Not Loading

**Symptoms**: API returns empty results, `SELECT COUNT(*) FROM stocks` returns 0.

**Diagnosis**:
```bash
docker compose exec postgres psql -U investment_user -d investment_db \
  -c "SELECT COUNT(*) FROM stocks;"

docker compose logs celery_worker | grep -i "error\|exception"
docker compose logs airflow | tail -30
```

**Solutions**:

```bash
# 1. Test API key connectivity
curl "https://finnhub.io/api/v1/quote?symbol=AAPL&token=$FINNHUB_API_KEY"

# 2. Run stock loader manually
docker compose exec backend python backend/scripts/load_stocks.py --verbose

# 3. Trigger Airflow DAG manually
docker compose exec airflow airflow dags trigger daily_stock_pipeline

# 4. Check ETL configuration
grep -E "ALPHA_VANTAGE|FINNHUB|POLYGON" .env
```

---

## Service-Specific Troubleshooting

### PostgreSQL

```bash
# Won't start - check logs
docker compose logs postgres

# Too many connections
docker compose exec postgres psql -U postgres \
  -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
docker compose exec postgres psql -U postgres -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE state = 'idle'
    AND state_change < now() - interval '10 minutes';"
```

### Redis

```bash
# Test connection
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" ping

# Check memory usage
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" INFO memory | grep used_memory_human

# Flush cache only (DB 0)
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" -n 0 FLUSHDB

# Check Celery queue depth
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" LLEN celery
```

### Celery Workers

```bash
# Check active tasks
docker compose exec celery_worker celery -A backend.tasks.celery_app inspect active

# View worker logs
docker compose logs celery_worker | grep -E "ERROR|WARNING"

# Restart workers
docker compose restart celery_worker celery_beat

# Scale workers up
docker compose up -d --scale celery_worker=3

# Purge stuck task queue (caution: loses pending tasks)
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" DEL celery
```

---

## Database Issues

### Find Slow Queries

```bash
# Enable slow query logging (threshold: 1 second)
docker compose exec postgres psql -U postgres << 'EOF'
ALTER SYSTEM SET log_min_duration_statement = 1000;
SELECT pg_reload_conf();
EOF

# View slow query log
docker compose exec postgres \
  tail -f /var/log/postgresql/postgresql.log | grep "duration:"
```

### Restore From Backup

```bash
# 1. Stop application (not database)
docker compose stop backend celery_worker celery_beat

# 2. Restore
docker compose exec -T postgres \
  psql -U investment_user -d investment_db < ./backups/backup-20260304.sql

# 3. Verify
docker compose exec postgres psql -U investment_user -d investment_db \
  -c "SELECT COUNT(*) FROM stocks;"

# 4. Restart
docker compose start backend celery_worker celery_beat
```

---

## Authentication Issues

### Can't Login (401 Unauthorized)

```bash
# 1. Verify user exists
docker compose exec postgres psql -U investment_user -d investment_db \
  -c "SELECT id, email FROM users WHERE email = 'user@example.com';"

# 2. Test login endpoint directly
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user@example.com","password":"yourpassword"}'

# 3. Check JWT configuration
grep JWT .env

# 4. Reset password manually
docker compose exec backend python3 << 'EOF'
from backend.utils.security import hash_password
from backend.models.unified_models import User
from backend.config.database import SessionLocal
db = SessionLocal()
user = db.query(User).filter_by(email='user@example.com').first()
if user:
    user.hashed_password = hash_password('NewPassword123!')
    db.commit()
    print("Password reset")
EOF
```

---

## Monitoring Issues

### Grafana Not Showing Data

```bash
# 1. Confirm Prometheus is scraping
curl http://localhost:9090/api/v1/targets | python3 -m json.tool | grep health

# 2. Check Prometheus datasource in Grafana
# Grafana > Configuration > Data Sources > Prometheus > Test

# 3. Verify backend exposes metrics
curl http://localhost:8000/metrics | head -20

# 4. Check Loki logs in Grafana
# Grafana > Explore > select Loki datasource > run query {container="investment_backend"}
```

### Loki Not Receiving Logs

```bash
# Check Loki readiness
curl http://localhost:3100/ready

# Check Promtail is running and connected
docker compose logs promtail | tail -20

# Verify log file path is mounted correctly
docker compose exec promtail ls /var/log/app/
```

### VictoriaMetrics Not Receiving Metrics

```bash
# Check VictoriaMetrics health
curl http://localhost:8428/health

# Verify PROMETHEUS_REMOTE_URL is set in .env
grep PROMETHEUS_REMOTE_URL .env

# Check Prometheus remote write in logs
docker compose logs prometheus | grep -i "remote\|victoria"
```

---

## Performance Issues

### High CPU Usage

```bash
# Find the top consumer
docker stats --no-stream | sort -k3 -rn

# Database CPU - find expensive queries
docker compose exec postgres psql -U investment_user -d investment_db \
  -c "SELECT query, calls, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time * calls DESC LIMIT 5;"
```

### Memory Leak Detection

```bash
# Watch backend memory over time
watch -n 5 'docker stats --no-stream investment_api_prod | grep investment'

# If growing, restart service
docker compose restart backend
```

### Database Lock Contention

```bash
# Find blocking sessions
docker compose exec postgres psql -U investment_user -d investment_db << 'EOF'
SELECT pid, usename, application_name, state, query_start, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;
EOF

# Terminate a blocking session
docker compose exec postgres psql -U postgres \
  -c "SELECT pg_terminate_backend(<pid>);"
```

---

## Useful Commands Reference

```bash
# Service management
docker compose ps                       # List all containers
docker compose logs -f <service>        # Follow service logs
docker compose restart <service>        # Restart a service
docker compose exec <service> bash      # Shell into container
docker compose up -d --scale <svc>=3    # Scale service

# Database
docker compose exec postgres psql -U investment_user -d investment_db
pg_dump -U investment_user investment_db > backup.sql

# Redis
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" INFO
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" KEYS "*"

# System resources
docker stats --no-stream
df -h
free -h
```

---

## Collecting Information for Bug Reports

When reporting an issue, include:

1. **Error message** - full text
2. **Service status** - `docker compose ps` output
3. **Recent logs** - `docker compose logs --tail 50 <service>`
4. **System resources** - `docker stats --no-stream`
5. **Environment** - dev/staging/production, OS, Docker version
6. **Steps to reproduce**

---

*Last Updated: 2026-03-04*
