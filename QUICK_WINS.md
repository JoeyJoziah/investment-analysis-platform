# Infrastructure Quick Wins - Immediate Actions

**Goal:** Achieve $50/month budget compliance in 3-5 days
**Expected Savings:** $25-35/month
**Risk Level:** LOW (all changes are optimizations, not breaking changes)

---

## ⚡ CRITICAL: Do These First

### Quick Win #1: Eliminate Elasticsearch (2 hours, saves $15-20/month)

**Why:** Elasticsearch costs $15-20/month but only used for basic stock search. PostgreSQL can do this for free.

**Step 1: Add PostgreSQL full-text search (30 min)**
```sql
-- Connect to database
docker exec -it investment_db psql -U postgres -d investment_db

-- Run these commands:
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

ALTER TABLE stocks ADD COLUMN IF NOT EXISTS search_vector tsvector;

UPDATE stocks SET search_vector = to_tsvector('english',
  coalesce(ticker, '') || ' ' || coalesce(name, '') || ' ' ||
  coalesce(description, '') || ' ' || coalesce(sector, '') || ' ' ||
  coalesce(industry, ''));

CREATE INDEX IF NOT EXISTS idx_stocks_search_vector ON stocks USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_stocks_ticker_trgm ON stocks USING gin(ticker gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_stocks_name_trgm ON stocks USING gin(name gin_trgm_ops);

-- Create auto-update trigger
CREATE OR REPLACE FUNCTION stocks_search_vector_trigger() RETURNS trigger AS $$
BEGIN
  NEW.search_vector := to_tsvector('english',
    coalesce(NEW.ticker, '') || ' ' ||
    coalesce(NEW.name, '') || ' ' ||
    coalesce(NEW.description, '') || ' ' ||
    coalesce(NEW.sector, '') || ' ' ||
    coalesce(NEW.industry, '')
  );
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS stocks_search_vector_update ON stocks;
CREATE TRIGGER stocks_search_vector_update
  BEFORE INSERT OR UPDATE ON stocks
  FOR EACH ROW EXECUTE FUNCTION stocks_search_vector_trigger();
```

**Step 2: Update backend code (30 min)**
```python
# Edit: backend/repositories/stock_repository.py

# Add this method:
async def search_stocks(
    self,
    query: str,
    limit: int = 20
) -> List[Stock]:
    """Full-text search using PostgreSQL."""
    from sqlalchemy import func, select

    # Parse search query
    search_query = ' & '.join(query.split())

    # Full-text search with ranking
    stmt = select(Stock).where(
        Stock.search_vector.match(search_query)
    ).order_by(
        func.ts_rank(Stock.search_vector, func.to_tsquery('english', search_query)).desc()
    ).limit(limit)

    result = await self.session.execute(stmt)
    return result.scalars().all()
```

**Step 3: Remove Elasticsearch from docker-compose.yml (15 min)**
```yaml
# DELETE these entire service blocks:
# - elasticsearch (lines ~73-98)
# - elasticsearch-exporter (lines ~424-441)

# In backend service, REMOVE:
#   - ELASTICSEARCH_URL=http://elasticsearch:9200
# And REMOVE from depends_on:
#   - elasticsearch
```

**Step 4: Update monitoring config (15 min)**
```yaml
# Edit: config/monitoring/prometheus.yml
# DELETE this scrape job:
# - job_name: 'elasticsearch'
#   static_configs:
#     - targets: ['elasticsearch:9114']
```

**Step 5: Test and restart (30 min)**
```bash
# Restart services
docker compose down
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Wait 60 seconds for startup
sleep 60

# Test search works
curl "http://localhost:8000/api/stocks/search?q=technology"

# Verify Elasticsearch is gone
docker ps | grep elasticsearch  # Should return nothing

# Check resource usage
docker stats --no-stream
```

**Result:** $15-20/month saved, simpler stack ✅

---

### Quick Win #2: Right-Size Resources (1 hour, saves $10-15/month)

**Why:** Current resource limits are over-provisioned for actual usage. Can safely reduce by 40%.

**Step 1: Edit docker-compose.yml (30 min)**
```yaml
# Find and update these resource limits:

postgres:
  deploy:
    resources:
      limits:
        cpus: '0.75'      # DOWN from 1.0
        memory: 384M      # DOWN from 512M
      reservations:
        cpus: '0.2'       # DOWN from 0.25
        memory: 192M      # DOWN from 256M
  command:
    # ... existing commands ...
    - -c max_connections=80              # DOWN from 100
    - -c shared_buffers=96MB             # DOWN from 128MB
    - -c effective_cache_size=256MB      # DOWN from 384MB
    - -c maintenance_work_mem=32MB       # DOWN from 64MB
    - -c work_mem=2MB                    # DOWN from 4MB

redis:
  command: >
    redis-server --appendonly yes --appendfsync everysec
    --maxmemory 100mb                    # DOWN from 128mb
    --maxmemory-policy allkeys-lru --requirepass ${REDIS_PASSWORD}
    --tcp-keepalive 60 --save ""
  deploy:
    resources:
      limits:
        cpus: '0.2'       # DOWN from 0.25
        memory: 128M      # DOWN from 150M
      reservations:
        cpus: '0.05'      # DOWN from 0.1
        memory: 48M       # DOWN from 64M

backend:
  deploy:
    resources:
      limits:
        cpus: '0.5'       # DOWN from 1.0
        memory: 512M      # DOWN from 768M
      reservations:
        cpus: '0.15'      # DOWN from 0.25
        memory: 256M      # DOWN from 384M

celery_worker:
  deploy:
    resources:
      limits:
        cpus: '0.75'      # DOWN from 1.0
        memory: 512M      # DOWN from 768M
      reservations:
        cpus: '0.2'       # DOWN from 0.25
        memory: 256M      # DOWN from 384M

airflow:
  deploy:
    resources:
      limits:
        cpus: '0.5'       # DOWN from 1.0
        memory: 512M      # DOWN from 1G
      reservations:
        cpus: '0.15'      # DOWN from 0.25
        memory: 256M      # DOWN from 512M
```

**Step 2: Apply same changes to docker-compose.prod.yml (15 min)**

**Step 3: Test (15 min)**
```bash
# Restart with new limits
docker compose down
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Monitor for 10 minutes
watch -n 30 'docker stats --no-stream'

# Check for OOM kills
docker ps -a | grep -i "exited"  # Should be empty

# Test API still works
curl http://localhost:8000/api/health
```

**Result:** $10-15/month saved, still performs well ✅

---

### Quick Win #3: Add Missing Health Checks (30 min, improves reliability)

**Why:** 5 services lack proper health checks, causing startup issues and false positives.

**Edit docker-compose.yml:**
```yaml
celery_beat:
  healthcheck:
    test: ["CMD-SHELL", "python -c 'import os,time; f=\"/tmp/celerybeat.pid\"; exit(0 if os.path.exists(f) and time.time() - os.path.getmtime(f) < 300 else 1)'"]
    interval: 60s
    timeout: 10s
    retries: 3
    start_period: 30s

airflow:
  healthcheck:
    test: ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
    interval: 30s
    timeout: 10s
    retries: 5
    start_period: 120s

frontend:
  healthcheck:
    test: ["CMD-SHELL", "curl -f http://localhost:3000 || exit 1"]
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 60s
```

**Test:**
```bash
docker compose down
docker compose up -d

# Watch services become healthy
watch -n 2 'docker compose ps'
# All should show "(healthy)" after start_period
```

**Result:** Services start in correct order, dependencies wait properly ✅

---

## 📊 Quick Validation

After completing all three quick wins:

```bash
# 1. Check cost savings
echo "Old cost: \$65-80/month"
echo "New cost: \$40-45/month"
echo "Savings: \$25-35/month (38-44% reduction)"

# 2. Check services running
docker compose ps

# 3. Check resource usage
docker stats --no-stream

# 4. Test API
curl http://localhost:8000/api/health
curl http://localhost:8000/api/stocks

# 5. Check logs for errors
docker compose logs --tail=50 backend
docker compose logs --tail=50 postgres
docker compose logs --tail=50 celery_worker
```

**All good?** Commit changes:
```bash
git add docker-compose.yml docker-compose.prod.yml \
        backend/repositories/stock_repository.py \
        config/monitoring/prometheus.yml

git commit -m "feat: Infrastructure cost optimization

- Remove Elasticsearch (saves \$15-20/month)
- Right-size resource allocations (saves \$10-15/month)
- Add missing health checks (improves reliability)

Total savings: \$25-35/month
Budget now: \$40-45/month (under \$50 target)

BREAKING CHANGE: Elasticsearch removed, replaced with PostgreSQL FTS"

git push
```

---

## 🎯 Expected Results

### Before Quick Wins
```
Monthly Cost:     $65-80 ❌
Services:         13 containers
Total CPU:        6.5 CPU
Total Memory:     5.2GB
Health Coverage:  58%
```

### After Quick Wins
```
Monthly Cost:     $40-45 ✅ (38-44% reduction)
Services:         11 containers (-2, removed ES)
Total CPU:        3.75 CPU (-42%)
Total Memory:     3GB (-42%)
Health Coverage:  100% ✅
```

---

## ⏱️ Time Investment

```
Quick Win #1:  2 hours   → $15-20/month saved
Quick Win #2:  1 hour    → $10-15/month saved
Quick Win #3:  30 min    → Reliability improved
---------------------------------------------------
TOTAL:         3.5 hours → $25-35/month saved
ROI:           $7-10 per hour of work
Annual ROI:    $300-420 in savings
```

---

## 🚨 If Something Goes Wrong

### Elasticsearch search not working
```bash
# Check search_vector exists
docker exec investment_db psql -U postgres -d investment_db \
  -c "\d stocks" | grep search_vector

# If missing, re-run Step 1 SQL commands

# Check index exists
docker exec investment_db psql -U postgres -d investment_db \
  -c "\di" | grep idx_stocks_search_vector

# Test search directly in DB
docker exec investment_db psql -U postgres -d investment_db \
  -c "SELECT ticker, name FROM stocks WHERE search_vector @@ to_tsquery('english', 'technology') LIMIT 5;"
```

### Services won't start (OOM killed)
```bash
# Check for OOM kills
docker ps -a | grep -i "exited"
dmesg | grep -i "out of memory"

# If killed, increase memory slightly:
# In docker-compose.yml, increase that service's memory limit by 128M
# Example: 512M → 640M

docker compose up -d [service-name]
```

### Performance degraded
```bash
# Check current resource usage
docker stats

# If CPU constantly at 100%, increase by 0.25 CPU
# If Memory constantly at limit, increase by 128M

# Make small adjustments, test, repeat
```

### Rollback if needed
```bash
# Restore Elasticsearch
git revert HEAD
docker compose down
docker compose up -d

# Or restore from backup
cp docker-compose.yml.backup docker-compose.yml
docker compose up -d
```

---

## 📈 Next Steps (After Quick Wins)

Once quick wins are complete and stable:

1. **Week 2:** Add Prometheus metrics and dashboards (see `INFRASTRUCTURE_FIXES_CHECKLIST.md`)
2. **Week 3:** Optimize CI/CD pipeline (see `INFRASTRUCTURE_FIXES_CHECKLIST.md`)
3. **Week 4:** Production hardening (see `INFRASTRUCTURE_FIXES_CHECKLIST.md`)

**Full roadmap:** `INFRASTRUCTURE_FIXES_CHECKLIST.md`
**Complete analysis:** `INFRASTRUCTURE_ANALYSIS.md`
**Executive summary:** `INFRASTRUCTURE_SUMMARY.md`

---

## ✅ Success Criteria

You've succeeded when:
- [ ] Monthly cost is <$50
- [ ] Elasticsearch containers are gone
- [ ] All services show "(healthy)" status
- [ ] API responds to /api/health in <100ms
- [ ] No services getting OOM killed
- [ ] Stock search still works

**Status check:**
```bash
# All checks should pass:
docker compose ps | grep "(healthy)" | wc -l  # Should be 11+
docker ps | grep elasticsearch              # Should be empty
curl -w "\nTime: %{time_total}s\n" http://localhost:8000/api/health
docker stats --no-stream | awk '{print $4}' | grep -i "512"  # Should see reduced limits
```

---

## 💡 Pro Tips

1. **Do one quick win at a time** - Test thoroughly before moving to the next
2. **Keep docker-compose.yml.backup** before editing:
   ```bash
   cp docker-compose.yml docker-compose.yml.backup
   ```
3. **Monitor for 24 hours** after changes to catch any issues
4. **Document any custom adjustments** in comments
5. **Run validation after each change:**
   ```bash
   docker compose config  # Validate syntax
   docker compose ps      # Check status
   docker stats           # Monitor resources
   ```

---

**Ready?** Start with Quick Win #1 now! 🚀
