# Infrastructure Fixes - Implementation Checklist

**Goal:** Bring infrastructure costs from $65-80/month down to <$50/month while improving reliability and observability.

**Total Estimated Time:** 3-4 weeks
**Expected Savings:** $25-35/month
**Expected Performance Improvement:** 66% faster CI/CD, 100% monitoring coverage

---

## Phase 1: Critical Budget Fixes (Week 1) ⚠️ DO FIRST

### Day 1-2: Eliminate Elasticsearch ($15-20/month savings)

- [ ] **Step 1:** Database Setup
  ```bash
  # Connect to PostgreSQL
  docker exec -it investment_db psql -U postgres -d investment_db

  # Run SQL commands:
  CREATE EXTENSION IF NOT EXISTS pg_trgm;
  CREATE EXTENSION IF NOT EXISTS unaccent;

  ALTER TABLE stocks ADD COLUMN search_vector tsvector;

  UPDATE stocks SET search_vector = to_tsvector('english',
    coalesce(ticker, '') || ' ' || coalesce(name, '') || ' ' ||
    coalesce(description, '') || ' ' || coalesce(sector, '') || ' ' ||
    coalesce(industry, ''));

  CREATE INDEX idx_stocks_search_vector ON stocks USING gin(search_vector);
  CREATE INDEX idx_stocks_ticker_trgm ON stocks USING gin(ticker gin_trgm_ops);
  CREATE INDEX idx_stocks_name_trgm ON stocks USING gin(name gin_trgm_ops);
  ```

- [ ] **Step 2:** Update Stock Repository
  ```bash
  # Edit: backend/repositories/stock_repository.py
  # Add: search_stocks() and fuzzy_search_stocks() methods
  # See Appendix B in INFRASTRUCTURE_ANALYSIS.md for code
  ```

- [ ] **Step 3:** Remove Elasticsearch from Docker
  ```bash
  # Edit: docker-compose.yml
  # - Remove elasticsearch service (lines 73-98)
  # - Remove elasticsearch-exporter service (lines 424-441)
  # - Update backend depends_on (remove elasticsearch)
  # - Remove ELASTICSEARCH_URL environment variable
  ```

- [ ] **Step 4:** Update Prometheus Config
  ```bash
  # Edit: config/monitoring/prometheus.yml
  # - Remove elasticsearch scrape job (lines 47-50)
  ```

- [ ] **Step 5:** Update Alert Rules
  ```bash
  # Edit: infrastructure/monitoring/alerts/investment-platform.yml
  # - Remove any elasticsearch-related alerts
  ```

- [ ] **Step 6:** Test
  ```bash
  docker compose down
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
  # Wait for services to start
  curl http://localhost:8000/api/stocks/search?q=technology
  ```

- [ ] **Step 7:** Commit Changes
  ```bash
  git add -A
  git commit -m "feat: Replace Elasticsearch with PostgreSQL full-text search

  - Add PostgreSQL FTS indexes and triggers
  - Remove Elasticsearch service (saves $15-20/month)
  - Update stock search to use PostgreSQL
  - Remove Elasticsearch monitoring

  BREAKING CHANGE: Elasticsearch no longer required"
  ```

**Validation:** Search works, Elasticsearch containers gone, budget reduced by $15-20/month

---

### Day 3-4: Right-Size Resources ($10-15/month savings)

- [ ] **Step 1:** Update docker-compose.yml
  ```bash
  # Edit resource limits for each service:
  # See section 1.2 in INFRASTRUCTURE_ANALYSIS.md for exact values

  # postgres: 1.0 CPU → 0.75 CPU, 512M → 384M
  # redis: 0.25 CPU → 0.2 CPU, 150M → 128M
  # backend: 1.0 CPU → 0.5 CPU, 768M → 512M
  # celery_worker: 1.0 CPU → 0.75 CPU, 768M → 512M
  # airflow: 1.0 CPU → 0.5 CPU, 1G → 512M
  ```

- [ ] **Step 2:** Update docker-compose.prod.yml
  ```bash
  # Apply same resource optimizations to production overrides
  ```

- [ ] **Step 3:** Test Resource Limits
  ```bash
  docker compose down
  docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

  # Monitor for 10 minutes
  watch -n 10 'docker stats --no-stream'

  # Check for OOM kills
  docker ps -a | grep -i "exited"
  ```

- [ ] **Step 4:** Load Test
  ```bash
  # Install apache bench if needed: sudo apt-get install apache2-utils

  # Test backend under load
  ab -n 5000 -c 25 http://localhost:8000/api/health

  # Monitor during load
  docker stats
  ```

- [ ] **Step 5:** Commit Changes
  ```bash
  git add docker-compose.yml docker-compose.prod.yml
  git commit -m "perf: Optimize resource allocation for $50/month budget

  - Reduce CPU/memory limits by 40% across services
  - Tested under load - no performance degradation
  - Projected savings: $10-15/month

  Services now right-sized for actual usage"
  ```

**Validation:** All services run smoothly, no OOM kills under load

---

### Day 5: Add Missing Health Checks

- [ ] **Step 1:** Update docker-compose.yml health checks
  ```yaml
  # Add/fix health checks for:
  # - celery_beat (improve PID check)
  # - airflow (add health endpoint check)
  # - frontend (add curl check)

  # See section 1.2 Change 4 in INFRASTRUCTURE_ANALYSIS.md
  ```

- [ ] **Step 2:** Update Service Dependencies
  ```yaml
  # backend depends_on:
  #   postgres:
  #     condition: service_healthy  # ✓ Already correct
  #   redis:
  #     condition: service_healthy  # ✓ Already correct
  ```

- [ ] **Step 3:** Test Service Startup
  ```bash
  docker compose down
  docker compose up -d

  # Watch services come up in order
  docker compose ps
  watch -n 2 'docker compose ps'
  ```

- [ ] **Step 4:** Commit Changes
  ```bash
  git add docker-compose.yml
  git commit -m "fix: Add missing health checks for reliability

  - Add health check to celery_beat
  - Add health check to airflow
  - Add health check to frontend
  - Improve service startup ordering

  Services now start in correct dependency order"
  ```

**Validation:** Services start in correct order, dependencies wait for health

**Phase 1 Complete:** Budget reduced from $65-80 to $45-50/month ✅

---

## Phase 2: Monitoring & Observability (Week 2)

### Day 1-2: Add Prometheus Instrumentation

- [ ] **Step 1:** Create prometheus_metrics.py
  ```bash
  # Create: backend/monitoring/prometheus_metrics.py
  # See section 2.2 in INFRASTRUCTURE_ANALYSIS.md for complete code
  ```

- [ ] **Step 2:** Create cost_metrics_updater.py
  ```bash
  # Create: backend/utils/cost_metrics_updater.py
  # See section 4.3 in INFRASTRUCTURE_ANALYSIS.md for complete code
  ```

- [ ] **Step 3:** Update main.py
  ```python
  # Edit: backend/api/main.py
  # Add metrics router
  # Add cost metrics updater to startup
  ```

- [ ] **Step 4:** Update prometheus.yml
  ```yaml
  # Edit: config/monitoring/prometheus.yml
  # Add backend-metrics scrape job
  # Add cost-monitor scrape job
  # See section 2.2 in INFRASTRUCTURE_ANALYSIS.md
  ```

- [ ] **Step 5:** Test Metrics
  ```bash
  # Restart services
  docker compose restart backend prometheus

  # Check metrics endpoint
  curl http://localhost:8000/api/metrics
  curl http://localhost:8000/api/metrics/cost

  # Check Prometheus targets
  open http://localhost:9090/targets
  ```

- [ ] **Step 6:** Commit
  ```bash
  git add backend/monitoring/prometheus_metrics.py \
         backend/utils/cost_metrics_updater.py \
         backend/api/main.py \
         config/monitoring/prometheus.yml
  git commit -m "feat: Add Prometheus instrumentation for business metrics

  - Add custom metrics exporter
  - Add cost tracking to Prometheus
  - Add real-time budget monitoring
  - Add API rate limit tracking

  Cost and business metrics now visible in Prometheus/Grafana"
  ```

**Validation:** Metrics visible in Prometheus, cost data updating

---

### Day 3: Add Celery Exporter

- [ ] **Step 1:** Add celery-exporter service
  ```yaml
  # Edit: docker-compose.yml
  # Add celery-exporter service after redis-exporter
  # See section 2.2 in INFRASTRUCTURE_ANALYSIS.md
  ```

- [ ] **Step 2:** Update Prometheus scrape config
  ```yaml
  # Edit: config/monitoring/prometheus.yml
  # Change celery job target from celery_worker:9540 to celery-exporter:9540
  ```

- [ ] **Step 3:** Test
  ```bash
  docker compose up -d celery-exporter
  sleep 10

  # Check exporter
  curl http://localhost:9540/metrics

  # Check Prometheus
  open http://localhost:9090/targets
  ```

- [ ] **Step 4:** Commit
  ```bash
  git add docker-compose.yml config/monitoring/prometheus.yml
  git commit -m "feat: Add Celery metrics exporter

  - Add celery-exporter service
  - Fix Celery metrics scraping in Prometheus
  - Worker queue depth now visible

  Celery worker performance now monitored"
  ```

**Validation:** Celery metrics appearing in Prometheus

---

### Day 4-5: Create Custom Dashboards

- [ ] **Step 1:** Create investment platform dashboard
  ```bash
  # Create: infrastructure/monitoring/dashboards/investment-platform.json
  # See section 2.2 in INFRASTRUCTURE_ANALYSIS.md for JSON
  ```

- [ ] **Step 2:** Create cost monitoring dashboard
  ```bash
  # Create: infrastructure/monitoring/dashboards/cost-monitoring.json
  # Custom dashboard focused on budget tracking
  ```

- [ ] **Step 3:** Update Grafana provisioning
  ```yaml
  # Edit: infrastructure/monitoring/grafana/provisioning/dashboards/dashboards.yml
  # Ensure it loads all dashboards from dashboards directory
  ```

- [ ] **Step 4:** Import dashboards
  ```bash
  docker compose restart grafana

  # Check dashboards loaded
  open http://localhost:3001
  # Login: admin / (check .env for password)
  # Navigate to Dashboards
  ```

- [ ] **Step 5:** Commit
  ```bash
  git add infrastructure/monitoring/dashboards/
  git commit -m "feat: Add custom investment platform dashboards

  - Add operations dashboard (services, metrics, alerts)
  - Add cost monitoring dashboard (budget, API limits)
  - Auto-provision on Grafana startup

  Real-time business metrics now visualized"
  ```

**Validation:** Dashboards visible in Grafana, showing real data

**Phase 2 Complete:** Full observability with cost tracking ✅

---

## Phase 3: CI/CD Optimization (Week 3)

### Day 1-2: Parallelize Workflows

- [ ] **Step 1:** Update CI workflow
  ```yaml
  # Edit: .github/workflows/ci.yml
  # Restructure jobs to run in parallel
  # Split frontend into quality and test jobs
  # See section 3.2 Change 1 in INFRASTRUCTURE_ANALYSIS.md
  ```

- [ ] **Step 2:** Add path filters
  ```yaml
  # Add paths filter to trigger conditions
  # See section 3.2 Change 3
  ```

- [ ] **Step 3:** Test workflow
  ```bash
  # Make a small change
  echo "# Test" >> README.md
  git add README.md
  git commit -m "test: CI optimization"
  git push

  # Watch workflow on GitHub Actions
  # Verify jobs run in parallel
  ```

- [ ] **Step 4:** Commit
  ```bash
  git add .github/workflows/ci.yml
  git commit -m "perf: Parallelize CI pipeline for faster builds

  - Run quality checks in parallel (backend, frontend, security)
  - Split frontend build into quality + test jobs
  - Add path filters to skip non-code changes

  CI duration reduced from 45-60 min to 15-20 min (66% faster)"
  ```

**Validation:** CI completes in <20 minutes

---

### Day 3-4: Add Caching

- [ ] **Step 1:** Improve Python dependency caching
  ```yaml
  # Edit: .github/workflows/ci.yml
  # Update cache configuration
  # See section 3.2 Change 2
  ```

- [ ] **Step 2:** Add Docker layer caching
  ```yaml
  # Edit: .github/workflows/ci.yml
  # Add Docker Buildx setup
  # Add cache configuration
  # See section 3.2 Change 4
  ```

- [ ] **Step 3:** Test caching
  ```bash
  # Run workflow twice
  # Second run should be much faster (cache hits)
  ```

- [ ] **Step 4:** Commit
  ```bash
  git add .github/workflows/ci.yml
  git commit -m "perf: Add aggressive caching to CI pipeline

  - Add Docker layer caching (reduces build time 80%)
  - Improve Python dependency caching
  - Cache restored across workflow runs

  Dependency install time: 5 min → 30 sec on cache hit"
  ```

**Validation:** Second workflow run significantly faster

---

### Day 5: Conditional Deployment

- [ ] **Step 1:** Add deployment conditions
  ```yaml
  # Edit: .github/workflows/ci.yml
  # Add if conditions to deployment job
  # See section 3.2 Change 5
  ```

- [ ] **Step 2:** Test
  ```bash
  # Create PR - deployment should NOT run
  git checkout -b test-branch
  git push origin test-branch
  # Create PR on GitHub

  # Merge to main - deployment SHOULD run
  ```

- [ ] **Step 3:** Commit
  ```bash
  git add .github/workflows/ci.yml
  git commit -m "fix: Only deploy on push to main branch

  - Skip deployment for PRs
  - Saves GitHub Actions minutes
  - Prevents accidental deployments

  Estimated savings: $5-8/month"
  ```

**Validation:** Deployment only runs on main branch

**Phase 3 Complete:** CI/CD optimized, $5-8/month saved ✅

---

## Phase 4: Production Hardening (Week 4)

### Day 1-2: Load Testing

- [ ] **Step 1:** Create load test script
  ```bash
  # Create: scripts/infrastructure/test_resource_limits.sh
  # See section 6.1 in INFRASTRUCTURE_ANALYSIS.md
  chmod +x scripts/infrastructure/test_resource_limits.sh
  ```

- [ ] **Step 2:** Run load tests
  ```bash
  ./scripts/infrastructure/test_resource_limits.sh
  ```

- [ ] **Step 3:** Analyze results
  ```bash
  # Check for:
  # - OOM kills (none expected)
  # - CPU throttling (some expected, acceptable)
  # - Response time degradation (should be minimal)
  ```

- [ ] **Step 4:** Fine-tune if needed
  ```bash
  # If issues found, adjust resource limits slightly
  # Re-test until stable
  ```

**Validation:** System stable under load with optimized resources

---

### Day 3-4: Backup & DR Testing

- [ ] **Step 1:** Test backup service
  ```bash
  # Trigger manual backup
  docker exec investment_backup /backup.sh

  # Check backup created
  ls -lh backups/
  ```

- [ ] **Step 2:** Test restore
  ```bash
  # Create test database
  docker exec investment_db psql -U postgres -c "CREATE DATABASE test_restore;"

  # Restore latest backup
  docker exec investment_backup /restore-backup.sh test_restore

  # Verify data
  docker exec investment_db psql -U postgres -d test_restore -c "SELECT COUNT(*) FROM stocks;"
  ```

- [ ] **Step 3:** Document DR procedure
  ```bash
  # Create: docs/DISASTER_RECOVERY.md
  # Include:
  # - Backup frequency
  # - Restore procedure
  # - RTO/RPO targets
  # - Contact information
  ```

**Validation:** Backups work, restore tested, DR documented

---

### Day 5: Documentation

- [ ] **Step 1:** Update deployment docs
  ```bash
  # Edit: docs/DEPLOYMENT.md
  # Add:
  # - Resource optimization notes
  # - Cost monitoring guide
  # - Troubleshooting section
  ```

- [ ] **Step 2:** Update monitoring docs
  ```bash
  # Edit: docs/MONITORING.md
  # Add:
  # - Custom metrics reference
  # - Dashboard guide
  # - Alert thresholds
  ```

- [ ] **Step 3:** Create cost optimization guide
  ```bash
  # Create: docs/COST_OPTIMIZATION.md
  # Document:
  # - Budget breakdown
  # - Cost monitoring
  # - Optimization strategies
  ```

- [ ] **Step 4:** Final commit
  ```bash
  git add docs/ scripts/
  git commit -m "docs: Complete infrastructure optimization documentation

  - Add disaster recovery procedures
  - Document cost optimization strategies
  - Update deployment and monitoring guides
  - Add load testing scripts

  Infrastructure now production-ready and well-documented"
  ```

**Phase 4 Complete:** Production-ready infrastructure ✅

---

## Validation Checklist

After completing all phases, verify:

### Budget Compliance
- [ ] Monthly cost projection: <$50/month
- [ ] Real-time cost tracking in Grafana
- [ ] Budget alerts configured

### Monitoring Coverage
- [ ] All services have health checks
- [ ] All services have Prometheus exporters
- [ ] Custom dashboards showing business metrics
- [ ] Cost metrics updating in real-time

### Performance
- [ ] CI/CD completes in <20 minutes
- [ ] API response time <500ms p95
- [ ] Database query time <100ms p95
- [ ] No OOM kills under load

### Reliability
- [ ] Services start in correct order
- [ ] Backup/restore tested and working
- [ ] Alerts firing for real issues
- [ ] DR procedure documented

### Documentation
- [ ] Deployment guide updated
- [ ] Monitoring guide updated
- [ ] Cost optimization guide created
- [ ] Troubleshooting guide created

---

## Quick Reference

### Cost Breakdown (Target)
```
Service          Monthly Cost
postgres         $6-8
redis            $2
backend          $8-10
celery_worker    $6-8
airflow          $5-7
prometheus       $3-4
grafana          $2
frontend         $2
nginx            $1-2
Domain/SSL       $10
----------------
TOTAL            $45-50 ✅
```

### Key Commands
```bash
# Start services
./start.sh prod

# Check resource usage
docker stats

# View cost metrics
curl http://localhost:8000/api/metrics/cost

# Check Prometheus targets
open http://localhost:9090/targets

# View Grafana dashboards
open http://localhost:3001

# Run load tests
./scripts/infrastructure/test_resource_limits.sh

# Validate costs
python scripts/infrastructure/validate_costs.py

# Backup database
docker exec investment_backup /backup.sh
```

### Support Files
- Full analysis: `INFRASTRUCTURE_ANALYSIS.md`
- Cost monitoring: `backend/utils/persistent_cost_monitor.py`
- Metrics: `backend/monitoring/prometheus_metrics.py`
- Alerts: `infrastructure/monitoring/alerts/investment-platform.yml`

---

**Implementation Status:** Ready to begin
**Expected Timeline:** 3-4 weeks
**Expected Outcome:** $50/month budget compliance, full observability, production-ready
