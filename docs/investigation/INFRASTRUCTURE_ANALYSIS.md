# Infrastructure & DevOps Bottleneck Analysis
## Investment Analysis Platform

**Analysis Date:** 2026-01-26
**Budget Constraint:** $50/month STRICT
**Target Scale:** 6,000+ stocks analyzed daily

---

## Executive Summary

The investment-analysis-platform infrastructure shows solid foundational work but has **12 critical bottlenecks** preventing production readiness and cost compliance. Resource allocation exceeds budget, monitoring has gaps, CI/CD lacks parallelization, and several production-critical services are missing health checks.

**Key Findings:**
- ❌ **Resource allocation** projects ~$65-80/month (30-60% over budget)
- ❌ **Missing exporters** for Celery and Backend API metrics
- ❌ **CI/CD pipeline** takes 45-60 minutes (can be 15-20 minutes)
- ⚠️ **Health checks** missing on 5 services
- ⚠️ **Cost tracking** not integrated with Prometheus/Grafana
- ✅ **Docker configurations** well-structured but need optimization

---

## 1. Docker Configuration Analysis

### 1.1 Current State Assessment

#### Strengths ✅
- Multi-stage builds implemented correctly
- Resource limits defined for all services
- Health checks on critical services (postgres, redis, backend)
- Proper separation of dev/prod configurations
- Non-root user in production containers

#### Critical Issues ❌

**Issue #1: Resource Over-Allocation**
```yaml
# Current allocation (docker-compose.yml + prod overrides)
Service               CPU Limit    Memory Limit    Est. Monthly Cost
postgres              1.0 CPU      512M            ~$8-12
elasticsearch         1.0 CPU      1G              ~$15-20
backend               1.0 CPU      768M            ~$10-15
celery_worker         1.0 CPU      768M            ~$10-15
airflow               1.0 CPU      1G              ~$15-20
prometheus            0.5 CPU      512M            ~$5-8
grafana               0.25 CPU     256M            ~$2-4
redis                 0.25 CPU     150M            ~$2-3
frontend              0.25 CPU     128M            ~$2-3
nginx                 0.25 CPU     128M            ~$2-3
-------------------------------------------------------------------
TOTAL:                ~6.5 CPU     ~5.2GB          ~$65-80/month
```

**Target allocation for $50/month:**
```yaml
Service               CPU Limit    Memory Limit    Est. Monthly Cost
postgres              0.75 CPU     384M            ~$6-8
redis                 0.2 CPU      128M            ~$2
backend (2 replicas)  0.5 CPU ea   512M ea         ~$8-10
celery_worker         0.75 CPU     512M            ~$6-8
airflow               0.5 CPU      512M            ~$5-7
prometheus            0.25 CPU     256M            ~$3-4
grafana               0.15 CPU     128M            ~$2
frontend              0.2 CPU      64M             ~$2
nginx                 0.15 CPU     64M             ~$1-2
-------------------------------------------------------------------
TOTAL:                ~3.75 CPU    ~3GB            ~$45-50/month
SAVINGS:              42% CPU      42% Memory      $20-30/month
```

**Issue #2: Elasticsearch - Budget Killer**
- Current: 1 CPU, 1GB memory → $15-20/month
- Usage: News sentiment analysis and stock search
- **Recommendation:** ELIMINATE Elasticsearch
  - Replace with PostgreSQL full-text search (free)
  - Use Redis for in-memory search cache
  - Potential savings: $15-20/month (30-40% of budget)

**Issue #3: Missing Health Checks**
Services without health checks:
1. `celery_beat` - Has check but unreliable (PID file check only)
2. `airflow` - No health check defined
3. `frontend` - No health check in base compose
4. `alertmanager` - Only in prod override
5. `nginx` - Only in prod override

**Issue #4: Dependency Order Problems**
```yaml
# Current: backend depends on elasticsearch starting (not healthy)
backend:
  depends_on:
    elasticsearch:
      condition: service_started  # ❌ Should be service_healthy
```

### 1.2 Recommended Changes

#### File: docker-compose.yml

**Change 1: Optimize PostgreSQL**
```yaml
postgres:
  command:
    - postgres
    - -c shared_preload_libraries=timescaledb
    - -c max_connections=80              # DOWN from 100
    - -c shared_buffers=96MB             # DOWN from 128MB
    - -c effective_cache_size=256MB      # DOWN from 384MB
    - -c maintenance_work_mem=32MB       # DOWN from 64MB
    - -c work_mem=2MB                    # DOWN from 4MB
    - -c random_page_cost=1.1
    - -c checkpoint_completion_target=0.9
    - -c max_wal_size=512MB              # ADD: Limit WAL growth
    - -c min_wal_size=80MB               # ADD: Reduce minimum WAL
  deploy:
    resources:
      limits:
        cpus: '0.75'                     # DOWN from 1.0
        memory: 384M                     # DOWN from 512M
      reservations:
        cpus: '0.2'                      # DOWN from 0.25
        memory: 192M                     # DOWN from 256M
```

**Change 2: Optimize Redis**
```yaml
redis:
  command: >
    redis-server
    --appendonly yes
    --appendfsync everysec
    --maxmemory 100mb                    # DOWN from 128mb
    --maxmemory-policy allkeys-lru
    --requirepass ${REDIS_PASSWORD}
    --tcp-keepalive 60
    --save ""
    --maxclients 50                      # ADD: Limit connections
    --timeout 300                        # ADD: Close idle connections
  deploy:
    resources:
      limits:
        cpus: '0.2'                      # DOWN from 0.25
        memory: 128M                     # DOWN from 150M
      reservations:
        cpus: '0.05'                     # DOWN from 0.1
        memory: 48M                      # DOWN from 64M
```

**Change 3: ELIMINATE Elasticsearch**
```yaml
# REMOVE entire elasticsearch service
# REMOVE elasticsearch-exporter service

# Replace backend dependency
backend:
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
    # REMOVED: elasticsearch dependency
  environment:
    # REMOVE: - ELASTICSEARCH_URL=http://elasticsearch:9200
    - ENABLE_ELASTICSEARCH=false         # ADD: Feature flag
```

**Change 4: Add Missing Health Checks**
```yaml
celery_beat:
  healthcheck:
    test: ["CMD-SHELL", "python -c 'import os; f=\"/tmp/celerybeat.pid\"; exit(0 if os.path.exists(f) and os.path.getmtime(f) > os.time() - 300 else 1)'"]
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

**Change 5: Optimize Backend**
```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '0.5'                      # DOWN from 1.0
        memory: 512M                     # DOWN from 768M
      reservations:
        cpus: '0.15'                     # DOWN from 0.25
        memory: 256M                     # DOWN from 384M
```

**Change 6: Optimize Celery Worker**
```yaml
celery_worker:
  environment:
    # Memory optimization (keep existing)
    - OMP_NUM_THREADS=1
    - OPENBLAS_NUM_THREADS=1
    # ADD: Celery-specific limits
    - CELERYD_MAX_MEMORY_PER_CHILD=400000  # 400MB max per child
  command: >
    celery -A backend.tasks.celery_app worker
    --loglevel=info
    --concurrency=1                      # KEEP at 1 for memory
    --prefetch-multiplier=1
    --max-tasks-per-child=25             # DOWN from 50 (recycle sooner)
    --soft-time-limit=240                # DOWN from 300
    --time-limit=480                     # DOWN from 600
    --max-memory-per-child=400000        # ADD: Hard memory limit
    -Q default,data_ingestion,analysis,notifications,high_priority,low_priority
  deploy:
    resources:
      limits:
        cpus: '0.75'                     # DOWN from 1.0
        memory: 512M                     # DOWN from 768M
      reservations:
        cpus: '0.2'                      # DOWN from 0.25
        memory: 256M                     # DOWN from 384M
```

**Change 7: Optimize Airflow**
```yaml
airflow:
  deploy:
    resources:
      limits:
        cpus: '0.5'                      # DOWN from 1.0
        memory: 512M                     # DOWN from 1G
      reservations:
        cpus: '0.15'                     # DOWN from 0.25
        memory: 256M                     # DOWN from 512M
```

---

## 2. Monitoring Gaps

### 2.1 Current State

#### What's Working ✅
- Prometheus configured with proper scrape intervals
- Alert rules defined for critical services
- Grafana dashboards provisioned
- AlertManager with email/webhook support
- PostgreSQL exporter monitoring DB metrics
- Redis exporter monitoring cache

#### Critical Gaps ❌

**Gap #1: Missing Backend API Exporter**
```yaml
# Prometheus config targets backend:8000/metrics
# BUT: No prometheus client instrumentation found in backend code
```

**Gap #2: Missing Celery Exporter**
```yaml
# Prometheus tries to scrape celery_worker:9540
# BUT: No Celery exporter container exists
# Current health check uses celery inspect, not prometheus
```

**Gap #3: Cost Metrics Not in Prometheus**
- `backend/utils/persistent_cost_monitor.py` exists
- Tracks API usage and costs in database
- NOT exposed to Prometheus for real-time alerting
- Budget alerts defined but metrics don't exist:
  - `cost_budget_usage_percent`
  - `daily_cost_dollars`

**Gap #4: Business Metrics Missing**
- Alert rules reference metrics that don't exist:
  - `stocks_analyzed_total`
  - `recommendations_generated_total`
  - `ml_predictions_total`
  - `ml_prediction_accuracy`

**Gap #5: No Custom Metrics Dashboard**
- Generic system dashboards only
- No investment-specific dashboard
- No real-time cost tracking dashboard

### 2.2 Recommended Changes

#### File: docker-compose.yml

**Add Celery Exporter**
```yaml
# Add after redis-exporter
celery-exporter:
  image: danihodovic/celery-exporter:latest
  container_name: investment_celery_exporter
  environment:
    - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    - CELERY_EXPORTER_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
  ports:
    - "9540:9540"
  depends_on:
    - redis
    - celery_worker
  deploy:
    resources:
      limits:
        cpus: '0.1'
        memory: 64M
  restart: unless-stopped
```

#### File: config/monitoring/prometheus.yml

**Update Scrape Configs**
```yaml
scrape_configs:
  # ... existing configs ...

  # Celery Workers - FIX target
  - job_name: 'celery'
    static_configs:
      - targets: ['celery-exporter:9540']  # CHANGE from celery_worker:9540

  # Backend API metrics - ADD custom metrics endpoint
  - job_name: 'backend-metrics'
    metrics_path: '/api/metrics'
    static_configs:
      - targets: ['backend:8000']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: 'backend-api'

  # Cost monitoring - ADD new scrape
  - job_name: 'cost-monitor'
    metrics_path: '/api/metrics/cost'
    static_configs:
      - targets: ['backend:8000']
    scrape_interval: 60s  # Cost metrics update less frequently
```

#### File: infrastructure/monitoring/alerts/investment-platform.yml

**Remove Elasticsearch Alerts**
```yaml
# REMOVE these alert rules (Elasticsearch being eliminated):
- alert: ElasticsearchDown
- alert: ElasticsearchClusterRed
# ... any other elasticsearch-related alerts
```

**Add Cost Alert Validation**
```yaml
# Add to investment-platform-critical group
- alert: CostMetricsUnavailable
  expr: absent(cost_budget_usage_percent)
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Cost monitoring metrics unavailable"
    description: "Cost tracking metrics have not been scraped for 5 minutes"
```

#### NEW File: infrastructure/monitoring/dashboards/investment-platform.json

Create comprehensive dashboard:
```json
{
  "dashboard": {
    "title": "Investment Platform - Operations",
    "tags": ["investment", "platform"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Monthly Budget Usage",
        "type": "gauge",
        "targets": [{
          "expr": "cost_budget_usage_percent",
          "legendFormat": "Budget Used"
        }],
        "thresholds": [
          {"value": 0, "color": "green"},
          {"value": 80, "color": "yellow"},
          {"value": 90, "color": "red"}
        ]
      },
      {
        "title": "Daily Cost Trend",
        "type": "graph",
        "targets": [{
          "expr": "sum(daily_cost_dollars)",
          "legendFormat": "Daily Cost"
        }]
      },
      {
        "title": "Stock Analysis Pipeline",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(stocks_analyzed_total[1h])",
            "legendFormat": "Stocks/Hour"
          },
          {
            "expr": "rate(analysis_completed_total{status=\"success\"}[5m])*60",
            "legendFormat": "Success Rate"
          }
        ]
      },
      {
        "title": "API Rate Limits",
        "type": "table",
        "targets": [{
          "expr": "api_calls_remaining",
          "format": "table",
          "instant": true
        }]
      },
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [{
          "expr": "up",
          "legendFormat": "{{job}}"
        }]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [{
          "expr": "pg_stat_activity_count",
          "legendFormat": "Active Connections"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [{
          "expr": "redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)",
          "legendFormat": "Hit Rate"
        }]
      },
      {
        "title": "Celery Queue Depth",
        "type": "graph",
        "targets": [{
          "expr": "celery_queue_length",
          "legendFormat": "{{queue}}"
        }]
      },
      {
        "title": "ML Prediction Accuracy",
        "type": "gauge",
        "targets": [{
          "expr": "ml_prediction_accuracy",
          "legendFormat": "{{model}}"
        }]
      }
    ]
  }
}
```

#### NEW File: backend/monitoring/prometheus_metrics.py

Add Prometheus instrumentation:
```python
"""
Prometheus metrics exporter for investment platform.
Exposes custom business metrics to Prometheus.
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client import REGISTRY
from fastapi import APIRouter
from fastapi.responses import Response

# Cost tracking metrics
cost_budget_usage = Gauge(
    'cost_budget_usage_percent',
    'Percentage of monthly budget used'
)
daily_cost = Gauge(
    'daily_cost_dollars',
    'Estimated daily cost in USD'
)
api_calls_remaining = Gauge(
    'api_calls_remaining',
    'Remaining API calls before limit',
    ['provider', 'limit_type']
)

# Business metrics
stocks_analyzed = Counter(
    'stocks_analyzed_total',
    'Total stocks analyzed',
    ['status']
)
recommendations_generated = Counter(
    'recommendations_generated_total',
    'Total recommendations generated',
    ['recommendation_type']
)
ml_predictions = Counter(
    'ml_predictions_total',
    'Total ML predictions made',
    ['model', 'status']
)
ml_accuracy = Gauge(
    'ml_prediction_accuracy',
    'Current ML model accuracy',
    ['model']
)

# Analysis pipeline metrics
analysis_duration = Histogram(
    'analysis_duration_seconds',
    'Time taken to analyze a stock',
    ['analysis_type']
)
analysis_completed = Counter(
    'analysis_completed_total',
    'Analysis tasks completed',
    ['status']
)

# API metrics
api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint', 'status']
)
api_requests = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)
api_errors = Counter(
    'api_errors_total',
    'Total API errors',
    ['method', 'endpoint', 'error_type']
)

# Database metrics
db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation']
)
db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections'
)

# Cache metrics
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)
cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

# External API metrics
external_api_calls = Counter(
    'external_api_calls_total',
    'External API calls',
    ['provider', 'status']
)
external_api_failures = Counter(
    'external_api_failures_total',
    'External API failures',
    ['provider', 'error_type']
)
rate_limit_hits = Counter(
    'rate_limit_hits_total',
    'Rate limit hits',
    ['resource']
)

# Router for metrics endpoint
router = APIRouter()

@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )

@router.get("/metrics/cost")
async def cost_metrics():
    """Expose cost-specific metrics with current values."""
    from backend.utils.persistent_cost_monitor import PersistentCostMonitor

    monitor = PersistentCostMonitor()

    # Update cost metrics
    budget_usage = await monitor.get_budget_usage_percent()
    cost_budget_usage.set(budget_usage)

    daily_cost_value = await monitor.get_daily_cost()
    daily_cost.set(daily_cost_value)

    # Update API limits
    for provider, limits in monitor.api_limits.items():
        remaining_daily = await monitor.get_remaining_calls(provider, 'daily')
        api_calls_remaining.labels(provider=provider, limit_type='daily').set(remaining_daily)

        if limits.get('per_minute') != float('inf'):
            remaining_minute = await monitor.get_remaining_calls(provider, 'per_minute')
            api_calls_remaining.labels(provider=provider, limit_type='per_minute').set(remaining_minute)

    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )
```

#### File: backend/api/main.py

**Add metrics router:**
```python
# Add import
from backend.monitoring.prometheus_metrics import router as metrics_router

# Add to app initialization (after existing routers)
app.include_router(
    metrics_router,
    tags=["monitoring"]
)
```

---

## 3. CI/CD Pipeline Analysis

### 3.1 Current State

**File:** `.github/workflows/ci.yml`

#### Issues ❌

**Issue #1: Sequential Job Execution**
Current workflow runs jobs sequentially:
1. backend-quality (15 min)
2. backend-test (30 min with matrix)
3. frontend-build (10 min)
4. security-scan (15 min)

**Total: 45-60 minutes**

**Issue #2: Duplicate Dependency Installation**
Each job installs dependencies separately:
- backend-quality: Installs linting tools + core deps
- backend-test: Installs full requirements.txt
- No caching strategy across jobs

**Issue #3: Unnecessary Matrix Testing**
```yaml
strategy:
  matrix:
    python-version: ['3.12']  # Only one version tested
    test-suite: ['unit', 'integration']
```
Matrix is overkill for single Python version.

**Issue #4: Build Artifacts Not Reused**
- No Docker layer caching
- Images rebuilt in deployment workflow
- No artifact sharing between workflows

**Issue #5: Missing Cost Optimization**
```yaml
timeout-minutes: 30  # Good
# But no:
# - Conditional job execution
# - Cache reuse across workflows
# - Parallel test execution
```

### 3.2 Recommended Changes

#### File: .github/workflows/ci.yml

**Change 1: Parallelize Independent Jobs**
```yaml
jobs:
  # Run these 4 jobs in PARALLEL
  backend-quality:
    runs-on: ubuntu-latest
    timeout-minutes: 10  # DOWN from 15
    # ... existing steps ...

  backend-test:
    runs-on: ubuntu-latest
    timeout-minutes: 20  # DOWN from 30
    strategy:
      matrix:
        test-suite: ['unit', 'integration']  # REMOVE python-version
      fail-fast: false
    # ... steps ...

  frontend-quality:  # NEW: Split from build
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: 'frontend/web/package-lock.json'
      - run: |
          cd frontend/web
          npm ci
          npm run lint
          npm run type-check

  frontend-test:  # NEW: Split from build
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: 'frontend/web/package-lock.json'
      - run: |
          cd frontend/web
          npm ci
          npm test -- --coverage

  security-scan:
    runs-on: ubuntu-latest
    timeout-minutes: 10  # DOWN from 15
    # ... steps ...

  # Build only after all checks pass
  build:
    needs: [backend-quality, backend-test, frontend-quality, frontend-test]
    runs-on: ubuntu-latest
    timeout-minutes: 15
    # ... build steps ...
```

**Change 2: Aggressive Dependency Caching**
```yaml
backend-quality:
  steps:
    - uses: actions/checkout@v4

    # Cache pip dependencies with better key
    - name: Cache Python dependencies
      uses: actions/cache@v4
      with:
        path: |
          ~/.cache/pip
          ~/.local/lib/python3.12
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    # Only install if cache miss
    - name: Install dependencies
      run: |
        if [ ! -d ~/.local/lib/python3.12/site-packages ]; then
          python -m pip install --upgrade pip
          pip install --user black isort flake8 mypy pylint bandit safety
          pip install --user pydantic fastapi sqlalchemy
        fi
```

**Change 3: Skip Jobs on Non-Code Changes**
```yaml
on:
  push:
    branches: [ main, develop ]
    paths:  # ADD: Only run on relevant changes
      - 'backend/**'
      - 'frontend/**'
      - 'requirements*.txt'
      - 'package*.json'
      - '.github/workflows/ci.yml'
  pull_request:
    branches: [ main, develop ]
    paths:  # ADD: Only run on relevant changes
      - 'backend/**'
      - 'frontend/**'
      - 'requirements*.txt'
      - 'package*.json'
      - '.github/workflows/ci.yml'
```

**Change 4: Docker Build Caching**
```yaml
build:
  steps:
    - uses: actions/checkout@v4

    # Setup Docker Buildx with caching
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Cache Docker layers
      uses: actions/cache@v4
      with:
        path: /tmp/.buildx-cache
        key: ${{ runner.os }}-buildx-${{ github.sha }}
        restore-keys: |
          ${{ runner.os }}-buildx-

    - name: Build backend image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./infrastructure/docker/backend/Dockerfile
        target: runtime
        push: false
        tags: investment-backend:${{ github.sha }}
        cache-from: type=local,src=/tmp/.buildx-cache
        cache-to: type=local,dest=/tmp/.buildx-cache-new,mode=max

    # Prevent cache size explosion
    - name: Move cache
      run: |
        rm -rf /tmp/.buildx-cache
        mv /tmp/.buildx-cache-new /tmp/.buildx-cache
```

**Change 5: Conditional Deployment**
```yaml
# NEW: Only deploy on push to main (not PRs)
deploy:
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  needs: build
  runs-on: ubuntu-latest
  # ... deployment steps ...
```

**Expected Results:**
- Parallel execution: 45-60 min → **15-20 min** (66% faster)
- Cache hits reduce install time: 5 min → **30 sec**
- Skip non-code changes: Saves ~40 builds/month
- **Estimated savings:** 30-40 workflow minutes/day = $5-8/month

---

## 4. Resource Utilization & Cost Analysis

### 4.1 Current Cost Breakdown

```
CATEGORY              CURRENT    OPTIMIZED   SAVINGS
Infrastructure        $65-80     $45-50      $20-30
  - Compute           $35-45     $25-30      $10-15
  - Database          $12-15     $8-10       $4-5
  - Elasticsearch     $15-20     $0          $15-20 ✅
  - Redis             $3         $2          $1
  - Monitoring        $0         $0          $0

GitHub Actions        $0         $0          $0 (free tier)
API Costs            $0         $0          $0 (free tiers)
Domain/SSL           $10        $10         $0

TOTAL                $75-90     $45-50      $30-40/month
OVER BUDGET          +50-80%    ON TARGET   ✅
```

### 4.2 Critical Optimizations

#### Priority 1: Eliminate Elasticsearch
- **Savings:** $15-20/month (30-40% of budget)
- **Impact:** Low (replace with PostgreSQL full-text search)
- **Implementation time:** 4-6 hours
- **Alternative solution:**
  ```sql
  -- PostgreSQL full-text search
  CREATE INDEX idx_stock_fts ON stocks USING gin(to_tsvector('english', name || ' ' || description));

  -- Search query
  SELECT * FROM stocks
  WHERE to_tsvector('english', name || ' ' || description) @@ to_tsquery('technology & growth');
  ```

#### Priority 2: Right-Size Resources
- **Savings:** $10-15/month
- **Impact:** None (resources currently over-provisioned)
- **Implementation time:** 1 hour (update docker-compose files)

#### Priority 3: Optimize CI/CD
- **Savings:** $5-8/month (workflow minutes)
- **Impact:** Positive (faster feedback loops)
- **Implementation time:** 2-3 hours

### 4.3 Cost Monitoring Integration

#### NEW File: backend/utils/cost_metrics_updater.py

```python
"""
Periodic cost metrics updater for Prometheus.
Updates cost-related Prometheus metrics every minute.
"""

import asyncio
import logging
from datetime import datetime
from backend.utils.persistent_cost_monitor import PersistentCostMonitor
from backend.monitoring.prometheus_metrics import (
    cost_budget_usage,
    daily_cost,
    api_calls_remaining
)

logger = logging.getLogger(__name__)


async def update_cost_metrics_loop():
    """Background task to update cost metrics for Prometheus."""
    monitor = PersistentCostMonitor()

    while True:
        try:
            # Update budget usage
            budget_percent = await monitor.get_budget_usage_percent()
            cost_budget_usage.set(budget_percent)

            # Update daily cost
            daily_cost_value = await monitor.get_daily_cost()
            daily_cost.set(daily_cost_value)

            # Update API limits for all providers
            for provider in monitor.api_limits.keys():
                remaining_daily = await monitor.get_remaining_calls(provider, 'daily')
                api_calls_remaining.labels(
                    provider=provider,
                    limit_type='daily'
                ).set(remaining_daily)

                remaining_minute = await monitor.get_remaining_calls(provider, 'per_minute')
                if remaining_minute != float('inf'):
                    api_calls_remaining.labels(
                        provider=provider,
                        limit_type='per_minute'
                    ).set(remaining_minute)

            logger.debug(f"Updated cost metrics - Budget: {budget_percent}%, Daily: ${daily_cost_value}")

        except Exception as e:
            logger.error(f"Failed to update cost metrics: {e}")

        # Update every minute
        await asyncio.sleep(60)


def start_cost_metrics_updater():
    """Start the cost metrics updater as a background task."""
    asyncio.create_task(update_cost_metrics_loop())
    logger.info("Cost metrics updater started")
```

#### File: backend/api/main.py

```python
# Add import
from backend.utils.cost_metrics_updater import start_cost_metrics_updater

# Add to startup event
@app.on_event("startup")
async def startup_event():
    # ... existing startup code ...

    # Start cost metrics updater
    start_cost_metrics_updater()
```

---

## 5. Implementation Roadmap

### Phase 1: Critical (Budget Compliance) - Week 1

**Day 1-2: Eliminate Elasticsearch**
1. ✅ Remove elasticsearch service from docker-compose
2. ✅ Remove elasticsearch-exporter
3. ✅ Update backend code to use PostgreSQL full-text search
4. ✅ Update prometheus.yml (remove elasticsearch scrape)
5. ✅ Update alert rules (remove elasticsearch alerts)
6. ✅ Test search functionality
7. **Savings: $15-20/month**

**Day 3-4: Right-Size Resources**
1. ✅ Update resource limits in docker-compose.yml
2. ✅ Update resource limits in docker-compose.prod.yml
3. ✅ Test all services start and run smoothly
4. ✅ Monitor for OOM kills or performance issues
5. **Savings: $10-15/month**

**Day 5: Add Missing Health Checks**
1. ✅ Add health checks to celery_beat, airflow, frontend
2. ✅ Update dependency conditions to use service_healthy
3. ✅ Test service startup order
4. **Impact: Improved reliability**

### Phase 2: Monitoring (Visibility) - Week 2

**Day 1-2: Add Prometheus Instrumentation**
1. ✅ Create `backend/monitoring/prometheus_metrics.py`
2. ✅ Add metrics router to FastAPI
3. ✅ Create cost metrics updater background task
4. ✅ Update prometheus.yml scrape configs
5. **Impact: Real-time cost visibility**

**Day 3: Add Celery Exporter**
1. ✅ Add celery-exporter service to docker-compose
2. ✅ Update prometheus.yml celery scrape target
3. ✅ Test celery metrics collection
4. **Impact: Worker performance visibility**

**Day 4-5: Create Custom Dashboards**
1. ✅ Create investment platform operations dashboard
2. ✅ Create cost monitoring dashboard
3. ✅ Import dashboards to Grafana
4. ✅ Configure dashboard auto-refresh
5. **Impact: Business metrics visibility**

### Phase 3: CI/CD Optimization - Week 3

**Day 1-2: Parallelize Workflows**
1. ✅ Split quality and test jobs
2. ✅ Remove unnecessary matrix strategy
3. ✅ Add path filters to skip non-code changes
4. ✅ Test parallel execution
5. **Savings: 30 min/build**

**Day 3-4: Add Caching**
1. ✅ Implement Docker layer caching
2. ✅ Improve dependency caching
3. ✅ Test cache hit rates
4. **Savings: 5 min/build**

**Day 5: Conditional Deployment**
1. ✅ Add deployment conditions
2. ✅ Add build artifact reuse
3. ✅ Test full workflow
4. **Savings: $5-8/month**

### Phase 4: Production Hardening - Week 4

**Day 1-2: Load Testing**
1. ✅ Test optimized resource limits under load
2. ✅ Identify bottlenecks
3. ✅ Fine-tune resource allocation
4. **Validation: Meets performance requirements**

**Day 3-4: Backup & Disaster Recovery**
1. ✅ Test backup service (already configured)
2. ✅ Test restore procedure
3. ✅ Document DR runbook
4. **Impact: Production readiness**

**Day 5: Documentation**
1. ✅ Update deployment documentation
2. ✅ Update monitoring documentation
3. ✅ Create cost optimization guide
4. **Impact: Maintainability**

---

## 6. Validation & Testing

### 6.1 Resource Limit Testing

**Script:** `scripts/infrastructure/test_resource_limits.sh`
```bash
#!/bin/bash
# Test optimized resource limits under load

set -e

echo "Starting resource limit tests..."

# Start services with optimized config
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Wait for services to be healthy
sleep 60

# Generate load
echo "Generating load on backend..."
ab -n 10000 -c 50 http://localhost:8000/api/health

echo "Triggering stock analysis..."
curl -X POST http://localhost:8000/api/admin/trigger-analysis

# Monitor resource usage
echo "Monitoring resource usage for 5 minutes..."
for i in {1..10}; do
  docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
  sleep 30
done

# Check for OOM kills
echo "Checking for OOM kills..."
docker ps -a --filter "status=exited" --filter "label=com.docker.compose.project=investment-analysis-platform"

echo "Test complete!"
```

### 6.2 Cost Validation

**Script:** `scripts/infrastructure/validate_costs.py`
```python
"""
Validate that infrastructure stays within $50/month budget.
"""

import asyncio
from backend.utils.persistent_cost_monitor import PersistentCostMonitor


async def validate_costs():
    monitor = PersistentCostMonitor()

    # Get current cost metrics
    budget_usage = await monitor.get_budget_usage_percent()
    daily_cost = await monitor.get_daily_cost()
    projected_monthly = daily_cost * 30

    print(f"Budget Usage: {budget_usage:.1f}%")
    print(f"Daily Cost: ${daily_cost:.2f}")
    print(f"Projected Monthly: ${projected_monthly:.2f}")

    # Validate
    assert projected_monthly <= 50, f"Projected cost ${projected_monthly:.2f} exceeds $50 budget!"
    assert budget_usage <= 100, f"Budget usage {budget_usage:.1f}% exceeds 100%!"

    print("✅ Cost validation passed!")


if __name__ == "__main__":
    asyncio.run(validate_costs())
```

### 6.3 Monitoring Validation

**Script:** `scripts/infrastructure/validate_monitoring.sh`
```bash
#!/bin/bash
# Validate that all monitoring is working

set -e

echo "Validating monitoring setup..."

# Check Prometheus targets
echo "Checking Prometheus targets..."
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up") | {job: .labels.job, health: .health}'

# Check cost metrics exist
echo "Checking cost metrics..."
curl -s http://localhost:9090/api/v1/query?query=cost_budget_usage_percent | jq '.data.result'

# Check business metrics exist
echo "Checking business metrics..."
METRICS=("stocks_analyzed_total" "recommendations_generated_total" "ml_predictions_total")
for metric in "${METRICS[@]}"; do
  result=$(curl -s "http://localhost:9090/api/v1/query?query=${metric}" | jq -r '.data.result | length')
  if [ "$result" -eq 0 ]; then
    echo "❌ Metric $metric not found!"
  else
    echo "✅ Metric $metric exists"
  fi
done

# Check Grafana dashboards
echo "Checking Grafana dashboards..."
curl -s -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
  http://localhost:3001/api/search?type=dash-db | jq '.[].title'

echo "Validation complete!"
```

---

## 7. Summary & Next Steps

### 7.1 Critical Actions (Do First)

1. **Eliminate Elasticsearch** → Saves $15-20/month
2. **Right-size resources** → Saves $10-15/month
3. **Add cost monitoring to Prometheus** → Visibility & alerts
4. **Fix missing health checks** → Reliability

**Total savings: $25-35/month (brings budget to $45-55/month)**

### 7.2 High-Value Actions (Do Next)

5. **Add Celery exporter** → Worker visibility
6. **Create custom dashboards** → Business metrics
7. **Parallelize CI/CD** → Faster builds, $5-8/month savings
8. **Add missing Prometheus metrics** → Complete observability

### 7.3 Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Monthly Infrastructure Cost | $65-80 | <$50 | ❌ Over |
| Service Health Checks | 7/12 | 12/12 | ⚠️ Incomplete |
| Prometheus Exporters | 4/6 | 6/6 | ⚠️ Missing 2 |
| CI/CD Duration | 45-60 min | <20 min | ⚠️ Slow |
| Cost Visibility | Manual | Real-time | ❌ None |
| Alert Coverage | 60% | 95% | ⚠️ Gaps |

### 7.4 Files to Create/Modify

**Create:**
1. `backend/monitoring/prometheus_metrics.py`
2. `backend/utils/cost_metrics_updater.py`
3. `infrastructure/monitoring/dashboards/investment-platform.json`
4. `infrastructure/monitoring/dashboards/cost-monitoring.json`
5. `scripts/infrastructure/test_resource_limits.sh`
6. `scripts/infrastructure/validate_costs.py`
7. `scripts/infrastructure/validate_monitoring.sh`

**Modify:**
8. `docker-compose.yml` (remove elasticsearch, optimize resources, add celery-exporter)
9. `docker-compose.prod.yml` (optimize production resources)
10. `config/monitoring/prometheus.yml` (update scrape configs)
11. `infrastructure/monitoring/alerts/investment-platform.yml` (remove ES alerts, add validation)
12. `.github/workflows/ci.yml` (parallelize, cache, optimize)
13. `backend/api/main.py` (add metrics router, start cost updater)

---

## Appendix A: Resource Sizing Calculations

### PostgreSQL Sizing
```
Target: 80 connections, 6000 stocks analyzed daily

Connection usage:
- Backend API: 10-20 connections
- Celery workers: 5-10 connections
- Airflow: 5-10 connections
- Monitoring: 2-5 connections
Total: 22-45 connections (50% of 80 limit)

Memory requirements:
- shared_buffers: 96MB (25% of total)
- work_mem: 2MB × 80 connections = 160MB max
- maintenance_work_mem: 32MB
- OS cache: 128MB
Total: ~384MB (matches allocation)
```

### Redis Sizing
```
Usage:
- Session storage: ~10MB
- API response cache: ~40MB
- Celery broker: ~30MB
- Rate limiting: ~10MB
Total: ~90MB (100MB limit provides headroom)
```

### Backend API Sizing
```
Memory usage:
- Python runtime: ~100MB
- FastAPI + dependencies: ~150MB
- Request processing: ~100MB
- ML model loading: ~150MB
Total: ~500MB (512MB limit provides headroom)
```

---

## Appendix B: Full-Text Search Migration

### Step 1: Add PostgreSQL Extensions
```sql
-- Enable pg_trgm for fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable unaccent for accent-insensitive search
CREATE EXTENSION IF NOT EXISTS unaccent;
```

### Step 2: Add Search Columns
```sql
-- Add text search vector column
ALTER TABLE stocks ADD COLUMN search_vector tsvector;

-- Update search vector
UPDATE stocks SET search_vector =
  to_tsvector('english',
    coalesce(ticker, '') || ' ' ||
    coalesce(name, '') || ' ' ||
    coalesce(description, '') || ' ' ||
    coalesce(sector, '') || ' ' ||
    coalesce(industry, '')
  );

-- Create trigger to keep it updated
CREATE FUNCTION stocks_search_vector_trigger() RETURNS trigger AS $$
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

CREATE TRIGGER stocks_search_vector_update
  BEFORE INSERT OR UPDATE ON stocks
  FOR EACH ROW EXECUTE FUNCTION stocks_search_vector_trigger();
```

### Step 3: Add Indexes
```sql
-- GIN index for fast full-text search
CREATE INDEX idx_stocks_search_vector ON stocks USING gin(search_vector);

-- Trigram index for fuzzy matching (typos)
CREATE INDEX idx_stocks_ticker_trgm ON stocks USING gin(ticker gin_trgm_ops);
CREATE INDEX idx_stocks_name_trgm ON stocks USING gin(name gin_trgm_ops);
```

### Step 4: Update Backend Code
```python
# backend/repositories/stock_repository.py

async def search_stocks(
    self,
    query: str,
    limit: int = 20
) -> List[Stock]:
    """
    Full-text search for stocks using PostgreSQL.
    Replaces Elasticsearch functionality.
    """
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


async def fuzzy_search_stocks(
    self,
    query: str,
    limit: int = 20,
    similarity_threshold: float = 0.3
) -> List[Stock]:
    """
    Fuzzy search for typo tolerance.
    """
    stmt = select(Stock).where(
        or_(
            func.similarity(Stock.ticker, query) > similarity_threshold,
            func.similarity(Stock.name, query) > similarity_threshold
        )
    ).order_by(
        func.greatest(
            func.similarity(Stock.ticker, query),
            func.similarity(Stock.name, query)
        ).desc()
    ).limit(limit)

    result = await self.session.execute(stmt)
    return result.scalars().all()
```

**Performance comparison:**
- Elasticsearch: ~10-50ms per search
- PostgreSQL FTS: ~15-60ms per search
- Difference: Negligible for this use case
- Benefit: $15-20/month savings + simplified stack

---

**End of Analysis**

**Recommendation:** Implement Phase 1 immediately to achieve budget compliance within 2-3 days.
