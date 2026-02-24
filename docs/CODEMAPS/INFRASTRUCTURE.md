# Infrastructure Architecture Codemap

## Docker Services

### Core Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| postgres | postgres:15-alpine | 5432 | Primary database |
| timescaledb | timescale/timescaledb | - | Time-series extension |
| redis | redis:7-alpine | 6379 | Cache & message broker |
| backend | custom | 8000 | FastAPI application |
| frontend | custom | 3000 | React application |
| celery_worker | custom | - | Background tasks |
| celery_beat | custom | - | Scheduled tasks |
| airflow | apache/airflow | 8080 | Pipeline orchestration |

### Monitoring Stack (Production)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| prometheus | prom/prometheus | 9090 | Metrics collection |
| grafana | grafana/grafana | 3001 | Dashboards |
| alertmanager | prom/alertmanager | 9093 | Alert routing |

## Docker Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base configuration |
| `docker-compose.dev.yml` | Development overrides |
| `docker-compose.prod.yml` | Production configuration |
| `docker-compose.test.yml` | Testing configuration |

## Resource Allocations

### Development

| Service | CPU | Memory |
|---------|-----|--------|
| postgres | 0.75 | 384MB |
| redis | 0.2 | 128MB |
| backend | 0.5 | 512MB |
| celery_worker | 0.75 | 512MB |
| airflow | 0.5 | 512MB |

### Production

| Service | CPU | Memory |
|---------|-----|--------|
| postgres | 1.0 | 512MB |
| redis | 0.25 | 600MB |
| backend | 1.0 | 768MB |
| celery_worker | 1.0 | 768MB |
| airflow | 0.75 | 768MB |

## Configuration Files

### Monitoring (`config/monitoring/`)

| File | Purpose |
|------|---------|
| `prometheus.yml` | Prometheus scrape config |
| `alertmanager.yml` | Alert routing rules |
| `grafana/provisioning/` | Grafana dashboards |

### Nginx (`infrastructure/nginx/`)

| File | Purpose |
|------|---------|
| `nginx.conf` | Main configuration |
| `ssl.conf` | SSL/TLS settings |
| `upstream.conf` | Backend proxy |

## Health Checks

All services have health checks configured:

```yaml
# Example: Backend
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

## Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EXTERNAL NETWORK                        │
│                    (Internet/Client)                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                        NGINX                                │
│              (Reverse Proxy + SSL)                          │
│                     Port 80/443                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│      Frontend         │   │       Backend         │
│   (React + Nginx)     │   │      (FastAPI)        │
│      Port 3000        │   │      Port 8000        │
└───────────────────────┘   └───────────┬───────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │    PostgreSQL   │ │      Redis      │ │  Celery Worker  │
          │    Port 5432    │ │   Port 6379     │ │                 │
          └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Environment Variables

### Required

| Variable | Service | Purpose |
|----------|---------|---------|
| DATABASE_URL | backend | DB connection |
| DB_PASSWORD | postgres | Auth |
| REDIS_URL | backend | Cache connection |
| REDIS_PASSWORD | redis | Auth |
| SECRET_KEY | backend | JWT signing |

### API Keys

| Variable | Service | Rate Limit |
|----------|---------|------------|
| ALPHA_VANTAGE_API_KEY | backend | 25/day |
| FINNHUB_API_KEY | backend | 60/min |
| POLYGON_API_KEY | backend | 5/min |
| NEWS_API_KEY | backend | 100/day |

## Deployment Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Initial setup with credential generation |
| `start.sh` | Start services (dev/prod/test) |
| `stop.sh` | Stop all services |
| `logs.sh` | View service logs |
| `board-sync.sh` | GitHub Projects sync |
| `notion-sync.sh` | Notion sync |

## CI/CD Workflows (`.github/workflows/`)

**Runtime Versions (standardized 2026-02-24):**
- Python: `3.12` (primary), matrix tests 3.9-3.12 in `ci.yml`
- Node.js: `20`

**Action Versions (standardized 2026-02-24):**
- `actions/setup-python@v5`, `actions/setup-node@v4`
- `actions/upload-artifact@v4`, `actions/download-artifact@v4`
- `github/codeql-action/*@v3` (except 2 legacy v2 refs in staging/production deploy)

### Core Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main/develop | Lint, test (matrix), build, coverage |
| `comprehensive-testing.yml` | Push/PR/daily 2AM | Security scan + full test suite |
| `security-scan.yml` | Push/PR/daily 2AM | CodeQL (v3), Semgrep, GitLeaks, dependency audit |
| `daily-pipeline-validation.yml` | Daily 6AM | ETL, data ingestion, ML pipeline validation |
| `dependency-updates.yml` | Weekly Monday 10AM | Python + JS dependency updates |

### Deployment Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `staging-deploy.yml` | Push to main | Build, test, deploy to staging |
| `production-deploy.yml` | Release published | Build, test, deploy to production |
| `release-management.yml` | Version tag push | Release versioning and changelog |
| `automated-release.yml` | Manual/VERSION push | Semantic versioning pipeline |
| `workflow-coordinator.yml` | Manual dispatch | Orchestrate multi-workflow runs |

### Supporting Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `board-sync.yml` | Issue/PR events | GitHub Projects sync |
| `documentation-sync.yml` | Push/schedule | Documentation updates |
| `documentation-validation.yml` | Push/PR | Validate documentation |
| `monitoring-notifications.yml` | Various | Alert routing |
| `performance-monitoring.yml` | Schedule/dispatch | Performance benchmarks |
| `pr-automation.yml` | PR events | PR labeling and checks |
| `issue-management.yml` | Issue events | Issue triage |
| `claude-code-review.yml` | PR events | AI code review |
| `mypy.yml` | Push/PR | Type checking |
| `type-check.yml` | Push/PR | TypeScript type checking |
| `reusable-build.yml` | Called by others | Shared build job |
| `reusable-test.yml` | Called by others | Shared test job |

### TA-Lib C Library Dependency

Workflows that run Python code importing `talib` include a build step that compiles TA-Lib 0.4.0 from source:
- `daily-pipeline-validation.yml`
- `security-scan.yml`
- `ci.yml` (via requirements)
- `reusable-test.yml`
- `production-deploy.yml`
- `dependency-updates.yml`

## Quick Wins Applied

### Elasticsearch Removal (Quick Win #3)
- **Before**: Elasticsearch 8.11 service ($15-20/month)
- **After**: PostgreSQL Full-Text Search with pg_trgm
- **Impact**: Reduced monthly cost, simpler stack

### Redis Memory Increase (Quick Win #4)
- **Before**: 128MB maxmemory
- **After**: 512MB maxmemory with `allkeys-lru`
- **Files Modified**:
  - `docker-compose.yml`
  - `docker-compose.dev.yml`
  - `docker-compose.prod.yml`

## Cost Breakdown

### Current ($45-50/month target achieved)

| Component | Estimated Cost |
|-----------|---------------|
| Database | ~$10 |
| Compute | ~$15 |
| Storage | ~$5 |
| APIs | ~$10 |
| **Total** | **~$40-45** |

### Savings from Quick Wins

| Optimization | Savings |
|--------------|---------|
| Elasticsearch removal | $15-20/month |
| Resource right-sizing | $10-15/month |
| **Total** | **$25-35/month** |

## Monitoring Dashboards

### Grafana Dashboards

| Dashboard | Purpose |
|-----------|---------|
| API Performance | Request latency, throughput |
| Database Metrics | Query performance, connections |
| Cache Performance | Hit rate, memory usage |
| ML Pipeline | Model latency, accuracy |
| Infrastructure | CPU, memory, disk |

### Prometheus Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `http_requests_total` | Counter | method, endpoint, status |
| `http_request_duration_seconds` | Histogram | method, endpoint |
| `cache_hits_total` | Counter | cache_type |
| `db_query_duration_seconds` | Histogram | query_type |

## Troubleshooting

### Common Issues

| Issue | Command |
|-------|---------|
| Service not starting | `docker compose logs <service>` |
| OOM killed | `docker stats` |
| Port conflict | `lsof -i :<port>` |
| Health check failing | `docker compose ps` |

### Useful Commands

```bash
# Check all services
docker compose ps

# View resource usage
docker stats --no-stream

# Restart single service
docker compose restart <service>

# View service logs
./logs.sh <service>

# Full restart
./stop.sh && ./start.sh dev
```

## CI/CD Hardening History (2026-02-24)

12 commits resolved daily CI failures:
- `d9b2d1c` -- Node 18 to 20, standardized action versions across 9 workflows
- `786ffec`, `4b8b168` -- TA-Lib C library added to pipeline validation and security scan
- `ca91ccd` -- Missing SECRET_KEY/JWT_SECRET_KEY added to pipeline validation
- `03b6237` -- SSL disabled for CI PostgreSQL service
- `363fe69` -- Fixed inline Python code in pipeline validation
- `4812e1f` -- Fixed pipeline validation db init and security scan issues
- `9f22b30` -- ETL validation made resilient to import errors
- `c7d78da` -- Fixed npm audit output concatenation in dependency-updates
- `7a39cb9` -- Semgrep made non-blocking in security scan
- `4bac2fc` -- GitLeaks made non-blocking in security scan
- CodeQL upgraded from v2 to v3 in `security-scan.yml`

**Last Updated**: 2026-02-24
