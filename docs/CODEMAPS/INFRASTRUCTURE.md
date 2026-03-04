# Infrastructure Architecture Codemap

**Last Updated:** 2026-03-04

## Docker Services

### Core Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| postgres | timescale/timescaledb:2.12.1-pg15 | 5432 | Primary database |
| redis | redis:7.2-alpine | 6379 | Cache & message broker |
| backend | custom (multi-stage, non-root) | 8000 | FastAPI application |
| frontend | custom (Node Alpine, multi-stage) | 3000 | React application |
| celery_worker | custom | - | Background tasks (5 queues) |
| celery_beat | custom | - | Scheduled tasks (beat scheduler) |
| nginx | nginx:alpine | 80/443 | Reverse proxy + SSL |

### Monitoring Stack (Production)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| prometheus | prom/prometheus | 9090 | Metrics collection |
| grafana | grafana/grafana | 3001 | Dashboards |
| alertmanager | prom/alertmanager | 9093 | Alert routing |
| loki | grafana/loki:2.9.3 | 3100 | Log aggregation |
| promtail | grafana/promtail:2.9.3 | - | Log shipping to Loki |
| certbot | certbot/certbot:v2.7.4 | - | SSL certificate auto-renewal |

### Additional Services (Production Only)

| Service | Purpose |
|---------|---------|
| cost_monitor | Budget monitoring (alert at 90% of $50/month) |
| node-exporter | Host metrics for Prometheus |
| cadvisor | Container metrics |
| airflow | Pipeline orchestration (Airflow DAGs) |

## Docker Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base configuration (17 services) |
| `docker-compose.dev.yml` | Development overrides |
| `docker-compose.production.yml` | Full production stack (canonical) — includes Loki, Promtail, certbot |
| `docker-compose.test.yml` | Testing configuration |
| `docker-compose.ml-production.yml` | ML-specific production extensions |

## Resource Allocations

### Production (`docker-compose.production.yml`)

| Service | CPU | Memory |
|---------|-----|--------|
| postgres | 1.0 | 1.5 GB |
| redis | 0.25 | 640 MB |
| backend | 0.75 | 512 MB |
| celery_worker | 2.0 | 1 GB |
| celery_beat | - | No limits (gap) |
| nginx | 0.25 | 128 MB |
| prometheus | 0.25 | 256 MB |
| grafana | 0.2 | 192 MB |

## Configuration Files

### Monitoring (`infrastructure/monitoring/`)

| File | Purpose |
|------|---------|
| `prometheus.yml` | Prometheus scrape config (10+ targets) |
| `prometheus.prod.yml` | Production Prometheus config |
| `alertmanager.yml` | Alert routing rules |
| `alerts/slo-targets.yml` | SLO target definitions |
| `grafana/provisioning/` | Grafana dashboards |
| `loki/loki-config.yaml` | Loki log aggregation config |
| `loki/promtail-config.yaml` | Promtail log shipping config |

### Nginx (`infrastructure/nginx/`)

| File | Purpose |
|------|---------|
| `nginx.conf` | Main configuration |
| `nginx-ssl.conf` | SSL/TLS settings (references ssl/ directory) |
| `upstream.conf` | Backend proxy |

**Note:** `ssl/` directory must contain `fullchain.pem`, `privkey.pem`, `dhparam.pem`,
`chain.pem` before nginx starts in production. Use certbot container for auto-provisioning.

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
│              (Reverse Proxy + SSL via certbot)              │
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
                    │
          ┌─────────────────┐ ┌─────────────────┐
          │   Prometheus    │ │      Loki       │
          │    Port 9090    │ │   Port 3100     │
          └─────────────────┘ └─────────────────┘
```

## Environment Variables

### Required

| Variable | Service | Purpose |
|----------|---------|---------|
| DATABASE_URL | backend | DB connection |
| DB_PASSWORD | postgres | Auth |
| REDIS_URL | backend | Cache connection |
| REDIS_PASSWORD | redis | Auth |
| SECRET_KEY | backend | JWT signing seed |
| JWT_SECRET_KEY | backend | JWT secret |
| GDPR_ENCRYPTION_KEY | backend | Fernet key for field-level encryption |

### API Keys

| Variable | Service | Rate Limit |
|----------|---------|------------|
| ALPHA_VANTAGE_API_KEY | backend | 25/day |
| FINNHUB_API_KEY | backend | 60/min |
| POLYGON_API_KEY | backend | 5/min |
| NEWS_API_KEY | backend | 100/day |

## Deployment Scripts (`scripts/deployment/`)

| Script | Quality | Notes |
|--------|---------|-------|
| `blue_green_deploy.sh` | HIGH | Error handling, dry-run, auto-rollback |
| `rollback.sh` | HIGH | Version-targeted, confirmation, JSON reports |
| `generate_secrets.sh` | GOOD | Python cryptography module |
| `validate-env.sh` | GOOD | Environment validation |
| `backup.sh` | MODERATE | S3 support, integrity verification |
| `restore-backup.sh` | MODERATE | S3 restore |

## CI/CD Workflows (`.github/workflows/`)

**Runtime Versions:** Python 3.12 (primary, matrix: 3.10/3.11/3.12), Node.js 20
**Action Versions:** `setup-python@v5`, `setup-node@v4`, `upload-artifact@v4`, `codeql@v3`

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
| `production-deploy.yml` | Release published | Build, test, deploy to production (blue-green) |
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

Workflows that run Python code importing `talib` include a build step that
compiles TA-Lib 0.4.0 from source:
- `daily-pipeline-validation.yml`
- `security-scan.yml`
- `ci.yml` (via requirements)
- `reusable-test.yml`
- `production-deploy.yml`
- `dependency-updates.yml`

## Monitoring Dashboards

### Grafana Dashboards

| Dashboard | Purpose |
|-----------|---------|
| API Performance | Request latency, throughput |
| Database Metrics | Query performance, connections |
| Cache Performance | Hit rate, memory usage |
| ML Pipeline | Model latency, accuracy |
| Infrastructure | CPU, memory, disk |
| Business Metrics | Active users, recommendations, portfolio updates |
| External APIs | Rate limit status, response times |

### Prometheus Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `http_requests_total` | Counter | method, endpoint, status |
| `http_request_duration_seconds` | Histogram | method, endpoint |
| `cache_hits_total` | Counter | cache_type |
| `db_query_duration_seconds` | Histogram | query_type |

### SLO Targets (`infrastructure/monitoring/alerts/slo-targets.yml`)

SLO definitions have been added including availability, latency p95, and error rate targets.

## Cost Profile

| Component | Monthly Cost |
|-----------|-------------|
| VPS/Compute | ~$20 |
| Database | ~$10 |
| Redis | ~$5 |
| Monitoring | ~$5 |
| **Total** | **~$40 (under $50 budget)** |

Elasticsearch was removed in favor of PostgreSQL full-text search (pg_trgm),
saving $15-20/month.

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

# Full restart
docker compose down && docker compose up -d

# View logs
docker compose logs -f <service>
```
