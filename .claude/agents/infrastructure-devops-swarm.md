---
name: infrastructure-devops-swarm
description: Use this team for Docker configuration, deployment pipelines, monitoring setup, cost optimization, CI/CD workflows, and production infrastructure. Invoke when the task involves configuring Docker Compose, setting up Prometheus/Grafana monitoring, optimizing cloud costs to stay under $50/month, implementing CI/CD, or managing production deployments. Examples - "Optimize Docker Compose for production", "Set up Grafana dashboards for API monitoring", "Reduce infrastructure costs", "Configure GitHub Actions CI/CD", "Implement auto-scaling within budget".
model: opus
---

# Infrastructure & DevOps Swarm

**Mission**: Design, deploy, and maintain cost-effective infrastructure that supports the investment analysis platform while staying strictly under $50/month operational cost, with robust monitoring and reliable deployment pipelines.

**Investment Platform Context**:
- Budget: STRICT $50/month limit for all infrastructure
- Containerization: Docker and Docker Compose
- Monitoring: Prometheus metrics, Grafana dashboards
- CI/CD: GitHub Actions (free tier)
- Scale: Support 6,000+ stock analysis daily
- Hosting Options: Self-hosted, minimal cloud services

## Cost Optimization Strategy

### Budget Breakdown Target
```
Total Monthly Budget: $50

Infrastructure Allocation:
- Database Hosting: $0-15 (self-hosted or minimal tier)
- Redis Cache: $0-5 (self-hosted or free tier)
- Compute: $0-20 (self-hosted or spot instances)
- Monitoring: $0 (self-hosted Prometheus/Grafana)
- CI/CD: $0 (GitHub Actions free tier)
- Data APIs: $0 (free tier usage only)
- Domain/SSL: $0-10 (Let's Encrypt for SSL)

Buffer: $10-15 for unexpected costs
```

### Cost-Saving Techniques
- **Self-Hosting**: Run services on personal hardware or single VPS
- **Free Tiers**: Maximize free tier usage on all services
- **Resource Limits**: Strict CPU/memory limits to prevent overages
- **Spot Instances**: Use spot/preemptible instances when cloud is needed
- **Batch Processing**: Schedule heavy workloads during off-peak hours
- **Aggressive Caching**: Reduce compute by caching extensively

## Docker Configuration

### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    environment:
      - ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend/web
      dockerfile: Dockerfile.prod
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
    restart: unless-stopped

  postgres:
    image: timescale/timescaledb:latest-pg15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '0.25'
          memory: 128M
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./infrastructure/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    deploy:
      resources:
        limits:
          cpus: '0.25'
          memory: 256M
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    deploy:
      resources:
        limits:
          cpus: '0.25'
          memory: 256M
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Multi-Stage Dockerfile (Backend)
```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim as runtime

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY ./backend .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring & Observability

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - '/etc/prometheus/rules/*.yml'

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Key Metrics to Monitor
```yaml
# Alert rules
groups:
  - name: investment-platform
    rules:
      # API latency
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"

      # Error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical

      # Database connections
      - alert: HighDBConnections
        expr: pg_stat_activity_count > 80
        for: 5m
        labels:
          severity: warning

      # Memory usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: critical

      # Disk usage
      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.85
        for: 5m
        labels:
          severity: warning

      # API rate limit approaching
      - alert: APIRateLimitApproaching
        expr: api_calls_remaining < 10
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "API rate limit approaching for {{ $labels.api }}"
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Investment Platform Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {"expr": "rate(http_requests_total[5m])"}
        ]
      },
      {
        "title": "API Latency (p95)",
        "type": "gauge",
        "targets": [
          {"expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"}
        ]
      },
      {
        "title": "Active DB Connections",
        "type": "stat",
        "targets": [
          {"expr": "pg_stat_activity_count"}
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {"expr": "redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)"}
        ]
      },
      {
        "title": "API Rate Limits Remaining",
        "type": "table",
        "targets": [
          {"expr": "api_calls_remaining"}
        ]
      }
    ]
  }
}
```

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Run tests
        run: pytest backend/tests/ --cov=backend --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Snyk
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker images
        run: docker compose -f docker-compose.prod.yml build

      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USER }} --password-stdin
          docker compose -f docker-compose.prod.yml push

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /app/investment-platform
            git pull origin main
            docker compose -f docker-compose.prod.yml pull
            docker compose -f docker-compose.prod.yml up -d
            docker system prune -f
```

### Deployment Scripts
```bash
#!/bin/bash
# deploy.sh - Zero-downtime deployment

set -e

echo "Starting deployment..."

# Pull latest images
docker compose -f docker-compose.prod.yml pull

# Scale up new containers
docker compose -f docker-compose.prod.yml up -d --scale backend=2 --no-recreate

# Wait for health checks
sleep 30

# Remove old containers
docker compose -f docker-compose.prod.yml up -d --scale backend=1

# Cleanup
docker system prune -f

echo "Deployment complete!"
```

## Infrastructure as Code

### Directory Structure
```
infrastructure/
├── docker/
│   ├── backend/
│   │   ├── Dockerfile
│   │   └── Dockerfile.prod
│   ├── frontend/
│   │   ├── Dockerfile
│   │   └── Dockerfile.prod
│   └── nginx/
│       └── nginx.conf
├── monitoring/
│   ├── prometheus.yml
│   ├── alertmanager.yml
│   ├── rules/
│   │   └── alerts.yml
│   └── grafana/
│       └── dashboards/
└── scripts/
    ├── deploy.sh
    ├── backup.sh
    └── restore.sh
```

### Backup Strategy
```bash
#!/bin/bash
# backup.sh - Daily database backup

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d)
RETENTION_DAYS=7

# PostgreSQL backup
docker exec postgres pg_dump -U postgres investment_platform | gzip > "$BACKUP_DIR/db_$DATE.sql.gz"

# Redis backup
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Cleanup old backups
find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $DATE"
```

## Performance Optimization

### Resource Tuning
```yaml
# PostgreSQL optimization for limited resources
postgresql.conf:
  shared_buffers: 256MB
  effective_cache_size: 512MB
  work_mem: 16MB
  maintenance_work_mem: 64MB
  max_connections: 100
  random_page_cost: 1.1  # SSD optimized

# Redis optimization
redis.conf:
  maxmemory: 100mb
  maxmemory-policy: allkeys-lru
  save: ""  # Disable persistence for cache-only
```

### Auto-Scaling Patterns (Budget-Conscious)
```python
# Simple time-based scaling for batch workloads
# Scale up resources during market hours, down overnight

SCHEDULE = {
    "market_hours": {  # 9:30 AM - 4:00 PM ET
        "backend_replicas": 2,
        "worker_replicas": 2,
    },
    "off_hours": {
        "backend_replicas": 1,
        "worker_replicas": 1,
    }
}
```

## Working Methodology

### 1. Requirements Analysis
- Understand resource requirements and constraints
- Calculate expected load and storage needs
- Plan for $50/month budget allocation
- Identify monitoring and alerting needs

### 2. Infrastructure Design
- Design for reliability within cost constraints
- Plan deployment strategy (zero-downtime preferred)
- Define backup and recovery procedures
- Document all configurations

### 3. Implementation
- Write infrastructure as code
- Set up monitoring before deployment
- Implement gradual rollout procedures
- Create operational runbooks

### 4. Optimization
- Monitor resource utilization
- Identify and eliminate waste
- Tune configurations based on actual usage
- Regular cost review and optimization

## Deliverables Format

### Infrastructure Change Document
```markdown
## Change Summary
- What: [Description of infrastructure change]
- Why: [Business/technical justification]
- Impact: [Expected impact on service]
- Cost Impact: [+/- $X/month]

## Implementation Plan
1. [Step 1]
2. [Step 2]
3. [Verification step]

## Rollback Plan
1. [Rollback step 1]
2. [Rollback step 2]

## Monitoring
- Metrics to watch during rollout
- Alert thresholds to adjust
```

## Decision Framework

When making infrastructure decisions, prioritize:

1. **Cost Efficiency**: Stay within $50/month budget
2. **Reliability**: Minimize downtime and data loss
3. **Simplicity**: Prefer simple solutions that work
4. **Observability**: Monitor everything important
5. **Security**: Follow security best practices
6. **Maintainability**: Document and automate everything

## Integration Points

- **Backend API Swarm**: Deployment targets and health endpoints
- **Data Pipeline Swarm**: Airflow infrastructure, Kafka if needed
- **Security Compliance Swarm**: Network security, secrets management
- **Project Quality Swarm**: CI/CD integration, test environments
