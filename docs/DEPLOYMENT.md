# Production Deployment Guide

**Last Updated**: 2026-03-04
**Status**: Production-Ready

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [SSL Certificate Setup](#ssl-certificate-setup)
3. [Domain Configuration](#domain-configuration)
4. [Production Environment](#production-environment)
5. [Database Setup](#database-setup)
6. [Service Startup](#service-startup)
7. [Smoke Testing](#smoke-testing)
8. [Monitoring & Verification](#monitoring--verification)
9. [Scaling Considerations](#scaling-considerations)
10. [Backup & Recovery](#backup--recovery)

---

## Pre-Deployment Checklist

### Infrastructure Requirements
- [ ] Server with minimum 4GB RAM (8GB recommended)
- [ ] 50GB+ disk space for databases and models
- [ ] Docker Engine 20.10+
- [ ] Docker Compose 2.0+
- [ ] Public IP address or domain name
- [ ] Port 80 (HTTP) accessible for Let's Encrypt challenge
- [ ] Port 443 (HTTPS) open for production traffic

### Configuration Files
- [ ] `.env.production` created from `.env.example`
- [ ] `GDPR_ENCRYPTION_KEY` set (generate: `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)
- [ ] `JWT_SECRET_KEY` configured (generate: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`)
- [ ] `DB_PASSWORD` and `REDIS_PASSWORD` set to strong random values
- [ ] `CORS_ORIGINS` set to your production domain(s)
- [ ] Financial API keys configured (Finnhub, Alpha Vantage, etc.)

### Domain & SSL
- [ ] Domain name registered and DNS A record pointing to server IP
- [ ] Email address for Let's Encrypt certificate

### Monitoring
- [ ] Slack webhook configured (optional)
- [ ] SMTP credentials set for alert emails (optional)
- [ ] AlertManager rules reviewed in `infrastructure/monitoring/alerts/`

---

## SSL Certificate Setup

### Let's Encrypt via Certbot (Production)

The production stack includes a `certbot` container that handles certificate issuance and automatic renewal. Certificates renew every 12 hours if needed (certificates expire after 90 days).

#### Initial Certificate Request

Before starting the full stack, obtain the initial certificate with standalone mode:

```bash
# Replace with your actual domain and email
DOMAIN="yourdomain.com"
EMAIL="admin@yourdomain.com"

# Request certificate (standalone - requires port 80 open, no nginx running yet)
certbot certonly --standalone \
  -d "$DOMAIN" \
  -d "www.$DOMAIN" \
  --email "$EMAIL" \
  --agree-tos \
  --non-interactive

# Copy to project ssl/ directory (mounted into nginx and certbot containers)
mkdir -p ./ssl/live/$DOMAIN
cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem ./ssl/live/$DOMAIN/
cp /etc/letsencrypt/live/$DOMAIN/privkey.pem ./ssl/live/$DOMAIN/
chmod 600 ./ssl/live/$DOMAIN/privkey.pem
```

Or use the provided init script which handles this automatically:

```bash
./scripts/init-ssl.sh yourdomain.com admin@yourdomain.com
```

#### How Auto-Renewal Works

The `certbot` container in `docker-compose.production.yml` runs a renewal loop every 12 hours using webroot mode (the `certbot_webroot` Docker volume is shared with nginx at `/.well-known/acme-challenge/`). No manual renewal is required.

#### Self-Signed Certificate (Development/Testing Only)

```bash
mkdir -p ./ssl
openssl req -x509 -newkey rsa:4096 -nodes \
  -out ./ssl/cert.pem \
  -keyout ./ssl/key.pem \
  -days 365 \
  -subj "/CN=localhost"
```

---

## Domain Configuration

### DNS Setup

```bash
# Get server public IP
curl ifconfig.me
```

In your domain registrar, create:

```
Record Type: A
Name: @ (root domain)
Value: <your-server-ip>
TTL: 3600

# Optional www redirect
Record Type: CNAME
Name: www
Value: yourdomain.com
TTL: 3600
```

Verify DNS has propagated:

```bash
dig yourdomain.com @8.8.8.8
nslookup yourdomain.com
```

### Firewall

```bash
sudo ufw allow 80/tcp    # Let's Encrypt HTTP challenge
sudo ufw allow 443/tcp   # HTTPS production traffic
sudo ufw status
```

---

## Production Environment

### Environment Variables

Create `.env.production` from the template:

```bash
cp .env.example .env.production
nano .env.production
```

**Required for production**:

```bash
# Runtime
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Database (uses investment_user role created by init.sql)
DB_HOST=postgres
DB_PORT=5432
DB_NAME=investment_db
DB_USER=postgres
DB_PASSWORD=<strong-random-password>
DATABASE_URL=postgresql://postgres:<password>@postgres:5432/investment_db

# Redis (broker on DB 0, Celery results on DB 1)
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
REDIS_PASSWORD=<strong-random-password>
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/1

# Security (all required)
SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>
JWT_SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>
GDPR_ENCRYPTION_KEY=<generate-with-Fernet.generate_key()>

# CORS (list your production domains)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Financial APIs
FINNHUB_API_KEY=<your-key>
ALPHA_VANTAGE_API_KEY=<your-key>
POLYGON_API_KEY=<your-key>
NEWS_API_KEY=<your-key>

# Monitoring (optional but recommended)
PROMETHEUS_REMOTE_URL=http://victoriametrics:8428/api/v1/write
GRAFANA_ADMIN_PASSWORD=<strong-password>
SLACK_WEBHOOK_URL=<optional>
```

### Generate Secure Keys

```bash
# Application and JWT secrets
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# GDPR encryption key (Fernet format)
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Database/Redis passwords
python3 -c "import secrets; print(secrets.token_urlsafe(24))"
```

---

## Database Setup

The `investment_user` database role is created automatically by `infrastructure/docker/postgres/init.sql` on first container start. It has DML privileges (SELECT, INSERT, UPDATE, DELETE) on all public tables but not DDL (schema changes). Alembic migrations run as the `postgres` superuser.

### Verify Database Role

```bash
docker compose -f docker-compose.production.yml exec postgres \
  psql -U postgres -c "\du investment_user"
```

### Run Migrations

```bash
docker compose -f docker-compose.production.yml exec backend \
  python -m alembic upgrade head
```

### Verify Database

```bash
# Count tables
docker compose -f docker-compose.production.yml exec postgres \
  psql -U investment_user -d investment_db -c "\dt" | wc -l

# Count stocks (non-zero after data load)
docker compose -f docker-compose.production.yml exec postgres \
  psql -U investment_user -d investment_db -c "SELECT COUNT(*) FROM stocks;"
```

### Manual Backup

```bash
# Full PostgreSQL dump
docker compose -f docker-compose.production.yml exec postgres \
  pg_dump -U investment_user -d investment_db > backup-$(date +%Y%m%d-%H%M%S).sql
```

---

## Service Startup

### Production Deployment

The production stack uses a Docker Compose overlay pattern:

```bash
# Build production images
docker compose -f docker-compose.production.yml build

# Start all services
docker compose -f docker-compose.production.yml up -d

# Verify all services are healthy
docker compose -f docker-compose.production.yml ps
```

Expected healthy services:

| Container | Purpose |
|-----------|---------|
| investment_db_prod | PostgreSQL + TimescaleDB |
| investment_cache_prod | Redis 7 |
| investment_api_prod | FastAPI backend |
| investment_web_prod | React frontend (Nginx) |
| investment_worker_prod | Celery worker |
| investment_scheduler_prod | Celery beat scheduler |
| investment_nginx_prod | TLS termination + reverse proxy |
| investment_certbot | Let's Encrypt auto-renewal |
| investment_prometheus_prod | Metrics scraping |
| investment_victoriametrics_prod | Long-term metric storage (90d) |
| investment_grafana_prod | Dashboards |
| investment_loki_prod | Log aggregation |
| investment_promtail_prod | Log shipping agent |

### Check Service Logs

```bash
# Follow all logs
docker compose -f docker-compose.production.yml logs -f

# Single service
docker compose -f docker-compose.production.yml logs -f investment_api_prod

# Last 100 lines
docker compose -f docker-compose.production.yml logs -n 100 investment_api_prod

# Filter errors
docker compose -f docker-compose.production.yml logs | grep ERROR
```

---

## Smoke Testing

```bash
# Backend health
curl https://yourdomain.com/api/health
# Expected: {"status": "healthy", ...}

# Database health
curl https://yourdomain.com/api/health/db

# Redis health
curl https://yourdomain.com/api/health/redis

# Frontend loads
curl -I https://yourdomain.com
# Expected: HTTP/2 200

# Authenticated API call (get JWT first)
JWT=$(curl -s -X POST https://yourdomain.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user@example.com","password":"..."}' | jq -r .access_token)

curl -H "Authorization: Bearer $JWT" https://yourdomain.com/api/stocks?limit=1
```

---

## Monitoring & Verification

### Monitoring Service URLs (Production)

| Service | Default Access | Purpose |
|---------|---------------|---------|
| Grafana | https://yourdomain.com:3001 | Dashboards (change admin/admin on first login) |
| Prometheus | https://yourdomain.com:9090 | Raw metrics (restrict access in prod) |
| VictoriaMetrics | internal :8428 | Long-term storage (not exposed externally) |
| Loki | internal :3100 | Log backend for Grafana (not exposed externally) |

### Key Metrics to Watch

```
API Performance:
- Request rate > 0 for active users
- p95 response time < 500ms
- Error rate < 1%

Database:
- Active connections < 80% of max_connections (300)
- p95 query time < 100ms
- Disk usage < 80%

Cache:
- Redis hit rate > 80%
- Memory < 512MB limit

System:
- CPU < 80% sustained
- Memory < 80%
- Disk < 80%
```

### Prometheus Remote Write to VictoriaMetrics

Set `PROMETHEUS_REMOTE_URL=http://victoriametrics:8428/api/v1/write` in your production `.env` to enable long-term metric retention (90-day default, configurable via `-retentionPeriod` in the `victoriametrics` service command).

### Log Aggregation (Loki + Promtail)

Promtail ships logs from `./logs/` and `/var/log/` on the host into Loki. Query logs in Grafana using the Loki datasource. No Elasticsearch required.

### SLO Alerts

Alert rules are defined in `infrastructure/monitoring/alerts/slo-targets.yml` and `infrastructure/monitoring/alerts/investment-platform.yml`. AlertManager routes to Slack/email/PagerDuty based on severity.

---

## Scaling Considerations

### Horizontal Scaling

```bash
# Scale backend API replicas
docker compose -f docker-compose.production.yml up -d --scale backend=3

# Scale Celery workers
docker compose -f docker-compose.production.yml up -d --scale celery_worker=5
```

Note: Nginx must be configured as a load balancer when running multiple backend replicas.

### Database Optimization

```sql
-- Check slow queries (requires pg_stat_statements extension)
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Vacuum and analyze
VACUUM ANALYZE;
```

### Redis Optimization

```bash
# Monitor Redis memory
docker compose -f docker-compose.production.yml exec redis \
  redis-cli -a "$REDIS_PASSWORD" INFO memory

# Check eviction policy (allkeys-lru is configured by default)
docker compose -f docker-compose.production.yml exec redis \
  redis-cli -a "$REDIS_PASSWORD" CONFIG GET maxmemory-policy
```

---

## Backup & Recovery

### Automated Backups

```bash
# Manual database backup
./db-backup.sh

# List existing backups
ls -lh ./backups/
```

### PostgreSQL Recovery

```bash
# 1. Stop application services (keep database running)
docker compose -f docker-compose.production.yml stop backend celery_worker celery_beat

# 2. Restore from backup
docker compose -f docker-compose.production.yml exec -T postgres \
  psql -U investment_user -d investment_db < ./backups/backup-20260304.sql

# 3. Verify row counts
docker compose -f docker-compose.production.yml exec postgres \
  psql -U investment_user -d investment_db -c "SELECT COUNT(*) FROM stocks;"

# 4. Restart application services
docker compose -f docker-compose.production.yml start backend celery_worker celery_beat
```

### Backup Retention

```bash
# Delete local backups older than 7 days
find ./backups/ -name "*.sql" -mtime +7 -delete
```

---

## Troubleshooting Common Issues

### Service Won't Start

```bash
docker compose -f docker-compose.production.yml logs investment_api_prod
docker stats --no-stream
docker compose -f docker-compose.production.yml restart investment_api_prod
```

### GDPR Key Missing (Backend Exits Immediately)

```bash
# Generate key
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
echo "GDPR_ENCRYPTION_KEY=$FERNET_KEY" >> .env.production

docker compose -f docker-compose.production.yml restart investment_api_prod
```

### SSL Certificate Issues

```bash
# Check certificate expiry
openssl x509 -in ./ssl/live/yourdomain.com/fullchain.pem -noout -dates

# Force renewal
docker compose -f docker-compose.production.yml exec certbot \
  certbot renew --force-renewal
```

### Database Connection Refused

```bash
# Verify container is healthy
docker compose -f docker-compose.production.yml ps postgres

# Check credentials
grep DB_ .env.production

# Test directly
docker compose -f docker-compose.production.yml exec postgres \
  pg_isready -U postgres
```

---

## Post-Deployment Checklist

- [ ] All services running and healthy (`docker compose ps`)
- [ ] SSL certificate active (`curl -I https://yourdomain.com`)
- [ ] API health endpoint returns 200
- [ ] Authentication working (login and JWT issued)
- [ ] Grafana dashboards showing live data
- [ ] Alerts configured and test alert fired
- [ ] Loki receiving logs (check Grafana > Explore > Loki)
- [ ] VictoriaMetrics receiving Prometheus remote write
- [ ] Database backed up and backup verified
- [ ] `GRAFANA_ADMIN_PASSWORD` changed from default
- [ ] `SESSION_COOKIE_SECURE=true` in production `.env`
- [ ] `FORCE_HTTPS=true` in production `.env`

---

*Last Updated: 2026-03-04*
