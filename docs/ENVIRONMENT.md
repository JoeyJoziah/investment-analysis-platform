# Environment Variables Reference

> Auto-generated from `.env.example`. Last updated: 2026-01-26

This document provides a comprehensive reference for all environment variables used in the investment-analysis-platform.

---

## Quick Start

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Replace placeholder values with actual secrets
3. Never commit `.env` to version control

---

## Environment Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Options: development, staging, production |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `NODE_ENV` | `development` | Node.js environment |

---

## Application Core

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Yes | Application secret (min 64 chars). Generate with: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `JWT_SECRET_KEY` | Yes | JWT signing key (min 64 chars) |
| `JWT_ALGORITHM` | No | Default: `HS256` |
| `JWT_EXPIRATION_HOURS` | No | Default: `24` |
| `FERNET_KEY` | Yes | Encryption key. Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |

---

## Database Configuration

### PostgreSQL (Primary Database)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `investment_db` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | - | Database password (required) |
| `DATABASE_URL` | - | Full connection string (auto-generated) |
| `DB_SSL_MODE` | `prefer` | Options: disable, allow, prefer, require, verify-ca, verify-full |

### Redis (Caching & Sessions)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | - | Redis password |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_URL` | - | Full Redis URL (auto-generated) |
| `REDIS_SSL` | `false` | Enable SSL |
| `REDIS_MAXMEMORY` | `256mb` | Maximum memory |
| `REDIS_MAXMEMORY_POLICY` | `allkeys-lru` | Eviction policy |

### Elasticsearch (Search & Analytics)

| Variable | Default | Description |
|----------|---------|-------------|
| `ELASTICSEARCH_HOST` | `localhost` | ES host |
| `ELASTICSEARCH_PORT` | `9200` | ES port |
| `ELASTICSEARCH_USER` | `elastic` | ES user |
| `ELASTICSEARCH_PASSWORD` | - | ES password |
| `ELASTICSEARCH_URL` | - | Full ES URL (auto-generated) |
| `ELASTICSEARCH_HEAP_SIZE` | `512m` | JVM heap size |

---

## Financial Data API Keys

| Variable | Free Tier Limit | Where to Get |
|----------|-----------------|--------------|
| `ALPHA_VANTAGE_API_KEY` | 25 calls/day | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `FINNHUB_API_KEY` | 60 calls/minute | [finnhub.io](https://finnhub.io/register) |
| `POLYGON_API_KEY` | 5 calls/minute | [polygon.io](https://polygon.io/dashboard/signup) |
| `NEWS_API_KEY` | 100 requests/day | [newsapi.org](https://newsapi.org/register) |

### Optional APIs

| Variable | Description |
|----------|-------------|
| `YAHOO_FINANCE_API_KEY` | Yahoo Finance API |
| `FMP_API_KEY` | Financial Modeling Prep |
| `MARKETAUX_API_KEY` | Market news API |
| `FRED_API_KEY` | Federal Reserve data |
| `OPENWEATHER_API_KEY` | Weather data (for agriculture stocks) |

---

## SEC Compliance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SEC_EDGAR_USER_AGENT` | - | Format: `CompanyName email@example.com` |
| `SEC_COMPLIANCE_MODE` | `enabled` | Enable SEC compliance checks |
| `AUDIT_LOG_ENABLED` | `true` | Enable audit logging |
| `AUDIT_LOG_RETENTION_DAYS` | `2555` | 7 years for SEC compliance |
| `DATA_RETENTION_DAYS` | `2555` | Data retention period |
| `TRANSACTION_LOGGING` | `true` | Log all transactions |
| `COMPLIANCE_REPORTS_ENABLED` | `true` | Generate compliance reports |

---

## GDPR Compliance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GDPR_COMPLIANCE` | `enabled` | Enable GDPR compliance |
| `PII_ENCRYPTION` | `enabled` | Encrypt PII data |
| `DATA_ANONYMIZATION` | `enabled` | Enable data anonymization |
| `RIGHT_TO_BE_FORGOTTEN` | `enabled` | Support data deletion requests |
| `DATA_PORTABILITY` | `enabled` | Support data export |
| `COOKIE_CONSENT_REQUIRED` | `true` | Require cookie consent |
| `PRIVACY_POLICY_VERSION` | `1.0` | Current privacy policy version |
| `TERMS_OF_SERVICE_VERSION` | `1.0` | Current ToS version |

---

## Cost Monitoring & Limits

**Target Budget: $50/month**

| Variable | Default | Description |
|----------|---------|-------------|
| `MONTHLY_BUDGET_LIMIT` | `50` | Monthly budget in USD |
| `DAILY_API_LIMIT_FINNHUB` | `1800` | Daily Finnhub call limit |
| `DAILY_API_LIMIT_ALPHA_VANTAGE` | `25` | Daily Alpha Vantage limit |
| `DAILY_API_LIMIT_POLYGON` | `150` | Daily Polygon limit |
| `DAILY_API_LIMIT_NEWS` | `100` | Daily NewsAPI limit |
| `API_RATE_LIMIT_BUFFER` | `0.8` | Use only 80% of limits |
| `COST_ALERT_THRESHOLD` | `40` | Alert at $40 (80% of budget) |
| `ENABLE_COST_MONITORING` | `true` | Enable cost tracking |

---

## Machine Learning & Models

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_TTL` | `900` | Model cache time (seconds) |
| `PREDICTION_CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence for predictions |
| `ENABLE_MODEL_VERSIONING` | `true` | Enable model versioning |
| `MODEL_REGISTRY_URL` | - | MLflow registry URL (optional) |
| `ENABLE_ONLINE_LEARNING` | `false` | Enable online learning |
| `BATCH_PREDICTION_SIZE` | `100` | Batch size for predictions |
| `GPU_ENABLED` | `false` | Enable GPU acceleration |
| `MAX_MODEL_MEMORY_MB` | `512` | Maximum model memory |

---

## Celery & Background Tasks

| Variable | Default | Description |
|----------|---------|-------------|
| `CELERY_BROKER_URL` | `${REDIS_URL}` | Message broker URL |
| `CELERY_RESULT_BACKEND` | `${REDIS_URL}` | Result backend URL |
| `CELERY_WORKER_CONCURRENCY` | `2` | Worker concurrency |
| `CELERY_WORKER_MAX_TASKS_PER_CHILD` | `100` | Tasks before worker restart |
| `CELERY_TASK_TIME_LIMIT` | `300` | Hard time limit (seconds) |
| `CELERY_TASK_SOFT_TIME_LIMIT` | `240` | Soft time limit (seconds) |
| `CELERY_TIMEZONE` | `UTC` | Timezone for scheduling |

---

## Airflow Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AIRFLOW_DB_HOST` | `${DB_HOST}` | Airflow database host |
| `AIRFLOW_DB_PORT` | `${DB_PORT}` | Airflow database port |
| `AIRFLOW_DB_NAME` | `airflow` | Airflow database name |
| `AIRFLOW_DB_USER` | `airflow` | Airflow database user |
| `AIRFLOW_DB_PASSWORD` | - | Airflow database password |
| `AIRFLOW_CORE_PARALLELISM` | `8` | Max parallel tasks |
| `AIRFLOW_DAG_CONCURRENCY` | `4` | Tasks per DAG |
| `AIRFLOW_MAX_ACTIVE_RUNS` | `2` | Active DAG runs |
| `AIRFLOW_WEBSERVER_PORT` | `8080` | Web UI port |
| `AIRFLOW_ADMIN_USERNAME` | `admin` | Admin username |
| `AIRFLOW_ADMIN_PASSWORD` | - | Admin password |
| `AIRFLOW_ADMIN_EMAIL` | - | Admin email |

---

## Monitoring & Observability

### Metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_METRICS` | `true` | Enable metrics collection |
| `METRICS_PORT` | `9090` | Prometheus metrics port |
| `PROMETHEUS_PUSHGATEWAY_URL` | - | Pushgateway URL (optional) |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_FORMAT` | `json` | Log format (json/text) |
| `LOG_FILE_PATH` | `/var/log/investment_app` | Log file location |
| `LOG_ROTATION_SIZE` | `100M` | Rotate at size |
| `LOG_RETENTION_DAYS` | `30` | Keep logs for days |
| `ENABLE_REQUEST_LOGGING` | `true` | Log all requests |
| `ENABLE_PERFORMANCE_LOGGING` | `true` | Log performance metrics |

### Error Tracking (Sentry)

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTRY_DSN` | - | Sentry DSN (optional) |
| `SENTRY_ENVIRONMENT` | `${ENVIRONMENT}` | Environment name |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Trace sampling rate |

### Grafana

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFANA_URL` | `http://localhost:3001` | Grafana URL |
| `GRAFANA_PORT` | `3001` | External port (internal: 3000) |
| `GRAFANA_ADMIN_USER` | `admin` | Admin username |
| `GRAFANA_ADMIN_PASSWORD` | - | Admin password |

### Slack Notifications

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_WEBHOOK_URL` | - | Slack webhook (optional) |
| `SLACK_CHANNEL` | `#alerts` | Alert channel |
| `ENABLE_SLACK_NOTIFICATIONS` | `false` | Enable Slack alerts |

---

## Frontend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_API_URL` | `http://localhost:8000` | Backend API URL |
| `REACT_APP_WS_URL` | `ws://localhost:8000/api/ws` | WebSocket URL |
| `REACT_APP_ENVIRONMENT` | `${ENVIRONMENT}` | Frontend environment |
| `REACT_APP_VERSION` | `1.0.0` | App version |
| `REACT_APP_GOOGLE_ANALYTICS_ID` | - | GA tracking ID (optional) |
| `REACT_APP_ENABLE_ANALYTICS` | `false` | Enable analytics |
| `REACT_APP_SENTRY_DSN` | - | Frontend Sentry DSN |

---

## Deployment & Infrastructure

### Docker

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_REGISTRY` | `docker.io` | Docker registry |
| `DOCKER_USERNAME` | - | Docker username |
| `DOCKER_PASSWORD` | - | Docker password |
| `DOCKER_IMAGE_PREFIX` | `investment-app` | Image name prefix |

### Kubernetes

| Variable | Default | Description |
|----------|---------|-------------|
| `KUBERNETES_NAMESPACE` | `investment-app` | K8s namespace |
| `KUBERNETES_CONTEXT` | - | K8s context |
| `ENABLE_AUTOSCALING` | `true` | Enable HPA |
| `MIN_REPLICAS` | `1` | Minimum replicas |
| `MAX_REPLICAS` | `5` | Maximum replicas |

### DigitalOcean

| Variable | Description |
|----------|-------------|
| `DIGITALOCEAN_ACCESS_TOKEN` | DO API token |
| `DIGITALOCEAN_CLUSTER_ID` | K8s cluster ID |
| `DIGITALOCEAN_REGION` | Default: `nyc3` |
| `DIGITALOCEAN_SPACES_KEY` | Spaces access key |
| `DIGITALOCEAN_SPACES_SECRET` | Spaces secret |

### AWS (Alternative)

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | Default: `us-east-1` |
| `AWS_S3_BUCKET` | S3 bucket name |

### SSL/TLS

| Variable | Default | Description |
|----------|---------|-------------|
| `SSL_ENABLED` | `false` | Enable SSL |
| `SSL_CERT_PATH` | `/etc/ssl/certs/cert.pem` | Certificate path |
| `SSL_KEY_PATH` | `/etc/ssl/private/key.pem` | Key path |
| `FORCE_HTTPS` | `false` | Force HTTPS redirects |

---

## Performance Optimization

### Database Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_SHARED_BUFFERS` | `256MB` | Shared buffer size |
| `POSTGRES_EFFECTIVE_CACHE_SIZE` | `1GB` | Effective cache size |
| `POSTGRES_MAX_CONNECTIONS` | `100` | Max connections |
| `POSTGRES_WORK_MEM` | `4MB` | Work memory |
| `DB_POOL_SIZE` | `20` | Connection pool size |
| `DB_POOL_TIMEOUT` | `30` | Pool timeout (seconds) |
| `DB_POOL_RECYCLE` | `3600` | Recycle connections (seconds) |

### Cache TTL Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TTL_DEFAULT` | `300` | Default TTL (5 min) |
| `CACHE_TTL_STOCK_PRICES` | `60` | Stock prices (1 min) |
| `CACHE_TTL_STOCK_FUNDAMENTALS` | `3600` | Fundamentals (1 hour) |
| `CACHE_TTL_NEWS` | `1800` | News (30 min) |
| `CACHE_TTL_RECOMMENDATIONS` | `900` | Recommendations (15 min) |
| `ENABLE_CACHE_WARMING` | `true` | Pre-warm cache |

### API Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `API_RATE_LIMIT_PER_MINUTE` | `60` | Requests per minute |
| `API_TIMEOUT_SECONDS` | `30` | Request timeout |
| `API_MAX_PAGE_SIZE` | `100` | Max pagination size |
| `API_DEFAULT_PAGE_SIZE` | `20` | Default page size |

### Workers

| Variable | Default | Description |
|----------|---------|-------------|
| `NGINX_WORKER_PROCESSES` | `2` | Nginx workers |
| `NGINX_WORKER_CONNECTIONS` | `1024` | Connections per worker |
| `GUNICORN_WORKERS` | `4` | Gunicorn workers |
| `GUNICORN_THREADS` | `2` | Threads per worker |
| `GUNICORN_TIMEOUT` | `120` | Request timeout |

---

## Security Settings

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Allowed origins |
| `CORS_ALLOW_CREDENTIALS` | `true` | Allow credentials |
| `CORS_MAX_AGE` | `86400` | Preflight cache (24h) |

### Session

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSION_COOKIE_SECURE` | `false` | HTTPS only (set true in prod) |
| `SESSION_COOKIE_HTTPONLY` | `true` | HTTP only |
| `SESSION_COOKIE_SAMESITE` | `lax` | SameSite policy |
| `SESSION_TIMEOUT_MINUTES` | `60` | Session timeout |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_PER_IP` | `100` | Requests per IP |
| `RATE_LIMIT_PER_USER` | `1000` | Requests per user |
| `RATE_LIMIT_WINDOW_MINUTES` | `15` | Time window |

### Security Headers

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SECURITY_HEADERS` | `true` | Enable security headers |
| `HSTS_MAX_AGE` | `31536000` | HSTS max age (1 year) |
| `CSP_POLICY` | `default-src 'self'` | Content Security Policy |

---

## Testing & Development

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_DATABASE_URL` | - | Test database URL |
| `TEST_REDIS_URL` | - | Test Redis URL |
| `ENABLE_TEST_ENDPOINTS` | `false` | Enable test endpoints |
| `MOCK_EXTERNAL_APIS` | `false` | Mock external APIs |
| `SELENIUM_GRID_URL` | - | Selenium Grid URL (E2E tests) |

---

## Backup & Disaster Recovery

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUTOMATIC_BACKUPS` | `false` | Enable auto backups |
| `BACKUP_RETENTION_DAYS` | `30` | Backup retention |
| `BACKUP_S3_BUCKET` | - | S3 bucket for backups |
| `BACKUP_ENCRYPTION_KEY` | - | Backup encryption key |
| `DISASTER_RECOVERY_MODE` | `false` | DR mode |

---

## Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SOCIAL_FEATURES` | `false` | Social features |
| `ENABLE_PREMIUM_FEATURES` | `false` | Premium tier |
| `ENABLE_BETA_FEATURES` | `false` | Beta features |
| `ENABLE_ADMIN_PANEL` | `true` | Admin panel |
| `ENABLE_API_DOCUMENTATION` | `true` | API docs (Swagger) |
| `ENABLE_WEBSOCKET_UPDATES` | `true` | Real-time updates |
| `ENABLE_EMAIL_NOTIFICATIONS` | `false` | Email notifications |
| `ENABLE_PUSH_NOTIFICATIONS` | `false` | Push notifications |

---

## Email Configuration (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `EMAIL_BACKEND` | `smtp` | Email backend |
| `EMAIL_HOST` | `smtp.gmail.com` | SMTP host |
| `EMAIL_PORT` | `587` | SMTP port |
| `EMAIL_USE_TLS` | `true` | Use TLS |
| `EMAIL_USERNAME` | - | SMTP username |
| `EMAIL_PASSWORD` | - | SMTP password |
| `DEFAULT_FROM_EMAIL` | - | Default sender |

---

## Security Best Practices

1. **Never commit `.env`** to version control
2. **Use strong passwords** - minimum 16 characters, mix of characters
3. **Rotate secrets regularly** - especially API keys
4. **Secure file permissions**: `chmod 600 .env`
5. **Use environment-specific files**: `.env.development`, `.env.production`
6. **Monitor for exposed secrets** in logs and error messages
