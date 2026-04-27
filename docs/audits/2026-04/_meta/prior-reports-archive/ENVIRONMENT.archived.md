> **ARCHIVED 2026-04-27 by 16-config-secrets**
> Original: docs/ENVIRONMENT.md
> Validation summary: 6/8 claims still current.
> See `../../reports/16-config-secrets.md` §2 for per-claim status.

# Environment Variables Reference

> Reflects current `.env.example`. Last updated: 2026-03-04

---

## Quick Start

```bash
cp .env.example .env
# Edit .env with your values
chmod 600 .env
```

Never commit `.env` to version control.

---

## Environment Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Options: `development`, `staging`, `production` |
| `DEBUG` | `false` | Enable debug mode (never `true` in production) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `NODE_ENV` | `development` | Node.js environment |

---

## Application Core

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Yes | Application secret (min 64 chars). Generate: `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `JWT_SECRET_KEY` | Yes | JWT signing key (min 64 chars, same generation command) |
| `MASTER_SECRET_KEY` | Yes | HMAC integrity key (min 128 chars). Generate: `python3 -c "import secrets; print(secrets.token_hex(64))"` |
| `JWT_ALGORITHM` | No | Default: `RS256` (RSA keypair auto-generated at startup) |
| `JWT_EXPIRATION_HOURS` | No | Default: `24` |
| `FERNET_KEY` | Yes | Symmetric encryption key. Generate: `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |

---

## Database Configuration

### PostgreSQL

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | Database host (use `postgres` inside Docker) |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `investment_db` | Database name |
| `DB_USER` | `postgres` | Superuser (used by Alembic migrations) |
| `DB_PASSWORD` | - | Required |
| `DATABASE_URL` | - | Full connection string (auto-composed from above) |
| `DB_SSL_MODE` | `prefer` | Options: `disable`, `allow`, `prefer`, `require`, `verify-ca`, `verify-full` |
| `DB_POOL_SIZE` | `20` | SQLAlchemy connection pool size |
| `DB_POOL_TIMEOUT` | `30` | Pool checkout timeout (seconds) |
| `DB_POOL_RECYCLE` | `3600` | Recycle connections after N seconds |

Note: An `investment_user` application role with DML-only privileges is created automatically by `infrastructure/docker/postgres/init.sql`.

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host (use `redis` inside Docker) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | - | Required |
| `REDIS_DB` | `0` | Redis DB number for general cache and Celery broker |
| `REDIS_URL` | - | Full Redis URL (auto-composed) |
| `REDIS_SSL` | `false` | Enable TLS |
| `REDIS_MAXMEMORY` | `512mb` | Maximum memory (configured directly in `redis-server` args) |
| `REDIS_MAXMEMORY_POLICY` | `allkeys-lru` | Eviction policy |

Note: Celery uses **DB 0** as the broker and **DB 1** as the result backend (see `CELERY_*` vars below).

---

## Security

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Comma-separated list of allowed origins. In production, set to your HTTPS domain(s). |
| `CORS_ALLOW_CREDENTIALS` | `true` | Allow cookies/auth headers cross-origin |
| `CORS_MAX_AGE` | `86400` | Preflight cache duration (seconds) |

### GDPR Compliance

| Variable | Default | Description |
|----------|---------|-------------|
| `GDPR_ENCRYPTION_KEY` | - | **Required.** Fernet key for PII encryption. Generate: `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `GDPR_COMPLIANCE` | `enabled` | Enable GDPR compliance checks |
| `PII_ENCRYPTION` | `enabled` | Encrypt PII at rest |
| `DATA_ANONYMIZATION` | `enabled` | Enable anonymization |
| `RIGHT_TO_BE_FORGOTTEN` | `enabled` | Support deletion requests |
| `DATA_PORTABILITY` | `enabled` | Support data export |

### Session

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSION_COOKIE_SECURE` | `false` | Set `true` in production (requires HTTPS) |
| `SESSION_COOKIE_HTTPONLY` | `true` | Prevent JS cookie access |
| `SESSION_COOKIE_SAMESITE` | `lax` | SameSite policy |
| `SESSION_TIMEOUT_MINUTES` | `60` | Session expiry |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_PER_IP` | `100` | Requests per IP per window |
| `RATE_LIMIT_PER_USER` | `1000` | Requests per authenticated user per window |
| `RATE_LIMIT_WINDOW_MINUTES` | `15` | Rate limit window |

### Security Headers

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SECURITY_HEADERS` | `true` | Add HSTS, CSP, X-Frame-Options headers |
| `HSTS_MAX_AGE` | `31536000` | HSTS max-age (1 year) |
| `FORCE_HTTPS` | `false` | Force HTTPS redirect (set `true` in production) |

### SSL/TLS

| Variable | Default | Description |
|----------|---------|-------------|
| `SSL_ENABLED` | `false` | Enable TLS (Nginx handles TLS in production) |
| `SSL_CERT_PATH` | `/etc/ssl/certs/cert.pem` | Certificate path |
| `SSL_KEY_PATH` | `/etc/ssl/private/key.pem` | Private key path |

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
| `FMP_API_KEY` | Financial Modeling Prep |
| `FRED_API_KEY` | Federal Reserve economic data |
| `MARKETAUX_API_KEY` | Market news |
| `YAHOO_FINANCE_API_KEY` | Yahoo Finance |

---

## Celery & Background Tasks

| Variable | Default | Description |
|----------|---------|-------------|
| `CELERY_BROKER_URL` | `redis://:password@redis:6379/0` | Message broker (Redis DB 0) |
| `CELERY_RESULT_BACKEND` | `redis://:password@redis:6379/1` | Result storage (Redis DB 1) |
| `CELERY_WORKER_CONCURRENCY` | `2` | Worker thread/process count |
| `CELERY_WORKER_MAX_TASKS_PER_CHILD` | `100` | Tasks before worker restart |
| `CELERY_TASK_TIME_LIMIT` | `300` | Hard task time limit (seconds) |
| `CELERY_TASK_SOFT_TIME_LIMIT` | `240` | Soft time limit (raises exception) |
| `CELERY_TIMEZONE` | `UTC` | Timezone for beat scheduler |

---

## Monitoring & Observability

### Prometheus / VictoriaMetrics

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_METRICS` | `true` | Expose `/metrics` endpoint |
| `METRICS_PORT` | `9090` | Prometheus scrape port |
| `PROMETHEUS_REMOTE_URL` | `http://victoriametrics:8428/api/v1/write` | Remote write endpoint for VictoriaMetrics long-term storage. Set this in production to enable 90-day metric retention. |

### Logging (Loki + Promtail)

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_FORMAT` | `json` | Log output format (`json` for structured logs; Promtail parses JSON) |
| `LOG_FILE_PATH` | `/var/log/investment_app` | Log file directory (mounted into Promtail) |
| `LOG_ROTATION_SIZE` | `100M` | Rotate at size |
| `LOG_RETENTION_DAYS` | `30` | Local log retention |
| `ENABLE_REQUEST_LOGGING` | `true` | Log all HTTP requests |
| `ENABLE_PERFORMANCE_LOGGING` | `true` | Log response times |

### Grafana

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFANA_URL` | `http://localhost:3001` | Grafana URL |
| `GRAFANA_PORT` | `3001` | External port (internal container port is 3000) |
| `GRAFANA_ADMIN_USER` | `admin` | Admin username |
| `GRAFANA_ADMIN_PASSWORD` | - | Admin password (change from default immediately) |

### Alerting

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_WEBHOOK_URL` | - | Slack incoming webhook URL |
| `SLACK_CHANNEL` | `#alerts` | Alert channel |
| `ENABLE_SLACK_NOTIFICATIONS` | `false` | Enable Slack alerts |
| `ALERTMANAGER_URL` | `http://alertmanager:9093` | AlertManager address |
| `PAGERDUTY_ROUTING_KEY` | - | PagerDuty integration key (optional) |

---

## SEC Compliance

| Variable | Default | Description |
|----------|---------|-------------|
| `SEC_EDGAR_USER_AGENT` | - | Format: `CompanyName email@example.com` (required for EDGAR API) |
| `SEC_COMPLIANCE_MODE` | `enabled` | Enable SEC compliance checks |
| `AUDIT_LOG_ENABLED` | `true` | Enable audit logging |
| `AUDIT_LOG_RETENTION_DAYS` | `2555` | 7 years (SEC requirement) |
| `DATA_RETENTION_DAYS` | `2555` | Data retention period |
| `TRANSACTION_LOGGING` | `true` | Log all recommendation events |

---

## Machine Learning & Models

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_TTL` | `900` | Model cache TTL (seconds) |
| `PREDICTION_CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence for recommendation output |
| `ENABLE_MODEL_VERSIONING` | `true` | Enable model versioning |
| `GPU_ENABLED` | `false` | Enable GPU acceleration |
| `MAX_MODEL_MEMORY_MB` | `512` | Maximum model memory |
| `HF_TOKEN` | - | HuggingFace token for model downloads |
| `HF_HOME` | `/app/ml_models/.hf_cache` | HuggingFace cache directory |
| `HF_HUB_ENABLED` | `false` | Enable HuggingFace Hub integration |

---

## Frontend Configuration

The frontend uses **Vite** (not Create React App). Environment variables must be prefixed with `VITE_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend API base URL |
| `VITE_WS_URL` | `http://localhost:8000` | WebSocket/Socket.IO origin (path handled automatically) |
| `VITE_APP_ENV` | `${ENVIRONMENT}` | Frontend environment label |
| `VITE_APP_VERSION` | `1.0.0` | App version displayed in UI |
| `VITE_GA_TRACKING_ID` | - | Google Analytics ID (optional) |
| `VITE_SENTRY_DSN` | - | Frontend Sentry DSN (optional) |

---

## Cost Monitoring

**Target budget: $50/month**

| Variable | Default | Description |
|----------|---------|-------------|
| `MONTHLY_BUDGET_LIMIT` | `50` | Monthly budget in USD |
| `COST_ALERT_THRESHOLD` | `40` | Alert at $40 (80% of budget) |
| `ENABLE_COST_MONITORING` | `true` | Enable cost tracking service |
| `API_RATE_LIMIT_BUFFER` | `0.8` | Use only 80% of free tier limits |
| `DAILY_API_LIMIT_FINNHUB` | `1800` | Daily Finnhub call cap |
| `DAILY_API_LIMIT_ALPHA_VANTAGE` | `25` | Daily Alpha Vantage cap |
| `DAILY_API_LIMIT_POLYGON` | `150` | Daily Polygon cap |
| `DAILY_API_LIMIT_NEWS` | `100` | Daily NewsAPI cap |

---

## Performance Tuning

### Cache TTL Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TTL_DEFAULT` | `300` | Default TTL (5 min) |
| `CACHE_TTL_STOCK_PRICES` | `60` | Real-time prices (1 min) |
| `CACHE_TTL_STOCK_FUNDAMENTALS` | `3600` | Fundamentals (1 hour) |
| `CACHE_TTL_NEWS` | `1800` | News (30 min) |
| `CACHE_TTL_RECOMMENDATIONS` | `900` | Recommendations (15 min) |
| `ENABLE_CACHE_WARMING` | `true` | Pre-warm cache on startup |

### API Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `API_RATE_LIMIT_PER_MINUTE` | `60` | Per-client request limit |
| `API_TIMEOUT_SECONDS` | `30` | Request timeout |
| `API_MAX_PAGE_SIZE` | `100` | Maximum pagination size |

### Workers

| Variable | Default | Description |
|----------|---------|-------------|
| `GUNICORN_WORKERS` | `4` | Gunicorn worker processes |
| `GUNICORN_THREADS` | `2` | Threads per worker |
| `GUNICORN_TIMEOUT` | `120` | Request timeout (seconds) |

---

## Testing & Development

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/investment_db_test` | Test database URL |
| `TEST_REDIS_URL` | `redis://localhost:6379/1` | Test Redis URL (DB 1 to avoid conflicts) |
| `ENABLE_TEST_ENDPOINTS` | `false` | Expose test-only API routes |
| `MOCK_EXTERNAL_APIS` | `false` | Mock financial data API calls |

---

## Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ADMIN_PANEL` | `true` | Admin UI panel |
| `ENABLE_API_DOCUMENTATION` | `true` | Swagger UI at `/docs` |
| `ENABLE_WEBSOCKET_UPDATES` | `true` | Real-time WebSocket updates |
| `ENABLE_EMAIL_NOTIFICATIONS` | `false` | Email alerts |

---

## Email Configuration (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `EMAIL_HOST` | `smtp.gmail.com` | SMTP host |
| `EMAIL_PORT` | `587` | SMTP port |
| `EMAIL_USE_TLS` | `true` | Use STARTTLS |
| `EMAIL_USERNAME` | - | SMTP username |
| `EMAIL_PASSWORD` | - | SMTP password or app password |
| `DEFAULT_FROM_EMAIL` | - | Default sender address |

---

## Security Best Practices

1. Never commit `.env` to version control (it is gitignored)
2. Use strong random values for all secrets (minimum 32 bytes of entropy)
3. Rotate API keys and passwords regularly
4. Set `chmod 600 .env` on the server
5. Use `.env.production` separate from `.env.development`
6. Verify no secrets appear in application logs or error messages
