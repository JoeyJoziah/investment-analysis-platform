# System Architecture Codemap

**Last Updated:** 2026-03-04
**Wave:** Post-Loki Remediation (Waves 1-14, P0-P5 Complete)
**Status:** Production-Approaching (Staging-Ready)

---

## High-Level Architecture

```
                                    +------------------+
                                    |   Load Balancer  |
                                    +--------+---------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
    +---------v---------+         +----------v----------+         +---------v---------+
    |   React Frontend  |         |   FastAPI Backend   |         |   WebSocket Hub   |
    |  (React 18/Vite)  |         | (12-layer middleware)|         |   (Real-time)     |
    +--------+----------+         +----------+----------+         +---------+---------+
             |                               |                              |
             +---------------+---------------+---------------+--------------+
                             |                               |
                   +---------v---------+           +---------v---------+
                   |   PostgreSQL DB   |           |   Redis Cache     |
                   | (TimescaleDB ext) |           | (Multi-tier TTL)  |
                   +-------------------+           +-------------------+
                             |
              +--------------+---------------+
              |                              |
    +---------v---------+          +---------v---------+
    |   ML Pipeline     |          |   ETL Pipeline    |
    | (XGBoost/Prophet/ |          | (Multi-source)    |
    |  LightGBM/LSTM)   |          +-------------------+
    +-------------------+
```

---

## API Routing Architecture

**Pattern:** Single prefix in `main.py`, no prefix in individual routers.

### Router Registration

```python
# backend/api/main.py
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
app.include_router(ml.router, prefix="/api/ml", tags=["ml"])
```

---

## API Route Map

### Core API Endpoints

| Endpoint | Router | Method | Purpose |
|----------|--------|--------|---------|
| `/api/health` | health | GET | Basic health check |
| `/api/health/readiness` | health | GET | Service readiness (DB, Redis) |
| `/api/health/liveness` | health | GET | Kubernetes liveness probe |
| `/api/health/metrics` | health | GET | System metrics |
| `/api/health/ping` | health | GET | Simple ping/pong |
| `/api/stocks` | stocks | GET | List stocks with filtering |
| `/api/stocks/search` | stocks | GET | Search stocks by name/symbol |
| `/api/stocks/{symbol}` | stocks | GET | Stock detail |
| `/api/stocks/{symbol}/quote` | stocks | GET | Real-time quote |
| `/api/stocks/{symbol}/history` | stocks | GET | Price history |
| `/api/auth/register` | auth | POST | User registration |
| `/api/auth/login` | auth | POST | User login |
| `/api/auth/me` | auth | GET | Current user info |
| `/api/recommendations` | recommendations | GET | Get recommendations |
| `/api/analysis/{symbol}` | analysis | GET | Stock analysis |
| `/api/portfolio` | portfolio | GET/POST | Portfolio management |
| `/api/admin/*` | admin | Various | Admin operations |
| `/api/trading/validate` | trading | POST | Validate order pre-flight |
| `/api/trading/execute` | trading | POST | Execute trade |
| `/api/trading/impact` | trading | POST | Portfolio impact analysis |
| `/api/ml/predictions` | ml | POST | Run ML prediction |
| `/api/ml/models` | ml | GET | List available models |
| `/api/ml/models/{model_id}` | ml | GET | Model details |
| `/ws/*` | websocket | WS | Real-time updates |

---

## Database Architecture

### Core Models (`backend/models/unified_models.py`)

```
Users (1) ----< Portfolios (N) ----< Positions (N)
   |                |
   |                +----< Transactions (N)
   |                +----< Orders (N)
   |
   +----< Watchlists (N) ----< Stock
   +----< Alerts (N)
   +----< UserSessions (N)
   +----< AuditLogs (N)

Exchanges (1) ----< Stocks (N) ----< PriceHistory (N)
                       |
                       +----< Fundamentals (N)
                       +----< TechnicalIndicators (N)
                       +----< NewsSentiment (N)
                       +----< MLPrediction (N)
                       +----< Recommendation (N)

Sectors (1) ----< Industries (N) ----< Stocks (N)
```

### Canonical ORM

`backend/models/unified_models.py` is the single source of truth for all ORM models
and Alembic migrations. All routers and services import from this module.

Key field conventions:
- `Position.avg_cost_basis` (not `cost_basis`)
- `Transaction.trade_date` (not `executed_at`)
- `Transaction.total_amount` (required field)
- `Watchlist.is_public` (Boolean, default False)
- `Stock.industry_id` (FK to industries, not string)

---

## Middleware Stack

### Execution Order (12 layers)

```
Request
   |
   v
+--------------------+
| PerformanceMiddleware | (timing, metrics)
+--------------------+
   |
   v
+--------------------+
| ConnectionPoolMiddleware | (connection management)
+--------------------+
   |
   v
+--------------------+
| TrustedHostMiddleware | (host validation)
+--------------------+
   |
   v
+--------------------+
| GZipMiddleware | (compression, min 1000 bytes)
+--------------------+
   |
   v
+--------------------+
| CORSMiddleware | (cross-origin, preflight cached 3600s)
+--------------------+
   |
   v
+--------------------+
| PrometheusInstrumentator | (metrics collection)
+--------------------+
   |
   v
Router Handler
```

### Rate Limiter Configuration

```python
# backend/security/rate_limiter.py
# In TESTING mode, rate limiting is bypassed
if os.getenv("TESTING") == "True":
    return RateLimitStatus(allowed=True, ...)
```

---

## Performance Architecture

### Cache Tiers (TTL Strategy)

| Tier | Storage | TTL | Use Case |
|------|---------|-----|----------|
| L1 | In-memory | 60-3600s | Real-time quotes |
| L2 | Redis | 300-14400s | API responses |
| L3 | Redis | 1800-604800s | Static data |

### Parallel Processing

```
Recommendation Generation:
   |
   +-- Parallel API Calls (asyncio.gather)
   |      +-- Finnhub (quote)
   |      +-- Alpha Vantage (fundamentals)
   |      +-- Polygon (historical)
   |
   +-- Batch Database Queries
          +-- get_bulk_price_history()
          +-- get_latest_prices_bulk()
          +-- get_top_stocks()
```

---

## Test Architecture

### Test Suite Statistics (2026-03-04)

| Metric | Value |
|--------|-------|
| Backend tests passing | 5,020 |
| Skipped (infra-only) | 8 |
| xfailed | 2 |
| Failed | 0 |
| Frontend tests | 201 |
| Total test files | 71+ |

### Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `backend/tests/unit/` | 28+ files, services/utils/ML |
| Integration | `backend/tests/integration/` | 16 files, API flows |
| Security | `backend/tests/security/` | 5 files, CSRF/OWASP |
| Middleware | `backend/tests/middleware/` | 4 files |
| Frontend | `frontend/web/src/**/*.test.tsx` | 13 files |

### Key Test Patterns

**Fixture setup (`backend/tests/conftest.py`):**
- `authenticated_client` bypasses JWT via dependency override
- `async_client` has NO auth (use for 401 testing)
- JWT uses RS256 with auto-generated RSA keys
- Two `get_current_user` paths: `oauth2.py` (returns User ORM) vs `utils/auth.py` (returns dict)
- conftest overrides both: `get_current_user` AND `get_current_user_utils`

---

## Security Architecture

### Authentication Flow

```
Client                    API                     Database
   |                       |                          |
   |-- POST /auth/login -->|                          |
   |                       |-- Validate credentials -->|
   |                       |<-- User record -----------|
   |                       |                          |
   |                       |-- Generate JWT (RS256)   |
   |<-- JWT token ---------|                          |
   |                       |                          |
   |-- GET /api/* -------->|                          |
   |   (Authorization:     |                          |
   |    Bearer <token>)    |                          |
   |                       |-- Verify JWT             |
   |<-- Response ----------|                          |
```

### Security Stack (Fully Implemented)

| Component | Implementation | Status |
|-----------|---------------|--------|
| JWT Auth | RS256 with auto-generated RSA keys | COMPLETE |
| RBAC | `security/rbac.py` - in-memory + optional DB-backed | COMPLETE |
| Crypto | `security/crypto_utils.py` - Fernet (AES-128-CBC) + RSA-2048 | COMPLETE |
| Passwords | `security/password_manager.py` - bcrypt (work factor 12) + legacy PBKDF2 verify | COMPLETE |
| CSP | `script-src 'self'` only; `style-src` allows `'unsafe-inline'` for MUI | HARDENED |
| Rate Limiting | Redis-backed distributed, 4 categories | COMPLETE |
| CSRF | 67 tests | COMPLETE |
| Audit Logging | Per-request logs | COMPLETE |
| GDPR | 13 endpoints, field-level encryption key via env | COMPLETE |

### Rate Limiting Categories

| Category | Limit | Window |
|----------|-------|--------|
| AUTHENTICATION | 5 | 15 min |
| REGISTRATION | 3 | 1 hour |
| API | 60 | 1 min |
| ADMIN | 100 | 1 min |

---

## Deployment Architecture

### Container Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| backend | fastapi | 8000 | API server |
| frontend | node/nginx | 3000 | Web app |
| postgres | timescale/timescaledb:2.12.1-pg15 | 5432 | Database |
| redis | redis:7.2-alpine | 6379 | Cache |
| celery_worker | backend | - | Background tasks |
| celery_beat | backend | - | Scheduled tasks |
| prometheus | prometheus | 9090 | Metrics |
| grafana | grafana | 3001 | Dashboards |
| loki | grafana/loki:2.9.3 | 3100 | Log aggregation |
| promtail | grafana/promtail:2.9.3 | - | Log shipping |
| certbot | certbot/certbot:v2.7.4 | - | SSL auto-renewal |

### CI/CD Runtime Versions

| Runtime | Version | Notes |
|---------|---------|-------|
| Python | 3.12 (primary) | CI matrix also tests 3.10, 3.11 |
| Node.js | 20 | Standardized across all workflows |
| `actions/setup-python` | v5 | |
| `actions/setup-node` | v4 | |
| `actions/upload-artifact` | v4 | |
| `github/codeql-action/*` | v3 | |

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Main API | `backend/api/main.py` |
| Unified Models | `backend/models/unified_models.py` |
| Router Registry | `backend/api/routers/__init__.py` |
| Settings | `backend/config/settings.py` |
| Security Config | `backend/security/security_config.py` |
| RBAC | `backend/security/rbac.py` |
| Crypto Utils | `backend/security/crypto_utils.py` |
| Password Manager | `backend/security/password_manager.py` |
| Rate Limiter | `backend/security/rate_limiter.py` |
| Cache Utils | `backend/utils/cache.py` |
| Test Fixtures | `backend/tests/conftest.py` |
| ETL Data Classes | `backend/etl/unlimited_data_extractor.py` |

---

## Related Codemaps

- [BACKEND.md](BACKEND.md) - Detailed backend structure
- [FRONTEND.md](FRONTEND.md) - Frontend architecture
- [DATA_FLOW.md](DATA_FLOW.md) - Data pipeline flows
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) - DevOps configuration
