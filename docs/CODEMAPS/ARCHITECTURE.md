# System Architecture Codemap

**Last Updated:** 2026-02-24
**Wave:** 5-6+ (CI/CD Hardening, Test Recovery)
**Status:** Production-Ready

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
    |   (Next.js/React) |         | (Performance Opt)   |         |   (Real-time)     |
    +--------+----------+         +----------+----------+         +---------+---------+
             |                               |                              |
             +---------------+---------------+---------------+--------------+
                             |                               |
                   +---------v---------+           +---------v---------+
                   |   PostgreSQL DB   |           |   Redis Cache     |
                   | (Primary Storage) |           | (Multi-tier TTL)  |
                   +-------------------+           +-------------------+
                             |
              +--------------+---------------+
              |                              |
    +---------v---------+          +---------v---------+
    |   ML Pipeline     |          |   ETL Pipeline    |
    | (XGBoost/Prophet) |          | (Multi-source)    |
    +-------------------+          +-------------------+
```

---

## API Routing Architecture (Wave 5 Fix)

**Pattern:** Single prefix in `main.py`, no prefix in individual routers.

### Router Registration (Correct Pattern)

```python
# backend/api/main_performance_optimized.py (lines 430-439)
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
```

### Wave 5 Double-Prefix Fix

**Problem:** Routers had `prefix="/xxx"` in their definition AND in `include_router()`.

| Router | Before (Double Prefix) | After (Single Prefix) |
|--------|------------------------|----------------------|
| auth.py | `prefix="/auth"` | `tags=["authentication"]` |
| portfolio.py | `prefix="/portfolio"` | `tags=["portfolio"]` |
| admin.py | `prefix="/admin"` | `tags=["admin"]` |
| analysis.py | `prefix="/analysis"` | `tags=["analysis"]` |
| recommendations.py | `prefix="/recommendations"` | `tags=["recommendations"]` |
| websocket.py | `prefix="/ws"` | `tags=["websocket"]` |

**Commit:** `178a92e`

---

## API Route Map

### Core API Endpoints

| Endpoint | Router | Method | Purpose |
|----------|--------|--------|---------|
| `/api/health` | health | GET | Basic health check |
| `/api/health/readiness` | health | GET | Service readiness (DB, Redis) |
| `/api/health/liveness` | health | GET | Kubernetes liveness probe |
| `/api/health/metrics` | health | GET | System metrics |
| `/api/health/ping` | health | GET | Simple ping/pong (Wave 5) |
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

### Wave 5 Model Updates

| Model | Field | Change | Commit |
|-------|-------|--------|--------|
| Watchlist | `is_public` | Added (Boolean, default False) | `9e51bc9` |
| Transaction | `trade_date` | Renamed from `executed_at` | Schema alignment |
| Transaction | `total_amount` | Added as required field | Schema alignment |
| Stock | `industry_id` | Changed from string to FK | Schema alignment |

---

## Middleware Stack

### Execution Order (Wave 5 Verified)

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
# Wave 5 Fix: TESTING mode bypass
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

### Integration Test Categories

| Category | Files | Purpose |
|----------|-------|---------|
| GDPR | `test_gdpr_data_lifecycle.py` | Data privacy compliance |
| Auth-Portfolio | `test_auth_to_portfolio_flow.py` | Authentication flows |
| Stock-Analysis | `test_stock_to_analysis_flow.py` | Analysis pipeline |
| Agents-Recommendations | `test_agents_to_recommendations_flow.py` | AI agent flows |
| Phase3 Integration | `test_phase3_integration.py` | End-to-end flows |

### Wave 5 Test Patterns

**Schema Validation Pattern:**
```python
# Always verify field names against unified_models.py
from backend.models.unified_models import Transaction, Stock, Fundamentals

# Use correct field names
transaction = Transaction(
    trade_date=datetime.utcnow(),      # NOT executed_at
    total_amount=Decimal("7505.00"),   # Required
    ...
)
```

**Async Fixture Pattern:**
```python
# Use sync MagicMock for cache fixtures
@pytest.fixture
def mock_cache():
    return MagicMock()  # NOT AsyncMock for Redis client
```

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
   |                       |-- Generate JWT           |
   |<-- JWT token ---------|                          |
   |                       |                          |
   |-- GET /api/* -------->|                          |
   |   (Authorization:     |                          |
   |    Bearer <token>)    |                          |
   |                       |-- Verify JWT             |
   |<-- Response ----------|                          |
```

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
| postgres | postgres:15 | 5432 | Database |
| redis | redis:7 | 6379 | Cache |
| celery_worker | backend | - | Background tasks |
| celery_beat | backend | - | Scheduled tasks |
| prometheus | prometheus | 9090 | Metrics |
| grafana | grafana | 3001 | Dashboards |

### CI/CD Runtime Versions (Standardized 2026-02-24)

| Runtime | Version | Notes |
|---------|---------|-------|
| Python | 3.12 (primary) | CI matrix also tests 3.9, 3.10, 3.11 |
| Node.js | 20 | Upgraded from 18 across 9 workflows |
| `actions/setup-python` | v5 | Upgraded from v4 |
| `actions/setup-node` | v4 | Upgraded from v3 |
| `actions/upload-artifact` | v4 | Upgraded from v3 |
| `github/codeql-action/*` | v3 | Upgraded from v2 in security-scan |

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Main API | `backend/api/main.py` |
| Unified Models | `backend/models/unified_models.py` |
| Router Registry | `backend/api/routers/__init__.py` |
| Settings | `backend/config/settings.py` |
| Security Config | `backend/security/security_config.py` |
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

---

**Wave 5 Commits:**
- `178a92e` - fix: Resolve double-prefix routing causing 404 errors
- `9e51bc9` - fix: Resolve integration test schema mismatches and fixture issues
- `f12c6b2` - fix: Skip rate limiting when TESTING=True

**Post-Wave 6 CI/CD Commits (2026-02-09 through 2026-02-24):**
- `d9b2d1c` - fix: Upgrade Node 18 to 20, Python 3.11 to 3.12, standardize action versions
- `786ffec` - fix(ci): Add TA-Lib C library to pipeline validation
- `4b8b168` - fix(ci): Add TA-Lib C library to security scan
- `ca91ccd` - fix(ci): Add missing SECRET_KEY/JWT_SECRET_KEY to pipeline validation
- `4812e1f` - fix(ci): Fix pipeline validation db init and security scan issues
- `9f22b30` - fix(ci): Make ETL validation resilient to import errors
- `7a39cb9` - fix(ci): Make Semgrep non-blocking
- `4bac2fc` - fix(ci): Make GitLeaks non-blocking
- `4cd4620` - fix: Add missing StockData and ExtractionResult dataclasses to ETL
