# Investment Analysis Platform - Codebase Architecture Map

**Last Updated:** 2026-02-08
**Version:** 2.1.0
**Source of Truth:** Generated from code inspection of `backend/api/main.py`, `backend/security/security_config.py`, `backend/config/database.py`, and directory listings. Updated post-30-agent Queen Audit.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Backend Architecture](#backend-architecture)
3. [Frontend Architecture](#frontend-architecture)
4. [Integration Points](#integration-points)
5. [Security Layer](#security-layer)
6. [Database Schema](#database-schema)
7. [Architectural Patterns](#architectural-patterns)
8. [Module Dependencies](#module-dependencies)

---

## Executive Summary

The Investment Analysis Platform is a full-stack financial analytics application:
- **Backend**: FastAPI (Python) with async PostgreSQL via SQLAlchemy + asyncpg, plus a legacy synchronous SQLAlchemy layer
- **Frontend**: React + TypeScript with Redux Toolkit state management and Material UI
- **Architecture**: Layered domain-driven design with an 11-layer security middleware stack, timezone-aware UTC throughout
- **Key Features**: Real-time stock analysis, ML predictions, portfolio management, investment thesis generation, background task scheduling, GDPR/SEC compliance, Kafka streaming

---

## Backend Architecture

### 1. API Layer (`backend/api/`)

#### Application Entry Point (`backend/api/main.py`)

The FastAPI app is created with a `lifespan` context manager that initializes, in order:
1. Async database (`backend/config/database.py` -- `initialize_database()`)
2. Legacy synchronous database (`backend/utils/database.py` -- `init_db()`)
3. Basic cache, then comprehensive cache manager, intelligent caching, cache monitoring, and cache invalidation triggers
4. Background task scheduler (`backend/tasks/scheduler.py`)
5. WebSocket cleanup task
6. ML model manager (optional; logs warning if unavailable)

Additional middleware added after security stack:
- `PrometheusMiddleware` -- Prometheus metrics collection
- `CacheControlMiddleware` -- HTTP cache-control headers (excludes `/api/v1/auth/`, `/api/v1/admin/`, `/api/v1/ws/`, `/api/v1/metrics`)
- `V1DeprecationMiddleware` -- V1 API deprecation warnings (disabled during testing)

Standardized error handlers are registered via `backend/middleware/error_handler.py`.

#### Router Files on Disk

```
backend/api/routers/
├── admin.py              # Admin operations
├── agents.py             # Trading agents
├── analysis.py           # Technical/fundamental analysis
├── auth.py               # Authentication & JWT management
├── cache_management.py   # Cache admin endpoints
├── gdpr.py               # GDPR data subject rights
├── health.py             # Health checks
├── monitoring.py         # System metrics (NOT mounted in main.py)
├── news.py               # News endpoints
├── portfolio.py          # Portfolio management
├── recommendations.py    # ML-powered recommendations
├── settings.py           # User settings
├── stocks.py             # Stock data & market info
├── stocks_legacy.py      # Legacy stock endpoints (NOT mounted in main.py)
├── thesis.py             # Investment thesis generation
├── watchlist.py          # User watchlists
└── websocket.py          # Real-time WebSocket updates
```

#### 17 Routers Mounted in `main.py`

The following 16 routers from `backend/api/routers/` plus 1 from `backend/api/versioning.py` are actually included. All API routers (except health) use the `/api/v1/` versioned prefix:

| # | Router | Prefix (as mounted) | Tags | Notes |
|---|--------|---------------------|------|-------|
| 1 | `health.router` | `/api/health` | health | Outside versioned prefix |
| 2 | `auth.router` | `/api/v1/auth` | authentication | |
| 3 | `stocks.router` | `/api/v1/stocks` | stocks | |
| 4 | `analysis.router` | `/api/v1/analysis` | analysis | |
| 5 | `recommendations.router` | `/api/v1/recommendations` | recommendations | |
| 6 | `portfolio.router` | `/api/v1/portfolio` | portfolio | |
| 7 | `websocket.router` | `/api/v1/ws` | websocket | |
| 8 | `admin.router` | `/api/v1/admin` | admin | |
| 9 | `agents.router` | `/api/v1/agents` | agents | |
| 10 | `cache_management.router` | `/api/v1/cache` | cache | |
| 11 | `gdpr.router` | `/api/v1` | gdpr | Router has no internal prefix |
| 12 | `watchlist.router` | `/api/v1/watchlists` | watchlists | |
| 13 | `thesis.router` | `/api/v1/thesis` | investment-thesis | |
| 14 | `news.router` | `/api/v1/news` | news | |
| 15 | `settings_router.router` | `/api/v1/settings` | settings | Imported as `settings_router` to avoid config clash |
| 16 | `v1_migration_router` | `/api/v1/admin/v1-migration` | v1-migration, admin | Self-prefixed, from `backend/api/versioning.py` |
| 17 | Root endpoint | `/` | -- | Returns API status JSON |

Additional non-router endpoint: `GET /api/v1/metrics` defined directly on the app, returns Prometheus metrics.

**Not mounted:** `monitoring.py` (defines its own prefix `/api/monitoring` internally but is not imported or included in `main.py`). Also not mounted: `stocks_legacy.py`.

### 2. Security Middleware Stack (`backend/security/`)

#### Middleware Execution Order

Defined in `add_comprehensive_security_middleware()` in `security_config.py`. In non-testing mode, the full stack is:

```
Request Flow (order of add_middleware calls):
 1. AuditMiddleware              -- Log all requests (skipped in testing)
 2. SecurityHeadersMiddleware    -- CSP, HSTS, X-Frame-Options, etc.
 3. RateLimitingMiddleware       -- Redis-backed rate limiting (skipped in testing)
 4. ValidationMiddleware         -- Input validation & sanitization (skipped in testing)
 5. InjectionPreventionMiddleware -- SQL/XSS/CSRF protection (skipped in testing)
 6. HTTPSRedirectMiddleware      -- Force HTTPS (production only, when FORCE_HTTPS=true)
 7. TrustedHostMiddleware        -- Allowed hosts whitelist
 8. GZipMiddleware               -- Response compression, min 1000 bytes (skipped in testing)
 9. CORSMiddleware               -- Cross-origin resource sharing
10. SessionMiddleware            -- Session cookies (skipped in testing)
11. IP Filter Middleware         -- Inline @app.middleware, IP allowlist/blocklist
```

Note: Due to Starlette's middleware stack semantics, the actual request processing order is reversed -- item 11 (IP Filter) runs first on the request, and item 1 (Audit) runs last before the route handler. Responses traverse in the opposite direction.

Several middleware layers are conditionally disabled during testing (when `TESTING=true` env var is set) to avoid `AsyncClient` compatibility issues.

After the security stack, `main.py` adds:
- `PrometheusMiddleware`
- `CacheControlMiddleware`
- `V1DeprecationMiddleware` (disabled in testing)

#### Security Module Files

```
backend/security/
├── advanced_rate_limiter.py     # Redis-backed rate limiting with rules
├── audit_logging.py             # SEC-compliant audit trail (2555-day retention)
├── code_analyzer.py             # Static code analysis
├── crypto_utils.py              # Cryptographic utilities
├── csrf_protection.py           # Token-based CSRF prevention
├── data_encryption.py           # Data encryption at rest
├── database_security.py         # Database security utilities
├── enhanced_auth.py             # Enhanced authentication
├── injection_prevention.py      # SQL/XSS/command injection prevention
├── input_validation.py          # Request validation middleware
├── jwt_manager.py               # RS256 JWT signing with key pairs
├── password_manager.py          # Password hashing and policy
├── rate_limiter.py              # Basic rate limiter (superseded by advanced)
├── rbac.py                      # Role-based access control
├── secrets_manager.py           # Encrypted secrets storage
├── secrets_vault.py             # Secrets vault
├── security_config.py           # Central config + middleware assembly
├── security_headers.py          # Security headers middleware
├── session_manager.py           # Session management
├── sql_injection_prevention.py  # SQL injection patterns
├── vulnerability_scanner.py     # Vulnerability scanning
└── websocket_security.py        # WebSocket security
```

#### JWT Configuration (Single Source of Truth in `SecurityConfig`)

```python
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")     # Primary
JWT_ALGORITHM_FALLBACK = "HS256"                         # Legacy compatibility
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30                     # Access token TTL
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7                        # Refresh token TTL
JWT_MFA_TOKEN_EXPIRE_MINUTES = 5                         # MFA verification TTL
JWT_RESET_TOKEN_EXPIRE_MINUTES = 15                      # Password reset TTL
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", <generated>) # HS256 fallback key
JWT_ISSUER = "investment-analysis-app"
JWT_AUDIENCE = "investment-analysis-users"
```

#### Rate Limiting

Default rate limit: `100/hour`. Strict rate limit: `10/minute`.

Redis connectivity is validated at startup with exponential backoff retry (3 attempts, 1s/2s/4s delays). In production, Redis failure raises `RedisHealthCheckError`. In development/testing, falls back to in-memory storage.

#### File Upload Security

- Max size: 10 MB
- Extension allowlist: `.csv`, `.json`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.txt`, `.xls`, `.xlsx`
- MIME type validation via magic bytes for binary files
- Content structure validation for text files (JSON parse check, CSV delimiter check)
- Malware pattern scanning (script tags, executable signatures, double-extension attacks)

### 3. Database Layer

#### Async Database (`backend/config/database.py`)

`AsyncDatabaseManager` class provides:
- SQLAlchemy async engine with `asyncpg` driver for PostgreSQL
- Connection pooling: `AsyncAdaptedQueuePool` (production/dev) or `NullPool` (testing)
- Pool sizes: 50/100 (production), 20/40 (development), 5/10 (testing)
- Prepared statement cache: 100 statements per connection
- SQLAlchemy query cache: 1000 entries
- Transaction isolation: `READ COMMITTED` (PostgreSQL), `SERIALIZABLE` (SQLite fallback)
- Event listeners for connection monitoring (connect, checkout, checkin, invalidate)
- Retry logic with exponential backoff for transient failures (serialization errors, deadlocks, disconnections)
- Bulk insert with conflict handling (ignore/update/error strategies)
- Health check endpoint querying `pg_prepared_statements` and `information_schema`

FastAPI dependency: `get_async_db_session()` yields an `AsyncSession`.

#### Legacy Synchronous Database (`backend/utils/database.py`)

Deprecated synchronous SQLAlchemy engine kept for backward compatibility:
- `QueuePool` with pool_size=10, max_overflow=20
- Emits `DeprecationWarning` when legacy functions are called
- Re-exports async functions from `backend/config/database.py` for migration convenience

Both are initialized during application startup (async first, then legacy).

#### Models

```
backend/models/
├── api_response.py          # Pydantic response wrappers
├── consolidated_models.py   # SQLAlchemy ORM models (main)
├── database.py              # Database model utilities
├── ml_models.py             # ML-specific models
├── monitoring_schemas.py    # Monitoring Pydantic schemas
├── schemas.py               # Request/response schemas
├── tables.py                # Additional table definitions
├── thesis.py                # Investment thesis models
└── unified_models.py        # Unified Base for all models
```

### 4. Domain Contracts (`backend/domain/contracts/`)

```
backend/domain/contracts/
├── base.py                         # DomainContract ABC + ContractResult
├── data_pipeline_contract.py       # ETL and data ingestion contract
├── investment_analysis_contract.py # Analysis & recommendations contract
├── market_data_contract.py         # Stock price & market data contract
├── ml_contract.py                  # Machine learning predictions contract
└── portfolio_contract.py           # Portfolio management contract
```

### 5. Repositories (`backend/repositories/`)

```
backend/repositories/
├── base.py                     # Base repository class
├── portfolio_repository.py     # Portfolio CRUD
├── price_repository.py         # Price history queries
├── recommendation_repository.py # Recommendation queries
├── stock_repository.py         # Stock data queries
├── thesis_repository.py        # Investment thesis CRUD
├── user_repository.py          # User account CRUD
└── watchlist_repository.py     # Watchlist CRUD
```

### 6. ETL & Data Ingestion

#### External Data Clients (`backend/data_ingestion/`)

```
backend/data_ingestion/
├── alpha_vantage_client.py    # Company overviews, historical prices
├── base_client.py             # Base HTTP client with retry logic
├── finnhub_client.py          # Real-time quotes, company profiles
├── market_scanner.py          # Market-wide scanning
├── polygon_client.py          # Market data, aggregates
├── robust_api_client.py       # Resilient API client wrapper
├── sec_edgar_client.py        # SEC filings, fundamentals
├── smart_data_fetcher.py      # Intelligent data routing
└── stock_tiers.json           # Stock tier classifications
```

#### ETL Pipeline (`backend/etl/`)

```
backend/etl/
├── concurrent_processor.py              # Async concurrent processing
├── data_extractor.py                    # Primary data extractor
├── data_extractor_original_backup.py    # Original extractor backup
├── data_extractor_unlimited.py          # Unlimited extraction variant
├── data_loader.py                       # Bulk insert with conflict handling
├── data_transformer.py                  # Normalization & derived fields
├── data_validation_pipeline.py          # Data quality pipeline
├── data_validator.py                    # Data integrity checks
├── distributed_batch_processor.py       # Distributed batch processing
├── etl_orchestrator.py                  # Pipeline coordination
├── intelligent_cache_system.py          # ETL-level caching
├── multi_source_extractor.py            # Multi-provider extraction
├── rate_limiting.py                     # API rate limit management
├── simple_unlimited_extractor.py        # Simplified unlimited extractor
├── stock_universe_manager.py            # Stock universe maintenance
├── unlimited_data_extractor.py          # Full universe extractor
├── unlimited_extractor_with_fallbacks.py # Extractor with provider fallbacks
└── web_scrapers.py                      # Web scraping utilities
```

### 7. ML & Analytics

#### ML Components (`backend/ml/`)

```
backend/ml/
├── backtesting.py               # Strategy backtesting engine
├── cost_monitoring.py           # ML operation cost tracking
├── dataset_hub.py               # Dataset management
├── feature_store.py             # Feature engineering & storage
├── gpu_utils.py                 # GPU detection & utilities
├── hf_hub_client.py             # HuggingFace Hub integration
├── minimal_training.py          # Lightweight training pipeline
├── ml_api_server.py             # Standalone ML API server
├── ml_monitoring_server.py      # Prometheus ML metrics
├── ml_tables.py                 # ML database tables
├── model_manager.py             # Model lifecycle management
├── model_monitoring.py          # Model performance tracking
├── model_versioning.py          # Model version control
├── online_learning.py           # Online/incremental learning
├── pipeline_optimization.py     # Pipeline performance tuning
├── simple_training_pipeline.py  # Simple training workflow
├── training_pipeline.py         # Full training pipeline
├── data_prep/                   # Data preparation utilities
├── models/ensemble/             # Ensemble model implementations
├── pipeline/                    # ML pipeline components
└── training/                    # Training utilities
```

#### Analytics Engines (`backend/analytics/`)

```
backend/analytics/
├── dividend_analyzer.py                    # Dividend analysis
├── finbert_analyzer.py                     # FinBERT NLP sentiment
├── fundamental_analysis.py                 # Fundamental metrics
├── recommendation_engine.py                # Base recommendation engine
├── recommendation_engine_optimized.py      # Optimized recommendation engine
├── sentiment_analysis.py                   # Sentiment scoring
├── technical_analysis.py                   # Technical indicators
├── agents/                                 # Agentic analysis
│   ├── cache_aware_agents.py              # Cache-integrated agents
│   ├── enhancement_levels.py              # Analysis enhancement tiers
│   ├── hybrid_engine.py                   # Multi-strategy engine
│   └── selective_orchestrator.py          # Selective agent dispatch
├── fundamental/                            # Fundamental analysis modules
├── portfolio/                              # Portfolio analytics
├── risk/                                   # Risk analysis modules
└── statistical/                            # Statistical analysis
```

### 8. Background Tasks (`backend/tasks/`)

```
backend/tasks/
├── analysis_tasks.py          # Scheduled analysis jobs
├── celery_app.py              # Celery configuration
├── data_pipeline.py           # Data pipeline tasks
├── data_tasks.py              # Data maintenance tasks
├── maintenance_tasks.py       # System maintenance
├── notification_tasks.py      # Alert/notification dispatch
├── portfolio_tasks.py         # Portfolio rebalancing tasks
├── scheduler.py               # APScheduler-based task scheduler
└── stock_universe_fetcher.py  # Stock universe update task
```

### 9. Middleware (`backend/middleware/`)

```
backend/middleware/
├── error_handler.py          # Standardized exception handlers
├── request_size_limiter.py   # Request body size limits
└── security_headers.py       # Additional security headers
```

### 10. Services (`backend/services/`)

```
backend/services/
└── realtime_price_service.py  # Real-time price data service
```

### 11. Auth (`backend/auth/`)

```
backend/auth/
├── oauth2.py               # OAuth2 password flow & token handling
└── password_validator.py    # Password strength validation
```

### 12. Compliance (`backend/compliance/`)

```
backend/compliance/
├── gdpr.py                 # GDPR data subject request handling
└── sec.py                  # SEC regulatory compliance
```

### 13. Streaming (`backend/streaming/`)

```
backend/streaming/
└── kafka_client.py         # Kafka event streaming client
```

### 14. Monitoring (`backend/monitoring/`)

```
backend/monitoring/
├── alerting_system.py          # Alert rule management
├── alertmanager_webhook.py     # Alertmanager webhook handler
├── api_performance.py          # API performance tracking
├── application_monitoring.py   # Application-level metrics
├── auto_scaler.py              # Auto-scaling logic
├── data_quality_dashboard.py   # Data quality visualization
├── data_quality_metrics.py     # Data quality metric collection
├── database_performance.py     # Database performance monitoring
├── financial_monitoring.py     # Financial metric tracking
├── health_checks.py            # Health check definitions
├── health_system.py            # Health system coordination
├── log_analysis.py             # Log analysis utilities
├── metrics_collector.py        # Prometheus metrics collector
├── real_time_alerts.py         # Real-time alert processing
└── sla_tracker.py              # SLA compliance tracking
```

### 15. Test Suite (`backend/tests/`)

```
backend/tests/
├── conftest.py                                # Shared test configuration & fixtures
├── async_fixtures.py                          # Async test fixtures
├── fixtures/
│   ├── comprehensive_mock_fixtures.py         # Comprehensive mock data
│   ├── integration_test_fixtures.py           # Integration test fixtures
│   ├── market_data_fixtures.py                # Market data fixtures
│   └── mock_api_fixtures.py                   # Mock API response fixtures
├── integration/
│   ├── test_agents_to_recommendations_flow.py # Agent -> recommendation pipeline
│   ├── test_analysis_router.py                # Analysis router endpoints
│   ├── test_auth_flow_complete.py             # Complete auth flow (login/register/refresh)
│   ├── test_auth_to_portfolio_flow.py         # Auth -> portfolio access flow
│   ├── test_domain_contracts.py               # Domain contract validation
│   ├── test_gdpr_data_lifecycle.py            # GDPR data lifecycle flow
│   ├── test_health_router.py                  # Health endpoint tests
│   ├── test_news_router.py                    # News router endpoints
│   ├── test_phase3_integration.py             # Phase 3 integration tests
│   ├── test_recommendations_router.py         # Recommendations router endpoints
│   ├── test_settings_router.py                # Settings router endpoints
│   ├── test_stock_to_analysis_flow.py         # Stock -> analysis pipeline
│   ├── test_stocks_router.py                  # Stocks router endpoints
│   └── test_websocket_router.py               # WebSocket router tests
├── middleware/
│   ├── test_request_size_limiter.py           # Request size limit tests
│   └── test_security_headers.py               # Security header tests
├── security/
│   ├── test_csrf_auth_integration.py          # CSRF + auth integration tests
│   ├── test_csrf_protection.py                # CSRF protection unit tests
│   ├── test_rate_limiter.py                   # Rate limiter tests
│   └── test_security_modules.py               # Security module tests
├── unit/                                      # Unit test directory
├── test_admin_api.py                          # Admin API tests
├── test_agents_api.py                         # Agents API tests
├── test_api_integration.py                    # API integration tests
├── test_bloom_filter.py                       # Bloom filter tests
├── test_cache_decorator.py                    # Cache decorator tests
├── test_cache_management_api.py               # Cache management API tests
├── test_circuit_breaker.py                    # Circuit breaker tests
├── test_cointegration.py                      # Cointegration analysis tests
├── test_comprehensive_units.py                # Comprehensive unit tests
├── test_data_pipeline_integration.py          # Data pipeline integration tests
├── test_data_quality.py                       # Data quality tests
├── test_database_integration.py               # Database integration tests
├── test_dividend_analyzer.py                  # Dividend analyzer tests
├── test_error_scenarios.py                    # Error scenario tests
├── test_financial_model_validation.py         # Financial model validation
├── test_gdpr_api.py                           # GDPR API tests
├── test_integration.py                        # General integration tests
├── test_integration_comprehensive.py          # Comprehensive integration tests
├── test_ml_performance.py                     # ML performance tests
├── test_ml_pipeline.py                        # ML pipeline tests
├── test_monitoring_api.py                     # Monitoring API tests
├── test_n1_query_fix.py                       # N+1 query fix tests
├── test_performance_load.py                   # Performance load tests
├── test_performance_optimizations.py          # Performance optimization tests
├── test_portfolio_optimizer.py                # Portfolio optimizer tests
├── test_rate_limiting.py                      # Rate limiting tests
├── test_recommendation_engine.py              # Recommendation engine tests
├── test_resilience_integration.py             # Resilience integration tests
├── test_risk_manager.py                       # Risk manager tests
├── test_security_compliance.py                # Security compliance tests
├── test_security_integration.py               # Security integration tests
├── test_simple_async.py                       # Simple async tests
├── test_thesis_api.py                         # Thesis API tests
├── test_watchlist.py                          # Watchlist tests
├── test_websocket_integration.py              # WebSocket integration tests
├── benchmark_n1_query_fix.py                  # N+1 query benchmark
└── locustfile.py                              # Locust load testing config
```

---

## Frontend Architecture

### 1. Application Structure

```
frontend/web/src/
├── App.tsx                     # Root component with routing
├── index.tsx                   # Entry point
├── components/                 # Reusable UI components
│   ├── Layout/                 # App shell with sidebar & header
│   ├── cards/                  # RecommendationCard, NewsCard, etc.
│   ├── charts/                 # StockChart, MarketHeatmap, etc.
│   ├── common/                 # Shared utility components
│   ├── dashboard/              # Dashboard-specific components
│   ├── monitoring/             # System monitoring widgets
│   ├── panels/                 # Panel components
│   ├── NotificationPanel/      # Real-time alerts & updates
│   ├── SearchModal/            # Stock search with autocomplete
│   ├── WebSocketIndicator/     # Connection status indicator
│   ├── CorrelationMatrix.tsx   # Asset correlation visualization
│   ├── EfficientFrontier.tsx   # Portfolio optimization chart
│   ├── EnhancedDashboard.tsx   # Enhanced dashboard view
│   └── RiskDecomposition.tsx   # Risk breakdown visualization
├── pages/                      # Route components
│   ├── Analysis.tsx
│   ├── Alerts.tsx
│   ├── Dashboard.tsx
│   ├── Help.tsx
│   ├── InvestmentThesis.tsx
│   ├── Login.tsx
│   ├── MarketOverview.tsx
│   ├── Portfolio.tsx
│   ├── Recommendations.tsx
│   ├── Reports.tsx
│   ├── Settings.tsx
│   └── Watchlist.tsx
├── services/                   # API integration
│   └── api.service.ts
├── store/                      # Redux Toolkit state management
│   ├── index.ts
│   └── slices/
│       ├── appSlice.ts
│       ├── dashboardSlice.ts
│       ├── marketSlice.ts
│       ├── portfolioSlice.ts
│       ├── recommendationsSlice.ts
│       └── stockSlice.ts
├── hooks/                      # Custom React hooks
├── config/                     # App configuration
├── design/                     # Design tokens/specs
├── styles/                     # CSS/styling
├── theme/                      # MUI theming
├── types/                      # TypeScript type definitions
└── utils/                      # Utilities (accessibility, env, etc.)
```

### 2. State Management (Redux Toolkit)

Six slices manage application state:
- `appSlice` -- Global app state (auth, navigation, notifications)
- `dashboardSlice` -- Dashboard aggregates and summary data
- `stockSlice` -- Selected stock, quotes, chart data, search results
- `portfolioSlice` -- Portfolio positions, transactions, performance
- `recommendationsSlice` -- ML-generated recommendations
- `marketSlice` -- Market overview, sector performance

### 3. Pages

12 page components correspond to the main application routes:

| Page | Purpose |
|------|---------|
| Dashboard | Main overview with portfolio summary and market data |
| Analysis | Technical and fundamental analysis tools |
| Portfolio | Position tracking and management |
| Recommendations | ML-generated investment recommendations |
| MarketOverview | Broad market data and sector performance |
| Watchlist | User stock watchlists |
| InvestmentThesis | AI-generated investment thesis |
| Alerts | Price and event alerts |
| Reports | Generated reports and exports |
| Settings | User preferences |
| Login | Authentication |
| Help | Documentation and support |

Tests exist for `Dashboard` and `Portfolio` pages (`Dashboard.test.tsx`, `Portfolio.test.tsx`).

---

## Integration Points

### 1. Frontend to Backend API Flow

```
React Component
    | dispatch(asyncThunk)
Redux Thunk (in slice)
    | api.service.ts call
Axios HTTP Client
    | JWT Bearer token injected by request interceptor
    | Token refresh on 401 via response interceptor
FastAPI Backend
    | Security middleware stack (11 layers)
    | Router handler
    | Repository / Service layer
    | AsyncSession (SQLAlchemy + asyncpg)
PostgreSQL Database
    | Response
JSON serialization
    | Redux store update
    | Component re-render
```

### 2. WebSocket Connection

```
/api/v1/ws/stocks/{ticker}
    | Price updates streamed to client
    | WebSocketIndicator component shows connection status
    | Cleanup task runs periodically to prune stale connections
```

### 3. Data Caching Strategy

Multi-tier caching implemented via `backend/utils/comprehensive_cache.py` and `backend/utils/intelligent_cache_policies.py`:

```
L1 Cache (60 seconds)   -- Real-time quotes, active trades
L2 Cache (5 minutes)    -- Technical indicators, intraday data
L3 Cache (30 minutes)   -- Fundamental data, company info
Database Cache (1 day)  -- Historical prices, SEC filings
```

Cache monitoring dashboard available via `backend/utils/cache_monitoring.py`. Database-level cache invalidation triggers set up at startup.

### 4. API Versioning

`backend/api/versioning.py` provides:
- `V1DeprecationMiddleware` -- Detects V1 requests (via URL path, header, or query param), adds deprecation warnings, tracks usage, supports optional auto-redirect to V2
- `v1_migration_router` -- Admin endpoints at `/api/v1/admin/v1-migration/` for monitoring migration progress (`/metrics`, `/clients`, `/endpoint-mapping`, `/version-info`)

---

## Security Layer

### 1. Security Headers (from `SecurityConfig.SECURITY_HEADERS`)

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' ws: wss:
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()
```

### 2. CORS Configuration (from `SecurityConfig`)

**Always allowed:** `http://localhost:3000`, `https://investment-analysis.com`, `https://api.investment-analysis.com`

**Development additions:** `http://localhost:8000`, `http://127.0.0.1:3000`, `http://127.0.0.1:8000`, `http://localhost:3001`, `http://127.0.0.1:3001`

**Production:** Only `https://` origins are kept.

Allowed Methods: `GET, POST, PUT, DELETE, OPTIONS`
Allowed Headers: `Authorization, Content-Type, X-Requested-With, X-API-Key`
Exposed Headers: `X-RateLimit-Remaining, X-RateLimit-Reset, X-Request-ID`
Credentials: Enabled

### 3. Trusted Hosts

`localhost`, `127.0.0.1`, `testserver`, `investment-analysis.com`, `api.investment-analysis.com`

### 4. Session Configuration

- Secret key: env var `SESSION_SECRET_KEY` or generated `token_urlsafe(32)`
- Max age: 3600 seconds (1 hour)
- SameSite: `strict` in production, `lax` otherwise
- HTTPS-only: Only when `FORCE_HTTPS=true` and in production

### 5. Password Policy (from `SecurityConfig`)

- Minimum length: 12 characters
- Required: uppercase, lowercase, digits, special characters
- Max age: 90 days

### 6. Redis Health Check

`RedisHealthChecker` class with:
- Exponential backoff retry: 3 attempts at 1s, 2s, 4s delays
- Connection timeout: 5 seconds
- URL masking for safe logging
- Latency measurement
- Redis version detection

---

## Database Schema

### Core Tables (from `consolidated_models.py`)

```sql
-- Reference Data
exchanges (id, code, name, timezone, country, currency)
sectors (id, name, description)
industries (id, name, sector_id)

-- Stock Data
stocks (id, ticker, name, exchange_id, sector_id, market_cap, is_active)
price_history (id, stock_id, date, open, high, low, close, volume, adjusted_close)
technical_indicators (id, stock_id, date, rsi_14, macd, sma_20, bollinger_upper, atr_14)
fundamentals (id, stock_id, period_date, revenue, net_income, eps, pe_ratio, debt_to_equity)

-- Analysis & ML
news_sentiment (id, stock_id, headline, sentiment_score, published_at, source)
ml_predictions (id, stock_id, model_name, predicted_price, confidence, target_date)
recommendations (id, stock_id, action, confidence, target_price, reasoning, created_at)

-- Monitoring
api_usage (id, provider, endpoint, calls_count, estimated_cost, timestamp)
cost_metrics (id, date, provider, api_calls, estimated_cost)

-- Users & Portfolios
users (id, email, hashed_password, full_name, role, created_at)
portfolios (id, user_id, name, cash_balance, strategy, created_at)
positions (id, portfolio_id, stock_id, quantity, average_cost, realized_gain)
transactions (id, portfolio_id, stock_id, type, quantity, price, timestamp)
```

### Entity Relationships

```
Exchange --+-- Stock --+-- PriceHistory
           |           +-- TechnicalIndicators
Sector ----+           +-- Fundamentals
           |           +-- NewsSentiment
Industry --+           +-- MLPrediction
                       +-- Recommendation

User --+-- Portfolio --+-- Position --+-- Stock
       |               +-- Transaction
       +-- Watchlist
```

---

## Architectural Patterns

### 1. Repository Pattern

Seven repositories abstract data access:
- `StockRepository`, `PriceRepository`, `PortfolioRepository`
- `RecommendationRepository`, `UserRepository`, `WatchlistRepository`, `ThesisRepository`

All use async sessions from `AsyncDatabaseManager`.

### 2. Domain Contracts

Five contracts enforce cross-domain boundaries:
- `MarketDataContract`, `PortfolioContract`, `DataPipelineContract`, `MLContract`, `InvestmentAnalysisContract`

Each returns `ContractResult<T>` with structured error handling.

### 3. Dual Database Strategy

- **Primary (async):** `AsyncDatabaseManager` in `backend/config/database.py` using `sqlalchemy[asyncio]` + `asyncpg`
- **Legacy (sync):** Synchronous engine in `backend/utils/database.py` using `QueuePool`, deprecated but retained for backward compatibility

Both initialize during app startup. Migration path documented in `backend/utils/database.py` docstring.

### 4. Middleware Chain

Security middleware is assembled in `add_comprehensive_security_middleware()` as a single composable function. Individual middleware components are independently testable. Several are conditionally disabled during testing.

### 5. Background Task Scheduling

`backend/tasks/scheduler.py` manages periodic tasks. A separate Celery configuration exists in `backend/tasks/celery_app.py` for distributed task execution. Task categories: analysis, data pipeline, maintenance, notifications, portfolio rebalancing, stock universe fetching.

---

## Module Dependencies

### Backend Dependency Graph

```
FastAPI Application (main.py)
+-- api/routers/ (16 routers, all at /api/v1/ except health at /api/health)
|   +-- auth.py --> security/jwt_manager.py, security/security_config.py, models/tables.py
|   +-- stocks.py --> repositories/stock_repository.py, data_ingestion/, utils/api_cache_decorators.py
|   +-- recommendations.py --> analytics/recommendation_engine_optimized.py, repositories/
|   +-- portfolio.py --> repositories/portfolio_repository.py, services/realtime_price_service.py
|   +-- agents.py --> analytics/agents/
|   +-- thesis.py --> repositories/thesis_repository.py
|   +-- news.py, settings.py, watchlist.py, gdpr.py, admin.py, health.py, cache_management.py
|   +-- websocket.py (WebSocket connections)
|
+-- security/ (middleware + utilities, 22 modules)
|   +-- security_config.py (assembles middleware stack)
|   +-- advanced_rate_limiter.py --> redis
|   +-- audit_logging.py --> database
|   +-- jwt_manager.py, csrf_protection.py, input_validation.py, injection_prevention.py
|
+-- auth/ (OAuth2 flow, password validation)
+-- compliance/ (GDPR, SEC regulatory)
+-- streaming/ (Kafka event streaming)
+-- monitoring/ (15 modules: alerts, health, SLA, metrics, auto-scaling)
|
+-- config/
|   +-- database.py (AsyncDatabaseManager, primary)
|   +-- settings.py (environment configuration)
|   +-- monitoring_config.py (monitoring configuration)
|
+-- utils/database.py (legacy sync engine, deprecated)
|
+-- domain/contracts/ (5 contracts)
+-- repositories/ (7 repositories) --> models/
+-- models/ (SQLAlchemy ORM + Pydantic schemas)
+-- analytics/ (engines + agents)
+-- ml/ (training, monitoring, inference)
+-- etl/ (extraction, transformation, loading)
+-- data_ingestion/ (external API clients)
+-- tasks/ (scheduler + Celery tasks)
+-- middleware/ (error handling, request limits)
+-- services/ (realtime price service)
```

### Frontend Dependency Graph

```
React Application (App.tsx)
+-- store/index.ts (Redux store, 6 slices)
+-- pages/ (12 page components)
+-- components/ (Layout, charts, cards, panels, common, monitoring)
+-- services/api.service.ts (Axios client with interceptors)
+-- hooks/ (custom React hooks)
+-- theme/ (MUI theming)
+-- types/ (TypeScript definitions)
+-- config/, design/, styles/, utils/
```

---

## Key Architectural Decisions

1. **Async-first database access** -- Financial apps are I/O bound; async SQLAlchemy + asyncpg gives high concurrency without threads.
2. **Dual database engines** -- Legacy sync engine retained to avoid breaking existing code during migration; deprecated with warnings.
3. **Repository pattern** -- Centralizes SQL, enables testing via mocks, allows future database swaps.
4. **Domain contracts** -- Explicit boundaries between market data, portfolio, ML, and analysis domains.
5. **Redis for rate limiting** -- Distributed, atomic, sub-millisecond operations; in-memory fallback for development.
6. **RS256 JWT** -- Asymmetric keys prevent token forgery; HS256 fallback for legacy compatibility.
7. **Conditional middleware in testing** -- Several middleware layers skip during tests to avoid `AsyncClient` stream compatibility issues.

---

**Document Version:** 2.1.0
**Last Updated:** 2026-02-08
**Generated From:** Code inspection of actual source files, verified post-Queen Audit
