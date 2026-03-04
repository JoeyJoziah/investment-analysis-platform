# Backend Architecture Codemap

**Last Updated:** 2026-03-04

## API Routers (`backend/api/routers/`)

19 router files (excluding `__init__.py`), 153+ endpoints total.

| Router | File | Purpose |
|--------|------|---------|
| admin | `admin.py` | Admin operations (16 endpoints) |
| agents | `agents.py` | AI agent management (7 endpoints) |
| analysis | `analysis.py` | Stock analysis endpoints (5 endpoints) |
| auth | `auth.py` | Authentication OAuth2/JWT RS256 (6 endpoints) |
| cache_management | `cache_management.py` | Cache control (8 endpoints) |
| gdpr | `gdpr.py` | GDPR compliance export/delete (13 endpoints) |
| health | `health.py` | Health check endpoints (7 endpoints) |
| ml | `ml.py` | ML predictions and model management (8 endpoints) |
| monitoring | `monitoring.py` | Metrics and monitoring (5 endpoints) |
| news | `news.py` | News and sentiment (4 endpoints) |
| portfolio | `portfolio.py` | Portfolio management (10 endpoints) |
| recommendations | `recommendations.py` | AI recommendations (8 endpoints) |
| settings | `settings.py` | User preferences (10 endpoints) |
| stocks | `stocks.py` | Stock CRUD operations (12 endpoints) |
| thesis | `thesis.py` | Investment thesis (6 endpoints) |
| trading | `trading.py` | Order validation and execution (3 endpoints) |
| watchlist | `watchlist.py` | Watchlist operations (13 endpoints) |
| websocket | `websocket.py` | Real-time WebSocket (3 HTTP + 3 WS) |

### Routing Pattern

```python
# In router file (e.g., auth.py):
router = APIRouter(tags=["authentication"])  # NO prefix here

# In main.py:
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])  # Prefix here only
```

## Services (`backend/services/`)

20 service files (10,241 total lines) providing the business logic layer
between routers and repositories.

| Service | Purpose |
|---------|---------|
| `analysis_service.py` | Stock analysis orchestration |
| `admin_service.py` | Administrative operations |
| `agents_service.py` | AI agent coordination |
| `gdpr_service.py` | Data export, deletion, anonymization |
| `market_data_service.py` | Market data aggregation |
| `news_service.py` | News collection and sentiment |
| `portfolio_helpers.py` | Portfolio calculation helpers |
| `portfolio_rebalancing.py` | Rebalancing logic |
| `portfolio_service.py` | Portfolio CRUD and analytics (1,162 lines) |
| `realtime_price_service.py` | Real-time price streaming |
| `recommendation_analysis.py` | Recommendation scoring |
| `recommendation_crud.py` | Recommendation CRUD |
| `recommendation_service.py` | Full recommendation engine (1,234 lines) |
| `settings_service.py` | User settings management |
| `socketio_service.py` | Socket.IO real-time layer |
| `stocks_service.py` | Stock data operations |
| `trading_service.py` | Order validation and execution |
| `watchlist_service.py` | Watchlist management |
| `websocket_service.py` | WebSocket connection management |

## Key Code Paths

### Cache Decorator
**File**: `backend/utils/cache.py:205-300`
```
get_redis()          → Async Redis client
cache_with_ttl()     → TTL-based caching decorator
get_cache_key()      → Cache key generation
CacheManager         → High-level cache operations
```

### API Parallelization
**File**: `backend/api/routers/analysis.py:335-404`
```
fetch_parallel_with_fallback()  → Parallel API fetching
safe_async_call()               → Timeout-protected calls
asyncio.gather()                → Concurrent execution
```

### Recommendations Engine (N+1 Query Fix)
**File**: `backend/services/recommendation_service.py`
```
generate_ml_powered_recommendations() → Optimized with batch queries
get_daily_recommendations()           → Cached daily picks
generate_personalized_recommendations() → User-specific recommendations
```

**Optimization (N+1 Query Fix):**
- Before: 201+ queries (1 stock query + 2 per stock)
- After: 2-3 queries (1 stock + 1 bulk price history)
- Improvement: 60-80% faster response time

## Security Modules (`backend/security/`)

20 security files providing comprehensive protection.

| Module | Purpose | Status |
|--------|---------|--------|
| `rbac.py` | Role-Based Access Control (in-memory + DB-backed) | COMPLETE |
| `crypto_utils.py` | Fernet encryption + RSA-2048 signing | COMPLETE |
| `password_manager.py` | bcrypt (work factor 12) + legacy PBKDF2 verify | COMPLETE |
| `security_config.py` | Overall security configuration (1,140 lines) | COMPLETE |
| `security_headers.py` | HSTS, CSP, X-Frame-Options, Referrer-Policy | COMPLETE |
| `rate_limiter.py` | Redis-backed distributed rate limiting | COMPLETE |
| `jwt_manager.py` | RS256 JWT lifecycle management | COMPLETE |

### RBAC Implementation (`backend/security/rbac.py`)

```python
class RoleBasedAccessControl:
    # Supports in-memory (default) or DB-backed (optional SQLAlchemy session)
    def has_permission(role, permission) -> bool    # Static role-permission map
    def get_user_roles(user_id) -> List[str]         # DB-aware role lookup
    def assign_role(user_id, role) -> bool           # Assign with DB persistence
    def revoke_role(user_id, role) -> bool           # Revoke with DB persistence
    def check_access(user_id, resource, action) -> bool  # Full RBAC check
```

Roles: `admin`, `analyst`, `user`, `viewer`
Permissions: `read`, `write`, `delete`, `admin`

### Crypto Utils (`backend/security/crypto_utils.py`)

```python
class CryptoUtils:
    # All methods implemented with cryptography library
    def encrypt_data(data, key) -> bytes     # Fernet AES-128-CBC + HMAC
    def decrypt_data(encrypted, key) -> bytes # Fernet decrypt
    def generate_key_pair() -> (priv, pub)   # RSA-2048 key pair (PEM)
    def sign_data(data, private_key) -> bytes # RSA-PSS + SHA-256
    def verify_signature(data, sig, pub) -> bool  # RSA-PSS verify
    def hash_data(data, algorithm) -> str    # SHA-256/512
```

## ML Pipeline (`backend/ml/`)

48 files organized across 4 subdirectories.

| Module | Purpose |
|--------|---------|
| `models/lstm_predictor.py` | LSTM neural network (weights absent from disk) |
| `models/xgboost_model.py` | XGBoost gradient boosting (690 KB on disk) |
| `models/prophet_forecaster.py` | Time-series forecasting (3 stocks: AAPL, ADBE, AMZN) |
| `models/lightgbm_model.py` | LightGBM integration |
| `feature_engineering.py` | TA-Lib feature extraction |
| `training_pipeline.py` | Model training orchestration |
| `inference_service.py` | Real-time predictions |
| `backtesting.py` | Strategy validation |
| `pipeline/deployment.py` | Model deployment management |
| `pipeline/optimization.py` | Hyperparameter optimization |

## ETL Pipeline (`backend/etl/`)

24 files with multi-layer caching.

| Module | Purpose |
|--------|---------|
| `multi_source_extractor.py` | API data extraction |
| `data_transformers.py` | Data transformation |
| `intelligent_cache_system.py` | Multi-layer caching (6 cache modules) |
| `data_quality_checker.py` | Data validation |
| `batch_processor.py` | Bulk processing |
| `unlimited_data_extractor.py` | StockData + ExtractionResult dataclasses |

## Repositories (`backend/repositories/`)

13 files implementing async CRUD patterns.

| Repository | Purpose |
|------------|---------|
| `stock_repository.py` | Stock CRUD + FTS search + `get_top_stocks()` |
| `price_repository.py` | Price history + batch queries (N+1 fix) |
| `recommendation_repository.py` | Recommendation storage |
| `portfolio_repository.py` | Portfolio management |
| `user_repository.py` | User data access |

### Key Repository Methods (N+1 Query Fix)

**StockRepository** (`backend/repositories/stock_repository.py`):
| Method | Purpose |
|--------|---------|
| `get_top_stocks()` | Optimized query for top stocks by market cap |
| `get_stocks_with_latest_prices()` | Join stocks with latest prices |
| `get_sector_summary()` | Aggregated sector statistics |

**PriceHistoryRepository** (`backend/repositories/price_repository.py`):
| Method | Purpose |
|--------|---------|
| `get_bulk_price_history()` | Batch fetch for multiple symbols (N+1 fix) |
| `get_latest_prices_bulk()` | Batch latest prices (N+1 fix) |
| `get_price_history()` | Single symbol price history |
| `calculate_returns()` | Period return calculations |
| `get_volatility()` | Historical volatility calculation |

## Database Migrations (`backend/migrations/versions/`)

| Migration | Purpose |
|-----------|---------|
| `001_initial_schema.py` | Base tables |
| `002_add_price_history.py` | Time-series tables |
| `003_add_recommendations.py` | Recommendation tables |
| `004_add_portfolios.py` | Portfolio tables |
| `005_add_ml_tables.py` | ML prediction storage |
| `006_add_audit_logging.py` | SEC compliance |
| `007_add_gdpr_fields.py` | GDPR compliance |
| `008_add_missing_query_indexes.py` | 45 performance indexes |
| `009` through `013` | Additional schema updates |

## Utilities (`backend/utils/`)

61 files (reduced from 87 via Loki remediation dead code cleanup).

| Module | Purpose |
|--------|---------|
| `cache.py` | Redis caching with TTL decorators |
| `auth.py` | Auth helpers — returns dict (vs oauth2.py which returns User ORM) |
| `validators.py` | Input validation |
| `enhanced_logging.py` | Application logging |
| `numpy_serializer.py` | NumPy JSON serialization for API responses |

## Tasks (`backend/tasks/`)

14 Celery task files across 5 queues.

| Module | Purpose |
|--------|---------|
| `data_collection.py` | Scheduled data fetching |
| `ml_training.py` | Model retraining tasks |
| `recommendation_generation.py` | Daily recommendations |
| `portfolio_updates.py` | Portfolio calculations |
| `alerts_processing.py` | Alert notifications |
| `maintenance_tasks.py` | System maintenance (1,111 lines) |

## Configuration (`backend/config/`)

| File | Purpose |
|------|---------|
| `settings.py` | Application settings |
| `database.py` | Async/sync database configuration (632 lines) |
| `celery_config.py` | 5-queue task scheduler settings |
| `logging_config.py` | Logging configuration |

## Tests (`backend/tests/`)

28 unit test files, 71+ total test files, 5,026 tests passing.

| Directory | Files | Coverage |
|-----------|-------|---------|
| `unit/` | 28 files | Services, utils, ML, middleware |
| `integration/` | 16 files | API flows, auth, data lifecycle |
| `security/` | 5 files | CSRF (67 tests), OWASP (48 tests) |
| `middleware/` | 4 files | Middleware stack |

### N+1 Query Fix Tests

| Test File | Purpose |
|-----------|---------|
| `test_n1_query_fix.py` | Unit tests for batch query methods |
| `benchmark_n1_query_fix.py` | Performance benchmarks |

**Expected Benchmark Results:**
| Stocks | N+1 Queries | Batch Queries | Speedup |
|--------|-------------|---------------|---------|
| 10 | 11 | 2 | 5.5x |
| 50 | 51 | 2 | 25.5x |
| 100 | 101 | 2 | 50.5x |

**Last Updated**: 2026-03-04
