# Backend Architecture Codemap

## API Routers (`backend/api/routers/`)

| Router | File | Purpose |
|--------|------|---------|
| admin | `admin.py` | Admin operations |
| agents | `agents.py` | AI agent management |
| analysis | `analysis.py` | Stock analysis endpoints |
| auth | `auth.py` | Authentication (OAuth2/JWT) |
| cache_management | `cache_management.py` | Cache control endpoints |
| gdpr | `gdpr.py` | GDPR compliance (export/delete) |
| health | `health.py` | Health check endpoints |
| monitoring | `monitoring.py` | Metrics and monitoring |
| portfolio | `portfolio.py` | Portfolio management |
| recommendations | `recommendations.py` | AI recommendations |
| stocks | `stocks.py` | Stock CRUD operations |
| stocks_legacy | `stocks_legacy.py` | Legacy stock endpoints |
| watchlist | `watchlist.py` | Watchlist operations |
| websocket | `websocket.py` | Real-time WebSocket |

## Key Code Paths

### Cache Decorator (Quick Win #1)
**File**: `backend/utils/cache.py:205-300`
```
get_redis()          → Async Redis client
cache_with_ttl()     → TTL-based caching decorator
get_cache_key()      → Cache key generation
CacheManager         → High-level cache operations
```

### API Parallelization (Quick Win #2)
**File**: `backend/api/routers/analysis.py:335-404`
```
fetch_parallel_with_fallback()  → Parallel API fetching
safe_async_call()               → Timeout-protected calls
asyncio.gather()                → Concurrent execution
```

### Recommendations Engine
**File**: `backend/api/routers/recommendations.py:315-461`
```
get_recommendations()           → Main recommendation logic
calculate_scores()              → ML-based scoring
filter_by_sector()              → Sector filtering
```

## ML Pipeline (`backend/ml/`)

| Module | Purpose |
|--------|---------|
| `models/lstm_predictor.py` | LSTM neural network |
| `models/xgboost_model.py` | XGBoost gradient boosting |
| `models/prophet_forecaster.py` | Time-series forecasting |
| `feature_engineering.py` | Feature extraction |
| `training_pipeline.py` | Model training orchestration |
| `inference_service.py` | Real-time predictions |
| `backtesting.py` | Strategy validation |

## ETL Pipeline (`backend/etl/`)

| Module | Purpose |
|--------|---------|
| `multi_source_extractor.py` | API data extraction |
| `data_transformers.py` | Data transformation |
| `intelligent_cache_system.py` | Multi-layer caching |
| `data_quality_checker.py` | Data validation |
| `batch_processor.py` | Bulk processing |

## Repositories (`backend/repositories/`)

| Repository | Purpose |
|------------|---------|
| `stock_repository.py` | Stock CRUD + FTS search |
| `price_repository.py` | Price history queries |
| `recommendation_repository.py` | Recommendation storage |
| `portfolio_repository.py` | Portfolio management |
| `user_repository.py` | User data access |

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
| `008_add_missing_query_indexes.py` | **45 performance indexes** |

## Utilities (`backend/utils/`)

| Module | Purpose |
|--------|---------|
| `cache.py` | Redis caching (fixed in Quick Wins) |
| `rate_limiter.py` | API rate limiting |
| `validators.py` | Input validation |
| `enhanced_logging.py` | Application logging |
| `security.py` | Auth helpers |

## Tasks (`backend/tasks/`)

| Module | Purpose |
|--------|---------|
| `data_collection.py` | Scheduled data fetching |
| `ml_training.py` | Model retraining tasks |
| `recommendation_generation.py` | Daily recommendations |
| `portfolio_updates.py` | Portfolio calculations |
| `alerts_processing.py` | Alert notifications |

## Configuration (`backend/config/`)

| File | Purpose |
|------|---------|
| `settings.py` | Application settings |
| `database.py` | Database configuration |
| `celery_config.py` | Task queue settings |
| `logging_config.py` | Logging configuration |

**Last Updated**: 2026-01-26
