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

### Recommendations Engine (N+1 Query Fix - CRITICAL-3)
**File**: `backend/api/routers/recommendations.py:302-540`
```
generate_ml_powered_recommendations() → Optimized with batch queries
get_daily_recommendations()           → Cached daily picks
generate_personalized_recommendations() → User-specific recommendations
```

**Optimization (N+1 Query Fix):**
- Before: 201+ queries (1 stock query + 2 per stock)
- After: 2-3 queries (1 stock + 1 bulk price history)
- Improvement: 60-80% faster response time

### Batch Price History Queries (Quick Win #5)
**File**: `backend/repositories/price_repository.py:411-512`
```
get_bulk_price_history()        → Single query for all symbols
get_latest_prices_bulk()        → Batch latest prices
```

**Usage in recommendations.py:**
```python
# BEFORE (N+1 pattern):
for stock in stocks:
    prices = await price_repository.get_price_history(stock.symbol)

# AFTER (batch pattern):
all_prices = await price_repository.get_bulk_price_history(symbols)
for symbol, prices in all_prices.items():
    # Process cached data
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
| `stock_repository.py` | Stock CRUD + FTS search + `get_top_stocks()` |
| `price_repository.py` | Price history + **batch queries (N+1 fix)** |
| `recommendation_repository.py` | Recommendation storage |
| `portfolio_repository.py` | Portfolio management |
| `user_repository.py` | User data access |

### Key Repository Methods (N+1 Query Fix)

**StockRepository** (`backend/repositories/stock_repository.py`):
| Method | Line | Purpose |
|--------|------|---------|
| `get_top_stocks()` | 442-483 | Optimized query for top stocks by market cap |
| `get_stocks_with_latest_prices()` | 217-285 | Join stocks with latest prices |
| `get_sector_summary()` | 287-324 | Aggregated sector statistics |

**PriceHistoryRepository** (`backend/repositories/price_repository.py`):
| Method | Line | Purpose |
|--------|------|---------|
| `get_bulk_price_history()` | 411-512 | **Batch fetch for multiple symbols** |
| `get_latest_prices_bulk()` | 514-539 | **Batch latest prices** |
| `get_price_history()` | 28-73 | Single symbol price history |
| `calculate_returns()` | 135-180 | Period return calculations |
| `get_volatility()` | 182-235 | Historical volatility calculation |

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

## Tests (`backend/tests/`)

### N+1 Query Fix Tests (CRITICAL-3)

| Test File | Purpose |
|-----------|---------|
| `test_n1_query_fix.py` | Unit tests for batch query methods |
| `benchmark_n1_query_fix.py` | Performance benchmarks for N+1 fix |

**Test Classes in `test_n1_query_fix.py`:**
| Class | Tests |
|-------|-------|
| `TestGetTopStocks` | Stock repository top stocks method |
| `TestGetBulkPriceHistory` | Batch price history fetching |
| `TestQueryCountReduction` | Verifies N+1 pattern elimination |
| `TestLatestPricesBulk` | Bulk latest prices method |
| `TestPerformance` | Query count comparison |
| `TestEdgeCases` | Missing symbols, insufficient data |
| `TestIntegrationWithRecommendations` | End-to-end batch data usage |

**Running Tests:**
```bash
# Run N+1 query fix tests
pytest backend/tests/test_n1_query_fix.py -v

# Run benchmark
python -m backend.tests.benchmark_n1_query_fix
```

**Expected Benchmark Results:**
| Stocks | N+1 Queries | Batch Queries | Speedup |
|--------|-------------|---------------|---------|
| 10 | 11 | 2 | 5.5x |
| 50 | 51 | 2 | 25.5x |
| 100 | 101 | 2 | 50.5x |

**Last Updated**: 2026-01-26
