# Architecture Codemaps

**Last Updated:** 2026-01-28 (Wave 5)

Quick reference to codebase structure for developers.

| Codemap | Purpose |
|---------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System-wide architecture overview (NEW) |
| [BACKEND.md](BACKEND.md) | API routers, ML pipeline, ETL modules |
| [FRONTEND.md](FRONTEND.md) | React pages, components, state |
| [DATA_FLOW.md](DATA_FLOW.md) | Data sources, caching, pipelines |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | Docker, monitoring, deployment |

## Quick Reference

| Component | Location | Key Files |
|-----------|----------|-----------|
| API Routers | `backend/api/routers/` | 15 routers (6 fixed in Wave 5) |
| ML Pipeline | `backend/ml/` | 22 modules |
| ETL Pipeline | `backend/etl/` | 17 modules |
| React Pages | `frontend/web/src/pages/` | 15 pages |
| Redux State | `frontend/web/src/store/` | 6 slices |
| Docker Config | `infrastructure/` | 4 compose files |
| Monitoring | `config/monitoring/` | Prometheus, Grafana |

## Wave 5 Updates (2026-01-28)

### Routing Architecture Fixes

| Change | Files | Impact |
|--------|-------|--------|
| Double-prefix fix | 6 routers | Resolved 404 errors |
| Added /ping endpoint | health.py | Better health checks |
| Rate limiter TESTING mode | rate_limiter.py | Tests run without rate limiting |

### Schema Alignment

| Model | Field Change | Purpose |
|-------|--------------|---------|
| Watchlist | Added `is_public` | Privacy control |
| Transaction | `trade_date` (not `executed_at`) | Correct field name |
| Stock | `industry_id` FK | Proper foreign key |

### Test Patterns Discovered

- **Schema validation**: Always verify field names against `unified_models.py`
- **Async fixtures**: Use sync `MagicMock` for Redis cache
- **CSRF handling**: `testing_mode=True` disables CSRF in tests

## Performance-Critical Paths

| Path | File | Line | Description |
|------|------|------|-------------|
| Cache Decorator | `backend/utils/cache.py` | 205-300 | Redis caching with TTL |
| API Parallelization | `backend/api/routers/analysis.py` | 335-404 | Parallel API calls |
| Database Indexes | `backend/migrations/versions/008_*` | - | 45 optimized indexes |

## Recent Optimizations (Quick Wins)

- **Cache**: Fixed decorator at `backend/utils/cache.py:205-300`
- **Parallelization**: Added at `backend/api/routers/analysis.py:335-404`
- **Indexes**: Migration `008_add_missing_query_indexes.py`
- **Search**: PostgreSQL FTS replaces Elasticsearch
- **N+1 Query Fix**: Batch queries at `backend/repositories/price_repository.py:411-512`

## N+1 Query Pattern Fix (CRITICAL-3)

Eliminated N+1 queries in recommendations generation:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Query Count | 201+ | 2-3 | 99% reduction |
| Response Time | 5-10s | 0.5-1s | 60-80% faster |
| DB Load | High | Minimal | Significant reduction |

**Key Changes:**
- `price_repository.get_bulk_price_history()` - Single query for all price histories
- `price_repository.get_latest_prices_bulk()` - Batch latest prices
- `stock_repository.get_top_stocks()` - Optimized top stocks query
- `recommendations.py:302-540` - Refactored to use batch queries

**Test Coverage:**
- `backend/tests/test_n1_query_fix.py` - Unit tests for batch queries
- `backend/tests/benchmark_n1_query_fix.py` - Performance benchmarks

**Last Updated**: 2026-01-28 (Wave 5)
