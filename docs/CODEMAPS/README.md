# Architecture Codemaps

**Last Updated:** 2026-02-24

Quick reference to codebase structure for developers.

| Codemap | Purpose |
|---------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System-wide architecture overview |
| [BACKEND.md](BACKEND.md) | API routers, ML pipeline, ETL modules |
| [FRONTEND.md](FRONTEND.md) | React pages, components, state |
| [DATA_FLOW.md](DATA_FLOW.md) | Data sources, caching, pipelines |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | Docker, CI/CD, monitoring, deployment |

## Quick Reference

| Component | Location | Key Files |
|-----------|----------|-----------|
| API Routers | `backend/api/routers/` | 18 mounted routers |
| ML Pipeline | `backend/ml/` | 22 modules |
| ETL Pipeline | `backend/etl/` | 17 modules (StockData/ExtractionResult dataclasses added) |
| React Pages | `frontend/web/src/pages/` | 12 pages |
| Redux State | `frontend/web/src/store/` | 6 slices |
| CI/CD Workflows | `.github/workflows/` | 29 workflows (Python 3.12, Node 20, Actions v4/v5) |
| Docker Config | `infrastructure/` | 4 compose files |
| Monitoring | `config/monitoring/` | Prometheus, Grafana |

## CI/CD Hardening (2026-02-24)

### Runtime Version Upgrades

| Runtime | Previous | Current | Affected Workflows |
|---------|----------|---------|-------------------|
| Python | 3.11 | 3.12 | 9 workflows |
| Node.js | 18 | 20 | 9 workflows |
| `actions/setup-python` | v4 | v5 | All Python workflows |
| `actions/setup-node` | v3 | v4 | All Node.js workflows |
| `actions/upload-artifact` | v3 | v4 | All artifact workflows |
| `github/codeql-action` | v2 | v3 | `security-scan.yml` |

### Workflow Fixes (12 commits)

| Fix | Workflows | Impact |
|-----|-----------|--------|
| TA-Lib C library install | `daily-pipeline-validation.yml`, `security-scan.yml` | Resolved `talib` import failures |
| Missing env vars (SECRET_KEY, JWT_SECRET_KEY) | `daily-pipeline-validation.yml` | Fixed pipeline startup crashes |
| SSL disabled for CI PostgreSQL | `daily-pipeline-validation.yml` | Fixed db connection errors |
| Semgrep/GitLeaks non-blocking | `security-scan.yml` | Prevented false-positive CI failures |
| npm audit output fix | `dependency-updates.yml` | Fixed garbled output concatenation |
| ETL validation resilience | `daily-pipeline-validation.yml` | Handles import errors gracefully |

### ETL Dataclass Additions

Added missing `StockData` and `ExtractionResult` dataclasses to `backend/etl/unlimited_data_extractor.py` -- these were referenced by `unlimited_extractor_with_fallbacks.py` but not defined.

### Orphan Submodule Removal

Removed orphan `excalidraw` git submodule. `.gitmodules` file deleted entirely (no remaining submodules).

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

## Test Status (2026-02-24)

| Metric | Value |
|--------|-------|
| Passing | 1543 |
| Skipped | 8 (infra-only: celery, testcontainers, sqlparse, memory_profiler, objgraph, psycopg2, requests_mock) |
| Failed | 0 |
| xfailed | 5 |

109 tests recovered across 4 commits (WebSocket, Celery, security, integration, monitoring).

## Performance-Critical Paths

| Path | File | Line | Description |
|------|------|------|-------------|
| Cache Decorator | `backend/utils/cache.py` | 205-300 | Redis caching with TTL |
| API Parallelization | `backend/api/routers/analysis.py` | 335-404 | Parallel API calls |
| Database Indexes | `backend/migrations/versions/008_*` | - | 45 optimized indexes |

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

**Last Updated**: 2026-02-24
