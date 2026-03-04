# Architecture Codemaps

**Last Updated:** 2026-03-04

Quick reference to codebase structure for developers.

| Codemap | Purpose |
|---------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System-wide architecture overview |
| [BACKEND.md](BACKEND.md) | API routers, services, ML pipeline, security |
| [FRONTEND.md](FRONTEND.md) | React pages, components, state, tests |
| [DATA_FLOW.md](DATA_FLOW.md) | Data sources, caching, pipelines |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | Docker, CI/CD, monitoring, deployment |

## Quick Reference

| Component | Location | Key Files |
|-----------|----------|-----------|
| API Routers | `backend/api/routers/` | 19 router files, 153+ endpoints |
| Service Layer | `backend/services/` | 20 service files (10,241 total lines) |
| ML Pipeline | `backend/ml/` | 48 modules |
| ETL Pipeline | `backend/etl/` | 24 modules |
| Security | `backend/security/` | 20 modules (RBAC, Fernet crypto, bcrypt passwords) |
| React Pages | `frontend/web/src/pages/` | 14 pages (all lazy-loaded) |
| Redux State | `frontend/web/src/store/slices/` | 6 slices |
| Frontend Tests | `frontend/web/src/**/*.test.tsx` | 13 test files, 201 tests |
| Backend Tests | `backend/tests/` | 71+ test files, 5,020 passing |
| CI/CD Workflows | `.github/workflows/` | 29 workflows (Python 3.12, Node 20, Actions v4/v5) |
| Docker Config | `docker-compose*.yml` | 5 compose files |
| Monitoring | `infrastructure/monitoring/` | Prometheus, Grafana, Loki, SLO targets |

## Current State (2026-03-04)

### Completed Priority Work

| Priority | Items | Status |
|----------|-------|--------|
| P0 — Security Hardening | RBAC, crypto_utils, password_manager, CSP | COMPLETE |
| P1 — CI Gates | Auth page tests (+30), slow test tagging | COMPLETE |
| P2 — API Completion | trading.py router, ml.py expanded to 8 endpoints | COMPLETE |
| P3 — Test Coverage | Test pollution fixed, 5,020 tests passing | COMPLETE |
| P4 — Deployment & Ops | certbot, Loki+Promtail, SLO alerts, GDPR key | COMPLETE |
| P5 — Frontend Polish | EnhancedDashboard deleted, analytics components to portfolio/ | COMPLETE |

### Remaining Work

| Item | Priority | Notes |
|------|----------|-------|
| SSL certificates provisioned | HIGH | nginx-ssl.conf references empty ssl/ dir |
| CI test gates made blocking | MEDIUM | `continue-on-error: true` still on lines 311, 457 |
| Coverage floor raised to 60% | MEDIUM | Currently 35% |
| Stock data loaded | MEDIUM | 0 stocks in DB, need NYSE/NASDAQ/AMEX |
| LSTM model weights | LOW | Training code exists, weights not saved |
| Trading router tests | LOW | test_trading_router.py exists |
| Frontend TS errors quantified | LOW | Run `tsc --noEmit` |
| Vitest/Playwright collision | LOW | Add `exclude: ['**/tests/e2e/**']` |
| Redux slices/hooks test coverage | LOW | Currently 0% |

## Test Status (2026-03-04)

| Metric | Value |
|--------|-------|
| Backend tests passing | 5,020 |
| Backend tests skipped | 8 (infra-only: celery, testcontainers, etc.) |
| Backend xfailed | 2 |
| Backend failed | 0 |
| Frontend tests passing | 197 |
| Frontend test files | 13 |
| Unit test files | 28 |
| Total test files | 71+ |

## Performance-Critical Paths

| Path | File | Description |
|------|------|-------------|
| Cache Decorator | `backend/utils/cache.py` | Redis caching with TTL |
| API Parallelization | `backend/api/routers/analysis.py:335-404` | Parallel API calls |
| Batch Price History | `backend/repositories/price_repository.py` | N+1 fix |
| Database Indexes | `backend/migrations/versions/008_*` | 45 optimized indexes |

## Security Stack Summary

| Component | Implementation | Status |
|-----------|---------------|--------|
| RBAC | `security/rbac.py` — in-memory + DB-backed role management | COMPLETE |
| Encryption | `security/crypto_utils.py` — Fernet AES + RSA-2048 | COMPLETE |
| Passwords | `security/password_manager.py` — bcrypt work factor 12 | COMPLETE |
| JWT | RS256 with auto-generated RSA keys | COMPLETE |
| CSP | `script-src 'self'` only (no unsafe-inline) | HARDENED |
| Rate Limiting | Redis-backed, 4 categories | COMPLETE |

**Last Updated**: 2026-03-04
