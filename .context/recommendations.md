# Recommendations

**Last Updated**: 2026-03-04 (Post P0-P5 completion)
**Previous**: 2026-03-03

## Executive Summary

All P0-P5 priority items from the March 3 roadmap are complete. Security stubs are replaced
with real implementations. The trading router and expanded ML router are live. Loki+Promtail
log aggregation, certbot SSL auto-renewal, and SLO alerts are wired into production compose.
Frontend dead code removed, analytics components organized. The platform is staging-ready.
Primary remaining blocker: SSL certificate provisioning.

## Priority Matrix

### Immediate (Days 1-2): Unblock Staging

| # | Action | Time | Impact |
|---|--------|------|--------|
| 1 | Provision SSL certificates — run certbot or generate self-signed for staging | 2 hrs | Unblocks nginx in production |
| 2 | Create `investment_user` database role in PostgreSQL | 30 min | Required for production DB |
| 3 | Generate and configure GDPR_ENCRYPTION_KEY in `.env` | 15 min | Already in compose, need actual key value |
| 4 | Remove `continue-on-error: true` from CI test steps (lines 311, 457) | 15 min | Makes CI a real gate |

### Week 1: Quality Gates

| # | Action | Time | Impact |
|---|--------|------|--------|
| 5 | Raise coverage floor from 35% to 60% in `ci.yml` line 387 | 15 min | Enforce quality gate |
| 6 | Add `exclude: ['**/tests/e2e/**']` to Vitest config | 15 min | Stop E2E collection errors |
| 7 | Fix 4 frontend test failures (Dashboard heatmap mock, Portfolio tab selectors) | 2 hrs | Green frontend CI |
| 8 | Load stock data (NYSE/NASDAQ/AMEX) — use ETL scripts or seed data | 1 day | Enable core functionality |
| 9 | Run `tsc --noEmit` to quantify TS errors. Fix Socket.IO generics in hooks. | 2 hrs | Clean TS build |

### Week 2: Coverage Expansion

| # | Action | Coverage Impact |
|---|--------|----------------|
| 10 | Add Redux slice tests (6 slices, zero coverage) | +5% frontend |
| 11 | Add custom hook tests (useRealTimePrices, usePortfolioWebSocket, etc.) | +3% frontend |
| 12 | Add typed methods for watchlist/alerts/settings to `services/api.service.ts` | Clean API layer |
| 13 | Expand TradingAgents test suite (3 tests for 39 files) | +5-8% backend |

### Week 3: Operations

| # | Action | Impact |
|---|--------|--------|
| 14 | Configure Prometheus remote storage for metric retention > 7 days | Metric history |
| 15 | Set up alertmanager paging (PagerDuty/OpsGenie) | On-call alerting |
| 16 | Train and save LSTM model weights | Complete ML suite |
| 17 | Set up Playwright E2E in CI (fix Vitest collision first) | E2E coverage |

## Architecture Recommendations

### Keep (Excellent Patterns)
- **Repository layer**: Generic async CRUD with locking and upsert — A-grade
- **Middleware stack**: Priority-based (10000 down to 1000) with testing skip — A-grade
- **Service layer**: 20 dedicated service files with clean router separation
- **RBAC**: Now fully functional with in-memory + optional DB-backed persistence
- **Security crypto**: Fernet + RSA-2048 properly implemented
- **Password security**: bcrypt with legacy fallback + strength scoring
- **Frontend code splitting**: Lazy loading with typed skeletons and route prefetching
- **Frontend hooks library**: 13 performance-oriented hooks (virtual scroll, Web Worker, etc.)
- **ML router**: Expanded to 8 endpoints with Redis caching

### Improve
- **ML API surface**: 48-file subsystem now has 8 endpoints — consider exposing drift detection,
  artifact management, and backtesting via additional endpoints
- **Frontend service layer**: Missing typed methods for 3 endpoint groups
- **TradingAgents coverage**: 3 test files for 39 source files (~8% coverage)
- **Monitoring**: Add distributed tracing (Jaeger/OpenTelemetry) for request correlation

### Watch
- **Security module file sizes**: 14 files >500 lines in `backend/security/` — cohesive but large
- **Task module sizes**: `maintenance_tasks.py` (1,112 lines) — candidate for splitting
- **4 charting libraries in frontend**: Recharts + Plotly + Chart.js + Lightweight Charts (overkill)
- **30 files >800 lines in backend**: Most cohesive, largest (recommendation_service 1,234 lines)

## Success Metrics (Updated 2026-03-04)

| Metric | Mar 3 | Mar 4 | Target (30 days) |
|--------|-------|-------|-------------------|
| Backend tests passing | 4,931 | 5,020 | 5,500+ |
| Backend pass rate | 99.98% | 100% (0 failed) | 100% |
| Frontend tests passing | 197 | 197 | 240+ |
| TS suppressions | 0 | 0 | 0 |
| Security stubs | 3 | 0 | 0 |
| CI test gates | Advisory | Advisory | Blocking |
| Coverage floor | 35% | 35% | 60% blocking |
| Deployment readiness | 79.5% | 84% | 90% |
| API endpoints | 153 | 153+ | 160+ |
| SSL provisioned | No | No | Yes |
| Stocks loaded | 0 | 0 | 1,000+ |
| RBAC functional | Stub | Complete | Complete |
| Crypto implemented | Stub | Complete | Complete |
| Passwords bcrypt | No | Yes | Yes |
| Loki configured | No | Yes | Yes |
| Certbot configured | No | Yes | Yes |
| SLO alerts defined | No | Yes | Yes |
| GDPR key wired | No | Yes (needs value) | Yes |
| Frontend dead code | 746 lines | 0 | 0 |
| Components organized | 3 misplaced | Resolved | Resolved |
