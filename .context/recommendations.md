# Recommendations

**Last Updated**: 2026-03-03
**Previous**: 2026-02-26

## Executive Summary

Five parallel analysis agents examined project structure, backend features, frontend status, test suite health, and deployment readiness. The platform has matured significantly since Feb 26 (88% -> 91%), but three critical security stubs and operational gaps must be addressed before production.

## Priority Matrix

### P0: Security Hardening (Immediate - Days 1-3)

| # | Action | Time | Impact |
|---|--------|------|--------|
| 1 | Implement `security/rbac.py` — replace NotImplementedError stubs with DB-backed role management | 2-3 days | Enables fine-grained access control |
| 2 | Implement `security/crypto_utils.py` — Fernet/AES encryption for PII at rest | 1 day | Unblocks GDPR field encryption |
| 3 | Upgrade `security/password_manager.py` to bcrypt/argon2id with complexity rules | 4 hrs | Prevents GPU-based password cracking |
| 4 | Remove `'unsafe-inline'` from CSP `script-src` in `security/security_headers.py:516` | 2 hrs | Closes XSS vector |
| 5 | Provision SSL certificates (certbot container or manual Let's Encrypt) | 2 hrs | Unblocks HTTPS in production |

### P1: Fix Failing Tests and CI Gates (Days 3-5)

| # | Action | Time | Impact |
|---|--------|------|--------|
| 6 | Fix 4 frontend test failures (Dashboard heatmap mock, Portfolio tab selectors) | 2 hrs | Green frontend CI |
| 7 | Fix Vitest/Playwright collision — add `exclude: ['**/tests/e2e/**']` to Vitest config | 15 min | Stop E2E collection errors |
| 8 | Fix flaky SEC Edgar test — refactor import pattern or add `@pytest.mark.flaky` | 1 hr | Eliminate false failures |
| 9 | Remove `continue-on-error: true` from CI test steps | 15 min | Make tests blocking |
| 10 | Raise coverage floor from 35% to 60% | 15 min | Enforce quality gate |
| 11 | Fix Dockerfile path mismatch — align CI (`./frontend/web/Dockerfile`) with repo (`./Dockerfile.frontend`) | 15 min | Unblock production image build |

### P2: API Completion (Week 1)

| # | Action | Time | Impact |
|---|--------|------|--------|
| 12 | Create `backend/api/routers/trading.py` — expose order CRUD from trading_service.py | 1 day | Complete trading feature |
| 13 | Expand `backend/api/routers/ml.py` — add drift detection, model management, backtesting endpoints | 1 day | Expose ML capabilities |
| 14 | Fix 15 frontend TypeScript errors (Socket.IO types, unused imports, @types/lodash) | 2 hrs | Clean TS build |
| 15 | Add typed methods for watchlist/alerts/settings to `services/api.service.ts` | 1 hr | Complete API integration |

### P3: Test Coverage Expansion (Weeks 1-2)

| # | Action | Coverage Impact |
|---|--------|----------------|
| 16 | Create TradingAgents test suite (~20 untested files) | +5-8% |
| 17 | Add frontend hook tests (useRealTimePrices, usePortfolioWebSocket, etc.) | +3% frontend |
| 18 | Add Redux slice tests (6 slices, zero coverage) | +5% frontend |
| 19 | Tag slow tests (>10s) with `@pytest.mark.slow` | Faster CI feedback |
| 20 | Fix 8 infrastructure-skipped tests (install celery, testcontainers, etc. in CI) | +2% |
| 21 | Add auth page tests (Login, Register, ForgotPassword) | +2% frontend |

### P4: Deployment & Operations (Weeks 2-3)

| # | Action | Impact |
|---|--------|--------|
| 22 | Add certbot container to docker-compose.production.yml for auto-renewal | Automated SSL |
| 23 | Add log aggregation (Loki + Promtail for Grafana) | Queryable logs |
| 24 | Define SLOs (availability 99.9%, latency p95 < 500ms, error rate < 0.5%) | Error budgets |
| 25 | Add prometheus-remote-storage container or configure external TSDB | Metric retention > 7 days |
| 26 | Load stock data (NYSE/NASDAQ/AMEX) + create investment_user role | Enable core functionality |
| 27 | Add GDPR_ENCRYPTION_KEY to .env | Unblock backend startup |

### P5: Frontend Polish (Week 2-3)

| # | Action | Impact |
|---|--------|--------|
| 28 | Investigate `EnhancedDashboard.tsx` (745 lines, not routed) — delete if dead code | Reduce bloat |
| 29 | Extract `SettingsTabs.tsx` (586 lines) into sub-components | Maintainability |
| 30 | Extract `PortfolioSummary.tsx` (534 lines) into sub-components | Maintainability |
| 31 | Resolve 4 unorganized root components (CorrelationMatrix, EfficientFrontier, etc.) | Clean structure |

## Architecture Recommendations

### Keep (Excellent Patterns)
- **Repository layer**: Generic async CRUD with locking and upsert — A-grade
- **Middleware stack**: Priority-based (10000 down to 1000) with testing skip — A-grade
- **Service layer**: 19 dedicated service files with clean router separation
- **Frontend code splitting**: Lazy loading with typed skeletons and route prefetching
- **Frontend hooks library**: 13 performance-oriented hooks (virtual scroll, Web Worker, etc.)

### Improve
- **RBAC**: Must move from boolean `is_admin` to role-permission matrix
- **ML API surface**: 48-file subsystem exposed through only 2 endpoints
- **Frontend service layer**: Missing typed methods for 3 endpoint groups
- **Password security**: Upgrade from PBKDF2 to bcrypt/argon2id
- **Monitoring**: Add SLOs, distributed tracing, log aggregation

### Watch
- **Security module file sizes**: 14 files >500 lines in `backend/security/` — cohesive but large
- **Task module sizes**: `maintenance_tasks.py` (1,112 lines) — candidate for splitting
- **4 charting libraries in frontend**: Recharts + Plotly + Chart.js + Lightweight Charts (overkill)

## Success Metrics (Updated)

| Metric | Feb 26 | Mar 3 | Target (30 days) |
|--------|--------|-------|-------------------|
| Test count | 3,569 | 5,132 | 6,000+ |
| Backend pass rate | 99.5% | 99.98% | 100% |
| Frontend pass rate | N/A | 98.0% | 100% |
| TS errors | Unknown | 15 | 0 |
| Security stubs | Unknown | 3 | 0 |
| CI test gates | Advisory | Advisory | Blocking |
| Coverage floor | 35% | 35% | 60% blocking |
| Deployment readiness | 75% | 78.5% | 90% |
| Stocks loaded | 0 | 0 | 1,000+ |
| SSL provisioned | No | No | Yes |
