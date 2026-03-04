# Identified Issues

**Last Updated**: 2026-03-04 (Post P0-P5 completion)

## Issue Summary

| Priority | Count | Status |
|----------|-------|--------|
| Critical | 0 | All security stubs resolved |
| High (Deployment Blockers) | 2 | SSL certs + CI gates |
| Medium (Quality/Completeness) | 7 | Mix of new and existing |
| Low (Tech Debt) | 5 | Tracked |
| Resolved since Mar 3 | 6 | P0-P5 items |

---

## Recently Resolved (P0-P5 Items — Mar 2026)

### RESOLVED: RBAC Module (was CRITICAL)
- **Location**: `backend/security/rbac.py`
- **Resolution**: Fully implemented with in-memory storage (default) and optional
  DB-backed persistence via SQLAlchemy session. All 5 methods functional:
  `has_permission()`, `get_user_roles()`, `assign_role()`, `revoke_role()`, `check_access()`.
  Roles: admin/analyst/user/viewer. Permissions: read/write/delete/admin.

### RESOLVED: Crypto Utils Module (was CRITICAL)
- **Location**: `backend/security/crypto_utils.py`
- **Resolution**: Fully implemented using Python `cryptography` library:
  `encrypt_data()` / `decrypt_data()` via Fernet (AES-128-CBC + HMAC),
  `generate_key_pair()` / `sign_data()` / `verify_signature()` via RSA-2048 PSS+SHA-256.

### RESOLVED: Weak Password Hashing (was HIGH)
- **Location**: `backend/security/password_manager.py`
- **Resolution**: Upgraded to bcrypt (work factor 12) via passlib CryptContext.
  Legacy PBKDF2-HMAC-SHA256 hashes are still verified for backward compatibility.
  `needs_rehash()` method triggers automatic upgrade on next login.
  `check_password_strength()` now scores on length + complexity (upper/lower/digit/special).

### RESOLVED: Trading Service Has No Router (was MEDIUM-HIGH)
- **Location**: `backend/api/routers/trading.py` (new file)
- **Resolution**: Created trading router with 3 endpoints: POST `/api/trading/validate`,
  POST `/api/trading/execute`, POST `/api/trading/impact`.

### RESOLVED: ML Router Underdeveloped (was MEDIUM)
- **Location**: `backend/api/routers/ml.py`
- **Resolution**: Expanded from 2 to 8 endpoints. Now covers predictions, model listing,
  model detail, and advanced ML operations with 15-minute Redis cache.

### RESOLVED: Frontend Dead Code and Disorganized Components (was LOW-MEDIUM)
- `EnhancedDashboard.tsx` (746 lines) deleted — confirmed dead code
- `CorrelationMatrix.tsx`, `EfficientFrontier.tsx`, `RiskDecomposition.tsx` relocated
  from root components/ to `frontend/web/src/components/portfolio/` (commit 46e7986)

---

## HIGH: Deployment Blockers

### 1. SSL Directory Empty
- **Severity**: HIGH
- **Location**: `infrastructure/docker/nginx/nginx-ssl.conf` references `ssl/fullchain.pem`,
  `ssl/privkey.pem`, `ssl/dhparam.pem`, `ssl/chain.pem`
- **Problem**: The `ssl/` directory is empty. Nginx will fail to start in production.
- **Certbot configured**: `docker-compose.production.yml` includes `certbot/certbot:v2.7.4`
  with auto-renewal cron. Initial certificate generation must be run manually once.
- **Fix**: `docker compose run certbot certonly --webroot -w /var/www/certbot -d yourdomain.com`
  Or generate self-signed certs for staging: `openssl req -x509 -newkey rsa:4096 -keyout privkey.pem -out fullchain.pem -days 365 -nodes`

### 2. CI Tests Non-Blocking
- **Severity**: HIGH
- **Location**: `.github/workflows/ci.yml` lines 311, 457
- **Problem**: `continue-on-error: true` on backend and frontend test steps. Failing tests
  do not block deployment.
- **Fix**: Remove `continue-on-error: true` from test steps. Verify 0 test failures first
  (currently 0 backend failures, 4 fixable frontend failures).

---

## MEDIUM: Quality and Completeness

### 3. CI Coverage Floor Misaligned
- **Severity**: MEDIUM
- **Location**: `.github/workflows/ci.yml` line 387
- **Problem**: Blocking coverage floor is 35%, far below the documented 80% target.
- **Fix**: Raise to 60% blocking, 80% advisory.

### 4. Vitest/Playwright Collision
- **Severity**: MEDIUM
- **Location**: `frontend/web/tests/e2e/auth.spec.ts`, `portfolio.spec.ts`
- **Problem**: Playwright E2E files collected by Vitest, causing `test.describe()` errors.
- **Fix**: Add `exclude: ['**/tests/e2e/**']` to Vitest config.

### 5. Frontend Test Coverage Gaps
- **Severity**: MEDIUM
- **Location**: `frontend/web/src/store/slices/`, `frontend/web/src/hooks/`, `frontend/web/src/services/`
- **Problem**: Redux slices (6), custom hooks (13), and API service layer have 0% test coverage.
- **Fix**: Add unit tests for Redux slices and hooks. Vitest with React Testing Library.

### 6. Frontend TypeScript Errors
- **Severity**: MEDIUM
- **Location**: Multiple frontend files
- **Details**: Zero @ts-ignore suppressions found (good). Actual compile errors may still
  exist, especially Socket.IO event handler typing in `usePortfolioWebSocket.ts`.
- **Fix**: Run `tsc --noEmit` to quantify exact errors. Fix Socket.IO generics.

### 7. LSTM Model Weights Not Persisted
- **Severity**: MEDIUM
- **Location**: `backend/ml/training/train_lstm.py`
- **Problem**: Training script exists but saved LSTM weights are absent from `ml_models/`.
  Only the scaler is saved. XGBoost, LightGBM, and Prophet are available as fallbacks.
- **Fix**: Run LSTM training or document as deferred.

### 8. TradingAgents Low Test Coverage
- **Severity**: MEDIUM
- **Location**: `backend/TradingAgents/` (39 files total)
- **Problem**: Only 3 test files cover core graph logic, agent state machines, signal
  processing, and 36 source files.
- **Fix**: Create comprehensive test suite covering trading graph, conditional logic,
  signal processing, and agent analysts.

### 9. Frontend Test Failures (4)
- **Severity**: LOW-MEDIUM
- **Location**: `pages/Dashboard.test.tsx` (1 failure), `pages/Portfolio.test.tsx` (3 failures)
- **Details**:
  - Dashboard: Test initializes `heatmap: []` but component requires non-empty array
  - Portfolio: Duplicate `role="tab"` elements with name `/analysis/i` after UI overhaul
- **Fix**: Update test mock data and use more specific selectors.

---

## LOW: Tech Debt

### 10. 30 Files >800 Lines
- **Top 5**: `recommendation_service.py` (1,234), `market_scanner.py` (1,211),
  `ml_models.py` (1,191), `unified_models.py` (1,168), `portfolio_service.py` (1,162)
- **Action**: Not urgent. Most are cohesive domain code. Monitor for further growth.

### 11. Slow Tests (>10 seconds)
- **Tests**: `test_memory_leak_detection` (62s), `test_api_retry_with_circuit_breaker` (60s),
  `test_batch_inference_performance` (17s)
- **Status**: Tagged with `@pytest.mark.slow` (P3 #19 complete). Excluded from fast CI runs.

### 12. Unawaited Coroutine Warnings
- **Count**: ~15+ instances across security modules
- **Files**: `security/rate_limiter.py`, `security/jwt_manager.py`
- **Fix**: Properly await async mock calls in tests.

### 13. Deprecated API Usage
- **`get_db()` deprecated**: Should use `get_async_db_session()`
- **`HTTP_422_UNPROCESSABLE_ENTITY` deprecated**: Renamed to `HTTP_422_UNPROCESSABLE_CONTENT`
  in Starlette

### 14. Four Charting Libraries in Frontend
- **Files**: Recharts + Plotly + Chart.js + Lightweight Charts
- **Impact**: Bundle size overhead
- **Action**: Evaluate consolidating to 2 libraries after feature stabilization

---

## Previously Resolved (Loki Mode Remediation — Feb 2026)

All items from the prior remediation remain resolved:
- [x] Utils directory sprawl: 87 -> 55 -> 61 files (sub-module splits)
- [x] Three competing ORM Base declarations: Unified in unified_models.py
- [x] Six dead ETL extractors: 4 files deleted
- [x] Triple-nested backend directory: Deleted
- [x] All routers under 750 lines via service extraction
- [x] JWT_ALGORITHM mismatch: Fixed to RS256
- [x] Duplicate docker-compose files: Consolidated
- [x] CI pipeline instability: Resolved
- [x] Test infrastructure patterns: Documented
- [x] Dockerfile path mismatch: `frontend/web/Dockerfile` exists

---

## Overall Health Summary

**Security**: STRONG — All three stubs implemented. CSP hardened. bcrypt passwords.
**Testing**: STRONG — 5,020 backend tests (0 failures), 197 frontend tests
**Architecture**: STRONG — Clean service layer, unified ORM, modular design
**CI/CD**: IMPROVING — Pipeline stable, tests still advisory (non-blocking)
**Frontend**: GOOD — Clean codebase, dead code removed, minor test coverage gaps
