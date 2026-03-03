# Identified Issues

**Last Updated**: 2026-03-03 (Refreshed with deep audit)

## Issue Summary

| Priority | Count | Status |
|----------|-------|--------|
| Critical (Security Stubs) | 3 | Must fix before production |
| High (Deployment Blockers) | 3 | 1 resolved, 2 remaining |
| Medium (Quality/Completeness) | 8 | Mix of new and existing |
| Low (Tech Debt) | 5 | Tracked |
| Previously Resolved | 23+ | Loki Mode + Dockerfile path fix |

---

## CRITICAL: Security Stubs (NEW - Mar 2026)

### 1. RBAC Module is a Stub
- **Severity**: CRITICAL
- **Location**: `backend/security/rbac.py` (lines 48-91)
- **Problem**: 4 of 5 core methods raise `NotImplementedError`: `get_user_roles()` (L48-57), `assign_role()` (L59-68), `revoke_role()` (L70-79), `check_access()` (L81-91). Only `has_permission()` (L43-46) works via static role-permission mapping.
- **Impact**: No fine-grained permission enforcement. Falls back to `is_admin` boolean on User model. Any authenticated user can access any non-admin endpoint.
- **Fix**: Implement role-permission mapping with database-backed role assignments

### 2. Crypto Utils Module is a Stub
- **Severity**: CRITICAL
- **Location**: `backend/security/crypto_utils.py` (lines 62-120)
- **Problem**: 5 methods raise `NotImplementedError`: `encrypt_data()` (L62-72), `decrypt_data()` (L75-83), `generate_key_pair()` (L86-96), `sign_data()` (L99-108), `verify_signature()` (L111-120). Working: `SecureRandom` (5 methods) and `CryptoUtils.hash_data()` (SHA256/512).
- **Impact**: Any feature requiring field-level encryption silently breaks. PII protection at rest is non-functional.
- **Fix**: Implement using Fernet symmetric encryption or AES-GCM, RSA/ECDSA for signing

### 3. Weak Password Hashing
- **Severity**: HIGH
- **Location**: `backend/security/password_manager.py` (line 24)
- **Problem**: Uses `hashlib.pbkdf2_hmac('sha256', ...)` with 100,000 iterations (L18). Password policy check is `len >= 8` only (L52-55). Lines 22, 42, 54 have TODO comments for bcrypt/argon2.
- **Impact**: Passwords are less resistant to GPU-based cracking attacks. No complexity requirements.
- **Fix**: Replace with `bcrypt` or `argon2id`. Add complexity validation (uppercase, digit, special char).

---

## HIGH: Deployment Blockers

### 4. SSL Directory Empty (Existing)
- **Severity**: HIGH
- **Location**: `infrastructure/docker/nginx/nginx-ssl.conf` references `ssl/fullchain.pem`, `ssl/privkey.pem`, `ssl/dhparam.pem`, `ssl/chain.pem`
- **Problem**: The `ssl/` directory is completely empty. Nginx will fail to start in production.
- **Fix**: Provision via Let's Encrypt with certbot container, or provide self-signed certs for staging.

### 5. CI Tests Non-Blocking (Existing)
- **Severity**: HIGH
- **Location**: `.github/workflows/ci.yml` lines 311, 457
- **Problem**: `continue-on-error: true` on backend and frontend test steps. Failing tests do not block deployment.
- **Fix**: Remove `continue-on-error: true` from test steps. Fix the 5 currently-failing tests first.

### 6. Trading Service Has No Router (NEW)
- **Severity**: MEDIUM-HIGH
- **Location**: `backend/services/trading_service.py`
- **Problem**: Order model is fully defined, trading_service.py is implemented, but no router exposes order creation, cancellation, or order book endpoints.
- **Fix**: Create `backend/api/routers/trading.py` with order CRUD endpoints.

### 7. ~~Frontend Dockerfile Path Mismatch~~ (RESOLVED)
- **Severity**: ~~HIGH~~ RESOLVED
- **Location**: `frontend/web/Dockerfile` now exists
- **Status**: `frontend/web/Dockerfile` has been created with valid Node.js Alpine multi-stage setup. CI path now resolves correctly.

---

## MEDIUM: Quality and Completeness

### 8. LSTM Model Weights Not Persisted
- **Severity**: MEDIUM
- **Location**: `backend/ml/training/train_lstm.py`
- **Problem**: Training script exists but saved LSTM weights are absent from `ml_models/`. Only the scaler is saved.
- **Fix**: Run LSTM training and save model weights, or document as deferred.

### 9. Frontend TypeScript Errors
- **Severity**: MEDIUM
- **Location**: Multiple frontend files
- **Details**:
  - Zero `@ts-ignore` or `@ts-expect-error` suppressions found (good type discipline)
  - Type-safety issues may still exist at compile time (Socket.IO event handler typing in `usePortfolioWebSocket.ts`)
  - Needs `tsc --noEmit` run to quantify actual error count
- **Fix**: Run `tsc --noEmit` to identify exact errors. Fix Socket.IO generics. Install `@types/lodash` if needed.

### 10. Frontend Test Failures (4)
- **Severity**: MEDIUM
- **Location**: `pages/Dashboard.test.tsx` (1 failure), `pages/Portfolio.test.tsx` (3 failures)
- **Details**:
  - Dashboard: Test initializes `heatmap: []` but component requires non-empty array
  - Portfolio: Duplicate `role="tab"` elements with name `/analysis/i` after UI overhaul
- **Fix**: Update test mock data and use more specific selectors.

### 11. Vitest/Playwright Collision
- **Severity**: MEDIUM
- **Location**: `frontend/web/tests/e2e/auth.spec.ts`, `portfolio.spec.ts`
- **Problem**: Playwright E2E files collected by Vitest, causing `test.describe()` errors.
- **Fix**: Add `exclude: ['**/tests/e2e/**']` to Vitest config.

### 12. TradingAgents Low Test Coverage
- **Severity**: MEDIUM
- **Location**: `backend/TradingAgents/` (39 files total)
- **Problem**: Only 3 test files exist (`test_agents_api.py`, `test_agents_to_recommendations_flow.py`, `test_agents_service.py`). Core graph logic, agent state machines, signal processing, and 36 source files largely untested.
- **Fix**: Create comprehensive test suite covering trading graph, conditional logic, signal processing, and agent analysts.

### 13. ML Router Underdeveloped
- **Severity**: MEDIUM
- **Location**: `backend/api/routers/ml.py`
- **Problem**: Only 2 endpoints (predictions POST, models GET) despite 48-file ML subsystem. Monitoring, drift detection, artifact management, backtesting all internal-only.
- **Fix**: Expand ML router to expose key pipeline capabilities.

### 14. Flaky Backend Test
- **Severity**: LOW-MEDIUM
- **Location**: `backend/tests/unit/test_data_ingestion.py` line 927
- **Test**: `TestSECEdgarClient::test_extract_section_returns_none_for_missing_section`
- **Problem**: Passes in isolation, fails in full suite run. Import-chain state + regex behavior with `\Z` anchor.
- **Fix**: Refactor to module-level imports or add `@pytest.mark.flaky(reruns=2)`.

### 15. Coverage Floor Misaligned
- **Severity**: MEDIUM
- **Location**: `.github/workflows/ci.yml` line 387
- **Problem**: Blocking coverage floor is 35%, far below the documented 80% target.
- **Fix**: Raise to 60% blocking, 80% advisory.

---

## LOW: Tech Debt

### 16. 30 Files >800 Lines
- **Files**: Distributed across services (2), security (6), ML (5), monitoring (4), ETL (2), utils (3), data_ingestion (1), models (2), TradingAgents (1), API (1), tasks (1), config (1)
- **Top 5**: `recommendation_service.py` (1,234), `market_scanner.py` (1,211), `ml_models.py` (1,191), `unified_models.py` (1,168), `portfolio_service.py` (1,162)
- **Action**: Not urgent. Most are cohesive domain code. Monitor for further growth.

### 17. Slow Tests (>10 seconds)
- **Tests**: `test_memory_leak_detection` (60s), `test_api_retry_with_circuit_breaker` (60s), `test_batch_inference_performance` (14s)
- **Fix**: Tag with `@pytest.mark.slow` and exclude from fast CI runs.

### 18. Unawaited Coroutine Warnings
- **Count**: ~15+ instances across security modules
- **Files**: `security/rate_limiter.py`, `security/jwt_manager.py`
- **Fix**: Properly await async mock calls in tests.

### 19. Deprecated API Usage
- **`get_db()` deprecated**: Should use `get_async_db_session()`
- **`HTTP_422_UNPROCESSABLE_ENTITY` deprecated**: Renamed to `HTTP_422_UNPROCESSABLE_CONTENT` in Starlette

### 20. Frontend Dead Code and Oversized Components
- **`EnhancedDashboard.tsx`**: 746 lines, CONFIRMED DEAD CODE (not imported anywhere in codebase). Features: drag-and-drop, undo/redo, export, fullscreen. Should be deleted or integrated.
- **`SettingsTabs.tsx`**: 586 lines, candidate for extraction
- **`PortfolioSummary.tsx`**: 534 lines, candidate for extraction

---

## Previously Resolved (Loki Mode Remediation - Feb 2026)

All items from the Feb 26 report remain resolved:
- [x] Utils directory sprawl: 87 -> 55 files
- [x] Three competing ORM Base declarations: Unified
- [x] Six dead ETL extractors: 4 files deleted
- [x] Triple-nested backend directory: Deleted
- [x] All routers under 750 lines via service extraction
- [x] JWT_ALGORITHM mismatch: Fixed to RS256
- [x] Duplicate docker-compose files: Consolidated
- [x] CI pipeline instability: Resolved
- [x] Test infrastructure patterns: Documented

---

## Overall Health Summary

**Security**: NEEDS ATTENTION - Three stubs must be implemented before production
**Testing**: STRONG - 5,132 tests, 99.98% backend pass rate
**Architecture**: STRONG - Clean service layer, unified ORM, modular design
**CI/CD**: IMPROVING - Pipeline stable but gates still advisory
**Frontend**: GOOD - Major refactor complete, minor fixups needed
