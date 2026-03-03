# Identified Issues

**Last Updated**: 2026-03-03

## Issue Summary

| Priority | Count | Status |
|----------|-------|--------|
| Critical (Security Stubs) | 3 | NEW - Must fix before production |
| High (Deployment Blockers) | 4 | 2 existing, 2 new |
| Medium (Quality/Completeness) | 8 | Mix of new and existing |
| Low (Tech Debt) | 5 | Tracked |
| Previously Resolved | 22+ | Loki Mode remediation (Waves 1-14) |

---

## CRITICAL: Security Stubs (NEW - Mar 2026)

### 1. RBAC Module is a Stub
- **Severity**: CRITICAL
- **Location**: `backend/security/rbac.py`
- **Problem**: `get_user_roles()`, `assign_role()`, `revoke_role()`, `check_access()` all raise `NotImplementedError`
- **Impact**: No fine-grained permission enforcement. Falls back to `is_admin` boolean on User model. Any authenticated user can access any non-admin endpoint.
- **Fix**: Implement role-permission mapping with database-backed role assignments

### 2. Crypto Utils Module is a Stub
- **Severity**: CRITICAL
- **Location**: `backend/security/crypto_utils.py`
- **Problem**: `encrypt_data()`, `decrypt_data()`, `sign_data()`, `verify_signature()`, `generate_key_pair()` all raise `NotImplementedError`
- **Impact**: Any feature requiring field-level encryption silently breaks. PII protection at rest is non-functional.
- **Fix**: Implement using Fernet symmetric encryption or AES-GCM

### 3. Weak Password Hashing
- **Severity**: HIGH
- **Location**: `backend/security/password_manager.py`
- **Problem**: Uses `pbkdf2_hmac` with SHA-256 instead of bcrypt or argon2id. Password policy check is `len >= 8` only.
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

### 7. Frontend Dockerfile Path Mismatch (NEW)
- **Severity**: HIGH
- **Location**: `.github/workflows/production-deploy.yml` line 234
- **Problem**: CI references `./frontend/web/Dockerfile` but only `./Dockerfile.frontend` exists at repo root. Production image build will fail.
- **Fix**: Either move Dockerfile or update the workflow path reference.

---

## MEDIUM: Quality and Completeness

### 8. LSTM Model Weights Not Persisted
- **Severity**: MEDIUM
- **Location**: `backend/ml/training/train_lstm.py`
- **Problem**: Training script exists but saved LSTM weights are absent from `ml_models/`. Only the scaler is saved.
- **Fix**: Run LSTM training and save model weights, or document as deferred.

### 9. Frontend TypeScript Errors (15)
- **Severity**: MEDIUM
- **Location**: Multiple frontend files
- **Details**:
  - 6 type-safety errors in `usePortfolioWebSocket.ts` (Socket.IO event handler typing)
  - 9 unused import warnings across 6 files
- **Fix**: Fix `_addSocketListener`/`_removeSocketListener` generics. Remove unused imports. Install `@types/lodash`.

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

### 12. TradingAgents Zero Test Coverage
- **Severity**: MEDIUM
- **Location**: `backend/TradingAgents/` (~20 files)
- **Problem**: No dedicated unit tests for trading graph, agent state machines, signal processing.
- **Fix**: Create test suite covering core agent logic.

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

### 16. 30+ Files >500 Lines
- **Files**: Concentrated in `backend/security/` (14 files 500-1141 lines) and `backend/tasks/` (5 files 500-1112 lines)
- **Largest**: `security_config.py` (1,141 lines), `maintenance_tasks.py` (1,112 lines)
- **Action**: Not urgent. Security modules are cohesive. Consider extraction if they grow further.

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

### 20. Frontend Oversized Components
- **`EnhancedDashboard.tsx`**: 745 lines, not connected to router (potential dead code)
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
