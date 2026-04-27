---
scope_id: "15-test-suite"
scope_name: "Test Suite (backend, frontend, integration)"
agent_type: "qa-expert"
date: "2026-04-27"
files_in_scope: 149
files_reviewed: 47
files_skipped:
  - "backend/tests/*.md — documentation files, not test code"
  - "backend/tests/fixtures/* — read as supporting context"
  - "tests/TEST_SUMMARY.md, TEST_METRICS.md, FILE_MANIFEST.md, QUICK_START.md, E2E_AND_INTEGRATION_TESTS.md — stub redirects confirmed by read"
prior_reports_validated:
  - path: "docs/TEST_REPORT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_REPORT.archived.md"
    claims_validated: 7
    claims_still_valid: 4
    claims_stale: 3
  - path: "docs/test-coverage-analysis-report.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/test-coverage-analysis-report.archived.md"
    claims_validated: 6
    claims_still_valid: 1
    claims_stale: 5
  - path: "docs/test-coverage-summary.txt"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/test-coverage-summary.archived.md"
    claims_validated: 4
    claims_still_valid: 1
    claims_stale: 3
  - path: "docs/test-fix-checklist.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/test-fix-checklist.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
  - path: "docs/test-fix-priority-matrix.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/test-fix-priority-matrix.archived.md"
    claims_validated: 5
    claims_still_valid: 3
    claims_stale: 2
  - path: "docs/pytest-asyncio-analysis.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/pytest-asyncio-analysis.archived.md"
    claims_validated: 8
    claims_still_valid: 3
    claims_stale: 5
  - path: "docs/pytest-asyncio-before-after.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/pytest-asyncio-before-after.archived.md"
    claims_validated: 4
    claims_still_valid: 2
    claims_stale: 2
  - path: "docs/pytest-asyncio-fixes-summary.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/pytest-asyncio-fixes-summary.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
  - path: "docs/pytest-asyncio-implementation-guide.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/pytest-asyncio-implementation-guide.archived.md"
    claims_validated: 3
    claims_still_valid: 1
    claims_stale: 2
  - path: "docs/QA_ACTION_PLAN.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/QA_ACTION_PLAN.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
  - path: "docs/README_VERIFICATION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/README_VERIFICATION.archived.md"
    claims_validated: 3
    claims_still_valid: 1
    claims_stale: 2
  - path: "docs/VERIFICATION_IMPLEMENTATION_SUMMARY.md"
    status: "unverifiable"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/VERIFICATION_IMPLEMENTATION_SUMMARY.archived.md"
    claims_validated: 2
    claims_still_valid: 0
    claims_stale: 2
  - path: "docs/VERIFICATION_QUICK_START.md"
    status: "unverifiable"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/VERIFICATION_QUICK_START.archived.md"
    claims_validated: 2
    claims_still_valid: 0
    claims_stale: 2
  - path: "docs/VERIFICATION_SCRIPTS_GUIDE.md"
    status: "unverifiable"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/VERIFICATION_SCRIPTS_GUIDE.archived.md"
    claims_validated: 2
    claims_still_valid: 0
    claims_stale: 2
  - path: "docs/VERIFICATION_SYSTEM_ARCHITECTURE.md"
    status: "unverifiable"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/VERIFICATION_SYSTEM_ARCHITECTURE.archived.md"
    claims_validated: 2
    claims_still_valid: 0
    claims_stale: 2
  - path: "docs/testing/INTEGRATION_TESTS.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/INTEGRATION_TESTS.archived.md"
    claims_validated: 4
    claims_still_valid: 2
    claims_stale: 2
  - path: "docs/testing/TEST_BASELINE_REPORT.md"
    status: "current"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_BASELINE_REPORT.archived.md"
    claims_validated: 6
    claims_still_valid: 5
    claims_stale: 1
  - path: "docs/testing/TEST_DOCUMENTATION_INDEX.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_DOCUMENTATION_INDEX.archived.md"
    claims_validated: 3
    claims_still_valid: 2
    claims_stale: 1
  - path: "docs/testing/TEST_EXECUTION_CHECKLIST.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_EXECUTION_CHECKLIST.archived.md"
    claims_validated: 4
    claims_still_valid: 2
    claims_stale: 2
  - path: "docs/testing/TEST_FAILURE_ANALYSIS.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_FAILURE_ANALYSIS.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
  - path: "docs/testing/TEST_VALIDATION_METRICS.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_VALIDATION_METRICS.archived.md"
    claims_validated: 5
    claims_still_valid: 1
    claims_stale: 4
  - path: "docs/testing/TESTING_GUIDE.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TESTING_GUIDE.archived.md"
    claims_validated: 4
    claims_still_valid: 2
    claims_stale: 2
  - path: "docs/reports/QUALITY_ASSURANCE_REPORT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/QUALITY_ASSURANCE_REPORT.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
  - path: "docs/reports/README_VALIDATION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/README_VALIDATION.archived.md"
    claims_validated: 3
    claims_still_valid: 1
    claims_stale: 2
  - path: "docs/reports/VALIDATION_EXECUTIVE_SUMMARY.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/VALIDATION_EXECUTIVE_SUMMARY.archived.md"
    claims_validated: 4
    claims_still_valid: 2
    claims_stale: 2
  - path: "docs/reports/VALIDATION_INDEX.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/VALIDATION_INDEX.archived.md"
    claims_validated: 3
    claims_still_valid: 1
    claims_stale: 2
  - path: "tests/TEST_SUMMARY.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_SUMMARY.archived.md"
    claims_validated: 1
    claims_still_valid: 0
    claims_stale: 1
  - path: "tests/TEST_METRICS.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TEST_METRICS.archived.md"
    claims_validated: 1
    claims_still_valid: 0
    claims_stale: 1
  - path: "tests/E2E_AND_INTEGRATION_TESTS.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/E2E_AND_INTEGRATION_TESTS.archived.md"
    claims_validated: 1
    claims_still_valid: 0
    claims_stale: 1
  - path: "tests/QUICK_START.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/QUICK_START.archived.md"
    claims_validated: 1
    claims_still_valid: 0
    claims_stale: 1
  - path: "tests/FILE_MANIFEST.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/FILE_MANIFEST.archived.md"
    claims_validated: 1
    claims_still_valid: 0
    claims_stale: 1
findings_summary:
  critical: 4
  high: 9
  medium: 9
  low: 5
  total: 27
estimated_remediation_effort_days: 14
agent_status: "complete"
agent_token_usage: 0
---

# Test Suite — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- The `tests/` root directory (3 Python test files, ~1,356 lines) and `frontend/web/tests/e2e/` (2 Playwright specs) are both **excluded from pytest discovery** — `pytest.ini` `testpaths = backend/tests` and `norecursedirs = ... tests ...` means `tests/security/*.py` and `tests/test_database_fixes.py` are dead test code never executed by any runner.
- **`test_performance_optimizations.py:34`** unconditionally imports `OptimizedRecommendationEngine` (a known-broken class per scope 09 — `scan_market_streaming` does not exist on `MarketScanner`) **outside** the `try/except ImportError` guard at lines 16–33; when `objgraph` is installed this causes a collection-time `AttributeError`/import error that silently kills all 50+ tests in that file.
- **`run-tests.sh:84`** invokes `npm test -- --watchAll=false --passWithNoTests`, passing Jest/CRA-specific flags to **vitest 4.x** — `--watchAll` and `--passWithNoTests` are unrecognised by vitest and cause the frontend test command to exit with error, meaning CI frontend tests are broken or silently bypassed.
- **Three deprecated `event_loop` fixture overrides** remain (`backend/tests/fixtures/integration_test_fixtures.py:650`, `backend/tests/middleware/test_response_optimizer.py:403`, `tests/security/conftest.py:14`) — with `asyncio_mode = auto` and pytest-asyncio 0.23.x these emit `DeprecationWarning` on every collection pass; additionally `requirements.txt` (0.23.8) and `requirements-dev.txt` (0.23.3) pin **different pytest-asyncio versions**, so dev and CI environments run different event-loop semantics.
- **1,622 redundant `@pytest.mark.asyncio` decorators** remain across 113 test files; with `asyncio_mode = auto` set in both `pytest.ini` and `pyproject.toml` these are harmless noise but conflict with documentation claiming they are required, slowing onboarding and obscuring the canonical async test pattern.

> Read these 5 before anything else in this report.

---

## 1. Scope & Files Reviewed

**Paths covered:**
- `backend/tests/**` — 113 `test_*.py` files plus conftest, fixture modules, middleware/ and security/ subdirs
- `frontend/web/tests/**` — 2 Playwright E2E specs (`auth.spec.ts`, `portfolio.spec.ts`)
- `tests/**` — 3 test `.py` files + 5 `.md` stub redirects
- `pytest.ini`, `run-tests.sh`, `run_integration_tests.py`, `test_performance.sh`
- Root-level `frontend/web/playwright.config.ts`, `frontend/web/vite.config.ts`, `frontend/web/src/` vitest test files (16)

**Files explicitly excluded:** `.md` documentation inside `backend/tests/` (COVERAGE_ANALYSIS.md, README.md, etc.) — these are supporting docs, not runnable tests.

**File counts:**
- backend/tests `test_*.py` files: 113 (confirmed by `find`)
- Total test functions discovered: 5,222 (confirmed by grep `def test_`)
- Async test functions: 1,771 (34% of total)
- Frontend vitest test files in `src/`: 16

---

## 2. Prior Report Reconciliation

### §2.A — Coverage & Pass-Rate Priors (Grouped)

**Files:** `docs/TEST_REPORT.md`, `docs/test-coverage-analysis-report.md`, `docs/test-coverage-summary.txt`, `docs/testing/TEST_VALIDATION_METRICS.md`, `docs/reports/QUALITY_ASSURANCE_REPORT.md`

**Validation method:** Compared claimed test counts against `find backend/tests -name "test_*.py" | wc -l` (returns 113) and `grep -rn "def test_" | wc -l` (returns 5,222). For coverage claims, compared to `TEST_BASELINE_REPORT.md` (2026-03-04, 5020+ passing).

**Archived to:** Five individual `.archived.md` files in `docs/audits/2026-04/_meta/prior-reports-archive/`.

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "10/38 tests passing (26.3%)" | test-coverage-analysis-report.md | fully_stale | `find backend/tests -name "test_*.py" \| wc -l` → 113 files; TEST_BASELINE_REPORT (2026-03-04) states "5020+ passing" |
| 2 | "Overall Coverage: 10.51%" | test-coverage-analysis-report.md | fully_stale | Was integration-only snapshot from 2026-01-28; suite grew from 38 to 113+ files |
| 3 | "27 test files, READY FOR EXECUTION" | TEST_VALIDATION_METRICS.md (2026-01-27) | fully_stale | `find backend/tests -name "test_*.py" \| wc -l` → 113 (not 27) |
| 4 | "Schema Errors: 0 (all 7 fixed)" | test-coverage-analysis-report.md | current | `conftest.py` uses `unified_models.Base`; no schema errors in import chain |
| 5 | "Overall Score: 87/100 PRODUCTION READY" | QUALITY_ASSURANCE_REPORT.md (2026-01-27) | partially_stale | Score was assessed against 27 test files; current suite is 4x larger; critical import bugs (F-15-001, F-15-003) now invalidate "production ready" |
| 6 | "Analytics/technical_analysis.py 7.03% coverage" | test-coverage-summary.txt | partially_stale | `unit/test_analytics_extended_agent4.py` added since 2026-01-28; exact current % unknown without live run |

---

### §2.B — pytest-asyncio Configuration Priors (Grouped)

**Files:** `docs/pytest-asyncio-analysis.md`, `docs/pytest-asyncio-before-after.md`, `docs/pytest-asyncio-fixes-summary.md`, `docs/pytest-asyncio-implementation-guide.md`, `docs/test-fix-checklist.md`, `docs/test-fix-priority-matrix.md`

**Validation method:** Direct file reads of `pytest.ini`, `backend/tests/conftest.py`, `backend/tests/fixtures/integration_test_fixtures.py`, `backend/tests/middleware/test_response_optimizer.py`, `tests/security/conftest.py`; grep for `event_loop` and `asyncio_mode`.

**Grouped archived to:** Four `.archived.md` files in prior-reports-archive.

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | "asyncio_mode = auto missing from pytest.ini" | fully_stale | `pytest.ini:58` reads `asyncio_mode = auto`; also confirmed in `pyproject.toml:83` |
| C2 | "Custom event_loop at conftest.py:31-37 (session scope)" | fully_stale | `grep -c "event_loop" backend/tests/conftest.py` → 0; conftest.py has no event_loop fixture |
| C3 | "async_fixtures.py:213 has event_loop" | fully_stale | `ls backend/tests/async_fixtures.py` → MISSING; file does not exist |
| C4 | "integration_test_fixtures.py:650 has deprecated event_loop" | current | `grep -n "event_loop" fixtures/integration_test_fixtures.py` → line 650 `@pytest.fixture(scope="session") def event_loop()` confirmed |
| C5 | "test_response_optimizer.py has event_loop" | current | `grep -n "event_loop" middleware/test_response_optimizer.py` → line 403 `@pytest.fixture def event_loop()` confirmed |
| C6 | "Session-scoped async fixtures incompatible with pytest-asyncio 1.3.0+" | partially_stale | Priors reference non-existent version "1.3.0+"; actual installed is 0.23.8 (req.txt) / 0.23.3 (req-dev.txt). Session-scoped event_loop IS deprecated since 0.21.x. Risk is real but description is inaccurate. |
| C7 | "1622 explicit @pytest.mark.asyncio redundant with asyncio_mode=auto" | current | `grep -rn "pytest.mark.asyncio" backend/tests/ \| wc -l` → 1622 (not harmful but noisy) |
| C8 | "pytest-asyncio version split (req.txt vs req-dev.txt)" | current | `grep pytest-asyncio requirements.txt requirements-dev.txt` → req.txt=0.23.8, req-dev.txt=0.23.3 |

---

### §2.C — Test Infrastructure & Baseline Priors

**Files:** `docs/testing/TEST_BASELINE_REPORT.md`, `docs/testing/TEST_FAILURE_ANALYSIS.md`, `docs/testing/TESTING_GUIDE.md`, `docs/QA_ACTION_PLAN.md`, `docs/README_VERIFICATION.md`, and 4 VERIFICATION_* docs.

| # | Claim | Source | Status | Evidence |
|---|---|---|---|---|
| D1 | "5020+ passing tests, 0 failures, 8 skipped, 2 xfailed" | TEST_BASELINE_REPORT.md (2026-03-04) | current | `grep -rn "def test_" backend/tests/ \| wc -l` → 5,222 (suite grew ~200 tests after 2026-03-04) |
| D2 | "testpaths = backend/tests" covers all tests | TEST_BASELINE_REPORT.md | partially_stale | pytest.ini `norecursedirs` explicitly lists `tests`, meaning `tests/security/*.py` and `tests/test_database_fixes.py` are never discovered |
| D3 | "Coverage Requirement: 85% (per pytest.ini)" | TEST_BASELINE_REPORT.md | partially_stale | `pytest.ini` comment has `--cov-fail-under=60`; run_integration_tests.py uses 75; README.md says 85. Three inconsistent thresholds. |
| D4 | "WebSocket latency testing <2s requirement" | TEST_FAILURE_ANALYSIS.md | current | `backend/tests/test_websocket_integration.py` exists (22 test functions); latency tests present |
| D5 | "VERIFICATION_* system describes runtime verification scripts" | 4 VERIFICATION docs | unverifiable | Scripts referenced by VERIFICATION_SCRIPTS_GUIDE.md not found in `scripts/`; claims cannot be validated from filesystem state |
| D6 | "tests/ root has test files referenced by documentation" | tests/TEST_SUMMARY.md etc. | fully_stale | All 5 docs in tests/ are stub redirects: content is literally "This file has been removed. See backend/tests/README.md" |

---

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-15-001 | critical | bug | backend/tests/test_performance_optimizations.py:34 | OptimizedRecommendationEngine import outside skip guard causes collection failure | `pytest.importorskip("objgraph")` at line 15 guards lines 16–33, but line 34 `from backend.analytics.recommendation_engine import OptimizedRecommendationEngine` is outside the try/except. When objgraph is installed, pytest reaches line 34 and the import of `OptimizedRecommendationEngine` triggers `AttributeError` because `_scan_market_optimized` calls `scanner.scan_market_streaming` which does not exist on `MarketScanner` (confirmed by scope 09 F-09-002). All 50+ tests in this file are silently killed at collection time. | Move line 34 inside the try block (lines 16–33) or add a separate `try/except ImportError` around it with `pytest.skip`. | `pytest --collect-only backend/tests/test_performance_optimizations.py` returns 50+ test items without errors | 1 | true | ["09-analytics"] |
| F-15-002 | critical | bug | run-tests.sh:84 | Jest flags passed to Vitest break frontend CI test run | `npm test -- --watchAll=false --passWithNoTests` passes Jest/CRA-specific flags to vitest 4.x. Vitest 4.x does not recognise `--watchAll` or `--passWithNoTests`; unrecognised CLI options cause vitest to exit with a non-zero error code, making the `--frontend` and `--all` modes of `run-tests.sh` fail or silently skip all frontend unit tests. The correct vitest equivalent is `vitest run`. | Replace line 84 with: `if npm run test -- --run; then` (vitest recognises `--run` for single-pass CI mode). Remove `--watchAll=false --passWithNoTests`. | `./run-tests.sh --frontend` exits 0 and shows vitest test results for 16 test files in src/ | 1 | true | ["12-frontend"] |
| F-15-003 | critical | broken_import | tests/security/test_phase2_improvements.py:1, tests/security/test_row_locking.py:1, tests/test_database_fixes.py:1 | Three Python test files (1,356 lines) excluded from pytest discovery — dead test code | `pytest.ini` `testpaths = backend/tests` and `norecursedirs = ... tests ...` explicitly exclude the `tests/` root directory. The three files in `tests/security/` and `tests/test_database_fixes.py` are never discovered or executed. `run-tests.sh` and `run_integration_tests.py` also only target `backend/tests/`. These 1,356 lines of tests cover admin API security, row-locking, and database schema fixes — all critical paths. | Either move the files into `backend/tests/security/` and `backend/tests/` respectively, or add `tests` to `pytest.ini:testpaths`. Also remove `tests` from `norecursedirs`. | `pytest --collect-only tests/` returns test items; `pytest tests/ -q` passes without import errors | 3 | true | ["08-auth-security-compliance", "07-database-persistence"] |
| F-15-004 | critical | testing_gap | backend/tests/unit/test_realtime_price_service.py | FinnhubWebSocketClient.connect() session-lifetime bug has no direct test coverage | Scope 02 (F-02-002) confirmed that `FinnhubWebSocketClient.connect()` opens `aiohttp.ClientSession` inside `async with`, schedules `_receive_task`, then exits the `async with` block — closing the session immediately on every connect. The test file (`test_realtime_price_service.py`) has `TestFinnhubWebSocketClientInit`, `TestSubscribeUnsubscribe`, `TestHandleConnectionError`, and `TestDisconnect` — but **no `TestConnect` class**. All reconnect tests (`TestHandleConnectionError:547-641`) mock `.connect()` via `patch.object(ws_client, "connect", new_callable=AsyncMock)`, meaning the actual session-lifetime bug is never exercised. | Add `TestConnect` class testing `connect()` with a real or minimally-mocked `aiohttp.ClientSession` to verify that the `ws` attribute remains open after the `async with` block exits. | New test `test_session_stays_open_after_connect` fails before fix and passes after fix | 4 | true | ["02-backend-services-domain"] |
| F-15-005 | high | testing_gap | backend/tests/test_recommendation_engine.py:203-215 | ranking_score test uses Mock(spec=StockRecommendation) masking the real dataclass bug | `StockRecommendation` dataclass has no `ranking_score` field (confirmed by scope 09 F-09-003: it's set via dynamic attribute assignment in `rank_recommendations`). `test_rank_recommendations` at line 199–215 uses `Mock(spec=StockRecommendation)` which allows any attribute access — `hasattr(r, 'ranking_score')` returns True on `Mock` regardless of whether the real dataclass has the field. The test therefore cannot catch the production bug. Same pattern in `test_analytics_extended_agent4.py:839-846`. | Replace `Mock(spec=StockRecommendation)` with real `StockRecommendation` instances. The test will fail, surfacing the missing field. Then add `ranking_score: float = 0.0` to the `StockRecommendation` dataclass as the proper fix. | `pytest backend/tests/test_recommendation_engine.py::TestRecommendationEngine::test_rank_recommendations -v` passes using real dataclass instances | 3 | true | ["09-analytics"] |
| F-15-006 | high | testing_gap | backend/tests/security/test_security_modules.py:73-74 | JWT manager fixture uses MagicMock — no test covers actual JWTManager.create_token with RS256 key | `TestJWTManagerTokenCreation.jwt_manager_mock` (line 72–91) creates a `MagicMock()` object, not a real `JWTManager`. No test in `backend/tests/security/` or `backend/tests/test_security_compliance.py` exercises the production `JWTManager` with the RSA key pair from `security_config.py`. Scope 08 (F-08-001) found hardcoded salt in `secrets_manager.py` that undermines all stored JWT keys; scope 01 (F-01-001) found the RS256/HS256 algorithm confusion. Neither risk is covered by a test that instantiates the real class. | Add integration-level test that calls `get_jwt_manager()` (the production factory) and verifies `create_token` + `validate_token` roundtrip succeeds with RS256 and fails gracefully when algorithm is wrong. | `pytest backend/tests/security/test_security_modules.py -k "jwt" -v` exercises real JWTManager | 4 | true | ["08-auth-security-compliance", "01-backend-api"] |
| F-15-007 | high | bug | backend/tests/test_security_compliance.py:74 | TestAuthenticationSecurity.jwt_manager fixture instantiates JWTManager with plain string secret | `JWTManager(secret_key="test_secret_key_12345678901234567890")` at line 74 creates a manager with an HS256-style string key. The production `JWTManager` uses RS256 with RSA key objects (confirmed by conftest.py:332 which uses `jwt_mgr.private_key`). The test therefore validates a different algorithm than production, meaning HS256-specific token behaviors (e.g., padding issues, algorithm confusion attacks) are tested against a configuration that doesn't match deployment. | Update test fixture to use `get_jwt_manager()` from `backend.security.jwt_manager` or generate a test RSA key pair (as `backend/tests/security/test_security_modules.py:59-70` already does for `rsa_keys` fixture). | `pytest backend/tests/test_security_compliance.py::TestAuthenticationSecurity -v` passes with RS256-based fixture | 2 | true | ["08-auth-security-compliance"] |
| F-15-008 | high | broken_dependency | backend/tests/test_database_integration.py:21-26 | Docker guard makes entire integration test file silently skip in all non-Docker environments | Lines 21–26 run `subprocess.run(["docker", "info"])` and call `pytest.skip(..., allow_module_level=True)` if Docker is unavailable. In CI (GitHub Actions without Docker socket), in dev containers, and in local macOS development without Docker Desktop running, all 9 tests in this file are silently skipped. This is the only real-database integration test file for PostgreSQL. The skip is silent (no warning to CI log). | Add an `--skip-docker` CLI flag or environment variable so CI can explicitly report a skip vs a pass; add a non-Docker integration test using `sqlite+aiosqlite:///:memory:` for the same CRUD and transaction scenarios. | `pytest backend/tests/test_database_integration.py -v` shows clear SKIP reason message; at least 1 test runs in SQLite mode | 3 | true | ["07-database-persistence"] |
| F-15-009 | high | architecture | pytest.ini:22, run_integration_tests.py:205, backend/tests/README.md:56 | Three inconsistent coverage thresholds: 60% (pytest.ini comment), 75% (run_integration_tests.py), 85% (README.md) | `pytest.ini` comment on line 22 shows `--cov-fail-under=60` (commented out). `run_integration_tests.py:205` hardcodes `"--cov-fail-under=75"`. `backend/tests/README.md:56` states "Minimum line coverage: 85%". The active `--cov-fail-under` in CI (`run_integration_tests.py`) is 75%, while the team documentation target is 85%. This means CI can pass at 75% while the stated target is 85% — a 10-point gap that may hide coverage regressions. | Decide on a single threshold (85% recommended per README.md); add `--cov-fail-under=85` to `pytest.ini:addopts` (uncomment the coverage block) so it applies to every test run uniformly. Remove the commented-out line and update `run_integration_tests.py`. | `pytest backend/tests/ --cov=backend --cov-fail-under=85` is the single canonical coverage gate | 2 | true | [] |
| F-15-010 | high | testing_gap | backend/tests/test_integration_comprehensive.py:30 | `from backend.models.database import Base` imports stale Base that may miss new models | `test_integration_comprehensive.py:30` imports `Base` from `backend.models.database`, while all other integration tests (including the main `conftest.py:105`) import `Base` from `backend.models.unified_models`. The `models/database.py` file exists but reading it shows it may not export the canonical `Base` (it has no `class Base` at top level based on search results). If `database.Base` is an older or empty declarative base, the testcontainer-based tests will create an incomplete schema, making the entire file's DB-backed tests unreliable even when Docker is available. | Change line 30 to `from backend.models.unified_models import Base` (consistent with conftest.py:105 and all other integration tests). | `pytest --collect-only backend/tests/test_integration_comprehensive.py` succeeds; schema creation includes all tables | 1 | true | ["07-database-persistence"] |
| F-15-011 | high | testing_gap | backend/tests/unit/test_repositories_unit.py | AsyncBaseRepository.transaction() async generator bug (F-07-002) has zero test coverage | Scope 07 (F-07-002) confirmed that `AsyncBaseRepository.transaction()` defines `_execute_transaction` as an async generator but passes it to `execute_with_retry` which calls `await operation()` — awaiting an async generator yields the generator object, not the result. `test_repositories_unit.py` covers `AsyncCRUDRepository`, `PortfolioRepository`, `PriceHistoryRepository`, and `StockRepository` using mocked sessions — but the `transaction()` method itself is never called in any test. `grep -rn "transaction()\|execute_with_retry" backend/tests/` returns no results. | Add unit tests for `AsyncBaseRepository.transaction()` that verify a transaction executes its body, commits on success, and rolls back on exception. These tests will initially expose the async generator bug. | `pytest backend/tests/unit/test_repositories_unit.py -k "transaction" -v` passes after fix | 3 | true | ["07-database-persistence"] |
| F-15-012 | high | security | test_performance.sh:15 | Hardcoded default psql credentials in performance test script | `test_performance.sh:15` runs `PGPASSWORD=postgres psql -h localhost -U postgres -d investment_db`. The password `postgres` is a default/trivial credential; while this is a local test script, its presence in the repo as a committed file sets a bad pattern and may be copied into CI scripts. Scope 17 found real production credentials committed elsewhere — this file adds to the surface. | Replace with `PGPASSWORD="${TEST_DB_PASSWORD:-postgres}"` reading from environment, and add a comment that this is a local-only script not for CI. Add to `.gitignore` if it contains environment-specific config. | `grep "PGPASSWORD=postgres" test_performance.sh` returns no hardcoded credential | 1 | true | ["17-scripts-tooling"] |
| F-15-013 | medium | architecture | tests/security/conftest.py:13-18 | Orphaned `tests/security/conftest.py` has session-scoped event_loop that also conflicts with asyncio_mode=auto | `tests/security/conftest.py` lines 13–18 define `@pytest.fixture(scope="session") def event_loop()` — the same deprecated pattern identified in the pytest-asyncio priors. Since `tests/` is excluded from pytest discovery (`pytest.ini:norecursedirs`), this is currently inert. If the `tests/` directory is ever added to `testpaths`, this fixture will conflict with the `asyncio_mode = auto` setting and generate ScopeMismatch warnings or collection errors. | Remove the `event_loop` fixture from `tests/security/conftest.py`; rely on `asyncio_mode = auto`'s automatic management. | After fix: `pytest tests/security/ -v` collects without DeprecationWarning about event_loop scope | 1 | true | [] |
| F-15-014 | medium | code_quality | backend/tests/fixtures/integration_test_fixtures.py:650-654 | Deprecated session-scoped event_loop fixture still present in active fixture module | `integration_test_fixtures.py:650-654` defines `@pytest.fixture(scope="session") def event_loop()` creating a new event loop manually. With pytest-asyncio 0.23.x and `asyncio_mode = auto`, overriding `event_loop` at session scope is deprecated (`DeprecationWarning: There is no current event loop`). This file IS imported by its package `__init__.py` making it active. Every pytest collection pass will emit this warning. | Remove lines 650–654 from `integration_test_fixtures.py`. pytest-asyncio 0.23.x with `asyncio_mode = auto` manages the event loop automatically. | `pytest backend/tests/ -W error::DeprecationWarning` runs without warnings from event_loop | 1 | true | [] |
| F-15-015 | medium | code_quality | backend/tests/middleware/test_response_optimizer.py:402-407 | Deprecated event_loop fixture in middleware test file | `test_response_optimizer.py:402-407` defines another `@pytest.fixture def event_loop()` (function-scoped this time, not session). While function-scoped is less harmful than session-scoped, overriding `event_loop` at all is deprecated in pytest-asyncio ≥ 0.21 with `asyncio_mode = auto`. The comment "Add asyncio import for async tests" at line 398 suggests this was added as a workaround. | Remove lines 398–407; the module-level `asyncio` import needed for the existing sync tests can remain at the top of the file. | `pytest backend/tests/middleware/test_response_optimizer.py -W error::DeprecationWarning` passes cleanly | 1 | true | [] |
| F-15-016 | medium | stale_code | backend/tests/unit/test_ml_pipeline.py:180,191 | `asyncio.get_event_loop().run_until_complete()` anti-pattern in sync test methods | `test_ml_pipeline.py:180` and `:191` call `asyncio.get_event_loop().run_until_complete(...)` inside synchronous test methods. With Python 3.10+ and pytest-asyncio 0.23.x, `asyncio.get_event_loop()` emits a DeprecationWarning when called without a running event loop in a non-main thread. The correct pattern with `asyncio_mode = auto` is to mark the test `async def test_...` and `await` directly. | Convert `test_step_validate_input_returns_true` (line 178) and `test_step_cleanup_noop` (line 183) to `async def` test methods. Remove `run_until_complete` wrappers. | `pytest backend/tests/unit/test_ml_pipeline.py::TestPipelineStepBase -W error::DeprecationWarning` passes | 1 | true | [] |
| F-15-017 | medium | testing_gap | backend/tests/ | No tests exercise the known-broken `RecommendationService` vs mixin divergence (F-02-001) | Scope 02 (F-02-001) found that `RecommendationService` (1,234 LOC) duplicates all methods from `RecommendationCrudMixin` and `RecommendationAnalysisMixin` which are dead code. `test_recommendation_service.py` (83 tests) covers `RecommendationService` directly and mocks its engines — but no test imports or exercises `RecommendationCrudMixin` or `RecommendationAnalysisMixin` to verify they are either equivalent or can be removed. | Add smoke-import tests: `from backend.services.recommendation_crud import RecommendationCrudMixin` etc., plus a test that verifies method signatures match `RecommendationService` (or confirms the mixins can be deleted). | `pytest backend/tests/unit/test_recommendation_service.py -k "mixin" -v` passes | 2 | true | ["02-backend-services-domain"] |
| F-15-018 | medium | testing_gap | backend/tests/ | PBKDF2 hardcoded salt vulnerability (F-08-001) has no regression test | Scope 08 (F-08-001) found `secrets_manager.py:173-180` uses hardcoded salt `b"investment_analysis_salt"` with 100k PBKDF2 iterations. `test_security_compliance.py:507-522` calls `SecretsManager().store_api_key(...)` and verifies roundtrip — but no test asserts that the salt is random (per-invocation or per-key) or that iterations meet a minimum (NIST 2023 minimum: 600k). | Add a test `test_pbkdf2_uses_random_salt_and_sufficient_iterations` that verifies `SecretsManager` uses a unique salt per encryption operation and iteration count ≥ 600000. Test will fail until F-08-001 is fixed. | `pytest backend/tests/security/ -k "pbkdf2" -v` passes | 2 | true | ["08-auth-security-compliance"] |
| F-15-019 | medium | doc_drift | backend/tests/README.md:56-59, pytest.ini:22 | README.md coverage targets (85%/95%/100%) contradict active CI configuration | `backend/tests/README.md:56-59` states minimums of 85% (line), 95% (critical path), 100% (security), 90% (error handling). The active coverage enforcement in `run_integration_tests.py:205` is 75%, and `pytest.ini` comments reference 60%. None of the higher targets in README.md are enforced by any runnable command. This creates a false sense of coverage compliance. | Either enforce the 85% minimum in `pytest.ini:addopts` (removing the comment marker) or update README.md to reflect what is actually enforced. Document which targets are aspirational and which are gates. | The enforced `--cov-fail-under` value in pytest.ini matches the README.md stated minimum | 1 | true | [] |
| F-15-020 | medium | testing_gap | backend/tests/test_database_integration.py | No SQLite-based fallback for critical database integration tests | `test_database_integration.py` guards all 9 tests behind `pytest.importorskip("testcontainers")` and a Docker availability check. When either is absent — which is the case in lightweight CI (GitHub Actions free tier, devcontainers without Docker-in-Docker) — all database integration tests are silently skipped. The SQLite in-memory backend used by `conftest.py` is capable of exercising CRUD, transaction rollback, and integrity error scenarios without Docker. | Extract 3–5 critical CRUD and transaction tests into a separate `test_database_sqlite.py` using `sqlite+aiosqlite:///:memory:` (same pattern as conftest.py) with no Docker dependency. | `pytest backend/tests/test_database_sqlite.py -v` passes in any environment without Docker | 4 | true | ["07-database-persistence"] |
| F-15-021 | medium | architecture | run_integration_tests.py:35-49 | IntegrationTestRunner hard-codes 6 integration test files; ignores entire backend/tests/integration/ subdirectory | `run_integration_tests.py:35-49` defines `self.test_categories["all"]` as exactly 6 files: `test_api_integration.py`, `test_database_integration.py`, `test_data_pipeline_integration.py`, `test_websocket_integration.py`, `test_security_integration.py`, `test_resilience_integration.py`. The `backend/tests/integration/` subdirectory contains 12 test files (`test_analysis_router.py`, `test_auth_flow_complete.py`, `test_domain_contracts.py`, etc.) — none of which are referenced by the runner. These 12 files are also not reachable via `pytest backend/tests/integration/` because `run_integration_tests.py` only passes the 6 hardcoded paths. | Update `run_integration_tests.py` to discover tests dynamically: `test_files = list((self.test_dir / "integration").glob("test_*.py"))` for the `integration` category, and `list(self.test_dir.rglob("test_*.py"))` for `all`. | `python run_integration_tests.py --categories integration` discovers all 12+ files in `backend/tests/integration/` | 2 | true | [] |
| F-15-022 | medium | better_pattern | backend/tests/test_integration_comprehensive.py:174-176 | 174 test methods using deprecated @pytest.mark.asyncio within a file that already has asyncio_mode=auto | `test_integration_comprehensive.py:174` uses `@pytest.mark.asyncio` — this file has 9 such redundant decorators. Across the full suite, 1,622 occurrences exist. While harmless in 0.23.x, they become an error in `asyncio_mode = strict` (the more defensible long-term mode) and mislead developers into thinking the decorator is required. | Run `sed -i 's/@pytest.mark.asyncio\n//g'` across the test suite (or use a codemod) to remove all redundant decorators. Consider migrating to `asyncio_mode = strict` for explicit control. | `grep -rn "pytest.mark.asyncio" backend/tests/ \| wc -l` returns 0 | 4 | true | [] |
| F-15-023 | low | doc_drift | tests/TEST_SUMMARY.md, tests/TEST_METRICS.md, tests/E2E_AND_INTEGRATION_TESTS.md, tests/QUICK_START.md, tests/FILE_MANIFEST.md | Five stub-redirect files in tests/ root add no value and are excluded from pytest | All five files contain only a one-line redirect to `backend/tests/README.md`. They are in the `tests/` directory which is excluded from pytest discovery and not linked from any active documentation index. They consume search/index bandwidth without providing content. | Delete all five files; add a `tests/README.md` that explains the root-level test/ is currently a placeholder for future integration test relocation, with a pointer to `backend/tests/README.md`. | `ls tests/*.md \| wc -l` returns ≤ 1 (just README.md if kept) | 0.5 | true | [] |
| F-15-024 | low | performance | backend/tests/conftest.py:460-477 | `setup_test_environment` autouse fixture re-monkeypatches envvars on every test function | `setup_test_environment` at line 460 is `autouse=True` and `scope=function` (default). It iterates over 8 environment variables and calls `monkeypatch.setenv` for each — but these same variables are already set at module load time (lines 7–12). The fixture runs ~5,222 times total per test session, adding ~40K no-op `os.environ` mutations with rollback overhead. | Move the module-level `os.environ` assignments (lines 7–12) to the conftest module scope only, and change `setup_test_environment` to `scope="session"` or remove it entirely since the OS env is already set before any tests run. | Benchmark: `pytest backend/tests/ --co -q` (collection only) is at least 15% faster | 1 | true | [] |
| F-15-025 | low | testing_gap | frontend/web/src/ | Frontend vitest suite covers 16 files but has no API integration layer tests | The 16 vitest test files in `frontend/web/src/` cover page-level components (`Dashboard.test.tsx`, `Portfolio.test.tsx`, etc.) and hooks, all mocking the `apiService` completely. No test exercises the `api.service.ts` module itself — error handling, retry logic, auth header injection, or the `401 → logout` refresh flow. | Add `src/services/__tests__/api.service.test.ts` exercising at least: successful GET, 401 trigger, network error, and token refresh path. | `vitest run src/services/__tests__/api.service.test.ts` passes | 3 | false | ["12-frontend"] |
| F-15-026 | low | stale_code | backend/tests/benchmark_n1_query_fix.py | Benchmark script is not a pytest file but lives in the pytest test directory | `benchmark_n1_query_fix.py` is a standalone script (uses `asyncio.run()`, no test functions) that must be invoked as `python -m backend.tests.benchmark_n1_query_fix`. It starts with module docstring saying "Usage: python -m..." yet it lives inside the `testpaths` directory. pytest will attempt to collect it (`python_files = test_*.py` skips it due to naming, but `--collect-all` would surface it). It belongs in `scripts/` not `backend/tests/`. | Move to `scripts/performance/benchmark_n1_query_fix.py`. | `pytest backend/tests/ --collect-only` returns no items from benchmark_n1_query_fix.py | 0.5 | true | [] |
| F-15-027 | low | doc_drift | backend/tests/README.md:56, backend/tests/README.md:382 | "Security code coverage: 100%" target is stated twice but not enforced | README.md lines 56 and 382 both state "Security code coverage: 100%". `backend/tests/security/` and `backend/tests/test_security_compliance.py` mock the core production classes rather than exercising them (F-15-006, F-15-007). 100% security coverage is not achievable with current mock-heavy approach. The claim misleads future maintainers. | Change to "Security code coverage: ≥ 90% (enforced)" and back it up with a `--cov=backend.security --cov-fail-under=90` marker in a dedicated security test run. | `pytest backend/tests/security/ --cov=backend.security --cov-report=term-missing` reports ≥ 90% | 1 | true | [] |

---

## 4. Cross-Scope Linkages

- **F-15-001** → scope 09 (`backend/analytics/recommendation_engine.py`) — `OptimizedRecommendationEngine._scan_market_optimized` calls `scanner.scan_market_streaming` which does not exist; the test failure is a direct symptom of this broken production code.
- **F-15-002** → scope 12 (frontend) — the broken run-tests.sh frontend command means 16 frontend vitest files are never validated in `--all` mode.
- **F-15-003** → scope 08 (auth-security) and scope 07 (database-persistence) — `tests/security/test_phase2_improvements.py` and `tests/security/test_row_locking.py` cover security behaviors documented in scope 08's priors; `tests/test_database_fixes.py` imports from `backend.utils.async_database_fixed` which is within scope 07's concern.
- **F-15-004** → scope 02 (backend-services-domain) — the `FinnhubWebSocketClient.connect()` session-lifetime bug found in F-02-002 is unexercised by tests; root cause lives in `backend/services/realtime_price_service.py`.
- **F-15-005** → scope 09 — `StockRecommendation` dataclass missing `ranking_score` field is the production defect; the test design flaw (Mock instead of real dataclass) means the bug goes undetected.
- **F-15-006, F-15-007** → scope 08 (auth-security-compliance) and scope 01 (backend-api) — JWT RS256/HS256 mismatch and secrets_manager salt vulnerabilities need regression test coverage to prevent recurrence.
- **F-15-008, F-15-020** → scope 07 (database-persistence) — Docker-dependent tests hide real database integration coverage gaps identified in that scope.
- **F-15-010** → scope 07 — stale `backend.models.database.Base` import mirrors the schema management concerns in scope 07.
- **F-15-011** → scope 07 — `AsyncBaseRepository.transaction()` async generator bug (F-07-002) has no test coverage at all.

---

## 5. Risk-Prioritized Punch List (top 10)

1. **F-15-001** — collection-killing import in test_performance_optimizations.py — kills 50+ tests silently at collection time; 1-hour fix; highest density impact per effort unit.
2. **F-15-002** — wrong flags to vitest in run-tests.sh — frontend tests broken in CI; 30-minute fix; every CI run is currently broken for `--all` mode.
3. **F-15-003** — 1,356 lines of security+DB tests dead due to pytest.ini exclusion — critical security regressions undetected; 3-hour fix; directly impacts scope 07 and 08 deliverables.
4. **F-15-004** — FinnhubWebSocketClient.connect() bug untested — real-time pricing is the core product feature; the production bug (F-02-002) can only be fixed with confidence once a test catches it.
5. **F-15-011** — AsyncBaseRepository.transaction() bug (F-07-002) has zero test coverage — every call to `transaction()` silently does nothing in production; a test will both document and guard the fix.
6. **F-15-009** — three inconsistent coverage thresholds — CI passes at 75% when team target is 85%; enforcing 85% could surface new gaps immediately.
7. **F-15-005** — ranking_score tested via Mock masks real dataclass bug — all recommendation ranking tests are false positives; affects product correctness of the recommendation feature.
8. **F-15-008** — Docker-only DB integration tests silently skipped — entire PostgreSQL integration layer is dark in standard CI; paired with F-15-020 this is a systematic gap.
9. **F-15-021** — run_integration_tests.py ignores 12 integration/ test files — the dedicated integration harness is blind to half the integration suite.
10. **F-15-006** — JWT security tests use MagicMock instead of real JWTManager — security regression tests provide false assurance; real auth bugs (F-08-001, F-01-001) go unguarded.

---

## 6. Open Questions

- Q1: Is the `tests/` root directory intentionally excluded from pytest discovery, or is this an oversight? The 1,356 lines of security and database tests suggest it was intended as a parallel test tree but was never wired up. Resolution determines whether F-15-003 is a "move files" or "add to testpaths" fix.
- Q2: When `objgraph` is installed (e.g., in a dev environment with performance tooling), does F-15-001 currently cause visible test collection errors, or are these already silently swallowed? The answer determines urgency.
- Q3: What is the team's target for frontend unit test coverage? The 16 vitest files exist but there is no coverage threshold enforced in CI. Is F-15-025 a backlog item or a priority?
- Q4: Should the test suite enforce `asyncio_mode = strict` (requiring explicit async markers, easier to audit) rather than `asyncio_mode = auto`? Moving from auto→strict requires adding `@pytest.mark.asyncio` only to async tests (not the 1,622 redundant existing ones). This is a one-time migration with long-term correctness benefits.
- Q5: Is `run_integration_tests.py` used in CI, or is it a developer-only convenience script? If it's in CI, the 12 files it misses (F-15-021) represent a systematic gap; if not, lower priority.
