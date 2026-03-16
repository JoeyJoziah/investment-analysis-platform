# Phase 2B: ERROR Test Categorization

## Overview
After fixing 17 import errors in Waves 1-3, we still have **139 ERROR tests** failing during test execution (not import). These are runtime errors that need systematic resolution.

## ERROR Tests by File (Top 20)

| File | Count | Category |
|------|-------|----------|
| test_performance_optimizations.py | 24 | Performance |
| test_security_integration.py | 14 | Security |
| test_comprehensive_units.py | 12 | Unit Tests |
| test_integration_comprehensive.py | 11 | Integration |
| test_error_scenarios.py | 9 | Error Handling |
| test_database_integration.py | 9 | Database |
| test_data_pipeline_integration.py | 9 | Data Pipeline |
| test_security_compliance.py | 8 | Security |
| test_performance_load.py | 7 | Performance |
| test_integration.py | 7 | Integration |
| test_resilience_integration.py | 6 | Resilience |
| test_cointegration.py | 6 | Data Pipeline |
| integration/test_stock_to_analysis_flow.py | 5 | Integration |
| integration/test_agents_to_recommendations_flow.py | 5 | Integration |
| integration/test_auth_to_portfolio_flow.py | 3 | Integration |
| integration/test_phase3_integration.py | 2 | Integration |
| integration/test_gdpr_data_lifecycle.py | 2 | GDPR |

**Total**: 139 ERROR tests across 17 test files

## Category Breakdown

| Category | Count | Files |
|----------|-------|-------|
| **Performance** | 31 | test_performance_optimizations.py (24), test_performance_load.py (7) |
| **Security** | 22 | test_security_integration.py (14), test_security_compliance.py (8) |
| **Integration** | 33 | test_integration_comprehensive.py (11), test_integration.py (7), integration/* (15) |
| **Data Pipeline** | 15 | test_data_pipeline_integration.py (9), test_cointegration.py (6) |
| **Unit Tests** | 12 | test_comprehensive_units.py (12) |
| **Error Handling** | 9 | test_error_scenarios.py (9) |
| **Database** | 9 | test_database_integration.py (9) |
| **Resilience** | 6 | test_resilience_integration.py (6) |
| **GDPR** | 2 | integration/test_gdpr_data_lifecycle.py (2) |

## Root Cause Hypotheses

### 1. Missing Test Fixtures or Setup
Many integration tests likely fail because they require:
- Database setup
- Redis connections
- API mock responses
- Authentication tokens
- Test data seeding

### 2. Missing Dependencies or Configuration
Tests may require:
- External services (Celery, Redis)
- Environment variables
- Configuration files
- API keys for test services

### 3. Async/Event Loop Issues
Seen in test_admin_api.py:
```
ERROR - Rate limiting middleware error: Event loop is closed
```

### 4. CSRF Token Validation
Seen in test_admin_api.py:
```
ERROR - Unhandled exception: 403: CSRF token validation failed
```

### 5. Missing Test Data or Mocks
Tests expecting specific data or service responses that aren't mocked properly.

## Phase 2B Strategy

### Step 1: Sample 5 Representative ERROR Tests
Pick one test from each major category and extract full traceback with `--tb=long`:

1. **Performance**: `test_performance_optimizations.py::TestMemoryManager::test_memory_metrics_collection`
2. **Security**: `test_security_integration.py::TestSecurityIntegration::test_jwt_token_creation_and_validation`
3. **Integration**: `test_integration_comprehensive.py::TestEndToEndWorkflows::test_stock_analysis_workflow`
4. **Data Pipeline**: `test_data_pipeline_integration.py::TestDataPipelineIntegration::test_complete_data_ingestion_pipeline`
5. **Database**: `test_database_integration.py::TestDatabaseIntegration::test_user_repository_operations`

### Step 2: Identify Common Root Causes
Analyze the 5 tracebacks to find patterns:
- Same ImportError across multiple tests?
- Same configuration issue?
- Same fixture missing?
- Same async/await problem?

### Step 3: Create Systematic Fixes
Based on root causes, create fixes that resolve entire categories:
- Add missing fixtures
- Fix async test configuration
- Add required mock services
- Fix CSRF token handling in tests
- Add required environment variables

### Step 4: Rerun and Measure Impact
After each batch of fixes, rerun tests to measure improvement.

## Expected Timeline

| Phase | Tasks | Expected Improvement |
|-------|-------|---------------------|
| **2B-1** | Extract 5 sample ERROR tracebacks | 0% (discovery) |
| **2B-2** | Fix common root cause #1 | +30-50 tests (3-6%) |
| **2B-3** | Fix common root cause #2 | +30-40 tests (3-5%) |
| **2B-4** | Fix common root cause #3 | +20-30 tests (2-4%) |
| **2B-5** | Fix remaining ERRORs individually | +20-30 tests (2-4%) |

**Target after Phase 2B**: 550/846 passing (65%), 296 remaining to reach 80%

## Next Commands

```bash
# Extract 5 sample ERROR test tracebacks
pytest backend/tests/test_performance_optimizations.py::TestMemoryManager::test_memory_metrics_collection -v --tb=long > /tmp/error-sample-1.txt 2>&1

pytest backend/tests/test_security_integration.py::TestSecurityIntegration::test_jwt_token_creation_and_validation -v --tb=long > /tmp/error-sample-2.txt 2>&1

pytest backend/tests/test_integration_comprehensive.py::TestEndToEndWorkflows::test_stock_analysis_workflow -v --tb=long > /tmp/error-sample-3.txt 2>&1

pytest backend/tests/test_data_pipeline_integration.py::TestDataPipelineIntegration::test_complete_data_ingestion_pipeline -v --tb=long > /tmp/error-sample-4.txt 2>&1

pytest backend/tests/test_database_integration.py::TestDatabaseIntegration::test_user_repository_operations -v --tb=long > /tmp/error-sample-5.txt 2>&1
```

## Files for Next Session

- `/tmp/error-sample-{1-5}.txt` - Full tracebacks for analysis
- `docs/deployment/PHASE2_ERROR_CATEGORIZATION.md` - This file
- `docs/deployment/PHASE2_WAVE_RESULTS.md` - Wave 1-3 results

## Conclusion

Phase 2 Waves 1-3 improved the pass rate from 46.2% to 48.8% (+2.6%) by fixing import errors. However, the 139 ERROR tests require runtime error fixes, not import fixes.

Phase 2B will systematically analyze and fix these runtime errors to push the pass rate toward 80%.
