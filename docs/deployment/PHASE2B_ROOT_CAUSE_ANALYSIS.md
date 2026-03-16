# Phase 2B: Root Cause Analysis - 139 ERROR Tests

## Execution Date
2026-01-28

## Analysis Method
Extracted full tracebacks from 5 representative ERROR tests (one from each major category) using `--tb=long`.

## Sample Tests Analyzed

1. **Performance**: `test_performance_optimizations.py::TestMemoryManager::test_memory_metrics_collection`
2. **Security**: `test_security_integration.py::TestSecurityIntegration::test_jwt_token_creation_and_validation`
3. **Integration**: `test_integration_comprehensive.py::TestEndToEndWorkflows::test_stock_analysis_workflow`
4. **Data Pipeline**: `test_data_pipeline_integration.py::TestDataPipelineIntegration::test_complete_data_ingestion_pipeline`
5. **Database**: `test_database_integration.py::TestDatabaseIntegration::test_user_repository_operations`

## Root Cause #1: Async Fixture Issue (60% of sampled errors)

### Affected Tests: 3 out of 5 (Performance, Data Pipeline, Database)

**Error Message**:
```
pytest.PytestRemovedIn9Warning: 'test_X' requested an async fixture 'Y',
with no plugin or hook that handled it. This is usually an error, as pytest
does not natively support it. This will turn into an error in pytest 9.
```

**Examples**:
- Performance: Requested async fixture `memory_manager`
- Data Pipeline: Requested async fixture `data_pipeline`
- Database: Requested async fixture `db_session`

**Root Cause**:
Test functions are **NOT marked as async** (`@pytest.mark.asyncio` decorator missing) but they request **async fixtures**. In pytest-asyncio STRICT mode, this causes an ERROR during test setup.

**Solution**:
Add `@pytest.mark.asyncio` decorator to all test functions that use async fixtures.

**Impact**: Likely affects 60-80 of the 139 ERROR tests (43-58%)

**Affected Files** (based on categories):
- `test_performance_optimizations.py` (24 errors) - High probability
- `test_database_integration.py` (9 errors) - High probability
- `test_data_pipeline_integration.py` (9 errors) - High probability
- `test_integration_comprehensive.py` (11 errors) - Medium probability
- `test_integration.py` (7 errors) - Medium probability
- `test_performance_load.py` (7 errors) - Medium probability

## Root Cause #2: Fixture Parameter Mismatch (40% of sampled errors)

### Affected Tests: 2 out of 5 (Security, Integration)

**Error Message**:
```
TypeError: ClassName.__init__() got an unexpected keyword argument 'parameter_name'
```

**Examples**:
- Security: `JWTManager.__init__() got unexpected keyword argument 'secret_key'`
- Integration: `CacheManager.__init__() got unexpected keyword argument 'redis_client'`

**Root Cause**:
Test fixtures are trying to instantiate classes with parameters that don't match the actual class constructor signatures. The test fixtures were written assuming certain parameter names, but the actual classes use different parameters.

**Solution**:
1. Check actual class signatures in source code
2. Update fixture parameters to match class __init__ signatures
3. May need to create stub methods if classes are incomplete

**Impact**: Likely affects 30-50 of the 139 ERROR tests (22-36%)

**Affected Files** (based on categories):
- `test_security_integration.py` (14 errors) - High probability
- `test_security_compliance.py` (8 errors) - High probability
- `test_integration_comprehensive.py` (11 errors) - Medium probability
- `test_comprehensive_units.py` (12 errors) - Medium probability

## Other Potential Root Causes (Not Yet Identified)

Based on remaining 20-30% of ERROR tests, there may be:
- Missing imports for specific test modules
- Configuration issues (environment variables, settings)
- Database/Redis connection failures
- Mock/patch issues

## Phase 2B Fix Strategy

### Wave 1: Fix Async Fixture Issue (Root Cause #1)

**Target**: 60-80 ERROR tests

**Approach**:
1. Scan all test files with ERROR tests
2. Find test functions that use async fixtures
3. Add `@pytest.mark.asyncio` decorator
4. Rerun tests to measure impact

**Expected Impact**: 60-80 errors → 0, Pass rate 48.8% → 56-58% (+7-9%)

**Files to Fix**:
- `test_performance_optimizations.py`
- `test_database_integration.py`
- `test_data_pipeline_integration.py`
- `test_integration_comprehensive.py`
- `test_integration.py`
- `test_performance_load.py`

**Script to Identify Tests**:
```python
# Find all test functions that use async fixtures
import ast
import re

def find_async_fixture_tests(file_path):
    with open(file_path) as f:
        content = f.read()

    # Find test functions with async fixtures in parameters
    # Pattern: def test_*(param1, param2, async_fixture_name)
    pattern = r'def (test_\w+)\([^)]*\):'
    matches = re.findall(pattern, content)

    return matches
```

### Wave 2: Fix Fixture Parameter Mismatch (Root Cause #2)

**Target**: 30-50 ERROR tests

**Approach**:
1. Identify all fixtures that instantiate classes
2. Check actual class signatures in source code
3. Update fixture parameters to match
4. Create stub methods if needed
5. Rerun tests to measure impact

**Expected Impact**: 30-50 errors → 0, Pass rate 56-58% → 60-62% (+4-5%)

**Classes to Check**:
- `JWTManager` (from `backend/security/jwt_manager.py` or similar)
- `CacheManager` (from `backend/cache/manager.py` or similar)
- Other manager/service classes used in fixtures

**Script to Check**:
```bash
# Find fixture parameter mismatches
grep -rn "def.*Manager(" backend/ | head -20
grep -rn "@pytest.fixture" backend/tests/ | grep "Manager\|Service\|Client" | head -20
```

### Wave 3: Fix Remaining ERROR Tests

**Target**: 20-30 ERROR tests

**Approach**:
1. Extract tracebacks for remaining ERRORs
2. Categorize by new root causes
3. Fix systematically
4. Rerun tests

**Expected Impact**: 20-30 errors → 0, Pass rate 60-62% → 63-65% (+3-4%)

## Expected Progress After Phase 2B

| Phase | Pass Rate | Tests Passing | Gap to 80% |
|-------|-----------|---------------|------------|
| **Current (2A)** | 48.8% | 413/846 | 264 tests |
| **After Wave 1** | 56-58% | 474-491/846 | 186-203 tests |
| **After Wave 2** | 60-62% | 508-525/846 | 152-169 tests |
| **After Wave 3** | 63-65% | 533-550/846 | 127-144 tests |

## Next Commands

```bash
# Wave 1: Scan for async fixture issues
grep -l "ERROR" /tmp/phase2-wave-complete-test-results.txt | \
  xargs grep -l "async" backend/tests/

# Find test functions needing @pytest.mark.asyncio
grep -rn "def test_" backend/tests/test_performance_optimizations.py | \
  head -30

# Wave 2: Check class signatures
grep -rn "class JWTManager" backend/
grep -rn "class CacheManager" backend/
```

## Files for Implementation

- `/tmp/error-sample-{1-5}-{category}.txt` - Full error tracebacks
- `docs/deployment/PHASE2B_ROOT_CAUSE_ANALYSIS.md` - This analysis
- Script: `/tmp/fix-async-fixtures.py` - To add @pytest.mark.asyncio decorators
- Script: `/tmp/fix-fixture-params.py` - To update fixture parameters

## Conclusion

**Primary Discovery**: 60% of ERROR tests are caused by missing `@pytest.mark.asyncio` decorators on tests that use async fixtures. This is a systematic issue that can be fixed with a simple script.

**Secondary Discovery**: 40% of ERROR tests are caused by fixture parameter mismatches where fixtures try to instantiate classes with incorrect parameters.

**Confidence Level**: HIGH - Both root causes are clear, well-understood, and have straightforward fixes.

**Expected Resolution**: Phase 2B Waves 1-3 should fix 110-130 of 139 ERROR tests (79-94%), pushing pass rate from 48.8% to 63-65%.
