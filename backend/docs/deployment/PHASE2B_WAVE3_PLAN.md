# Phase 2B Wave 3 Plan - Remaining 110 ERROR Tests

## Overview
Analysis of remaining 110 ERROR tests after Wave 1 (async fixtures) and Wave 2 (JWT/CacheManager parameter fixes).

## Error Distribution by Test File

| Test File | ERROR Count |
|-----------|-------------|
| test_security_integration.py | 14 |
| test_comprehensive_units.py | 12 |
| test_integration_comprehensive.py | 11 |
| test_error_scenarios.py | 9 |
| test_database_integration.py | 9 |
| test_security_compliance.py | 8 |
| test_performance_load.py | 7 |
| test_integration.py | 7 |
| test_resilience_integration.py | 6 |
| test_cointegration.py | 6 |
| integration/test_stock_to_analysis_flow.py | 5 |
| integration/test_agents_to_recommendations_flow.py | 5 |
| integration/test_auth_to_portfolio_flow.py | 3 |
| integration/test_phase3_integration.py | 2 |
| integration/test_gdpr_data_lifecycle.py | 2 |
| **TOTAL** | **110** |

## Root Cause Categories

### 1. Remaining Async Fixture Decorators (Root Cause #1 Continuation)
**Pattern:** `@pytest.fixture` used for async functions instead of `@pytest_asyncio.fixture`

**Example:**
```python
# ❌ Wrong
@pytest.fixture
async def async_client(self):
    async with AsyncClient(...) as client:
        yield client

# ✅ Correct
@pytest_asyncio.fixture
async def async_client(self):
    async with AsyncClient(...) as client:
        yield client
```

**Identified Instance:**
- `tests/test_security_integration.py:76` - `async_client` fixture

**Expected Impact:** 5-10 tests

---

### 2. Fixture Parameter Mismatches (Root Cause #2 Continuation)
**Pattern:** Test fixtures instantiate classes with incorrect/outdated parameter names

**Examples:**

#### AlphaVantageClient
```python
# ❌ Wrong
@pytest.fixture
def alpha_vantage_client(self):
    return AlphaVantageClient(api_key="test_key")

# ✅ Need to check actual __init__ signature
```
**Location:** `tests/test_comprehensive_units.py:351`
**Affected Tests:** 12 tests in test_comprehensive_units.py

#### CacheManager (from Wave 2)
```python
# ✅ Fixed
@pytest.fixture
def cache_client(test_redis):
    return CacheManager(prefix="test")
```

**Expected Impact:** 30-40 tests

---

### 3. Method Name Mismatches (New Pattern)
**Pattern:** Test code calls methods that don't exist on objects

**Examples:**

#### User.set_password()
```python
# ❌ Wrong
user.set_password(password)

# ✅ Need to find actual method name (likely set_hashed_password or similar)
```
**Location:** `tests/test_error_scenarios.py` - `authenticated_client` fixture
**Error:** `AttributeError: 'User' object has no attribute 'set_password'`
**Affected Tests:** 9 tests in test_error_scenarios.py

#### JWTManager (from Wave 2 - FIXED)
- `create_token` → `create_access_token` ✅
- `decode_token` → `verify_token` ✅

**Expected Impact:** 15-20 tests

---

### 4. Database Driver Configuration (New Pattern)
**Pattern:** Tests expect async database driver (asyncpg) but sync driver (psycopg2) is loaded

**Error:**
```python
sqlalchemy.exc.InvalidRequestError: The asyncio extension requires an async driver
to be used. The loaded 'psycopg2' is not async.
```

**Location:** `tests/test_database_integration.py:49` - `db_session` fixture

**Cause:** Connection string specifies `postgresql+asyncpg://` but psycopg2 driver is being loaded instead

**Solution Options:**
1. Ensure `asyncpg` package is installed
2. Fix database URL construction to use correct driver
3. Update fixture to use testcontainers or in-memory SQLite

**Affected Tests:** 9 tests in test_database_integration.py

**Expected Impact:** 9-15 tests

---

### 5. Integration Test File Issues (Unknown Pattern)
**Pattern:** Integration tests in `tests/integration/` directory showing ERRORs

**Files:**
- test_stock_to_analysis_flow.py (5 errors)
- test_agents_to_recommendations_flow.py (5 errors)
- test_auth_to_portfolio_flow.py (3 errors)
- test_phase3_integration.py (2 errors)
- test_gdpr_data_lifecycle.py (2 errors)

**Need Investigation:** These may be a mix of the above root causes or new issues

**Expected Impact:** 17 tests

---

## Wave 3 Strategy

### Phase 3A: Quick Wins (5-10 tests)
1. Fix remaining async fixture decorators
   - `tests/test_security_integration.py` - `async_client` fixture
   - Search for other async fixtures with wrong decorator

### Phase 3B: Fixture Parameters (30-40 tests)
2. Fix AlphaVantageClient fixture parameters
   - Check actual `__init__` signature
   - Update `tests/test_comprehensive_units.py:351`
3. Search for similar API client fixture issues
   - FinnhubClient
   - PolygonClient
   - Other data source clients

### Phase 3C: Method Names (15-20 tests)
4. Fix User.set_password() calls
   - Find actual method name in User model
   - Update `tests/test_error_scenarios.py` fixture
5. Search for other method name mismatches

### Phase 3D: Database Configuration (9-15 tests)
6. Fix database driver configuration
   - Option 1: Ensure asyncpg is installed
   - Option 2: Use testcontainers (recommended)
   - Option 3: Use SQLite in-memory for unit tests
7. Update `tests/test_database_integration.py` fixture

### Phase 3E: Integration Tests (17 tests)
8. Investigate integration test errors
   - May be combination of above root causes
   - Create sub-plan after initial investigation

---

## Expected Outcomes

**Conservative Estimate:**
- Phase 3A: +8 tests (5-10)
- Phase 3B: +35 tests (30-40)
- Phase 3C: +17 tests (15-20)
- Phase 3D: +12 tests (9-15)
- Phase 3E: +15 tests (unknown, conservative)
- **Total:** +87 tests fixed

**Optimistic Estimate:**
- Phase 3A: +10 tests
- Phase 3B: +40 tests
- Phase 3C: +20 tests
- Phase 3D: +15 tests
- Phase 3E: +25 tests (if same root causes)
- **Total:** +110 tests fixed (100%)

**Pass Rate Projections:**
- Current: 51.1% (432/846)
- Conservative: 61.4% (519/846)
- Optimistic: 64.1% (542/846)

**Target:** 63-65% pass rate requires fixing ~95-110 tests

---

## Implementation Order

1. **Start:** Phase 3A (async fixtures) - Fastest, highest confidence
2. **Then:** Phase 3B (fixture parameters) - Similar to Wave 2, proven approach
3. **Then:** Phase 3C (method names) - Check actual implementations first
4. **Then:** Phase 3D (database driver) - May need infrastructure changes
5. **Finally:** Phase 3E (integration tests) - Investigate and adapt

---

## Risk Mitigation

1. **Test Each Category Individually** - Don't mix fixes, easier to verify impact
2. **Commit After Each Phase** - Separate git commits for easy rollback
3. **Sample Test One Fix** - Always test a single instance before batch fixing
4. **Check Actual Implementations** - Never assume method/parameter names

---

## Next Steps

1. ✅ Complete Wave 3 root cause analysis
2. 🔄 Execute Phase 3A (async fixtures)
3. 🔄 Execute Phase 3B (fixture parameters)
4. 🔄 Execute Phase 3C (method names)
5. 🔄 Execute Phase 3D (database driver)
6. 🔄 Execute Phase 3E (integration tests)
7. 📊 Run full test suite to measure actual impact
8. 📝 Document Wave 3 results

---

**Generated:** 2026-01-28 15:50 PST
**Status:** Wave 3 Planning Complete
**Next:** Execute Phase 3A - Async Fixtures
