# Phase 2B Root Cause Analysis - Complete Summary

## Overview
Comprehensive analysis of ERROR test root causes across all waves of Phase 2B remediation.

---

## Root Cause Summary Table

| # | Root Cause | Pattern | Tests Affected | Status | Wave |
|---|------------|---------|----------------|--------|------|
| **1** | Async Fixture Decorators | `@pytest.fixture` → `@pytest_asyncio.fixture` | 60-80 | ✅ Mostly Fixed | Wave 1 |
| **2A** | JWT Fixture Parameters | `JWTManager(secret_key=...)` → `JWTManager(redis_client=...)` | 1+ | ✅ Fixed | Wave 2 |
| **2B** | JWT Method Names | `create_token` → `create_access_token`, `decode_token` → `verify_token` | 5+ | ✅ Fixed | Wave 2 |
| **2C** | CacheManager Fixture | `CacheManager(redis_client=...)` → `CacheManager(prefix=...)` | 1+ | 🔄 In Progress | Wave 2 |
| **3** | AlphaVantage Client Params | `AlphaVantageClient(api_key=...)` → Check actual signature | 12+ | ⏳ Pending | Wave 3B |
| **4** | User Method Names | `user.set_password()` doesn't exist | 9+ | ⏳ Pending | Wave 3C |
| **5** | Database Driver Config | psycopg2 loaded instead of asyncpg | 9-15 | ⏳ Pending | Wave 3D |
| **6** | Async Client Fixture | Remaining `async_client` with wrong decorator | 5-10 | ⏳ Pending | Wave 3A |
| **7** | Integration Test Issues | Multiple causes in integration/ tests | 17 | ⏳ Pending | Wave 3E |

---

## Detailed Root Cause Breakdown

### Root Cause #1: Async Fixture Decorators (Wave 1)
**Impact:** 60-80 tests → **29 confirmed fixed** in Wave 1

**Pattern:**
```python
# ❌ Wrong (causes pytest.PytestRemovedIn9Warning)
@pytest.fixture
async def memory_manager():
    return MemoryManager()

# ✅ Correct
@pytest_asyncio.fixture
async def memory_manager():
    return MemoryManager()
```

**Files Fixed (Wave 1):**
- test_performance_optimizations.py (6 fixtures)
- test_database_integration.py (1 fixture)
- test_data_pipeline_integration.py (2 fixtures)
- test_integration.py (2 fixtures)

**Remaining:**
- test_security_integration.py - `async_client` fixture (Wave 3A)

**Why This Happens:**
pytest-asyncio STRICT mode (v0.21+) requires explicit `@pytest_asyncio.fixture` decorator for async fixtures

---

### Root Cause #2: Fixture Parameter Mismatches (Wave 2 & 3B)

#### 2A. JWTManager (Wave 2) ✅
**Error:** `TypeError: JWTManager.__init__() got an unexpected keyword argument 'secret_key'`

**Actual Signature:**
```python
def __init__(self, redis_client: Optional[redis.Redis] = None):
```

**Fix:**
```python
# ❌ Old
jwt_manager = JWTManager(
    secret_key=settings.SECRET_KEY,
    algorithm="HS256",
    access_token_expire_minutes=30
)

# ✅ Fixed
jwt_manager = JWTManager(redis_client=mock_redis)
```

#### 2B. JWTManager Method Names (Wave 2) ✅
**Errors:**
- `AttributeError: 'JWTManager' object has no attribute 'create_token'`
- `AttributeError: 'JWTManager' object has no attribute 'decode_token'`

**Actual Methods:**
- `create_access_token(claims: TokenClaims, ...)`
- `verify_token(token: str, ...) -> Optional[Dict]`

**Fix:**
```python
# ❌ Old
token = jwt_manager.create_token({"sub": user.id})
decoded = jwt_manager.decode_token(token)

# ✅ Fixed
claims = TokenClaims(user_id=user.id, username=user.username, ...)
token = jwt_manager.create_access_token(claims)
decoded = jwt_manager.verify_token(token)
```

**Additional Fixes:**
- Added `TokenClaims` import
- Fixed mock Redis with smart `exists()` for blacklist vs session keys
- Fixed test assertions: `sub` contains username, `user_id` field contains ID
- Fixed exception handling: `verify_token` returns `None`, doesn't raise

#### 2C. CacheManager (Wave 2) 🔄
**Error:** `TypeError: CacheManager.__init__() got an unexpected keyword argument 'redis_client'`

**Actual Signature:**
```python
def __init__(self, prefix: str = ""):
```

**Fix:**
```python
# ❌ Old
cache_client = CacheManager(redis_client=test_redis.client)

# ✅ Fixed
cache_client = CacheManager(prefix="test")
```

**Status:** Fixed but test still has issues (needs investigation)

---

### Root Cause #3: AlphaVantageClient Fixture (Wave 3B) ⏳
**Impact:** 12+ tests in test_comprehensive_units.py

**Error:** `TypeError: AlphaVantageClient.__init__() got an unexpected keyword argument 'api_key'`

**Location:** tests/test_comprehensive_units.py:351

**Investigation Needed:**
1. Check `AlphaVantageClient.__init__` actual signature
2. Identify correct parameter names
3. Update fixture in test file
4. Check for similar issues in FinnhubClient, PolygonClient

---

### Root Cause #4: User.set_password() Method (Wave 3C) ⏳
**Impact:** 9+ tests in test_error_scenarios.py

**Error:** `AttributeError: 'User' object has no attribute 'set_password'`

**Location:** tests/test_error_scenarios.py - `authenticated_client` fixture

**Investigation Needed:**
1. Check `User` model actual methods
2. Find correct password-setting method
3. Update all test fixtures using `set_password()`

**Hint:** Error suggests using `hashed_password` field directly

---

### Root Cause #5: Database Driver Configuration (Wave 3D) ⏳
**Impact:** 9-15 tests in test_database_integration.py

**Error:**
```
sqlalchemy.exc.InvalidRequestError: The asyncio extension requires an async driver
to be used. The loaded 'psycopg2' is not async.
```

**Location:** tests/test_database_integration.py:49 - `db_session` fixture

**Issue:** Connection string specifies `postgresql+asyncpg://` but psycopg2 is loaded

**Solution Options:**
1. ✅ **Preferred:** Use testcontainers with PostgreSQL (like other tests)
2. Ensure asyncpg package is installed
3. Use SQLite in-memory for unit tests
4. Fix database URL construction

---

### Root Cause #6: Remaining Async Fixtures (Wave 3A) ⏳
**Impact:** 5-10 tests

**Same as Root Cause #1, but additional instances found:**
- tests/test_security_integration.py:76 - `async_client` fixture

**Quick Fix:** Add `@pytest_asyncio.fixture` decorator

---

### Root Cause #7: Integration Test Issues (Wave 3E) ⏳
**Impact:** 17 tests across 5 files

**Files:**
- test_stock_to_analysis_flow.py (5 errors)
- test_agents_to_recommendations_flow.py (5 errors)
- test_auth_to_portfolio_flow.py (3 errors)
- test_phase3_integration.py (2 errors)
- test_gdpr_data_lifecycle.py (2 errors)

**Status:** Requires investigation - likely combination of above root causes

---

## Wave-by-Wave Progress

### Wave 1: Async Fixture Decorators
- **Targeted:** 60-80 tests
- **Fixed:** 29 tests confirmed (ERROR → lower count)
- **Pass Rate:** 48.8% → 51.1%
- **Commit:** 5df51f4

### Wave 2: JWT & CacheManager Fixture Parameters
- **Targeted:** 30-50 tests
- **Fixed:** 1+ tests confirmed (`test_jwt_token_creation_and_validation`)
- **Pass Rate:** 51.1% (unchanged - need fresh test run)
- **Commit:** f494023

### Wave 3: Remaining 110 ERRORs (Planned)
- **Phase 3A:** Async fixtures (5-10 tests)
- **Phase 3B:** Fixture parameters (30-40 tests)
- **Phase 3C:** Method names (15-20 tests)
- **Phase 3D:** Database driver (9-15 tests)
- **Phase 3E:** Integration tests (17 tests)
- **Target:** 95-110 tests fixed → 63-65% pass rate

---

## Key Patterns & Lessons

### Pattern Recognition
1. **Fixture decorator mismatch** → Async fixtures need `@pytest_asyncio.fixture`
2. **API signature changes** → Always check actual `__init__` signatures
3. **Method name assumptions** → Verify method names before fixing tests
4. **Driver configuration** → Async operations need async drivers
5. **Test isolation** → Mock dependencies properly (smart mocks for context-aware behavior)

### Best Practices
1. ✅ **Sample First:** Test one instance before batch fixing
2. ✅ **Read Implementation:** Check actual class signatures
3. ✅ **Verify Fix:** Run test to confirm ERROR → PASSED
4. ✅ **Commit Incrementally:** Separate commits for each root cause
5. ✅ **Document Findings:** Record root causes for future reference

---

## Statistics

### Current State (After Wave 1 & 2)
- **Total Tests:** 846
- **Passed:** 432 (51.1%)
- **Failed:** 304
- **Errors:** 110
- **Target:** 533+ passed (63%)

### Tests Fixed by Wave
- **Wave 1:** 29 tests (ERROR → PASSED/FAILED)
- **Wave 2:** 1+ tests (conservative, likely more)
- **Wave 3:** 95-110 tests (projected)
- **Total:** 125-140 tests (14.8-16.5% improvement)

### Pass Rate Trajectory
- **Start:** 48.8% (413/846)
- **Wave 1:** 51.1% (432/846)
- **Wave 2:** ~51-52% (estimated, need fresh run)
- **Wave 3:** 63-65% (target)

---

**Generated:** 2026-01-28 15:52 PST
**Status:** Complete Root Cause Analysis
**Next:** Execute Wave 3 Phases
