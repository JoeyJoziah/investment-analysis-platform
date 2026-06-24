# Pytest-Asyncio Configuration Analysis

## Executive Summary

The `backend/tests/conftest.py` file contains session-scoped async fixtures that are incompatible with pytest 9.x and pytest-asyncio 1.3.0+ behavior. This analysis provides specific recommendations for fixing the configuration issues.

---

## Problem Identification

### 1. Event Loop Fixture Issues

**Current Implementation (Lines 31-37):**
```python
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
```

**Problems:**
- **Scope mismatch**: Session-scoped `event_loop` fixture conflicts with pytest-asyncio's default function-scoped event loop
- **Decorator type**: Using `@pytest.fixture` instead of `@pytest_asyncio.fixture` causes confusion about ownership
- **Loop lifecycle**: Creating a single event loop for entire session can cause issues with test isolation
- **pytest-asyncio 1.3.0+**: This version has stricter event loop handling and deprecates session-scoped event loops

### 2. Session-Scoped Async Fixtures

**Current Implementation:**

Lines 102-118 - `test_db_engine`:
```python
@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    """Create test database engine."""
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )

    engine = create_async_engine(
        test_db_url,
        echo=False,
        pool_pre_ping=True
    )

    yield engine
    await engine.dispose()
```

Lines 121-129 - `test_db_session_factory`:
```python
@pytest_asyncio.fixture(scope="session")
async def test_db_session_factory(test_db_engine):
    """Create test database session factory."""
    TestSessionLocal = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    return TestSessionLocal
```

**Problems:**
- **pytest-asyncio limitation**: pytest-asyncio 1.3.0+ doesn't properly support async fixtures with session scope
- **Event loop dependency**: Session-scoped async fixtures need a session-scoped event loop, which is deprecated
- **Resource lifecycle**: Database engine created at session start can have stale connections across tests
- **Test isolation**: Shared engine across all tests can cause test pollution

### 3. Multiple Event Loop Definitions

**Duplicate Definitions Found:**
1. `backend/tests/conftest.py:32` - Session-scoped, `@pytest.fixture`
2. `backend/tests/async_fixtures.py:213` - Session-scoped, `@pytest_asyncio.fixture`
3. `backend/tests/fixtures/integration_test_fixtures.py:650` - Session-scoped, `@pytest.fixture`

**Problems:**
- **Fixture override conflicts**: Multiple event_loop fixtures cause pytest collection errors
- **Inconsistent behavior**: Different implementations create unpredictable test behavior
- **Scope conflicts**: All trying to use session scope when pytest-asyncio prefers function scope

---

## Pytest-Asyncio 1.3.0+ Best Practices

### Key Changes in pytest-asyncio 1.3.0+

1. **Event loop management**: pytest-asyncio now manages event loops automatically
2. **asyncio_mode configuration**: New configuration option in pytest.ini
3. **Scope restrictions**: Session-scoped async fixtures are discouraged
4. **Auto mode**: `asyncio_mode = auto` automatically detects async tests

### Recommended asyncio_mode Settings

**Option 1: Auto Mode (Recommended)**
```ini
[pytest]
asyncio_mode = auto
```
- Automatically detects async test functions
- No need for `@pytest.mark.asyncio` decorator
- Simplest configuration

**Option 2: Strict Mode**
```ini
[pytest]
asyncio_mode = strict
```
- Requires explicit `@pytest.mark.asyncio` decorator
- More control over which tests are async
- Better for mixed sync/async test suites

**Option 3: Legacy Mode**
```ini
[pytest]
asyncio_mode = legacy
```
- Backward compatible with older pytest-asyncio behavior
- Not recommended for new projects

---

## Recommended Solutions

### Solution 1: Remove Custom Event Loop (Recommended)

**Remove the custom event_loop fixture entirely and let pytest-asyncio manage it.**

**Implementation:**
1. Delete lines 31-37 from `conftest.py`
2. Delete duplicate event_loop fixtures from other files
3. Add pytest.ini configuration:

```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
```

**Pros:**
- Follows pytest-asyncio best practices
- Simplest solution
- Best test isolation
- Future-proof

**Cons:**
- Function-scoped fixtures recreated for each test (slight performance overhead)

### Solution 2: Function-Scoped Database Fixtures

**Convert session-scoped async fixtures to function-scoped.**

**Implementation:**

```python
@pytest_asyncio.fixture  # Remove scope="session"
async def test_db_engine():
    """Create test database engine."""
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )

    engine = create_async_engine(
        test_db_url,
        echo=False,
        pool_pre_ping=True,
        poolclass=NullPool  # Important: Disable pooling for test isolation
    )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session_factory(test_db_engine):
    """Create test database session factory."""
    TestSessionLocal = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    return TestSessionLocal


@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    """Provide database session for tests."""
    async with test_db_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
```

**Pros:**
- Better test isolation
- Compatible with pytest-asyncio 1.3.0+
- No event loop management needed

**Cons:**
- Engine recreated for each test (performance impact)
- May need database initialization optimization

### Solution 3: Hybrid Approach with Module Scope

**Use module scope as a middle ground between session and function.**

**Implementation:**

```python
@pytest_asyncio.fixture(scope="module")
async def test_db_engine():
    """Create test database engine per test module."""
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )

    engine = create_async_engine(
        test_db_url,
        echo=False,
        pool_pre_ping=True
    )

    yield engine
    await engine.dispose()
```

**Pros:**
- Balance between performance and isolation
- Compatible with pytest-asyncio 1.3.0+
- Reduces setup overhead

**Cons:**
- Tests within same module share engine
- Still requires proper event loop configuration

### Solution 4: Synchronous Setup with Async Runtime

**Make setup fixtures synchronous, only the usage async.**

**Implementation:**

```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


@pytest.fixture(scope="session")
def test_db_url():
    """Provide test database URL (sync fixture)."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )


@pytest.fixture(scope="session")
def async_engine_factory(test_db_url):
    """Factory for creating async engines (sync fixture)."""
    def _create_engine():
        return create_async_engine(
            test_db_url,
            echo=False,
            pool_pre_ping=True,
            poolclass=NullPool
        )
    return _create_engine


@pytest_asyncio.fixture
async def test_db_engine(async_engine_factory):
    """Create async engine per test (async fixture)."""
    engine = async_engine_factory()
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_db_engine):
    """Provide database session for tests."""
    TestSessionLocal = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
```

**Pros:**
- Clear separation between sync setup and async usage
- Session-scoped configuration without async issues
- Good test isolation
- pytest-asyncio 1.3.0+ compatible

**Cons:**
- More complex fixture design
- Requires factory pattern understanding

---

## Configuration Requirements

### Required pytest.ini Configuration

**Create `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/pytest.ini`:**

```ini
[pytest]
# pytest-asyncio configuration
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function

# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output options
addopts =
    -v
    --strict-markers
    --tb=short
    --disable-warnings
    -p no:warnings

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    api: marks tests as API tests
    database: marks tests as database tests
    security: marks tests as security tests
    performance: marks tests as performance tests
    external_api: marks tests that call external APIs

# Timeout settings
timeout = 300
timeout_method = thread

# Coverage settings (if using pytest-cov)
# [tool:pytest]
# addopts = --cov=backend --cov-report=html --cov-report=term
```

### Required Dependencies

Ensure these versions in requirements:
```
pytest>=9.0.0
pytest-asyncio>=1.3.0
pytest-timeout>=2.2.0
sqlalchemy>=2.0.0
aiosqlite>=0.19.0
httpx>=0.25.0
```

---

## Fixture Dependency Ordering

### Proper Ordering

1. **Configuration fixtures** (session scope, sync)
   - `test_db_url`
   - `setup_test_environment`
   - `performance_threshold`

2. **Factory fixtures** (session scope, sync)
   - `async_engine_factory`

3. **Resource fixtures** (function/module scope, async)
   - `test_db_engine`
   - `test_db_session_factory`

4. **Usage fixtures** (function scope, async)
   - `db_session`
   - `async_client`

5. **Mock fixtures** (function scope, sync/async)
   - `mock_current_user`
   - `mock_cache`
   - `mock_external_apis`

---

## Recommended Implementation Plan

### Phase 1: Configuration Cleanup
1. Create `backend/pytest.ini` with proper asyncio_mode
2. Remove custom `event_loop` fixtures from all files
3. Standardize on pytest-asyncio automatic event loop management

### Phase 2: Fixture Refactoring
1. Convert session-scoped async fixtures to function-scoped
2. Or implement synchronous factory pattern
3. Add `NullPool` to engine configuration for test isolation
4. Test each fixture individually

### Phase 3: Test Validation
1. Run full test suite: `pytest backend/tests/ -v`
2. Check for deprecation warnings: `pytest backend/tests/ -W error::DeprecationWarning`
3. Verify test isolation: `pytest backend/tests/ --lf --ff`
4. Performance testing: `pytest backend/tests/ -m "not slow" --durations=10`

### Phase 4: Documentation
1. Document fixture usage patterns
2. Add fixture dependency diagrams
3. Create testing guidelines for new tests

---

## Edge Cases and Gotchas

### 1. SQLite In-Memory Database Gotcha
**Problem**: SQLite `:memory:` databases are lost when engine is disposed.

**Solution**: Use file-based SQLite for session-scoped tests:
```python
test_db_url = "sqlite+aiosqlite:///./test.db"
# Clean up in session teardown
```

### 2. Connection Pool Issues
**Problem**: Connection pooling can cause tests to interfere with each other.

**Solution**: Use `NullPool` for tests:
```python
from sqlalchemy.pool import NullPool

engine = create_async_engine(
    test_db_url,
    poolclass=NullPool  # Disable connection pooling
)
```

### 3. Fixture Cleanup Order
**Problem**: Async fixtures may not cleanup in reverse dependency order.

**Solution**: Use explicit cleanup with try/finally:
```python
@pytest_asyncio.fixture
async def db_session(test_db_engine):
    session_factory = sessionmaker(test_db_engine, class_=AsyncSession)
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()
```

### 4. Event Loop Already Running
**Problem**: Tests fail with "Event loop is already running" error.

**Solution**: Ensure no nested `asyncio.run()` calls in test code:
```python
# BAD
async def test_something():
    result = asyncio.run(my_async_function())  # Don't do this!

# GOOD
async def test_something():
    result = await my_async_function()  # Use await instead
```

### 5. Fixture Scope Inheritance
**Problem**: Async fixture with function scope depends on async fixture with session scope.

**Solution**: Match scopes or use factory pattern:
```python
# Option 1: Match scopes
@pytest_asyncio.fixture(scope="session")
async def parent_fixture(): ...

@pytest_asyncio.fixture(scope="session")
async def child_fixture(parent_fixture): ...

# Option 2: Factory pattern (recommended)
@pytest.fixture(scope="session")
def parent_factory(): ...

@pytest_asyncio.fixture
async def child_fixture(parent_factory): ...
```

---

## Testing the Fix

### Validation Commands

```bash
# 1. Check for deprecation warnings
pytest backend/tests/conftest.py -v -W error::DeprecationWarning

# 2. Run single test to verify fixture loading
pytest backend/tests/test_integration.py::test_health_check -v -s

# 3. Run all tests with verbose output
pytest backend/tests/ -v --tb=short

# 4. Check fixture setup/teardown
pytest backend/tests/ -v --setup-show

# 5. Verify no event loop warnings
pytest backend/tests/ -v 2>&1 | grep -i "event loop"

# 6. Test with multiple workers (check isolation)
pytest backend/tests/ -n 4 -v
```

### Success Criteria

- [ ] No deprecation warnings about event loops
- [ ] No "session-scoped async fixture" errors
- [ ] All tests pass consistently
- [ ] Test isolation verified (can run tests in any order)
- [ ] No "Event loop is closed" errors
- [ ] Parallel test execution works (pytest-xdist)

---

## Memory Storage for Agent Coordination

**Key Findings to Store:**

1. **Root Cause**: Session-scoped async fixtures incompatible with pytest-asyncio 1.3.0+
2. **Primary Issue**: Custom event_loop fixture conflicts with pytest-asyncio's automatic management
3. **Best Solution**: Remove custom event_loop, use function-scoped async fixtures with pytest.ini configuration
4. **Alternative**: Factory pattern with sync session fixtures and async function fixtures
5. **Configuration**: Add `asyncio_mode = auto` to pytest.ini
6. **Critical**: Use NullPool for database engines in tests
7. **Testing**: Verify with `-W error::DeprecationWarning` flag

---

## References

- pytest-asyncio documentation: https://pytest-asyncio.readthedocs.io/
- pytest-asyncio 1.3.0 changelog: https://github.com/pytest-dev/pytest-asyncio/releases/tag/v1.3.0
- SQLAlchemy async testing: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- pytest fixtures: https://docs.pytest.org/en/stable/fixture.html

---

**Analysis Date**: 2026-01-27
**Analyzed By**: TDD Research Agent
**Status**: Complete - Ready for Implementation
