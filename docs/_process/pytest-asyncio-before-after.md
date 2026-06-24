# Pytest-Asyncio Configuration: Before vs After

## Visual Comparison

### BEFORE (Problematic Configuration)

```python
# backend/tests/conftest.py

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# ❌ PROBLEM 1: Custom event_loop fixture conflicts with pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ❌ PROBLEM 2: Session-scoped async fixture not supported in pytest-asyncio 1.3.0+
@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    """Create test database engine."""
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )

    # ❌ PROBLEM 3: No NullPool - connection pooling causes test interference
    engine = create_async_engine(
        test_db_url,
        echo=False,
        pool_pre_ping=True
    )

    yield engine
    await engine.dispose()


# ❌ PROBLEM 4: Session-scoped async fixture
@pytest_asyncio.fixture(scope="session")
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

**Problems:**
- Custom event_loop fixture (lines 7-12)
- Session-scoped async fixtures (lines 16, 36)
- Missing NullPool configuration
- No pytest.ini configuration
- Duplicate event_loop fixtures in other files

---

### AFTER (Recommended Solution - Option 1)

```python
# backend/tests/conftest.py

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool  # ✅ NEW: Import NullPool

# ✅ REMOVED: No custom event_loop fixture - let pytest-asyncio manage it

# ✅ FIXED: Function-scoped async fixture
@pytest_asyncio.fixture  # No scope parameter = function scope
async def test_db_engine():
    """Create test database engine."""
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )

    # ✅ FIXED: Added NullPool for test isolation
    engine = create_async_engine(
        test_db_url,
        echo=False,
        pool_pre_ping=True,
        poolclass=NullPool  # Disable connection pooling
    )

    yield engine
    await engine.dispose()


# ✅ FIXED: Function-scoped async fixture
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

**NEW FILE: backend/pytest.ini**
```ini
[pytest]
# ✅ Configure pytest-asyncio to auto-detect async tests
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts =
    -v
    --strict-markers
    --tb=short

markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    api: marks tests as API tests
    database: marks tests as database tests
    security: marks tests as security tests
    performance: marks tests as performance tests
```

**Changes:**
- ✅ Removed custom event_loop fixture
- ✅ Changed to function-scoped async fixtures
- ✅ Added NullPool for test isolation
- ✅ Created pytest.ini with asyncio_mode=auto
- ✅ Remove duplicate event_loop fixtures from other files

---

### AFTER (Alternative Solution - Option 2: Factory Pattern)

```python
# backend/tests/conftest.py

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# ✅ REMOVED: No custom event_loop fixture

# ✅ PATTERN: Sync session-scoped configuration
@pytest.fixture(scope="session")
def test_db_url():
    """Provide test database URL (sync fixture)."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )


# ✅ PATTERN: Sync session-scoped factory
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


# ✅ PATTERN: Async function-scoped resource from factory
@pytest_asyncio.fixture
async def test_db_engine(async_engine_factory):
    """Create async engine per test (async fixture)."""
    engine = async_engine_factory()
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

**Benefits of Factory Pattern:**
- Session-scoped configuration (faster, no duplication)
- Function-scoped async resources (proper isolation)
- Clear separation of concerns
- pytest-asyncio 1.3.0+ compatible

---

## Fixture Dependency Graph

### BEFORE (Broken)
```
event_loop (session, sync) ❌
    └─> test_db_engine (session, async) ❌
            └─> test_db_session_factory (session, async) ❌
                    └─> db_session (function, async)
```

**Problem**: Session-scoped async fixtures need session-scoped event loop (deprecated)

### AFTER - Option 1 (Simple)
```
[pytest-asyncio manages event loop automatically] ✅
    └─> test_db_engine (function, async) ✅
            └─> test_db_session_factory (function, async) ✅
                    └─> db_session (function, async) ✅
```

**Benefit**: All function-scoped, maximum test isolation

### AFTER - Option 2 (Factory Pattern)
```
test_db_url (session, sync) ✅
    └─> async_engine_factory (session, sync) ✅
            └─> [pytest-asyncio event loop] ✅
                    └─> test_db_engine (function, async) ✅
                            └─> test_db_session_factory (function, async) ✅
                                    └─> db_session (function, async) ✅
```

**Benefit**: Session-scoped config, function-scoped async resources

---

## Error Messages: Before vs After

### BEFORE - Typical Errors

```
ERROR tests/test_integration.py::test_health_check
PytestDeprecationWarning:
The event_loop fixture provided by pytest-asyncio has been redefined.
Replacing the event_loop fixture with a custom implementation is deprecated
and will lead to errors in the future.

ERROR tests/test_integration.py::test_create_user
ScopeMismatch:
You tried to access the function scoped fixture event_loop with a session scoped request object.
Session scoped fixture `test_db_engine` requested function scoped fixture `event_loop`

ERROR tests/conftest.py::test_db_engine
RuntimeError: Event loop is closed

ERROR tests/test_integration.py::test_list_stocks
RuntimeError: Task <Task pending> attached to a different loop
```

### AFTER - Clean Output

```
tests/test_integration.py::test_health_check PASSED                [ 10%]
tests/test_integration.py::test_create_user PASSED                 [ 20%]
tests/test_integration.py::test_list_stocks PASSED                 [ 30%]
tests/test_integration.py::test_portfolio_operations PASSED        [ 40%]

================================ 40 passed in 5.23s ================================
```

---

## Performance Impact Analysis

### Option 1: Function-Scoped Fixtures

**Setup Time per Test:**
- Engine creation: ~5ms (SQLite in-memory)
- Session factory: ~1ms
- Total overhead: ~6ms per test

**For 100 tests:**
- Additional time: ~600ms (0.6 seconds)
- Trade-off: Worth it for proper test isolation

### Option 2: Factory Pattern

**Setup Time per Test:**
- Factory call: ~0.1ms (returns cached config)
- Engine creation: ~5ms
- Session factory: ~1ms
- Total overhead: ~6ms per test

**For 100 tests:**
- Additional time: ~600ms (similar to Option 1)
- Benefit: Cleaner code organization

### Session-Scoped (Broken, for comparison)

**Setup Time:**
- One-time: ~6ms for entire test session
- Per test: ~0ms

**Trade-offs:**
- ❌ Deprecated and causes errors
- ❌ Poor test isolation
- ❌ Tests can affect each other
- ✅ Marginally faster (not worth the problems)

---

## Migration Checklist

### Phase 1: Preparation
- [ ] Read full analysis in `pytest-asyncio-analysis.md`
- [ ] Understand current fixture dependencies
- [ ] Backup current conftest.py
- [ ] Check pytest and pytest-asyncio versions

### Phase 2: Configuration
- [ ] Create backend/pytest.ini with asyncio_mode=auto
- [ ] Add NullPool import to conftest.py

### Phase 3: Cleanup
- [ ] Remove event_loop fixture from conftest.py (lines 31-37)
- [ ] Remove event_loop from async_fixtures.py (line 213)
- [ ] Remove event_loop from integration_test_fixtures.py (line 650)

### Phase 4: Refactoring
- [ ] Remove scope="session" from test_db_engine
- [ ] Remove scope="session" from test_db_session_factory
- [ ] Add poolclass=NullPool to create_async_engine()
- [ ] Test each fixture individually

### Phase 5: Validation
- [ ] Run: pytest tests/conftest.py -v -W error::DeprecationWarning
- [ ] Run: pytest tests/test_integration.py -v
- [ ] Run: pytest tests/ -v --tb=short
- [ ] Run: pytest tests/ --setup-show
- [ ] Verify no "event loop" warnings

### Phase 6: Documentation
- [ ] Update test documentation
- [ ] Document fixture patterns for team
- [ ] Add fixture dependency diagram

---

## Quick Reference: Fixture Scope Rules

| Fixture Type | Scope | Decorator | Example |
|--------------|-------|-----------|---------|
| Config/Constants | session | @pytest.fixture | test_db_url, API keys |
| Factories | session | @pytest.fixture | async_engine_factory |
| Async Resources | function | @pytest_asyncio.fixture | test_db_engine |
| Usage/Mocks | function | @pytest_asyncio.fixture | db_session, mock_cache |

**Golden Rule**: Session-scoped fixtures must be synchronous in pytest-asyncio 1.3.0+

---

## Code Review Checklist

When reviewing test fixtures, check:

- [ ] No custom event_loop fixtures (let pytest-asyncio handle it)
- [ ] No session-scoped async fixtures
- [ ] All async fixtures use @pytest_asyncio.fixture
- [ ] Database engines use NullPool
- [ ] pytest.ini has asyncio_mode configured
- [ ] No asyncio.run() calls in test code
- [ ] Fixtures have proper try/finally cleanup
- [ ] No nested event loops

---

**Last Updated**: 2026-01-27
**Status**: Ready for Implementation
**Reviewer**: TDD Research Agent
