# Pytest-Asyncio Configuration Fixes - Quick Reference

## Problem Summary

Session-scoped async fixtures in `backend/tests/conftest.py` are causing pytest 9.x deprecation warnings and setup errors due to incompatibility with pytest-asyncio 1.3.0+.

## Root Causes

1. **Custom event_loop fixture** (line 32) conflicts with pytest-asyncio's automatic event loop management
2. **Session-scoped async fixtures** (`test_db_engine`, `test_db_session_factory`) are not properly supported in pytest-asyncio 1.3.0+
3. **Multiple event_loop fixtures** across different test files cause fixture override conflicts
4. **Missing pytest.ini** configuration for asyncio_mode

## Files Affected

- `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/conftest.py` (lines 31-37, 102-129)
- `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/async_fixtures.py` (line 213)
- `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/fixtures/integration_test_fixtures.py` (line 650)

## Recommended Solution (Option 1: Simplest)

### Step 1: Create pytest.ini

Create `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
testpaths = tests
python_files = test_*.py
addopts = -v --strict-markers --tb=short
```

### Step 2: Remove Custom Event Loop Fixtures

Delete these fixtures:
- `conftest.py` lines 31-37
- `async_fixtures.py` lines 212-217
- `integration_test_fixtures.py` lines 649-654

### Step 3: Convert to Function-Scoped Fixtures

Replace in `conftest.py`:

```python
from sqlalchemy.pool import NullPool

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
        poolclass=NullPool  # CRITICAL: Disable pooling for test isolation
    )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture  # Remove scope="session"
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

## Alternative Solution (Option 2: Factory Pattern)

Use synchronous session-scoped factories with async function-scoped resources:

```python
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
    from sqlalchemy.pool import NullPool

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

## Critical Implementation Details

1. **Import NullPool**: `from sqlalchemy.pool import NullPool`
2. **Add to engine config**: `poolclass=NullPool` in `create_async_engine()`
3. **Remove all @pytest.fixture decorators** on event_loop (let pytest-asyncio handle it)
4. **Use @pytest_asyncio.fixture** for all async fixtures
5. **Remove scope="session"** from all async fixtures

## Validation Steps

```bash
# 1. Check for deprecation warnings
cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend
pytest tests/conftest.py -v -W error::DeprecationWarning

# 2. Run single test
pytest tests/test_integration.py::test_health_check -v -s

# 3. Run all tests
pytest tests/ -v --tb=short

# 4. Check fixture setup/teardown
pytest tests/ -v --setup-show

# 5. Verify no event loop warnings
pytest tests/ -v 2>&1 | grep -i "event loop"
```

## Success Criteria

- [ ] No deprecation warnings about event loops
- [ ] No "session-scoped async fixture" errors
- [ ] All tests pass consistently
- [ ] Test isolation verified
- [ ] No "Event loop is closed" errors

## Key Takeaways for TDD Agent

1. **Always use function-scoped async fixtures** in pytest-asyncio 1.3.0+
2. **Let pytest-asyncio manage event loops** - don't create custom event_loop fixtures
3. **Use NullPool for database engines** in tests to ensure isolation
4. **Factory pattern** allows session-scoped config with function-scoped resources
5. **pytest.ini with asyncio_mode=auto** is the recommended configuration

## Performance Considerations

Function-scoped fixtures have slight overhead compared to session-scoped, but:
- SQLite in-memory databases are extremely fast (milliseconds to create)
- Test isolation is more valuable than marginal performance gains
- Can use module scope (`@pytest_asyncio.fixture(scope="module")`) as middle ground

## Next Steps for Implementation Agent

1. Create pytest.ini file
2. Remove all custom event_loop fixtures
3. Add NullPool imports and configuration
4. Convert session-scoped async fixtures to function-scoped
5. Run validation suite
6. Update test documentation

---

**For detailed analysis, see**: `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/pytest-asyncio-analysis.md`
