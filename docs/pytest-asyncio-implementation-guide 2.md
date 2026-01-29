# Pytest-Asyncio Implementation Guide - Copy-Paste Ready

## Quick Fix (5 Minutes)

This guide provides exact code changes needed to fix the pytest-asyncio configuration issues.

---

## File 1: Create pytest.ini

**Location**: `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/pytest.ini`

**Action**: Create new file with this content:

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
```

---

## File 2: Fix conftest.py

**Location**: `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/conftest.py`

### Change 1: Add NullPool Import (Line 14)

**FIND:**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
```

**REPLACE WITH:**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
```

### Change 2: Delete Event Loop Fixture (Lines 31-37)

**FIND AND DELETE:**
```python
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
```

**REPLACE WITH:**
```python
# Event loop is now managed automatically by pytest-asyncio
# No custom event_loop fixture needed
```

### Change 3: Fix test_db_engine Fixture (Lines 102-118)

**FIND:**
```python
@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    """Create test database engine."""
    # Use test database URL if available, otherwise use in-memory SQLite
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"  # Fallback to in-memory for CI/CD
    )

    engine = create_async_engine(
        test_db_url,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True
    )

    yield engine
    await engine.dispose()
```

**REPLACE WITH:**
```python
@pytest_asyncio.fixture  # Removed scope="session" - now function-scoped
async def test_db_engine():
    """Create test database engine per test for isolation."""
    # Use test database URL if available, otherwise use in-memory SQLite
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"  # Fallback to in-memory for CI/CD
    )

    engine = create_async_engine(
        test_db_url,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,
        poolclass=NullPool  # CRITICAL: Disable pooling for test isolation
    )

    yield engine
    await engine.dispose()
```

### Change 4: Fix test_db_session_factory Fixture (Lines 121-129)

**FIND:**
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

**REPLACE WITH:**
```python
@pytest_asyncio.fixture  # Removed scope="session" - now function-scoped
async def test_db_session_factory(test_db_engine):
    """Create test database session factory."""
    TestSessionLocal = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    return TestSessionLocal
```

---

## File 3: Fix async_fixtures.py

**Location**: `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/async_fixtures.py`

### Change: Delete Event Loop Fixture (Lines 212-217)

**FIND AND DELETE:**
```python
@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for entire test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
```

**REPLACE WITH:**
```python
# Event loop is now managed automatically by pytest-asyncio
# Removed custom event_loop fixture to prevent conflicts
```

---

## File 4: Fix integration_test_fixtures.py

**Location**: `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/fixtures/integration_test_fixtures.py`

### Change: Delete Event Loop Fixture (Lines 649-654)

**FIND AND DELETE:**
```python
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

**REPLACE WITH:**
```python
# Event loop is now managed automatically by pytest-asyncio
# Removed custom event_loop fixture to prevent conflicts
```

---

## Complete Fixed conftest.py (Key Sections)

Here's what the fixed sections should look like:

```python
"""
Global Test Configuration for Investment Analysis Platform Integration Tests
Provides shared fixtures, test utilities, and configuration for all test modules.
"""

import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import patch, AsyncMock
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool  # ✅ ADDED
from httpx import AsyncClient

from backend.api.main import app
from backend.config.database import get_async_db_session, initialize_database
from backend.auth.oauth2 import get_current_user, create_access_token
from backend.models.unified_models import User
from backend.utils.comprehensive_cache import get_cache_manager
from backend.config.settings import settings


# Configure test logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ✅ REMOVED: Event loop fixture - pytest-asyncio handles this automatically


# ============================================================================
# ApiResponse Wrapper Validation Helpers
# ============================================================================

def assert_success_response(response, expected_status=200):
    """
    Validate ApiResponse wrapper structure and return unwrapped data.

    [... rest of function unchanged ...]
    """
    # ... existing code ...


def assert_api_error_response(response, expected_status, expected_error_substring=None):
    """
    Validate ApiResponse error structure.

    [... rest of function unchanged ...]
    """
    # ... existing code ...


# ============================================================================
# Database Fixtures (Function-scoped for test isolation)
# ============================================================================

@pytest_asyncio.fixture  # ✅ CHANGED: Removed scope="session"
async def test_db_engine():
    """Create test database engine per test for isolation."""
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )

    engine = create_async_engine(
        test_db_url,
        echo=False,
        pool_pre_ping=True,
        poolclass=NullPool  # ✅ ADDED: Disable pooling for test isolation
    )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture  # ✅ CHANGED: Removed scope="session"
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
            await session.rollback()  # Rollback any changes
            await session.close()


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def async_client():
    """Provide async HTTP client for API testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def client(async_client):
    """Alias for async_client for backward compatibility."""
    return async_client


# [... rest of file unchanged ...]
```

---

## Validation Script

Create this script to verify the fix:

**Location**: `/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/validate_fix.sh`

```bash
#!/bin/bash

echo "==================================="
echo "Pytest-Asyncio Configuration Check"
echo "==================================="

cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend

echo ""
echo "1. Checking for pytest.ini..."
if [ -f "pytest.ini" ]; then
    echo "✅ pytest.ini exists"
    grep "asyncio_mode" pytest.ini && echo "✅ asyncio_mode configured"
else
    echo "❌ pytest.ini missing"
fi

echo ""
echo "2. Checking for custom event_loop fixtures..."
event_loop_count=$(grep -r "def event_loop" tests/ | wc -l)
if [ "$event_loop_count" -eq 0 ]; then
    echo "✅ No custom event_loop fixtures found"
else
    echo "❌ Found $event_loop_count custom event_loop fixtures:"
    grep -rn "def event_loop" tests/
fi

echo ""
echo "3. Checking for NullPool imports..."
nullpool_count=$(grep -r "from sqlalchemy.pool import NullPool" tests/ | wc -l)
if [ "$nullpool_count" -gt 0 ]; then
    echo "✅ NullPool imported"
else
    echo "⚠️  NullPool not imported (optional but recommended)"
fi

echo ""
echo "4. Checking for session-scoped async fixtures..."
session_async_count=$(grep -r "@pytest_asyncio.fixture(scope=\"session\")" tests/ | wc -l)
if [ "$session_async_count" -eq 0 ]; then
    echo "✅ No session-scoped async fixtures"
else
    echo "❌ Found $session_async_count session-scoped async fixtures:"
    grep -rn "@pytest_asyncio.fixture(scope=\"session\")" tests/
fi

echo ""
echo "5. Running pytest with deprecation warnings..."
pytest tests/conftest.py -v -W error::DeprecationWarning 2>&1 | head -20

echo ""
echo "6. Running a sample test..."
pytest tests/test_integration.py::test_health_check -v 2>&1 | tail -20

echo ""
echo "==================================="
echo "Validation Complete"
echo "==================================="
```

**Make it executable:**
```bash
chmod +x /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend/tests/validate_fix.sh
```

---

## Step-by-Step Implementation

### Step 1: Backup Current Files
```bash
cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend
cp tests/conftest.py tests/conftest.py.backup
cp tests/async_fixtures.py tests/async_fixtures.py.backup
cp tests/fixtures/integration_test_fixtures.py tests/fixtures/integration_test_fixtures.py.backup
```

### Step 2: Create pytest.ini
Copy the pytest.ini content from above into new file.

### Step 3: Edit conftest.py
Apply the 4 changes listed above:
1. Add NullPool import
2. Delete event_loop fixture
3. Fix test_db_engine (remove scope, add NullPool)
4. Fix test_db_session_factory (remove scope)

### Step 4: Edit async_fixtures.py
Delete the event_loop fixture (lines 212-217).

### Step 5: Edit integration_test_fixtures.py
Delete the event_loop fixture (lines 649-654).

### Step 6: Run Validation
```bash
cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/backend

# Check for deprecation warnings
pytest tests/conftest.py -v -W error::DeprecationWarning

# Run single test
pytest tests/test_integration.py::test_health_check -v -s

# Run all tests
pytest tests/ -v --tb=short

# Check fixture setup
pytest tests/ -v --setup-show | head -50
```

### Step 7: Verify Success
Look for:
- ✅ No "event_loop fixture has been redefined" warnings
- ✅ No "ScopeMismatch" errors
- ✅ No "Event loop is closed" errors
- ✅ All tests pass

---

## Common Issues After Implementation

### Issue 1: "fixture 'event_loop' not found"

**Cause**: pytest-asyncio not installed or wrong version

**Fix**:
```bash
pip install "pytest-asyncio>=1.3.0"
```

### Issue 2: Tests still fail with event loop errors

**Cause**: Cached pytest files

**Fix**:
```bash
rm -rf .pytest_cache
rm -rf tests/__pycache__
pytest --cache-clear tests/
```

### Issue 3: "poolclass is not a valid keyword argument"

**Cause**: Wrong SQLAlchemy version

**Fix**:
```bash
pip install "sqlalchemy>=2.0.0"
```

### Issue 4: Slow test execution

**Cause**: Creating engine per test

**Consider**: Module-scoped fixtures
```python
@pytest_asyncio.fixture(scope="module")
async def test_db_engine():
    # ... same code but module scope
```

---

## Alternative Implementation (Factory Pattern)

If you prefer the factory pattern for better performance:

```python
# Sync session-scoped configuration
@pytest.fixture(scope="session")
def test_db_url():
    """Provide test database URL."""
    return os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")

@pytest.fixture(scope="session")
def async_engine_factory(test_db_url):
    """Factory for creating async engines."""
    def _create_engine():
        return create_async_engine(
            test_db_url,
            echo=False,
            pool_pre_ping=True,
            poolclass=NullPool
        )
    return _create_engine

# Async function-scoped resource
@pytest_asyncio.fixture
async def test_db_engine(async_engine_factory):
    """Create async engine per test."""
    engine = async_engine_factory()
    yield engine
    await engine.dispose()

# Rest stays the same
@pytest_asyncio.fixture
async def test_db_session_factory(test_db_engine):
    """Create session factory."""
    TestSessionLocal = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    return TestSessionLocal

@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    """Provide database session."""
    async with test_db_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
```

---

## Final Checklist

Before committing changes:

- [ ] Created pytest.ini with asyncio_mode=auto
- [ ] Removed all custom event_loop fixtures (3 files)
- [ ] Added NullPool import to conftest.py
- [ ] Removed scope="session" from async fixtures
- [ ] Added poolclass=NullPool to engine configuration
- [ ] Backed up original files
- [ ] Ran validation commands
- [ ] All tests pass
- [ ] No deprecation warnings
- [ ] Committed changes with descriptive message

---

**Implementation Time**: ~5-10 minutes
**Testing Time**: ~5 minutes
**Total Time**: ~15 minutes

**Status**: Ready to implement
**Risk Level**: Low (reversible with backups)
**Impact**: Fixes all pytest-asyncio configuration issues
