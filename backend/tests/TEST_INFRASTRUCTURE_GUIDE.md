# Test Infrastructure Guide

**Last Updated**: 2026-01-27
**Version**: 1.0
**Status**: Production Ready

## Overview

This guide covers the comprehensive test infrastructure for the Investment Analysis Platform, including async fixture patterns, API response validation, pytest-asyncio configuration, and best practices for writing reliable tests.

## Table of Contents

1. [ApiResponse Wrapper Testing Pattern](#apiresponse-wrapper-testing-pattern)
2. [Validation Helper Functions](#validation-helper-functions)
3. [Pytest-Asyncio Configuration](#pytest-asyncio-configuration)
4. [Fixture Usage Guide](#fixture-usage-guide)
5. [Common Testing Patterns](#common-testing-patterns)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## ApiResponse Wrapper Testing Pattern

### Overview

All API endpoints in the Investment Analysis Platform return a standardized `ApiResponse` wrapper structure:

```python
{
    "success": true,
    "data": { ... },
    "meta": { ... }
}
```

For error responses:

```python
{
    "success": false,
    "error": "Error message"
}
```

### Why This Pattern?

- **Consistency**: Every API endpoint returns the same structure
- **Type Safety**: Response structure is validated by FastAPI
- **Client Friendliness**: Clients can always check `success` field
- **Metadata Support**: Includes pagination, timestamps, and other metadata
- **Error Standardization**: All errors follow the same format

### Response Structure

```typescript
// Success Response
interface ApiResponse<T> {
  success: boolean   // true
  data: T            // The actual response data
  meta?: {           // Optional metadata
    total?: number
    page?: number
    limit?: number
    timestamp?: string
  }
}

// Error Response
interface ApiErrorResponse {
  success: boolean   // false
  error: string      // Error message
  details?: unknown  // Additional error details
}
```

---

## Validation Helper Functions

### assert_success_response()

**Purpose**: Validate successful API responses and extract unwrapped data

**Signature**:
```python
def assert_success_response(
    response,
    expected_status: int = 200
) -> dict | list | Any:
    """
    Validate ApiResponse wrapper structure and return unwrapped data.

    Args:
        response: FastAPI TestClient response object
        expected_status: Expected HTTP status code (default: 200)

    Returns:
        Unwrapped data from response["data"]

    Raises:
        AssertionError: If response structure is invalid
    """
```

**Usage Examples**:

```python
# Basic usage
async def test_create_thesis(async_client):
    response = await async_client.post(
        "/api/theses",
        json={"title": "Tech Growth", "description": "..."}
    )

    # Validate and extract data
    data = assert_success_response(response, expected_status=201)

    # Now assert on unwrapped data
    assert data["title"] == "Tech Growth"
    assert "id" in data

# With status code validation
async def test_get_thesis(async_client):
    response = await async_client.get("/api/theses/123")

    # Expects 200, verify with custom status
    data = assert_success_response(response, expected_status=200)
    assert data["id"] == 123

# List responses
async def test_list_theses(async_client):
    response = await async_client.get("/api/theses")

    # Returns list from data field
    theses = assert_success_response(response)
    assert isinstance(theses, list)
    assert len(theses) > 0

# With metadata
async def test_paginated_list(async_client):
    response = await async_client.get("/api/theses?page=1&limit=10")

    data = assert_success_response(response)
    # Metadata is NOT in returned data, but in response JSON
    json_data = response.json()
    assert json_data["meta"]["page"] == 1
```

**Internal Assertions**:

```python
# The function validates:
assert response.status_code == expected_status
assert response.json()["success"] == True
assert "data" in response.json()
# Then returns: response.json()["data"]
```

---

### assert_api_error_response()

**Purpose**: Validate API error responses

**Signature**:
```python
def assert_api_error_response(
    response,
    expected_status: int,
    expected_error_substring: str | None = None
) -> dict:
    """
    Validate ApiResponse error structure.

    Args:
        response: FastAPI TestClient response object
        expected_status: Expected HTTP error status code
        expected_error_substring: Optional substring to verify in error message

    Returns:
        Full response JSON data

    Raises:
        AssertionError: If response structure is invalid
    """
```

**Usage Examples**:

```python
# Basic 404 error
async def test_thesis_not_found(async_client):
    response = await async_client.get("/api/theses/999")

    # Validate error structure
    error_data = assert_api_error_response(response, expected_status=404)
    assert "error" in error_data

# With message validation
async def test_invalid_thesis_title(async_client):
    response = await async_client.post(
        "/api/theses",
        json={"title": "", "description": "..."}  # Invalid
    )

    error_data = assert_api_error_response(
        response,
        expected_status=422,
        expected_error_substring="title"
    )

# Unauthorized access
async def test_thesis_unauthorized(async_client):
    # No auth headers
    response = await async_client.get("/api/theses/private/123")

    error_data = assert_api_error_response(
        response,
        expected_status=401,
        expected_error_substring="authentication"
    )

# Forbidden access
async def test_thesis_forbidden(async_client, auth_headers):
    # Different user's thesis
    response = await async_client.delete(
        "/api/theses/123",
        headers=auth_headers  # User doesn't own thesis
    )

    error_data = assert_api_error_response(
        response,
        expected_status=403,
        expected_error_substring="permission"
    )
```

**Internal Assertions**:

```python
# The function validates:
assert response.status_code == expected_status
assert response.json()["success"] == False
# If expected_error_substring provided:
assert expected_error_substring.lower() in response.json()["error"].lower()
# Then returns: response.json()
```

---

## Pytest-Asyncio Configuration

### Configuration File

Configuration is in `/pytest.ini`:

```ini
[tool:pytest]
# Asyncio configuration
asyncio_mode = strict
asyncio_default_fixture_loop_scope = function
```

### asyncio_mode = strict

**What it means**: Pytest-asyncio operates in strict mode, ensuring:
- Proper event loop management
- No implicit async context
- Explicit async marker required

**Why it matters**: Prevents hard-to-debug async issues and race conditions

### asyncio_default_fixture_loop_scope = function

**What it means**: Event loop scope defaults to function-level
- Each test function gets its own event loop
- Fixtures can specify different scopes

**Why it matters**: Ensures test isolation and prevents cross-test pollution

---

### Using Async Fixtures

All database and client fixtures are async:

```python
# Session-scoped (created once per test session)
@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    """Create test database engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()

# Function-scoped (created for each test)
@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    """Provide database session for tests."""
    async with test_db_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

# Used in async tests
@pytest.mark.asyncio
async def test_something(db_session):
    # db_session is automatically awaited and injected
    result = await db_session.execute(...)
```

### Event Loop Lifecycle

```
Session Start
    ↓
create event_loop (session-scoped)
    ↓
test_db_engine (session-scoped) - created once
    ↓
test_db_session_factory (session-scoped) - created once
    ↓
    ├─ Test 1 starts
    │   ├─ db_session (function-scoped) - created
    │   ├─ Run test
    │   └─ db_session cleanup (rollback)
    │
    ├─ Test 2 starts
    │   ├─ db_session (function-scoped) - created
    │   ├─ Run test
    │   └─ db_session cleanup (rollback)
    │
    └─ More tests...
    ↓
test_db_engine cleanup (dispose)
    ↓
event_loop cleanup
    ↓
Session End
```

---

## Fixture Usage Guide

### Available Fixtures

All fixtures are defined in `/backend/tests/conftest.py`:

#### Database Fixtures

**1. test_db_engine** (Session-scoped)
```python
@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    """Create test database engine."""
```

Usage:
```python
async def test_something(test_db_engine):
    # Access the engine
    async with test_db_engine.connect() as conn:
        result = await conn.execute(...)
```

**2. test_db_session_factory** (Session-scoped)
```python
@pytest_asyncio.fixture(scope="session")
async def test_db_session_factory(test_db_engine):
    """Create test database session factory."""
```

Usage:
```python
async def test_something(test_db_session_factory):
    # Create a session
    async with test_db_session_factory() as session:
        result = await session.execute(...)
```

**3. db_session** (Function-scoped)
```python
@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    """Provide database session for tests."""
```

Usage:
```python
async def test_something(db_session):
    # Session automatically available, rolled back after test
    result = await db_session.execute(...)
    # Automatic rollback on exit
```

---

#### HTTP Client Fixtures

**1. async_client** (Function-scoped)
```python
@pytest_asyncio.fixture
async def async_client():
    """Provide async HTTP client for API testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
```

Usage:
```python
async def test_api_endpoint(async_client):
    response = await async_client.get("/api/endpoint")
    assert response.status_code == 200

async def test_post_endpoint(async_client):
    response = await async_client.post(
        "/api/endpoint",
        json={"data": "value"}
    )
    data = assert_success_response(response)
```

**2. client** (Function-scoped, alias)
```python
@pytest_asyncio.fixture
async def client(async_client):
    """Alias for async_client for backward compatibility."""
    return async_client
```

---

#### Authentication Fixtures

**1. test_user** (Function-scoped)
```python
@pytest.fixture
def test_user():
    """Provide test user."""
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        ...
    )
```

**2. auth_token** (Function-scoped)
```python
@pytest.fixture
def auth_token(test_user):
    """Provide authentication token."""
    return create_access_token(
        data={"sub": str(test_user.id), "username": test_user.username}
    )
```

**3. auth_headers** (Function-scoped)
```python
@pytest.fixture
def auth_headers(auth_token):
    """Provide authentication headers."""
    return {"Authorization": f"Bearer {auth_token}"}
```

Usage:
```python
async def test_authenticated_endpoint(async_client, auth_headers):
    response = await async_client.get(
        "/api/protected",
        headers=auth_headers
    )
    data = assert_success_response(response)
```

---

#### Mock Fixtures

**1. mock_current_user** (Function-scoped)
```python
@pytest.fixture
def mock_current_user(test_user):
    """Mock current user dependency."""
    with patch.object(app, 'dependency_overrides') as mock_overrides:
        mock_overrides[get_current_user] = lambda: test_user
        yield test_user
        mock_overrides.clear()
```

Usage:
```python
async def test_with_mocked_user(mock_current_user, async_client):
    # Current user is automatically mocked
    response = await async_client.get("/api/protected")
    data = assert_success_response(response)
```

**2. mock_cache** (Function-scoped)
```python
@pytest.fixture
async def mock_cache():
    """Provide mock cache manager."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    # ... more mocks
```

**3. mock_external_apis** (Function-scoped)
```python
@pytest.fixture
def mock_external_apis():
    """Mock all external API calls."""
    # Provides realistic mock responses for Alpha Vantage, Finnhub, Polygon
```

Usage:
```python
async def test_with_mocked_apis(async_client, mock_external_apis):
    # External APIs are mocked
    response = await async_client.get("/api/stock/AAPL")
    data = assert_success_response(response)
    # Uses mocked data instead of real API
```

---

#### Other Fixtures

**setup_test_environment** (Function-scoped, autouse)
```python
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    # Automatically runs before each test
    # Sets required test environment variables
```

**performance_threshold** (Function-scoped)
```python
@pytest.fixture
def performance_threshold():
    """Define performance thresholds for tests."""
    return {
        "api_response": 2.0,      # seconds
        "database_query": 1.0,    # seconds
        "cache_operation": 0.1,   # seconds
        "websocket_message": 0.5, # seconds
        "bulk_operation": 10.0,   # seconds
    }
```

Usage:
```python
import time

async def test_api_performance(async_client, performance_threshold):
    start = time.time()
    response = await async_client.get("/api/endpoint")
    duration = time.time() - start

    assert_performance_threshold(
        duration,
        performance_threshold["api_response"],
        "GET /api/endpoint"
    )
```

---

### Fixture Dependencies

Fixtures can depend on other fixtures. The dependency order is:

```
event_loop (session)
    ↓
test_db_engine (session)
    ↓
test_db_session_factory (session)
    ↓
db_session (function)
    ↓
async_client (function)
    ↓
test_user (function)
    ↓
auth_token (function)
    ↓
auth_headers (function)
```

**Never use a session-scoped fixture inside a function-scoped fixture.**

---

## Common Testing Patterns

### Pattern 1: Basic API Test

```python
@pytest.mark.api
async def test_get_thesis(async_client):
    """Test getting a thesis by ID."""
    # Create test data
    thesis = await create_test_thesis()

    # Make request
    response = await async_client.get(f"/api/theses/{thesis.id}")

    # Validate response
    data = assert_success_response(response)
    assert data["id"] == thesis.id
    assert data["title"] == thesis.title
```

### Pattern 2: Authenticated API Test

```python
@pytest.mark.api
async def test_create_thesis(async_client, auth_headers):
    """Test creating a thesis with authentication."""
    # Prepare request
    payload = {
        "title": "AI Investments",
        "description": "Analysis of AI sector",
        "thesis_type": "bullish"
    }

    # Make authenticated request
    response = await async_client.post(
        "/api/theses",
        json=payload,
        headers=auth_headers
    )

    # Validate response
    data = assert_success_response(response, expected_status=201)
    assert data["title"] == payload["title"]
    assert "id" in data
```

### Pattern 3: Error Handling Test

```python
@pytest.mark.api
async def test_thesis_not_found(async_client):
    """Test error when thesis doesn't exist."""
    # Make request for non-existent resource
    response = await async_client.get("/api/theses/999")

    # Validate error response
    error_data = assert_api_error_response(
        response,
        expected_status=404,
        expected_error_substring="not found"
    )
```

### Pattern 4: Database Test with Fixture

```python
@pytest.mark.database
async def test_create_portfolio(db_session):
    """Test portfolio creation in database."""
    # Use provided session
    portfolio = Portfolio(
        name="Test Portfolio",
        description="For testing",
        cash_balance=10000.0
    )

    db_session.add(portfolio)
    await db_session.flush()

    # Verify in database
    result = await db_session.execute(
        select(Portfolio).where(Portfolio.id == portfolio.id)
    )
    retrieved = result.scalar_one()
    assert retrieved.name == "Test Portfolio"
```

### Pattern 5: End-to-End Flow Test

```python
@pytest.mark.integration
async def test_create_and_update_thesis(async_client, auth_headers):
    """Test creating and updating a thesis."""
    # Step 1: Create
    create_response = await async_client.post(
        "/api/theses",
        json={"title": "Original", "description": "..."},
        headers=auth_headers
    )
    thesis_data = assert_success_response(create_response, expected_status=201)
    thesis_id = thesis_data["id"]

    # Step 2: Update
    update_response = await async_client.put(
        f"/api/theses/{thesis_id}",
        json={"title": "Updated", "description": "..."},
        headers=auth_headers
    )
    updated_data = assert_success_response(update_response)
    assert updated_data["title"] == "Updated"

    # Step 3: Verify
    get_response = await async_client.get(
        f"/api/theses/{thesis_id}",
        headers=auth_headers
    )
    final_data = assert_success_response(get_response)
    assert final_data["title"] == "Updated"
```

### Pattern 6: Performance Test

```python
@pytest.mark.performance
async def test_list_theses_performance(async_client, performance_threshold):
    """Test that listing theses meets performance requirements."""
    start = time.time()

    response = await async_client.get("/api/theses?limit=100")

    duration = time.time() - start

    # Validate performance
    assert_performance_threshold(
        duration,
        performance_threshold["api_response"],
        "GET /api/theses"
    )

    # Validate correctness
    data = assert_success_response(response)
    assert isinstance(data, list)
```

### Pattern 7: Mocked External API Test

```python
@pytest.mark.integration
async def test_fetch_stock_with_mocked_api(async_client, mock_external_apis):
    """Test fetching stock data with mocked external API."""
    response = await async_client.get("/api/stocks/AAPL")

    data = assert_success_response(response)
    assert data["symbol"] == "AAPL"
    # Uses mocked API data, not real API
```

### Pattern 8: Test with Multiple Assertions

```python
@pytest.mark.api
async def test_thesis_comprehensive(async_client, auth_headers):
    """Comprehensive test of thesis endpoint."""
    # Setup
    thesis_data = {
        "title": "Growth Stocks",
        "description": "Focus on growth",
        "thesis_type": "bullish",
        "confidence_score": 0.85
    }

    # Create
    response = await async_client.post(
        "/api/theses",
        json=thesis_data,
        headers=auth_headers
    )

    # Multiple assertions
    created = assert_success_response(response, expected_status=201)

    # Verify all fields
    assert created["title"] == thesis_data["title"]
    assert created["description"] == thesis_data["description"]
    assert created["thesis_type"] == thesis_data["thesis_type"]
    assert created["confidence_score"] == thesis_data["confidence_score"]
    assert "id" in created
    assert "created_at" in created
    assert "updated_at" in created
```

---

## Troubleshooting Guide

### Issue 1: "RuntimeError: Event loop is closed"

**Cause**: Event loop cleanup issue, typically with session-scoped fixtures

**Solution**:
```python
# In conftest.py, ensure proper cleanup:
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()  # Important!
```

---

### Issue 2: "assert_success_response() missing 'data' field"

**Cause**: Endpoint not returning ApiResponse wrapper

**Solution**: Check that all endpoints use `success_response()` wrapper:

```python
# WRONG - returns plain dict
@router.get("/thesis")
async def get_thesis():
    return {"id": 1, "title": "..."}

# CORRECT - returns ApiResponse wrapper
@router.get("/thesis")
async def get_thesis():
    return success_response(data={"id": 1, "title": "..."})
```

---

### Issue 3: "Database session not yielding"

**Cause**: Async fixture not properly awaited or yielded

**Solution**: Ensure fixture uses `async` and `yield`:

```python
# WRONG - missing async
@pytest_asyncio.fixture
def db_session(test_db_session_factory):
    async with test_db_session_factory() as session:
        yield session

# CORRECT - async and yield
@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    async with test_db_session_factory() as session:
        yield session
```

---

### Issue 4: "Test hangs indefinitely"

**Cause**: Missing await on async operation

**Solution**: Always await async operations:

```python
# WRONG - missing await
async def test_something(async_client):
    response = async_client.get("/api/endpoint")  # Missing await!

# CORRECT - with await
async def test_something(async_client):
    response = await async_client.get("/api/endpoint")
```

---

### Issue 5: "assert_success_response() fails unexpectedly"

**Cause**: Response status code doesn't match expected

**Solution**: Check the actual response:

```python
async def test_endpoint(async_client):
    response = await async_client.get("/api/endpoint")

    # Debug: print actual response
    print(f"Status: {response.status_code}")
    print(f"Body: {response.json()}")

    # Then assert with correct status
    data = assert_success_response(response, expected_status=200)
```

---

### Issue 6: "Database not rolling back after test"

**Cause**: Session not properly cleaned up

**Solution**: Ensure fixture properly rollbacks:

```python
@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    async with test_db_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()  # Important!
            await session.close()
```

---

### Issue 7: "Fixture scope mismatch error"

**Cause**: Function-scoped fixture depending on session-scoped

**Solution**: Match fixture scopes properly:

```python
# WRONG - function fixture can't depend on session fixture that yields function data
@pytest_asyncio.fixture
async def db_session(session_factory):  # Session-scoped
    session = session_factory()
    yield session

# CORRECT - both same scope or function depends on session
@pytest_asyncio.fixture(scope="session")
async def session_factory(engine):
    return async_sessionmaker(engine)

@pytest_asyncio.fixture
async def db_session(session_factory):  # Function scope OK
    async with session_factory() as session:
        yield session
```

---

### Issue 8: "Import error for assert_success_response()"

**Cause**: Helper function not imported from conftest

**Solution**: Ensure it's in conftest.py and import properly:

```python
# In conftest.py - already defined there
def assert_success_response(response, expected_status=200):
    """..."""

# In test file - it's available automatically
from backend.tests.conftest import assert_success_response

# OR just use it directly
async def test_something(async_client):
    response = await async_client.get("/api/endpoint")
    data = assert_success_response(response)
```

---

### Issue 9: "Mock fixture not working"

**Cause**: Mock not properly patched

**Solution**: Verify patch path is correct:

```python
# WRONG - patch where used, not where defined
@patch('backend.api.main.get_cache_manager')
async def test_with_cache(mock_cache):
    ...

# CORRECT - patch where it's imported
with patch('backend.utils.comprehensive_cache.get_cache_manager'):
    ...
```

---

### Issue 10: "Performance threshold exceeded"

**Cause**: Test execution too slow

**Solution**:
1. Check if test is legitimately slow (database, network)
2. Optimize the tested code
3. Adjust threshold if appropriate

```python
async def test_slow_operation(async_client, performance_threshold):
    start = time.time()
    response = await async_client.get("/api/expensive-endpoint")
    duration = time.time() - start

    # Increase threshold if needed
    threshold = performance_threshold.get("bulk_operation", 10.0)
    assert_performance_threshold(duration, threshold * 1.5, "expensive operation")
```

---

## Best Practices

### 1. Always Use Helper Functions

```python
# GOOD
data = assert_success_response(response)
assert data["field"] == value

# BAD
assert response.json()["success"] == True
assert response.json()["data"]["field"] == value
```

### 2. Explicit Status Codes

```python
# GOOD - explicit status code
data = assert_success_response(response, expected_status=201)

# BAD - defaults to 200
data = assert_success_response(response)
```

### 3. Use Appropriate Fixtures

```python
# GOOD - use async_client fixture
async def test_endpoint(async_client):
    response = await async_client.get("/api/endpoint")

# BAD - creating client manually
async def test_endpoint():
    client = AsyncClient(app=app)
    response = await client.get("/api/endpoint")
```

### 4. Proper Error Message Checking

```python
# GOOD - check substring
error_data = assert_api_error_response(
    response,
    expected_status=404,
    expected_error_substring="not found"
)

# BAD - no verification
error_data = assert_api_error_response(response, expected_status=404)
```

### 5. Clean Test Data

```python
# GOOD - use fixtures
async def test_endpoint(async_client, db_session):
    portfolio = await create_test_portfolio(db_session)
    response = await async_client.get(f"/api/portfolios/{portfolio.id}")

# BAD - hardcoded IDs
async def test_endpoint(async_client):
    response = await async_client.get("/api/portfolios/1")
```

---

## Additional Resources

- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [FastAPI Testing Documentation](https://fastapi.tiangolo.com/advanced/testing/)
- [SQLAlchemy AsyncIO Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- Main test README: [backend/tests/README.md](./README.md)
- Example tests: See any test file in this directory

---

**Last Updated**: 2026-01-27
**Maintained By**: Investment Analysis Platform Team
