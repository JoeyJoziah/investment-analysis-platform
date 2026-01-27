# Testing Guide for Investment Analysis Platform

## Quick Start

### Running Tests

```bash
# All tests
pytest backend/tests/ -v

# Specific test file
pytest backend/tests/test_api_integration.py -v

# Specific test
pytest backend/tests/test_api_integration.py::TestAPIEndpointsIntegration::test_health_endpoint -v

# With coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Failed tests only
pytest backend/tests/ --lf

# With output
pytest backend/tests/ -v -s
```

### Test Organization

Tests are organized by functionality in `backend/tests/`:
- **test_api_integration.py** - API endpoints
- **test_websocket_integration.py** - Real-time updates
- **test_error_scenarios.py** - Error handling & resilience
- **test_security_compliance.py** - Security & compliance
- **test_database_integration.py** - Database operations
- **test_performance_load.py** - Performance & load testing
- **test_watchlist.py** - Watchlist management
- And 20 more specialized test files

### Key Fixtures

```python
# Async HTTP client
async def test_api(async_client):
    response = await async_client.get("/api/endpoint")
    assert response.status_code == 200

# Database session
async def test_db(db_session):
    await db_session.execute(query)
    await db_session.commit()

# Authenticated user
def test_auth(test_user):
    assert test_user.email == "test@example.com"

# Event loop for async tests
def test_async(event_loop):
    result = event_loop.run_until_complete(async_func())
```

---

## Writing Tests

### Test Structure (Arrange-Act-Assert)

```python
@pytest.mark.asyncio
async def test_user_registration_success(async_client, db_session):
    # ARRANGE - Set up test data
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecurePass123!"
    }

    # ACT - Perform the action
    response = await async_client.post("/api/users/register", json=user_data)

    # ASSERT - Verify the result
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == user_data["email"]
    assert "id" in data
```

### Unit Test Example

```python
from backend.utils.validators import validate_email

def test_validate_email_valid():
    """Test email validation with valid email."""
    assert validate_email("test@example.com") is True

def test_validate_email_invalid():
    """Test email validation with invalid email."""
    assert validate_email("invalid-email") is False

def test_validate_email_empty():
    """Test email validation with empty string."""
    assert validate_email("") is False
```

### Integration Test Example

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_portfolio_add_stock(async_client, db_session, test_user):
    """Test adding stock to portfolio."""
    # Set up authentication
    token = create_access_token({"sub": str(test_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Add stock
    stock_data = {
        "symbol": "AAPL",
        "quantity": 10,
        "purchase_price": 150.00
    }

    response = await async_client.post(
        "/api/portfolio/stocks",
        json=stock_data,
        headers=headers
    )

    # Verify response
    assert response.status_code == 201

    # Verify database
    stocks = await db_session.execute(
        select(Stock).where(Stock.user_id == test_user.id)
    )
    assert len(stocks.scalars().all()) > 0
```

### Error Handling Test Example

```python
@pytest.mark.asyncio
async def test_api_rate_limiting(async_client):
    """Test API rate limiting."""
    # Make requests until rate limited
    for i in range(101):  # Assume 100 req/min limit
        response = await async_client.get("/api/endpoint")

        if i < 100:
            assert response.status_code == 200
        else:
            assert response.status_code == 429
            assert "Retry-After" in response.headers
```

### Async Test Example

```python
@pytest.mark.asyncio
async def test_websocket_subscription(async_client):
    """Test WebSocket price subscription."""
    async with async_client.websocket_connect(
        "/ws/prices?token=test_token"
    ) as ws:
        # Subscribe
        await ws.send_json({"action": "subscribe", "symbols": ["AAPL"]})

        # Receive subscription confirmation
        msg = await ws.receive_json()
        assert msg["type"] == "subscription_confirmed"

        # Receive price update
        msg = await ws.receive_json()
        assert msg["type"] == "price_update"
        assert msg["symbol"] == "AAPL"
```

---

## Markers & Filtering

### Available Markers

```python
@pytest.mark.unit              # Unit tests
@pytest.mark.integration       # Integration tests
@pytest.mark.performance       # Performance tests
@pytest.mark.security          # Security tests
@pytest.mark.api               # API tests
@pytest.mark.database          # Database tests
@pytest.mark.websocket         # WebSocket tests
@pytest.mark.error_handling    # Error scenario tests
@pytest.mark.slow              # Slow tests (skipped in CI)
@pytest.mark.flaky             # Flaky tests
@pytest.mark.compliance        # Compliance tests
@pytest.mark.financial         # Financial analysis tests
```

### Running Specific Categories

```bash
# Only unit tests
pytest backend/tests/ -m unit

# Only integration tests
pytest backend/tests/ -m integration

# Security tests
pytest backend/tests/ -m security

# Skip slow tests
pytest backend/tests/ -m "not slow"

# API or WebSocket tests
pytest backend/tests/ -m "api or websocket"
```

---

## Mocking Best Practices

### Mock External Dependencies

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mocked_external_api(async_client):
    """Test with mocked external API."""
    with patch('backend.api.external.fetch_prices') as mock_fetch:
        # Mock the external API
        mock_fetch.return_value = {
            'AAPL': 150.00,
            'GOOGL': 2800.00
        }

        # Call endpoint that uses external API
        response = await async_client.get("/api/prices")
        assert response.status_code == 200

        # Verify mock was called
        mock_fetch.assert_called_once()
```

### Mock Async Functions

```python
@pytest.mark.asyncio
async def test_async_mock():
    """Test with async mock."""
    mock_fn = AsyncMock(return_value="mocked_result")

    result = await mock_fn()
    assert result == "mocked_result"
    mock_fn.assert_called_once()
```

### Patch Context Manager

```python
@pytest.mark.asyncio
async def test_patched_database(async_client):
    """Test with patched database."""
    mock_session = AsyncMock()
    mock_session.execute.return_value.scalars.return_value.all.return_value = []

    with patch('backend.config.database.get_async_db_session', return_value=mock_session):
        response = await async_client.get("/api/users")
        assert response.status_code == 200
```

---

## Fixtures

### Creating Custom Fixtures

```python
@pytest.fixture
async def authenticated_user(db_session, test_user):
    """Fixture providing authenticated user with JWT token."""
    await db_session.add(test_user)
    await db_session.commit()

    token = create_access_token({"sub": str(test_user.id)})
    return test_user, token

@pytest.fixture
async def sample_portfolio(db_session, test_user):
    """Fixture providing user with sample portfolio."""
    # Create user
    user = User(username="test", email="test@example.com")
    db_session.add(user)

    # Create portfolio items
    for symbol in ["AAPL", "GOOGL", "MSFT"]:
        stock = Stock(user_id=user.id, symbol=symbol, quantity=10)
        db_session.add(stock)

    await db_session.commit()
    return user, [stock for stock in await db_session.query(Stock)]
```

### Using Fixtures

```python
async def test_with_sample_portfolio(sample_portfolio):
    """Test using sample portfolio fixture."""
    user, stocks = sample_portfolio
    assert len(stocks) == 3
    assert stocks[0].symbol == "AAPL"
```

---

## Coverage Requirements

### Minimum Coverage: 85%

```bash
# Check coverage
pytest backend/tests/ --cov=backend --cov-report=html

# View HTML report
open htmlcov/index.html

# Coverage by file
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

### Improving Coverage

1. **Identify uncovered lines**
   ```bash
   pytest backend/tests/ --cov=backend --cov-report=term-missing | grep -A5 "0 lines"
   ```

2. **Add tests for missing coverage**
   - Focus on error paths
   - Cover edge cases
   - Test validation logic

3. **Example of high-coverage test**
   ```python
   def test_validate_stock_quantity():
       """Test stock quantity validation."""
       # Happy path
       assert validate_quantity(100) is True

       # Edge cases
       assert validate_quantity(0) is False
       assert validate_quantity(-1) is False
       assert validate_quantity(999999) is True

       # Invalid types
       with pytest.raises(TypeError):
           validate_quantity("100")
   ```

---

## Debugging Tests

### Print Debugging

```bash
# Show print output
pytest backend/tests/test_file.py::test_name -v -s

# With logging
pytest backend/tests/ -v --log-cli-level=DEBUG
```

### Using pdb

```bash
# Drop to debugger on failure
pytest backend/tests/ --pdb

# Drop to debugger on first failure
pytest backend/tests/ --pdb -x

# Drop to debugger when test passes (for verification)
pytest backend/tests/ --pdb-trace
```

### Inspecting Test State

```python
def test_with_debugging():
    """Test with debugging info."""
    import pdb

    result = perform_operation()

    # Drop to debugger here
    pdb.set_trace()

    # Now can inspect 'result' variable
    assert result == expected
```

### Verbose Output

```bash
# Show test setup/teardown
pytest backend/tests/ --setup-show

# Show local variables on failure
pytest backend/tests/ -l

# Full traceback
pytest backend/tests/ --tb=long

# Short traceback
pytest backend/tests/ --tb=short
```

---

## Performance Testing

### Timing Tests

```bash
# Show slowest 20 tests
pytest backend/tests/ --durations=20

# Show all test durations
pytest backend/tests/ --durations=0

# Fail if test takes >10 seconds
pytest backend/tests/ --timeout=10
```

### Load Testing

```python
@pytest.mark.performance
async def test_concurrent_users(async_client):
    """Test with 100 concurrent users."""
    import asyncio

    async def make_request():
        return await async_client.get("/api/endpoint")

    # Make 100 concurrent requests
    tasks = [make_request() for _ in range(100)]
    results = await asyncio.gather(*tasks)

    # Verify all succeeded
    assert all(r.status_code == 200 for r in results)
```

---

## Common Issues & Solutions

### Issue: Tests Pass Locally but Fail in CI

**Solution**:
- Check for hardcoded paths (use fixtures instead)
- Check for timezone issues (use UTC)
- Check for timing issues (increase timeouts)
- Check for environment variables (set in CI config)

### Issue: Flaky WebSocket Tests

**Solution**:
- Add explicit wait times instead of relying on delays
- Use explicit sync points (events, conditions)
- Increase timeout values
- Add retry logic in tests

### Issue: Database Lock Errors

**Solution**:
- Use `await session.rollback()` in fixture cleanup
- Avoid cross-session queries
- Use function-scoped sessions
- Add transaction isolation level

### Issue: Mock Return Value Not Used

**Solution**:
```python
# Wrong
mock.return_value = value
# Correct for async
mock.return_value = value
result = await mock()  # Now uses return_value

# Or use AsyncMock
mock = AsyncMock(return_value=value)
```

### Issue: Fixture Dependency Chain

**Solution**:
- Keep fixtures simple (single responsibility)
- Use clear naming (test_* for inputs, *_data for outputs)
- Document dependencies in docstring
- Consider merging if >2 dependencies

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest backend/tests/ --cov=backend --cov-report=xml
      - uses: codecov/codecov-action@v2
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
pytest backend/tests/ -x -q
exit $?
```

---

## Best Practices Checklist

- [ ] Tests are isolated (no shared state)
- [ ] Tests are fast (<1 second each)
- [ ] Tests are clear (self-documenting)
- [ ] Tests follow Arrange-Act-Assert pattern
- [ ] Tests use fixtures for setup
- [ ] Tests have descriptive names
- [ ] Tests check one thing (one assertion or related assertions)
- [ ] Tests are repeatable (no randomness)
- [ ] Tests handle async properly
- [ ] Tests clean up resources (fixtures)
- [ ] Tests use mocks for external dependencies
- [ ] Coverage is >85%
- [ ] No skip without reason
- [ ] No flaky tests
- [ ] Documented test purposes

---

## Further Reading

- **pytest documentation**: https://docs.pytest.org/
- **asyncio documentation**: https://docs.python.org/3/library/asyncio.html
- **FastAPI testing**: https://fastapi.tiangolo.com/advanced/testing-websockets/
- **Testing Best Practices**: See `.claude/rules/testing.md`

---

## Questions?

See the investigation guide: `TEST_FAILURE_ANALYSIS.md`
See the baseline report: `TEST_BASELINE_REPORT.md`
