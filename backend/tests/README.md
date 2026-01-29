# Integration Tests for Investment Analysis Platform

This directory contains comprehensive integration tests for the Investment Analysis Platform, covering all critical paths and system components.

**Complete Guide**: See [TEST_INFRASTRUCTURE_GUIDE.md](./TEST_INFRASTRUCTURE_GUIDE.md) for detailed documentation of test infrastructure, fixtures, and patterns.

## Quick Start

### Run All Tests

```bash
# Run all tests with coverage
pytest backend/tests/ -v

# Run specific test file
pytest backend/tests/test_thesis_api.py -v

# Run tests matching pattern
pytest backend/tests/ -k "test_create" -v

# Run with coverage report
pytest backend/tests/ --cov=backend --cov-report=html
```

### Test Organization

All tests follow a consistent pattern using the ApiResponse wrapper validation helpers defined in `conftest.py`:

```python
# Good: Use helper function to validate and extract data
async def test_create_thesis(async_client, auth_headers):
    response = await async_client.post(
        "/api/theses",
        json={"title": "AI Investments", "description": "..."},
        headers=auth_headers
    )

    # Validates response structure and returns unwrapped data
    data = assert_success_response(response, expected_status=201)
    assert data["title"] == "AI Investments"

# Error handling
async def test_thesis_not_found(async_client):
    response = await async_client.get("/api/theses/999")

    # Validates error structure
    error_data = assert_api_error_response(
        response,
        expected_status=404,
        expected_error_substring="not found"
    )
```

### Coverage Requirements

- **Minimum line coverage**: 85%
- **Critical path coverage**: 95%
- **Security code coverage**: 100%
- **Error handling coverage**: 90%

Coverage threshold is enforced by pytest (see `pytest.ini`):
```bash
# Will fail if coverage drops below 85%
pytest backend/tests/ --cov=backend --cov-fail-under=85
```

## Test Infrastructure

### Fixture System

The test suite uses a comprehensive fixture system defined in `conftest.py` and `async_fixtures.py`:

**Key Async Fixtures:**
- `test_db_engine` - Session-scoped database engine
- `test_db_session_factory` - Session-scoped session factory
- `db_session` - Function-scoped database session with automatic rollback
- `async_client` / `client` - HTTP client for API testing
- `test_user`, `auth_token`, `auth_headers` - Authentication fixtures
- `mock_cache`, `mock_external_apis` - Mock dependencies

**See [TEST_INFRASTRUCTURE_GUIDE.md](./TEST_INFRASTRUCTURE_GUIDE.md#fixture-usage-guide) for complete fixture documentation.**

### ApiResponse Validation Helpers

All API responses follow a standardized wrapper structure. Two helper functions validate responses:

**For successful responses:**
```python
data = assert_success_response(response, expected_status=200)
# Validates: status code, success=true, "data" field exists
# Returns: unwrapped data from response["data"]
```

**For error responses:**
```python
error_data = assert_api_error_response(response, expected_status=404, expected_error_substring="not found")
# Validates: status code, success=false, optional substring match
# Returns: full response JSON
```

**See [TEST_INFRASTRUCTURE_GUIDE.md#validation-helper-functions](./TEST_INFRASTRUCTURE_GUIDE.md#validation-helper-functions) for complete documentation.**

### Pytest Configuration

Configuration in `pytest.ini`:
- **Async mode**: `asyncio_mode = strict` - Proper event loop management
- **Test discovery**: Automatic detection of test files and functions
- **Markers**: 10+ custom markers for organizing tests
- **Coverage**: 85% minimum, with detailed reporting
- **Asyncio fixture scope**: Function-level by default

**See [TEST_INFRASTRUCTURE_GUIDE.md#pytest-asyncio-configuration](./TEST_INFRASTRUCTURE_GUIDE.md#pytest-asyncio-configuration) for configuration details.**

## Test Structure

### Test Categories

0. **N+1 Query Fix Tests** (`test_n1_query_fix.py`) - NEW
   - Batch query method validation
   - Query count reduction verification
   - Performance benchmarking
   - Integration with recommendations engine
   - Edge case handling

1. **API Integration Tests** (`test_api_integration.py`)
   - Comprehensive API endpoint testing
   - Authentication and authorization
   - Request/response validation
   - Error handling and edge cases
   - Performance under load
   - Caching integration

2. **Data Pipeline Integration Tests** (`test_data_pipeline_integration.py`)
   - End-to-end data ingestion pipeline
   - External API integration (Alpha Vantage, Finnhub, Polygon)
   - Rate limiting and error handling
   - Cache integration and invalidation
   - Real-time data processing
   - Performance and scalability

3. **WebSocket Integration Tests** (`test_websocket_integration.py`)
   - Real-time connection management
   - Message broadcasting and subscriptions
   - Connection lifecycle and recovery
   - Authentication and authorization
   - Error handling and resilience
   - Load testing and performance

4. **Security Integration Tests** (`test_security_integration.py`)
   - OAuth2 authentication flow
   - JWT token management
   - Rate limiting and abuse prevention
   - SQL injection prevention
   - XSS and CSRF protection
   - Data privacy and encryption
   - Authorization and access control

5. **Database Integration Tests** (`test_database_integration.py`)
   - CRUD operations across all repositories
   - Transaction integrity and rollback
   - Concurrent operations and deadlock handling
   - Performance and indexing
   - Data validation and constraints
   - Connection pooling and recovery

6. **Resilience Integration Tests** (`test_resilience_integration.py`)
   - Circuit breaker patterns
   - Retry logic and backoff strategies
   - Fallback mechanisms
   - System recovery procedures
   - Cascade failure prevention
   - Disaster recovery testing

### Test Fixtures and Utilities

- **`fixtures/integration_test_fixtures.py`**: Comprehensive test fixtures and utilities
  - Realistic test data generators
  - Mock factories for external dependencies
  - Database and cache mocking utilities
  - Performance measurement tools
  - Assertion helpers

- **`conftest.py`**: Global test configuration
  - Session-scoped fixtures
  - Environment setup
  - Mock configurations
  - Custom assertions
  - Performance thresholds

## Running Tests

### Quick Start

```bash
# Run all integration tests
python run_integration_tests.py

# Run specific category
python run_integration_tests.py --categories api database

# Run with coverage
python run_integration_tests.py --suite full

# Run smoke tests
python run_integration_tests.py --suite smoke
```

### Test Suites

#### Smoke Tests
Quick validation of core functionality:
```bash
python run_integration_tests.py --suite smoke
```

#### Regression Tests  
Comprehensive testing for releases:
```bash
python run_integration_tests.py --suite regression
```

#### Full Test Suite
Complete integration testing:
```bash
python run_integration_tests.py --suite full
```

#### N+1 Query Fix Tests
Tests for CRITICAL-3 optimization:
```bash
# Run unit tests
pytest backend/tests/test_n1_query_fix.py -v

# Run performance benchmark
python -m backend.tests.benchmark_n1_query_fix
```

#### Security Tests
Security-focused testing:
```bash
python run_integration_tests.py --suite security
```

#### Performance Tests
Performance and load testing:
```bash
python run_integration_tests.py --suite performance
```

### Custom Test Execution

```bash
# Run specific markers
python run_integration_tests.py --markers "not slow"

# Run with parallel execution
python run_integration_tests.py --parallel

# Fail fast on first error
python run_integration_tests.py --fail-fast

# Skip coverage reporting
python run_integration_tests.py --no-coverage
```

### Using pytest directly

```bash
# Run all integration tests
pytest backend/tests/ -m integration -v

# Run specific test file
pytest backend/tests/test_api_integration.py -v

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Run with markers
pytest backend/tests/ -m "integration and not slow" -v
```

## Test Environments

### Test Environment
- In-memory SQLite database
- Mock external APIs
- Minimal dependencies
- Fast execution

```bash
python run_integration_tests.py --environment test
```

### Integration Environment  
- Real PostgreSQL database
- Real Redis cache
- Some external API calls
- More realistic conditions

```bash
python run_integration_tests.py --environment integration
```

### CI/CD Environment
- Optimized for CI/CD pipelines
- Skip slow tests
- Minimal external dependencies
- Error-level logging only

```bash
python run_integration_tests.py --environment ci
```

## Test Configuration

### Environment Variables

Required for testing:
```bash
# Database
DATABASE_URL=postgresql+asyncpg://test_user:test_pass@localhost/test_db
TEST_DATABASE_URL=postgresql+asyncpg://test_user:test_pass@localhost/test_db

# Cache
REDIS_URL=redis://localhost:6379/1
TEST_REDIS_URL=redis://localhost:6379/1

# Security
SECRET_KEY=your-test-secret-key

# API Keys (for integration environment)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key
POLYGON_API_KEY=your_polygon_key
NEWS_API_KEY=your_news_api_key

# Test Control
SKIP_EXTERNAL_API_TESTS=false
SKIP_SLOW_TESTS=false
```

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.database` - Database tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.external_api` - Tests requiring external APIs
- `@pytest.mark.n1_query_fix` - N+1 query pattern fix tests

### Performance Thresholds

Default performance expectations:
- API response: < 2.0 seconds
- Database query: < 1.0 seconds  
- Cache operation: < 0.1 seconds
- WebSocket message: < 0.5 seconds
- Bulk operation: < 10.0 seconds

## Test Data

### Realistic Test Data
Tests use realistic data generated by Faker library:
- User profiles with proper demographics
- Stock data with realistic price movements
- Portfolio data with proper allocations
- Transaction histories with proper patterns

### Mock Data
External dependencies are mocked with realistic responses:
- API responses match actual API formats
- Database responses simulate real query results
- Cache responses simulate Redis behavior
- WebSocket messages simulate real-time updates

## Coverage Requirements

- Minimum line coverage: 85%
- Critical path coverage: 95%
- Security code coverage: 100%
- Error handling coverage: 90%

Coverage reports are generated in multiple formats:
- Terminal output for immediate feedback
- HTML report for detailed analysis
- XML report for CI/CD integration
- JSON report for programmatic analysis

## CI/CD Integration

### GitHub Actions Integration

```yaml
- name: Run Integration Tests
  run: |
    python run_integration_tests.py --environment ci --suite regression
    
- name: Upload Coverage Reports
  uses: codecov/codecov-action@v3
  with:
    file: ./test_reports/coverage.xml
```

### Test Reports

All test reports are saved to `test_reports/` directory:
- `test_report_YYYYMMDD_HHMMSS.html` - Detailed HTML report
- `test_report_YYYYMMDD_HHMMSS.json` - JSON report for automation
- `junit_YYYYMMDD_HHMMSS.xml` - JUnit XML for CI/CD
- `coverage.xml` - Coverage report in XML format
- `coverage_html/` - Detailed HTML coverage report
- `test_summary_YYYYMMDD_HHMMSS.json` - High-level summary

## Debugging Tests

### Running Individual Tests

```bash
# Run single test method
pytest backend/tests/test_api_integration.py::TestAPIEndpointsIntegration::test_health_endpoint_integration -v -s

# Run with debugger
pytest backend/tests/test_api_integration.py::TestAPIEndpointsIntegration::test_health_endpoint_integration -v -s --pdb
```

### Debugging Tips

1. **Use verbose output**: Add `-v` flag for detailed test names
2. **Capture output**: Add `-s` flag to see print statements
3. **Stop on first failure**: Add `--maxfail=1` to stop immediately
4. **Show local variables**: Add `-l` flag to show locals in tracebacks
5. **Use debugger**: Add `--pdb` to drop into debugger on failures

### Logging Configuration

Tests use structured logging with different levels:
- ERROR: Critical failures only
- WARNING: Important issues
- INFO: General progress information
- DEBUG: Detailed execution information

Set log level via environment variable:
```bash
export LOG_LEVEL=DEBUG
pytest backend/tests/test_api_integration.py -v
```

## Best Practices

### Writing Integration Tests

1. **Test Real Scenarios**: Focus on realistic user workflows
2. **Test Error Conditions**: Include error handling and edge cases
3. **Test Performance**: Validate response times and throughput
4. **Test Security**: Verify authentication and authorization
5. **Test Resilience**: Include failure and recovery scenarios

### Test Organization

1. **Clear Test Names**: Use descriptive test method names
2. **Proper Fixtures**: Use appropriate fixtures for setup/teardown
3. **Isolation**: Ensure tests don't depend on each other
4. **Documentation**: Include docstrings explaining test purpose
5. **Assertions**: Use meaningful assertions with clear messages

### Maintenance

1. **Regular Updates**: Keep tests updated with code changes
2. **Performance Monitoring**: Track test execution times
3. **Coverage Analysis**: Regular coverage analysis and improvement
4. **Flaky Test Management**: Identify and fix unreliable tests
5. **Documentation**: Keep test documentation current

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Ensure test database is running
   - Check connection parameters
   - Verify user permissions

2. **Redis Connection Errors**
   - Ensure Redis is running
   - Check Redis URL configuration
   - Verify Redis is accessible

3. **External API Errors**
   - Check API key validity
   - Verify rate limits not exceeded
   - Ensure internet connectivity

4. **Import Errors**
   - Verify all dependencies installed
   - Check PYTHONPATH configuration
   - Ensure virtual environment activated

5. **Permission Errors**
   - Check file system permissions
   - Verify test user has required access
   - Check database user permissions

### Getting Help

1. Check the test logs in `test_reports/`
2. Run tests with verbose output (`-v`)
3. Check the main application logs
4. Verify environment configuration
5. Consult the main project documentation

## Adding New Tests

### Step 1: Choose Test File

- **API endpoints**: Add to `test_api_integration.py` or create `test_<feature>_api.py`
- **Database operations**: Add to `test_database_integration.py`
- **Business logic**: Create `test_<feature>.py`
- **Security**: Add to `test_security_integration.py` or `test_security_compliance.py`
- **Performance**: Add to `test_performance_load.py`

### Step 2: Import Required Fixtures

```python
import pytest
import pytest_asyncio
from conftest import assert_success_response, assert_api_error_response

@pytest.mark.api
async def test_my_feature(async_client, auth_headers, db_session):
    """Test description."""
```

### Step 3: Use ApiResponse Validation

```python
# Always wrap response validation
response = await async_client.get("/api/endpoint")
data = assert_success_response(response)

# For errors
response = await async_client.post("/api/endpoint", json={})
error_data = assert_api_error_response(response, expected_status=422)
```

### Step 4: Proper Test Structure

```python
@pytest.mark.api
@pytest.mark.integration
async def test_create_and_retrieve(async_client, auth_headers):
    """Test complete workflow: create and retrieve."""
    # Setup
    payload = {"title": "Test", "description": "..."}

    # Action 1: Create
    create_response = await async_client.post(
        "/api/resource",
        json=payload,
        headers=auth_headers
    )
    created = assert_success_response(create_response, expected_status=201)
    resource_id = created["id"]

    # Action 2: Retrieve
    get_response = await async_client.get(
        f"/api/resource/{resource_id}",
        headers=auth_headers
    )
    retrieved = assert_success_response(get_response)

    # Assertions
    assert retrieved["id"] == resource_id
    assert retrieved["title"] == payload["title"]
```

### Step 5: Add Markers and Documentation

```python
@pytest.mark.api                    # Type of test
@pytest.mark.integration            # Test scope
@pytest.mark.security              # (Optional) Special category
async def test_endpoint(async_client):
    """
    Test endpoint behavior.

    This test verifies:
    - Response structure is valid
    - Data is returned correctly
    - Error handling works

    See: https://jira.company.com/browse/PROJ-123
    """
```

### Step 6: Include Both Success and Error Cases

```python
# Good: Success case
@pytest.mark.api
async def test_create_thesis(async_client, auth_headers):
    response = await async_client.post(
        "/api/theses",
        json={"title": "Test", "description": "..."},
        headers=auth_headers
    )
    data = assert_success_response(response, expected_status=201)

# Also add: Error case
@pytest.mark.api
async def test_create_thesis_missing_title(async_client, auth_headers):
    response = await async_client.post(
        "/api/theses",
        json={"description": "..."},  # Missing title
        headers=auth_headers
    )
    error = assert_api_error_response(response, expected_status=422)
```

### Step 7: Run and Verify

```bash
# Run your new tests
pytest backend/tests/test_my_feature.py -v

# Check coverage
pytest backend/tests/test_my_feature.py --cov=backend.api.routers

# Full test suite
pytest backend/tests/ -v
```

## Contributing

When adding new integration tests:

1. Follow existing test patterns (use `assert_success_response()` and `assert_api_error_response()`)
2. Add appropriate markers (`@pytest.mark.api`, `@pytest.mark.database`, etc.)
3. Include both positive (success) and negative (error) test cases
4. Use proper fixtures (don't create clients/sessions manually)
5. Verify tests pass in all environments
6. Update test documentation as needed
7. Ensure adequate test coverage (80%+ minimum)
8. Consider performance implications
9. Add docstrings explaining test purpose

### Test Pattern Examples

See [TEST_INFRASTRUCTURE_GUIDE.md#common-testing-patterns](./TEST_INFRASTRUCTURE_GUIDE.md#common-testing-patterns) for complete pattern examples:
- Basic API tests
- Authenticated API tests
- Error handling tests
- Database tests
- End-to-end flow tests
- Performance tests
- Tests with mocked external APIs

### Troubleshooting

Common test issues and solutions are documented in [TEST_INFRASTRUCTURE_GUIDE.md#troubleshooting-guide](./TEST_INFRASTRUCTURE_GUIDE.md#troubleshooting-guide).

## Documentation

**Complete test infrastructure documentation**: [TEST_INFRASTRUCTURE_GUIDE.md](./TEST_INFRASTRUCTURE_GUIDE.md)

Topics covered:
- ApiResponse wrapper pattern and validation
- Helper functions: `assert_success_response()`, `assert_api_error_response()`
- Pytest-asyncio configuration and event loop management
- Fixture system and dependencies
- Common testing patterns (8 complete examples)
- Troubleshooting guide for 10 common issues
- Best practices for test writing

For questions or issues, please refer to the test infrastructure guide or create an issue in the project repository.