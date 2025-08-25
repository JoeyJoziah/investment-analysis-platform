# Integration Tests for Investment Analysis Platform

This directory contains comprehensive integration tests for the Investment Analysis Platform, covering all critical paths and system components.

## Test Structure

### Test Categories

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

## Contributing

When adding new integration tests:

1. Follow existing test patterns and structure
2. Add appropriate markers and documentation
3. Include both positive and negative test cases
4. Verify tests pass in all environments
5. Update test documentation as needed
6. Ensure adequate test coverage
7. Consider performance implications

For questions or issues, please refer to the main project documentation or create an issue in the project repository.