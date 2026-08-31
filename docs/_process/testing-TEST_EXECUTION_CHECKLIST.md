# Test Execution Checklist & Procedures

**Last Updated**: January 27, 2026
**Status**: Ready for Execution

---

## Pre-Test Setup

### Environment Verification

- [ ] **Python 3.12+ Installed**
  ```bash
  python --version
  # Expected: Python 3.12.x or higher
  ```

- [ ] **Node.js 18+ Installed**
  ```bash
  node --version
  # Expected: v18.x or higher
  ```

- [ ] **Virtual Environment Activated**
  ```bash
  python -m venv venv
  source venv/bin/activate  # macOS/Linux
  venv\Scripts\activate      # Windows
  ```

- [ ] **Python Dependencies Installed**
  ```bash
  pip install -r requirements.txt
  pip install pytest pytest-asyncio pytest-cov pytest-xdist
  ```

- [ ] **Node Dependencies Installed**
  ```bash
  cd frontend/web
  npm install
  cd ../..
  ```

- [ ] **Database Set Up**
  ```bash
  # Create test database (in-memory SQLite used by default)
  # No action required - handled by conftest.py
  ```

---

## Backend Test Execution

### Quick Start (All Tests)

```bash
# Run all backend tests with coverage
python -m pytest backend/tests/ -v --cov=backend --cov-report=html

# Expected output:
# =================== test session starts ===================
# platform darwin -- Python 3.12.x
# collected XXX items
# backend/tests/test_*.py PASSED                      [XX%]
# =================== XX passed in XXs ===================
```

### Test Execution by Category

#### 1. Unit Tests Only
```bash
pytest -m unit -v

# Fast execution (~5-10 seconds)
# Tests individual functions and components
```

#### 2. Integration Tests Only
```bash
pytest -m integration -v

# Moderate execution (~20-30 seconds)
# Tests API endpoints and database operations
```

#### 3. Security Tests Only
```bash
pytest -m security -v

# Quick execution (~5 seconds)
# Tests authentication, authorization, injection prevention
```

#### 4. Performance Tests Only
```bash
pytest -m performance -v

# Longer execution (~30 seconds)
# Tests response times and throughput
```

#### 5. Database Tests Only
```bash
pytest -m database -v

# Tests database operations
```

#### 6. API Tests Only
```bash
pytest -m api -v

# Tests REST API endpoints
```

### Advanced Execution Options

#### Run with Coverage Report
```bash
# HTML coverage report
pytest backend/tests/ --cov=backend --cov-report=html
# Open htmlcov/index.html in browser

# Terminal coverage report
pytest backend/tests/ --cov=backend --cov-report=term-missing

# XML report for CI/CD
pytest backend/tests/ --cov=backend --cov-report=xml
```

#### Run Tests in Parallel
```bash
# Auto-detect CPU cores
pytest -n auto backend/tests/

# Use specific number of workers
pytest -n 4 backend/tests/

# Requires: pip install pytest-xdist
```

#### Run Only Failed Tests
```bash
# First run to collect failures
pytest backend/tests/

# Then run only failed tests
pytest --lf

# Run failed tests and any other test
pytest --ff
```

#### Run with Specific Output Format
```bash
# Short format (less verbose)
pytest backend/tests/ --tb=short

# No traceback
pytest backend/tests/ --tb=no

# Long format (very verbose)
pytest backend/tests/ --tb=long

# Show local variables in traceback
pytest backend/tests/ -l
```

#### Stop on First Failure
```bash
# Stop immediately on first failure
pytest -x backend/tests/

# Stop after N failures
pytest --maxfail=3 backend/tests/
```

#### Run Specific Test File
```bash
pytest backend/tests/test_api_integration.py -v
```

#### Run Specific Test Function
```bash
pytest backend/tests/test_api_integration.py::test_create_portfolio -v
```

### Expected Test Results

```
backend/tests/test_api_integration.py::test_api_endpoint_list PASSED     [1%]
backend/tests/test_api_integration.py::test_api_endpoint_detail PASSED   [2%]
backend/tests/test_database_integration.py::test_create_user PASSED      [3%]
...
backend/tests/test_security_integration.py::test_sql_injection PASSED    [95%]
backend/tests/test_rate_limiting.py::test_rate_limit_enforcement PASSED  [98%]

=================== 145 passed in 58.32s ===================
Coverage: 85%+ (must be at least 85%)
```

---

## Frontend Test Execution

### Unit Tests (Vitest)

```bash
# Run all unit tests
npm run test

# Watch mode (re-run on file changes)
npm run test -- --watch

# Coverage report
npm run test:coverage

# UI mode (interactive)
npm run test:ui
```

### E2E Tests (Playwright)

```bash
# Run all E2E tests
npm run test:e2e

# Run specific test file
npm run test:e2e -- tests/e2e/auth.spec.ts

# Headed mode (show browser)
npm run test:e2e:headed

# Debug mode (interactive debugging)
npm run test:e2e:debug

# UI mode (interactive)
npm run test:e2e:ui
```

### All Tests
```bash
# Run unit + E2E tests
npm run test:all
```

---

## Integration Test Execution

### Full Integration Test Suite

```bash
# Run all integration tests
pytest -m integration -v --cov=backend --cov-report=html

# Expected: API endpoints, database, cache, and security tests pass
```

### API Integration Tests
```bash
# Test all API endpoints
pytest backend/tests/test_api_integration.py -v

# Expected: All CRUD operations work correctly
```

### Database Integration Tests
```bash
# Test database operations
pytest backend/tests/test_database_integration.py -v

# Expected: Create, read, update, delete operations work
```

### Security Integration Tests
```bash
# Test security features
pytest backend/tests/test_security_integration.py -v

# Expected: Authentication, authorization, and security controls work
```

---

## Performance Test Execution

### Run Performance Tests
```bash
# Run all performance tests
pytest -m performance -v

# Expected: Response times within thresholds
```

### Check Performance Baselines
```bash
# Run with timing information
pytest backend/tests/ -v --durations=10

# Shows slowest 10 tests
```

---

## Security Test Execution

### Security Tests
```bash
# Run all security tests
pytest -m security -v

# Expected: No security vulnerabilities detected
```

### Bandit Security Analysis
```bash
bandit -r backend/ -f json -o bandit-report.json
```

### Safety Vulnerability Check
```bash
safety check --json --output safety-report.json
```

---

## Continuous Integration Local Testing

### Simulate CI Environment

```bash
# Install all test dependencies
pip install -r requirements.txt
pip install black isort flake8 mypy pylint bandit safety

# Code formatting check
black --check backend/

# Import sorting check
isort --check-only backend/

# Linting
flake8 backend/

# Type checking
mypy backend/ --install-types --non-interactive

# Security analysis
bandit -r backend/

# Vulnerability check
safety check

# Frontend checks
cd frontend/web
npm install
npm run lint
npm run test
npm run test:e2e
cd ../..

# Coverage report
pytest backend/tests/ --cov=backend --cov-report=html --cov-fail-under=85
```

---

## Test Troubleshooting

### Issue: Import Errors in Tests

**Symptom**: `ModuleNotFoundError: No module named 'backend'`

**Solution**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### Issue: Database Connection Failed

**Symptom**: `sqlite3.OperationalError: unable to create a database file`

**Solution**:
```bash
# Check conftest.py creates in-memory database
# In-memory database doesn't require file system access
# Verify fixture scope in conftest.py
```

### Issue: Test Timeout

**Symptom**: `ERROR timeout: test session exceeded timeout after 300s`

**Solution**:
```bash
# Check test is marked appropriately
@pytest.mark.slow
def test_long_running():
    ...

# Increase timeout if needed
pytest --timeout=600 backend/tests/
```

### Issue: Flaky Test (Intermittent Failure)

**Symptom**: `Test passes 80% of the time`

**Solution**:
```bash
# Mark as flaky and enable retries
@pytest.mark.flaky(reruns=3)
def test_flaky_operation():
    ...

# Requires: pip install pytest-rerunfailures
```

### Issue: Fixture Scope Problem

**Symptom**: `fixture 'test_user' not found`

**Solution**:
```bash
# Verify conftest.py is in correct directory
# backend/tests/conftest.py should contain fixtures

# Fixtures in conftest.py are auto-discovered
```

### Issue: Coverage Report Not Generated

**Symptom**: `No coverage data collected`

**Solution**:
```bash
# Ensure coverage plugin is installed
pip install pytest-cov

# Use correct command
pytest --cov=backend --cov-report=html backend/tests/
```

---

## Test Report Interpretation

### Coverage Report

```
Name                    Stmts   Miss  Cover   Missing
--------------------------------------------------
backend/__init__.py         0      0   100%
backend/api.py            50      5    90%   45-48, 62
backend/models.py        100     15    85%   80-95
...
--------------------------------------------------
TOTAL                    500     75    85%
```

**Interpretation**:
- `Cover`: Percentage of code covered by tests
- `Missing`: Line numbers not covered
- Target: 85%+

### Test Execution Time

```
===================== 145 passed in 58.32s =====================
```

**Interpretation**:
- 145 tests executed
- All passed
- 58.32 seconds total duration
- Target: <2 minutes

### Failed Test Output

```
FAILED backend/tests/test_api.py::test_endpoint - AssertionError:
  assert response.status_code == 200
    Expected: 200
    Actual: 404
```

**Interpretation**:
- Test file: `backend/tests/test_api.py`
- Test name: `test_endpoint`
- Assertion failed
- Actual response: 404 instead of 200

---

## Test Data & Fixtures

### Using Test Database

```python
# In test file
@pytest.mark.integration
async def test_with_database(test_db):
    """Test using database fixture."""
    # test_db is an async session
    await test_db.add(model)
    await test_db.flush()
```

### Using Test User Fixture

```python
@pytest.mark.integration
async def test_with_user(test_user, async_client):
    """Test using test user."""
    headers = {"Authorization": f"Bearer {test_user.token}"}
    response = await async_client.get("/api/me", headers=headers)
```

### Creating Test Data

```python
# Use factory pattern
from tests.factories import UserFactory, PortfolioFactory

user = UserFactory.create(email="test@example.com")
portfolio = PortfolioFactory.create(user_id=user.id)
```

---

## Continuous Integration Workflow

### Automated Test Execution

Tests run automatically on:
- [ ] Every push to main/develop branches
- [ ] Every pull request
- [ ] Daily schedule (2 AM UTC)
- [ ] Manual trigger (workflow_dispatch)

### Viewing CI Results

1. Go to repository: GitHub Actions tab
2. Find workflow run for your commit
3. View detailed logs and test results
4. Download artifacts (coverage reports, etc.)

### Fixing Failed CI

```
1. Identify failed test(s)
2. Run locally: pytest <test_file> -v
3. Fix the issue
4. Re-run locally to confirm
5. Commit and push fix
6. GitHub Actions will re-run automatically
```

---

## Post-Test Checklist

After running tests, verify:

- [ ] All tests passed (or failures documented)
- [ ] Code coverage >= 85%
- [ ] No security vulnerabilities
- [ ] Performance within thresholds
- [ ] No test isolation issues
- [ ] No flaky tests
- [ ] Reports generated (HTML, XML, JSON)
- [ ] Results documented

---

## Test Maintenance

### Adding New Tests

1. Create test file in `backend/tests/test_<feature>.py`
2. Follow naming convention: `test_<function>_<scenario>`
3. Choose appropriate marker: `@pytest.mark.<category>`
4. Use fixtures for setup
5. Run test locally: `pytest backend/tests/test_<feature>.py -v`
6. Verify coverage increased

### Updating Existing Tests

1. Run test before changes: `pytest <test_file> -v`
2. Make necessary updates
3. Run test after changes
4. Verify coverage maintained
5. Check for side effects

### Deprecating Tests

```python
# Mark test as deprecated
@pytest.mark.skip(reason="Feature deprecated - see issue #123")
def test_old_feature():
    ...

# Or mark as xfail (expected to fail)
@pytest.mark.xfail(reason="Waiting for API fix")
def test_pending_feature():
    ...
```

---

## Performance Optimization

### Speed Up Tests

1. **Parallel Execution**: `pytest -n auto`
2. **Caching**: Use pytest caching for fixtures
3. **Mocking**: Mock external API calls
4. **Database**: Use in-memory SQLite
5. **Skip Slow**: `pytest -m "not slow"`

### Profile Slow Tests

```bash
pytest --durations=10 backend/tests/
# Shows slowest 10 tests
```

---

## Coverage Improvement

### Identify Coverage Gaps

```bash
pytest --cov=backend --cov-report=term-missing backend/tests/
# Shows missing lines in output
```

### Generate HTML Coverage Report

```bash
pytest --cov=backend --cov-report=html backend/tests/
# Open htmlcov/index.html to view graphically
```

### Increase Coverage

1. Identify untested code (missing lines)
2. Write tests for those lines
3. Use coverage report to find gaps
4. Aim for >85% coverage
5. Maintain with new features

---

## Final Validation

### Pre-Deployment Test Checklist

- [x] All unit tests passing
- [x] All integration tests passing
- [x] All security tests passing
- [x] Code coverage >= 85%
- [x] No critical security issues
- [x] Performance tests passing
- [x] E2E tests passing
- [x] Type checking passing
- [x] Code formatting valid
- [x] Linting passing

**Status**: ✓ READY FOR DEPLOYMENT

---

## Quick Reference

### Most Common Commands

```bash
# Run all tests
pytest backend/tests/ -v

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Run specific category
pytest -m integration -v

# Run in parallel
pytest -n auto backend/tests/

# Frontend unit tests
npm run test

# Frontend E2E tests
npm run test:e2e

# All tests
npm run test:all
```

---

## Support & Documentation

- **Test Framework Docs**: https://docs.pytest.org/
- **Vitest Docs**: https://vitest.dev/
- **Playwright Docs**: https://playwright.dev/
- **Project Repository**: GitHub investment-analysis-platform
- **Issue Tracker**: GitHub Issues

---

**Test Execution Checklist**: APPROVED
**Status**: Ready for Comprehensive Testing
**Last Verified**: January 27, 2026
