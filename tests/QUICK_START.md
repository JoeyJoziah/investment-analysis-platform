# Quick Start - E2E and Integration Tests

## Installation

### Frontend Dependencies
```bash
cd frontend/web
npm install  # Includes Playwright
```

### Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
# Ensure pytest, pytest-asyncio, websockets are installed
```

## Running Tests

### Quick Commands

#### All Frontend E2E Tests
```bash
cd frontend/web
npm run test:e2e
```

#### All Backend Integration Tests
```bash
cd backend
pytest tests/test_websocket_integration.py tests/test_error_scenarios.py -v
```

#### Specific Test File

**Frontend Auth Tests**
```bash
cd frontend/web
npx playwright test tests/e2e/auth.spec.ts
```

**Frontend Portfolio Tests**
```bash
cd frontend/web
npx playwright test tests/e2e/portfolio.spec.ts
```

**Backend WebSocket Tests**
```bash
cd backend
pytest tests/test_websocket_integration.py -v
```

**Backend Error Scenario Tests**
```bash
cd backend
pytest tests/test_error_scenarios.py -v
```

#### Interactive Testing

**Playwright Test UI**
```bash
cd frontend/web
npm run test:e2e:ui
```

**Playwright Headed Mode (see browser)**
```bash
cd frontend/web
npm run test:e2e:headed
```

**Playwright Debug Mode**
```bash
cd frontend/web
npm run test:e2e:debug
```

#### With Filters

**Run single test**
```bash
# Frontend
npx playwright test -g "should successfully login"

# Backend
pytest tests/test_websocket_integration.py::TestWebSocketConnection::test_websocket_connection_succeeds -v
```

**Run test class**
```bash
# Backend
pytest tests/test_websocket_integration.py::TestPriceSubscription -v
```

## What Each Test Suite Covers

### Authentication (19 tests)
- User registration with validation
- Login/logout flows
- JWT token management
- Protected route access
- Token refresh
- Error handling

**Run**: `npx playwright test tests/e2e/auth.spec.ts`

### Portfolio Management (20 tests)
- Add stocks with validation
- View performance metrics
- Real-time price updates
- Remove positions
- Transaction history
- Portfolio analysis

**Run**: `npx playwright test tests/e2e/portfolio.spec.ts`

### WebSocket Integration (19 tests)
- Connection establishment
- Authentication
- Price subscriptions
- Real-time updates (<2s latency)
- Reconnection
- Error handling

**Run**: `pytest tests/test_websocket_integration.py -v`

### Error Scenarios (25 tests)
- Rate limiting (6 tests)
- Database recovery (5 tests)
- Circuit breaker (6 tests)
- Graceful degradation (6 tests)
- Concurrency (2 tests)

**Run**: `pytest tests/test_error_scenarios.py -v`

## Prerequisites

### Frontend
- Node.js 18+
- npm or yarn
- Chromium/Firefox/Safari (auto-installed by Playwright)

### Backend
- Python 3.9+
- FastAPI
- pytest and pytest-asyncio
- sqlalchemy

## Common Issues & Solutions

### Issue: "Chrome not found"
```bash
# Playwright auto-installs browsers, but if needed:
npx playwright install
```

### Issue: "Backend not responding"
```bash
# Start backend in separate terminal
cd backend
python -m uvicorn backend.api.main:app --reload --port 8000
```

### Issue: "Tests timeout"
```bash
# Increase timeout for slow environments
npx playwright test --config playwright.config.ts --timeout 30000
```

### Issue: "Database locked" (Python tests)
```bash
# Remove test database and retry
rm test.db
pytest tests/test_websocket_integration.py -v
```

### Issue: "Port 8000 in use"
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

## Test Reports

### Frontend
- HTML Report: `frontend/web/playwright-report/index.html`
- JUnit XML: `frontend/web/test-results/junit.xml`
- JSON: `frontend/web/test-results/results.json`

### Backend
Generate coverage report:
```bash
pytest tests/test_websocket_integration.py tests/test_error_scenarios.py --cov=backend --cov-report=html
# Report: htmlcov/index.html
```

## Performance Expectations

### Frontend E2E Tests
- Total time: 2-5 minutes
- Per test: 10-30 seconds
- Parallel execution: 4 workers

### Backend Tests
- Total time: 1-2 minutes
- WebSocket tests: ~30 seconds
- Error scenario tests: ~30 seconds

## Continuous Integration

### GitHub Actions Example
```yaml
- name: Run Frontend E2E Tests
  run: |
    cd frontend/web
    npm install
    npm run test:e2e

- name: Run Backend Integration Tests
  run: |
    cd backend
    pip install -r requirements.txt
    pytest tests/test_websocket_integration.py tests/test_error_scenarios.py -v
```

## Next Steps After Tests Pass

1. Review test reports in `playwright-report/` or `htmlcov/`
2. Check for any flaky tests or timeouts
3. Integrate into CI/CD pipeline
4. Set up automated test runs
5. Monitor test performance over time

## Documentation Files

- **Full Reference**: `tests/E2E_AND_INTEGRATION_TESTS.md`
- **Test Summary**: `tests/TEST_SUMMARY.md`
- **This File**: `tests/QUICK_START.md`

## Support

For issues or questions:
1. Check test output for specific error messages
2. Run with `-v` flag for verbose output
3. Use `--debug` flag for Playwright tests
4. Check test files for comments explaining behavior
