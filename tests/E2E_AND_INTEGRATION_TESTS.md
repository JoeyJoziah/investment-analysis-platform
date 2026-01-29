# E2E and Integration Tests - Phase 4.1

## Overview

This document describes the comprehensive E2E and integration tests for the Investment Analysis Platform frontend-backend integration.

## Test Suites

### 1. Frontend E2E Tests (Playwright)

#### Auth Tests (`frontend/web/tests/e2e/auth.spec.ts`)

**User Registration Flows (5 tests)**
- Register with valid credentials
- Reject duplicate email registration
- Validate password strength requirements
- Require password confirmation match
- Display validation errors

**User Login Flows (5 tests)**
- Successfully login with valid credentials
- Reject invalid credentials
- Validate empty field requirements
- Provide forgot password link
- Redirect to dashboard on success

**JWT Token Verification (3 tests)**
- Include valid JWT in Authorization header
- Refresh expired tokens automatically
- Decode JWT and extract user claims

**Protected Route Access (4 tests)**
- Redirect unauthenticated users to login
- Allow authenticated users to protected routes
- Display user profile when authenticated
- Clear token and redirect on 401 response

**Logout (2 tests)**
- Logout user and clear token
- Invalidate refresh token on logout

**Total: 19 authentication tests**

#### Portfolio Tests (`frontend/web/tests/e2e/portfolio.spec.ts`)

**Add Stock to Portfolio (5 tests)**
- Successfully add a stock position
- Validate required fields
- Reject invalid price
- Reject invalid quantity
- Handle duplicate ticker correctly

**View Performance Metrics (4 tests)**
- Display portfolio performance summary
- Show performance tab with charts
- Display allocation breakdown
- Show position-level gains/losses

**Real-time Price Updates (3 tests)**
- Receive WebSocket price updates
- Update position values based on price changes
- Handle WebSocket disconnection gracefully

**Remove Position (4 tests)**
- Remove a position from portfolio
- Show confirmation dialog before delete
- Keep position when cancel is clicked
- Handle deletion of last position

**Transaction History (2 tests)**
- Display transaction history
- Show transaction details

**Portfolio Analysis (2 tests)**
- Display analysis tab
- Show correlation analysis

**Total: 20 portfolio management tests**

### 2. Backend WebSocket Integration Tests (`backend/tests/test_websocket_integration.py`)

#### Connection Tests (5 tests)
- WebSocket connection succeeds with valid token
- Requires authentication
- Rejects invalid token
- Rejects expired token
- Rejects inactive user connection

#### Price Subscription (5 tests)
- Subscribe to price updates for specific tickers
- Unsubscribe from price updates
- Multiple subscriptions to different symbols
- Invalid subscription symbol handling
- Error handling for invalid symbols

#### Price Update Delivery (3 tests)
- Price update message format validation
- Price update latency verification (<2 seconds)
- Batch price updates handling

#### WebSocket Reconnection (3 tests)
- Reconnection with same token
- Preserve subscriptions on reconnect
- Connection cleanup on disconnect

#### Error Handling (3 tests)
- Invalid message format handling
- Malformed JSON handling
- Server error notification

**Total: 19 WebSocket integration tests**

### 3. Backend Error Scenario Tests (`backend/tests/test_error_scenarios.py`)

#### API Rate Limiting (6 tests)
- Track requests per user
- Reset limits after time period
- Return 429 status when exceeded
- Include Retry-After header in 429 response
- Different rate limits for user tiers
- Per-endpoint rate limiting

#### Database Connection Loss (5 tests)
- Handle connection error gracefully
- Handle database query timeouts
- Handle connection pool exhaustion
- Transaction rollback on error
- Recovery after connection loss

#### Circuit Breaker Pattern (6 tests)
- Initial CLOSED state
- Opens after threshold failures
- Rejects calls when OPEN
- Transitions to HALF_OPEN state
- Detects and handles slow responses
- Collects metrics
- Protects external API calls

#### Graceful Degradation (6 tests)
- Return cached data on DB error
- Missing external service fallback
- Partial response on service failure
- WebSocket graceful disconnect
- Authentication fallback
- Data validation prevents corruption

#### Concurrency (2 tests)
- Handle simultaneous portfolio updates
- Prevent duplicate transactions

**Total: 25 error scenario tests**

## Test Statistics

| Category | Count |
|----------|-------|
| Frontend Auth Tests | 19 |
| Frontend Portfolio Tests | 20 |
| Backend WebSocket Tests | 19 |
| Backend Error Scenario Tests | 25 |
| **Total Tests** | **83** |

## Running the Tests

### Frontend E2E Tests

```bash
# Install dependencies
cd frontend/web
npm install

# Run all E2E tests
npm run test:e2e

# Run with UI
npm run test:e2e:ui

# Run headed (see browser)
npm run test:e2e:headed

# Run single test file
npx playwright test tests/e2e/auth.spec.ts

# Run specific test
npx playwright test -g "should successfully login"

# Run with debugging
npm run test:e2e:debug
```

### Backend Integration Tests

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run WebSocket tests
pytest tests/test_websocket_integration.py -v

# Run error scenario tests
pytest tests/test_error_scenarios.py -v

# Run all tests
pytest tests/test_websocket_integration.py tests/test_error_scenarios.py -v

# Run with coverage
pytest tests/test_websocket_integration.py tests/test_error_scenarios.py --cov=backend --cov-report=html

# Run specific test class
pytest tests/test_websocket_integration.py::TestWebSocketConnection -v

# Run specific test
pytest tests/test_websocket_integration.py::TestWebSocketConnection::test_websocket_connection_succeeds -v
```

### Run All Tests

```bash
# Run all unit, integration, and E2E tests
npm run test:all
```

## Test Configuration

### Frontend (Playwright)

**File**: `frontend/web/playwright.config.ts`

- **Base URL**: http://localhost:5173
- **API URL**: http://localhost:8000
- **Browsers**: Chromium, Firefox, WebKit
- **Mobile**: Pixel 5, iPhone 12
- **Retries**: 2 in CI, 0 locally
- **Timeout**: 10 seconds per action
- **Screenshots**: On failure
- **Video**: Retained on failure
- **Reports**: HTML, JUnit XML, JSON

### Backend

**Test Database**: SQLite (in-memory or test.db)
**Async Support**: pytest-asyncio
**Mocking**: unittest.mock, patch
**Fixtures**: pytest fixtures with dependency injection

## Test Data

### Frontend Test Users

```
Email: portfolio-test@example.com
Username: portfolio-test
Password: PortfolioTest123!

Email: existing@example.com
Username: existinguser
Password: ExistingPass123!@#
```

### Backend Test Fixtures

- `test_user_data`: Default test user
- `authenticated_client`: TestClient with auth headers
- `db_session`: Database session for tests
- `auth_headers`: Authorization headers

## Critical User Flows Covered

### 1. Authentication Flow
```
Register → Login → Receive JWT → Access Protected Routes → Logout
```
- Token validation
- Token refresh
- Token expiration
- Session management

### 2. Portfolio Management
```
View Portfolio → Add Position → Monitor Updates → Remove Position
```
- Real-time price updates via WebSocket
- Position performance tracking
- Risk metrics calculation
- Portfolio allocation

### 3. Error Recovery
```
Connection Loss → Automatic Reconnection → Resume Operations
```
- Circuit breaker activation
- Graceful degradation
- Rate limit handling
- Data consistency

## Performance Requirements

### WebSocket Latency
- **Target**: < 2 seconds for price updates
- **Measurement**: From subscription to first update

### API Response Time
- **Target**: < 1 second for 95th percentile
- **Measurement**: From request to response

### Rate Limiting
- **Basic**: 100 calls/hour
- **Premium**: 1000 calls/hour
- **Admin**: 10000 calls/hour

## Acceptance Criteria

✓ 15+ E2E test scenarios pass
✓ All critical user flows covered
✓ Error handling verified
✓ WebSocket connectivity stable
✓ 83 total tests covering:
  - Authentication (19 tests)
  - Portfolio operations (20 tests)
  - Real-time updates (19 tests)
  - Error scenarios (25 tests)

## Known Limitations

### Frontend Tests
- Mock API responses for unit behavior
- Live WebSocket requires running backend
- Some latency tests approximate network behavior
- Mobile tests use emulation, not real devices

### Backend Tests
- Database tests use test database
- Rate limiting tests use in-memory counters
- Circuit breaker tests simulate failures
- External API calls are mocked

## Future Improvements

1. Add performance benchmarking
2. Load testing for concurrent users
3. Security penetration testing
4. Mobile-specific WebSocket tests
5. Cross-browser compatibility matrix
6. Accessibility testing (WCAG)
7. Visual regression testing
8. API contract testing

## Troubleshooting

### E2E Tests Fail

**Issue**: Tests timeout waiting for API response
```bash
# Increase timeout
npx playwright test --config playwright.config.ts --timeout 30000
```

**Issue**: Frontend not loading
```bash
# Start dev server in separate terminal
npm run dev

# Then run tests
npm run test:e2e
```

**Issue**: Backend not responding
```bash
# Start backend in separate terminal
python -m uvicorn backend.api.main:app --reload --port 8000

# Verify health endpoint
curl http://localhost:8000/api/health
```

### Backend Tests Fail

**Issue**: Database locked
```bash
# Remove test database
rm test.db

# Run tests again
pytest tests/test_websocket_integration.py -v
```

**Issue**: Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Run tests again
pytest tests/test_error_scenarios.py -v
```

## CI/CD Integration

### GitHub Actions

Tests run on:
- Pull requests to main/develop
- Commits to main
- Manual trigger via workflow_dispatch

### Test Reporting

- JUnit XML for CI integration
- HTML report for visual review
- JSON for programmatic access
- Screenshots/videos on failure

### Coverage Requirements

- Statements: >80%
- Branches: >75%
- Functions: >80%
- Lines: >80%

## Security Testing

### Covered
- JWT token validation
- Authentication bypass prevention
- Rate limit enforcement
- Input validation
- Error message sanitization

### Recommended
- SQL injection prevention
- XSS prevention
- CSRF protection
- API key security
- Encryption verification

## References

- [Playwright Documentation](https://playwright.dev/docs/intro)
- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/advanced/testing-websockets/)
- [WebSocket Testing](https://fastapi.tiangolo.com/advanced/testing-websockets/)
