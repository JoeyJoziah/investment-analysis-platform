# Quick Test Checklist - Investment Analysis Platform

## Pre-Test Setup (5 minutes)

```bash
# 1. Start all services
./start.sh dev

# 2. Wait for services to be healthy (check with)
docker-compose ps

# 3. Verify all services are running
# Expected: postgres, redis, elasticsearch, backend, frontend all "Up"
```

---

## Critical Path Tests (15 minutes)

### 1. Backend Health (2 minutes)

```bash
# Basic health check
curl http://localhost:8000/api/health
# Expected: {"status":"healthy","timestamp":"...","version":"1.0.0"}

# Readiness check
curl http://localhost:8000/api/health/readiness
# Expected: {"status":"ready","checks":{"database":true,"cache":true,"api":true}}

# Metrics
curl http://localhost:8000/api/health/metrics
# Expected: System metrics with CPU, memory, database pool stats
```

**Pass Criteria:**
- [ ] All health endpoints return 200
- [ ] Database check: true
- [ ] Cache check: true

---

### 2. Authentication Flow (3 minutes)

```bash
# Register new user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!",
    "full_name": "Test User"
  }'
# Save the access_token from response

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!"
  }'
# Expected: {"access_token":"eyJ...","token_type":"bearer"}

# Export token for next tests
export TOKEN="paste_token_here"

# Get current user
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN"
# Expected: User object with email, role, etc.
```

**Pass Criteria:**
- [ ] Registration returns JWT token
- [ ] Login returns JWT token
- [ ] /me endpoint returns user data with valid token
- [ ] Returns 401 without token

---

### 3. Stock Data APIs (3 minutes)

```bash
# Get stock list
curl "http://localhost:8000/api/stocks?limit=10"
# Expected: Array of stocks with symbol, name, sector

# Get specific stock
curl http://localhost:8000/api/stocks/AAPL
# Expected: Detailed stock info including price, market_cap

# Get stock history
curl "http://localhost:8000/api/stocks/AAPL/history?period=1M"
# Expected: Array of OHLCV data

# Search stocks
curl "http://localhost:8000/api/stocks/search?query=apple"
# Expected: Search results with matching stocks
```

**Pass Criteria:**
- [ ] Stock list returns data
- [ ] Stock detail returns for valid symbol
- [ ] Returns 404 for invalid symbol
- [ ] History returns price data
- [ ] Search returns results

---

### 4. Analysis APIs (2 minutes)

```bash
# Get comprehensive analysis
curl http://localhost:8000/api/analysis/AAPL
# Expected: Technical, fundamental, sentiment analysis

# Get technical analysis
curl http://localhost:8000/api/analysis/AAPL/technical
# Expected: RSI, MACD, moving averages, signals
```

**Pass Criteria:**
- [ ] Analysis endpoint returns data
- [ ] Contains technical indicators
- [ ] Contains fundamental metrics
- [ ] Contains sentiment scores

---

### 5. Recommendations (2 minutes)

```bash
# Get daily recommendations (requires auth)
curl http://localhost:8000/api/recommendations \
  -H "Authorization: Bearer $TOKEN"
# Expected: Array of recommendations with symbols, confidence, rationale

# Get personalized recommendations
curl "http://localhost:8000/api/recommendations/personalized?risk_level=moderate" \
  -H "Authorization: Bearer $TOKEN"
# Expected: Filtered recommendations based on risk level
```

**Pass Criteria:**
- [ ] Returns recommendations with auth
- [ ] Returns 401 without auth
- [ ] Each recommendation has required fields (symbol, recommendation, confidence)
- [ ] Risk filtering works

---

### 6. Portfolio Management (3 minutes)

```bash
# Create portfolio
curl -X POST http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Portfolio",
    "description": "Test portfolio",
    "initial_balance": 10000
  }'
# Save portfolio ID from response

export PORTFOLIO_ID="1"

# Get portfolios
curl http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer $TOKEN"
# Expected: Array of user portfolios

# Add position
curl -X POST "http://localhost:8000/api/portfolio/$PORTFOLIO_ID/positions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 10,
    "purchase_price": 150.00
  }'
# Expected: Position created confirmation

# Get portfolio details
curl "http://localhost:8000/api/portfolio/$PORTFOLIO_ID" \
  -H "Authorization: Bearer $TOKEN"
# Expected: Portfolio with positions, performance metrics
```

**Pass Criteria:**
- [ ] Portfolio creation works
- [ ] Portfolio list shows created portfolio
- [ ] Position addition works
- [ ] Portfolio value calculated correctly
- [ ] Requires authentication

---

## WebSocket Tests (5 minutes)

### Using Browser Console

```javascript
// Open browser console at http://localhost:3000
// Or use browser's WebSocket testing tools

// 1. Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/api/ws/stream?client_id=test-123');

ws.onopen = () => {
  console.log('✓ WebSocket connected');

  // 2. Subscribe to stocks
  ws.send(JSON.stringify({
    type: 'subscribe',
    symbols: ['AAPL', 'GOOGL', 'MSFT']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);

  if (data.type === 'price_update') {
    console.log(`✓ Price update for ${data.symbol}: $${data.price}`);
  }
};

ws.onerror = (error) => {
  console.error('✗ WebSocket error:', error);
};

// Wait 10 seconds to receive price updates

// 3. Unsubscribe
ws.send(JSON.stringify({
  type: 'unsubscribe',
  symbols: ['AAPL']
}));

// 4. Send heartbeat
ws.send(JSON.stringify({
  type: 'heartbeat'
}));

// 5. Close connection
// ws.close();
```

### Using wscat (CLI)

```bash
# Install wscat if needed
npm install -g wscat

# Connect to WebSocket
wscat -c "ws://localhost:8000/api/ws/stream?client_id=test-cli"

# After connection, type and press Enter:
{"type":"subscribe","symbols":["AAPL","GOOGL"]}

# You should see price updates streaming

# Send heartbeat
{"type":"heartbeat"}

# Unsubscribe
{"type":"unsubscribe","symbols":["AAPL"]}
```

**Pass Criteria:**
- [ ] WebSocket connection established
- [ ] Welcome message received
- [ ] Subscribe confirmation received
- [ ] Price updates streaming
- [ ] Heartbeat response received
- [ ] Unsubscribe works
- [ ] Connection can be closed gracefully

---

## Frontend Tests (10 minutes)

### 1. Application Load (1 minute)

```bash
# Open browser to
http://localhost:3000
```

**Visual Checks:**
- [ ] Page loads without errors
- [ ] No console errors (F12 -> Console)
- [ ] Navigation bar visible
- [ ] Login/Register buttons visible

---

### 2. Login Flow (2 minutes)

**Steps:**
1. Click "Login" button
2. Enter email: `test@example.com`
3. Enter password: `SecurePass123!`
4. Click "Sign In"

**Pass Criteria:**
- [ ] Login form displays correctly
- [ ] Form validation works
- [ ] Successful login redirects to dashboard
- [ ] User menu shows logged-in state
- [ ] Logout button available

---

### 3. Dashboard (2 minutes)

**URL:** `http://localhost:3000/dashboard`

**Visual Checks:**
- [ ] Market overview widget displays
- [ ] Portfolio summary card shows (if portfolio exists)
- [ ] Recent recommendations section visible
- [ ] Charts render without errors
- [ ] All data loads without errors

---

### 4. Stock Analysis (2 minutes)

**Steps:**
1. Navigate to Analysis page
2. Search for "AAPL"
3. Click on Apple Inc.

**Pass Criteria:**
- [ ] Stock search works
- [ ] Stock detail page loads
- [ ] Price chart renders
- [ ] Technical indicators display
- [ ] Company info shows
- [ ] News section populated

---

### 5. Portfolio Management (2 minutes)

**Steps:**
1. Navigate to Portfolio page
2. View existing portfolio (created via API)
3. Try to add a position via UI
4. Verify position appears in list

**Pass Criteria:**
- [ ] Portfolio list displays
- [ ] Portfolio details load
- [ ] Positions table shows data
- [ ] Performance metrics calculated
- [ ] Add position dialog works
- [ ] Charts render correctly

---

### 6. Recommendations (1 minute)

**URL:** `http://localhost:3000/recommendations`

**Pass Criteria:**
- [ ] Daily recommendations load
- [ ] Filter dropdowns work
- [ ] Recommendation cards display all info
- [ ] "View Details" button works
- [ ] Risk levels visible

---

## Database Integration (5 minutes)

```bash
# Connect to PostgreSQL
docker exec -it investment_db psql -U postgres -d investment_db

# Run these SQL commands:

-- Check tables exist
\dt

-- Check TimescaleDB
SELECT * FROM pg_extension WHERE extname = 'timescaledb';

-- Verify data
SELECT COUNT(*) FROM stocks;
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM portfolios;

-- Check recent activity
SELECT * FROM users ORDER BY created_at DESC LIMIT 5;

-- Exit
\q
```

**Pass Criteria:**
- [ ] Database connection successful
- [ ] All tables exist
- [ ] TimescaleDB extension enabled
- [ ] Data present in tables
- [ ] Recent test user visible

---

## Redis Cache (3 minutes)

```bash
# Connect to Redis
docker exec -it investment_cache redis-cli -a "${REDIS_PASSWORD}"

# Run these commands:

# Check connection
PING
# Expected: PONG

# List cached keys
KEYS *

# Check specific cache
KEYS stock:*
KEYS analysis:*

# Get cache info
INFO stats

# Check memory usage
INFO memory

# Exit
quit
```

**Pass Criteria:**
- [ ] Redis connection successful
- [ ] Cache keys present
- [ ] Memory usage reasonable (<100MB)
- [ ] Hit/miss ratio tracked

---

## Security Quick Tests (5 minutes)

### 1. Authentication Required

```bash
# Try to access protected endpoint without token
curl http://localhost:8000/api/portfolio
# Expected: 401 Unauthorized

# Try with invalid token
curl http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer invalid_token"
# Expected: 401 Unauthorized

# Try with valid token
curl http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer $TOKEN"
# Expected: 200 OK with data
```

**Pass Criteria:**
- [ ] Protected endpoints require auth
- [ ] Invalid tokens rejected
- [ ] Valid tokens accepted

---

### 2. Rate Limiting

```bash
# Make rapid login attempts (should trigger rate limit)
for i in {1..20}; do
  curl -X POST http://localhost:8000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@example.com","password":"wrong"}' \
    -w "%{http_code}\n" -o /dev/null -s
done
```

**Expected:** After several attempts, should see `429` status code

**Pass Criteria:**
- [ ] Rate limiting triggers
- [ ] 429 status code returned
- [ ] X-RateLimit headers present

---

### 3. Input Validation

```bash
# Invalid email format
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"not-an-email","password":"Test123!","full_name":"Test"}'
# Expected: 422 Validation Error

# Weak password
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test2@example.com","password":"123","full_name":"Test"}'
# Expected: 422 Validation Error or 400 Bad Request
```

**Pass Criteria:**
- [ ] Email validation works
- [ ] Password validation works
- [ ] Returns appropriate error codes
- [ ] Error messages are descriptive

---

## Performance Quick Check (5 minutes)

```bash
# Test response times
time curl http://localhost:8000/api/health
# Expected: <100ms

time curl http://localhost:8000/api/stocks/AAPL
# Expected: <500ms (first call, cache miss)

time curl http://localhost:8000/api/stocks/AAPL
# Expected: <100ms (second call, cache hit)

time curl http://localhost:8000/api/analysis/AAPL
# Expected: <2s (includes external API calls)
```

**Pass Criteria:**
- [ ] Health endpoint: <100ms
- [ ] Cached requests: <100ms
- [ ] Uncached stock data: <1s
- [ ] Analysis endpoint: <3s

---

## Quick Test Summary

### All Tests Pass Checklist

**Backend API:**
- [ ] Health endpoints working
- [ ] Authentication flow complete
- [ ] Stock data endpoints functional
- [ ] Analysis endpoints returning data
- [ ] Recommendations working
- [ ] Portfolio CRUD operations work

**Frontend:**
- [ ] Application loads
- [ ] Login works
- [ ] Dashboard displays
- [ ] Stock analysis page works
- [ ] Portfolio page functional
- [ ] No console errors

**Integration:**
- [ ] Database connected and populated
- [ ] Redis caching working
- [ ] WebSocket connections stable
- [ ] External APIs responding

**Security:**
- [ ] Authentication enforced
- [ ] Rate limiting active
- [ ] Input validation working

**Performance:**
- [ ] Response times acceptable
- [ ] Caching working
- [ ] No memory leaks observed

---

## If Tests Fail

### Backend Issues

```bash
# Check backend logs
docker-compose logs backend | tail -50

# Restart backend
docker-compose restart backend

# Check backend health
curl http://localhost:8000/api/health/readiness
```

### Database Issues

```bash
# Check database logs
docker-compose logs postgres | tail -50

# Verify database running
docker-compose ps postgres

# Test connection
docker exec -it investment_db psql -U postgres -d investment_db -c "SELECT 1;"
```

### Redis Issues

```bash
# Check Redis logs
docker-compose logs redis | tail -50

# Test Redis
docker exec -it investment_cache redis-cli -a "${REDIS_PASSWORD}" PING
```

### Frontend Issues

```bash
# Check frontend logs
docker-compose logs frontend | tail -50

# Rebuild frontend
docker-compose build frontend
docker-compose up -d frontend
```

---

## Quick Commands Reference

```bash
# Start everything
./start.sh dev

# Stop everything
./stop.sh

# View all logs
./logs.sh

# View specific service
./logs.sh backend

# Restart a service
docker-compose restart backend

# Check service status
docker-compose ps

# Execute command in container
docker-compose exec backend bash

# Run pytest
docker-compose exec backend pytest

# Check database
docker exec -it investment_db psql -U postgres -d investment_db

# Check Redis
docker exec -it investment_cache redis-cli -a "${REDIS_PASSWORD}"
```

---

## Time Estimates

- **Pre-Test Setup:** 5 minutes
- **Backend API Tests:** 15 minutes
- **WebSocket Tests:** 5 minutes
- **Frontend Tests:** 10 minutes
- **Database/Redis:** 8 minutes
- **Security Tests:** 5 minutes
- **Performance Tests:** 5 minutes

**Total Execution Time:** ~50 minutes for complete test suite

**Minimum Critical Path:** ~20 minutes (Backend API + Frontend + Integration basics)

---

## Next Steps After Quick Tests

1. If all tests pass → Proceed with detailed testing using `TEST_PLAN.md`
2. If tests fail → Check troubleshooting section
3. Document any issues found
4. Run automated test suite: `pytest backend/tests/`
5. Review logs for warnings/errors

---

**Last Updated:** 2026-01-02
**Version:** 1.0.0
