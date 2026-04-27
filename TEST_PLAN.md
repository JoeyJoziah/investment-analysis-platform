# Investment Analysis Platform - Comprehensive Test Plan

## Test Plan Overview

This document provides a structured test checklist for the Investment Analysis Platform. Execute these tests once all services are running.

**Platform Architecture:**
- Backend API: FastAPI (Port 8000)
- Frontend: React + TypeScript (Port 3000)
- Database: PostgreSQL with TimescaleDB (Port 5432)
- Cache: Redis (Port 6379)
- Search: Elasticsearch (Port 9200)

---

## Table of Contents
1. [Test Environment Setup](#test-environment-setup)
2. [Backend API Tests](#backend-api-tests)
3. [Frontend Tests](#frontend-tests)
4. [Integration Tests](#integration-tests)
5. [Security Tests](#security-tests)
6. [Performance Tests](#performance-tests)
7. [WebSocket Tests](#websocket-tests)

---

## Test Environment Setup

### Prerequisites Checklist

- [ ] Docker and Docker Compose installed
- [ ] All environment variables configured in `.env` file
- [ ] API keys available (Alpha Vantage, Finnhub, Polygon)
- [ ] Python 3.11+ installed (for local testing)
- [ ] Node.js 18+ installed (for frontend testing)
- [ ] curl or httpie installed for API testing
- [ ] WebSocket client available (wscat, websocat, or browser dev tools)

### Start Services

```bash
# Start all services
./start.sh dev

# Verify all containers are running
docker ps

# Check service health
docker-compose ps
```

### Expected Running Services
- [ ] postgres (port 5432) - healthy
- [ ] redis (port 6379) - healthy
- [ ] elasticsearch (port 9200) - healthy
- [ ] backend (port 8000) - running
- [ ] frontend (port 3000) - running

---

## Backend API Tests

### 1. Health Endpoints

#### 1.1 Basic Health Check
**Endpoint:** `GET /api/health`

```bash
curl http://localhost:8000/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-02T...",
  "version": "1.0.0",
  "service": "investment-analysis-api"
}
```

**Checklist:**
- [ ] Returns 200 status code
- [ ] Contains valid timestamp
- [ ] Status is "healthy"
- [ ] Version matches expected version

---

#### 1.2 Readiness Check
**Endpoint:** `GET /api/health/readiness`

```bash
curl http://localhost:8000/api/health/readiness
```

**Expected Response:**
```json
{
  "status": "ready",
  "checks": {
    "database": true,
    "cache": true,
    "api": true
  },
  "timestamp": "..."
}
```

**Checklist:**
- [ ] Returns 200 status code
- [ ] Database check passes
- [ ] Cache check passes
- [ ] API check passes
- [ ] Overall status is "ready"

---

#### 1.3 Metrics Endpoint
**Endpoint:** `GET /api/health/metrics`

```bash
curl http://localhost:8000/api/health/metrics
```

**Checklist:**
- [ ] Returns 200 status code
- [ ] Contains system metrics (CPU, memory, disk)
- [ ] Contains database pool stats
- [ ] Contains Redis info
- [ ] All numeric values are valid

---

#### 1.4 Liveness Probe
**Endpoint:** `GET /api/health/liveness`

```bash
curl http://localhost:8000/api/health/liveness
```

**Checklist:**
- [ ] Returns 200 status code
- [ ] Status is "alive"
- [ ] Response time < 1 second

---

### 2. Authentication Endpoints

#### 2.1 User Registration
**Endpoint:** `POST /api/auth/register`

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!",
    "full_name": "Test User"
  }'
```

**Expected Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

**Checklist:**
- [ ] Returns 200 status code for new user
- [ ] Returns JWT access token
- [ ] Token type is "bearer"
- [ ] Returns 400 for duplicate email
- [ ] Validates email format
- [ ] Requires password complexity

---

#### 2.2 User Login (OAuth2)
**Endpoint:** `POST /api/auth/token`

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=SecurePass123!"
```

**Checklist:**
- [ ] Returns 200 for valid credentials
- [ ] Returns JWT token
- [ ] Returns 401 for invalid credentials
- [ ] Updates last_login timestamp
- [ ] Rate limiting works (429 after many attempts)

---

#### 2.3 Alternative Login
**Endpoint:** `POST /api/auth/login`

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePass123!"
  }'
```

**Checklist:**
- [ ] Returns 200 for valid credentials
- [ ] Returns JWT token
- [ ] Returns 401 for invalid credentials

---

#### 2.4 Get Current User
**Endpoint:** `GET /api/auth/me`

```bash
TOKEN="your_jwt_token_here"
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

**Expected Response:**
```json
{
  "id": 1,
  "email": "test@example.com",
  "full_name": "Test User",
  "role": "free_user",
  "is_active": true,
  "created_at": "..."
}
```

**Checklist:**
- [ ] Returns 200 with valid token
- [ ] Returns 401 without token
- [ ] Returns 401 with invalid/expired token
- [ ] Returns correct user information

---

#### 2.5 Token Refresh
**Endpoint:** `POST /api/auth/refresh`

```bash
curl -X POST http://localhost:8000/api/auth/refresh \
  -H "Authorization: Bearer $TOKEN"
```

**Checklist:**
- [ ] Returns new access token
- [ ] New token is valid
- [ ] Old token still works (until expiry)

---

#### 2.6 Logout
**Endpoint:** `POST /api/auth/logout`

```bash
curl -X POST http://localhost:8000/api/auth/logout \
  -H "Authorization: Bearer $TOKEN"
```

**Checklist:**
- [ ] Returns 200 status
- [ ] Returns success message
- [ ] Requires valid token

---

### 3. Stock Data Endpoints

#### 3.1 Get Stock List
**Endpoint:** `GET /api/stocks`

```bash
curl "http://localhost:8000/api/stocks?limit=10&offset=0"
```

**Checklist:**
- [ ] Returns 200 status code
- [ ] Returns array of stocks
- [ ] Respects limit parameter
- [ ] Respects offset parameter
- [ ] Contains stock symbol, name, sector
- [ ] Pagination works correctly

---

#### 3.2 Get Stock by Symbol
**Endpoint:** `GET /api/stocks/{symbol}`

```bash
curl http://localhost:8000/api/stocks/AAPL
```

**Expected Response Fields:**
- symbol
- name
- sector
- market_cap
- current_price
- company_overview

**Checklist:**
- [ ] Returns 200 for valid symbol
- [ ] Returns 404 for invalid symbol
- [ ] Contains complete stock data
- [ ] Real-time quote data present
- [ ] Company overview present

---

#### 3.3 Get Stock Price History
**Endpoint:** `GET /api/stocks/{symbol}/history`

```bash
curl "http://localhost:8000/api/stocks/AAPL/history?period=1M"
```

**Checklist:**
- [ ] Returns 200 status
- [ ] Returns array of price data
- [ ] Contains OHLCV data
- [ ] Respects period parameter (1D, 1W, 1M, 3M, 1Y)
- [ ] Data is sorted by date
- [ ] Contains volume data

---

#### 3.4 Search Stocks
**Endpoint:** `GET /api/stocks/search`

```bash
curl "http://localhost:8000/api/stocks/search?query=apple"
```

**Checklist:**
- [ ] Returns 200 status
- [ ] Returns matching stocks
- [ ] Search works for symbol
- [ ] Search works for company name
- [ ] Results are ranked by relevance

---

### 4. Analysis Endpoints

#### 4.1 Get Stock Analysis
**Endpoint:** `GET /api/analysis/{symbol}`

```bash
curl http://localhost:8000/api/analysis/AAPL
```

**Expected Response Sections:**
- technical_analysis (RSI, MACD, moving averages)
- fundamental_analysis (P/E, revenue, earnings)
- sentiment_analysis (news sentiment, social sentiment)
- price_prediction (ML-based forecast)

**Checklist:**
- [ ] Returns 200 for valid symbol
- [ ] Returns 404 for invalid symbol
- [ ] Technical indicators present
- [ ] Fundamental metrics present
- [ ] Sentiment scores present
- [ ] Price predictions present
- [ ] Analysis timestamp present

---

#### 4.2 Get Technical Analysis
**Endpoint:** `GET /api/analysis/{symbol}/technical`

```bash
curl http://localhost:8000/api/analysis/AAPL/technical
```

**Expected Indicators:**
- RSI (14-day)
- MACD
- Moving Averages (SMA 20, 50, 200)
- Bollinger Bands
- Support/Resistance levels

**Checklist:**
- [ ] Returns 200 status
- [ ] All technical indicators present
- [ ] Values are within valid ranges
- [ ] Buy/sell signals included
- [ ] Trend direction provided

---

#### 4.3 Get Fundamental Analysis
**Endpoint:** `GET /api/analysis/{symbol}/fundamental`

```bash
curl http://localhost:8000/api/analysis/AAPL/fundamental
```

**Expected Metrics:**
- P/E ratio
- EPS
- Revenue
- Market cap
- Debt-to-equity
- ROE, ROA

**Checklist:**
- [ ] Returns 200 status
- [ ] Key financial ratios present
- [ ] Earnings data present
- [ ] Balance sheet metrics present
- [ ] Valuation metrics present

---

#### 4.4 Get Sentiment Analysis
**Endpoint:** `GET /api/analysis/{symbol}/sentiment`

```bash
curl http://localhost:8000/api/analysis/AAPL/sentiment
```

**Checklist:**
- [ ] Returns 200 status
- [ ] News sentiment score (-1 to 1)
- [ ] Social media sentiment
- [ ] Recent news articles included
- [ ] Sentiment trend over time

---

### 5. Recommendations Endpoints

#### 5.1 Get Daily Recommendations
**Endpoint:** `GET /api/recommendations`

```bash
curl http://localhost:8000/api/recommendations \
  -H "Authorization: Bearer $TOKEN"
```

**Expected Response:**
```json
{
  "recommendations": [
    {
      "symbol": "AAPL",
      "recommendation": "buy",
      "confidence": 0.85,
      "target_price": 200.50,
      "time_horizon": "medium_term",
      "rationale": "...",
      "risk_level": "moderate"
    }
  ],
  "generated_at": "...",
  "total": 10
}
```

**Checklist:**
- [ ] Returns 200 with auth
- [ ] Returns 401 without auth
- [ ] Recommendations have all required fields
- [ ] Confidence scores between 0-1
- [ ] Includes rationale
- [ ] Risk levels are valid enums

---

#### 5.2 Get Personalized Recommendations
**Endpoint:** `GET /api/recommendations/personalized`

```bash
curl "http://localhost:8000/api/recommendations/personalized?risk_level=moderate" \
  -H "Authorization: Bearer $TOKEN"
```

**Checklist:**
- [ ] Returns 200 status
- [ ] Respects risk level filter
- [ ] Respects user portfolio context
- [ ] Different from general recommendations

---

#### 5.3 Get Recommendation Details
**Endpoint:** `GET /api/recommendations/{symbol}`

```bash
curl http://localhost:8000/api/recommendations/AAPL \
  -H "Authorization: Bearer $TOKEN"
```

**Checklist:**
- [ ] Returns 200 for valid symbol
- [ ] Detailed analysis included
- [ ] Multiple recommendation sources
- [ ] Historical accuracy metrics
- [ ] Risk assessment included

---

### 6. Portfolio Endpoints

#### 6.1 Get User Portfolios
**Endpoint:** `GET /api/portfolio`

```bash
curl http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer $TOKEN"
```

**Checklist:**
- [ ] Returns 200 with auth
- [ ] Returns 401 without auth
- [ ] Returns array of portfolios
- [ ] Each portfolio has name, value, positions
- [ ] Performance metrics included

---

#### 6.2 Create Portfolio
**Endpoint:** `POST /api/portfolio`

```bash
curl -X POST http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Portfolio",
    "description": "Test portfolio for testing",
    "initial_balance": 10000
  }'
```

**Checklist:**
- [ ] Returns 201 for successful creation
- [ ] Returns portfolio ID
- [ ] Portfolio appears in list
- [ ] Initial balance set correctly

---

#### 6.3 Get Portfolio Details
**Endpoint:** `GET /api/portfolio/{portfolio_id}`

```bash
curl http://localhost:8000/api/portfolio/1 \
  -H "Authorization: Bearer $TOKEN"
```

**Expected Response:**
```json
{
  "id": "1",
  "name": "Test Portfolio",
  "total_value": 10500.50,
  "cash_balance": 5000.00,
  "positions": [...],
  "performance": {
    "total_return": 5.00,
    "total_return_percent": 5.00
  }
}
```

**Checklist:**
- [ ] Returns 200 for owned portfolio
- [ ] Returns 403 for other user's portfolio
- [ ] Contains all positions
- [ ] Performance metrics calculated
- [ ] Real-time values updated

---

#### 6.4 Add Position to Portfolio
**Endpoint:** `POST /api/portfolio/{portfolio_id}/positions`

```bash
curl -X POST http://localhost:8000/api/portfolio/1/positions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 10,
    "purchase_price": 150.00
  }'
```

**Checklist:**
- [ ] Returns 201 for successful addition
- [ ] Position appears in portfolio
- [ ] Cash balance reduced correctly
- [ ] Returns 400 for insufficient funds
- [ ] Returns 404 for invalid symbol

---

#### 6.5 Update Position
**Endpoint:** `PUT /api/portfolio/{portfolio_id}/positions/{position_id}`

```bash
curl -X PUT http://localhost:8000/api/portfolio/1/positions/1 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "quantity": 5
  }'
```

**Checklist:**
- [ ] Returns 200 for successful update
- [ ] Position quantity updated
- [ ] Portfolio value recalculated
- [ ] Transaction recorded

---

#### 6.6 Delete Position
**Endpoint:** `DELETE /api/portfolio/{portfolio_id}/positions/{position_id}`

```bash
curl -X DELETE http://localhost:8000/api/portfolio/1/positions/1 \
  -H "Authorization: Bearer $TOKEN"
```

**Checklist:**
- [ ] Returns 204 for successful deletion
- [ ] Position removed from portfolio
- [ ] Cash balance updated
- [ ] Sale transaction recorded

---

#### 6.7 Get Portfolio Performance
**Endpoint:** `GET /api/portfolio/{portfolio_id}/performance`

```bash
curl "http://localhost:8000/api/portfolio/1/performance?period=1M" \
  -H "Authorization: Bearer $TOKEN"
```

**Checklist:**
- [ ] Returns 200 status
- [ ] Historical performance data
- [ ] Benchmark comparison (S&P 500)
- [ ] Risk metrics (volatility, Sharpe ratio)
- [ ] Returns over time

---

### 7. Admin Endpoints

#### 7.1 System Status
**Endpoint:** `GET /api/admin/status`

```bash
curl http://localhost:8000/api/admin/status \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Checklist:**
- [ ] Returns 200 for admin user
- [ ] Returns 403 for non-admin
- [ ] System metrics present
- [ ] Service health status
- [ ] Database statistics

---

#### 7.2 Cache Management
**Endpoint:** `GET /api/cache/stats`

```bash
curl http://localhost:8000/api/cache/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Checklist:**
- [ ] Returns cache statistics
- [ ] Hit/miss ratios
- [ ] Memory usage
- [ ] Key counts by type

---

#### 7.3 Clear Cache
**Endpoint:** `POST /api/cache/clear`

```bash
curl -X POST http://localhost:8000/api/cache/clear \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**Checklist:**
- [ ] Returns 200 status
- [ ] Cache cleared successfully
- [ ] Only admin can clear cache

---

## WebSocket Tests

### 8.1 WebSocket Connection
**Endpoint:** `WS /api/ws/stream`

```javascript
// Browser console test
const ws = new WebSocket('ws://localhost:8000/api/ws/stream?client_id=test-123');

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  console.log('Received:', JSON.parse(event.data));
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

**Using wscat (CLI):**
```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket
wscat -c "ws://localhost:8000/api/ws/stream?client_id=test-123"
```

**Checklist:**
- [ ] Connection established successfully
- [ ] Welcome message received
- [ ] Client ID acknowledged
- [ ] Server version in welcome message

---

### 8.2 Subscribe to Stock Updates
**Message Format:**
```json
{
  "type": "subscribe",
  "symbols": ["AAPL", "GOOGL", "MSFT"]
}
```

**Checklist:**
- [ ] Subscription confirmation received
- [ ] Price updates start streaming
- [ ] Updates for all subscribed symbols
- [ ] Update frequency reasonable (1-3 seconds)

---

### 8.3 Unsubscribe from Stock
**Message Format:**
```json
{
  "type": "unsubscribe",
  "symbols": ["AAPL"]
}
```

**Checklist:**
- [ ] Unsubscribe confirmation received
- [ ] Price updates stop for unsubscribed symbol
- [ ] Other subscriptions continue

---

### 8.4 Heartbeat/Ping
**Message Format:**
```json
{
  "type": "heartbeat"
}
```

**Checklist:**
- [ ] Pong response received
- [ ] Server timestamp in response
- [ ] Connection kept alive

---

### 8.5 Authenticated WebSocket
**Endpoint:** `WS /api/ws/stream?token=JWT_TOKEN`

**Checklist:**
- [ ] Connection with valid token succeeds
- [ ] Connection without token limited features
- [ ] Invalid token rejected
- [ ] Token expiry handled gracefully

---

### 8.6 WebSocket Price Updates Format

**Expected Price Update:**
```json
{
  "type": "price_update",
  "symbol": "AAPL",
  "price": 195.50,
  "change": 2.30,
  "change_percent": 1.19,
  "volume": 45000000,
  "bid": 195.48,
  "ask": 195.52,
  "timestamp": "2026-01-02T12:34:56Z"
}
```

**Checklist:**
- [ ] All fields present
- [ ] Prices are realistic
- [ ] Timestamp is current
- [ ] Volume is positive integer

---

### 8.7 WebSocket Alert Messages

**Expected Alert:**
```json
{
  "type": "alert",
  "alert": {
    "alert_type": "price_target",
    "message": "AAPL reached target price",
    "severity": "info"
  },
  "timestamp": "..."
}
```

**Checklist:**
- [ ] Alerts received for configured triggers
- [ ] Alert format is consistent
- [ ] Severity levels correct

---

### 8.8 WebSocket Connection Management

**Checklist:**
- [ ] Multiple concurrent connections work
- [ ] Disconnection handled gracefully
- [ ] Automatic reconnection attempts
- [ ] Stale connections cleaned up
- [ ] Connection limits enforced

---

### 8.9 Market Data Stream
**Endpoint:** `WS /api/ws/market`

```bash
wscat -c "ws://localhost:8000/api/ws/market"
```

**Checklist:**
- [ ] Market overview data received
- [ ] Major indices data (SPY, QQQ, DIA)
- [ ] Market sentiment metrics
- [ ] VIX data
- [ ] Advance/decline data
- [ ] Updates every 5 seconds

---

### 8.10 Portfolio Stream
**Endpoint:** `WS /api/ws/portfolio/{portfolio_id}`

```bash
wscat -c "ws://localhost:8000/api/ws/portfolio/1?token=JWT_TOKEN"
```

**Checklist:**
- [ ] Requires authentication
- [ ] Portfolio value updates
- [ ] Position updates
- [ ] Alert notifications
- [ ] Updates every 3 seconds

---

## Frontend Tests

### 9. Frontend Functionality Tests

#### 9.1 Application Load
**URL:** `http://localhost:3000`

**Checklist:**
- [ ] Page loads without errors
- [ ] No console errors
- [ ] Assets load correctly (CSS, JS)
- [ ] Favicon displays
- [ ] Loading states show properly

---

#### 9.2 Dashboard Page
**URL:** `http://localhost:3000/dashboard`

**Checklist:**
- [ ] Dashboard loads successfully
- [ ] Market overview widget displays
- [ ] Portfolio summary visible
- [ ] Recent recommendations shown
- [ ] Charts render correctly
- [ ] Real-time updates work
- [ ] Responsive design works

---

#### 9.3 Login Page
**URL:** `http://localhost:3000/login`

**Checklist:**
- [ ] Login form displays
- [ ] Email validation works
- [ ] Password validation works
- [ ] Submit button functional
- [ ] Error messages display correctly
- [ ] Success redirects to dashboard
- [ ] "Remember me" option works
- [ ] "Forgot password" link present

---

#### 9.4 Registration Flow

**Checklist:**
- [ ] Registration form displays
- [ ] All required fields present
- [ ] Email format validation
- [ ] Password strength indicator
- [ ] Password confirmation match check
- [ ] Terms acceptance required
- [ ] Successful registration redirects
- [ ] Duplicate email error shown

---

#### 9.5 Stock Analysis Page
**URL:** `http://localhost:3000/analysis/AAPL`

**Checklist:**
- [ ] Stock symbol accepted in URL
- [ ] Company info displays
- [ ] Price chart renders
- [ ] Technical indicators shown
- [ ] Fundamental data displayed
- [ ] News section populated
- [ ] Recommendation visible
- [ ] Time period selector works
- [ ] Export data option available

---

#### 9.6 Portfolio Page
**URL:** `http://localhost:3000/portfolio`

**Checklist:**
- [ ] Portfolio list displays
- [ ] Create portfolio button works
- [ ] Portfolio cards show summary
- [ ] Add position dialog works
- [ ] Edit position functional
- [ ] Delete position confirmation
- [ ] Performance charts render
- [ ] Allocation pie chart displays
- [ ] Export portfolio data works

---

#### 9.7 Recommendations Page
**URL:** `http://localhost:3000/recommendations`

**Checklist:**
- [ ] Daily recommendations load
- [ ] Filter by risk level works
- [ ] Filter by category works
- [ ] Sort options functional
- [ ] Recommendation cards complete
- [ ] "View Details" button works
- [ ] Save recommendation works
- [ ] Share recommendation works

---

#### 9.8 Market Overview Page
**URL:** `http://localhost:3000/market`

**Checklist:**
- [ ] Market indices display
- [ ] Sector performance shown
- [ ] Top gainers/losers lists
- [ ] Heat map renders
- [ ] Real-time updates work
- [ ] Charts interactive

---

#### 9.9 Watchlist Page
**URL:** `http://localhost:3000/watchlist`

**Checklist:**
- [ ] Watchlist items display
- [ ] Add symbol dialog works
- [ ] Remove symbol confirmation
- [ ] Real-time price updates
- [ ] Quick actions functional
- [ ] Sort/filter options work

---

#### 9.10 Settings Page
**URL:** `http://localhost:3000/settings`

**Checklist:**
- [ ] User profile section
- [ ] Email preferences
- [ ] Notification settings
- [ ] Theme toggle (light/dark)
- [ ] Risk tolerance setting
- [ ] Save settings works
- [ ] Password change functional

---

### 10. Frontend Navigation

**Checklist:**
- [ ] Navigation bar always visible
- [ ] All menu items clickable
- [ ] Active page highlighted
- [ ] Mobile menu works
- [ ] Breadcrumbs functional
- [ ] Back button works
- [ ] Deep linking works

---

### 11. Frontend Error Handling

**Checklist:**
- [ ] API errors displayed to user
- [ ] Network errors show retry option
- [ ] 404 page for invalid routes
- [ ] Form validation errors clear
- [ ] Loading states prevent double-submit
- [ ] Error boundaries catch crashes

---

## Integration Tests

### 12. Database Integration

#### 12.1 PostgreSQL Connection

```bash
# Connect to database
docker exec -it investment_db psql -U postgres -d investment_db
```

**SQL Tests:**
```sql
-- Check tables exist
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public';

-- Check TimescaleDB extension
SELECT * FROM pg_extension WHERE extname = 'timescaledb';

-- Verify hypertables (for time-series data)
SELECT * FROM timescaledb_information.hypertables;

-- Check stock data
SELECT COUNT(*) FROM stocks;

-- Check price history
SELECT COUNT(*) FROM price_history;

-- Check user table
SELECT COUNT(*) FROM users;
```

**Checklist:**
- [ ] Database connection successful
- [ ] All required tables present
- [ ] TimescaleDB extension enabled
- [ ] Hypertables configured
- [ ] Indexes created
- [ ] Foreign keys enforced
- [ ] Sample data present

---

#### 12.2 Database Migrations

```bash
# Check migration status
alembic current

# Show migration history
alembic history
```

**Checklist:**
- [ ] All migrations applied
- [ ] Migration version tracked
- [ ] No pending migrations
- [ ] Rollback works

---

### 13. Redis Cache Integration

#### 13.1 Redis Connection

```bash
# Connect to Redis
docker exec -it investment_cache redis-cli -a "${REDIS_PASSWORD}"
```

**Redis Commands:**
```redis
# Check connection
PING

# Check info
INFO

# List all keys
KEYS *

# Check memory usage
INFO memory

# Check specific cache keys
KEYS stock:*
KEYS analysis:*
KEYS user:*

# Get cache hit/miss stats
INFO stats

# Check TTL on a key
TTL stock:AAPL:quote
```

**Checklist:**
- [ ] Redis connection successful
- [ ] Password authentication works
- [ ] Cache keys present
- [ ] TTL set correctly
- [ ] Memory usage reasonable
- [ ] Eviction policy configured
- [ ] Persistence enabled

---

#### 13.2 Cache Functionality

**Test Cache Read/Write:**
```bash
# API call should cache result
curl http://localhost:8000/api/stocks/AAPL

# Check if cached
docker exec -it investment_cache redis-cli -a "${REDIS_PASSWORD}" KEYS "stock:AAPL:*"

# Second call should be faster (from cache)
time curl http://localhost:8000/api/stocks/AAPL
```

**Checklist:**
- [ ] First request caches data
- [ ] Second request faster (cache hit)
- [ ] Cache invalidation works
- [ ] TTL expiration works
- [ ] Cache warming on startup

---

### 14. API Response Caching

**Test Scenarios:**

```bash
# Test 1: Cache miss (first request)
time curl http://localhost:8000/api/analysis/AAPL

# Test 2: Cache hit (second request within TTL)
time curl http://localhost:8000/api/analysis/AAPL

# Test 3: Different parameter (new cache key)
time curl "http://localhost:8000/api/analysis/AAPL?period=1M"
```

**Checklist:**
- [ ] Cache headers present (X-Cache: HIT/MISS)
- [ ] Cache-Control headers set
- [ ] ETag support
- [ ] Conditional requests work (304 Not Modified)
- [ ] Vary header for different users

---

### 15. Background Tasks Integration

**Checklist:**
- [ ] Scheduler starts on boot
- [ ] Daily recommendation task runs
- [ ] Data update tasks execute
- [ ] ML model updates scheduled
- [ ] Error notifications sent
- [ ] Task status logged

---

### 16. External API Integration

#### 16.1 Alpha Vantage Integration

```bash
# Test direct API call
curl "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=${ALPHA_VANTAGE_API_KEY}"
```

**Checklist:**
- [ ] API key valid
- [ ] Rate limiting respected
- [ ] Error handling for API failures
- [ ] Fallback to other sources
- [ ] Cost tracking enabled

---

#### 16.2 Finnhub Integration

**Checklist:**
- [ ] API key valid
- [ ] Real-time quotes work
- [ ] Company profiles retrieved
- [ ] Rate limits respected

---

#### 16.3 Polygon.io Integration

**Checklist:**
- [ ] API key valid
- [ ] Historical data retrieval
- [ ] Aggregates work
- [ ] Free tier limits respected

---

## Security Tests

### 17. Authentication & Authorization

#### 17.1 JWT Token Validation

**Test Cases:**

```bash
# Test 1: No token
curl http://localhost:8000/api/portfolio

# Test 2: Invalid token
curl http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer invalid_token"

# Test 3: Expired token
curl http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer expired_token"

# Test 4: Valid token
curl http://localhost:8000/api/portfolio \
  -H "Authorization: Bearer $VALID_TOKEN"
```

**Checklist:**
- [ ] No token returns 401
- [ ] Invalid token returns 401
- [ ] Expired token returns 401
- [ ] Valid token returns 200
- [ ] Token signature verified
- [ ] Token payload validated

---

#### 17.2 Rate Limiting

**Test Rate Limits:**

```bash
# Make multiple rapid requests
for i in {1..100}; do
  curl http://localhost:8000/api/auth/login \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"email":"test@example.com","password":"wrong"}' \
    -w "%{http_code}\n" -o /dev/null -s
done
```

**Checklist:**
- [ ] Rate limits enforced
- [ ] 429 status code returned when exceeded
- [ ] X-RateLimit headers present
- [ ] Retry-After header set
- [ ] Different limits for different endpoints
- [ ] Rate limit resets correctly

---

#### 17.3 CORS Configuration

```bash
# Test CORS preflight
curl -X OPTIONS http://localhost:8000/api/stocks \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" \
  -v
```

**Checklist:**
- [ ] CORS headers present
- [ ] Allowed origins configured
- [ ] Preflight requests work
- [ ] Credentials allowed if needed
- [ ] Methods whitelist correct

---

#### 17.4 SQL Injection Prevention

**Test Cases:**

```bash
# Attempt SQL injection in search
curl "http://localhost:8000/api/stocks/search?query=AAPL'; DROP TABLE stocks; --"

# Check if database intact
docker exec -it investment_db psql -U postgres -d investment_db -c "SELECT COUNT(*) FROM stocks;"
```

**Checklist:**
- [ ] SQL injection attempts blocked
- [ ] Parameterized queries used
- [ ] Input sanitized
- [ ] Database intact after attempts

---

#### 17.5 XSS Prevention

**Test Cases:**

```bash
# Attempt XSS in user input
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test123!",
    "full_name": "<script>alert('xss')</script>"
  }'
```

**Checklist:**
- [ ] Script tags escaped
- [ ] HTML entities encoded
- [ ] Content-Type headers correct
- [ ] CSP headers present

---

#### 17.6 HTTPS/TLS (Production)

**Production Only:**
- [ ] HTTPS enforced
- [ ] Valid SSL certificate
- [ ] TLS 1.2+ only
- [ ] Strong cipher suites
- [ ] HSTS header present

---

### 18. Input Validation

**Test Cases:**

```bash
# Test 1: Invalid email format
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"invalid-email","password":"Test123!","full_name":"Test"}'

# Test 2: Weak password
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"123","full_name":"Test"}'

# Test 3: SQL injection attempt
curl "http://localhost:8000/api/stocks/'; DROP TABLE stocks; --"

# Test 4: Extremely long input
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"test@example.com\",\"password\":\"Test123!\",\"full_name\":\"$(python3 -c 'print("A"*10000)')\"}"
```

**Checklist:**
- [ ] Invalid formats rejected
- [ ] Length limits enforced
- [ ] Type validation works
- [ ] Range validation works
- [ ] Special characters handled
- [ ] Null/undefined handled

---

## Performance Tests

### 19. Load Testing

#### 19.1 API Endpoint Load Test

**Using Apache Bench:**

```bash
# Test health endpoint
ab -n 1000 -c 10 http://localhost:8000/api/health

# Test stock endpoint
ab -n 500 -c 10 http://localhost:8000/api/stocks/AAPL

# Test analysis endpoint
ab -n 100 -c 5 http://localhost:8000/api/analysis/AAPL
```

**Expected Performance:**
- Health endpoint: >1000 req/sec
- Stock data: >100 req/sec
- Analysis: >20 req/sec

**Checklist:**
- [ ] Response times acceptable
- [ ] No errors under load
- [ ] Memory usage stable
- [ ] CPU usage reasonable
- [ ] Database connections managed

---

#### 19.2 Concurrent Users Test

```bash
# Using siege
siege -c 50 -t 1M http://localhost:8000/api/stocks
```

**Checklist:**
- [ ] Handles 50 concurrent users
- [ ] Handles 100 concurrent users
- [ ] No connection pool exhaustion
- [ ] No memory leaks
- [ ] Response times degrade gracefully

---

### 20. Database Query Performance

```sql
-- Test query performance
EXPLAIN ANALYZE SELECT * FROM stocks WHERE symbol = 'AAPL';

EXPLAIN ANALYZE SELECT * FROM price_history
WHERE symbol = 'AAPL'
AND date >= CURRENT_DATE - INTERVAL '1 year';

-- Check slow queries
SELECT * FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Checklist:**
- [ ] Queries use indexes
- [ ] Query plans optimized
- [ ] No sequential scans on large tables
- [ ] Join performance acceptable
- [ ] Slow query log configured

---

### 21. Caching Performance

**Test Cache Performance:**

```bash
# Measure without cache (first request)
time curl http://localhost:8000/api/analysis/AAPL

# Measure with cache (subsequent requests)
time curl http://localhost:8000/api/analysis/AAPL
time curl http://localhost:8000/api/analysis/AAPL
time curl http://localhost:8000/api/analysis/AAPL
```

**Expected Results:**
- First request: 200-500ms
- Cached requests: <50ms

**Checklist:**
- [ ] Cache hit rate >80%
- [ ] Response time improvement significant
- [ ] Memory usage acceptable
- [ ] Cache eviction working

---

### 22. WebSocket Performance

**Checklist:**
- [ ] Handles 100+ concurrent connections
- [ ] Message latency <100ms
- [ ] No message loss
- [ ] Graceful degradation under load
- [ ] Memory usage per connection reasonable

---

## End-to-End Test Scenarios

### 23. Complete User Journey

#### Scenario 1: New User Registration to First Trade

**Steps:**
1. [ ] Visit homepage
2. [ ] Click "Register"
3. [ ] Fill registration form
4. [ ] Verify email (if enabled)
5. [ ] Login with credentials
6. [ ] View dashboard
7. [ ] Browse recommendations
8. [ ] Click on a recommendation
9. [ ] View detailed analysis
10. [ ] Create portfolio
11. [ ] Add position to portfolio
12. [ ] View updated portfolio
13. [ ] Check performance metrics

---

#### Scenario 2: Daily Recommendation Flow

**Steps:**
1. [ ] Login as existing user
2. [ ] Navigate to recommendations
3. [ ] View daily recommendations
4. [ ] Filter by risk level
5. [ ] Click recommendation details
6. [ ] Review analysis
7. [ ] Add to watchlist
8. [ ] Set price alert
9. [ ] Verify alert configuration

---

#### Scenario 3: Portfolio Management

**Steps:**
1. [ ] Login
2. [ ] Navigate to portfolio
3. [ ] View portfolio summary
4. [ ] Add new position
5. [ ] Edit existing position
6. [ ] View performance charts
7. [ ] Export portfolio report
8. [ ] Delete position
9. [ ] Verify cash balance updated

---

#### Scenario 4: Real-time Market Monitoring

**Steps:**
1. [ ] Login
2. [ ] Navigate to market overview
3. [ ] Subscribe to stock updates (WebSocket)
4. [ ] Verify real-time price updates
5. [ ] Check multiple stocks simultaneously
6. [ ] Verify data accuracy
7. [ ] Test reconnection on network drop

---

## Test Execution Checklist

### Pre-Test Setup

- [ ] All services running
- [ ] Database seeded with test data
- [ ] Test users created
- [ ] API keys valid
- [ ] Logs monitoring enabled
- [ ] Backup created (if testing destructive operations)

### During Testing

- [ ] Document all failures
- [ ] Screenshot errors
- [ ] Save error logs
- [ ] Note performance metrics
- [ ] Record API response times

### Post-Test

- [ ] Review test results
- [ ] File bug reports
- [ ] Update test documentation
- [ ] Clean up test data
- [ ] Archive test logs

---

## Test Tools & Commands

### Useful cURL Commands

```bash
# Pretty print JSON response
curl http://localhost:8000/api/stocks/AAPL | jq '.'

# Show response headers
curl -i http://localhost:8000/api/health

# Follow redirects
curl -L http://localhost:8000/

# Measure response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/stocks/AAPL

# Save response to file
curl http://localhost:8000/api/recommendations > recommendations.json
```

### curl-format.txt
```
time_namelookup:  %{time_namelookup}s\n
time_connect:  %{time_connect}s\n
time_appconnect:  %{time_appconnect}s\n
time_pretransfer:  %{time_pretransfer}s\n
time_redirect:  %{time_redirect}s\n
time_starttransfer:  %{time_starttransfer}s\n
----------\n
time_total:  %{time_total}s\n
```

### Docker Commands

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend

# Check service health
docker-compose ps

# Restart service
docker-compose restart backend

# Execute command in container
docker-compose exec backend bash

# View container stats
docker stats
```

### Database Commands

```bash
# Connect to PostgreSQL
docker exec -it investment_db psql -U postgres -d investment_db

# Backup database
docker exec investment_db pg_dump -U postgres investment_db > backup.sql

# Restore database
docker exec -i investment_db psql -U postgres investment_db < backup.sql
```

### Python Test Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest backend/tests/test_api_integration.py

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific markers
pytest -m "api"
pytest -m "unit"
pytest -m "integration"

# Run verbose
pytest -v

# Stop on first failure
pytest -x
```

---

## Success Criteria

### Backend API
- [ ] All health endpoints return 200
- [ ] Authentication flow works end-to-end
- [ ] All CRUD operations functional
- [ ] Real-time data updates working
- [ ] Error handling appropriate
- [ ] Response times acceptable
- [ ] No memory leaks
- [ ] No database connection issues

### Frontend
- [ ] All pages load without errors
- [ ] Navigation works smoothly
- [ ] Forms validate correctly
- [ ] API integration works
- [ ] Real-time updates display
- [ ] Responsive design functional
- [ ] No console errors
- [ ] Loading states appropriate

### Integration
- [ ] Database connectivity stable
- [ ] Cache hit rates acceptable
- [ ] External APIs responding
- [ ] WebSocket connections stable
- [ ] Background tasks running
- [ ] Monitoring data collecting

### Security
- [ ] Authentication enforced
- [ ] Authorization working
- [ ] Rate limiting active
- [ ] Input validation working
- [ ] CORS configured correctly
- [ ] No security vulnerabilities

### Performance
- [ ] Response times under thresholds
- [ ] Handles concurrent users
- [ ] Cache performance good
- [ ] Database queries optimized
- [ ] Memory usage stable
- [ ] No performance degradation

---

## Common Issues & Troubleshooting

### Issue: Services won't start

**Symptoms:** `docker-compose up` fails

**Solutions:**
```bash
# Check Docker is running
docker version

# Remove old containers
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Check logs for specific service
docker-compose logs backend
```

---

### Issue: Database connection failed

**Symptoms:** Backend can't connect to PostgreSQL

**Solutions:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Verify credentials
echo $DATABASE_URL

# Test connection manually
docker exec -it investment_db psql -U postgres -d investment_db
```

---

### Issue: Redis connection failed

**Symptoms:** Cache operations fail

**Solutions:**
```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker exec -it investment_cache redis-cli -a "${REDIS_PASSWORD}" PING

# Check Redis logs
docker-compose logs redis
```

---

### Issue: API returns 500 errors

**Symptoms:** Internal server errors

**Solutions:**
```bash
# Check backend logs
docker-compose logs -f backend

# Check for Python errors
docker-compose exec backend python -c "import backend.api.main"

# Restart backend
docker-compose restart backend
```

---

### Issue: Frontend won't load

**Symptoms:** Blank page or connection refused

**Solutions:**
```bash
# Check frontend logs
docker-compose logs frontend

# Verify frontend is running
docker-compose ps frontend

# Check if port 3000 is accessible
curl http://localhost:3000

# Rebuild frontend
docker-compose build frontend
docker-compose up -d frontend
```

---

### Issue: WebSocket connection fails

**Symptoms:** Can't establish WebSocket connection

**Solutions:**
```bash
# Check WebSocket endpoint
wscat -c "ws://localhost:8000/api/ws/stream?client_id=test"

# Check backend supports WebSocket
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8000/api/ws/stream

# Check firewall/proxy settings
```

---

### Issue: Authentication token issues

**Symptoms:** 401 Unauthorized errors with valid credentials

**Solutions:**
```bash
# Verify JWT_SECRET_KEY is set
echo $JWT_SECRET_KEY

# Check token expiration
# Decode JWT at https://jwt.io

# Generate new token
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"YourPassword"}'
```

---

## Test Report Template

```markdown
# Test Execution Report

**Date:** YYYY-MM-DD
**Tester:** Name
**Environment:** Development/Staging/Production
**Version:** X.Y.Z

## Summary
- Total Tests: X
- Passed: Y
- Failed: Z
- Skipped: W

## Test Results

### Backend API Tests
- Health Endpoints: PASS/FAIL
- Authentication: PASS/FAIL
- Stock Data: PASS/FAIL
- Analysis: PASS/FAIL
- Recommendations: PASS/FAIL
- Portfolio: PASS/FAIL

### Frontend Tests
- Dashboard: PASS/FAIL
- Login/Auth: PASS/FAIL
- Stock Analysis: PASS/FAIL
- Portfolio Management: PASS/FAIL

### Integration Tests
- Database: PASS/FAIL
- Cache: PASS/FAIL
- External APIs: PASS/FAIL
- WebSocket: PASS/FAIL

### Security Tests
- Authentication: PASS/FAIL
- Authorization: PASS/FAIL
- Rate Limiting: PASS/FAIL
- Input Validation: PASS/FAIL

### Performance Tests
- Load Testing: PASS/FAIL
- Response Times: PASS/FAIL
- Caching: PASS/FAIL

## Issues Found
1. Issue description
   - Severity: High/Medium/Low
   - Steps to reproduce
   - Expected vs Actual

## Recommendations
- List of improvements
- Priority items
- Follow-up actions

## Sign-off
- [ ] All critical tests passed
- [ ] All blockers resolved
- [ ] Ready for next phase
```

---

## Appendix: Test Data

### Sample Test Users

```json
[
  {
    "email": "admin@example.com",
    "password": "Admin123!",
    "role": "admin"
  },
  {
    "email": "user@example.com",
    "password": "User123!",
    "role": "free_user"
  },
  {
    "email": "premium@example.com",
    "password": "Premium123!",
    "role": "premium_user"
  }
]
```

### Sample Stock Symbols for Testing

- **Large Cap:** AAPL, MSFT, GOOGL, AMZN, META
- **Mid Cap:** PLTR, SNAP, PINS, SQ, SHOP
- **Small Cap:** FVRR, UPST, OPEN, ROOT, WISH
- **ETFs:** SPY, QQQ, DIA, IWM, VTI
- **Invalid:** INVALID, FAKE, XXXX

---

## Contact & Support

For issues with this test plan:
- Create issue in repository
- Contact development team
- Review documentation at `/docs`

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-02
**Next Review:** 2026-02-01
