#!/bin/bash

################################################################################
# Quick Test Execution Script for Investment Analysis Platform
# This script runs automated tests to verify system functionality
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
TOTAL=0

# Base URL
BASE_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"

# Test result tracking
declare -a FAILED_TESTS

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_test() {
    echo -e "${YELLOW}Testing:${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    ((PASSED++))
    ((TOTAL++))
}

print_failure() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    echo -e "${RED}       $2${NC}"
    FAILED_TESTS+=("$1: $2")
    ((FAILED++))
    ((TOTAL++))
}

check_response() {
    local url=$1
    local expected_code=${2:-200}
    local description=$3

    print_test "$description"

    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")

    if [ "$response" -eq "$expected_code" ]; then
        print_success "$description (HTTP $response)"
    else
        print_failure "$description" "Expected HTTP $expected_code, got HTTP $response"
    fi
}

check_json_field() {
    local url=$1
    local field=$2
    local description=$3

    print_test "$description"

    response=$(curl -s "$url")

    if echo "$response" | jq -e ".$field" > /dev/null 2>&1; then
        print_success "$description"
    else
        print_failure "$description" "Field '$field' not found in response"
    fi
}

check_service_health() {
    local service=$1
    local description=$2

    print_test "$description"

    if docker-compose ps | grep -q "$service.*Up"; then
        print_success "$description"
    else
        print_failure "$description" "Service $service is not running"
    fi
}

################################################################################
# Pre-Flight Checks
################################################################################

print_header "PRE-FLIGHT CHECKS"

# Check if Docker is running
print_test "Docker daemon"
if docker info > /dev/null 2>&1; then
    print_success "Docker is running"
else
    print_failure "Docker is running" "Docker daemon is not running"
    exit 1
fi

# Check if docker-compose is available
print_test "Docker Compose"
if command -v docker-compose > /dev/null 2>&1; then
    print_success "Docker Compose is available"
else
    print_failure "Docker Compose is available" "docker-compose not found"
    exit 1
fi

# Check if services are running
print_header "SERVICE HEALTH CHECKS"

check_service_health "investment_db" "PostgreSQL Database"
check_service_health "investment_cache" "Redis Cache"
check_service_health "investment_api" "Backend API"
check_service_health "investment_frontend" "Frontend Web"

################################################################################
# Backend API Health Tests
################################################################################

print_header "BACKEND API - HEALTH ENDPOINTS"

check_response "$BASE_URL/api/health" 200 "Basic health check"
check_json_field "$BASE_URL/api/health" "status" "Health status field exists"
check_json_field "$BASE_URL/api/health" "version" "Version field exists"

check_response "$BASE_URL/api/health/readiness" 200 "Readiness check"
check_json_field "$BASE_URL/api/health/readiness" "checks.database" "Database readiness check"
check_json_field "$BASE_URL/api/health/readiness" "checks.cache" "Cache readiness check"

check_response "$BASE_URL/api/health/liveness" 200 "Liveness probe"
check_response "$BASE_URL/api/health/metrics" 200 "System metrics"

################################################################################
# Authentication Tests
################################################################################

print_header "AUTHENTICATION ENDPOINTS"

# Test user registration
print_test "User registration"
TIMESTAMP=$(date +%s)
TEST_EMAIL="test${TIMESTAMP}@example.com"

REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/api/auth/register" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$TEST_EMAIL\",
    \"password\": \"SecurePass123!\",
    \"full_name\": \"Test User\"
  }")

if echo "$REGISTER_RESPONSE" | jq -e '.access_token' > /dev/null 2>&1; then
    TOKEN=$(echo "$REGISTER_RESPONSE" | jq -r '.access_token')
    print_success "User registration (received JWT token)"
else
    print_failure "User registration" "No access_token in response"
    TOKEN=""
fi

# Test login
print_test "User login"
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$TEST_EMAIL\",
    \"password\": \"SecurePass123!\"
  }")

if echo "$LOGIN_RESPONSE" | jq -e '.access_token' > /dev/null 2>&1; then
    print_success "User login (received JWT token)"
else
    print_failure "User login" "No access_token in response"
fi

# Test get current user (with token)
if [ -n "$TOKEN" ]; then
    print_test "Get current user (authenticated)"
    ME_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/auth/me")

    if echo "$ME_RESPONSE" | jq -e '.email' > /dev/null 2>&1; then
        print_success "Get current user"
    else
        print_failure "Get current user" "No email in response"
    fi
fi

# Test unauthorized access
print_test "Protected endpoint without token (should fail)"
UNAUTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/portfolio")

if [ "$UNAUTH_RESPONSE" -eq 401 ]; then
    print_success "Unauthorized access blocked (HTTP 401)"
else
    print_failure "Unauthorized access blocked" "Expected HTTP 401, got HTTP $UNAUTH_RESPONSE"
fi

################################################################################
# Stock Data Tests
################################################################################

print_header "STOCK DATA ENDPOINTS"

check_response "$BASE_URL/api/stocks?limit=10" 200 "Get stock list"
check_response "$BASE_URL/api/stocks/AAPL" 200 "Get stock by symbol (AAPL)"
check_response "$BASE_URL/api/stocks/INVALID" 404 "Invalid stock symbol returns 404"
check_response "$BASE_URL/api/stocks/AAPL/history?period=1M" 200 "Get stock price history"
check_response "$BASE_URL/api/stocks/search?query=apple" 200 "Search stocks"

################################################################################
# Analysis Tests
################################################################################

print_header "ANALYSIS ENDPOINTS"

check_response "$BASE_URL/api/analysis/AAPL" 200 "Get comprehensive analysis"
check_response "$BASE_URL/api/analysis/AAPL/technical" 200 "Get technical analysis"
check_response "$BASE_URL/api/analysis/AAPL/fundamental" 200 "Get fundamental analysis"
check_response "$BASE_URL/api/analysis/AAPL/sentiment" 200 "Get sentiment analysis"

################################################################################
# Recommendations Tests
################################################################################

print_header "RECOMMENDATIONS ENDPOINTS"

if [ -n "$TOKEN" ]; then
    print_test "Get daily recommendations (authenticated)"
    RECS_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/recommendations")

    if echo "$RECS_RESPONSE" | jq -e '.recommendations' > /dev/null 2>&1; then
        print_success "Get daily recommendations"
    else
        print_failure "Get daily recommendations" "No recommendations field in response"
    fi
fi

# Test without auth
print_test "Recommendations without auth (should fail)"
RECS_UNAUTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/recommendations")

if [ "$RECS_UNAUTH" -eq 401 ]; then
    print_success "Recommendations require authentication (HTTP 401)"
else
    print_failure "Recommendations require authentication" "Expected HTTP 401, got HTTP $RECS_UNAUTH"
fi

################################################################################
# Portfolio Tests
################################################################################

print_header "PORTFOLIO ENDPOINTS"

if [ -n "$TOKEN" ]; then
    # Create portfolio
    print_test "Create portfolio"
    CREATE_PORTFOLIO=$(curl -s -X POST "$BASE_URL/api/portfolio" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Test Portfolio",
        "description": "Automated test portfolio",
        "initial_balance": 10000
      }')

    if echo "$CREATE_PORTFOLIO" | jq -e '.id' > /dev/null 2>&1; then
        PORTFOLIO_ID=$(echo "$CREATE_PORTFOLIO" | jq -r '.id')
        print_success "Create portfolio (ID: $PORTFOLIO_ID)"

        # Get portfolios
        print_test "Get user portfolios"
        GET_PORTFOLIOS=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/portfolio")

        if echo "$GET_PORTFOLIOS" | jq -e '.[0].id' > /dev/null 2>&1; then
            print_success "Get user portfolios"
        else
            print_failure "Get user portfolios" "No portfolios in response"
        fi

        # Get portfolio details
        print_test "Get portfolio details"
        GET_PORTFOLIO=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/portfolio/$PORTFOLIO_ID")

        if echo "$GET_PORTFOLIO" | jq -e '.name' > /dev/null 2>&1; then
            print_success "Get portfolio details"
        else
            print_failure "Get portfolio details" "No portfolio data in response"
        fi
    else
        print_failure "Create portfolio" "No portfolio ID in response"
    fi
fi

################################################################################
# WebSocket Tests (Basic)
################################################################################

print_header "WEBSOCKET CONNECTIVITY"

print_test "WebSocket endpoint availability"
# Note: This is a basic check. Full WebSocket testing requires a WebSocket client
WS_CHECK=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  "$BASE_URL/api/ws/stream?client_id=test")

# WebSocket upgrade returns various codes, just check it's not a complete failure
if [ "$WS_CHECK" -ge 200 ] && [ "$WS_CHECK" -lt 500 ]; then
    print_success "WebSocket endpoint accessible"
else
    print_failure "WebSocket endpoint accessible" "HTTP $WS_CHECK returned"
fi

################################################################################
# Database Integration Tests
################################################################################

print_header "DATABASE INTEGRATION"

print_test "PostgreSQL connection"
DB_TEST=$(docker exec investment_db psql -U postgres -d investment_db -c "SELECT 1;" 2>&1)

if echo "$DB_TEST" | grep -q "1 row"; then
    print_success "PostgreSQL connection"
else
    print_failure "PostgreSQL connection" "Cannot connect to database"
fi

print_test "TimescaleDB extension"
TS_CHECK=$(docker exec investment_db psql -U postgres -d investment_db \
  -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb';" -t 2>&1)

if [ "$(echo $TS_CHECK | tr -d '[:space:]')" -eq "1" ]; then
    print_success "TimescaleDB extension enabled"
else
    print_failure "TimescaleDB extension enabled" "TimescaleDB not found"
fi

print_test "Required tables exist"
TABLE_COUNT=$(docker exec investment_db psql -U postgres -d investment_db \
  -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" -t 2>&1)

if [ "$(echo $TABLE_COUNT | tr -d '[:space:]')" -gt "0" ]; then
    print_success "Database tables exist (count: $(echo $TABLE_COUNT | tr -d '[:space:]'))"
else
    print_failure "Database tables exist" "No tables found"
fi

################################################################################
# Redis Integration Tests
################################################################################

print_header "REDIS CACHE INTEGRATION"

print_test "Redis connection"
REDIS_TEST=$(docker exec investment_cache redis-cli -a "${REDIS_PASSWORD:-redis_password}" PING 2>&1)

if echo "$REDIS_TEST" | grep -q "PONG"; then
    print_success "Redis connection"
else
    print_failure "Redis connection" "Redis did not respond with PONG"
fi

print_test "Redis memory usage"
REDIS_MEMORY=$(docker exec investment_cache redis-cli -a "${REDIS_PASSWORD:-redis_password}" \
  INFO memory 2>&1 | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')

if [ -n "$REDIS_MEMORY" ]; then
    print_success "Redis memory usage: $REDIS_MEMORY"
else
    print_failure "Redis memory usage" "Cannot get memory info"
fi

################################################################################
# Frontend Tests (Basic)
################################################################################

print_header "FRONTEND ACCESSIBILITY"

check_response "$FRONTEND_URL" 200 "Frontend homepage loads"
check_response "$FRONTEND_URL/static/js/main" 200 "Frontend JavaScript bundle"

################################################################################
# Performance Checks
################################################################################

print_header "BASIC PERFORMANCE CHECKS"

print_test "Health endpoint response time"
HEALTH_TIME=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/api/health")

if [ "$(echo "$HEALTH_TIME < 1.0" | bc)" -eq 1 ]; then
    print_success "Health endpoint response time: ${HEALTH_TIME}s (< 1s)"
else
    print_failure "Health endpoint response time" "Took ${HEALTH_TIME}s (should be < 1s)"
fi

print_test "Stock data endpoint response time"
STOCK_TIME=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/api/stocks/AAPL")

if [ "$(echo "$STOCK_TIME < 2.0" | bc)" -eq 1 ]; then
    print_success "Stock data response time: ${STOCK_TIME}s (< 2s)"
else
    print_failure "Stock data response time" "Took ${STOCK_TIME}s (should be < 2s)"
fi

################################################################################
# Security Checks
################################################################################

print_header "BASIC SECURITY CHECKS"

print_test "CORS headers present"
CORS_HEADERS=$(curl -s -I "$BASE_URL/api/health" | grep -i "access-control")

if [ -n "$CORS_HEADERS" ]; then
    print_success "CORS headers present"
else
    print_failure "CORS headers present" "No CORS headers found"
fi

print_test "Security headers present"
SECURITY_HEADERS=$(curl -s -I "$BASE_URL/api/health" | grep -iE "x-frame-options|x-content-type-options")

if [ -n "$SECURITY_HEADERS" ]; then
    print_success "Security headers present"
else
    echo -e "${YELLOW}⚠ WARNING:${NC} Security headers not found (acceptable in dev)"
    ((TOTAL++))
fi

################################################################################
# Test Summary
################################################################################

print_header "TEST SUMMARY"

echo -e "Total Tests:  $TOTAL"
echo -e "${GREEN}Passed:       $PASSED${NC}"
echo -e "${RED}Failed:       $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed Tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}✗${NC} $test"
    done
fi

echo -e "\n${BLUE}========================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    exit 1
fi
