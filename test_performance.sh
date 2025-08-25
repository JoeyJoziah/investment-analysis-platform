#!/bin/bash
# Quick performance test for the investment platform

echo "🔍 Running performance tests..."

# Test API response time
test_api() {
    echo -n "API Health Check: "
    time curl -s http://localhost:8000/api/health > /dev/null
}

# Test database query
test_db() {
    echo -n "Database Query: "
    time PGPASSWORD=postgres psql -h localhost -U postgres -d investment_db -c "SELECT COUNT(*) FROM stocks;" > /dev/null 2>&1
}

# Test Redis
test_redis() {
    echo -n "Redis Ping: "
    time redis-cli ping > /dev/null
}

# Test file search
test_search() {
    echo -n "File Search (ripgrep): "
    time rg --files | wc -l
}

# Run tests
test_api
test_db
test_redis
test_search

echo ""
echo "✅ Performance tests complete"
