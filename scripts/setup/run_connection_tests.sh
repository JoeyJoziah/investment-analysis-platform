#!/bin/bash

echo "=== Investment Analysis App - Connection Tests ==="
echo "Testing all database and service connections..."
echo ""

# Check if we're in a Docker environment
if [ -f /.dockerenv ]; then
    echo "🐳 Running inside Docker container"
    ENVIRONMENT="docker"
else
    echo "🖥️  Running on host system"
    ENVIRONMENT="host"
fi

echo ""

# Install required Python packages if not present
echo "📦 Checking Python dependencies..."
pip install -q psycopg2-binary redis elasticsearch asyncpg sqlalchemy 2>/dev/null || {
    echo "⚠️  Some packages may be missing. Install with:"
    echo "pip install psycopg2-binary redis elasticsearch asyncpg sqlalchemy"
}

echo ""

# Run the appropriate test script
if [ "$ENVIRONMENT" = "docker" ]; then
    echo "🔧 Running Docker container tests..."
    python3 test_docker_connections.py
else
    echo "🔧 Running host system tests..."
    python3 test_services_quick.py
fi

echo ""

# Additional Docker-specific tests
if command -v docker &> /dev/null; then
    echo "🐳 Checking Docker container status..."
    echo "Running containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(postgres|redis|elasticsearch)" || echo "No relevant containers found"
    echo ""
fi

# Check if services are responding on expected ports
echo "🌐 Checking service ports..."

if nc -z localhost 5432 2>/dev/null; then
    echo "✅ PostgreSQL port 5432: Open"
else
    echo "❌ PostgreSQL port 5432: Closed"
fi

if nc -z localhost 6379 2>/dev/null; then
    echo "✅ Redis port 6379: Open"
else
    echo "❌ Redis port 6379: Closed"
fi

if nc -z localhost 9200 2>/dev/null; then
    echo "✅ Elasticsearch port 9200: Open"
else
    echo "❌ Elasticsearch port 9200: Closed"
fi

echo ""
echo "=== Connection Tests Complete ==="