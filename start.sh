#!/bin/bash
# Unified Start Script for Investment Platform

set -e

# Determine environment (default to dev)
ENV=${1:-dev}

echo "🚀 Starting Investment Platform in $ENV mode"
echo "==========================================="

# Validate environment
if [[ ! "$ENV" =~ ^(dev|prod|test)$ ]]; then
    echo "❌ Invalid environment: $ENV"
    echo "Usage: ./start.sh [dev|prod|test]"
    exit 1
fi

# Check .env file
if [ ! -f .env ]; then
    echo "❌ .env file not found. Run ./setup.sh first"
    exit 1
fi

# Start services based on environment
case $ENV in
    dev)
        echo "🔧 Starting development environment..."
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
        echo ""
        echo "✅ Development environment started!"
        echo ""
        echo "Services available at:"
        echo "  Frontend:    http://localhost:3000"
        echo "  Backend API: http://localhost:8000"
        echo "  API Docs:    http://localhost:8000/docs"
        echo "  PgAdmin:     http://localhost:5050"
        echo "  Redis UI:    http://localhost:8081"
        echo "  Flower:      http://localhost:5555"
        echo ""
        echo "Run './logs.sh' to view logs"
        ;;
        
    prod)
        echo "🚀 Starting production environment..."
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
        echo ""
        echo "✅ Production environment started!"
        echo ""
        echo "Services available at:"
        echo "  Application: http://localhost"
        echo "  Monitoring:  http://localhost:3001"
        echo ""
        echo "Run './logs.sh' to view logs"
        ;;
        
    test)
        echo "🧪 Starting test environment..."
        docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
        echo ""
        echo "✅ Test environment started!"
        echo "Running tests..."
        docker-compose exec -T backend pytest --cov=backend --cov-report=html
        echo ""
        echo "Test results available in htmlcov/index.html"
        ;;
esac

# Health check
echo ""
echo "🏥 Running health check..."
sleep 5
curl -s http://localhost:8000/api/health > /dev/null 2>&1 && echo "✅ Backend is healthy" || echo "⚠️ Backend health check failed"

echo ""
echo "To stop the platform, run: ./stop.sh"