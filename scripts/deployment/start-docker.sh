#!/bin/bash

# Investment Analysis App - Docker Startup Script
# This script handles all Docker-related issues and starts the application

set -e

echo "================================================"
echo "Investment Analysis App - Docker Startup"
echo "================================================"

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker Desktop first."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to clean up old containers
cleanup() {
    echo "🧹 Cleaning up old containers..."
    docker compose down 2>/dev/null || true
    docker compose -f docker-compose.test.yml down 2>/dev/null || true
    docker compose -f docker-compose.fixed.yml down 2>/dev/null || true
}

# Function to pull base images
pull_images() {
    echo "📦 Pulling base Docker images..."
    docker pull python:3.11-slim || {
        echo "⚠️  Failed to pull Python image. Trying without credentials..."
        docker logout
        docker pull python:3.11-slim
    }
    docker pull node:18-alpine || true
    docker pull postgres:15-alpine || true
    docker pull redis:7-alpine || true
    echo "✅ Base images pulled"
}

# Function to build services
build_services() {
    echo "🔨 Building services..."
    
    # Check which docker-compose file to use
    if [ -f "docker-compose.test.yml" ]; then
        echo "Using test configuration..."
        docker compose -f docker-compose.test.yml build --no-cache backend
    elif [ -f "docker-compose.fixed.yml" ]; then
        echo "Using fixed configuration..."
        docker compose -f docker-compose.fixed.yml build --no-cache backend
    else
        echo "Using default configuration..."
        docker compose build --no-cache backend
    fi
    
    echo "✅ Services built"
}

# Function to start infrastructure
start_infrastructure() {
    echo "🚀 Starting infrastructure services..."
    
    if [ -f "docker-compose.test.yml" ]; then
        docker compose -f docker-compose.test.yml up -d postgres redis
    elif [ -f "docker-compose.fixed.yml" ]; then
        docker compose -f docker-compose.fixed.yml up -d postgres redis
    else
        docker compose up -d postgres redis
    fi
    
    echo "⏳ Waiting for database to be ready..."
    sleep 10
    
    # Check if database is ready
    for i in {1..30}; do
        if docker exec investment_db_test pg_isready -U postgres >/dev/null 2>&1 || \
           docker exec investment_db pg_isready -U postgres >/dev/null 2>&1; then
            echo "✅ Database is ready"
            break
        fi
        echo "Waiting for database... ($i/30)"
        sleep 2
    done
}

# Function to initialize database
init_database() {
    echo "🗄️ Initializing database..."
    
    # Create database if it doesn't exist
    docker exec investment_db_test psql -U postgres -c "CREATE DATABASE investment_db;" 2>/dev/null || \
    docker exec investment_db psql -U postgres -c "CREATE DATABASE investment_db;" 2>/dev/null || true
    
    echo "✅ Database initialized"
}

# Function to start application
start_application() {
    echo "🚀 Starting application services..."
    
    if [ -f "docker-compose.test.yml" ]; then
        docker compose -f docker-compose.test.yml up -d backend
        echo "✅ Application started with test configuration"
    elif [ -f "docker-compose.fixed.yml" ]; then
        docker compose -f docker-compose.fixed.yml up -d backend
        docker compose -f docker-compose.fixed.yml up -d celery_worker celery_beat 2>/dev/null || true
        docker compose -f docker-compose.fixed.yml up -d frontend 2>/dev/null || true
        echo "✅ Application started with fixed configuration"
    else
        docker compose up -d backend
        docker compose up -d celery_worker celery_beat 2>/dev/null || true
        docker compose up -d frontend 2>/dev/null || true
        echo "✅ Application started with default configuration"
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "================================================"
    echo "📊 Service Status"
    echo "================================================"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo "================================================"
    echo "🔗 Access Points"
    echo "================================================"
    echo "API:        http://localhost:8000"
    echo "API Docs:   http://localhost:8000/docs"
    echo "Frontend:   http://localhost:3000 (if running)"
    echo "Database:   localhost:5432"
    echo "Redis:      localhost:6379"
    echo ""
    echo "================================================"
    echo "📝 Useful Commands"
    echo "================================================"
    echo "View logs:       docker compose logs -f backend"
    echo "Stop services:   docker compose down"
    echo "Clean all:       docker compose down -v"
    echo "Shell access:    docker exec -it investment_api_test bash"
    echo ""
}

# Function to check API health
check_health() {
    echo "🏥 Checking API health..."
    sleep 5
    
    if curl -f http://localhost:8000/api/health >/dev/null 2>&1; then
        echo "✅ API is healthy"
    else
        echo "⚠️  API health check failed. Checking logs..."
        docker logs investment_api_test --tail 20 2>/dev/null || docker logs investment_api --tail 20
    fi
}

# Main execution
main() {
    echo "Starting Investment Analysis App..."
    echo ""
    
    check_docker
    cleanup
    pull_images
    build_services
    start_infrastructure
    init_database
    start_application
    check_health
    show_status
    
    echo "✅ Startup complete!"
    echo ""
    echo "To view real-time logs, run:"
    echo "  docker compose logs -f backend"
}

# Run main function
main "$@"