#!/bin/bash
# Investment Analysis Platform - Setup Script
# This script sets up and starts the entire application

set -e

echo "=========================================="
echo "Investment Analysis Platform Setup"
echo "=========================================="

# Check for required tools
check_requirements() {
    echo "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    echo "All requirements met."
}

# Create .env file if it does not exist
setup_env() {
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            echo "Creating .env from .env.example..."
            cp .env.example .env
            
            # Generate random secrets
            SECRET_KEY=$(openssl rand -hex 32)
            JWT_SECRET=$(openssl rand -hex 32)
            DB_PASSWORD=$(openssl rand -hex 16)
            REDIS_PASSWORD=$(openssl rand -hex 16)
            
            # Update secrets in .env
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s/your_secret_key_here/$SECRET_KEY/" .env
                sed -i '' "s/your_jwt_secret_here/$JWT_SECRET/" .env
                sed -i '' "s/your_db_password/$DB_PASSWORD/" .env
                sed -i '' "s/your_redis_password/$REDIS_PASSWORD/" .env
            else
                sed -i "s/your_secret_key_here/$SECRET_KEY/" .env
                sed -i "s/your_jwt_secret_here/$JWT_SECRET/" .env
                sed -i "s/your_db_password/$DB_PASSWORD/" .env
                sed -i "s/your_redis_password/$REDIS_PASSWORD/" .env
            fi
            
            echo "Generated secure credentials in .env"
        else
            echo "ERROR: .env.example not found"
            exit 1
        fi
    else
        echo ".env file already exists"
    fi
}

# Build Docker images
build_images() {
    echo "Building Docker images..."
    docker compose build --no-cache
}

# Start services
start_services() {
    MODE=${1:-dev}
    echo "Starting services in $MODE mode..."
    
    if [ "$MODE" == "prod" ]; then
        docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    fi
}

# Wait for services to be healthy
wait_for_services() {
    echo "Waiting for services to be healthy..."
    
    # Wait for PostgreSQL
    echo "Waiting for PostgreSQL..."
    until docker exec investment_db pg_isready -U postgres > /dev/null 2>&1; do
        sleep 2
    done
    echo "PostgreSQL is ready"
    
    # Wait for Redis
    echo "Waiting for Redis..."
    until docker exec investment_cache redis-cli ping > /dev/null 2>&1; do
        sleep 2
    done
    echo "Redis is ready"
    
    # Wait for backend
    echo "Waiting for Backend API..."
    until curl -s http://localhost:8000/api/health > /dev/null 2>&1; do
        sleep 2
    done
    echo "Backend is ready"
    
    echo "All services are healthy!"
}

# Initialize database
init_database() {
    echo "Initializing database..."
    docker exec investment_api python -c "from backend.utils.database import init_db; import asyncio; asyncio.run(init_db())" 2>/dev/null || true
    echo "Database initialized"
}

# Show service URLs
show_urls() {
    echo ""
    echo "=========================================="
    echo "Application is ready!"
    echo "=========================================="
    echo ""
    echo "Frontend:        http://localhost:3000"
    echo "Backend API:     http://localhost:8000"
    echo "API Docs:        http://localhost:8000/docs"
    echo "Health Check:    http://localhost:8000/api/health"
    echo ""
    echo "To stop the application, run: ./stop.sh"
    echo "To view logs, run: ./logs.sh"
    echo ""
}

# Main
main() {
    MODE=${1:-dev}
    
    check_requirements
    setup_env
    build_images
    start_services $MODE
    wait_for_services
    init_database
    show_urls
}

main "$@"
