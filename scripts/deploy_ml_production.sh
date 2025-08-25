#!/bin/bash
# ML Production Deployment Script
# Deploys ML services in production mode

set -e

echo "============================================================"
echo "ML Production Deployment - $(date)"
echo "============================================================"

# Check if running in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must be run from project root directory"
    exit 1
fi

# Check if Docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

echo "✓ Environment checks passed"

# Create necessary directories
echo "Creating ML directories..."
mkdir -p backend/ml_models
mkdir -p backend/ml_logs
mkdir -p backend/ml_registry
mkdir -p data/training
mkdir -p data/predictions

# Ensure we have at least one model
if [ ! -f "backend/ml_models/sample_model.pkl" ]; then
    echo "Creating initial model..."
    python3 backend/ml/minimal_training.py
fi

echo "✓ Directories and models ready"

# Build and start ML services
echo "Building ML Docker images..."
docker-compose -f docker-compose.ml-production.yml build

echo "Starting ML services in production mode..."
docker-compose -f docker-compose.ml-production.yml up -d

echo "Waiting for services to start..."
sleep 10

# Health checks
echo "Performing health checks..."

# Check ML API
if curl -f -s http://localhost:8001/health > /dev/null; then
    echo "✓ ML API is healthy"
else
    echo "✗ ML API health check failed"
    docker-compose -f docker-compose.ml-production.yml logs ml-api
    exit 1
fi

# Check ML Monitoring
if curl -f -s http://localhost:8002/health > /dev/null; then
    echo "✓ ML Monitoring is healthy"
else
    echo "✗ ML Monitoring health check failed"
    docker-compose -f docker-compose.ml-production.yml logs ml-monitoring
    exit 1
fi

echo "============================================================"
echo "ML Production Deployment completed successfully!"
echo ""
echo "Services:"
echo "  - ML API: http://localhost:8001"
echo "  - ML API Docs: http://localhost:8001/docs"
echo "  - ML Monitoring: http://localhost:8002"
echo ""
echo "To view logs: docker-compose -f docker-compose.ml-production.yml logs"
echo "To stop services: docker-compose -f docker-compose.ml-production.yml down"
echo "============================================================"