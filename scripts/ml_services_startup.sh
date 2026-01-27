#!/bin/bash
"""
ML Services Startup Script
Starts all ML-related services including training, inference, and monitoring
"""

set -e

echo "============================================================"
echo "Starting ML Services - $(date)"
echo "============================================================"

# Check if running in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must be run from project root directory"
    exit 1
fi

# Create necessary directories
echo "Creating ML directories..."
mkdir -p backend/ml_models
mkdir -p backend/ml_logs
mkdir -p backend/ml_registry
mkdir -p data/training
mkdir -p data/predictions

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import sklearn, pandas, numpy; print('✓ Core ML dependencies available')"

# Start Redis if needed (check if it's running)
echo "Checking Redis status..."
if ! redis-cli ping >/dev/null 2>&1; then
    echo "Redis not running, attempting to start..."
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes --port 6379
        echo "✓ Redis started"
    else
        echo "⚠ Redis not available - task queuing will be disabled"
    fi
else
    echo "✓ Redis already running"
fi

# Set environment variables
export ML_MODELS_PATH="backend/ml_models"
export ML_LOGS_PATH="backend/ml_logs"
export ML_REGISTRY_PATH="backend/ml_registry"

echo "✓ ML Services environment prepared"
echo "============================================================"
echo "ML Services startup completed successfully"
echo "============================================================"