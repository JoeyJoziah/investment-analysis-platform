#!/bin/bash
# Start ML API Server Script
# Starts the ML inference API on port 8001

set -e

echo "============================================================"
echo "Starting ML API Server - $(date)"
echo "============================================================"

# Check if running in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must be run from project root directory"
    exit 1
fi

# Check if models exist
if [ ! -d "backend/ml_models" ]; then
    echo "Creating models directory..."
    mkdir -p backend/ml_models
fi

# Check if sample model exists, create if not
if [ ! -f "backend/ml_models/sample_model.pkl" ]; then
    echo "No models found, creating sample model..."
    python3 backend/ml/minimal_training.py
fi

# Check dependencies
echo "Checking API dependencies..."
python3 -c "import fastapi, uvicorn, pandas, numpy, sklearn; print('✓ API dependencies available')"

# Set environment variables
export ML_MODELS_PATH="backend/ml_models"
export ML_LOGS_PATH="backend/ml_logs"

echo "Starting ML API server on port 8001..."
echo "API Documentation: http://localhost:8001/docs"
echo "Health Check: http://localhost:8001/health"
echo "Models List: http://localhost:8001/models"

# Start the server
python3 backend/ml/ml_api_server.py