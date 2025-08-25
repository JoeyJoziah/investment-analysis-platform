#!/bin/bash

# Investment Analysis Platform - Backend Startup Script
# This script properly configures the environment and starts the backend

# Set the project root directory
PROJECT_ROOT="/mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/7b51113a393465a37d4f1fda36b4d190088ac69ea8d5cf2f90400b3c14148ad3"

# Export PYTHONPATH to include project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Load environment variables
if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(cat "${PROJECT_ROOT}/.env" | grep -v '^#' | xargs)
fi

# Navigate to backend directory
cd "${PROJECT_ROOT}/backend"

echo "======================================"
echo "Starting Investment Analysis Backend"
echo "======================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Python Path: ${PYTHONPATH}"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, sqlalchemy, redis" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Some dependencies may be missing"
fi

# Start the backend with uvicorn
echo "Starting FastAPI backend on http://localhost:8000"
echo "API Documentation will be available at http://localhost:8000/docs"
echo ""

# Run with proper module path
exec python3 -m uvicorn backend.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info