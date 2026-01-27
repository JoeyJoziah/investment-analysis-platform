#!/bin/bash

# Minimal Airflow startup script
echo "Starting minimal Airflow setup..."

# Export required environment variables
export AIRFLOW_UID=$(id -u)
export AIRFLOW_GID=0

# Create necessary directories
mkdir -p data_pipelines/airflow/{dags,logs,plugins,config}

# Start only essential services
echo "Starting PostgreSQL and Redis..."
docker-compose -f docker-compose.airflow.yml up -d postgres redis

# Wait for services
sleep 10

# Initialize Airflow database (run in background)
echo "Initializing Airflow..."
docker-compose -f docker-compose.airflow.yml up airflow-init &

# Wait for initialization
sleep 30

# Start core Airflow services
echo "Starting Airflow services..."
docker-compose -f docker-compose.airflow.yml up -d airflow-webserver airflow-scheduler

# Wait for services to be ready
sleep 20

# Check status
echo ""
echo "Checking service status..."
docker-compose -f docker-compose.airflow.yml ps

echo ""
echo "Airflow should be accessible at:"
echo "  http://localhost:8080 (admin/airflow)"
echo ""
echo "To check logs:"
echo "  docker logs airflow-webserver"
echo "  docker logs airflow-scheduler"