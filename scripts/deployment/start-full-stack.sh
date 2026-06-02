#!/bin/bash

echo "Starting Investment Analysis Platform (Resource Optimized)..."

# Stage 1: Core Infrastructure
echo "Stage 1: Starting core services..."
docker-compose up -d postgres redis
echo "Waiting for databases to initialize..."
sleep 15

# Stage 2: Backend Services
echo "Stage 2: Starting backend API..."
docker-compose up -d backend
sleep 10

# Stage 3: Task Processing (Lower Priority)
echo "Stage 3: Starting task workers..."
docker-compose up -d celery_worker celery_beat
sleep 5

# Stage 4: Data Pipeline (Optional)
echo "Stage 4: Starting Airflow..."
docker-compose up -d airflow
sleep 10

# Stage 5: Monitoring (Optional)
echo "Stage 5: Starting monitoring..."
docker-compose up -d prometheus grafana

# Stage 6: Frontend
echo "Stage 6: Starting frontend..."
docker-compose up -d frontend nginx

echo "All services started! Checking status..."
docker-compose ps