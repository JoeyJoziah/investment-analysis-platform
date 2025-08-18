#!/bin/bash

echo "Starting Investment Analysis Platform (Resource Optimized)..."

# Stage 1: Core Infrastructure
echo "Stage 1: Starting core services..."
docker-compose up -d postgres redis
echo "Waiting for databases to initialize..."
sleep 15

# Stage 2: Search and Cache
echo "Stage 2: Starting Elasticsearch..."
docker-compose up -d elasticsearch
sleep 10

# Stage 3: Backend Services
echo "Stage 3: Starting backend API..."
docker-compose up -d backend
sleep 10

# Stage 4: Task Processing (Lower Priority)
echo "Stage 4: Starting task workers..."
docker-compose up -d celery_worker celery_beat
sleep 5

# Stage 5: Data Pipeline (Optional)
echo "Stage 5: Starting Airflow..."
docker-compose up -d airflow
sleep 10

# Stage 6: Monitoring (Optional)
echo "Stage 6: Starting monitoring..."
docker-compose up -d prometheus grafana

# Stage 7: Frontend
echo "Stage 7: Starting frontend..."
docker-compose up -d frontend nginx

echo "All services started! Checking status..."
docker-compose ps