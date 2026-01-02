#!/bin/bash
# Start services

MODE=${1:-dev}

echo "Starting Investment Analysis Platform in $MODE mode..."

if [ "$MODE" == "prod" ]; then
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
elif [ "$MODE" == "test" ]; then
    docker compose -f docker-compose.yml -f docker-compose.test.yml up -d
else
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
fi

echo ""
echo "Services starting..."
echo "Frontend:     http://localhost:3000"
echo "Backend API:  http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo ""
echo "Use ./logs.sh to view logs"
echo "Use ./stop.sh to stop all services"
