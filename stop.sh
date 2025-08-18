#!/bin/bash
# Unified Stop Script for Investment Platform

echo "🛑 Stopping Investment Platform"
echo "=============================="

# Stop all docker-compose services
docker-compose down

# Option to clean volumes
if [ "$1" == "--clean" ]; then
    echo "🧹 Cleaning up volumes and data..."
    docker-compose down -v
    docker system prune -f
    echo "✅ Cleanup complete"
fi

echo "✅ Platform stopped"

# Show status
echo ""
echo "Container status:"
docker ps --filter "label=com.docker.compose.project=investment-platform" --format "table {{.Names}}\t{{.Status}}"

if [ "$1" != "--clean" ]; then
    echo ""
    echo "To completely clean up (including volumes), run: ./stop.sh --clean"
fi