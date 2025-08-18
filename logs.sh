#!/bin/bash
# Log Viewing Script for Investment Platform

SERVICE=$1
FOLLOW=${2:---follow}

echo "📋 Investment Platform Logs"
echo "=========================="

if [ -z "$SERVICE" ]; then
    echo "Showing logs for all services (Ctrl+C to stop)..."
    echo ""
    docker-compose logs $FOLLOW
else
    echo "Showing logs for $SERVICE service (Ctrl+C to stop)..."
    echo ""
    docker-compose logs $FOLLOW $SERVICE
fi