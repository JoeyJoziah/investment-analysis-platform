#!/bin/bash
# Start services

set -euo pipefail

MODE=${1:-dev}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Investment Analysis Platform in $MODE mode..."

# Detect OS - enable linux-monitoring profile only on Linux
# (node-exporter and cadvisor require Linux-specific mounts)
PROFILE_FLAG=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PROFILE_FLAG="--profile linux-monitoring"
    echo "Linux detected: enabling system monitoring services"
else
    echo "macOS detected: system monitoring services disabled (Linux-only)"
fi

if [ "$MODE" == "prod" ]; then
    docker compose -f docker-compose.yml -f docker-compose.prod.yml $PROFILE_FLAG up -d
elif [ "$MODE" == "test" ]; then
    docker compose -f docker-compose.yml -f docker-compose.test.yml $PROFILE_FLAG up -d
else
    docker compose -f docker-compose.yml -f docker-compose.dev.yml $PROFILE_FLAG up -d
fi

echo ""
echo "Services starting..."
echo "Frontend:     http://localhost:3000"
echo "Backend API:  http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo ""
echo "Use ./logs.sh to view logs"
echo "Use ./stop.sh to stop all services"

# Trigger board sync in background after services start
SYNC_SCRIPT="$SCRIPT_DIR/.claude/hooks/board-sync/sync-boards.sh"
if [ -x "$SYNC_SCRIPT" ]; then
    export BOARD_SYNC_TRIGGER="start-script"
    "$SYNC_SCRIPT" --silent &
    echo "[Board Sync] Triggered in background"
fi
