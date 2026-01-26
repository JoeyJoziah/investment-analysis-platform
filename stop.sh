#!/bin/bash
# Stop all services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping all services..."
docker compose down
echo "All services stopped."

# Trigger board sync in background after services stop
SYNC_SCRIPT="$SCRIPT_DIR/.claude/hooks/board-sync/sync-boards.sh"
if [ -x "$SYNC_SCRIPT" ]; then
    export BOARD_SYNC_TRIGGER="stop-script"
    "$SYNC_SCRIPT" --silent &
    echo "[Board Sync] Triggered in background"
fi
