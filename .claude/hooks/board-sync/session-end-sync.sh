#!/bin/bash
# Session-End Board Sync Hook
# Called when Claude Code session ends
#
# This is the primary sync point - ensures all changes are synced before session closes.
# Runs in foreground to ensure completion before session terminates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
QUEUE_FILE="$PROJECT_ROOT/.claude/.board-sync-queue"

# Read stdin (hook input)
INPUT=$(cat)

# Check if there were any queued syncs
if [ -f "$QUEUE_FILE" ]; then
    QUEUED_COUNT=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
    if [ "$QUEUED_COUNT" -gt 0 ]; then
        echo "[Board Sync] Processing $QUEUED_COUNT queued sync requests..." >&2
    fi
    rm -f "$QUEUE_FILE"
fi

# Run sync in foreground with force flag to bypass debounce
# This ensures session-end always syncs
export BOARD_SYNC_TRIGGER="session-end"
"$SCRIPT_DIR/sync-boards.sh" --foreground --force 2>&1 | while read -r line; do
    echo "[Board Sync] $line" >&2
done

echo "[Board Sync] Session-end sync completed" >&2

# Pass through input unchanged
echo "$INPUT"
