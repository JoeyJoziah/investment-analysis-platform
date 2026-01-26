#!/bin/bash
# Post-Edit Board Sync Hook
# Called after file edits in Claude Code sessions
#
# This hook is lightweight and non-blocking to avoid slowing down edits.
# It queues a sync rather than running immediately.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
QUEUE_FILE="$PROJECT_ROOT/.claude/.board-sync-queue"

# Read stdin (hook input)
INPUT=$(cat)

# Queue a sync (don't run immediately to avoid slowing edits)
# The sync will run on session-end or task completion
echo "$(date +%s):post-edit:$PWD" >> "$QUEUE_FILE" 2>/dev/null || true

# Pass through input unchanged
echo "$INPUT"
