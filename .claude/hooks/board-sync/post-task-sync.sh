#!/bin/bash
# Post-Task Board Sync Hook
# Called after task completion in Claude Code sessions
#
# Tasks represent meaningful units of work, so this triggers an actual sync.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Read stdin (hook input)
INPUT=$(cat)

# Trigger sync in background (non-blocking)
export BOARD_SYNC_TRIGGER="post-task"
"$SCRIPT_DIR/sync-boards.sh" --silent &

# Pass through input unchanged
echo "$INPUT"
