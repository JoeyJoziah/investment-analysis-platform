#!/bin/bash
# GitHub Projects Board Sync - Convenience Wrapper
# Investment Analysis Platform
#
# Usage: ./board-sync.sh [command]
#
# Commands:
#   init     - Initialize project board
#   sync     - Full synchronization
#   report   - Generate status report
#   help     - Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Forward to main script
exec "$SCRIPT_DIR/scripts/github-board-sync.sh" "$@"
