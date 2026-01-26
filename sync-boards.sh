#!/bin/bash
# Project Board Sync - CLI Interface
# Manual trigger for syncing GitHub Projects and Notion boards
#
# Usage:
#   ./sync-boards.sh [command] [options]
#
# Commands:
#   sync        Full sync (GitHub + Notion) - default
#   github      Sync GitHub Projects only
#   notion      Sync Notion only
#   status      Show sync status and logs
#   logs        View recent sync logs
#   help        Show this help
#
# Options:
#   --force     Bypass debounce and run immediately
#   --verbose   Show detailed output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/.claude/hooks/board-sync/sync-boards.sh"
LOG_DIR="$SCRIPT_DIR/.claude/logs"
LOG_FILE="$LOG_DIR/board-sync.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo -e "${CYAN}Project Board Sync - Investment Analysis Platform${NC}"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  sync        Full sync (GitHub Projects + Notion) - default"
    echo "  github      Sync GitHub Projects only"
    echo "  notion      Sync Notion only"
    echo "  status      Show sync status and recent activity"
    echo "  logs        View recent sync logs"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  --force     Bypass debounce timer and run immediately"
    echo "  --verbose   Show detailed output during sync"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full sync"
    echo "  $0 github --force     # Force GitHub sync"
    echo "  $0 status             # Show sync status"
    echo ""
    echo "Automatic Triggers:"
    echo "  - Claude Code: post-edit, task-complete, session-end"
    echo "  - Git: post-commit, post-merge, post-checkout"
    echo "  - Scripts: start.sh, stop.sh, run-tests.sh"
}

show_status() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Board Sync Status${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Check lock file
    LOCK_FILE="$SCRIPT_DIR/.claude/.board-sync.lock"
    if [ -f "$LOCK_FILE" ]; then
        PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Status: Sync in progress (PID: $PID)${NC}"
        else
            echo -e "${GREEN}Status: Idle (stale lock file)${NC}"
            rm -f "$LOCK_FILE"
        fi
    else
        echo -e "${GREEN}Status: Idle${NC}"
    fi
    echo ""

    # Check debounce
    DEBOUNCE_FILE="$SCRIPT_DIR/.claude/.board-sync.last"
    if [ -f "$DEBOUNCE_FILE" ]; then
        LAST_SYNC=$(cat "$DEBOUNCE_FILE")
        NOW=$(date +%s)
        DIFF=$((NOW - LAST_SYNC))
        LAST_DATE=$(date -r "$LAST_SYNC" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d "@$LAST_SYNC" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "unknown")
        echo -e "${BLUE}Last Sync:${NC} $LAST_DATE (${DIFF}s ago)"
    else
        echo -e "${BLUE}Last Sync:${NC} Never"
    fi
    echo ""

    # Check queue
    QUEUE_FILE="$SCRIPT_DIR/.claude/.board-sync-queue"
    if [ -f "$QUEUE_FILE" ]; then
        QUEUE_COUNT=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
        echo -e "${BLUE}Queued Syncs:${NC} $QUEUE_COUNT"
    else
        echo -e "${BLUE}Queued Syncs:${NC} 0"
    fi
    echo ""

    # Check scripts availability
    echo -e "${BLUE}Scripts:${NC}"
    if [ -x "$SCRIPT_DIR/board-sync.sh" ]; then
        echo -e "  GitHub Sync: ${GREEN}Available${NC}"
    else
        echo -e "  GitHub Sync: ${RED}Not Found${NC}"
    fi

    if [ -x "$SCRIPT_DIR/notion-sync.sh" ]; then
        echo -e "  Notion Sync: ${GREEN}Available${NC}"
    else
        echo -e "  Notion Sync: ${RED}Not Found${NC}"
    fi
    echo ""

    # Recent log entries
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Recent Activity (last 10 entries):${NC}"
        tail -10 "$LOG_FILE" | while read -r line; do
            echo "  $line"
        done
    fi

    echo ""
    echo -e "${CYAN}========================================${NC}"
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${CYAN}Board Sync Logs${NC}"
        echo -e "${CYAN}File: $LOG_FILE${NC}"
        echo ""
        tail -50 "$LOG_FILE"
    else
        echo -e "${YELLOW}No log file found at $LOG_FILE${NC}"
    fi
}

# Parse command
COMMAND="${1:-sync}"
shift || true

# Parse options
FORCE=""
VERBOSE=""
SYNC_FLAGS=""

for arg in "$@"; do
    case $arg in
        --force) FORCE="--force" ;;
        --verbose) VERBOSE="--foreground" ;;
    esac
done

# Execute command
case "$COMMAND" in
    sync|"")
        echo -e "${BLUE}Starting full board sync...${NC}"
        export BOARD_SYNC_TRIGGER="manual-cli"
        if [ -x "$SYNC_SCRIPT" ]; then
            "$SYNC_SCRIPT" $VERBOSE $FORCE
        else
            echo -e "${RED}Error: Sync script not found at $SYNC_SCRIPT${NC}"
            exit 1
        fi
        ;;
    github)
        echo -e "${BLUE}Starting GitHub Projects sync...${NC}"
        export BOARD_SYNC_TRIGGER="manual-cli-github"
        if [ -x "$SYNC_SCRIPT" ]; then
            "$SYNC_SCRIPT" --github-only $VERBOSE $FORCE
        else
            echo -e "${RED}Error: Sync script not found${NC}"
            exit 1
        fi
        ;;
    notion)
        echo -e "${BLUE}Starting Notion sync...${NC}"
        export BOARD_SYNC_TRIGGER="manual-cli-notion"
        if [ -x "$SYNC_SCRIPT" ]; then
            "$SYNC_SCRIPT" --notion-only $VERBOSE $FORCE
        else
            echo -e "${RED}Error: Sync script not found${NC}"
            exit 1
        fi
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
