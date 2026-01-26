#!/bin/bash
# Board Sync Hook Script
# Triggers sync for both GitHub Projects and Notion boards
#
# Features:
# - Non-blocking (runs in background by default)
# - Respects debounce to prevent rapid consecutive syncs
# - Logs all sync operations
# - Silent mode for post-edit hooks
#
# Usage:
#   ./sync-boards.sh [options]
#   Options:
#     --foreground    Run in foreground (blocking)
#     --silent        Suppress output (for hooks)
#     --github-only   Only sync GitHub Projects
#     --notion-only   Only sync Notion
#     --force         Bypass debounce check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/.claude/logs"
LOCK_FILE="$PROJECT_ROOT/.claude/.board-sync.lock"
DEBOUNCE_FILE="$PROJECT_ROOT/.claude/.board-sync.last"
DEBOUNCE_SECONDS=30

# Ensure log directory exists
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/board-sync.log"

# Parse arguments
FOREGROUND=false
SILENT=false
GITHUB_ONLY=false
NOTION_ONLY=false
FORCE=false

for arg in "$@"; do
    case $arg in
        --foreground) FOREGROUND=true ;;
        --silent) SILENT=true ;;
        --github-only) GITHUB_ONLY=true ;;
        --notion-only) NOTION_ONLY=true ;;
        --force) FORCE=true ;;
    esac
done

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" >> "$LOG_FILE"
    if [ "$SILENT" = false ]; then
        echo "$msg" >&2
    fi
}

check_debounce() {
    if [ "$FORCE" = true ]; then
        return 0
    fi

    if [ -f "$DEBOUNCE_FILE" ]; then
        local last_sync
        last_sync=$(cat "$DEBOUNCE_FILE")
        local now
        now=$(date +%s)
        local diff=$((now - last_sync))

        if [ "$diff" -lt "$DEBOUNCE_SECONDS" ]; then
            log "Debounce: Skipping sync (last sync ${diff}s ago, threshold ${DEBOUNCE_SECONDS}s)"
            return 1
        fi
    fi
    return 0
}

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid
        pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log "Lock: Another sync process is running (PID: $pid)"
            return 1
        fi
        rm -f "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
    return 0
}

release_lock() {
    rm -f "$LOCK_FILE"
}

update_debounce() {
    date +%s > "$DEBOUNCE_FILE"
}

sync_github() {
    log "GitHub Sync: Starting..."

    local github_script="$PROJECT_ROOT/board-sync.sh"
    if [ -x "$github_script" ]; then
        if "$github_script" sync >> "$LOG_FILE" 2>&1; then
            log "GitHub Sync: Completed successfully"
            return 0
        else
            log "GitHub Sync: Failed (exit code: $?)"
            return 1
        fi
    else
        log "GitHub Sync: Script not found or not executable at $github_script"
        return 1
    fi
}

sync_notion() {
    log "Notion Sync: Starting..."

    local notion_script="$PROJECT_ROOT/notion-sync.sh"
    if [ -x "$notion_script" ]; then
        if "$notion_script" push >> "$LOG_FILE" 2>&1; then
            log "Notion Sync: Completed successfully"
            return 0
        else
            log "Notion Sync: Failed (exit code: $?)"
            return 1
        fi
    else
        log "Notion Sync: Script not found or not executable at $notion_script"
        return 1
    fi
}

run_sync() {
    if ! check_debounce; then
        return 0
    fi

    if ! acquire_lock; then
        return 1
    fi

    trap release_lock EXIT

    log "=== Board Sync Started ==="
    log "Trigger: ${BOARD_SYNC_TRIGGER:-manual}"

    local github_status=0
    local notion_status=0

    # Sync GitHub Projects
    if [ "$NOTION_ONLY" = false ]; then
        sync_github || github_status=$?
    fi

    # Sync Notion
    if [ "$GITHUB_ONLY" = false ]; then
        sync_notion || notion_status=$?
    fi

    update_debounce

    # Summary
    if [ "$github_status" -eq 0 ] && [ "$notion_status" -eq 0 ]; then
        log "=== Board Sync Completed Successfully ==="
    else
        log "=== Board Sync Completed with Errors (GitHub: $github_status, Notion: $notion_status) ==="
    fi

    return 0
}

# Main execution
if [ "$FOREGROUND" = true ]; then
    run_sync
else
    # Run in background, detached from terminal
    (
        export BOARD_SYNC_TRIGGER="${BOARD_SYNC_TRIGGER:-background}"
        run_sync
    ) </dev/null >/dev/null 2>&1 &

    if [ "$SILENT" = false ]; then
        echo "[Board Sync] Started in background (PID: $!)" >&2
    fi
fi
