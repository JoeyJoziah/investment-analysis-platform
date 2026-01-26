#!/bin/bash
# =============================================================================
# Unified Sync Coordinator - Wrapper Script
# Investment Analysis Platform
# =============================================================================
# Synchronizes GitHub Projects, Issues, and Notion in one command
#
# Usage:
#   ./sync-all.sh                    # Full bidirectional sync
#   ./sync-all.sh --github-only      # Sync to GitHub only
#   ./sync-all.sh --notion-only      # Sync to Notion only
#   ./sync-all.sh --dry-run          # Preview without applying changes
#   ./sync-all.sh status             # Show sync status
#   ./sync-all.sh diff               # Show differences
# =============================================================================

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/.sync_logs"
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
MAX_RETRIES=3
RETRY_DELAY=5
PYTHON_CMD="${PYTHON_CMD:-python3}"

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        INFO)  color="$BLUE" ;;
        SUCCESS) color="$GREEN" ;;
        WARN)  color="$YELLOW" ;;
        ERROR) color="$RED" ;;
        *)     color="$NC" ;;
    esac

    echo -e "${color}[$timestamp] [$level]${NC} $message" | tee -a "$LOG_FILE"
}

log_info() { log INFO "$@"; }
log_success() { log SUCCESS "$@"; }
log_warn() { log WARN "$@"; }
log_error() { log ERROR "$@"; }

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
    # Keep only last 20 log files
    local old_files
    old_files=$(find "$LOG_DIR" -name "sync_*.log" -type f 2>/dev/null | wc -l)
    if [[ "$old_files" -gt 20 ]]; then
        (cd "$LOG_DIR" && ls -t sync_*.log 2>/dev/null | tail -n +21 | xargs rm -f 2>/dev/null) || true
    fi
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_python() {
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        log_error "Python not found. Please install Python 3.8+"
        exit 1
    fi

    local version
    version=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    local major=${version%.*}
    local minor=${version#*.}

    if [[ "$major" -lt 3 ]] || [[ "$major" -eq 3 && "$minor" -lt 8 ]]; then
        log_error "Python 3.8+ required. Found: $version"
        exit 1
    fi

    log_info "Python version: $version"
}

check_gh_cli() {
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) not installed"
        echo "  Install with: brew install gh"
        exit 1
    fi

    if ! gh auth status &> /dev/null; then
        log_error "Not authenticated with GitHub CLI"
        echo "  Run: gh auth login"
        exit 1
    fi

    log_info "GitHub CLI: authenticated"
}

check_notion_api() {
    local api_key_path="$HOME/.config/notion/api_key"

    if [[ -f "$api_key_path" ]]; then
        log_info "Notion API key: configured"
        return 0
    else
        log_warn "Notion API key not configured"
        echo "  Configure at: $api_key_path"
        echo "  Notion sync will be disabled"
        return 1
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check if requests is installed (for Notion)
    if ! "$PYTHON_CMD" -c "import requests" 2>/dev/null; then
        log_warn "Python 'requests' library not installed"
        echo "  Install with: pip install requests"
        echo "  Notion sync will be disabled"
    fi

    # Check if pyyaml is installed (for config parsing)
    if ! "$PYTHON_CMD" -c "import yaml" 2>/dev/null; then
        log_warn "Python 'pyyaml' library not installed (optional)"
        echo "  Install with: pip install pyyaml"
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    check_python
    check_gh_cli
    check_notion_api || true  # Don't fail if Notion not configured
    check_dependencies
    log_success "Prerequisites check complete"
}

# =============================================================================
# Sync Functions
# =============================================================================

run_unified_sync() {
    local sync_mode="$1"
    local dry_run="${2:-false}"
    local conflict_resolution="${3:-github_wins}"
    local attempt=1
    local success=false

    local cmd="$PYTHON_CMD $PROJECT_ROOT/scripts/unified_sync.py sync $sync_mode"

    if [[ "$dry_run" == "true" ]]; then
        cmd="$cmd --dry-run"
    fi

    cmd="$cmd --conflict-resolution $conflict_resolution"

    log_info "Running unified sync: $sync_mode"
    log_info "Command: $cmd"

    while [[ $attempt -le $MAX_RETRIES ]] && [[ "$success" != "true" ]]; do
        log_info "Attempt $attempt of $MAX_RETRIES..."

        if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
            success=true
            log_success "Sync completed successfully"
        else
            log_warn "Sync failed on attempt $attempt"

            if [[ $attempt -lt $MAX_RETRIES ]]; then
                log_info "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi

        ((attempt++))
    done

    if [[ "$success" != "true" ]]; then
        log_error "Sync failed after $MAX_RETRIES attempts"
        return 1
    fi

    return 0
}

run_github_board_sync() {
    local command="${1:-sync}"

    log_info "Running GitHub board sync: $command"

    if [[ -x "$PROJECT_ROOT/scripts/github-board-sync.sh" ]]; then
        "$PROJECT_ROOT/scripts/github-board-sync.sh" "$command" 2>&1 | tee -a "$LOG_FILE"
    else
        log_warn "github-board-sync.sh not found or not executable"
    fi
}

run_notion_sync() {
    local action="${1:-push}"
    local dry_run="${2:-false}"

    log_info "Running Notion sync: $action"

    if [[ ! -f "$HOME/.config/notion/api_key" ]]; then
        log_warn "Skipping Notion sync - API key not configured"
        return 0
    fi

    local cmd="$PYTHON_CMD $PROJECT_ROOT/scripts/notion_sync.py $action"

    if [[ "$dry_run" == "true" ]]; then
        cmd="$cmd --dry-run"
    fi

    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Notion sync completed"
    else
        log_warn "Notion sync encountered issues"
    fi
}

# =============================================================================
# Commands
# =============================================================================

cmd_sync() {
    local mode="--bidirectional"
    local dry_run="false"
    local conflict_resolution="github_wins"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --github-only)
                mode="--github-only"
                shift
                ;;
            --notion-only)
                mode="--notion-only"
                shift
                ;;
            --bidirectional)
                mode="--bidirectional"
                shift
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            --conflict-resolution)
                conflict_resolution="$2"
                shift 2
                ;;
            --github-wins|--notion-wins|--newest-wins)
                conflict_resolution="${1#--}"
                conflict_resolution="${conflict_resolution//-/_}"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    log_info "Starting sync with mode: $mode"
    log_info "Dry run: $dry_run"
    log_info "Conflict resolution: $conflict_resolution"

    # Run unified sync
    run_unified_sync "$mode" "$dry_run" "$conflict_resolution"

    # Also sync to GitHub Projects board if not dry run and bidirectional
    if [[ "$dry_run" == "false" ]] && [[ "$mode" == "--bidirectional" ]]; then
        run_github_board_sync "sync-board"
    fi

    log_success "All sync operations complete"
}

cmd_status() {
    log_info "Gathering sync status..."

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}        Unified Sync Status${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Run unified sync status
    "$PYTHON_CMD" "$PROJECT_ROOT/scripts/unified_sync.py" status 2>&1 | tee -a "$LOG_FILE"

    echo ""

    # Show GitHub board status
    if [[ -x "$PROJECT_ROOT/scripts/github-board-sync.sh" ]]; then
        echo -e "${BLUE}GitHub Board Status:${NC}"
        "$PROJECT_ROOT/scripts/github-board-sync.sh" report 2>/dev/null | tail -n +5
    fi

    echo ""
    echo -e "${CYAN}========================================${NC}"
}

cmd_diff() {
    log_info "Showing differences between systems..."
    "$PYTHON_CMD" "$PROJECT_ROOT/scripts/unified_sync.py" diff 2>&1 | tee -a "$LOG_FILE"
}

cmd_init() {
    log_info "Initializing sync infrastructure..."

    # Initialize GitHub Projects board
    if [[ -x "$PROJECT_ROOT/scripts/github-board-sync.sh" ]]; then
        log_info "Initializing GitHub Projects board..."
        run_github_board_sync "init"
    fi

    # Create initial sync state
    log_info "Creating initial sync state..."
    run_unified_sync "--bidirectional" "true" "github_wins"  # Dry run first

    log_success "Initialization complete"
    echo ""
    echo "Next steps:"
    echo "  1. Review the dry run output above"
    echo "  2. Run './sync-all.sh' to perform actual sync"
}

cmd_clean() {
    log_info "Cleaning up sync state..."

    # Remove sync state files
    if [[ -f "$PROJECT_ROOT/.sync_state.json" ]]; then
        rm -f "$PROJECT_ROOT/.sync_state.json"
        log_info "Removed .sync_state.json"
    fi

    if [[ -f "$PROJECT_ROOT/.notion_sync_state.json" ]]; then
        rm -f "$PROJECT_ROOT/.notion_sync_state.json"
        log_info "Removed .notion_sync_state.json"
    fi

    if [[ -f "$PROJECT_ROOT/.github_board_sync_state.json" ]]; then
        rm -f "$PROJECT_ROOT/.github_board_sync_state.json"
        log_info "Removed .github_board_sync_state.json"
    fi

    log_success "Sync state cleaned. Run './sync-all.sh init' to reinitialize."
}

show_help() {
    cat << 'EOF'
Unified Sync Coordinator
========================

Keeps GitHub Projects and Notion in perfect sync.

USAGE:
    ./sync-all.sh [COMMAND] [OPTIONS]

COMMANDS:
    (default)       Full bidirectional sync
    sync            Synchronize systems (with options below)
    status          Show current sync status
    diff            Show differences between systems
    init            Initialize sync infrastructure
    clean           Remove all sync state files
    help            Show this help message

SYNC OPTIONS:
    --bidirectional     Full bidirectional sync (default)
    --github-only       Sync to GitHub only (Notion -> GitHub)
    --notion-only       Sync to Notion only (GitHub -> Notion)
    --dry-run           Preview changes without applying
    --github-wins       Resolve conflicts with GitHub data (default)
    --notion-wins       Resolve conflicts with Notion data
    --newest-wins       Resolve conflicts with newest data

EXAMPLES:
    ./sync-all.sh                              # Full bidirectional sync
    ./sync-all.sh --dry-run                    # Preview full sync
    ./sync-all.sh --github-only                # Sync Notion -> GitHub only
    ./sync-all.sh --notion-only --dry-run      # Preview GitHub -> Notion
    ./sync-all.sh sync --notion-wins           # Sync with Notion priority
    ./sync-all.sh status                       # Show status
    ./sync-all.sh diff                         # Show differences

FIELD MAPPINGS:
    GitHub Issue title    <->  Notion task name
    GitHub labels         <->  Notion category/priority
    GitHub state          <->  Notion status (open->To Do, closed->Done)
    GitHub assignees      <->  Notion assignees
    GitHub milestone      <->  Notion milestone

CONFIGURATION:
    Config file:     .github/board-sync.yml
    Sync state:      .sync_state.json
    Notion API key:  ~/.config/notion/api_key
    Sync logs:       .sync_logs/

RELATED SCRIPTS:
    scripts/unified_sync.py       Main Python sync coordinator
    scripts/notion_sync.py        Notion-specific sync
    scripts/github-board-sync.sh  GitHub Projects board sync

For more information, see the project documentation.
EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Ensure we're in project root
    cd "$PROJECT_ROOT"

    # Create log directory
    ensure_log_dir

    # Parse command
    local command="${1:-sync}"

    case "$command" in
        sync)
            shift || true
            check_prerequisites
            cmd_sync "$@"
            ;;
        --bidirectional|--github-only|--notion-only|--dry-run)
            # Allow calling directly with sync options
            check_prerequisites
            cmd_sync "$@"
            ;;
        status)
            check_prerequisites
            cmd_status
            ;;
        diff)
            check_prerequisites
            cmd_diff
            ;;
        init)
            check_prerequisites
            cmd_init
            ;;
        clean)
            cmd_clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main with all arguments
main "$@"
