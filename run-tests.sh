#!/bin/bash
# Test Runner with Board Sync
# Runs tests and syncs project boards on completion
#
# Usage:
#   ./run-tests.sh [options]
#   Options:
#     --backend    Run backend tests only (pytest)
#     --frontend   Run frontend tests only (npm test)
#     --all        Run all tests (default)
#     --no-sync    Skip board sync after tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/.claude/hooks/board-sync/sync-boards.sh"

# Parse arguments
RUN_BACKEND=false
RUN_FRONTEND=false
SKIP_SYNC=false

if [ $# -eq 0 ]; then
    RUN_BACKEND=true
    RUN_FRONTEND=true
fi

for arg in "$@"; do
    case $arg in
        --backend) RUN_BACKEND=true ;;
        --frontend) RUN_FRONTEND=true ;;
        --all)
            RUN_BACKEND=true
            RUN_FRONTEND=true
            ;;
        --no-sync) SKIP_SYNC=true ;;
    esac
done

# Default to all if nothing specified
if [ "$RUN_BACKEND" = false ] && [ "$RUN_FRONTEND" = false ]; then
    RUN_BACKEND=true
    RUN_FRONTEND=true
fi

TEST_RESULTS=0

echo "========================================"
echo "  Investment Analysis Platform Tests"
echo "========================================"
echo ""

# Run backend tests
if [ "$RUN_BACKEND" = true ]; then
    echo "Running backend tests (pytest)..."
    echo "----------------------------------------"

    if [ -d "$SCRIPT_DIR/.venv" ]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi

    cd "$SCRIPT_DIR"
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

    if pytest backend/tests/ --tb=short -q; then
        echo ""
        echo "[Backend Tests] PASSED"
    else
        echo ""
        echo "[Backend Tests] FAILED"
        TEST_RESULTS=1
    fi

    echo ""
fi

# Run frontend tests
if [ "$RUN_FRONTEND" = true ]; then
    echo "Running frontend tests (npm test)..."
    echo "----------------------------------------"

    cd "$SCRIPT_DIR/frontend/web"

    if npm test -- --watchAll=false --passWithNoTests; then
        echo ""
        echo "[Frontend Tests] PASSED"
    else
        echo ""
        echo "[Frontend Tests] FAILED"
        TEST_RESULTS=1
    fi

    echo ""
fi

cd "$SCRIPT_DIR"

# Summary
echo "========================================"
if [ "$TEST_RESULTS" -eq 0 ]; then
    echo "  All Tests PASSED"
else
    echo "  Some Tests FAILED"
fi
echo "========================================"
echo ""

# Trigger board sync
if [ "$SKIP_SYNC" = false ] && [ -x "$SYNC_SCRIPT" ]; then
    export BOARD_SYNC_TRIGGER="test-completion"
    "$SYNC_SCRIPT" --silent &
    echo "[Board Sync] Triggered in background after test completion"
fi

exit $TEST_RESULTS
