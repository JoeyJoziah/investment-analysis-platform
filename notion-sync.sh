#!/bin/bash
#
# Notion Sync Wrapper Script
# Syncs TODO.md with Notion Product Development Tracker
#
# Usage:
#   ./notion-sync.sh push     # Push TODO.md to Notion
#   ./notion-sync.sh pull     # Pull from Notion to TODO.md
#   ./notion-sync.sh status   # Show sync status
#   ./notion-sync.sh          # Same as status
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/scripts/notion_sync.py"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found"
    exit 1
fi

# Check if requests is installed
if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing required dependency: requests"
    pip3 install requests
fi

# Check if API key is configured
API_KEY_FILE="$HOME/.config/notion/api_key"
if [ ! -f "$API_KEY_FILE" ]; then
    echo "Error: Notion API key not configured"
    echo ""
    echo "To configure:"
    echo "  1. Create an integration at https://notion.so/my-integrations"
    echo "  2. Copy the API key and run:"
    echo "     mkdir -p ~/.config/notion"
    echo "     echo 'your-api-key' > ~/.config/notion/api_key"
    echo "  3. Share your Notion pages with the integration"
    exit 1
fi

# Default action is status
ACTION="${1:-status}"

case "$ACTION" in
    push|pull|status)
        python3 "$PYTHON_SCRIPT" "$@"
        ;;
    -h|--help|help)
        python3 "$PYTHON_SCRIPT" --help
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: $0 {push|pull|status}"
        exit 1
        ;;
esac
