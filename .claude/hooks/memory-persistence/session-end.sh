#!/bin/bash
# Stop Hook (Session End) - Persist learnings and memory when session ends
#
# Runs when Claude session ends. Creates/updates session log file
# with timestamp for continuity tracking and consolidates memory.
#
# Hook config (in ~/.claude/settings.json or .claude/hooks/hooks.json):
# {
#   "hooks": {
#     "SessionEnd": [{
#       "matcher": "*",
#       "hooks": [{
#         "type": "command",
#         "command": "bash .claude/hooks/memory-persistence/session-end.sh"
#       }]
#     }]
#   }
# }

SESSIONS_DIR="${HOME}/.claude/sessions"
PROJECT_ROOT="$(pwd)"
MEMORY_DIR="$PROJECT_ROOT/.claude/memory"
TODAY=$(date '+%Y-%m-%d')
TIMESTAMP=$(date '+%Y-%m-%dT%H:%M:%S')
SESSION_FILE="${SESSIONS_DIR}/${TODAY}-session.tmp"

mkdir -p "$SESSIONS_DIR"
mkdir -p "$MEMORY_DIR"

echo "[SessionEnd] Persisting session state..." >&2

# If session file exists for today, update the end time
if [ -f "$SESSION_FILE" ]; then
  # Update Last Updated timestamp (compatible with both macOS and Linux)
  sed -i '' "s/\*\*Last Updated:\*\*.*/\*\*Last Updated:\*\* $(date '+%H:%M')/" "$SESSION_FILE" 2>/dev/null || \
  sed -i "s/\*\*Last Updated:\*\*.*/\*\*Last Updated:\*\* $(date '+%H:%M')/" "$SESSION_FILE" 2>/dev/null
  echo "[SessionEnd] Updated session file: $SESSION_FILE" >&2
else
  # Create new session file with template
  cat > "$SESSION_FILE" << EOF
# Session: $(date '+%Y-%m-%d')
**Date:** $TODAY
**Started:** $(date '+%H:%M')
**Last Updated:** $(date '+%H:%M')
**Project:** investment-analysis-platform

---

## Current State

[Session context goes here]

### Completed
- [ ]

### In Progress
- [ ]

### Notes for Next Session
-

### Context to Load
\`\`\`
[relevant files]
\`\`\`

### Agents Used
-

### Memory Namespaces Accessed
-
EOF
  echo "[SessionEnd] Created session file: $SESSION_FILE" >&2
fi

# Consolidate session findings to shared memory
SHARED_CONTEXT="$MEMORY_DIR/shared-context.json"
if [ ! -f "$SHARED_CONTEXT" ]; then
  cat > "$SHARED_CONTEXT" << EOF
{
  "version": "1.0.0",
  "lastUpdated": "$TIMESTAMP",
  "sessions": [],
  "patterns": [],
  "decisions": [],
  "namespaces": {
    "investment-analysis": [],
    "portfolios": [],
    "github-operations": [],
    "development": []
  }
}
EOF
  echo "[SessionEnd] Created shared context file: $SHARED_CONTEXT" >&2
else
  # Update lastUpdated timestamp
  if command -v jq &> /dev/null; then
    jq --arg ts "$TIMESTAMP" '.lastUpdated = $ts' "$SHARED_CONTEXT" > "$SHARED_CONTEXT.tmp" && \
    mv "$SHARED_CONTEXT.tmp" "$SHARED_CONTEXT"
  fi
fi

# Log session end
echo "[SessionEnd] Session ended at $TIMESTAMP" >&2
echo "[SessionEnd] Memory consolidated to $MEMORY_DIR" >&2
echo "[SessionEnd] Session state persisted successfully" >&2
