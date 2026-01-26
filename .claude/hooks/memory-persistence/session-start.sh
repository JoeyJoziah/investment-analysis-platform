#!/bin/bash
# SessionStart Hook - Load previous context and memory on new session
#
# Runs when a new Claude session starts. Checks for recent session
# files, learned skills, and shared memory namespaces.
#
# Hook config (in ~/.claude/settings.json or .claude/hooks/hooks.json):
# {
#   "hooks": {
#     "SessionStart": [{
#       "matcher": "*",
#       "hooks": [{
#         "type": "command",
#         "command": "bash .claude/hooks/memory-persistence/session-start.sh"
#       }]
#     }]
#   }
# }

SESSIONS_DIR="${HOME}/.claude/sessions"
LEARNED_DIR="${HOME}/.claude/skills/learned"
PROJECT_ROOT="$(pwd)"
MEMORY_FILE="$PROJECT_ROOT/.claude/memory/shared-context.json"
AGENT_STORE="$PROJECT_ROOT/.claude-flow/agents/store.json"

echo "[SessionStart] Initializing session for investment-analysis-platform" >&2

# Check for recent session files (last 7 days)
recent_sessions=$(find "$SESSIONS_DIR" -name "*.tmp" -mtime -7 2>/dev/null | wc -l | tr -d ' ')

if [ "$recent_sessions" -gt 0 ]; then
  latest=$(ls -t "$SESSIONS_DIR"/*.tmp 2>/dev/null | head -1)
  echo "[SessionStart] Found $recent_sessions recent session(s)" >&2
  echo "[SessionStart] Latest: $latest" >&2
fi

# Check for learned skills
learned_count=$(find "$LEARNED_DIR" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')

if [ "$learned_count" -gt 0 ]; then
  echo "[SessionStart] $learned_count learned skill(s) available in $LEARNED_DIR" >&2
fi

# Check agent store
if [ -f "$AGENT_STORE" ]; then
  agent_count=$(grep -c '"agentId"' "$AGENT_STORE" 2>/dev/null || echo "0")
  echo "[SessionStart] Agent store: $agent_count agents registered" >&2
fi

# Load memory namespaces (if claude-flow is available)
if command -v npx &> /dev/null; then
  # Check for patterns namespace (cross-session learnings)
  if [ -f "$MEMORY_FILE" ]; then
    echo "[SessionStart] Shared context available at $MEMORY_FILE" >&2
  fi
fi

# Report available swarm configurations
if [ -f "$PROJECT_ROOT/.claude/config/agent-registry.json" ]; then
  echo "[SessionStart] Agent registry loaded with routing configuration" >&2
fi

# Report investment-specific context
echo "[SessionStart] Memory namespaces available:" >&2
echo "  - investment-analysis: Deal data, analysis results" >&2
echo "  - portfolios: Portfolio metrics and positions" >&2
echo "  - github-operations: PR and issue tracking" >&2
echo "  - patterns: Cross-session learned patterns" >&2

echo "[SessionStart] Session initialized successfully" >&2
