#!/bin/bash
# Mandatory continuous learning wrapper
# CRITICAL: Use this for ALL tasks to enable continuous learning

TASK_ID="task-$(date +%s)"
DESCRIPTION="$1"
shift  # Remove description from args

echo "🧠 Starting task with continuous learning enabled"
echo "Task ID: $TASK_ID"
echo "Description: $DESCRIPTION"
echo ""

# Pre-task hook (MANDATORY)
echo "📝 Running pre-task analysis..."
npx @claude-flow/cli@latest hooks pre-task \
  --task-id "$TASK_ID" \
  --description "$DESCRIPTION" 2>/dev/null

echo ""

# Search memory for similar patterns
echo "🔍 Searching memory for relevant patterns..."
npx @claude-flow/cli@latest memory search \
  --query "$DESCRIPTION" \
  --namespace patterns \
  --limit 5 2>/dev/null

echo ""

# Store task ID for post-task
echo "$TASK_ID" > /tmp/claude-flow-current-task

echo "✅ Pre-task complete. Proceed with your work."
echo "💡 When done, run: .claude/scripts/complete-task.sh [success|failure]"
echo ""
