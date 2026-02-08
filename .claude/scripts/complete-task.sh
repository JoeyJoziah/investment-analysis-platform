#!/bin/bash
# Complete task and store learnings
# CRITICAL: Run this after EVERY task to enable continuous learning

SUCCESS="${1:-true}"
TASK_ID=$(cat /tmp/claude-flow-current-task 2>/dev/null)

if [[ -z "$TASK_ID" ]]; then
  echo "❌ No active task found. Did you run mandatory-learning.sh first?"
  exit 1
fi

echo "🎯 Completing task: $TASK_ID"
echo "Success: $SUCCESS"
echo ""

# Post-task hook (MANDATORY)
npx @claude-flow/cli@latest hooks post-task \
  --task-id "$TASK_ID" \
  --success "$SUCCESS" \
  --store-results true 2>/dev/null

echo ""

# Prompt for learnings
echo "📚 What did you learn from this task?"
read -p "Learning (or Enter to skip): " LEARNING

if [[ -n "$LEARNING" ]]; then
  npx @claude-flow/cli@latest memory store \
    --namespace patterns \
    --key "learning-$TASK_ID" \
    --value "$LEARNING" 2>/dev/null
  echo "✅ Learning stored"
fi

# Clean up
rm -f /tmp/claude-flow-current-task

echo ""
echo "✅ Task complete. Learning pipeline executed."
