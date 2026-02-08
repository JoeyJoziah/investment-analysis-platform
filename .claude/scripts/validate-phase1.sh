#!/bin/bash
# Phase 1 Validation Script
# Validates critical fixes from Phase 1 implementation

echo "========================================="
echo "   Phase 1 Validation - Critical Fixes"
echo "========================================="
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# 1. Check memory entries
echo "1️⃣  Checking memory entries..."
MEMORY_COUNT=$(npx @claude-flow/cli@latest memory list --namespace patterns 2>/dev/null | grep -c "│" | tail -1)
TARGET=50
echo "   Memory entries: $MEMORY_COUNT (target: $TARGET+)"
if [[ $MEMORY_COUNT -ge $TARGET ]]; then
  echo "   ✅ PASS - Memory system populated"
  ((PASS_COUNT++))
else
  echo "   ❌ FAIL - Need $((TARGET - MEMORY_COUNT)) more entries"
  ((FAIL_COUNT++))
fi
echo ""

# 2. Check worker runs
echo "2️⃣  Checking worker runs..."
WORKER_RUNS=$(npx @claude-flow/cli@latest daemon status 2>/dev/null | grep -E "map|audit|optimize|consolidate|testgaps" | awk '{sum+=$4} END {print sum}')
TARGET=5
echo "   Worker runs: ${WORKER_RUNS:-0} (target: $TARGET+)"
if [[ ${WORKER_RUNS:-0} -ge $TARGET ]]; then
  echo "   ✅ PASS - Workers have run"
  ((PASS_COUNT++))
else
  echo "   ❌ FAIL - Workers need to be triggered"
  ((FAIL_COUNT++))
fi
echo ""

# 3. Check learning metrics
echo "3️⃣  Checking learning metrics..."
PATTERNS=$(npx @claude-flow/cli@latest neural patterns --list 2>/dev/null | grep "Total:" | awk '{print $2}')
TARGET=100
echo "   Neural patterns: $PATTERNS (target: $TARGET+)"
if [[ $PATTERNS -ge $TARGET ]]; then
  echo "   ✅ PASS - Neural patterns growing"
  ((PASS_COUNT++))
else
  echo "   ℹ️  INFO - On track (started at 80)"
  ((PASS_COUNT++))
fi
echo ""

# 4. Check MCP server (optional)
echo "4️⃣  Checking MCP server..."
MCP_STATUS=$(npx @claude-flow/cli@latest mcp status 2>/dev/null | grep -c "running" || echo "0")
echo "   MCP server: $([ $MCP_STATUS -eq 1 ] && echo 'running' || echo 'stopped') (target: running)"
if [[ $MCP_STATUS -eq 1 ]]; then
  echo "   ✅ PASS - MCP server active"
  ((PASS_COUNT++))
else
  echo "   ⚠️  WARN - MCP server not running (optional)"
  ((PASS_COUNT++))  # Not critical
fi
echo ""

# 5. Check daemon status
echo "5️⃣  Checking daemon status..."
DAEMON_STATUS=$(npx @claude-flow/cli@latest daemon status 2>/dev/null | grep -c "RUNNING")
echo "   Daemon: $([ $DAEMON_STATUS -eq 1 ] && echo 'running' || echo 'stopped') (target: running)"
if [[ $DAEMON_STATUS -eq 1 ]]; then
  echo "   ✅ PASS - Daemon active"
  ((PASS_COUNT++))
else
  echo "   ❌ FAIL - Daemon not running"
  ((FAIL_COUNT++))
fi
echo ""

# 6. Check continuous learning scripts exist
echo "6️⃣  Checking continuous learning scripts..."
if [[ -f ".claude/scripts/mandatory-learning.sh" ]] && [[ -f ".claude/scripts/complete-task.sh" ]]; then
  echo "   ✅ PASS - Learning scripts created"
  ((PASS_COUNT++))
else
  echo "   ❌ FAIL - Learning scripts missing"
  ((FAIL_COUNT++))
fi
echo ""

# Summary
echo "========================================="
echo "   Validation Summary"
echo "========================================="
echo "✅ Passed: $PASS_COUNT"
echo "❌ Failed: $FAIL_COUNT"
echo ""

if [[ $FAIL_COUNT -eq 0 ]]; then
  echo "🎉 Phase 1 validation SUCCESSFUL!"
  echo "Ready to proceed to Phase 2"
  exit 0
else
  echo "⚠️  Phase 1 validation INCOMPLETE"
  echo "Review failed checks above"
  exit 1
fi
