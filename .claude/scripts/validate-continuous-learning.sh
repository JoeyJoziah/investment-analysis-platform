#!/bin/bash
# Continuous Learning Validation Script
# Validates that pre-task/post-task hooks are being used and effective

set -e

echo "🧠 Continuous Learning Validation"
echo "==================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

check_pass() {
  echo -e "${GREEN}✓${NC} $1"
  ((PASSED++))
}

check_fail() {
  echo -e "${RED}✗${NC} $1"
  ((FAILED++))
}

check_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

check_info() {
  echo -e "${BLUE}ℹ${NC} $1"
}

# Test 1: Check daemon is running
echo "1. Checking daemon status..."
if npx @claude-flow/cli@latest daemon status &>/dev/null; then
  check_pass "Daemon is running (required for learning)"
else
  check_fail "Daemon is NOT running - continuous learning disabled!"
  echo "   Fix: npx @claude-flow/cli@latest daemon start"
fi
echo ""

# Test 2: Check neural patterns exist
echo "2. Checking neural patterns..."
PATTERN_COUNT=$(npx @claude-flow/cli@latest neural patterns --list 2>/dev/null | grep "Total:" | grep -oE "[0-9]+" | head -1 || echo "0")
if [ "$PATTERN_COUNT" -gt 0 ]; then
  check_pass "Neural patterns found: $PATTERN_COUNT patterns"
  if [ "$PATTERN_COUNT" -lt 10 ]; then
    check_warn "Pattern count is low. Need more training."
  fi
else
  check_fail "No neural patterns found - learning not happening"
  echo "   Fix: Run tasks with pre-task/post-task hooks"
fi
echo ""

# Test 3: Check trajectories
echo "3. Checking learning trajectories..."
TRAJ_COUNT=$(npx @claude-flow/cli@latest neural patterns --list 2>/dev/null | grep "Trajectories:" | grep -oE "[0-9]+" || echo "0")
if [ "$TRAJ_COUNT" -gt 0 ]; then
  check_pass "Learning trajectories: $TRAJ_COUNT paths"
else
  check_fail "No trajectories found - post-task hooks not being used"
fi
echo ""

# Test 4: Check memory database
echo "4. Checking memory database..."
if [ -f ".swarm/memory.db" ]; then
  DB_SIZE=$(du -h .swarm/memory.db | cut -f1)
  check_pass "Memory database exists: $DB_SIZE"

  # Check if database has entries
  ENTRY_COUNT=$(npx @claude-flow/cli@latest memory list --limit 1000 2>/dev/null | grep -c "│" || echo "0")
  if [ "$ENTRY_COUNT" -gt 3 ]; then  # Header rows
    check_pass "Memory database has entries"
  else
    check_warn "Memory database is empty - no patterns stored yet"
  fi
else
  check_fail "Memory database not found"
  echo "   Fix: npx @claude-flow/cli@latest memory init --force"
fi
echo ""

# Test 5: Test pre-task hook
echo "5. Testing pre-task hook..."
if npx @claude-flow/cli@latest hooks pre-task --task-id "validate-$(date +%s)" --description "Validation test" &>/dev/null; then
  check_pass "Pre-task hook is functional"
else
  check_fail "Pre-task hook failed"
  echo "   Fix: npx @claude-flow/cli@latest daemon start"
fi
echo ""

# Test 6: Test post-task hook
echo "6. Testing post-task hook..."
TEST_TASK_ID="validate-test-$(date +%s)"
if npx @claude-flow/cli@latest hooks post-task --task-id "$TEST_TASK_ID" --success true --store-results true &>/dev/null; then
  check_pass "Post-task hook is functional"

  # Verify pattern was created
  sleep 1
  NEW_PATTERN_COUNT=$(npx @claude-flow/cli@latest neural patterns --list 2>/dev/null | grep "Total:" | grep -oE "[0-9]+" | head -1 || echo "0")
  if [ "$NEW_PATTERN_COUNT" -gt "$PATTERN_COUNT" ]; then
    check_pass "Post-task hook successfully created new pattern"
  else
    check_warn "Post-task hook didn't create new pattern (may be normal)"
  fi
else
  check_fail "Post-task hook failed"
fi
echo ""

# Test 7: Check workers are active
echo "7. Checking background workers..."
ACTIVE_WORKERS=$(npx @claude-flow/cli@latest daemon status 2>/dev/null | grep "Workers Enabled:" | grep -oE "[0-9]+" || echo "0")
if [ "$ACTIVE_WORKERS" -gt 0 ]; then
  check_pass "Background workers active: $ACTIVE_WORKERS workers"
else
  check_warn "No background workers active"
  echo "   Recommend: Enable workers for continuous optimization"
fi
echo ""

# Test 8: Check recent learning activity
echo "8. Checking recent learning activity..."
RECENT_PATTERNS=$(npx @claude-flow/cli@latest memory list --namespace patterns --limit 10 2>/dev/null | grep "ago\|now" | wc -l || echo "0")
if [ "$RECENT_PATTERNS" -gt 0 ]; then
  check_pass "Recent learning activity detected"
else
  check_warn "No recent learning activity (>24h)"
  echo "   Action: Use pre-task/post-task hooks on next task"
fi
echo ""

# Test 9: Verify HNSW indexing
echo "9. Checking HNSW vector indexing..."
if npx @claude-flow/cli@latest memory list 2>&1 | grep -q "Vector"; then
  check_pass "HNSW vector indexing enabled"
else
  check_warn "Vector indexing status unclear"
fi
echo ""

# Test 10: Check model routing intelligence
echo "10. Checking intelligent model routing..."
ROUTING_TEST=$(npx @claude-flow/cli@latest hooks pre-task --task-id "routing-test-$(date +%s)" --description "Simple test task" 2>&1)
if echo "$ROUTING_TEST" | grep -q "TASK_MODEL_RECOMMENDATION"; then
  check_pass "Intelligent model routing is working"
  RECOMMENDED_MODEL=$(echo "$ROUTING_TEST" | grep "TASK_MODEL_RECOMMENDATION" | grep -oE "(haiku|sonnet|opus)")
  check_info "Test task routed to: $RECOMMENDED_MODEL"
else
  check_warn "Model routing not providing recommendations"
fi
echo ""

# Summary
echo "=========================================="
echo "📊 Validation Summary"
echo "=========================================="
echo ""
echo -e "Passed:   ${GREEN}$PASSED${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

TOTAL=$((PASSED + FAILED + WARNINGS))
if [ $TOTAL -gt 0 ]; then
  PERCENTAGE=$((PASSED * 100 / TOTAL))
  echo "Success Rate: $PERCENTAGE%"
fi
echo ""

# Recommendations
echo "📋 Recommendations"
echo "===================="
echo ""

if [ $FAILED -gt 0 ]; then
  echo -e "${RED}CRITICAL:${NC} Fix failed tests immediately"
  echo "1. Ensure daemon is running: npx @claude-flow/cli@latest daemon start"
  echo "2. Initialize memory: npx @claude-flow/cli@latest memory init --force"
  echo "3. Test hooks manually (see commands above)"
fi

if [ "$PATTERN_COUNT" -lt 50 ]; then
  echo -e "${YELLOW}TRAINING NEEDED:${NC} Pattern count is low ($PATTERN_COUNT)"
  echo "1. Run: npx @claude-flow/cli@latest hooks pretrain --model-type moe --epochs 10"
  echo "2. Use pre-task/post-task hooks on every task"
  echo "3. Target: 500+ patterns for optimal performance"
fi

if [ "$ACTIVE_WORKERS" -lt 5 ]; then
  echo -e "${YELLOW}WORKERS:${NC} Enable more background workers"
  echo "1. Enable all workers in settings.json (12 total available)"
  echo "2. Workers provide continuous optimization"
fi

echo ""
echo "Next Steps:"
echo "1. ALWAYS use pre-task hook before tasks"
echo "2. ALWAYS use post-task hook after tasks"
echo "3. Check progress weekly: $0"
echo "4. Train neural patterns: npx @claude-flow/cli@latest neural train --pattern-type coordination"
echo ""

# Set exit code based on failures
if [ $FAILED -gt 0 ]; then
  exit 1
else
  exit 0
fi
