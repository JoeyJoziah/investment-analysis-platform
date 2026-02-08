#!/bin/bash
# Claude Flow V3 - Setup Validation Script
# Validates that all V3 systems are working correctly

set -e

echo "🔍 Claude Flow V3 Setup Validation"
echo "==================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Test 1: CLI Installation
echo "1. Testing CLI installation..."
if npx @claude-flow/cli@latest --version &>/dev/null; then
  VERSION=$(npx @claude-flow/cli@latest --version 2>/dev/null | head -1)
  check_pass "CLI installed: $VERSION"
else
  check_fail "CLI not installed or not responding"
fi
echo ""

# Test 2: Daemon Status
echo "2. Testing daemon status..."
if npx @claude-flow/cli@latest daemon status &>/dev/null; then
  check_pass "Daemon is running"
else
  check_warn "Daemon is not running (run: npx @claude-flow/cli@latest daemon start)"
fi
echo ""

# Test 3: Memory Operations
echo "3. Testing memory operations..."
TEST_KEY="validation-test-$(date +%s)"
if npx @claude-flow/cli@latest memory store --key "$TEST_KEY" --value "test" --namespace patterns &>/dev/null; then
  check_pass "Memory store works"

  if npx @claude-flow/cli@latest memory retrieve --key "$TEST_KEY" --namespace patterns &>/dev/null; then
    check_pass "Memory retrieve works"
  else
    check_fail "Memory retrieve failed"
  fi

  if npx @claude-flow/cli@latest memory search --query "validation" --namespace patterns &>/dev/null; then
    check_pass "Memory search works"
  else
    check_fail "Memory search failed"
  fi
else
  check_fail "Memory store failed"
fi
echo ""

# Test 4: Hooks System
echo "4. Testing hooks system..."
if npx @claude-flow/cli@latest hooks list &>/dev/null; then
  HOOK_COUNT=$(npx @claude-flow/cli@latest hooks list 2>/dev/null | wc -l)
  check_pass "Hooks are accessible ($HOOK_COUNT hooks)"
else
  check_fail "Hooks list failed"
fi
echo ""

# Test 5: Background Workers
echo "5. Testing background workers..."
if npx @claude-flow/cli@latest hooks worker list &>/dev/null; then
  WORKER_COUNT=$(npx @claude-flow/cli@latest hooks worker list 2>/dev/null | grep -c "Worker:" || echo "0")
  if [ "$WORKER_COUNT" -gt 0 ]; then
    check_pass "Workers configured ($WORKER_COUNT workers)"
  else
    check_warn "No workers found (expected 12 workers)"
  fi
else
  check_fail "Worker list failed"
fi
echo ""

# Test 6: Swarm Capabilities
echo "6. Testing swarm capabilities..."
if npx @claude-flow/cli@latest swarm status &>/dev/null; then
  check_pass "Swarm commands work"
else
  check_warn "Swarm not initialized (run: npx @claude-flow/cli@latest swarm init)"
fi
echo ""

# Test 7: Agent Spawning
echo "7. Testing agent capabilities..."
if npx @claude-flow/cli@latest agent list &>/dev/null; then
  AGENT_COUNT=$(npx @claude-flow/cli@latest agent list 2>/dev/null | grep -c "Agent:" || echo "0")
  check_pass "Agent commands work ($AGENT_COUNT active agents)"
else
  check_warn "Agent list failed (no agents spawned yet)"
fi
echo ""

# Test 8: Neural/Learning Systems
echo "8. Testing neural/learning systems..."
if npx @claude-flow/cli@latest neural patterns --list &>/dev/null; then
  check_pass "Neural pattern system accessible"
else
  check_warn "Neural patterns not trained yet (run: npx @claude-flow/cli@latest hooks pretrain)"
fi
echo ""

# Test 9: Performance Tools
echo "9. Testing performance tools..."
if npx @claude-flow/cli@latest performance benchmark --help &>/dev/null; then
  check_pass "Performance tools available"
else
  check_warn "Performance tools not accessible"
fi
echo ""

# Test 10: Security Tools
echo "10. Testing security tools..."
if npx @claude-flow/cli@latest security scan --help &>/dev/null; then
  check_pass "Security tools available"
else
  check_warn "Security tools not accessible"
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

if [ $FAILED -eq 0 ]; then
  echo -e "${GREEN}✓ All critical systems operational!${NC}"
  echo ""
  echo "Your Claude Flow V3 setup is working correctly."
  echo ""
  echo "Next steps:"
  echo "1. Initialize swarm: npx @claude-flow/cli@latest swarm init --topology hierarchical-mesh"
  echo "2. Train patterns: npx @claude-flow/cli@latest hooks pretrain --model-type moe"
  echo "3. View metrics: npx @claude-flow/cli@latest hooks metrics --v3-dashboard"
  echo "4. Enable workers: npx @claude-flow/cli@latest hooks worker dispatch --trigger audit"
  exit 0
else
  echo -e "${RED}✗ Some systems failed validation${NC}"
  echo ""
  echo "Run the fix script first:"
  echo "  ./.claude/scripts/fix-cli-infrastructure.sh"
  echo ""
  echo "Then re-run this validation:"
  echo "  ./.claude/scripts/validate-v3-setup.sh"
  exit 1
fi
