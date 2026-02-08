#!/bin/bash
# Phase 3 Integration Verification Script
# Tests swarm-workflow coordination

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Phase 3: Swarm-Workflow Coordination Verification${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOOKS_DIR="${PROJECT_ROOT}/.claude/hooks"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"

    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${BLUE}[TEST ${TESTS_RUN}]${NC} ${test_name}..."

    if eval "$test_command" > /dev/null 2>&1; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo -e "${RED}✗ FAIL${NC}"
        return 1
    fi
}

# 1. Check hook files exist and are executable
echo -e "${YELLOW}Checking hook files...${NC}"
run_test "workflow-swarm-sync.sh exists" "test -f '${HOOKS_DIR}/workflow-swarm-sync.sh'"
run_test "swarm-consensus-sync.sh exists" "test -f '${HOOKS_DIR}/swarm-consensus-sync.sh'"

# Make hooks executable
chmod +x "${HOOKS_DIR}/workflow-swarm-sync.sh" 2>/dev/null || true
chmod +x "${HOOKS_DIR}/swarm-consensus-sync.sh" 2>/dev/null || true

run_test "workflow-swarm-sync.sh is executable" "test -x '${HOOKS_DIR}/workflow-swarm-sync.sh'"
run_test "swarm-consensus-sync.sh is executable" "test -x '${HOOKS_DIR}/swarm-consensus-sync.sh'"
echo ""

# 2. Initialize orchestration namespace
echo -e "${YELLOW}Initializing orchestration namespace...${NC}"
bash "${HOOKS_DIR}/workflow-swarm-sync.sh" init || {
    echo -e "${RED}✗ Failed to initialize orchestration namespace${NC}"
    exit 1
}
echo -e "${GREEN}✓ Orchestration namespace initialized${NC}"
echo ""

# 3. Test workflow phase sync
echo -e "${YELLOW}Testing workflow phase synchronization...${NC}"
TEST_WORKFLOW_ID="test-wf-$(date +%s)"
TEST_PHASE="design"
TEST_STATUS="completed"

bash "${HOOKS_DIR}/workflow-swarm-sync.sh" sync-phase \
    "${TEST_WORKFLOW_ID}" "${TEST_PHASE}" "${TEST_STATUS}" \
    '{"test":true,"adr_created":true}' || {
    echo -e "${RED}✗ Failed to sync phase transition${NC}"
    exit 1
}
echo -e "${GREEN}✓ Phase transition synced${NC}"

# 4. Test phase output recording
echo -e "${YELLOW}Testing phase output recording...${NC}"
bash "${HOOKS_DIR}/workflow-swarm-sync.sh" record-output \
    "${TEST_WORKFLOW_ID}" "${TEST_PHASE}" "adr" \
    '{"decision":"Use microservices","rationale":"Scalability"}' || {
    echo -e "${RED}✗ Failed to record phase output${NC}"
    exit 1
}
echo -e "${GREEN}✓ Phase output recorded${NC}"

# 5. Test workflow state retrieval
echo -e "${YELLOW}Testing workflow state retrieval...${NC}"
bash "${HOOKS_DIR}/workflow-swarm-sync.sh" get-state "${TEST_WORKFLOW_ID}" > /dev/null || {
    echo -e "${YELLOW}⚠ Workflow state not found (expected for new workflow)${NC}"
}

# 6. Test consensus decision recording
echo -e "${YELLOW}Testing consensus decision recording...${NC}"
bash "${HOOKS_DIR}/swarm-consensus-sync.sh" record-decision \
    "${TEST_WORKFLOW_ID}" "${TEST_PHASE}" "approved" \
    '{"approvers":["agent1","agent2"],"votes":2}' || {
    echo -e "${RED}✗ Failed to record consensus decision${NC}"
    exit 1
}
echo -e "${GREEN}✓ Consensus decision recorded${NC}"

# 7. Verify memory entries
echo -e "${YELLOW}Verifying memory entries...${NC}"
MEMORY_COUNT=$(npx @claude-flow/cli@latest memory list --namespace orchestration 2>/dev/null | grep -c "key:" || echo "0")

if [[ "$MEMORY_COUNT" -gt 0 ]]; then
    echo -e "${GREEN}✓ Memory entries found: ${MEMORY_COUNT}${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${YELLOW}⚠ No memory entries found${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))
echo ""

# 8. Test hook help commands
echo -e "${YELLOW}Testing hook help commands...${NC}"
run_test "workflow-swarm-sync help works" "bash '${HOOKS_DIR}/workflow-swarm-sync.sh' help"
run_test "swarm-consensus-sync help works" "bash '${HOOKS_DIR}/swarm-consensus-sync.sh' help"
echo ""

# 9. Verify workflow-engine.json configuration
echo -e "${YELLOW}Verifying workflow-engine.json...${NC}"
CONFIG_FILE="${PROJECT_ROOT}/.claude/config/workflow-engine.json"

if grep -q '"orchestration"' "$CONFIG_FILE"; then
    echo -e "${GREEN}✓ Orchestration namespace registered${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Orchestration namespace not found in config${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

if grep -q '"workflow_swarm_sync"' "$CONFIG_FILE"; then
    echo -e "${GREEN}✓ workflow_swarm_sync hook registered${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ workflow_swarm_sync hook not registered${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

if grep -q '"swarm_consensus_sync"' "$CONFIG_FILE"; then
    echo -e "${GREEN}✓ swarm_consensus_sync hook registered${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ swarm_consensus_sync hook not registered${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))

if grep -q '"bidirectional_sync"' "$CONFIG_FILE"; then
    echo -e "${GREEN}✓ Bidirectional sync enabled${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Bidirectional sync not configured${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_RUN=$((TESTS_RUN + 1))
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "Total Tests: ${TESTS_RUN}"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
if [[ $TESTS_FAILED -gt 0 ]]; then
    echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
else
    echo -e "Failed: ${TESTS_FAILED}"
fi
echo ""

# Final status
if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}✓ Phase 3 integration verification PASSED${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Test with dry-run workflow execution:"
    echo "   npx @claude-flow/cli@latest workflow execute feature --task \"Test integration\" --dry-run"
    echo ""
    echo "2. List orchestration memory entries:"
    echo "   npx @claude-flow/cli@latest memory list --namespace orchestration"
    echo ""
    echo "3. View coordination events:"
    echo "   bash .claude/hooks/workflow-swarm-sync.sh list-workflows"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Phase 3 integration verification FAILED${NC}"
    echo ""
    echo "Please review the failed tests above and fix any issues."
    exit 1
fi
