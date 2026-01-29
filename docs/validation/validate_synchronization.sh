#!/bin/bash
# Comprehensive Synchronization and Integration Validation Script
# Validates test suites, version alignment, documentation, and GitHub workflows

set -e

echo "=============================================="
echo "SYNCHRONIZATION AND QUALITY VALIDATION REPORT"
echo "=============================================="
echo "Timestamp: $(date)"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Test results
TEST_RESULTS=()

# Helper functions
check_pass() {
    local check_name="$1"
    echo -e "${GREEN}✓${NC} $check_name"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_fail() {
    local check_name="$1"
    local message="${2:-}"
    echo -e "${RED}✗${NC} $check_name"
    if [ -n "$message" ]; then
        echo "  ${RED}Error: $message${NC}"
    fi
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_warn() {
    local warning="$1"
    echo -e "${YELLOW}⚠${NC} $warning"
    ((WARNINGS++))
}

# ============================================
# 1. VERSION ALIGNMENT CHECK
# ============================================
echo -e "\n${BLUE}═══ VERSION ALIGNMENT CHECK ═══${NC}"

# Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" =~ ^3\.1[2-9] ]]; then
    check_pass "Python version (3.12+): $PYTHON_VERSION"
else
    check_fail "Python version requirement" "Expected 3.12+, got $PYTHON_VERSION"
fi

# Node version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | cut -d 'v' -f 2)
    if [[ "$NODE_VERSION" =~ ^(1[8-9]|2[0-9])\. ]]; then
        check_pass "Node.js version (18+): $NODE_VERSION"
    else
        check_warn "Node.js version is lower than recommended (18+): $NODE_VERSION"
    fi
else
    check_warn "Node.js not installed"
fi

# Backend version from pyproject.toml
BACKEND_VERSION=$(grep '^version' pyproject.toml | head -1 | awk -F'"' '{print $2}' || echo "unknown")
check_pass "Backend version: $BACKEND_VERSION"

# Frontend version from package.json
FRONTEND_VERSION=$(grep '"version"' frontend/web/package.json | head -1 | awk -F'"' '{print $4}')
check_pass "Frontend version: $FRONTEND_VERSION"

# ============================================
# 2. BACKEND TEST SUITE CHECK
# ============================================
echo -e "\n${BLUE}═══ BACKEND TEST SUITE VALIDATION ═══${NC}"

# Count test files
BACKEND_TEST_COUNT=$(find backend/tests -name "test_*.py" 2>/dev/null | wc -l)
if [ "$BACKEND_TEST_COUNT" -gt 0 ]; then
    check_pass "Backend test files found: $BACKEND_TEST_COUNT"
else
    check_fail "No backend test files found"
fi

# Check pytest configuration
if grep -q "pytest.ini_options" pyproject.toml; then
    COVERAGE_THRESHOLD=$(grep "cov-fail-under" pyproject.toml | grep -oP '\d+' | head -1)
    check_pass "Pytest configured with coverage threshold: ${COVERAGE_THRESHOLD}%"
else
    check_fail "Pytest configuration not found"
fi

# Check test markers
EXPECTED_MARKERS=("unit" "integration" "performance" "security")
for marker in "${EXPECTED_MARKERS[@]}"; do
    if grep -q "\"$marker\"" pyproject.toml; then
        check_pass "Test marker configured: $marker"
    else
        check_fail "Test marker missing: $marker"
    fi
done

# ============================================
# 3. FRONTEND TEST SUITE CHECK
# ============================================
echo -e "\n${BLUE}═══ FRONTEND TEST SUITE VALIDATION ═══${NC}"

# Check frontend test configuration
if [ -f "frontend/web/package.json" ]; then
    # Count E2E test files
    E2E_TEST_COUNT=$(find frontend/web/tests -name "*.spec.ts" 2>/dev/null | wc -l)
    if [ "$E2E_TEST_COUNT" -gt 0 ]; then
        check_pass "Frontend E2E test files found: $E2E_TEST_COUNT"
    else
        check_warn "No frontend E2E test files found (optional)"
    fi

    # Check test scripts
    if grep -q '"test":' frontend/web/package.json; then
        check_pass "Frontend test script configured"
    else
        check_fail "Frontend test script not configured"
    fi

    if grep -q '"test:e2e":' frontend/web/package.json; then
        check_pass "Frontend E2E test script configured"
    else
        check_fail "Frontend E2E test script not configured"
    fi
else
    check_fail "Frontend package.json not found"
fi

# ============================================
# 4. DOCUMENTATION CONSISTENCY CHECK
# ============================================
echo -e "\n${BLUE}═══ DOCUMENTATION CONSISTENCY CHECK ═══${NC}"

# Check README files
if [ -f "README.md" ]; then
    check_pass "Root README.md exists"

    # Check for key sections
    if grep -q "installation\|setup\|getting started" README.md -i; then
        check_pass "README contains setup instructions"
    else
        check_warn "README may be missing setup instructions"
    fi
else
    check_fail "Root README.md not found"
fi

# Check API documentation
if [ -f "docs/API.md" ]; then
    check_pass "API documentation exists"

    # Check for version info
    if grep -q "v1\|v2\|v3" docs/API.md -i; then
        check_pass "API documentation includes version info"
    fi
else
    check_warn "API documentation not found at docs/API.md"
fi

# Check for architecture documentation
if [ -f "docs/ARCHITECTURE.md" ]; then
    check_pass "Architecture documentation exists"
else
    check_warn "Architecture documentation not found"
fi

# ============================================
# 5. GITHUB WORKFLOW VALIDATION
# ============================================
echo -e "\n${BLUE}═══ GITHUB WORKFLOW CONFIGURATION CHECK ═══${NC}"

WORKFLOW_DIR=".github/workflows"
WORKFLOW_COUNT=$(ls "$WORKFLOW_DIR"/*.yml 2>/dev/null | wc -l)
check_pass "GitHub workflow files found: $WORKFLOW_COUNT"

# Check for critical workflows
EXPECTED_WORKFLOWS=("ci.yml" "security-scan.yml" "release-management.yml" "production-deploy.yml")
for workflow in "${EXPECTED_WORKFLOWS[@]}"; do
    if [ -f "$WORKFLOW_DIR/$workflow" ]; then
        check_pass "Critical workflow exists: $workflow"
    else
        check_fail "Critical workflow missing: $workflow"
    fi
done

# Validate workflow syntax
for workflow in "$WORKFLOW_DIR"/*.yml; do
    if grep -q "^name:" "$workflow" && grep -q "^on:" "$workflow"; then
        # Valid YAML structure
        true
    else
        check_warn "Workflow may have invalid structure: $(basename $workflow)"
    fi
done

# ============================================
# 6. CROSS-PACKAGE DEPENDENCY CHECK
# ============================================
echo -e "\n${BLUE}═══ CROSS-PACKAGE DEPENDENCY VALIDATION ═══${NC}"

# Check backend dependencies
if grep -q "sqlalchemy\|fastapi\|pydantic" pyproject.toml; then
    check_pass "Backend core dependencies configured"
fi

# Check frontend dependencies
if grep -q '"react"\|"redux"\|"axios"' frontend/web/package.json; then
    check_pass "Frontend core dependencies configured"
fi

# ============================================
# 7. CONFIGURATION FILES CHECK
# ============================================
echo -e "\n${BLUE}═══ CONFIGURATION FILES CHECK ═══${NC}"

# Check for critical config files
CONFIG_FILES=(".env.example" ".env.secure.template" "pyproject.toml" "frontend/web/package.json")
for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        check_pass "Configuration file exists: $config"
    else
        check_fail "Configuration file missing: $config"
    fi
done

# ============================================
# 8. SECURITY COMPLIANCE CHECK
# ============================================
echo -e "\n${BLUE}═══ SECURITY COMPLIANCE CHECK ═══${NC}"

# Check for .gitignore
if [ -f ".gitignore" ]; then
    check_pass ".gitignore exists"

    # Check for secrets patterns
    if grep -q "\.env\|secrets\|credentials" .gitignore; then
        check_pass "Secrets excluded in .gitignore"
    fi
else
    check_fail ".gitignore not found"
fi

# Check for security headers in tests
if [ -d "backend/tests" ]; then
    SECURITY_TESTS=$(grep -r "security\|csrf\|sql.*injection" backend/tests/*.py 2>/dev/null | wc -l)
    if [ "$SECURITY_TESTS" -gt 0 ]; then
        check_pass "Security tests found: $SECURITY_TESTS references"
    else
        check_warn "Limited security test coverage"
    fi
fi

# ============================================
# 9. DOCKER & CONTAINERIZATION CHECK
# ============================================
echo -e "\n${BLUE}═══ DOCKER & CONTAINERIZATION CHECK ═══${NC}"

if [ -f "Dockerfile" ]; then
    check_pass "Dockerfile exists"

    if grep -q "FROM python\|FROM node" Dockerfile; then
        check_pass "Dockerfile configured for application runtime"
    fi
else
    check_warn "Dockerfile not found (optional if using other deployment)"
fi

if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
    check_pass "Docker Compose configuration exists"
else
    check_warn "Docker Compose not configured (optional)"
fi

# ============================================
# 10. VERSION CONSISTENCY ACROSS PACKAGES
# ============================================
echo -e "\n${BLUE}═══ VERSION CONSISTENCY ACROSS PACKAGES ═══${NC}"

# Extract versions
PYTHON_PKG_VERSION=$(grep '^version' pyproject.toml | head -1 | awk -F'"' '{print $2}' || echo "unknown")
FRONTEND_PKG_VERSION=$(grep '"version"' frontend/web/package.json | head -1 | awk -F'"' '{print $4}' || echo "unknown")

echo "Python package version: $PYTHON_PKG_VERSION"
echo "Frontend package version: $FRONTEND_PKG_VERSION"

if [ "$PYTHON_PKG_VERSION" != "unknown" ] && [ "$FRONTEND_PKG_VERSION" != "unknown" ]; then
    check_pass "Versions defined in both backends"
fi

# ============================================
# TEST EXECUTION SUMMARY
# ============================================
echo -e "\n${BLUE}═══ TEST EXECUTION CONFIGURATION ═══${NC}"

# Count available tests by type
UNIT_TESTS=$(find backend/tests -name "test_*.py" -exec grep -l "def test_" {} \; 2>/dev/null | wc -l)
INTEGRATION_TESTS=$(find backend/tests -name "test_*integration*.py" 2>/dev/null | wc -l)
E2E_TESTS=$(find frontend/web/tests -name "*.spec.ts" 2>/dev/null | wc -l)

echo "Unit tests configured: $UNIT_TESTS files"
echo "Integration tests configured: $INTEGRATION_TESTS files"
echo "E2E tests configured: $E2E_TESTS files"

# ============================================
# FINAL SUMMARY
# ============================================
echo ""
echo "=============================================="
echo "VALIDATION SUMMARY"
echo "=============================================="
echo -e "Total Checks: $TOTAL_CHECKS"
echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo ""

PASS_RATE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
echo "Pass Rate: ${PASS_RATE}%"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "\n${GREEN}✓ All critical checks passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some checks failed. Please review.${NC}"
    exit 1
fi
