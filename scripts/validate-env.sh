#!/bin/bash
# Environment Validation Script
# Validates required environment variables are set with proper formats
# Usage: ./scripts/validate-env.sh [.env file]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default to .env if no argument provided
ENV_FILE="${1:-.env}"

# Check if file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}✗ Error: Environment file '$ENV_FILE' not found${NC}"
    exit 1
fi

echo -e "${BLUE}=== Environment Validation: $ENV_FILE ===${NC}\n"

# Load environment file
set -a
source "$ENV_FILE"
set +a

# Track validation status
ERRORS=0
WARNINGS=0

# Validation helper functions
check_required() {
    local var_name=$1
    local var_value="${!var_name}"

    if [ -z "$var_value" ]; then
        echo -e "${RED}✗ REQUIRED: $var_name is not set${NC}"
        ((ERRORS++))
        return 1
    else
        echo -e "${GREEN}✓ $var_name is set${NC}"
        return 0
    fi
}

check_placeholder() {
    local var_name=$1
    local var_value="${!var_name}"
    local placeholders=("your_" "CHANGE_THIS" "placeholder" "example" "test_" "TODO" "FIXME")

    for placeholder in "${placeholders[@]}"; do
        if [[ "$var_value" == *"$placeholder"* ]]; then
            echo -e "${YELLOW}⚠ WARNING: $var_name appears to use placeholder value${NC}"
            ((WARNINGS++))
            return 1
        fi
    done
    return 0
}

check_url_format() {
    local var_name=$1
    local var_value="${!var_name}"

    if [[ ! "$var_value" =~ ^https?:// ]] && [[ ! "$var_value" =~ ^postgresql:// ]] && [[ ! "$var_value" =~ ^redis:// ]]; then
        echo -e "${RED}✗ ERROR: $var_name has invalid URL format${NC}"
        ((ERRORS++))
        return 1
    fi
    return 0
}

check_port() {
    local var_name=$1
    local var_value="${!var_name}"

    if ! [[ "$var_value" =~ ^[0-9]+$ ]] || [ "$var_value" -lt 1 ] || [ "$var_value" -gt 65535 ]; then
        echo -e "${RED}✗ ERROR: $var_name must be a valid port (1-65535)${NC}"
        ((ERRORS++))
        return 1
    fi
    return 0
}

check_min_length() {
    local var_name=$1
    local var_value="${!var_name}"
    local min_length=$2

    if [ ${#var_value} -lt $min_length ]; then
        echo -e "${RED}✗ ERROR: $var_name must be at least $min_length characters${NC}"
        ((ERRORS++))
        return 1
    fi
    return 0
}

# 1. Application Core
echo -e "${BLUE}--- Application Core ---${NC}"
check_required "ENVIRONMENT"
check_required "SECRET_KEY" && check_min_length "SECRET_KEY" 64 && check_placeholder "SECRET_KEY"
check_required "JWT_SECRET_KEY" && check_min_length "JWT_SECRET_KEY" 64 && check_placeholder "JWT_SECRET_KEY"
check_required "FERNET_KEY" && check_placeholder "FERNET_KEY"

# Check ENVIRONMENT value
if [ -n "$ENVIRONMENT" ] && [[ ! "$ENVIRONMENT" =~ ^(development|staging|production|testing)$ ]]; then
    echo -e "${YELLOW}⚠ WARNING: ENVIRONMENT should be one of: development, staging, production, testing${NC}"
    ((WARNINGS++))
fi

echo ""

# 2. Database Configuration
echo -e "${BLUE}--- Database Configuration ---${NC}"
check_required "DATABASE_URL" && check_url_format "DATABASE_URL"
check_required "DB_PASSWORD" && check_min_length "DB_PASSWORD" 16 && check_placeholder "DB_PASSWORD"

# Check database URL doesn't contain localhost in production
if [ "$ENVIRONMENT" = "production" ] && [[ "$DATABASE_URL" == *"localhost"* ]]; then
    echo -e "${RED}✗ ERROR: DATABASE_URL contains 'localhost' in production${NC}"
    ((ERRORS++))
fi

echo ""

# 3. Redis Configuration
echo -e "${BLUE}--- Redis Configuration ---${NC}"
check_required "REDIS_URL" && check_url_format "REDIS_URL"
if [ -n "$REDIS_PASSWORD" ]; then
    check_min_length "REDIS_PASSWORD" 16 && check_placeholder "REDIS_PASSWORD"
fi

echo ""

# 4. API Keys (Production only)
if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${BLUE}--- API Keys (Production) ---${NC}"
    check_required "ALPHA_VANTAGE_API_KEY" && check_placeholder "ALPHA_VANTAGE_API_KEY"
    check_required "FINNHUB_API_KEY" && check_placeholder "FINNHUB_API_KEY"
    check_required "POLYGON_API_KEY" && check_placeholder "POLYGON_API_KEY"
    check_required "NEWS_API_KEY" && check_placeholder "NEWS_API_KEY"
    echo ""
fi

# 5. Security Settings
echo -e "${BLUE}--- Security Settings ---${NC}"

# Check SESSION_COOKIE_SECURE in production
if [ "$ENVIRONMENT" = "production" ]; then
    if [ "$SESSION_COOKIE_SECURE" != "true" ]; then
        echo -e "${RED}✗ ERROR: SESSION_COOKIE_SECURE must be 'true' in production${NC}"
        ((ERRORS++))
    else
        echo -e "${GREEN}✓ SESSION_COOKIE_SECURE is enabled${NC}"
    fi

    # Check FORCE_HTTPS
    if [ "$FORCE_HTTPS" != "true" ]; then
        echo -e "${YELLOW}⚠ WARNING: FORCE_HTTPS should be 'true' in production${NC}"
        ((WARNINGS++))
    fi
fi

# Check CSRF_SECRET_KEY
if [ -n "$CSRF_SECRET_KEY" ]; then
    check_min_length "CSRF_SECRET_KEY" 32 && check_placeholder "CSRF_SECRET_KEY"
fi

echo ""

# 6. Test Environment Specific
if [ "$ENVIRONMENT" = "testing" ]; then
    echo -e "${BLUE}--- Test Environment ---${NC}"

    # Ensure test database is used
    if [[ "$DATABASE_URL" != *"test"* ]] && [[ "$DATABASE_URL" != *":memory:"* ]]; then
        echo -e "${YELLOW}⚠ WARNING: DATABASE_URL should contain 'test' or use in-memory database${NC}"
        ((WARNINGS++))
    fi

    # Check MOCK_EXTERNAL_APIS
    if [ "$MOCK_EXTERNAL_APIS" != "true" ]; then
        echo -e "${YELLOW}⚠ WARNING: MOCK_EXTERNAL_APIS should be 'true' in testing${NC}"
        ((WARNINGS++))
    fi

    echo ""
fi

# 7. Port Validation
echo -e "${BLUE}--- Port Configuration ---${NC}"
if [ -n "$DB_PORT" ]; then check_port "DB_PORT"; fi
if [ -n "$REDIS_PORT" ]; then check_port "REDIS_PORT"; fi

echo ""

# Summary
echo -e "${BLUE}=== Validation Summary ===${NC}"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All validations passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Validation completed with $WARNINGS warning(s)${NC}"
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    exit 1
fi
