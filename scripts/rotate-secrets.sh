#!/bin/bash
# Secret Rotation Helper Script
# Generates new secrets for rotation after security breach
# Usage: ./scripts/rotate-secrets.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Secret Rotation Helper ===${NC}\n"
echo -e "${YELLOW}This script generates NEW secrets to replace the exposed ones.${NC}"
echo -e "${YELLOW}You must MANUALLY update your .env file and rotate API keys.${NC}\n"

# Generate secrets
echo -e "${BLUE}--- Generated Secrets ---${NC}\n"

echo "# Application Secrets (Copy these to your .env file)"
echo ""

echo "# SECRET_KEY (64 characters)"
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
echo ""

echo "# JWT_SECRET_KEY (64 characters)"
echo "JWT_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
echo ""

echo "# FERNET_KEY (base64 encoded)"
echo "FERNET_KEY=$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
echo ""

echo "# CSRF_SECRET_KEY (64 characters)"
echo "CSRF_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
echo ""

echo "# GDPR_ENCRYPTION_KEY (base64 encoded)"
echo "GDPR_ENCRYPTION_KEY=$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
echo ""

echo "# MASTER_SECRET_KEY (64 characters)"
echo "MASTER_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
echo ""

echo "# Database Passwords (16+ characters)"
echo "DB_PASSWORD=$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(24)))')"
echo "REDIS_PASSWORD=$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(24)))')"
echo "ELASTICSEARCH_PASSWORD=$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(24)))')"
echo ""

echo "# Airflow Password"
echo "AIRFLOW_DB_PASSWORD=$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(24)))')"
echo "AIRFLOW_ADMIN_PASSWORD=$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(24)))')"
echo ""

echo "# Grafana Password"
echo "GRAFANA_ADMIN_PASSWORD=$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(24)))')"
echo ""

echo -e "\n${BLUE}--- API Keys to Rotate Manually ---${NC}\n"
echo "1. Alpha Vantage: https://www.alphavantage.co/support/#api-key"
echo "2. Finnhub: https://finnhub.io/dashboard"
echo "3. Polygon: https://polygon.io/dashboard/api-keys"
echo "4. NewsAPI: https://newsapi.org/account"
echo "5. HuggingFace: https://huggingface.co/settings/tokens"
echo "6. Google Cloud: https://console.cloud.google.com/apis/credentials"
echo ""

echo -e "${YELLOW}--- Next Steps ---${NC}\n"
echo "1. Copy the generated secrets above to your .env file"
echo "2. Rotate API keys using the links above"
echo "3. Update database with new passwords (requires downtime)"
echo "4. Update Airflow with new credentials"
echo "5. Update Grafana with new admin password"
echo "6. Restart all services to use new secrets"
echo "7. Remove .env from git: git rm --cached .env"
echo "8. Add .env to .gitignore"
echo ""

echo -e "${GREEN}Secret generation complete!${NC}"
