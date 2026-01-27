#!/bin/bash
###############################################################################
# SECRET GENERATION SCRIPT
# Purpose: Generate cryptographically secure secrets for rotation
# Safe to run - only generates values, doesn't modify anything
###############################################################################

set -euo pipefail

echo "=== SECRET GENERATION SCRIPT ==="
echo "Generated: $(date)"
echo ""
echo "Copy these values to your .env file"
echo "=================================="
echo ""

echo "# Application Core Secrets"
echo "SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")"
echo "JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")"
echo "FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")"
echo "SESSION_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")"
echo ""

echo "# Database Credentials (32 chars alphanumeric)"
echo "DB_PASSWORD=$(python3 -c "import secrets, string; chars=string.ascii_letters+string.digits; print(''.join(secrets.choice(chars) for _ in range(32)))")"
echo ""

echo "# Redis Credentials (32 chars alphanumeric)"
echo "REDIS_PASSWORD=$(python3 -c "import secrets, string; chars=string.ascii_letters+string.digits; print(''.join(secrets.choice(chars) for _ in range(32)))")"
echo ""

echo "# Monitoring Passwords"
echo "GRAFANA_PASSWORD=$(python3 -c "import secrets, string; chars=string.ascii_letters+string.digits+string.punctuation; print(''.join(secrets.choice(chars) for _ in range(24)))")"
echo "AIRFLOW_ADMIN_PASSWORD=$(python3 -c "import secrets, string; chars=string.ascii_letters+string.digits+string.punctuation; print(''.join(secrets.choice(chars) for _ in range(24)))")"
echo ""

echo "# API Keys Placeholder (Get from providers)"
echo "# ANTHROPIC_API_KEY=sk-ant-api03-xxxxx  # Get from: https://console.anthropic.com/settings/keys"
echo "# OPENAI_API_KEY=sk-xxxxx               # Get from: https://platform.openai.com/api-keys"
echo "# GOOGLE_API_KEY=xxxxx                  # Get from: https://console.cloud.google.com/apis/credentials"
echo "# ALPACA_API_KEY=xxxxx                  # Get from: https://app.alpaca.markets/paper/dashboard/overview"
echo "# ALPACA_SECRET_KEY=xxxxx               # Get from: https://app.alpaca.markets/paper/dashboard/overview"
echo "# ALPHA_VANTAGE_API_KEY=xxxxx           # Get from: https://www.alphavantage.co/support/#api-key"
echo "# FINNHUB_API_KEY=xxxxx                 # Get from: https://finnhub.io/dashboard"
echo "# NEWS_API_KEY=xxxxx                    # Get from: https://newsapi.org/account"
echo "# FRED_API_KEY=xxxxx                    # Get from: https://fred.stlouisfed.org/docs/api/api_key.html"
echo ""

echo "=================================="
echo "⚠️  SECURITY REMINDERS:"
echo "  1. Never commit these values to git"
echo "  2. Store in password manager or secrets vault"
echo "  3. Rotate secrets every 90 days"
echo "  4. Use environment-specific secrets (dev/staging/prod)"
echo "=================================="
