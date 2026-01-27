#!/bin/bash

# Generate secure secrets for Investment Analysis Platform
# Usage: ./scripts/generate_secrets.sh > .env

echo "# Investment Analysis Platform - Generated Environment Variables"
echo "# Generated on: $(date)"
echo "# WARNING: Keep these secrets secure and never commit to version control"
echo ""

echo "# ====================
# Core Application
# ====================
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO"

echo ""
echo "# Security Keys (Generated with openssl)"
echo "SECRET_KEY=$(openssl rand -hex 32)"
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)"
echo "SESSION_SECRET_KEY=$(openssl rand -hex 32)"

echo ""
echo "# ====================
# Database Configuration
# ====================
DB_HOST=postgres
DB_PORT=5432
DB_NAME=investment_db
DB_USER=postgres"
echo "DB_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
echo 'DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}'

echo ""
echo "# Redis"
echo "REDIS_HOST=redis
REDIS_PORT=6379"
echo "REDIS_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
echo 'REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/0'

echo ""
echo "# Elasticsearch"
echo "ELASTICSEARCH_URL=http://localhost:9200"
echo "ELASTIC_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"

echo ""
echo "# ====================
# API Keys (Replace with actual keys)
# ====================
# Alpha Vantage - https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=

# Finnhub - https://finnhub.io/register
FINNHUB_API_KEY=

# Polygon.io - https://polygon.io/dashboard/signup
POLYGON_API_KEY=

# Financial Modeling Prep - https://site.financialmodelingprep.com/developer/docs
FMP_API_KEY=

# NewsAPI - https://newsapi.org/register
NEWS_API_KEY=

# MarketAux - https://www.marketaux.com/
MARKETAUX_API_KEY=

# FRED - https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=

# OpenWeather - https://openweathermap.org/api
OPENWEATHER_API_KEY="

echo ""
echo "# ====================
# Service Passwords
# ====================
# Airflow"
echo "AIRFLOW_DB_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
echo "AIRFLOW_SECRET_KEY=$(openssl rand -hex 32)"

# Generate Fernet key for Airflow
echo "# Fernet key for Airflow (generated with Python cryptography)"
echo "FERNET_KEY=$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' 2>/dev/null || echo 'GENERATE_WITH_PYTHON_CRYPTOGRAPHY')"

echo ""
echo "# Grafana"
echo "GRAFANA_USER=admin"
echo "GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d '=+/')"
echo "GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16 | tr -d '=+/')"

echo ""
echo "# RabbitMQ (Production only)"
echo "RABBITMQ_USER=admin"
echo "RABBITMQ_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"

echo ""
echo "# ====================
# Development/Test Specific
# ====================
# Development Database"
echo "DEV_DB_PASSWORD=$(openssl rand -base64 16 | tr -d '=+/')"

echo ""
echo "# Test Database"
echo "TEST_DB_PASSWORD=$(openssl rand -base64 16 | tr -d '=+/')"
echo "TEST_JWT_SECRET_KEY=$(openssl rand -hex 32)"

echo ""
echo "# PgAdmin (Development only)"
echo "PGADMIN_PASSWORD=$(openssl rand -base64 16 | tr -d '=+/')"

echo ""
echo "# ====================
# Email Configuration (Optional)
# ====================
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD="

echo ""
echo "# ====================
# Monitoring & Analytics (Optional)
# ====================
SENTRY_DSN=
NEW_RELIC_LICENSE_KEY=
DATADOG_API_KEY="

echo ""
echo "# ====================
# Cloud Storage (Optional)
# ====================
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_S3_BUCKET="

echo ""
echo "# ====================
# Feature Flags
# ====================
ENABLE_ALTERNATIVE_DATA=true
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_TECHNICAL_ANALYSIS=true
ENABLE_FUNDAMENTAL_ANALYSIS=true
ENABLE_ML_PREDICTIONS=true
ENABLE_PORTFOLIO_OPTIMIZATION=true"

echo ""
echo "# ====================
# Cost Management
# ====================
MONTHLY_BUDGET_USD=50.0
ALERT_THRESHOLD_PERCENT=80"

echo ""
echo "# ====================
# Performance Settings
# ====================
MAX_WORKERS=4
ASYNC_TIMEOUT_SECONDS=30
API_TIMEOUT_SECONDS=60
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=10000"