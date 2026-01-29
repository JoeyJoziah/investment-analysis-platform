#!/bin/bash

# Environment Setup Script for Investment Analysis Platform
# This script helps set up the production environment

set -e

echo "🚀 Investment Analysis Platform - Environment Setup"
echo "================================================="

# Function to generate secure keys
generate_key() {
    openssl rand -hex 32
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command_exists openssl; then
    echo "❌ openssl is required but not installed. Please install it."
    exit 1
fi

# Create .env.production if it doesn't exist
if [ ! -f .env.production ]; then
    echo "📝 Creating .env.production from template..."
    cp .env.production.template .env.production 2>/dev/null || cp .env.example .env.production
fi

# Generate secure keys
echo "🔐 Generating secure keys..."
SECRET_KEY=$(generate_key)
JWT_SECRET=$(generate_key)

# Update .env.production with generated keys
echo "📝 Updating .env.production with secure keys..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/SECRET_KEY=GENERATE_NEW_SECRET_KEY_HERE/SECRET_KEY=$SECRET_KEY/" .env.production
    sed -i '' "s/JWT_SECRET_KEY=GENERATE_NEW_JWT_SECRET_HERE/JWT_SECRET_KEY=$JWT_SECRET/" .env.production
else
    # Linux
    sed -i "s/SECRET_KEY=GENERATE_NEW_SECRET_KEY_HERE/SECRET_KEY=$SECRET_KEY/" .env.production
    sed -i "s/JWT_SECRET_KEY=GENERATE_NEW_JWT_SECRET_HERE/JWT_SECRET_KEY=$JWT_SECRET/" .env.production
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p infrastructure/ssl/certs
mkdir -p infrastructure/ssl/private
mkdir -p logs
mkdir -p data/backups

# Set up database password
echo ""
echo "🔐 Database Configuration"
echo "========================"
echo "Please enter a strong password for the PostgreSQL database:"
read -s DB_PASSWORD
echo ""

# Update database URL with password
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/CHANGE_THIS_PASSWORD/$DB_PASSWORD/" .env.production
else
    sed -i "s/CHANGE_THIS_PASSWORD/$DB_PASSWORD/" .env.production
fi

# API Key validation
echo ""
echo "🔑 API Key Validation"
echo "===================="
echo "Checking configured API keys..."

# Check each API key
check_api_key() {
    local key_name=$1
    local key_value=$(grep "^$key_name=" .env.production | cut -d'=' -f2)
    
    if [[ -z "$key_value" ]] || [[ "$key_value" == *"your_"*"_here"* ]]; then
        echo "⚠️  $key_name not configured"
        return 1
    else
        echo "✅ $key_name configured"
        return 0
    fi
}

missing_keys=0
for key in ALPHA_VANTAGE_API_KEY FINNHUB_API_KEY POLYGON_API_KEY NEWS_API_KEY; do
    if ! check_api_key $key; then
        ((missing_keys++))
    fi
done

# Optional API keys
echo ""
echo "📋 Optional API Keys (not required for basic functionality):"
for key in FMP_API_KEY MARKETAUX_API_KEY FRED_API_KEY OPENWEATHER_API_KEY; do
    check_api_key $key || true
done

# SSL Certificate information
echo ""
echo "🔒 SSL Certificate Setup"
echo "======================="
echo "For production deployment, you'll need SSL certificates."
echo "Options:"
echo "1. Use Let's Encrypt (recommended for production)"
echo "2. Use self-signed certificates (development only)"
echo "3. Use existing certificates"
echo ""
echo "Would you like to generate self-signed certificates for testing? (y/n)"
read -r GENERATE_SSL

if [[ "$GENERATE_SSL" == "y" ]]; then
    echo "Generating self-signed certificates..."
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout infrastructure/ssl/private/investment-analysis.key \
        -out infrastructure/ssl/certs/investment-analysis.crt \
        -days 365 \
        -subj "/C=US/ST=State/L=City/O=Investment Analysis/CN=investment-analysis.com"
    echo "✅ Self-signed certificates generated"
fi

# Create Kubernetes secrets file
echo ""
echo "☸️  Creating Kubernetes secrets template..."
cat > infrastructure/kubernetes/secrets.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: investment-api-secrets
  namespace: investment-analysis
type: Opaque
stringData:
  # Copy values from .env.production
  ALPHA_VANTAGE_API_KEY: "$(grep '^ALPHA_VANTAGE_API_KEY=' .env.production | cut -d'=' -f2)"
  FINNHUB_API_KEY: "$(grep '^FINNHUB_API_KEY=' .env.production | cut -d'=' -f2)"
  POLYGON_API_KEY: "$(grep '^POLYGON_API_KEY=' .env.production | cut -d'=' -f2)"
  NEWS_API_KEY: "$(grep '^NEWS_API_KEY=' .env.production | cut -d'=' -f2)"
  SECRET_KEY: "$SECRET_KEY"
  JWT_SECRET_KEY: "$JWT_SECRET"
  DATABASE_PASSWORD: "$DB_PASSWORD"
EOF

# Summary
echo ""
echo "✅ Environment Setup Complete!"
echo "=============================="
echo ""
echo "📋 Summary:"
echo "- ✅ Production environment file created: .env.production"
echo "- ✅ Secure keys generated"
echo "- ✅ Database password configured"
if [ $missing_keys -eq 0 ]; then
    echo "- ✅ All required API keys configured"
else
    echo "- ⚠️  $missing_keys required API keys missing"
fi
echo ""
echo "📌 Next Steps:"
echo "1. Review and update .env.production with any missing API keys"
echo "2. Update domain names and URLs for your deployment"
echo "3. Configure SSL certificates for production"
echo "4. Run validation: python debug_validate.py"
echo "5. Build Docker images: docker-compose -f docker-compose.prod.yml build"
echo ""
echo "⚠️  Security Reminders:"
echo "- Never commit .env.production to version control"
echo "- Keep all API keys and secrets secure"
echo "- Use strong passwords for database"
echo "- Enable firewall rules for production servers"
echo ""
echo "For production SSL setup with Let's Encrypt, run:"
echo "  certbot certonly --standalone -d api.investment-analysis.com -d investment-analysis.com"