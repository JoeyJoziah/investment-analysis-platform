#!/bin/bash
# Investment Analysis App - Complete Setup Script

set -e  # Exit on any error

echo "🚀 Setting up Investment Analysis App"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

print_status "Python 3 is available"

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Activated virtual environment"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first
print_status "Installing core dependencies..."
pip install python-dotenv sqlalchemy[asyncio] asyncpg psycopg2-binary redis pydantic fastapi uvicorn cryptography

# Create necessary directories
print_status "Creating directories..."
mkdir -p secrets logs data data/cache reports backend/ml/models scripts

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file..."
    cat > .env << 'EOF'
# Development Environment
DEBUG=true
ENVIRONMENT=development

# Security
MASTER_SECRET_KEY=dev-master-secret-32-chars-long!!
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production

# Local paths
SECRETS_DIR=./secrets
LOGS_DIR=./logs
DATA_DIR=./data

# Database (adjust for your local setup)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/investment_db_dev
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_db_dev
DB_USER=postgres
DB_PASSWORD=postgres

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys (replace with actual keys)
ALPHA_VANTAGE_API_KEY=demo
FINNHUB_API_KEY=demo
POLYGON_API_KEY=demo
NEWS_API_KEY=demo
EOF
    print_status "Created .env file"
else
    print_status ".env file already exists"
fi

# Create simple test script
cat > scripts/test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test basic setup"""
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment loaded")
except ImportError:
    print("⚠️  dotenv not available")

# Test basic imports
try:
    import sqlalchemy
    print("✓ SQLAlchemy available")
except ImportError:
    print("❌ SQLAlchemy not available")

try:
    import redis
    print("✓ Redis available")
except ImportError:
    print("❌ Redis not available")

try:
    import fastapi
    print("✓ FastAPI available")
except ImportError:
    print("❌ FastAPI not available")

# Test secrets directory
secrets_dir = Path(os.getenv("SECRETS_DIR", "./secrets"))
if secrets_dir.exists():
    print(f"✓ Secrets directory exists: {secrets_dir}")
else:
    print(f"❌ Secrets directory missing: {secrets_dir}")

print("\n🎉 Basic setup validation complete!")
EOF

# Make test script executable
chmod +x scripts/test_setup.py

# Run test
print_status "Testing basic setup..."
python scripts/test_setup.py

echo ""
print_status "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Install full dependencies: pip install -r requirements-clean.txt"
echo "3. Set up PostgreSQL and Redis locally"
echo "4. Update API keys in .env file"
echo "5. Test: python scripts/test_setup.py"
echo "6. Start app: python -m uvicorn backend.api.main:app --reload"
echo ""
print_warning "Make sure to update your API keys in the .env file!"