#!/bin/bash

# ============================================================================
# Environment Setup Script for Investment Analysis App
# Fixes Python/pip issues and sets up the development environment
# ============================================================================

set -e  # Exit on error

echo "=================================================="
echo "Investment Analysis App - Environment Setup"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ============================================================================
# STEP 1: Check and Install Python Requirements
# ============================================================================

print_status "Step 1: Checking Python installation..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python3 found: $PYTHON_VERSION"
else
    print_error "Python3 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-dev
fi

# ============================================================================
# STEP 2: Install pip
# ============================================================================

print_status "Step 2: Checking pip installation..."

if ! python3 -m pip --version &> /dev/null; then
    print_warning "pip not found. Installing pip..."
    
    # Try to install python3-pip package
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-pip
    else
        # Alternative: Download get-pip.py
        print_warning "Installing pip using get-pip.py..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py --user
        rm get-pip.py
    fi
else
    PIP_VERSION=$(python3 -m pip --version)
    print_status "pip found: $PIP_VERSION"
fi

# ============================================================================
# STEP 3: Create and Activate Virtual Environment
# ============================================================================

print_status "Step 3: Setting up virtual environment..."

# Install venv if not available
if ! python3 -m venv --help &> /dev/null; then
    print_warning "Installing python3-venv..."
    sudo apt-get install -y python3-venv
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# ============================================================================
# STEP 4: Install System Dependencies
# ============================================================================

print_status "Step 4: Installing system dependencies..."

# Check if running on Linux/WSL
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ -f /proc/version ]] && grep -q Microsoft /proc/version; then
    print_status "Installing required system packages..."
    
    # Update package list
    sudo apt-get update
    
    # Install required packages
    sudo apt-get install -y \
        build-essential \
        libpq-dev \
        postgresql-client \
        redis-tools \
        curl \
        wget \
        git \
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libgdbm-dev \
        libnss3-dev \
        libedit-dev \
        libc6-dev
    
    print_status "System dependencies installed"
else
    print_warning "Not on Linux/WSL. Please install system dependencies manually."
fi

# ============================================================================
# STEP 5: Upgrade pip and Install Core Tools
# ============================================================================

print_status "Step 5: Upgrading pip and installing core tools..."

python3 -m pip install --upgrade pip setuptools wheel

# ============================================================================
# STEP 6: Install Python Requirements
# ============================================================================

print_status "Step 6: Installing Python requirements..."

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_status "Installing from requirements.txt..."
    
    # Install in chunks to avoid memory issues
    print_status "Installing core dependencies..."
    python3 -m pip install \
        fastapi==0.104.1 \
        uvicorn[standard]==0.24.0 \
        sqlalchemy==2.0.23 \
        alembic==1.12.1 \
        pydantic==2.5.0 \
        pydantic-settings==2.1.0
    
    print_status "Installing data processing dependencies..."
    python3 -m pip install \
        pandas==2.1.3 \
        numpy==1.26.2 \
        scipy==1.11.4 \
        scikit-learn==1.3.2
    
    print_status "Installing ML dependencies..."
    python3 -m pip install \
        torch==2.1.1 \
        transformers==4.35.2 \
        xgboost==2.0.2
    
    print_status "Installing remaining dependencies..."
    # Install remaining packages, ignoring duplicates and version conflicts
    python3 -m pip install -r requirements.txt --upgrade --no-deps 2>/dev/null || true
    
    # Now install with dependencies to resolve conflicts
    python3 -m pip install -r requirements.txt --upgrade
    
    print_status "Python requirements installed successfully"
else
    print_error "requirements.txt not found!"
fi

# ============================================================================
# STEP 7: Fix Script Files
# ============================================================================

print_status "Step 7: Fixing script files to use python3..."

# Fix shebang in Python scripts
for script in scripts/*.py; do
    if [ -f "$script" ]; then
        # Check if first line is a shebang
        first_line=$(head -n 1 "$script")
        if [[ "$first_line" == "#!"* ]]; then
            # Replace with python3 shebang
            sed -i '1s|.*|#!/usr/bin/env python3|' "$script"
            print_status "Fixed shebang in $(basename $script)"
        else
            # Add python3 shebang
            sed -i '1i#!/usr/bin/env python3' "$script"
            print_status "Added shebang to $(basename $script)"
        fi
        # Make executable
        chmod +x "$script"
    fi
done

# ============================================================================
# STEP 8: Initialize Database
# ============================================================================

print_status "Step 8: Initializing database..."

# Check if PostgreSQL is running
if command -v pg_isready &> /dev/null; then
    if pg_isready -h localhost -p 5432 &> /dev/null; then
        print_status "PostgreSQL is running"
        
        # Run database initialization
        if [ -f "scripts/init_database.py" ]; then
            print_status "Running database initialization..."
            python3 scripts/init_database.py || print_warning "Database initialization failed - may already be initialized"
        fi
    else
        print_warning "PostgreSQL is not running. Please start PostgreSQL first."
        print_warning "You can start it with: sudo service postgresql start"
    fi
else
    print_warning "PostgreSQL client not found. Please install PostgreSQL."
fi

# ============================================================================
# STEP 9: Download ML Models
# ============================================================================

print_status "Step 9: Setting up ML models..."

# Create models directory
mkdir -p models

# Check if download script exists
if [ -f "scripts/download_ml_models.py" ]; then
    print_status "Downloading ML models..."
    python3 scripts/download_ml_models.py || print_warning "ML model download failed - will download on first use"
elif [ -f "scripts/download_models.py" ]; then
    print_status "Using alternative model download script..."
    python3 scripts/download_models.py || print_warning "ML model download failed - will download on first use"
fi

# ============================================================================
# STEP 10: Create .env file if missing
# ============================================================================

print_status "Step 10: Checking environment configuration..."

if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from template..."
    
    cat > .env << 'EOL'
# API Keys - REPLACE WITH YOUR ACTUAL KEYS
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
NEWS_API_KEY=your_key_here

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/investment_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=investment_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
SECRET_KEY=your-secret-key-here-change-in-production
JWT_SECRET_KEY=your-jwt-secret-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
APP_ENV=development
DEBUG=True
LOG_LEVEL=INFO

# ML Settings
ML_MODELS_PATH=/app/ml_models
ENABLE_ML=true

# Email Settings (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
FROM_EMAIL=noreply@investmentapp.com
ENABLE_EMAIL=false
EOL
    
    print_warning "Created .env template. Please update with your actual API keys!"
else
    print_status ".env file exists"
fi

# ============================================================================
# STEP 11: Verify Installation
# ============================================================================

print_status "Step 11: Verifying installation..."

echo ""
echo "Python Version:"
python3 --version

echo ""
echo "Pip Version:"
python3 -m pip --version

echo ""
echo "Key Python Packages Installed:"
python3 -m pip list | grep -E "fastapi|sqlalchemy|pandas|torch|redis" || true

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=================================================="
print_status "Environment Setup Complete!"
echo "=================================================="

echo ""
echo "Next steps:"
echo "1. If you haven't already, update your .env file with actual API keys"
echo "2. Start PostgreSQL if not running: sudo service postgresql start"
echo "3. Start Redis if not running: sudo service redis-server start"
echo "4. Run database initialization: python3 scripts/init_database.py"
echo "5. Start the application: python3 -m uvicorn backend.api.main:app --reload"

echo ""
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"

echo ""
print_status "Setup complete! 🚀"