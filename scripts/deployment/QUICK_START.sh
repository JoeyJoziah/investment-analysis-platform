#!/bin/bash

# ============================================================================
# Quick Start Script - Minimal setup to get started
# ============================================================================

echo "=================================================="
echo "Investment Analysis App - Quick Start"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please run this script from the project root."
    exit 1
fi

# Step 1: Install pip if not present
echo "Step 1: Checking pip installation..."
if ! python3 -m pip --version &> /dev/null; then
    echo "Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Step 2: Install minimal requirements
echo "Step 2: Installing minimal Python packages..."
python3 -m pip install --user \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    sqlalchemy==2.0.23 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    python-dotenv==1.0.0 \
    pandas==2.1.3 \
    numpy==1.26.2

# Step 3: Create necessary directories
echo "Step 3: Creating necessary directories..."
mkdir -p models
mkdir -p data/cache
mkdir -p data/exports
mkdir -p logs

# Step 4: Initialize the database (simple version)
echo "Step 4: Initializing database..."
python3 << 'EOF'
import sys
import os
sys.path.append(os.getcwd())

print("Attempting database initialization...")

try:
    from backend.utils.db_init import DatabaseInitializer
    db_init = DatabaseInitializer()
    if db_init.initialize():
        print("✓ Database initialized successfully!")
    else:
        print("⚠ Database initialization failed - may already be initialized")
except ImportError as e:
    print(f"⚠ Could not import database initializer: {e}")
    print("  You may need to initialize the database manually later")
except Exception as e:
    print(f"⚠ Database initialization error: {e}")
    print("  You may need to initialize the database manually later")
EOF

# Step 5: Create a simple test script
echo "Step 5: Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test if basic setup is working"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if basic imports work"""
    print("Testing imports...")
    
    try:
        import backend.api.main
        print("✓ Backend API module found")
    except ImportError as e:
        print(f"✗ Backend API module error: {e}")
    
    try:
        import backend.config.settings
        print("✓ Settings module found")
    except ImportError as e:
        print(f"✗ Settings module error: {e}")
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import error: {e}")
    
    try:
        import fastapi
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import error: {e}")

if __name__ == "__main__":
    test_imports()
    print("\nSetup test complete!")
EOF

chmod +x test_setup.py

# Step 6: Run the test
echo "Step 6: Testing setup..."
python3 test_setup.py

echo ""
echo "=================================================="
echo "Quick Start Complete!"
echo "=================================================="
echo ""
echo "To start the application:"
echo "  python3 -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "To run the full setup with all dependencies:"
echo "  bash SETUP_ENVIRONMENT.sh"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"