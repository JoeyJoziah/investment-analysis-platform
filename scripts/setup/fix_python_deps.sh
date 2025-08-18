#!/bin/bash
"""
Python Dependency Fix Script for WSL Investment Analysis Platform
Fixes ModuleNotFoundError issues and sets up proper Python environment
"""

set -e  # Exit on any error

PROJECT_ROOT="/mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/7b51113a393465a37d4f1fda36b4d190088ac69ea8d5cf2f90400b3c14148ad3"
VENV_PATH="$PROJECT_ROOT/venv"

echo "🔧 Python Dependency Fix Script"
echo "================================"

# Step 1: Navigate to project directory
echo "📁 Navigating to project directory..."
cd "$PROJECT_ROOT"

# Step 2: Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found. Creating new one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment found"
fi

# Step 3: Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Step 4: Verify activation
echo "🔍 Verifying virtual environment..."
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo "Python version: $(python --version)"

# Step 5: Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Step 6: Install dependencies based on user preference
echo ""
echo "📦 Available dependency installation options:"
echo "1. Minimal (23 packages) - For quick testing"
echo "2. Core (16 packages) - Recommended for development"
echo "3. Full (131 packages) - Complete feature set"
echo "4. Just aiohttp - Fix the immediate error only"
echo ""

# For automation, install core dependencies
echo "🚀 Installing core dependencies (recommended)..."
if [ -f "requirements-core.txt" ]; then
    pip install -r requirements-core.txt
    echo "✅ Core dependencies installed"
elif [ -f "requirements-minimal.txt" ]; then
    pip install -r requirements-minimal.txt
    echo "✅ Minimal dependencies installed"
else
    # Fallback: install just aiohttp to fix immediate error
    echo "⚠️ Installing aiohttp only to fix immediate error..."
    pip install aiohttp==3.9.1
    echo "✅ aiohttp installed"
fi

# Step 7: Verify aiohttp installation
echo "🧪 Testing aiohttp import..."
python -c "import aiohttp; print(f'✅ aiohttp {aiohttp.__version__} installed successfully')" || echo "❌ aiohttp import failed"

# Step 8: Test background_loader_enhanced.py imports
echo "🧪 Testing background_loader_enhanced.py imports..."
python -c "
try:
    import sys
    sys.path.append('.')
    import aiohttp, asyncio, json, logging, multiprocessing, os, pickle, psutil, sys, time, pandas as pd
    print('✅ All critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('💡 You may need to install additional dependencies')
"

# Step 9: Create activation helper script
cat > activate_env.sh << 'EOF'
#!/bin/bash
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/7b51113a393465a37d4f1fda36b4d190088ac69ea8d5cf2f90400b3c14148ad3
source venv/bin/activate
echo "✅ Investment Analysis Platform environment activated"
echo "Python: $(which python)"
echo "Ready to run: python background_loader_enhanced.py"
EOF

chmod +x activate_env.sh

echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo "✅ Virtual environment: $VENV_PATH"
echo "✅ Dependencies installed"
echo "✅ aiohttp module available"
echo ""
echo "🚀 Next Steps:"
echo "1. Always activate virtual environment first:"
echo "   source venv/bin/activate"
echo "   # OR use the helper script:"
echo "   ./activate_env.sh"
echo ""
echo "2. Run your script:"
echo "   python background_loader_enhanced.py"
echo ""
echo "3. To install more dependencies later:"
echo "   pip install -r requirements.txt  # Full dependencies"
echo "   pip install package_name          # Individual packages"
echo ""
echo "⚠️  NEVER use 'apt install' for Python packages!"
echo "   Always use 'pip install' inside the activated virtual environment"