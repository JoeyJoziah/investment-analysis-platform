#!/bin/bash
# Install critical missing dependencies for the investment platform

echo "Installing critical missing dependencies..."

# Activate virtual environment
source ./venv/bin/activate

# Install missing packages one by one with shorter timeout
echo "Installing selenium..."
pip install selenium --timeout 30 --no-cache-dir || echo "Failed to install selenium"

echo "Installing lightgbm..."
pip install lightgbm --timeout 30 --no-cache-dir || echo "Failed to install lightgbm"

echo "Installing optuna..."
pip install optuna --timeout 30 --no-cache-dir || echo "Failed to install optuna"

echo "Installation attempt complete. Checking installed packages..."
echo "Installed critical packages:"
pip list | grep -E "(selenium|torch|transformers|lightgbm|optuna)" || echo "Check failed"

echo "Done!"