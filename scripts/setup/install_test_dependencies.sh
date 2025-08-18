#!/bin/bash
# Installation script for test_all_passwords.py dependencies

echo "=================================================="
echo "Installing Dependencies for Password Test Script"
echo "=================================================="

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed. Please install Python 3 and pip first."
    exit 1
fi

# List of required packages
PACKAGES=(
    "psycopg2-binary==2.9.9"
    "redis==5.0.1"
    "requests==2.31.0"
    "python-dotenv==1.0.0"
)

echo ""
echo "Installing required packages..."
echo ""

# Install each package
for package in "${PACKAGES[@]}"; do
    echo "Installing $package..."
    pip3 install "$package"
    if [ $? -eq 0 ]; then
        echo "✅ $package installed successfully"
    else
        echo "❌ Failed to install $package"
        exit 1
    fi
    echo ""
done

echo "=================================================="
echo "✅ All dependencies installed successfully!"
echo "=================================================="
echo ""
echo "You can now run the test script with:"
echo "  python3 test_all_passwords.py"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "
try:
    import psycopg2
    import redis
    import requests
    from dotenv import load_dotenv
    print('✅ All imports verified successfully!')
except ImportError as e:
    print(f'❌ Import verification failed: {e}')
    exit(1)
"