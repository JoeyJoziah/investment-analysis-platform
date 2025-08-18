#!/bin/bash
# Investment Platform Setup Script
# Simplified setup for development and production environments

set -e

echo "🚀 Investment Platform Setup"
echo "=========================="

# Check for .env file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f .env.template ]; then
        cp .env.template .env
        echo "✅ .env file created. Please update with your API keys."
    else
        echo "❌ Error: .env.template not found"
        exit 1
    fi
fi

# Generate secure passwords if not set
if grep -q "CHANGE_ME" .env; then
    echo "🔐 Generating secure passwords..."
    DB_PASSWORD=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    SECRET_KEY=$(openssl rand -hex 32)
    JWT_SECRET=$(openssl rand -hex 32)
    
    sed -i "s/DB_PASSWORD=CHANGE_ME/DB_PASSWORD=$DB_PASSWORD/g" .env
    sed -i "s/REDIS_PASSWORD=CHANGE_ME/REDIS_PASSWORD=$REDIS_PASSWORD/g" .env
    sed -i "s/SECRET_KEY=CHANGE_ME/SECRET_KEY=$SECRET_KEY/g" .env
    sed -i "s/JWT_SECRET_KEY=CHANGE_ME/JWT_SECRET_KEY=$JWT_SECRET/g" .env
    
    echo "✅ Secure passwords generated"
fi

# Setup Python virtual environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and install dependencies
if [ -f requirements.txt ]; then
    echo "📦 Installing Python dependencies..."
    source venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo "✅ Python dependencies installed"
fi

# Install frontend dependencies (if npm is available)
if [ -d frontend/web ] && command -v npm &> /dev/null; then
    echo "📦 Installing frontend dependencies..."
    cd frontend/web
    npm install --silent 2>/dev/null || npm install
    cd ../..
    echo "✅ Frontend dependencies installed"
elif [ -d frontend/web ]; then
    echo "⚠️ npm not found. Skipping frontend dependency installation."
    echo "   Frontend dependencies will be installed in the Docker container."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs data models/trained archive
echo "✅ Directories created"

# Initialize database (if Docker is available)
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "🗄️ Initializing database..."
    docker-compose up -d postgres redis
    sleep 5
    docker-compose exec -T postgres psql -U postgres -c "CREATE DATABASE investment_db;" 2>/dev/null || echo "Database may already exist"
    echo "✅ Database services started"
    
    # Note about migrations
    echo "📝 Note: Run database migrations after starting the backend service"
else
    echo "⚠️ Docker/Docker Compose not found. Skipping database initialization."
    echo "   Please install Docker and run this script again."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update API keys in .env file (if not already set)"
echo "2. Run: ./start.sh dev"
echo "3. Access the application at http://localhost:3000"
echo ""
echo "For Python development, activate the virtual environment:"
echo "  source venv/bin/activate"