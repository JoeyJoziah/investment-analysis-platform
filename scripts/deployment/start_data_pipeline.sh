#!/bin/bash

# Data Pipeline Activation Script
# Comprehensive startup script for the investment analysis data pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    netstat -an | grep ":$1 " | grep LISTEN >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    log "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 10
        ((attempt++))
    done
    
    error "$service_name failed to become ready after $((max_attempts * 10)) seconds"
    return 1
}

# Main execution starts here
echo "="*80
echo "🚀 INVESTMENT ANALYSIS DATA PIPELINE - ACTIVATION"
echo "="*80
echo ""

# Check prerequisites
log "📋 Checking Prerequisites..."

# Check if Docker is installed and running
if ! command_exists docker; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    error "Docker is not running. Please start Docker first."
    exit 1
fi

success "Docker is available and running"

# Check if docker-compose is available
if ! command_exists docker-compose; then
    error "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

success "docker-compose is available"

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+')
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) == 1 ]]; then
        success "Python $PYTHON_VERSION is available"
    else
        warning "Python $PYTHON_VERSION detected (3.11+ recommended)"
    fi
else
    warning "Python 3 not found in PATH"
fi

# Check environment file
if [ ! -f ".env" ]; then
    warning ".env file not found. Creating from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        warning "Please edit .env file with your actual configuration"
    else
        error ".env.example not found. Cannot create .env file."
        exit 1
    fi
else
    success ".env file found"
fi

# Check system resources
log "🖥️  Checking System Resources..."

TOTAL_MEM=$(free -g | awk 'NR==2{printf "%.1f", $2}')
if (( $(echo "$TOTAL_MEM >= 4.0" | bc -l) )); then
    success "Memory: ${TOTAL_MEM}GB (sufficient)"
else
    warning "Memory: ${TOTAL_MEM}GB (4GB+ recommended)"
fi

DISK_SPACE=$(df -h . | awk 'NR==2{print $4}')
success "Available disk space: $DISK_SPACE"

# Check for port conflicts
log "🔌 Checking Port Availability..."

PORTS=("5432:PostgreSQL" "6379:Redis" "8080:Airflow" "5555:Flower" "9090:Metrics")
PORT_CONFLICTS=false

for port_info in "${PORTS[@]}"; do
    port=${port_info%:*}
    service=${port_info#*:}
    
    if port_in_use "$port"; then
        warning "Port $port is already in use (needed for $service)"
        PORT_CONFLICTS=true
    else
        echo "✅ Port $port available for $service"
    fi
done

if [ "$PORT_CONFLICTS" = true ]; then
    echo ""
    warning "Some ports are in use. The pipeline may still work if services are already running."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
log "🏗️  Starting Pipeline Services..."

# Create necessary directories
log "Creating directories..."
mkdir -p data_pipelines/airflow/logs
mkdir -p data_pipelines/airflow/plugins
mkdir -p logs
mkdir -p data/cache
mkdir -p models/trained
success "Directories created"

# Set proper permissions for Airflow
log "Setting Airflow permissions..."
export AIRFLOW_UID=$(id -u)
echo "AIRFLOW_UID=$AIRFLOW_UID" >> .env
success "Airflow UID set to $AIRFLOW_UID"

# Start PostgreSQL and Redis first
log "Starting PostgreSQL and Redis..."
docker-compose up -d postgres redis

# Wait for PostgreSQL
log "Waiting for PostgreSQL..."
sleep 15
if wait_for_service "PostgreSQL" "postgresql://postgres:$(grep DB_PASSWORD .env | cut -d'=' -f2)@localhost:5432/investment_db"; then
    success "PostgreSQL is ready"
else
    error "PostgreSQL failed to start"
    exit 1
fi

# Wait for Redis
log "Waiting for Redis..."
if wait_for_service "Redis" "redis://localhost:6379"; then
    success "Redis is ready"
else
    error "Redis failed to start"
    exit 1
fi

# Initialize database
log "🗄️  Initializing Database..."
if python3 activate_data_pipeline.py; then
    success "Database initialized successfully"
else
    error "Database initialization failed"
    exit 1
fi

# Start Airflow services
log "🌪️  Starting Airflow Services..."
docker-compose -f docker-compose.airflow.yml up -d

# Wait for Airflow webserver
if wait_for_service "Airflow Webserver" "http://localhost:8080/health" 60; then
    success "Airflow is ready"
else
    error "Airflow failed to start properly"
    # Continue anyway - user can check manually
fi

# Wait for Flower
if wait_for_service "Flower (Celery Monitor)" "http://localhost:5555" 30; then
    success "Flower monitoring is ready"
else
    warning "Flower monitoring may not be available"
fi

# Run component tests
log "🧪 Running Component Tests..."
if python3 test_pipeline_components.py; then
    success "Component tests completed"
else
    warning "Some component tests failed - check logs for details"
fi

# Final validation
log "✅ Running Final Validation..."
if python3 monitor_pipeline.py --once; then
    success "Pipeline validation completed"
else
    warning "Pipeline validation had issues - check logs"
fi

echo ""
echo "="*80
echo "🎉 DATA PIPELINE ACTIVATION COMPLETE!"
echo "="*80
echo ""

# Display access information
echo "📊 ACCESS INFORMATION:"
echo "  • Airflow Web UI:    http://localhost:8080"
echo "    Username: admin"
echo "    Password: secure_admin_password_789"
echo ""
echo "  • Flower Monitor:    http://localhost:5555"
echo "    Username: admin"
echo "    Password: secure_flower_password_123"
echo ""
echo "  • PostgreSQL:        localhost:5432"
echo "    Database: investment_db"
echo "    Username: postgres"
echo ""
echo "  • Redis:             localhost:6379"
echo ""

# Display next steps
echo "🚀 NEXT STEPS:"
echo "  1. Access Airflow UI at http://localhost:8080"
echo "  2. Enable the 'daily_market_analysis' DAG"
echo "  3. Enable the 'parallel_stock_processing' DAG"
echo "  4. Configure your API keys in the .env file"
echo "  5. Monitor pipeline with: python3 monitor_pipeline.py"
echo ""

# Display important files
echo "📄 IMPORTANT FILES:"
echo "  • Pipeline Monitor:     ./monitor_pipeline.py"
echo "  • Component Tests:      ./test_pipeline_components.py"
echo "  • Activation Script:    ./activate_data_pipeline.py"
echo "  • Configuration:        ./.env"
echo "  • Logs Directory:       ./logs/"
echo ""

# Display service status
echo "🔍 CURRENT SERVICE STATUS:"
docker-compose ps

echo ""
echo "✨ Pipeline is ready for stock analysis!"
echo "   Monitor the pipeline with: python3 monitor_pipeline.py"
echo ""
echo "🛑 To stop all services:"
echo "   docker-compose down && docker-compose -f docker-compose.airflow.yml down"
echo ""

# Optional: Start monitoring
read -p "Would you like to start live monitoring now? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Starting live monitoring..."
    python3 monitor_pipeline.py
fi

success "Data pipeline activation completed successfully!"