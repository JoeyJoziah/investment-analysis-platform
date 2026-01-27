#!/bin/bash

# Complete Apache Airflow Setup for Investment Analysis Application
# This script sets up the entire Airflow infrastructure from scratch

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log "ERROR" "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log "ERROR" "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log "ERROR" "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Set Docker Compose command
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        DOCKER_COMPOSE="docker compose"
    fi
    
    log "INFO" "Prerequisites check passed!"
}

# Function to validate environment variables
validate_environment() {
    log "INFO" "Validating environment variables..."
    
    # Load .env file
    if [ -f .env ]; then
        # More robust loading of .env file that handles comments and spaces
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue
            
            # Remove inline comments and trim whitespace
            value="${value%%#*}"
            value="${value%% }"
            value="${value## }"
            
            # Export the variable
            export "$key=$value"
        done < .env
        log "INFO" "Loaded .env file"
    else
        log "ERROR" ".env file not found. Please create it first."
        exit 1
    fi
    
    # Required variables
    required_vars=(
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "ALPHA_VANTAGE_API_KEY"
        "FINNHUB_API_KEY"
        "POLYGON_API_KEY"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log "ERROR" "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    
    # Generate Airflow-specific secrets if not present
    if [ -z "${AIRFLOW_FERNET_KEY}" ]; then
        log "WARN" "AIRFLOW_FERNET_KEY not set. Generating one..."
        FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
        echo "AIRFLOW_FERNET_KEY=$FERNET_KEY" >> .env
        export AIRFLOW_FERNET_KEY=$FERNET_KEY
    fi
    
    if [ -z "${AIRFLOW_SECRET_KEY}" ]; then
        log "WARN" "AIRFLOW_SECRET_KEY not set. Generating one..."
        SECRET_KEY=$(openssl rand -hex 32)
        echo "AIRFLOW_SECRET_KEY=$SECRET_KEY" >> .env
        export AIRFLOW_SECRET_KEY=$SECRET_KEY
    fi
    
    if [ -z "${AIRFLOW_DB_PASSWORD}" ]; then
        log "WARN" "AIRFLOW_DB_PASSWORD not set. Using default..."
        echo "AIRFLOW_DB_PASSWORD=airflow_password_123" >> .env
        export AIRFLOW_DB_PASSWORD="airflow_password_123"
    fi
    
    if [ -z "${FLOWER_PASSWORD}" ]; then
        log "WARN" "FLOWER_PASSWORD not set. Using default..."
        echo "FLOWER_PASSWORD=flower123" >> .env
        export FLOWER_PASSWORD="flower123"
    fi
    
    log "INFO" "Environment validation completed!"
}

# Function to create necessary directories
create_directories() {
    log "INFO" "Creating necessary directories..."
    
    # Airflow directories
    mkdir -p data_pipelines/airflow/{dags,logs,plugins,config}
    mkdir -p infrastructure/docker/airflow
    mkdir -p infrastructure/monitoring/prometheus/alerts
    mkdir -p infrastructure/monitoring/grafana/dashboards
    
    # Set proper permissions for Airflow
    export AIRFLOW_UID=$(id -u)
    export AIRFLOW_GID=0
    
    # Create .env entry for AIRFLOW_UID if not exists
    if ! grep -q "AIRFLOW_UID" .env; then
        echo "AIRFLOW_UID=$AIRFLOW_UID" >> .env
    fi
    
    log "INFO" "Directories created successfully!"
}

# Function to build custom Airflow image
build_airflow_image() {
    log "INFO" "Building custom Airflow image..."
    
    # Copy requirements file to Docker build context
    cp requirements-airflow.txt infrastructure/docker/airflow/
    
    # Build the custom image
    $DOCKER_COMPOSE -f docker-compose.airflow.yml build airflow-webserver
    
    if [ $? -eq 0 ]; then
        log "INFO" "Custom Airflow image built successfully!"
    else
        log "ERROR" "Failed to build custom Airflow image"
        exit 1
    fi
}

# Function to start core services
start_core_services() {
    log "INFO" "Starting core services (PostgreSQL, Redis)..."
    
    # Start PostgreSQL and Redis
    $DOCKER_COMPOSE -f docker-compose.airflow.yml up -d postgres redis
    
    # Wait for services to be ready
    log "INFO" "Waiting for PostgreSQL to be ready..."
    timeout=60
    while ! $DOCKER_COMPOSE -f docker-compose.airflow.yml exec postgres pg_isready -U postgres &>/dev/null; do
        if [ $timeout -le 0 ]; then
            log "ERROR" "PostgreSQL failed to start within expected time"
            exit 1
        fi
        sleep 2
        timeout=$((timeout-2))
    done
    
    log "INFO" "Waiting for Redis to be ready..."
    timeout=60
    while ! $DOCKER_COMPOSE -f docker-compose.airflow.yml exec redis redis-cli ping &>/dev/null; do
        if [ $timeout -le 0 ]; then
            log "ERROR" "Redis failed to start within expected time"
            exit 1
        fi
        sleep 2
        timeout=$((timeout-2))
    done
    
    log "INFO" "Core services are ready!"
}

# Function to initialize Airflow database
initialize_airflow_database() {
    log "INFO" "Initializing Airflow database..."
    
    # Run Airflow database initialization
    $DOCKER_COMPOSE -f docker-compose.airflow.yml run --rm airflow-init
    
    if [ $? -eq 0 ]; then
        log "INFO" "Airflow database initialized successfully!"
    else
        log "ERROR" "Failed to initialize Airflow database"
        exit 1
    fi
}

# Function to start all Airflow services
start_airflow_services() {
    log "INFO" "Starting all Airflow services..."
    
    # Start all Airflow services
    $DOCKER_COMPOSE -f docker-compose.airflow.yml up -d
    
    # Wait for webserver to be ready
    log "INFO" "Waiting for Airflow webserver to be ready..."
    timeout=120
    while ! curl -f http://localhost:8080/health &>/dev/null; do
        if [ $timeout -le 0 ]; then
            log "ERROR" "Airflow webserver failed to start within expected time"
            exit 1
        fi
        sleep 5
        timeout=$((timeout-5))
        log "DEBUG" "Still waiting for webserver... (${timeout}s remaining)"
    done
    
    log "INFO" "Airflow webserver is ready!"
}

# Function to run the Airflow initialization script
run_airflow_initialization() {
    log "INFO" "Running Airflow initialization (connections, pools, variables)..."
    
    # Give webserver a moment to fully initialize
    sleep 10
    
    # Run the initialization script we created
    if [ -f "scripts/init_airflow.sh" ]; then
        # Extract just the setup parts from init_airflow.sh since services are already running
        log "INFO" "Creating Airflow connections..."
        $DOCKER_COMPOSE -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
            'investment_postgres' \
            --conn-type 'postgres' \
            --conn-host 'postgres' \
            --conn-port '5432' \
            --conn-login 'postgres' \
            --conn-password "${DB_PASSWORD}" \
            --conn-schema 'investment_db' \
            --conn-extra '{"sslmode": "prefer"}' || log "WARN" "Connection investment_postgres may already exist"
        
        log "INFO" "Creating Airflow pools..."
        $DOCKER_COMPOSE -f docker-compose.airflow.yml exec -T airflow-webserver airflow pools set \
            'api_calls' 5 'Pool for external API calls with rate limiting' || log "WARN" "Pool api_calls may already exist"
        
        $DOCKER_COMPOSE -f docker-compose.airflow.yml exec -T airflow-webserver airflow pools set \
            'compute_intensive' 8 'Pool for compute-intensive ML and analytics tasks' || log "WARN" "Pool compute_intensive may already exist"
        
        log "INFO" "Setting Airflow variables..."
        $DOCKER_COMPOSE -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
            'DAILY_API_LIMIT_ALPHA_VANTAGE' "${DAILY_API_LIMIT_ALPHA_VANTAGE:-25}"
        
        $DOCKER_COMPOSE -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
            'MONTHLY_BUDGET_LIMIT' "${MONTHLY_BUDGET_LIMIT:-50}"
        
        log "INFO" "Creating admin user..."
        $DOCKER_COMPOSE -f docker-compose.airflow.yml exec -T airflow-webserver airflow users create \
            --username "${AIRFLOW_ADMIN_USERNAME:-admin}" \
            --firstname "Investment" \
            --lastname "Admin" \
            --role "Admin" \
            --email "${AIRFLOW_ADMIN_EMAIL:-admin@investment-analysis.com}" \
            --password "${AIRFLOW_ADMIN_PASSWORD:-admin123}" || log "WARN" "Admin user may already exist"
        
        log "INFO" "Airflow initialization completed!"
    else
        log "WARN" "init_airflow.sh not found, skipping detailed initialization"
    fi
}

# Function to verify installation
verify_installation() {
    log "INFO" "Verifying Airflow installation..."
    
    # Check service health
    services=("airflow-webserver" "airflow-scheduler" "airflow-worker-api" "airflow-worker-compute" "airflow-flower")
    
    for service in "${services[@]}"; do
        if $DOCKER_COMPOSE -f docker-compose.airflow.yml ps | grep -q "$service.*Up"; then
            log "INFO" "✓ $service is running"
        else
            log "WARN" "✗ $service is not running properly"
        fi
    done
    
    # Test webserver endpoint
    if curl -f http://localhost:8080/health &>/dev/null; then
        log "INFO" "✓ Airflow webserver health check passed"
    else
        log "WARN" "✗ Airflow webserver health check failed"
    fi
    
    # Test Flower endpoint
    if curl -f http://localhost:5555/ &>/dev/null; then
        log "INFO" "✓ Flower monitoring is accessible"
    else
        log "WARN" "✗ Flower monitoring is not accessible"
    fi
    
    log "INFO" "Installation verification completed!"
}

# Function to show deployment summary
show_deployment_summary() {
    log "INFO" "🎉 Airflow deployment completed successfully!"
    echo ""
    echo -e "${GREEN}📊 Airflow Services:${NC}"
    echo "  • Webserver: http://localhost:8080"
    echo "  • Flower (Celery monitoring): http://localhost:5555"
    echo "  • StatsD metrics: http://localhost:9102/metrics"
    echo ""
    echo -e "${GREEN}👤 Admin Credentials:${NC}"
    echo "  • Username: ${AIRFLOW_ADMIN_USERNAME:-admin}"
    echo "  • Password: ${AIRFLOW_ADMIN_PASSWORD:-admin123}"
    echo ""
    echo -e "${GREEN}🔧 Management Commands:${NC}"
    echo "  • View logs: $DOCKER_COMPOSE -f docker-compose.airflow.yml logs -f [service]"
    echo "  • Stop services: $DOCKER_COMPOSE -f docker-compose.airflow.yml down"
    echo "  • Restart services: $DOCKER_COMPOSE -f docker-compose.airflow.yml restart"
    echo ""
    echo -e "${GREEN}📋 Next Steps:${NC}"
    echo "  1. Access Airflow dashboard at http://localhost:8080"
    echo "  2. Verify DAGs are loaded and visible"
    echo "  3. Check connections in Admin > Connections"
    echo "  4. Review pools in Admin > Pools"
    echo "  5. Trigger a test run of daily_market_analysis DAG"
    echo "  6. Monitor performance in Flower at http://localhost:5555"
    echo ""
    echo -e "${GREEN}🏗️ Architecture:${NC}"
    echo "  • Executor: CeleryExecutor with 3 specialized workers"
    echo "  • Database: PostgreSQL (shared with main application)"
    echo "  • Message Broker: Redis"
    echo "  • Monitoring: Prometheus + Grafana integration"
    echo "  • Pools: api_calls (5), compute_intensive (8), database_tasks (12)"
    echo ""
}

# Function to handle cleanup on failure
cleanup_on_failure() {
    log "ERROR" "Setup failed. Cleaning up..."
    $DOCKER_COMPOSE -f docker-compose.airflow.yml down -v 2>/dev/null || true
    exit 1
}

# Set trap for cleanup on failure
trap cleanup_on_failure ERR

# Main execution function
main() {
    log "INFO" "🚀 Starting complete Airflow setup for Investment Analysis Application..."
    
    # Run all setup steps
    check_prerequisites
    validate_environment
    create_directories
    build_airflow_image
    start_core_services
    initialize_airflow_database
    start_airflow_services
    run_airflow_initialization
    verify_installation
    show_deployment_summary
    
    log "INFO" "✨ Complete Airflow setup finished successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-init)
            SKIP_INIT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-build    Skip building custom Airflow image"
            echo "  --skip-init     Skip Airflow database initialization"
            echo "  --help, -h      Show this help message"
            echo ""
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [ ! -f "docker-compose.airflow.yml" ]; then
    log "ERROR" "docker-compose.airflow.yml not found. Please run this script from the project root."
    exit 1
fi

# Run main function
main