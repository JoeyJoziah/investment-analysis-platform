#!/bin/bash

# =============================================================================
# Investment Analysis Platform - Production Deployment Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV="${1:-production}"
COMPOSE_FILE="docker-compose.${DEPLOYMENT_ENV}.yml"
ENV_FILE=".env.${DEPLOYMENT_ENV}"
BACKUP_BEFORE_DEPLOY=true
RUN_TESTS=true
RUN_MIGRATIONS=true

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose is not installed"
    fi
    
    # Check environment file
    if [ ! -f "${ENV_FILE}" ]; then
        log_error "Environment file ${ENV_FILE} not found. Copy from .env.${DEPLOYMENT_ENV}.example"
    fi
    
    # Check compose file
    if [ ! -f "${COMPOSE_FILE}" ]; then
        log_error "Docker Compose file ${COMPOSE_FILE} not found"
    fi
    
    log_success "Prerequisites check passed"
}

validate_environment() {
    log_info "Validating environment variables..."
    
    # Source environment file
    set -a
    source ${ENV_FILE}
    set +a
    
    # Check required variables
    required_vars=(
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "ELASTIC_PASSWORD"
        "RABBITMQ_PASSWORD"
        "JWT_SECRET_KEY"
        "SECRET_KEY"
        "AIRFLOW_FERNET_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable ${var} is not set"
        fi
    done
    
    log_success "Environment validation passed"
}

backup_database() {
    if [ "$BACKUP_BEFORE_DEPLOY" = true ]; then
        log_info "Creating database backup..."
        
        # Check if database container is running
        if docker ps | grep -q investment_db; then
            docker exec investment_db pg_dump -U postgres investment_db | gzip > backups/pre_deploy_$(date +%Y%m%d_%H%M%S).sql.gz
            log_success "Database backup created"
        else
            log_warning "Database container not running, skipping backup"
        fi
    fi
}

run_tests() {
    if [ "$RUN_TESTS" = true ]; then
        log_info "Running tests..."
        
        # Build test image
        docker build -f Dockerfile.backend -t investment-backend:test .
        
        # Run tests
        docker run --rm \
            --env-file ${ENV_FILE} \
            -e APP_ENV=test \
            investment-backend:test \
            pytest backend/tests/ -v --cov=backend --cov-report=term-missing
        
        if [ $? -ne 0 ]; then
            log_error "Tests failed. Deployment aborted."
        fi
        
        log_success "All tests passed"
    fi
}

pull_images() {
    log_info "Pulling latest images..."
    docker compose -f ${COMPOSE_FILE} pull --ignore-pull-failures
    log_success "Images pulled"
}

build_images() {
    log_info "Building application images..."
    
    # Build with BuildKit for better caching
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    docker compose -f ${COMPOSE_FILE} build --parallel
    
    if [ $? -ne 0 ]; then
        log_error "Image build failed"
    fi
    
    log_success "Images built successfully"
}

stop_services() {
    log_info "Stopping existing services..."
    docker compose -f ${COMPOSE_FILE} down --remove-orphans
    log_success "Services stopped"
}

start_infrastructure() {
    log_info "Starting infrastructure services..."
    
    docker compose -f ${COMPOSE_FILE} up -d \
        postgres \
        redis \
        elasticsearch \
        rabbitmq
    
    # Wait for services to be healthy
    log_info "Waiting for infrastructure services to be healthy..."
    sleep 10
    
    for service in postgres redis elasticsearch rabbitmq; do
        for i in {1..30}; do
            if docker compose -f ${COMPOSE_FILE} ps ${service} | grep -q "healthy"; then
                log_success "${service} is healthy"
                break
            fi
            
            if [ $i -eq 30 ]; then
                log_error "${service} failed to become healthy"
            fi
            
            sleep 2
        done
    done
}

run_migrations() {
    if [ "$RUN_MIGRATIONS" = true ]; then
        log_info "Running database migrations..."
        
        docker compose -f ${COMPOSE_FILE} run --rm backend \
            alembic upgrade head
        
        if [ $? -ne 0 ]; then
            log_error "Migrations failed"
        fi
        
        log_success "Migrations completed"
    fi
}

initialize_data() {
    log_info "Initializing data..."
    
    # Run initialization script
    docker compose -f ${COMPOSE_FILE} run --rm backend \
        python scripts/init_database.py
    
    # Load ML models
    docker compose -f ${COMPOSE_FILE} run --rm backend \
        python scripts/download_models.py
    
    log_success "Data initialization completed"
}

start_application() {
    log_info "Starting application services..."
    
    docker compose -f ${COMPOSE_FILE} up -d \
        backend \
        celery_worker \
        celery_beat \
        airflow \
        frontend
    
    # Wait for services to start
    sleep 10
    
    # Check health
    for service in backend frontend; do
        if docker compose -f ${COMPOSE_FILE} ps ${service} | grep -q "Up"; then
            log_success "${service} is running"
        else
            log_error "${service} failed to start"
        fi
    done
}

start_monitoring() {
    log_info "Starting monitoring services..."
    
    docker compose -f ${COMPOSE_FILE} up -d \
        prometheus \
        grafana
    
    log_success "Monitoring services started"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check API health
    for i in {1..10}; do
        if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
            log_success "API is healthy"
            break
        fi
        
        if [ $i -eq 10 ]; then
            log_error "API health check failed"
        fi
        
        sleep 3
    done
    
    # Check frontend
    if curl -f http://localhost/health > /dev/null 2>&1; then
        log_success "Frontend is healthy"
    else
        log_warning "Frontend health check failed"
    fi
}

show_status() {
    echo ""
    echo "=============================================="
    echo "         DEPLOYMENT STATUS                    "
    echo "=============================================="
    
    docker compose -f ${COMPOSE_FILE} ps
    
    echo ""
    echo "=============================================="
    echo "         ACCESS POINTS                        "
    echo "=============================================="
    echo "Frontend:       http://localhost"
    echo "API:            http://localhost:8000"
    echo "API Docs:       http://localhost:8000/docs"
    echo "Grafana:        http://localhost:3001"
    echo "Prometheus:     http://localhost:9090"
    echo "RabbitMQ:       http://localhost:15672"
    echo "Airflow:        http://localhost:8080"
    echo ""
    echo "=============================================="
    echo "         USEFUL COMMANDS                      "
    echo "=============================================="
    echo "View logs:      docker compose -f ${COMPOSE_FILE} logs -f [service]"
    echo "Stop all:       docker compose -f ${COMPOSE_FILE} down"
    echo "Restart:        docker compose -f ${COMPOSE_FILE} restart [service]"
    echo "Scale:          docker compose -f ${COMPOSE_FILE} up -d --scale backend=3"
    echo ""
}

cleanup_old_images() {
    log_info "Cleaning up old images..."
    docker image prune -f
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    echo "=============================================="
    echo "   INVESTMENT ANALYSIS PLATFORM DEPLOYMENT    "
    echo "=============================================="
    echo "Environment: ${DEPLOYMENT_ENV}"
    echo "Compose File: ${COMPOSE_FILE}"
    echo ""
    
    check_prerequisites
    validate_environment
    backup_database
    run_tests
    pull_images
    build_images
    stop_services
    start_infrastructure
    run_migrations
    initialize_data
    start_application
    start_monitoring
    verify_deployment
    cleanup_old_images
    show_status
    
    log_success "DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Next steps:"
    echo "1. Configure SSL certificates for HTTPS"
    echo "2. Set up domain DNS records"
    echo "3. Configure firewall rules"
    echo "4. Set up backup automation"
    echo "5. Configure monitoring alerts"
}

# Run main function
main "$@"