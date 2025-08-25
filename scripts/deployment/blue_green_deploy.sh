#!/bin/bash

# Blue-Green Deployment Script for Investment Analysis Platform
# Zero-downtime deployment with automatic rollback capability
# Optimized for cost-efficient production deployment (<$50/month)

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly DEPLOYMENT_LOG="${PROJECT_ROOT}/logs/deployment.log"
readonly HEALTH_CHECK_TIMEOUT=300  # 5 minutes
readonly HEALTH_CHECK_INTERVAL=10  # 10 seconds
readonly ROLLBACK_TIMEOUT=180      # 3 minutes

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
VERSION="latest"
SKIP_TESTS=false
SKIP_BACKUP=false
AUTO_ROLLBACK=true
DRY_RUN=false
SERVICES_TO_DEPLOY="backend frontend"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$DEPLOYMENT_LOG" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$DEPLOYMENT_LOG" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" | tee -a "$DEPLOYMENT_LOG" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$DEPLOYMENT_LOG" ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$DEPLOYMENT_LOG"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Blue-Green Deployment Script for Investment Analysis Platform

OPTIONS:
    -e, --environment ENV     Target environment (default: production)
    -v, --version VERSION     Version/tag to deploy (default: latest)
    -s, --services SERVICES   Services to deploy (default: "backend frontend")
    --skip-tests              Skip pre-deployment tests
    --skip-backup             Skip database backup
    --no-auto-rollback        Disable automatic rollback on failure
    --dry-run                 Show what would be done without executing
    -h, --help               Show this help message

EXAMPLES:
    $0                                    # Deploy latest version with all checks
    $0 -v v1.2.3 -s backend              # Deploy specific version of backend only
    $0 --dry-run                         # Show deployment plan without executing
    $0 --skip-tests --skip-backup        # Fast deployment (not recommended for prod)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -s|--services)
                SERVICES_TO_DEPLOY="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --no-auto-rollback)
                AUTO_ROLLBACK=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking deployment prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in docker docker-compose curl jq; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "ERROR" "Missing required tools: ${missing_tools[*]}"
        return 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log "ERROR" "Docker daemon is not running"
        return 1
    fi
    
    # Check environment file
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        log "ERROR" "Environment file not found: ${PROJECT_ROOT}/.env"
        return 1
    fi
    
    # Create logs directory
    mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
    
    log "INFO" "Prerequisites check passed"
    return 0
}

# Get current active environment (blue or green)
get_current_environment() {
    local current_env="blue"  # Default to blue
    
    # Check which environment is currently active by querying load balancer
    if curl -sf http://localhost/api/health | jq -r '.environment' | grep -q "green"; then
        current_env="green"
    fi
    
    echo "$current_env"
}

# Get target environment (opposite of current)
get_target_environment() {
    local current_env="$1"
    
    if [[ "$current_env" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Build application images
build_images() {
    local target_env="$1"
    
    log "INFO" "Building application images for $target_env environment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would build images with version: $VERSION"
        return 0
    fi
    
    # Build backend image
    if [[ "$SERVICES_TO_DEPLOY" =~ "backend" ]]; then
        log "INFO" "Building backend image..."
        docker build \
            -f "${PROJECT_ROOT}/infrastructure/docker/backend/Dockerfile.optimized" \
            -t "investment-backend:${VERSION}" \
            -t "investment-backend:${target_env}" \
            --target runtime \
            --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
            --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
            "$PROJECT_ROOT"
    fi
    
    # Build frontend image
    if [[ "$SERVICES_TO_DEPLOY" =~ "frontend" ]]; then
        log "INFO" "Building frontend image..."
        docker build \
            -f "${PROJECT_ROOT}/infrastructure/docker/frontend/Dockerfile.optimized" \
            -t "investment-frontend:${VERSION}" \
            -t "investment-frontend:${target_env}" \
            --build-arg REACT_APP_VERSION="$VERSION" \
            --build-arg REACT_APP_ENVIRONMENT="$ENVIRONMENT" \
            "${PROJECT_ROOT}/frontend/web"
    fi
    
    log "INFO" "Image build completed successfully"
}

# Run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "WARN" "Skipping pre-deployment tests"
        return 0
    fi
    
    log "INFO" "Running pre-deployment tests..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would run test suite"
        return 0
    fi
    
    # Run unit tests
    log "INFO" "Running unit tests..."
    if ! docker run --rm \
        -v "${PROJECT_ROOT}/backend:/app/backend" \
        -v "${PROJECT_ROOT}/tests:/app/tests" \
        -e ENVIRONMENT=test \
        "investment-backend:${VERSION}" \
        python -m pytest tests/unit/ -v --tb=short; then
        log "ERROR" "Unit tests failed"
        return 1
    fi
    
    # Run integration tests on test database
    log "INFO" "Running integration tests..."
    docker-compose -f "${PROJECT_ROOT}/docker-compose.test.yml" up -d
    sleep 30  # Wait for services to be ready
    
    if ! docker run --rm \
        --network "$(basename "$PROJECT_ROOT")_test_network" \
        -e DATABASE_URL="postgresql://test:test@postgres_test:5432/test_db" \
        -e REDIS_URL="redis://:testpass@redis_test:6379/0" \
        "investment-backend:${VERSION}" \
        python -m pytest tests/integration/ -v --tb=short; then
        log "ERROR" "Integration tests failed"
        docker-compose -f "${PROJECT_ROOT}/docker-compose.test.yml" down
        return 1
    fi
    
    docker-compose -f "${PROJECT_ROOT}/docker-compose.test.yml" down
    
    log "INFO" "All tests passed successfully"
}

# Create database backup
create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log "WARN" "Skipping database backup"
        return 0
    fi
    
    log "INFO" "Creating database backup..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would create database backup"
        return 0
    fi
    
    local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup PostgreSQL database
    log "INFO" "Backing up PostgreSQL database..."
    docker exec investment_db_prod pg_dump \
        -U postgres \
        -d investment_db \
        --format=custom \
        --compress=9 \
        --verbose \
        > "${backup_dir}/postgres_backup.sql"
    
    # Backup Redis data
    log "INFO" "Backing up Redis data..."
    docker exec investment_cache_prod redis-cli \
        --rdb "${backup_dir}/redis_backup.rdb"
    docker cp investment_cache_prod:"${backup_dir}/redis_backup.rdb" \
        "${backup_dir}/redis_backup.rdb"
    
    # Create backup metadata
    cat > "${backup_dir}/backup_metadata.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version_before": "$(docker inspect investment_api_prod --format '{{.Config.Image}}' 2>/dev/null || echo 'unknown')",
    "deployment_version": "$VERSION",
    "environment": "$ENVIRONMENT",
    "services": "$SERVICES_TO_DEPLOY"
}
EOF
    
    log "INFO" "Database backup created: $backup_dir"
    echo "$backup_dir"
}

# Deploy to target environment
deploy_to_environment() {
    local target_env="$1"
    
    log "INFO" "Deploying to $target_env environment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would deploy to $target_env environment"
        return 0
    fi
    
    # Create environment-specific compose override
    local compose_override="${PROJECT_ROOT}/docker-compose.${target_env}.yml"
    cat > "$compose_override" << EOF
version: '3.8'

services:
  backend:
    image: investment-backend:${VERSION}
    container_name: investment_api_${target_env}
    ports:
      - "${target_env == 'blue' && echo '8001' || echo '8002'}:8000"
    environment:
      - ENVIRONMENT=${target_env}
    networks:
      - investment_network_${target_env}

  frontend:
    image: investment-frontend:${VERSION}
    container_name: investment_web_${target_env}
    ports:
      - "${target_env == 'blue' && echo '3001' || echo '3002'}:80"
    environment:
      - ENVIRONMENT=${target_env}
    networks:
      - investment_network_${target_env}

networks:
  investment_network_${target_env}:
    driver: bridge
EOF
    
    # Deploy services
    log "INFO" "Starting $target_env environment services..."
    docker-compose \
        -f "${PROJECT_ROOT}/docker-compose.yml" \
        -f "${PROJECT_ROOT}/docker-compose.production.yml" \
        -f "$compose_override" \
        up -d --force-recreate
    
    log "INFO" "Deployment to $target_env environment completed"
}

# Health check function
health_check() {
    local target_env="$1"
    local port
    
    if [[ "$target_env" == "blue" ]]; then
        port="8001"
    else
        port="8002"
    fi
    
    log "INFO" "Performing health check on $target_env environment (port $port)..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would perform health check"
        return 0
    fi
    
    local start_time=$(date +%s)
    local end_time=$((start_time + HEALTH_CHECK_TIMEOUT))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        # Check backend health
        if curl -sf "http://localhost:${port}/api/health" > /dev/null 2>&1; then
            local health_response=$(curl -s "http://localhost:${port}/api/health")
            local health_status=$(echo "$health_response" | jq -r '.status // "unknown"')
            
            if [[ "$health_status" == "healthy" ]]; then
                log "INFO" "Health check passed for $target_env environment"
                return 0
            else
                log "WARN" "Health check returned status: $health_status"
            fi
        fi
        
        log "DEBUG" "Health check failed, retrying in ${HEALTH_CHECK_INTERVAL}s..."
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    log "ERROR" "Health check timed out after ${HEALTH_CHECK_TIMEOUT}s"
    return 1
}

# Switch traffic to target environment
switch_traffic() {
    local target_env="$1"
    local target_port
    
    if [[ "$target_env" == "blue" ]]; then
        target_port="8001"
    else
        target_port="8002"
    fi
    
    log "INFO" "Switching traffic to $target_env environment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would switch traffic to $target_env environment"
        return 0
    fi
    
    # Update nginx configuration to point to new environment
    local nginx_conf="${PROJECT_ROOT}/infrastructure/docker/nginx/upstream.conf"
    cat > "$nginx_conf" << EOF
upstream backend {
    server backend:${target_port};
}

upstream frontend {
    server frontend:$(if [[ "$target_env" == "blue" ]]; then echo "3001"; else echo "3002"; fi);
}
EOF
    
    # Reload nginx configuration
    docker exec investment_nginx_prod nginx -s reload
    
    # Wait for traffic switch to take effect
    sleep 10
    
    # Verify traffic switch
    local active_env=$(get_current_environment)
    if [[ "$active_env" == "$target_env" ]]; then
        log "INFO" "Traffic successfully switched to $target_env environment"
        return 0
    else
        log "ERROR" "Traffic switch failed - still on $active_env environment"
        return 1
    fi
}

# Cleanup old environment
cleanup_old_environment() {
    local old_env="$1"
    
    log "INFO" "Cleaning up $old_env environment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would cleanup $old_env environment"
        return 0
    fi
    
    # Stop and remove old environment containers
    docker stop "investment_api_${old_env}" "investment_web_${old_env}" 2>/dev/null || true
    docker rm "investment_api_${old_env}" "investment_web_${old_env}" 2>/dev/null || true
    
    # Remove old networks
    docker network rm "investment_network_${old_env}" 2>/dev/null || true
    
    # Clean up old compose override
    rm -f "${PROJECT_ROOT}/docker-compose.${old_env}.yml"
    
    log "INFO" "Cleanup of $old_env environment completed"
}

# Rollback function
rollback() {
    local current_env="$1"
    local target_env="$2"
    
    log "WARN" "Initiating rollback from $target_env to $current_env..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would rollback to $current_env environment"
        return 0
    fi
    
    # Switch traffic back to old environment
    if switch_traffic "$current_env"; then
        log "INFO" "Rollback completed successfully"
        
        # Cleanup failed deployment
        cleanup_old_environment "$target_env"
        
        return 0
    else
        log "ERROR" "Rollback failed - manual intervention required"
        return 1
    fi
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    log "INFO" "Starting blue-green deployment..."
    log "INFO" "Environment: $ENVIRONMENT, Version: $VERSION, Services: $SERVICES_TO_DEPLOY"
    
    # Check prerequisites
    if ! check_prerequisites; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    fi
    
    # Determine current and target environments
    local current_env=$(get_current_environment)
    local target_env=$(get_target_environment "$current_env")
    
    log "INFO" "Current environment: $current_env, Target environment: $target_env"
    
    # Create backup
    local backup_dir=""
    if ! backup_dir=$(create_backup); then
        log "ERROR" "Database backup failed"
        exit 1
    fi
    
    # Build new images
    if ! build_images "$target_env"; then
        log "ERROR" "Image build failed"
        exit 1
    fi
    
    # Run tests
    if ! run_tests; then
        log "ERROR" "Pre-deployment tests failed"
        exit 1
    fi
    
    # Deploy to target environment
    if ! deploy_to_environment "$target_env"; then
        log "ERROR" "Deployment to $target_env failed"
        exit 1
    fi
    
    # Health check
    if ! health_check "$target_env"; then
        log "ERROR" "Health check failed for $target_env environment"
        
        if [[ "$AUTO_ROLLBACK" == "true" ]]; then
            log "WARN" "Auto-rollback is enabled, rolling back..."
            if rollback "$current_env" "$target_env"; then
                log "INFO" "Automatic rollback completed"
                exit 1
            else
                log "ERROR" "Automatic rollback failed - manual intervention required"
                exit 2
            fi
        else
            log "ERROR" "Auto-rollback is disabled - manual intervention required"
            exit 1
        fi
    fi
    
    # Switch traffic
    if ! switch_traffic "$target_env"; then
        log "ERROR" "Traffic switch failed"
        
        if [[ "$AUTO_ROLLBACK" == "true" ]]; then
            log "WARN" "Rolling back due to traffic switch failure..."
            rollback "$current_env" "$target_env"
        fi
        
        exit 1
    fi
    
    # Final verification
    sleep 30  # Allow time for traffic to stabilize
    
    if health_check "$target_env"; then
        log "INFO" "Final verification passed"
        
        # Cleanup old environment
        cleanup_old_environment "$current_env"
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "INFO" "Blue-green deployment completed successfully in ${duration}s"
        log "INFO" "Active environment: $target_env"
        log "INFO" "Backup location: $backup_dir"
        
        # Send success notification (if configured)
        # notify_success "$target_env" "$VERSION" "$duration"
        
    else
        log "ERROR" "Final verification failed"
        
        if [[ "$AUTO_ROLLBACK" == "true" ]]; then
            rollback "$current_env" "$target_env"
        fi
        
        exit 1
    fi
}

# Trap for cleanup on exit
cleanup_on_exit() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR" "Deployment failed with exit code: $exit_code"
        
        # Additional cleanup if needed
        docker-compose -f "${PROJECT_ROOT}/docker-compose.test.yml" down 2>/dev/null || true
    fi
    
    exit $exit_code
}

trap cleanup_on_exit EXIT

# Parse arguments and run main function
parse_args "$@"
main