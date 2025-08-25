#!/bin/bash

# Rollback Script for Investment Analysis Platform
# Quick rollback to previous deployment with minimal downtime
# Includes database restoration and service verification

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly ROLLBACK_LOG="${PROJECT_ROOT}/logs/rollback.log"
readonly VERIFICATION_TIMEOUT=120  # 2 minutes

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default values
BACKUP_PATH=""
TARGET_VERSION=""
RESTORE_DATABASE=false
FORCE=false
DRY_RUN=false

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$ROLLBACK_LOG" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$ROLLBACK_LOG" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" | tee -a "$ROLLBACK_LOG" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$ROLLBACK_LOG" ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$ROLLBACK_LOG"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Rollback Script for Investment Analysis Platform

OPTIONS:
    -b, --backup-path PATH    Path to backup directory for database restoration
    -v, --version VERSION     Specific version to rollback to
    --restore-db              Also restore database from backup
    --force                   Force rollback without confirmation
    --dry-run                 Show what would be done without executing
    -h, --help               Show this help message

EXAMPLES:
    $0                                    # Quick rollback to previous environment
    $0 -v v1.2.0                        # Rollback to specific version
    $0 -b /path/to/backup --restore-db   # Rollback with database restoration
    $0 --dry-run                         # Show rollback plan

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--backup-path)
                BACKUP_PATH="$2"
                shift 2
                ;;
            -v|--version)
                TARGET_VERSION="$2"
                shift 2
                ;;
            --restore-db)
                RESTORE_DATABASE=true
                shift
                ;;
            --force)
                FORCE=true
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
    log "INFO" "Checking rollback prerequisites..."
    
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
    
    # Create logs directory
    mkdir -p "$(dirname "$ROLLBACK_LOG")"
    
    log "INFO" "Prerequisites check passed"
    return 0
}

# Get current deployment information
get_current_deployment() {
    log "INFO" "Getting current deployment information..."
    
    local current_info="{}"
    
    # Get current backend version
    if docker ps --filter "name=investment_api" --format "table {{.Names}}\t{{.Image}}" | grep -q "investment_api"; then
        local backend_image=$(docker inspect investment_api_prod --format '{{.Config.Image}}' 2>/dev/null || echo "unknown")
        current_info=$(echo "$current_info" | jq --arg img "$backend_image" '. + {backend_image: $img}')
    fi
    
    # Get current environment (blue/green)
    local current_env="unknown"
    if curl -sf http://localhost/api/health | jq -r '.environment' &>/dev/null; then
        current_env=$(curl -s http://localhost/api/health | jq -r '.environment // "unknown"')
    fi
    current_info=$(echo "$current_info" | jq --arg env "$current_env" '. + {current_environment: $env}')
    
    # Get running services
    local running_services=$(docker ps --filter "name=investment_" --format "{{.Names}}" | tr '\n' ',' | sed 's/,$//')
    current_info=$(echo "$current_info" | jq --arg services "$running_services" '. + {running_services: $services}')
    
    echo "$current_info"
}

# List available backups
list_available_backups() {
    log "INFO" "Available backups:"
    
    if [[ ! -d "${PROJECT_ROOT}/backups" ]]; then
        log "WARN" "No backup directory found"
        return 0
    fi
    
    for backup_dir in "${PROJECT_ROOT}/backups"/*; do
        if [[ -d "$backup_dir" && -f "$backup_dir/backup_metadata.json" ]]; then
            local metadata=$(cat "$backup_dir/backup_metadata.json")
            local timestamp=$(echo "$metadata" | jq -r '.timestamp')
            local version=$(echo "$metadata" | jq -r '.deployment_version')
            
            echo "  - $(basename "$backup_dir") (${timestamp}, version: ${version})"
        fi
    done
}

# Find previous deployment
find_previous_deployment() {
    log "INFO" "Finding previous deployment..."
    
    # If specific version provided, use it
    if [[ -n "$TARGET_VERSION" ]]; then
        log "INFO" "Target version specified: $TARGET_VERSION"
        echo "$TARGET_VERSION"
        return 0
    fi
    
    # Try to find previous version from Docker images
    local previous_version=""
    
    # Look for previous backend image
    local images=$(docker images "investment-backend" --format "{{.Tag}}" | grep -v "latest" | head -2)
    if [[ $(echo "$images" | wc -l) -ge 2 ]]; then
        previous_version=$(echo "$images" | tail -1)
    fi
    
    # If no previous version found, look for blue/green environment
    if [[ -z "$previous_version" ]]; then
        local current_env=$(get_current_deployment | jq -r '.current_environment')
        if [[ "$current_env" == "blue" ]]; then
            previous_version="green"
        elif [[ "$current_env" == "green" ]]; then
            previous_version="blue"
        fi
    fi
    
    if [[ -n "$previous_version" ]]; then
        log "INFO" "Found previous deployment: $previous_version"
        echo "$previous_version"
    else
        log "ERROR" "Could not determine previous deployment"
        return 1
    fi
}

# Confirm rollback
confirm_rollback() {
    if [[ "$FORCE" == "true" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    local current_info=$(get_current_deployment)
    local current_image=$(echo "$current_info" | jq -r '.backend_image')
    local current_env=$(echo "$current_info" | jq -r '.current_environment')
    
    echo
    log "WARN" "ROLLBACK CONFIRMATION REQUIRED"
    echo "Current deployment:"
    echo "  - Environment: $current_env"
    echo "  - Backend image: $current_image"
    echo
    
    if [[ -n "$TARGET_VERSION" ]]; then
        echo "Rolling back to version: $TARGET_VERSION"
    else
        echo "Rolling back to previous deployment"
    fi
    
    if [[ "$RESTORE_DATABASE" == "true" ]]; then
        echo "Database will be restored from: $BACKUP_PATH"
    fi
    
    echo
    read -p "Are you sure you want to proceed? (yes/no): " -r confirmation
    
    if [[ "$confirmation" != "yes" ]]; then
        log "INFO" "Rollback cancelled by user"
        exit 0
    fi
}

# Stop current services
stop_current_services() {
    log "INFO" "Stopping current services..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would stop current services"
        return 0
    fi
    
    # Get running services
    local running_services=$(docker ps --filter "name=investment_" --format "{{.Names}}")
    
    if [[ -n "$running_services" ]]; then
        log "INFO" "Stopping services: $running_services"
        
        # Stop services gracefully
        echo "$running_services" | xargs -r docker stop -t 30
        
        # Wait for services to stop
        sleep 10
        
        log "INFO" "Services stopped successfully"
    else
        log "WARN" "No running services found"
    fi
}

# Restore database
restore_database() {
    if [[ "$RESTORE_DATABASE" != "true" ]]; then
        return 0
    fi
    
    if [[ -z "$BACKUP_PATH" ]]; then
        log "ERROR" "Database restore requested but no backup path provided"
        return 1
    fi
    
    if [[ ! -d "$BACKUP_PATH" ]]; then
        log "ERROR" "Backup directory not found: $BACKUP_PATH"
        return 1
    fi
    
    log "INFO" "Restoring database from backup..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would restore database from $BACKUP_PATH"
        return 0
    fi
    
    # Start database if not running
    if ! docker ps --filter "name=investment_db" --format "{{.Names}}" | grep -q "investment_db"; then
        log "INFO" "Starting database container..."
        docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" \
                      -f "${PROJECT_ROOT}/docker-compose.production.yml" \
                      up -d postgres
        
        # Wait for database to be ready
        local timeout=60
        local elapsed=0
        
        while [[ $elapsed -lt $timeout ]]; do
            if docker exec investment_db_prod pg_isready -U postgres > /dev/null 2>&1; then
                break
            fi
            sleep 2
            ((elapsed+=2))
        done
        
        if [[ $elapsed -ge $timeout ]]; then
            log "ERROR" "Database failed to start within ${timeout}s"
            return 1
        fi
    fi
    
    # Restore PostgreSQL backup
    if [[ -f "$BACKUP_PATH/postgres_backup.sql" ]]; then
        log "INFO" "Restoring PostgreSQL database..."
        
        # Drop existing connections
        docker exec investment_db_prod psql -U postgres -d investment_db \
            -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'investment_db' AND pid <> pg_backend_pid();"
        
        # Restore database
        docker exec -i investment_db_prod pg_restore \
            -U postgres \
            -d investment_db \
            --clean \
            --if-exists \
            --verbose < "$BACKUP_PATH/postgres_backup.sql"
        
        log "INFO" "PostgreSQL database restored successfully"
    fi
    
    # Restore Redis backup
    if [[ -f "$BACKUP_PATH/redis_backup.rdb" ]]; then
        log "INFO" "Restoring Redis data..."
        
        # Start Redis if not running
        if ! docker ps --filter "name=investment_cache" --format "{{.Names}}" | grep -q "investment_cache"; then
            docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" \
                          -f "${PROJECT_ROOT}/docker-compose.production.yml" \
                          up -d redis
            sleep 5
        fi
        
        # Stop Redis, restore data, and restart
        docker exec investment_cache_prod redis-cli SHUTDOWN SAVE || true
        docker cp "$BACKUP_PATH/redis_backup.rdb" investment_cache_prod:/data/dump.rdb
        docker restart investment_cache_prod
        
        log "INFO" "Redis data restored successfully"
    fi
    
    log "INFO" "Database restoration completed"
}

# Start previous services
start_previous_services() {
    local target_version="$1"
    
    log "INFO" "Starting services with version: $target_version..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would start services with version $target_version"
        return 0
    fi
    
    # Create temporary compose override for rollback
    local rollback_override="${PROJECT_ROOT}/docker-compose.rollback.yml"
    cat > "$rollback_override" << EOF
version: '3.8'

services:
  backend:
    image: investment-backend:${target_version}
    environment:
      - ROLLBACK_VERSION=${target_version}

  frontend:
    image: investment-frontend:${target_version}
    environment:
      - ROLLBACK_VERSION=${target_version}
EOF
    
    # Start services
    log "INFO" "Starting rolled-back services..."
    docker-compose \
        -f "${PROJECT_ROOT}/docker-compose.yml" \
        -f "${PROJECT_ROOT}/docker-compose.production.yml" \
        -f "$rollback_override" \
        up -d --force-recreate
    
    # Clean up temporary override
    rm -f "$rollback_override"
    
    log "INFO" "Services started successfully"
}

# Verify rollback
verify_rollback() {
    local target_version="$1"
    
    log "INFO" "Verifying rollback..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would verify rollback"
        return 0
    fi
    
    local start_time=$(date +%s)
    local end_time=$((start_time + VERIFICATION_TIMEOUT))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        # Check if services are responding
        if curl -sf http://localhost/api/health > /dev/null 2>&1; then
            local health_response=$(curl -s http://localhost/api/health)
            local health_status=$(echo "$health_response" | jq -r '.status // "unknown"')
            
            if [[ "$health_status" == "healthy" ]]; then
                log "INFO" "Health check passed"
                
                # Verify version if specified
                if [[ -n "$target_version" && "$target_version" != "blue" && "$target_version" != "green" ]]; then
                    local current_version=$(echo "$health_response" | jq -r '.version // "unknown"')
                    if [[ "$current_version" == "$target_version" ]]; then
                        log "INFO" "Version verification passed: $current_version"
                    else
                        log "WARN" "Version mismatch: expected $target_version, got $current_version"
                    fi
                fi
                
                log "INFO" "Rollback verification completed successfully"
                return 0
            else
                log "DEBUG" "Health check returned status: $health_status"
            fi
        fi
        
        log "DEBUG" "Verification failed, retrying in 10s..."
        sleep 10
    done
    
    log "ERROR" "Rollback verification timed out after ${VERIFICATION_TIMEOUT}s"
    return 1
}

# Create rollback report
create_rollback_report() {
    local target_version="$1"
    
    local report_file="${PROJECT_ROOT}/logs/rollback_report_$(date +%Y%m%d_%H%M%S).json"
    
    local current_info=$(get_current_deployment)
    
    cat > "$report_file" << EOF
{
    "rollback_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "target_version": "$target_version",
    "backup_path": "$BACKUP_PATH",
    "database_restored": $RESTORE_DATABASE,
    "final_state": $current_info,
    "rollback_duration": "$(($(date +%s) - rollback_start_time))s",
    "verification_status": "$(curl -s http://localhost/api/health | jq -r '.status // "unknown"')"
}
EOF
    
    log "INFO" "Rollback report created: $report_file"
}

# Main rollback function
main() {
    local rollback_start_time=$(date +%s)
    
    log "INFO" "Starting rollback procedure..."
    
    # Check prerequisites
    if ! check_prerequisites; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    fi
    
    # Get current deployment info
    local current_info=$(get_current_deployment)
    log "INFO" "Current deployment: $(echo "$current_info" | jq -c .)"
    
    # Find target version for rollback
    local target_version=""
    if ! target_version=$(find_previous_deployment); then
        log "ERROR" "Could not determine rollback target"
        
        log "INFO" "Available options:"
        list_available_backups
        exit 1
    fi
    
    # Confirm rollback
    confirm_rollback
    
    # Stop current services
    if ! stop_current_services; then
        log "ERROR" "Failed to stop current services"
        exit 1
    fi
    
    # Restore database if requested
    if ! restore_database; then
        log "ERROR" "Database restoration failed"
        exit 1
    fi
    
    # Start previous services
    if ! start_previous_services "$target_version"; then
        log "ERROR" "Failed to start previous services"
        exit 1
    fi
    
    # Wait for services to stabilize
    log "INFO" "Waiting for services to stabilize..."
    sleep 30
    
    # Verify rollback
    if ! verify_rollback "$target_version"; then
        log "ERROR" "Rollback verification failed"
        
        # Attempt basic recovery
        log "WARN" "Attempting to restart all services..."
        docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" \
                      -f "${PROJECT_ROOT}/docker-compose.production.yml" \
                      restart
        
        exit 1
    fi
    
    # Create rollback report
    create_rollback_report "$target_version"
    
    local rollback_duration=$(($(date +%s) - rollback_start_time))
    log "INFO" "Rollback completed successfully in ${rollback_duration}s"
    log "INFO" "System rolled back to: $target_version"
    
    # Send notification (if configured)
    # notify_rollback "$target_version" "$rollback_duration"
}

# Trap for cleanup on exit
cleanup_on_exit() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR" "Rollback failed with exit code: $exit_code"
    fi
    
    # Remove temporary files
    rm -f "${PROJECT_ROOT}/docker-compose.rollback.yml"
    
    exit $exit_code
}

trap cleanup_on_exit EXIT

# Parse arguments and run main function
parse_args "$@"
main