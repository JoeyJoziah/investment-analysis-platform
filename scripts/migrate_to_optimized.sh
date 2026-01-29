#!/bin/bash

# Migration Script: Transition to Optimized Architecture
# This script helps migrate from the current over-engineered setup to the budget-optimized configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Investment Analysis App Architecture Migration"
echo "Migrating to budget-optimized configuration"
echo "========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if .env exists
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_warn ".env file not found. Creating from template..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env" 2>/dev/null || {
            log_error "No .env or .env.example found"
            exit 1
        }
    fi
    
    log_info "Prerequisites check passed"
}

# Backup current configuration
backup_current_config() {
    log_info "Backing up current configuration..."
    
    BACKUP_DIR="$PROJECT_ROOT/backups/migration_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup docker-compose files
    if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
        cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/"
    fi
    
    if [ -f "$PROJECT_ROOT/docker-compose.postgres-replicas.yml" ]; then
        cp "$PROJECT_ROOT/docker-compose.postgres-replicas.yml" "$BACKUP_DIR/"
    fi
    
    # Backup kubernetes configs
    if [ -d "$PROJECT_ROOT/infrastructure/kubernetes" ]; then
        cp -r "$PROJECT_ROOT/infrastructure/kubernetes" "$BACKUP_DIR/"
    fi
    
    # Backup database settings
    if [ -f "$PROJECT_ROOT/backend/utils/database.py" ]; then
        cp "$PROJECT_ROOT/backend/utils/database.py" "$BACKUP_DIR/database.py.bak"
    fi
    
    log_info "Backup completed at: $BACKUP_DIR"
}

# Stop current services
stop_current_services() {
    log_info "Stopping current services..."
    
    # Stop PostgreSQL replicas if running
    if [ -f "$PROJECT_ROOT/docker-compose.postgres-replicas.yml" ]; then
        log_info "Stopping PostgreSQL replicas..."
        docker-compose -f "$PROJECT_ROOT/docker-compose.postgres-replicas.yml" down || true
    fi
    
    # Stop main services
    log_info "Stopping main services..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" down || true
    
    # Remove Istio if installed
    if kubectl get namespace istio-system &> /dev/null; then
        log_warn "Istio detected. Run 'istioctl uninstall --purge' manually to remove it"
    fi
    
    log_info "Services stopped"
}

# Apply optimized configuration
apply_optimized_config() {
    log_info "Applying optimized configuration..."
    
    # Use optimized docker-compose
    if [ -f "$PROJECT_ROOT/docker-compose.optimized.yml" ]; then
        log_info "Switching to optimized Docker Compose configuration..."
        mv "$PROJECT_ROOT/docker-compose.yml" "$PROJECT_ROOT/docker-compose.original.yml"
        cp "$PROJECT_ROOT/docker-compose.optimized.yml" "$PROJECT_ROOT/docker-compose.yml"
    fi
    
    # Update database configuration
    if [ -f "$PROJECT_ROOT/backend/utils/database_optimized.py" ]; then
        log_info "Updating database configuration..."
        mv "$PROJECT_ROOT/backend/utils/database.py" "$PROJECT_ROOT/backend/utils/database.original.py"
        cp "$PROJECT_ROOT/backend/utils/database_optimized.py" "$PROJECT_ROOT/backend/utils/database.py"
    fi
    
    # Update main.py
    if [ -f "$PROJECT_ROOT/backend/api/main_optimized.py" ]; then
        log_info "Updating FastAPI main application..."
        mv "$PROJECT_ROOT/backend/api/main.py" "$PROJECT_ROOT/backend/api/main.original.py"
        cp "$PROJECT_ROOT/backend/api/main_optimized.py" "$PROJECT_ROOT/backend/api/main.py"
    fi
    
    # Add graceful shutdown
    if [ -f "$PROJECT_ROOT/backend/utils/graceful_shutdown.py" ]; then
        log_info "Graceful shutdown handler added"
    fi
    
    log_info "Optimized configuration applied"
}

# Update environment variables
update_env_vars() {
    log_info "Updating environment variables for optimized settings..."
    
    # Check if env vars need updating
    if ! grep -q "DB_POOL_SIZE" "$PROJECT_ROOT/.env"; then
        cat >> "$PROJECT_ROOT/.env" << EOF

# Optimized Connection Pool Settings
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=5
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
REDIS_MAX_CONNECTIONS=20

# Resource Limits
MAX_WORKERS=2
MAX_CONCURRENT_REQUESTS=100
CACHE_WARMING_BATCH_SIZE=10
EOF
        log_info "Added optimized environment variables"
    fi
}

# Migrate data if needed
migrate_data() {
    log_info "Checking if data migration is needed..."
    
    # Check if PostgreSQL data exists
    if docker volume ls | grep -q "postgres_data"; then
        log_info "PostgreSQL data volume found. Data will be preserved."
    fi
    
    # Check if Redis data exists
    if docker volume ls | grep -q "redis_data"; then
        log_info "Redis data volume found. Cache will be preserved."
    fi
    
    log_info "No data migration needed - volumes will be reused"
}

# Start optimized services
start_optimized_services() {
    log_info "Starting optimized services..."
    
    # Build images
    log_info "Building Docker images..."
    docker-compose build
    
    # Start services
    log_info "Starting services with optimized configuration..."
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_info "Backend API is healthy"
    else
        log_warn "Backend API health check failed"
    fi
    
    log_info "Services started successfully"
}

# Verify migration
verify_migration() {
    log_info "Verifying migration..."
    
    # Check running containers
    RUNNING_CONTAINERS=$(docker-compose ps --services | wc -l)
    log_info "Running containers: $RUNNING_CONTAINERS"
    
    # Check database connections
    if docker-compose exec -T backend python -c "
from backend.utils.database import check_database_health
health = check_database_health()
print(f'Database health: {health[\"status\"]}')
print(f'Connection pool utilization: {health[\"utilization_percent\"]}%')
" 2>/dev/null; then
        log_info "Database connection verified"
    else
        log_warn "Could not verify database connection"
    fi
    
    # Check memory usage
    MEMORY_USAGE=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | tail -n +2)
    log_info "Memory usage:"
    echo "$MEMORY_USAGE"
    
    # Calculate total memory
    TOTAL_MEM=$(docker stats --no-stream --format "{{.MemUsage}}" | awk -F'/' '{gsub(/[^0-9.]/, "", $1); sum+=$1} END {print sum}')
    log_info "Total memory usage: ${TOTAL_MEM}MB"
    
    if (( $(echo "$TOTAL_MEM < 2000" | bc -l) )); then
        log_info "Memory usage is within budget constraints (<2GB)"
    else
        log_warn "Memory usage exceeds budget constraints (>2GB)"
    fi
}

# Cleanup old resources
cleanup_old_resources() {
    log_info "Cleaning up old resources..."
    
    # Remove old replicas volumes
    docker volume rm postgres-replica1-data postgres-replica2-data pgpool-data 2>/dev/null || true
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused networks
    docker network prune -f
    
    log_info "Cleanup completed"
}

# Main migration flow
main() {
    echo ""
    log_info "Starting migration process..."
    echo ""
    
    # Confirm migration
    read -p "This will migrate your setup to the optimized configuration. Continue? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Migration cancelled"
        exit 0
    fi
    
    # Run migration steps
    check_prerequisites
    backup_current_config
    stop_current_services
    apply_optimized_config
    update_env_vars
    migrate_data
    start_optimized_services
    verify_migration
    cleanup_old_resources
    
    echo ""
    echo "========================================="
    log_info "Migration completed successfully!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Monitor the application at http://localhost:8000/health"
    echo "2. Check logs: docker-compose logs -f"
    echo "3. View metrics: http://localhost:8000/metrics"
    echo "4. If using Kubernetes, apply: kubectl apply -f infrastructure/kubernetes/deployment-optimized.yaml"
    echo ""
    echo "To rollback if needed:"
    echo "  ./scripts/rollback_migration.sh"
    echo ""
}

# Run main function
main "$@"