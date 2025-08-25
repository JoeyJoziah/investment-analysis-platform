#!/bin/bash

# Production Deployment Script - Investment Analysis Platform
# Complete production deployment with cost optimization and monitoring
# Designed for <$50/month operational budget

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly DEPLOYMENT_LOG="${PROJECT_ROOT}/logs/production_deployment.log"
readonly COST_BUDGET_MONTHLY=50.00
readonly COST_BUDGET_DAILY=1.67

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default values
VERSION="latest"
DEPLOYMENT_TYPE="blue-green"
SKIP_TESTS=false
SKIP_BACKUP=false
ENABLE_MONITORING=true
DRY_RUN=false
AUTO_OPTIMIZE=true

# Logging function with cost tracking
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local cost_info=""
    
    # Add cost information to critical logs
    if [[ "$level" == "INFO" && "$message" =~ (deploy|start|scale) ]]; then
        local current_cost=$(calculate_current_cost)
        cost_info=" [Cost: $${current_cost}/day]"
    fi
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message${cost_info}" | tee -a "$DEPLOYMENT_LOG" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message${cost_info}" | tee -a "$DEPLOYMENT_LOG" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message${cost_info}" | tee -a "$DEPLOYMENT_LOG" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$DEPLOYMENT_LOG" ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$DEPLOYMENT_LOG"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Production Deployment Script for Investment Analysis Platform

OPTIONS:
    -v, --version VERSION     Version to deploy (default: latest)
    -t, --type TYPE          Deployment type: blue-green|rolling (default: blue-green)
    --skip-tests             Skip pre-deployment tests
    --skip-backup            Skip database backup
    --no-monitoring          Disable monitoring setup
    --no-optimize            Disable automatic optimization
    --dry-run                Show deployment plan without executing
    -h, --help              Show this help message

DEPLOYMENT TYPES:
    blue-green              Zero-downtime deployment with environment switch
    rolling                 Rolling update with service-by-service deployment

EXAMPLES:
    $0                                      # Deploy latest with all optimizations
    $0 -v v1.2.3 -t rolling               # Rolling deployment of specific version
    $0 --dry-run                           # Show deployment plan
    $0 --skip-tests --skip-backup          # Fast deployment (emergency only)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -t|--type)
                DEPLOYMENT_TYPE="$2"
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
            --no-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            --no-optimize)
                AUTO_OPTIMIZE=false
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

# Calculate current operational cost
calculate_current_cost() {
    local cost=0.0
    
    # Base infrastructure cost
    cost=$(echo "$cost + 0.50" | bc)  # Base server cost per day
    
    # Database cost (TimescaleDB)
    cost=$(echo "$cost + 0.30" | bc)  # Database cost per day
    
    # Redis cache cost
    cost=$(echo "$cost + 0.15" | bc)  # Cache cost per day
    
    # Container costs based on running containers
    local backend_containers=$(docker ps --filter "name=investment.*backend" --quiet | wc -l)
    local frontend_containers=$(docker ps --filter "name=investment.*frontend" --quiet | wc -l)
    local worker_containers=$(docker ps --filter "name=investment.*worker" --quiet | wc -l)
    
    cost=$(echo "$cost + ($backend_containers * 0.20)" | bc)    # $0.20/day per backend
    cost=$(echo "$cost + ($frontend_containers * 0.10)" | bc)   # $0.10/day per frontend
    cost=$(echo "$cost + ($worker_containers * 0.15)" | bc)     # $0.15/day per worker
    
    # Monitoring cost
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        cost=$(echo "$cost + 0.25" | bc)  # Monitoring stack cost
    fi
    
    printf "%.2f" "$cost"
}

# Cost validation
validate_cost_budget() {
    local projected_cost=$(calculate_current_cost)
    local cost_percentage=$(echo "scale=1; $projected_cost / $COST_BUDGET_DAILY * 100" | bc)
    
    log "INFO" "Projected daily cost: \$${projected_cost} (${cost_percentage}% of budget)"
    
    if (( $(echo "$projected_cost > $COST_BUDGET_DAILY" | bc -l) )); then
        log "ERROR" "Projected cost exceeds daily budget of \$${COST_BUDGET_DAILY}"
        
        if [[ "$DRY_RUN" != "true" ]]; then
            log "WARN" "Consider scaling down or optimizing deployment"
            read -p "Continue anyway? (yes/no): " -r confirmation
            if [[ "$confirmation" != "yes" ]]; then
                exit 1
            fi
        fi
    fi
}

# Pre-deployment checks
run_pre_deployment_checks() {
    log "INFO" "Running pre-deployment checks..."
    
    # Check Docker environment
    if ! docker info &> /dev/null; then
        log "ERROR" "Docker daemon is not running"
        return 1
    fi
    
    # Check available disk space
    local available_gb=$(df / | awk 'NR==2 {print $4/1024/1024}')
    if (( $(echo "$available_gb < 10" | bc -l) )); then
        log "ERROR" "Insufficient disk space: ${available_gb}GB available, 10GB minimum required"
        return 1
    fi
    
    # Check memory
    local available_memory=$(free -g | awk 'NR==2{print $7}')
    if [[ "$available_memory" -lt 2 ]]; then
        log "WARN" "Low available memory: ${available_memory}GB (4GB recommended)"
    fi
    
    # Check environment file
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        log "ERROR" "Environment file not found: ${PROJECT_ROOT}/.env"
        return 1
    fi
    
    # Validate API keys
    source "${PROJECT_ROOT}/.env"
    if [[ -z "${ALPHA_VANTAGE_API_KEY:-}" ]]; then
        log "WARN" "Alpha Vantage API key not set"
    fi
    
    # Check network connectivity
    if ! curl -sf https://api.polygon.io/v1/meta/symbols &> /dev/null; then
        log "WARN" "External API connectivity test failed - may affect data collection"
    fi
    
    log "INFO" "Pre-deployment checks completed"
}

# Optimize static assets
optimize_static_assets() {
    if [[ "$AUTO_OPTIMIZE" != "true" ]]; then
        return 0
    fi
    
    log "INFO" "Optimizing static assets..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would optimize static assets"
        return 0
    fi
    
    # Optimize frontend assets
    if [[ -d "${PROJECT_ROOT}/frontend/web/src" ]]; then
        python3 "${PROJECT_ROOT}/scripts/optimization/static_asset_optimizer.py" \
            "${PROJECT_ROOT}/frontend/web/src" \
            "${PROJECT_ROOT}/frontend/web/build/static" \
            --verbose
    fi
    
    log "INFO" "Static asset optimization completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        return 0
    fi
    
    log "INFO" "Setting up production monitoring..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would setup monitoring stack"
        return 0
    fi
    
    # Start monitoring services
    docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" \
                  -f "${PROJECT_ROOT}/docker-compose.production.yml" \
                  up -d prometheus grafana alertmanager
    
    # Wait for services to be ready
    local timeout=60
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf http://localhost:9090/-/ready &> /dev/null && \
           curl -sf http://localhost:3001/api/health &> /dev/null; then
            log "INFO" "Monitoring services are ready"
            break
        fi
        
        sleep 5
        ((elapsed += 5))
    done
    
    if [[ $elapsed -ge $timeout ]]; then
        log "WARN" "Monitoring services startup timeout - continuing without monitoring"
        return 1
    fi
    
    # Configure cost monitoring alerts
    setup_cost_alerts
    
    log "INFO" "Monitoring setup completed"
}

# Setup cost monitoring alerts
setup_cost_alerts() {
    log "INFO" "Configuring cost monitoring alerts..."
    
    # Create cost alert rules
    cat > "${PROJECT_ROOT}/infrastructure/monitoring/alerts/cost-alerts.yml" << EOF
groups:
- name: cost-management
  rules:
  - alert: DailyCostBudgetExceeded
    expr: investment_daily_cost_projection > ${COST_BUDGET_DAILY}
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Daily cost budget exceeded"
      description: "Current daily cost projection: \${{ \$value }} exceeds budget of \$${COST_BUDGET_DAILY}"
      
  - alert: DailyCostBudgetWarning
    expr: investment_daily_cost_projection > $(echo "$COST_BUDGET_DAILY * 0.8" | bc)
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Daily cost budget warning"
      description: "Current daily cost projection: \${{ \$value }} approaching budget of \$${COST_BUDGET_DAILY}"
      
  - alert: APIRateLimitApproaching
    expr: api_usage_rate > 80
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "API rate limit approaching"
      description: "API usage at {{ \$value }}% of rate limit"
EOF

    # Reload Prometheus configuration
    curl -X POST http://localhost:9090/-/reload &> /dev/null || true
    
    log "INFO" "Cost monitoring alerts configured"
}

# Deploy application
deploy_application() {
    log "INFO" "Starting application deployment..."
    
    case $DEPLOYMENT_TYPE in
        "blue-green")
            deploy_blue_green
            ;;
        "rolling")
            deploy_rolling
            ;;
        *)
            log "ERROR" "Unknown deployment type: $DEPLOYMENT_TYPE"
            return 1
            ;;
    esac
}

# Blue-green deployment
deploy_blue_green() {
    log "INFO" "Executing blue-green deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would execute blue-green deployment"
        return 0
    fi
    
    # Use the existing blue-green deployment script
    local blue_green_args=""
    
    if [[ "$SKIP_TESTS" == "true" ]]; then
        blue_green_args="$blue_green_args --skip-tests"
    fi
    
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        blue_green_args="$blue_green_args --skip-backup"
    fi
    
    "${PROJECT_ROOT}/scripts/deployment/blue_green_deploy.sh" \
        --version "$VERSION" \
        $blue_green_args
}

# Rolling deployment
deploy_rolling() {
    log "INFO" "Executing rolling deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would execute rolling deployment"
        return 0
    fi
    
    # Deploy backend first
    log "INFO" "Deploying backend services..."
    docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" \
                  -f "${PROJECT_ROOT}/docker-compose.production.yml" \
                  up -d --force-recreate --no-deps backend celery_worker celery_beat
    
    # Wait for backend health
    wait_for_service_health "http://localhost:8000/api/health" 120
    
    # Deploy frontend
    log "INFO" "Deploying frontend services..."
    docker-compose -f "${PROJECT_ROOT}/docker-compose.yml" \
                  -f "${PROJECT_ROOT}/docker-compose.production.yml" \
                  up -d --force-recreate --no-deps frontend nginx
    
    # Final health check
    wait_for_service_health "http://localhost/api/health" 60
    
    log "INFO" "Rolling deployment completed"
}

# Wait for service health
wait_for_service_health() {
    local url="$1"
    local timeout="$2"
    local elapsed=0
    
    log "INFO" "Waiting for service health: $url"
    
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "$url" &> /dev/null; then
            log "INFO" "Service is healthy: $url"
            return 0
        fi
        
        sleep 5
        ((elapsed += 5))
    done
    
    log "ERROR" "Service health check timeout: $url"
    return 1
}

# Post-deployment optimization
post_deployment_optimization() {
    if [[ "$AUTO_OPTIMIZE" != "true" ]]; then
        return 0
    fi
    
    log "INFO" "Running post-deployment optimizations..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would run post-deployment optimizations"
        return 0
    fi
    
    # Warm up cache
    log "INFO" "Warming up application cache..."
    curl -sf "http://localhost/api/stocks/prices?symbols=AAPL,GOOGL,MSFT" &> /dev/null || true
    curl -sf "http://localhost/api/market/overview" &> /dev/null || true
    
    # Database optimization
    log "INFO" "Running database optimization..."
    docker exec investment_db_prod psql -U postgres -d investment_db -c "ANALYZE;" &> /dev/null || true
    
    # Check and optimize container resource usage
    optimize_container_resources
    
    log "INFO" "Post-deployment optimizations completed"
}

# Optimize container resources based on actual usage
optimize_container_resources() {
    log "INFO" "Optimizing container resources..."
    
    # Check container resource usage
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | \
    while IFS=$'\t' read -r container cpu memory; do
        if [[ "$container" == "CONTAINER" ]]; then
            continue
        fi
        
        # Extract numeric CPU percentage
        cpu_numeric=$(echo "$cpu" | sed 's/%//')
        
        # If CPU usage is very low, consider scaling down
        if (( $(echo "$cpu_numeric < 10" | bc -l) 2>/dev/null )); then
            log "DEBUG" "Low CPU usage detected for $container: $cpu"
        fi
        
        # Log resource usage for monitoring
        log "DEBUG" "Resource usage - $container: CPU $cpu, Memory $memory"
    done
}

# Generate deployment report
generate_deployment_report() {
    log "INFO" "Generating deployment report..."
    
    local report_file="${PROJECT_ROOT}/logs/deployment_report_$(date +%Y%m%d_%H%M%S).json"
    local deployment_end_time=$(date +%s)
    local deployment_duration=$((deployment_end_time - deployment_start_time))
    local current_cost=$(calculate_current_cost)
    
    # Gather deployment information
    local deployment_info=$(cat << EOF
{
    "deployment": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "version": "$VERSION",
        "type": "$DEPLOYMENT_TYPE",
        "duration_seconds": $deployment_duration,
        "dry_run": $DRY_RUN
    },
    "cost_analysis": {
        "daily_projection": $current_cost,
        "monthly_projection": $(echo "$current_cost * 30" | bc),
        "budget_utilization": $(echo "scale=1; $current_cost / $COST_BUDGET_DAILY * 100" | bc)
    },
    "infrastructure": {
        "containers": $(docker ps --format "{{.Names}}" | grep "investment_" | wc -l),
        "images": $(docker images --format "{{.Repository}}:{{.Tag}}" | grep "investment-" | wc -l),
        "networks": $(docker network ls --format "{{.Name}}" | grep "investment" | wc -l)
    },
    "health_status": $(curl -s http://localhost/api/health 2>/dev/null || echo '{"status":"unknown"}'),
    "optimization": {
        "static_assets": $AUTO_OPTIMIZE,
        "monitoring_enabled": $ENABLE_MONITORING,
        "tests_skipped": $SKIP_TESTS,
        "backup_skipped": $SKIP_BACKUP
    }
}
EOF
)
    
    echo "$deployment_info" > "$report_file"
    log "INFO" "Deployment report generated: $report_file"
    
    # Display cost summary
    echo
    echo "==============================================="
    echo "DEPLOYMENT COST SUMMARY"
    echo "==============================================="
    echo "Daily Cost Projection: \$${current_cost}"
    echo "Monthly Cost Projection: \$$(echo "$current_cost * 30" | bc)"
    echo "Budget Utilization: $(echo "scale=1; $current_cost / $COST_BUDGET_DAILY * 100" | bc)%"
    echo "Deployment Duration: ${deployment_duration}s"
    echo "==============================================="
    echo
}

# Main deployment function
main() {
    local deployment_start_time=$(date +%s)
    
    log "INFO" "Starting production deployment process..."
    log "INFO" "Version: $VERSION, Type: $DEPLOYMENT_TYPE"
    
    # Create logs directory
    mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
    
    # Run pre-deployment checks
    if ! run_pre_deployment_checks; then
        log "ERROR" "Pre-deployment checks failed"
        exit 1
    fi
    
    # Validate cost budget
    validate_cost_budget
    
    # Optimize static assets
    optimize_static_assets
    
    # Setup monitoring
    if ! setup_monitoring; then
        log "WARN" "Monitoring setup failed - continuing deployment"
    fi
    
    # Deploy application
    if ! deploy_application; then
        log "ERROR" "Application deployment failed"
        
        # Attempt automatic rollback
        log "WARN" "Attempting automatic rollback..."
        "${PROJECT_ROOT}/scripts/deployment/rollback.sh" --force || true
        
        exit 1
    fi
    
    # Post-deployment optimization
    post_deployment_optimization
    
    # Generate deployment report
    generate_deployment_report
    
    local deployment_end_time=$(date +%s)
    local total_duration=$((deployment_end_time - deployment_start_time))
    
    log "INFO" "Production deployment completed successfully in ${total_duration}s"
    log "INFO" "Daily cost projection: \$$(calculate_current_cost)"
    log "INFO" "Monitoring dashboard: http://localhost:3001"
    
    # Final health check
    if curl -sf http://localhost/api/health &> /dev/null; then
        log "INFO" "Application is healthy and ready for traffic"
    else
        log "WARN" "Application may not be fully ready - check logs"
    fi
}

# Cleanup function
cleanup_on_exit() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR" "Production deployment failed with exit code: $exit_code"
        
        # Save failure state for debugging
        docker ps > "${PROJECT_ROOT}/logs/containers_state_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
        docker-compose logs --tail=100 > "${PROJECT_ROOT}/logs/compose_logs_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
    fi
    
    exit $exit_code
}

# Set trap for cleanup
trap cleanup_on_exit EXIT

# Parse arguments and run main function
parse_args "$@"
main