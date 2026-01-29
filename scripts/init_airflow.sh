#!/bin/bash

# Initialize Apache Airflow for Investment Analysis Application
# This script sets up Airflow database, connections, pools, users, and variables

set -e  # Exit on any error

echo "🚀 Initializing Apache Airflow for Investment Analysis Application..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found! Please create it first."
    exit 1
fi

# Set Airflow environment variables
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__WEBSERVER__EXPOSE_CONFIG=False
export AIRFLOW__CORE__STORE_SERIALIZED_DAGS=True
export AIRFLOW_UID=${AIRFLOW_UID:-50000}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start within expected time"
    return 1
}

# Function to check if Airflow database is initialized
check_airflow_db() {
    echo "🔍 Checking if Airflow database is initialized..."
    
    # Try to connect to Airflow database
    if PGPASSWORD="${AIRFLOW_DB_PASSWORD:-airflow_password_123}" psql -h localhost -U airflow_user -d airflow_db -c "SELECT 1 FROM dag WHERE dag_id='daily_market_analysis' LIMIT 1;" >/dev/null 2>&1; then
        echo "✅ Airflow database is already initialized!"
        return 0
    else
        echo "ℹ️  Airflow database needs initialization"
        return 1
    fi
}

# Function to initialize Airflow database
init_airflow_database() {
    echo "🗄️  Initializing Airflow database..."
    
    # Initialize the database
    docker-compose -f docker-compose.airflow.yml run --rm airflow-init || {
        echo "❌ Failed to initialize Airflow database"
        exit 1
    }
    
    # Wait a moment for the database to be fully ready
    sleep 10
    
    echo "✅ Airflow database initialized successfully!"
}

# Function to create Airflow connections
create_connections() {
    echo "🔗 Creating Airflow connections..."
    
    # PostgreSQL connection for investment database
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
        'investment_postgres' \
        --conn-type 'postgres' \
        --conn-host 'postgres' \
        --conn-port '5432' \
        --conn-login 'postgres' \
        --conn-password "${DB_PASSWORD}" \
        --conn-schema 'investment_db' \
        --conn-extra '{"sslmode": "prefer"}' || echo "Connection investment_postgres may already exist"
    
    # Redis connection
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
        'investment_redis' \
        --conn-type 'redis' \
        --conn-host 'redis' \
        --conn-port '6379' \
        --conn-password "${REDIS_PASSWORD}" \
        --conn-extra '{"db": 0}' || echo "Connection investment_redis may already exist"
    
    # Elasticsearch connection
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
        'investment_elasticsearch' \
        --conn-type 'elasticsearch' \
        --conn-host 'elasticsearch' \
        --conn-port '9200' \
        --conn-extra '{"http_compress": true, "timeout": 30}' || echo "Connection investment_elasticsearch may already exist"
    
    # HTTP connection for Alpha Vantage
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
        'alpha_vantage_http' \
        --conn-type 'http' \
        --conn-host 'www.alphavantage.co' \
        --conn-extra "{\"endpoint_prefix\": \"/query\", \"headers\": {\"User-Agent\": \"Investment-Analysis-App/1.0\"}}" || echo "Connection alpha_vantage_http may already exist"
    
    # HTTP connection for Finnhub
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
        'finnhub_http' \
        --conn-type 'http' \
        --conn-host 'finnhub.io' \
        --conn-extra "{\"endpoint_prefix\": \"/api/v1\", \"headers\": {\"User-Agent\": \"Investment-Analysis-App/1.0\"}}" || echo "Connection finnhub_http may already exist"
    
    # HTTP connection for Polygon
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
        'polygon_http' \
        --conn-type 'http' \
        --conn-host 'api.polygon.io' \
        --conn-extra "{\"endpoint_prefix\": \"/v2\", \"headers\": {\"User-Agent\": \"Investment-Analysis-App/1.0\"}}" || echo "Connection polygon_http may already exist"
    
    # SMTP connection for email notifications
    if [ ! -z "${SMTP_USER}" ]; then
        docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections add \
            'smtp_default' \
            --conn-type 'email' \
            --conn-host 'smtp.gmail.com' \
            --conn-port '587' \
            --conn-login "${SMTP_USER}" \
            --conn-password "${SMTP_PASSWORD}" \
            --conn-extra '{"use_tls": true}' || echo "Connection smtp_default may already exist"
    fi
    
    echo "✅ Connections created successfully!"
}

# Function to create Airflow pools for resource management
create_pools() {
    echo "🏊 Creating Airflow pools for resource management..."
    
    # Pool for API calls (limited by external API rate limits)
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow pools set \
        'api_calls' 5 'Pool for external API calls with rate limiting' || echo "Pool api_calls may already exist"
    
    # Pool for compute-intensive tasks (ML, analytics)
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow pools set \
        'compute_intensive' 8 'Pool for compute-intensive ML and analytics tasks' || echo "Pool compute_intensive may already exist"
    
    # Pool for database operations
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow pools set \
        'database_tasks' 12 'Pool for database read/write operations' || echo "Pool database_tasks may already exist"
    
    # Pool for low priority maintenance tasks
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow pools set \
        'low_priority' 3 'Pool for low priority maintenance and cleanup tasks' || echo "Pool low_priority may already exist"
    
    # Pool for high-frequency data processing
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow pools set \
        'high_frequency' 2 'Pool for high-frequency intraday data processing' || echo "Pool high_frequency may already exist"
    
    echo "✅ Pools created successfully!"
}

# Function to set Airflow variables
create_variables() {
    echo "📝 Creating Airflow variables..."
    
    # API rate limits
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'DAILY_API_LIMIT_ALPHA_VANTAGE' "${DAILY_API_LIMIT_ALPHA_VANTAGE:-25}"
    
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'DAILY_API_LIMIT_FINNHUB' "${DAILY_API_LIMIT_FINNHUB:-1800}"
    
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'DAILY_API_LIMIT_POLYGON' "${DAILY_API_LIMIT_POLYGON:-150}"
    
    # Cost monitoring
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'MONTHLY_BUDGET_LIMIT' "${MONTHLY_BUDGET_LIMIT:-50}"
    
    # Processing configuration
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'STOCK_PROCESSING_BATCH_SIZE' '100'
    
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'MAX_PARALLEL_API_CALLS' '5'
    
    # Data quality thresholds
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'DATA_QUALITY_THRESHOLD' '0.85'
    
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'CACHE_TTL_HOURS' '24'
    
    # Emergency mode settings
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'EMERGENCY_MODE_ENABLED' 'false'
    
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow variables set \
        'FALLBACK_TO_CACHE_HOURS' '72'
    
    echo "✅ Variables created successfully!"
}

# Function to create admin user
create_admin_user() {
    echo "👤 Creating admin user..."
    
    # Create admin user if it doesn't exist
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow users create \
        --username "${AIRFLOW_ADMIN_USERNAME:-admin}" \
        --firstname "Investment" \
        --lastname "Admin" \
        --role "Admin" \
        --email "${AIRFLOW_ADMIN_EMAIL:-admin@investment-analysis.com}" \
        --password "${AIRFLOW_ADMIN_PASSWORD:-admin123}" || echo "Admin user may already exist"
    
    echo "✅ Admin user created successfully!"
}

# Function to enable DAGs
enable_dags() {
    echo "🔄 Enabling DAGs..."
    
    # Enable main DAG
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow dags unpause \
        'daily_market_analysis' || echo "DAG daily_market_analysis not found yet"
    
    # Enable optimized DAG
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow dags unpause \
        'daily_market_analysis_optimized' || echo "DAG daily_market_analysis_optimized not found yet"
    
    # Enable bulk operations DAG
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow dags unpause \
        'bulk_operations_enhanced' || echo "DAG bulk_operations_enhanced not found yet"
    
    echo "✅ DAGs enabled successfully!"
}

# Function to test connections
test_connections() {
    echo "🧪 Testing connections..."
    
    # Test PostgreSQL connection
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections test \
        'investment_postgres' || echo "⚠️  PostgreSQL connection test failed"
    
    # Test Redis connection
    docker-compose -f docker-compose.airflow.yml exec -T airflow-webserver airflow connections test \
        'investment_redis' || echo "⚠️  Redis connection test failed"
    
    echo "✅ Connection tests completed!"
}

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    # Add any cleanup tasks here if needed
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution flow
main() {
    echo "🚀 Starting Airflow initialization process..."
    
    # Start services if not already running
    echo "🚀 Starting Airflow services..."
    docker-compose -f docker-compose.airflow.yml up -d postgres redis
    
    # Wait for core services
    wait_for_service "localhost" "5432" "PostgreSQL"
    wait_for_service "localhost" "6379" "Redis"
    
    # Initialize database if needed
    if ! check_airflow_db; then
        init_airflow_database
    fi
    
    # Start Airflow services
    echo "🚀 Starting Airflow webserver and scheduler..."
    docker-compose -f docker-compose.airflow.yml up -d airflow-webserver airflow-scheduler
    
    # Wait for Airflow webserver
    wait_for_service "localhost" "8080" "Airflow Webserver"
    
    # Give the webserver a moment to fully initialize
    echo "⏳ Waiting for Airflow to fully initialize..."
    sleep 15
    
    # Setup Airflow components
    create_connections
    create_pools
    create_variables
    create_admin_user
    enable_dags
    test_connections
    
    # Start all workers
    echo "🚀 Starting Airflow workers..."
    docker-compose -f docker-compose.airflow.yml up -d
    
    echo ""
    echo "🎉 Airflow initialization completed successfully!"
    echo ""
    echo "📊 Airflow Dashboard: http://localhost:8080"
    echo "🌺 Celery Flower: http://localhost:5555"
    echo ""
    echo "👤 Admin Credentials:"
    echo "   Username: ${AIRFLOW_ADMIN_USERNAME:-admin}"
    echo "   Password: ${AIRFLOW_ADMIN_PASSWORD:-admin123}"
    echo ""
    echo "🔧 Next steps:"
    echo "   1. Access the Airflow dashboard at http://localhost:8080"
    echo "   2. Verify that DAGs are visible and unpaused"
    echo "   3. Check connections in Admin > Connections"
    echo "   4. Monitor pools in Admin > Pools"
    echo "   5. Trigger a test run of daily_market_analysis DAG"
    echo ""
    echo "📋 Environment Summary:"
    echo "   - Database: PostgreSQL with TimescaleDB"
    echo "   - Cache: Redis"
    echo "   - Executor: CeleryExecutor"
    echo "   - Workers: 3 (api_calls, compute_intensive, default)"
    echo "   - Monitoring: StatsD + Prometheus integration"
    echo ""
}

# Check if running as root (for Docker)
if [ "$(id -u)" = "0" ]; then
    echo "⚠️  Running as root. Make sure Docker is properly configured."
fi

# Check required environment variables
required_vars=("DB_PASSWORD" "REDIS_PASSWORD" "AIRFLOW_FERNET_KEY" "AIRFLOW_SECRET_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set!"
        echo "Please set it in your .env file."
        exit 1
    fi
done

# Run main function
main

echo "✨ Airflow initialization script completed!"