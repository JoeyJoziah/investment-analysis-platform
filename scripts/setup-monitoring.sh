#!/bin/bash
set -e

echo "Setting up comprehensive monitoring for Investment Analysis Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from project root
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Create monitoring directories if they don't exist
print_status "Creating monitoring directories..."
mkdir -p infrastructure/monitoring/{alerts,grafana/{provisioning/{dashboards,datasources,alerting},dashboards}}
mkdir -p infrastructure/monitoring/grafana/templates

# Set permissions for Grafana
print_status "Setting up Grafana permissions..."
sudo chown -R 472:472 infrastructure/monitoring/grafana/ || print_warning "Could not set Grafana permissions (you may need to run with sudo)"

# Validate Prometheus configuration
print_status "Validating Prometheus configuration..."
if command -v promtool &> /dev/null; then
    promtool check config infrastructure/monitoring/prometheus.yml || {
        print_error "Prometheus configuration validation failed"
        exit 1
    }
    print_status "Prometheus configuration is valid"
else
    print_warning "promtool not found, skipping Prometheus config validation"
fi

# Validate AlertManager configuration
print_status "Validating AlertManager configuration..."
if command -v amtool &> /dev/null; then
    amtool check-config infrastructure/monitoring/alertmanager.yml || {
        print_error "AlertManager configuration validation failed"
        exit 1
    }
    print_status "AlertManager configuration is valid"
else
    print_warning "amtool not found, skipping AlertManager config validation"
fi

# Check if monitoring services are already running
print_status "Checking existing monitoring services..."
if docker ps | grep -q "investment_prometheus\|investment_grafana"; then
    print_warning "Some monitoring services are already running"
    read -p "Do you want to restart them? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping existing monitoring services..."
        docker-compose stop prometheus grafana alertmanager node-exporter cadvisor postgres-exporter redis-exporter elasticsearch-exporter nginx-exporter || true
    fi
fi

# Start monitoring stack
print_status "Starting monitoring services..."
docker-compose up -d prometheus grafana alertmanager node-exporter cadvisor postgres-exporter redis-exporter elasticsearch-exporter nginx-exporter

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check service health
print_status "Checking service health..."
services=(
    "prometheus:9090"
    "grafana:3001" 
    "alertmanager:9093"
    "node-exporter:9100"
    "cadvisor:8080"
    "postgres-exporter:9187"
    "redis-exporter:9121"
    "elasticsearch-exporter:9114"
)

failed_services=()
for service in "${services[@]}"; do
    service_name=$(echo $service | cut -d':' -f1)
    port=$(echo $service | cut -d':' -f2)
    
    if curl -s -f "http://localhost:$port" > /dev/null 2>&1; then
        print_status "$service_name is healthy"
    else
        print_error "$service_name failed health check"
        failed_services+=("$service_name")
    fi
done

# Report results
if [ ${#failed_services[@]} -eq 0 ]; then
    print_status "All monitoring services are running successfully!"
    echo
    echo "Access URLs:"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3001 (admin/\$GRAFANA_PASSWORD)"
    echo "- AlertManager: http://localhost:9093"
    echo
    echo "Dashboards imported:"
    echo "- System Overview"
    echo "- API Performance"
    echo "- Business Metrics" 
    echo "- Database Performance"
    echo "- External APIs"
    echo
    print_status "Monitoring setup completed successfully!"
else
    print_error "Some services failed to start: ${failed_services[*]}"
    echo "Check the logs with: docker-compose logs <service_name>"
    exit 1
fi

# Display cost optimization tips
echo
print_status "Cost Optimization Tips:"
echo "- Monitor the 'Business Metrics' dashboard for budget usage"
echo "- Set up alerts when approaching the \$50/month limit"
echo "- Use the 'External APIs' dashboard to track API call costs"
echo "- Review resource usage regularly in the 'System Overview' dashboard"
echo
print_status "The monitoring system is optimized for your \$50/month budget constraint"
