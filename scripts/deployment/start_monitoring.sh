#!/bin/bash
# Start comprehensive monitoring infrastructure for Investment Analysis Application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if Docker and Docker Compose are installed
check_requirements() {
    log "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    log "Requirements check passed."
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    # Monitoring directories
    mkdir -p infrastructure/monitoring/prometheus/dynamic
    mkdir -p infrastructure/monitoring/alertmanager/templates
    mkdir -p infrastructure/monitoring/grafana/provisioning/datasources
    mkdir -p infrastructure/monitoring/grafana/provisioning/dashboards
    mkdir -p infrastructure/monitoring/logstash/pipeline
    mkdir -p infrastructure/monitoring/logstash/config
    mkdir -p infrastructure/monitoring/blackbox
    
    # Log directories
    mkdir -p logs
    mkdir -p data/cache
    mkdir -p data/exports
    
    log "Directories created successfully."
}

# Generate monitoring configuration files
generate_configs() {
    log "Generating monitoring configuration files..."
    
    # Grafana datasource configuration
    cat > infrastructure/monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: "investment-analysis-logs-*"
    timeField: "@timestamp"
    
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
EOF

    # Grafana dashboard provisioning
    cat > infrastructure/monitoring/grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'investment-analysis'
    orgId: 1
    folder: 'Investment Analysis'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    # Blackbox exporter configuration
    cat > infrastructure/monitoring/blackbox/blackbox.yml << EOF
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: []
      method: GET
      preferred_ip_protocol: "ip4"
      
  tcp_connect:
    prober: tcp
    timeout: 5s
    
  ssl_expiry:
    prober: tcp
    timeout: 10s
    tcp:
      query_response:
        - expect: "^SSH-2.0-"
      tls: true
      tls_config:
        insecure_skip_verify: false
EOF

    # Logstash pipeline configuration
    cat > infrastructure/monitoring/logstash/pipeline/investment-analysis.conf << EOF
input {
  file {
    path => "/app/logs/*.log"
    start_position => "beginning"
    sincedb_path => "/dev/null"
    codec => json
  }
  
  beats {
    port => 5044
  }
}

filter {
  if [service] == "investment_analysis" {
    mutate {
      add_tag => ["investment-analysis"]
    }
    
    # Parse log level
    if [level] {
      mutate {
        uppercase => ["level"]
      }
    }
    
    # Add timestamp processing
    date {
      match => ["timestamp", "ISO8601"]
    }
    
    # Extract error information
    if [level] == "ERROR" and [exception] {
      mutate {
        add_field => {
          "error_type" => "%{[exception][type]}"
          "error_message" => "%{[exception][message]}"
        }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "investment-analysis-logs-%{+YYYY.MM}"
  }
  
  stdout {
    codec => rubydebug
  }
}
EOF

    # Logstash main config
    cat > infrastructure/monitoring/logstash/config/logstash.yml << EOF
http.host: "0.0.0.0"
xpack.monitoring.enabled: false
EOF

    log "Configuration files generated successfully."
}

# Validate Docker Compose files
validate_compose_files() {
    log "Validating Docker Compose files..."
    
    if [[ ! -f "docker-compose.yml" ]]; then
        error "Main docker-compose.yml file not found."
    fi
    
    if [[ ! -f "docker-compose.monitoring.yml" ]]; then
        error "Monitoring docker-compose.monitoring.yml file not found."
    fi
    
    # Validate compose files
    docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml config > /dev/null || error "Invalid Docker Compose configuration"
    
    log "Docker Compose files validation passed."
}

# Pull required Docker images
pull_images() {
    log "Pulling required Docker images..."
    
    images=(
        "prom/prometheus:v2.45.0"
        "grafana/grafana:10.0.0"
        "grafana/grafana-image-renderer:3.8.0"
        "prom/alertmanager:v0.26.0"
        "docker.elastic.co/elasticsearch/elasticsearch:8.9.0"
        "docker.elastic.co/kibana/kibana:8.9.0"
        "docker.elastic.co/logstash/logstash:8.9.0"
        "prom/node-exporter:v1.6.0"
        "gcr.io/cadvisor/cadvisor:v0.47.0"
        "prom/blackbox-exporter:v0.24.0"
        "jaegertracing/all-in-one:1.47"
        "redis:7-alpine"
    )
    
    for image in "${images[@]}"; do
        log "Pulling $image..."
        docker pull "$image" || warn "Failed to pull $image"
    done
    
    log "Image pulling completed."
}

# Start monitoring services
start_monitoring() {
    log "Starting monitoring infrastructure..."
    
    # Start monitoring services
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    check_service_health() {
        local service=$1
        local port=$2
        local endpoint=${3:-"/"}
        
        for i in {1..30}; do
            if curl -sf "http://localhost:$port$endpoint" > /dev/null 2>&1; then
                log "$service is healthy"
                return 0
            fi
            sleep 5
        done
        warn "$service may not be healthy"
        return 1
    }
    
    # Check all services
    check_service_health "Prometheus" 9090 "/-/healthy"
    check_service_health "Grafana" 3001 "/api/health"
    check_service_health "Alertmanager" 9093 "/-/healthy"
    check_service_health "Elasticsearch" 9200 "/_cluster/health"
    check_service_health "Kibana" 5601 "/api/status"
    check_service_health "Node Exporter" 9100 "/metrics"
    check_service_health "cAdvisor" 8080 "/healthz"
    check_service_health "Blackbox Exporter" 9115 "/-/healthy"
    check_service_health "Jaeger" 16686 "/"
    
    log "Monitoring infrastructure started successfully!"
}

# Configure Grafana dashboards
configure_grafana() {
    log "Configuring Grafana dashboards..."
    
    # Wait for Grafana to be fully ready
    sleep 10
    
    # Import dashboards using Grafana API
    grafana_url="http://admin:admin123@localhost:3001"
    
    # Import system health dashboard
    if [[ -f "infrastructure/monitoring/grafana/dashboards/system_health_dashboard.json" ]]; then
        curl -X POST \
            -H "Content-Type: application/json" \
            -d @infrastructure/monitoring/grafana/dashboards/system_health_dashboard.json \
            "$grafana_url/api/dashboards/db" > /dev/null 2>&1 || warn "Failed to import system health dashboard"
    fi
    
    # Import business metrics dashboard
    if [[ -f "infrastructure/monitoring/grafana/dashboards/business_metrics_dashboard.json" ]]; then
        curl -X POST \
            -H "Content-Type: application/json" \
            -d @infrastructure/monitoring/grafana/dashboards/business_metrics_dashboard.json \
            "$grafana_url/api/dashboards/db" > /dev/null 2>&1 || warn "Failed to import business metrics dashboard"
    fi
    
    # Import cost monitoring dashboard
    if [[ -f "infrastructure/monitoring/grafana/dashboards/cost_monitoring_dashboard.json" ]]; then
        curl -X POST \
            -H "Content-Type: application/json" \
            -d @infrastructure/monitoring/grafana/dashboards/cost_monitoring_dashboard.json \
            "$grafana_url/api/dashboards/db" > /dev/null 2>&1 || warn "Failed to import cost monitoring dashboard"
    fi
    
    log "Grafana dashboards configured."
}

# Start main application with monitoring enabled
start_application_with_monitoring() {
    log "Starting main application with monitoring enabled..."
    
    # Set monitoring environment variables
    export ENABLE_MONITORING=true
    export PROMETHEUS_ENABLED=true
    export GRAFANA_URL=http://localhost:3001
    export ELASTICSEARCH_URL=http://localhost:9200
    export JAEGER_URL=http://localhost:16686
    
    # Start main application
    docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
    
    log "Application started with monitoring enabled."
}

# Display monitoring endpoints
show_endpoints() {
    log "Monitoring endpoints:"
    echo ""
    echo -e "${BLUE}📊 Monitoring Dashboards:${NC}"
    echo "  Grafana:           http://localhost:3001 (admin/admin123)"
    echo "  Prometheus:        http://localhost:9090"
    echo "  Alertmanager:      http://localhost:9093"
    echo ""
    echo -e "${BLUE}📋 Log Management:${NC}"
    echo "  Kibana:            http://localhost:5601"
    echo "  Elasticsearch:     http://localhost:9200"
    echo ""
    echo -e "${BLUE}🔍 Tracing & Metrics:${NC}"
    echo "  Jaeger:            http://localhost:16686"
    echo "  Node Exporter:     http://localhost:9100"
    echo "  cAdvisor:          http://localhost:8080"
    echo ""
    echo -e "${BLUE}📱 Application:${NC}"
    echo "  Backend API:       http://localhost:8000"
    echo "  Frontend:          http://localhost:3000"
    echo "  API Docs:          http://localhost:8000/docs"
    echo ""
    echo -e "${YELLOW}📚 Default Dashboards:${NC}"
    echo "  System Health:     http://localhost:3001/d/investment-system-health"
    echo "  Business Metrics:  http://localhost:3001/d/investment-business-metrics"
    echo "  Cost Monitoring:   http://localhost:3001/d/investment-cost-monitoring"
    echo ""
}

# Main execution
main() {
    log "🚀 Starting Investment Analysis Monitoring Infrastructure"
    echo "=================================================="
    
    check_requirements
    create_directories
    generate_configs
    validate_compose_files
    pull_images
    start_monitoring
    configure_grafana
    start_application_with_monitoring
    
    echo ""
    log "✅ Monitoring infrastructure started successfully!"
    show_endpoints
    
    echo ""
    log "🎯 Quick Start Tips:"
    echo "  1. Visit Grafana at http://localhost:3001 (admin/admin123)"
    echo "  2. Check system health dashboard"
    echo "  3. Monitor cost usage to stay under $50/month budget"
    echo "  4. Set up alert notifications in Alertmanager"
    echo ""
    log "📖 View logs: docker-compose logs -f [service-name]"
    log "🛑 Stop monitoring: docker-compose -f docker-compose.monitoring.yml down"
}

# Handle script interruption
trap 'error "Script interrupted by user"' INT TERM

# Run main function
main "$@"