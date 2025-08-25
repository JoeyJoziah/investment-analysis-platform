#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Investment Platform - Monitoring Health Check${NC}"
echo "=============================================="

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✓${NC} $service_name is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name is unhealthy"
        return 1
    fi
}

# Function to check metric availability
check_metrics() {
    local service_name=$1
    local url=$2
    local metric_name=$3
    
    if curl -s "$url" | grep -q "$metric_name"; then
        echo -e "${GREEN}✓${NC} $service_name metrics available"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name metrics missing"
        return 1
    fi
}

# Check core monitoring services
echo -e "\n${BLUE}Core Monitoring Services:${NC}"
failed_services=0

check_service "Prometheus" "http://localhost:9090/-/healthy" || ((failed_services++))
check_service "Grafana" "http://localhost:3001/api/health" || ((failed_services++))
check_service "AlertManager" "http://localhost:9093/-/healthy" || ((failed_services++))

# Check exporters
echo -e "\n${BLUE}Monitoring Exporters:${NC}"
check_service "Node Exporter" "http://localhost:9100/metrics" || ((failed_services++))
check_service "cAdvisor" "http://localhost:8080/healthz" || ((failed_services++))
check_service "PostgreSQL Exporter" "http://localhost:9187/metrics" || ((failed_services++))
check_service "Redis Exporter" "http://localhost:9121/metrics" || ((failed_services++))
check_service "Elasticsearch Exporter" "http://localhost:9114/metrics" || ((failed_services++))

# Check application metrics
echo -e "\n${BLUE}Application Metrics:${NC}"
check_metrics "Investment API" "http://localhost:8000/api/metrics" "api_requests_total" || ((failed_services++))
check_metrics "Business Metrics" "http://localhost:8000/api/metrics" "daily_recommendations_generated_total" || ((failed_services++))

# Check Prometheus targets
echo -e "\n${BLUE}Prometheus Targets Status:${NC}"
if curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | select(.health != "up") | "\(.labels.job): \(.health)"' | head -5; then
    unhealthy_targets=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | select(.health != "up")' | wc -l)
    if [ "$unhealthy_targets" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All Prometheus targets are healthy"
    else
        echo -e "${YELLOW}⚠${NC} $unhealthy_targets targets are unhealthy"
        ((failed_services++))
    fi
else
    echo -e "${RED}✗${NC} Could not check Prometheus targets"
    ((failed_services++))
fi

# Check Grafana dashboards
echo -e "\n${BLUE}Grafana Dashboards:${NC}"
dashboard_count=$(curl -s -H "Authorization: Bearer admin:admin" http://localhost:3001/api/search | jq length 2>/dev/null || echo "0")
if [ "$dashboard_count" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} $dashboard_count dashboards available"
else
    echo -e "${RED}✗${NC} No dashboards found"
    ((failed_services++))
fi

# Check recent alerts
echo -e "\n${BLUE}Recent Alerts:${NC}"
recent_alerts=$(curl -s http://localhost:9093/api/v1/alerts | jq '.data | length' 2>/dev/null || echo "0")
if [ "$recent_alerts" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No active alerts"
else
    echo -e "${YELLOW}⚠${NC} $recent_alerts active alerts"
    curl -s http://localhost:9093/api/v1/alerts | jq -r '.data[] | "- \(.labels.alertname): \(.annotations.summary)"' | head -5
fi

# Check disk usage for monitoring data
echo -e "\n${BLUE}Storage Usage:${NC}"
prometheus_size=$(docker exec investment_prometheus du -sh /prometheus 2>/dev/null | cut -f1 || echo "Unknown")
grafana_size=$(docker exec investment_grafana du -sh /var/lib/grafana 2>/dev/null | cut -f1 || echo "Unknown")
echo "Prometheus data: $prometheus_size"
echo "Grafana data: $grafana_size"

# Check cost metrics
echo -e "\n${BLUE}Cost Tracking:${NC}"
if curl -s http://localhost:8000/api/metrics | grep -q "monthly_budget_usage_percent"; then
    budget_usage=$(curl -s http://localhost:8000/api/metrics | grep "monthly_budget_usage_percent" | awk '{print $2}')
    echo "Monthly budget usage: ${budget_usage}%"
    
    if (( $(echo "$budget_usage > 80" | bc -l) )); then
        echo -e "${YELLOW}⚠${NC} High budget usage detected"
    else
        echo -e "${GREEN}✓${NC} Budget usage within limits"
    fi
else
    echo -e "${RED}✗${NC} Budget metrics not available"
    ((failed_services++))
fi

# Summary
echo -e "\n${BLUE}Health Check Summary:${NC}"
echo "========================"
if [ $failed_services -eq 0 ]; then
    echo -e "${GREEN}✓ All monitoring services are healthy${NC}"
    echo "Dashboard URLs:"
    echo "- Grafana: http://localhost:3001"
    echo "- Prometheus: http://localhost:9090" 
    echo "- AlertManager: http://localhost:9093"
    exit 0
else
    echo -e "${RED}✗ $failed_services issues detected${NC}"
    echo "Run: docker-compose logs <service_name> to debug issues"
    exit 1
fi
