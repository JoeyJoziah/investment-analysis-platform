#!/bin/bash

# Quick status check for Airflow deployment

echo "======================================"
echo "  Airflow Deployment Status Check"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
echo "1. Checking Docker status..."
if docker info >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker is running${NC}"
else
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi
echo ""

# Check Airflow containers
echo "2. Checking Airflow containers..."
containers=(
    "airflow-webserver"
    "airflow-scheduler"
    "airflow-worker-api"
    "airflow-worker-compute"
    "airflow-worker-default"
    "investment_db_airflow"
    "investment_redis_airflow"
)

running_count=0
for container in "${containers[@]}"; do
    if docker ps --format "table {{.Names}}" | grep -q "$container"; then
        echo -e "${GREEN}✓ $container is running${NC}"
        ((running_count++))
    else
        echo -e "${YELLOW}⚠ $container is not running${NC}"
    fi
done
echo ""

# Check service health
echo "3. Checking service health..."

# Check Airflow webserver
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200"; then
    echo -e "${GREEN}✓ Airflow webserver is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Airflow webserver is not responding (may still be starting)${NC}"
fi

# Check Flower
if curl -s -o /dev/null -w "%{http_code}" http://localhost:5555 | grep -q "200"; then
    echo -e "${GREEN}✓ Flower (Celery monitor) is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Flower is not responding${NC}"
fi

# Check metrics endpoint
if curl -s -o /dev/null -w "%{http_code}" http://localhost:9102/metrics | grep -q "200"; then
    echo -e "${GREEN}✓ Prometheus metrics are available${NC}"
else
    echo -e "${YELLOW}⚠ Metrics endpoint is not responding${NC}"
fi
echo ""

# Check DAGs
echo "4. Checking DAGs..."
if docker exec airflow-webserver airflow dags list 2>/dev/null | grep -q "daily_market_analysis"; then
    echo -e "${GREEN}✓ Main DAG 'daily_market_analysis' is loaded${NC}"
    
    # Check if DAG is paused
    if docker exec airflow-webserver airflow dags state daily_market_analysis 2>/dev/null | grep -q "paused"; then
        echo -e "${YELLOW}  ⚠ DAG is paused (run: docker exec airflow-webserver airflow dags unpause daily_market_analysis)${NC}"
    else
        echo -e "${GREEN}  ✓ DAG is active${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Main DAG not found (may still be loading)${NC}"
fi
echo ""

# Check pools
echo "5. Checking resource pools..."
pools_output=$(docker exec airflow-webserver airflow pools list 2>/dev/null)
if echo "$pools_output" | grep -q "api_calls"; then
    echo -e "${GREEN}✓ Resource pools are configured${NC}"
    echo "$pools_output" | grep -E "api_calls|compute_intensive|database_tasks" | head -3
else
    echo -e "${YELLOW}⚠ Resource pools not configured yet${NC}"
fi
echo ""

# Summary
echo "======================================"
echo "  Summary"
echo "======================================"
if [ $running_count -ge 5 ]; then
    echo -e "${GREEN}✓ Airflow is deployed and running!${NC}"
    echo ""
    echo "Access points:"
    echo "  • Airflow UI: http://localhost:8080 (admin/admin123)"
    echo "  • Flower: http://localhost:5555"
    echo "  • Metrics: http://localhost:9102/metrics"
    echo ""
    echo "Next steps:"
    echo "  1. Access Airflow UI and unpause the DAG"
    echo "  2. Run: python3 scripts/validate_rate_limits.py"
    echo "  3. Run: python3 scripts/test_sample_stocks.py"
else
    echo -e "${YELLOW}⚠ Airflow deployment is incomplete${NC}"
    echo "Run: ./scripts/setup_airflow_complete.sh"
fi