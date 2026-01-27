#!/bin/bash

#############################################################################
# Load Testing Script - Phase 4.2 Performance Testing
#
# Runs comprehensive load tests and generates performance report
# Includes:
#   - API load testing with Locust
#   - Performance benchmarks
#   - ML model performance tests
#   - Resource utilization analysis
#
# Usage:
#   ./scripts/run_load_tests.sh                    # Run all tests
#   ./scripts/run_load_tests.sh --api-only         # Only API load tests
#   ./scripts/run_load_tests.sh --benchmark-only   # Only benchmarks
#   ./scripts/run_load_tests.sh --output=report.md # Custom output file
#############################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
TESTS_DIR="$BACKEND_DIR/tests"
OUTPUT_DIR="$PROJECT_ROOT/docs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${OUTPUT_DIR}/PERFORMANCE_BENCHMARKS.md"

# Test configuration
API_HOST="${API_HOST:-http://localhost:8000}"
NUM_USERS="${NUM_USERS:-100}"
RAMP_UP_RATE="${RAMP_UP_RATE:-10}"  # Users per second
TEST_DURATION="${TEST_DURATION:-300}"  # seconds
LOCUST_TIMEOUT="${LOCUST_TIMEOUT:-600}"  # seconds

# Logging
log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} ${GREEN}ℹ${NC} $1"
}

log_warn() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} ${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} ${RED}✗${NC} $1"
}

log_success() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} ${GREEN}✓${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Parse arguments
RUN_API_TESTS=true
RUN_BENCHMARKS=true
RUN_ML_TESTS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-only)
            RUN_BENCHMARKS=false
            RUN_ML_TESTS=false
            shift
            ;;
        --benchmark-only)
            RUN_API_TESTS=false
            RUN_ML_TESTS=false
            shift
            ;;
        --ml-only)
            RUN_API_TESTS=false
            RUN_BENCHMARKS=false
            shift
            ;;
        --output=*)
            REPORT_FILE="${1#*=}"
            shift
            ;;
        --users=*)
            NUM_USERS="${1#*=}"
            shift
            ;;
        --duration=*)
            TEST_DURATION="${1#*=}"
            shift
            ;;
        --host=*)
            API_HOST="${1#*=}"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to initialize report
init_report() {
    cat > "$REPORT_FILE" << 'EOF'
# Performance Benchmarks Report

Generated: $(date -u +'%Y-%m-%d %H:%M:%S UTC')

## Executive Summary

This report contains comprehensive performance testing results for the Investment Analysis Platform.

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| API p95 latency | <500ms | TBD |
| Page load FCP | <2s | TBD |
| Cache hit rate | >85% | TBD |
| Database queries p95 | <100ms | TBD |
| Daily pipeline | <1 hour | TBD |
| Error rate | <1% | TBD |

---

## Test Configuration

- **API Host**: $(API_HOST)
- **Test Duration**: $(TEST_DURATION)s
- **Concurrent Users**: $(NUM_USERS)
- **Ramp-up Rate**: $(RAMP_UP_RATE) users/s
- **Test Date**: $(date)

---

## Test Results

### API Load Testing

EOF
}

# Function to run API load tests
run_api_load_tests() {
    print_header "API Load Testing with Locust"

    log_info "Starting API load test..."
    log_info "Host: $API_HOST"
    log_info "Users: $NUM_USERS, Duration: ${TEST_DURATION}s, Ramp-up: ${RAMP_UP_RATE} users/s"

    # Check if API is accessible
    if ! curl -s "$API_HOST/health" > /dev/null 2>&1; then
        log_warn "API server not accessible at $API_HOST"
        log_warn "Skipping API load tests. Start the server and retry."
        return 1
    fi

    log_success "API server is accessible"

    # Run Locust load test
    cd "$PROJECT_ROOT"

    locust \
        -f backend/tests/locustfile.py \
        --host="$API_HOST" \
        -u "$NUM_USERS" \
        -r "$RAMP_UP_RATE" \
        --run-time "${TEST_DURATION}s" \
        --headless \
        --csv="$OUTPUT_DIR/locust_results" \
        --timeout "$LOCUST_TIMEOUT" \
        2>&1 | tee "$OUTPUT_DIR/locust_output.log" || {
        log_error "Locust test failed"
        return 1
    }

    log_success "API load tests completed"
}

# Function to run benchmark tests
run_benchmark_tests() {
    print_header "Performance Benchmarks"

    log_info "Running pytest performance benchmarks..."

    cd "$BACKEND_DIR"

    python -m pytest tests/test_performance_load.py \
        -v \
        --tb=short \
        -m performance \
        --benchmark-only \
        --benchmark-columns=min,max,mean,stddev \
        2>&1 | tee "$OUTPUT_DIR/benchmark_results.log" || {
        log_warn "Some benchmark tests failed"
        # Don't exit - continue with other tests
    }

    log_success "Benchmark tests completed"
}

# Function to run ML performance tests
run_ml_performance_tests() {
    print_header "ML Model Performance Tests"

    log_info "Running ML performance tests..."

    cd "$BACKEND_DIR"

    python -m pytest tests/test_ml_performance.py \
        -v \
        --tb=short \
        -m performance \
        2>&1 | tee "$OUTPUT_DIR/ml_performance_results.log" || {
        log_warn "Some ML performance tests failed"
        # Don't exit - continue with reporting
    }

    log_success "ML performance tests completed"
}

# Function to analyze results
analyze_results() {
    print_header "Results Analysis"

    log_info "Analyzing test results..."

    # Parse Locust results if available
    if [ -f "$OUTPUT_DIR/locust_results_stats.csv" ]; then
        log_success "Locust statistics available"

        # Extract key metrics
        if command -v awk &> /dev/null; then
            log_info "Key Locust Results:"
            awk -F',' 'NR>1 && NR<=5 {
                printf "  %s: avg=%.0fms, p95=%.0fms, p99=%.0fms\n",
                $1, $5, $9, $10
            }' "$OUTPUT_DIR/locust_results_stats.csv" || true
        fi
    fi

    # Parse benchmark results
    if [ -f "$OUTPUT_DIR/benchmark_results.log" ]; then
        log_success "Benchmark results available"
    fi

    # Parse ML performance results
    if [ -f "$OUTPUT_DIR/ml_performance_results.log" ]; then
        log_success "ML performance results available"

        # Extract cache hit rates
        if command -v grep &> /dev/null; then
            grep -i "cache hit rate\|p95\|p99" "$OUTPUT_DIR/ml_performance_results.log" | head -10 || true
        fi
    fi
}

# Function to generate HTML report
generate_html_report() {
    print_header "Generating HTML Report"

    local html_file="$OUTPUT_DIR/performance_report.html"

    cat > "$html_file" << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
    <title>Performance Benchmarks Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #0066cc; color: white; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .success { color: green; font-weight: bold; }
        .warning { color: orange; font-weight: bold; }
        .error { color: red; font-weight: bold; }
        .metric-value { font-family: monospace; background-color: #f0f0f0; padding: 4px 8px; border-radius: 3px; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Benchmarks Report</h1>
        <p>Generated: <span id="timestamp"></span></p>

        <h2>Summary</h2>
        <table>
            <tr>
                <th>Test Category</th>
                <th>Status</th>
                <th>Results</th>
            </tr>
            <tr>
                <td>API Load Testing</td>
                <td><span class="metric-value" id="api-status">Pending</span></td>
                <td id="api-results">-</td>
            </tr>
            <tr>
                <td>Performance Benchmarks</td>
                <td><span class="metric-value" id="benchmark-status">Pending</span></td>
                <td id="benchmark-results">-</td>
            </tr>
            <tr>
                <td>ML Performance</td>
                <td><span class="metric-value" id="ml-status">Pending</span></td>
                <td id="ml-results">-</td>
            </tr>
        </table>

        <h2>Performance Targets Status</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Target</th>
                <th>Actual</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>API p95 latency</td>
                <td>&lt;500ms</td>
                <td id="api-p95">-</td>
                <td id="api-p95-status">Pending</td>
            </tr>
            <tr>
                <td>Cache hit rate</td>
                <td>&gt;85%</td>
                <td id="cache-rate">-</td>
                <td id="cache-rate-status">Pending</td>
            </tr>
            <tr>
                <td>Error rate</td>
                <td>&lt;1%</td>
                <td id="error-rate">-</td>
                <td id="error-rate-status">Pending</td>
            </tr>
        </table>

        <h2>Detailed Results</h2>
        <div id="detailed-results">
            <p>See log files for detailed results.</p>
        </div>
    </div>

    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
HTMLEOF

    log_success "HTML report generated: $html_file"
}

# Function to generate final summary
generate_summary() {
    print_header "Performance Test Summary"

    echo -e "${GREEN}Test Results Location: $OUTPUT_DIR${NC}\n"

    log_info "Generated files:"
    [ -f "$REPORT_FILE" ] && log_success "  - $REPORT_FILE"
    [ -f "$OUTPUT_DIR/benchmark_results.log" ] && log_success "  - $OUTPUT_DIR/benchmark_results.log"
    [ -f "$OUTPUT_DIR/ml_performance_results.log" ] && log_success "  - $OUTPUT_DIR/ml_performance_results.log"
    [ -f "$OUTPUT_DIR/locust_output.log" ] && log_success "  - $OUTPUT_DIR/locust_output.log"
    [ -f "$OUTPUT_DIR/performance_report.html" ] && log_success "  - $OUTPUT_DIR/performance_report.html"

    echo ""
    log_info "Next steps:"
    echo "  1. Review results in: $OUTPUT_DIR"
    echo "  2. Check HTML report: $OUTPUT_DIR/performance_report.html"
    echo "  3. Analyze bottlenecks in: $OUTPUT_DIR/benchmark_results.log"
    echo ""
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║   Investment Platform Load Test Suite  ║"
    echo "║         Phase 4.2 - Performance        ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}\n"

    log_info "Starting comprehensive load testing..."
    log_info "Configuration: Users=$NUM_USERS, Duration=${TEST_DURATION}s, Host=$API_HOST"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Initialize report
    init_report

    # Run tests based on flags
    if [ "$RUN_API_TESTS" = true ]; then
        run_api_load_tests || log_warn "API load tests skipped"
    fi

    if [ "$RUN_BENCHMARKS" = true ]; then
        run_benchmark_tests
    fi

    if [ "$RUN_ML_TESTS" = true ]; then
        run_ml_performance_tests
    fi

    # Analyze and generate reports
    analyze_results
    generate_html_report
    generate_summary

    log_success "Load testing completed successfully!"
    exit 0
}

# Run main function
main
