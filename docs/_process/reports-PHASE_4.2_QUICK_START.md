# Phase 4.2 - Quick Start Guide

**Phase**: Performance Load Testing (Phase 4.2)
**Status**: ✅ Ready to Execute
**Date**: 2026-01-27

---

## What Was Delivered

✅ **Locust Load Testing** - `backend/tests/locustfile.py`
- 100 concurrent user simulation
- 6 realistic task categories
- Real-world request patterns

✅ **Expanded Performance Tests** - `backend/tests/test_performance_load.py`
- 4 new high-impact test scenarios
- 6,000+ stock processing validation
- Daily pipeline end-to-end testing

✅ **ML Performance Tests** - `backend/tests/test_ml_performance.py`
- Inference latency benchmarking
- Memory profiling and leak detection
- Cache hit rate validation
- 8 comprehensive test methods

✅ **Test Orchestration** - `scripts/run_load_tests.sh`
- Automated all-in-one test runner
- Flexible configuration options
- Comprehensive result reporting

✅ **Complete Documentation** - 28KB
- Main performance benchmarks (18KB)
- Quick reference guide (6KB)
- Implementation details (12KB)
- This quick start guide

---

## Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio locust memory-profiler psutil

# Install project dependencies (if not already)
pip install -r requirements.txt

# Start services
docker-compose up -d postgres redis
```

---

## Run All Tests (Recommended)

```bash
# Make sure script is executable
chmod +x scripts/run_load_tests.sh

# Run all performance tests
./scripts/run_load_tests.sh
```

This will:
1. Start Locust load testing (100 users, 5 minutes)
2. Run pytest performance benchmarks
3. Run ML performance tests
4. Generate HTML report
5. Save results in `/docs/`

**Estimated Duration**: 30-45 minutes

---

## Run Individual Tests

### Option 1: Locust API Load Testing

```bash
# Start API first (in separate terminal)
python -m uvicorn backend.main:app --port 8000

# In another terminal, run Locust
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 100 \
    -r 10 \
    --run-time 300s \
    --headless

# Results appear in terminal and stats.csv
```

**Duration**: 5-10 minutes
**Users**: 100 concurrent
**Requests**: ~15,000 total

### Option 2: Performance Load Benchmarks

```bash
cd backend

# All tests
python -m pytest tests/test_performance_load.py -v -m performance

# Specific test
python -m pytest tests/test_performance_load.py::TestRecommendationsUnderLoad -v -s

# With output
python -m pytest tests/test_performance_load.py::TestDailyPipelinePerformance -v -s
```

**Duration**: 10-20 minutes
**Includes**: 6 existing + 4 new tests

### Option 3: ML Performance Tests

```bash
cd backend

# All ML tests
python -m pytest tests/test_ml_performance.py -v -m performance

# Specific class
python -m pytest tests/test_ml_performance.py::TestMLModelInference -v -s

# Cache validation
python -m pytest tests/test_ml_performance.py::TestCacheHitRate -v -s
```

**Duration**: 5-10 minutes
**Includes**: 8 comprehensive tests

---

## Performance Targets Quick Reference

| Metric | Target | Tested By |
|--------|--------|-----------|
| API p95 latency | <500ms | Locust |
| Cache hit rate | >85% | ML tests |
| Bulk query (6000 stocks) | <5min | Benchmark |
| ML inference | <100ms | ML tests |
| Error rate | <1% | All tests |
| Memory peak | <2GB | Monitoring |
| Concurrent users | 100+ | Locust |
| Daily pipeline | <1 hour | Benchmark |

---

## Where to Find Results

After running tests, check:

```
/docs/
├── PERFORMANCE_BENCHMARKS.md       # Main report
├── benchmark_results.log            # Detailed results
├── ml_performance_results.log       # ML results
├── locust_output.log                # API load test output
├── locust_results_stats.csv         # Locust statistics
└── performance_report.html          # HTML dashboard

/backend/tests/
├── README_PERFORMANCE.md            # Quick reference
└── [test output files]
```

---

## Interpreting Results

### Latency Metrics

- **Median (p50)**: Typical response time
  - Target: <200ms
  - 50% of requests faster than this

- **P95**: 95th percentile
  - Target: <500ms
  - MOST IMPORTANT - user experience
  - Only 1 in 20 requests slower

- **P99**: 99th percentile
  - Target: <1000ms
  - Edge cases
  - 1 in 100 requests slower

### Cache Hit Rate

- **Calculation**: hits / (hits + misses)
- **Target**: >85%
- **Example**: 870 hits out of 1000 = 87% (good!)
- **Impact**: Each miss = ~200ms extra latency

### Throughput

- **Calculation**: requests / duration
- **Target**: >100 req/s sustained
- **Example**: 15,000 requests in 300 seconds = 50 req/s (at 100 concurrent)

### Error Rate

- **Calculation**: failures / total
- **Target**: <1%
- **Example**: 10 failures out of 1000 = 1% (acceptable)

---

## Common Test Configurations

### Light Load Test (Quick Validation)
```bash
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 10 \
    -r 2 \
    --run-time 60s \
    --headless
```
**Duration**: 2-3 minutes

### Standard Load Test (Full Validation)
```bash
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 100 \
    -r 10 \
    --run-time 300s \
    --headless
```
**Duration**: 5-10 minutes

### Heavy Load Test (Stress Testing)
```bash
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 500 \
    -r 50 \
    --run-time 600s \
    --headless
```
**Duration**: 10-15 minutes

---

## Troubleshooting

### "Connection refused" localhost:8000

```bash
# Start API server first
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### "locust not found"

```bash
pip install locust
```

### Tests timeout

Increase timeout in pytest.ini or reduce load:
```bash
# Smaller load
locust -u 50 -r 5 --run-time 300s

# With longer timeout
pytest --timeout=600
```

### Out of memory

```bash
# Reduce concurrent users
locust -u 50 -r 5

# Reduce batch sizes in test code
batch_size = 100  # was 500
```

---

## Next Steps After Testing

1. **Review Results** (30 minutes)
   - Check `/docs/PERFORMANCE_BENCHMARKS.md`
   - Identify any metrics not meeting targets
   - Note slowest endpoints/operations

2. **Analyze Bottlenecks** (1 hour)
   - Use results to identify causes
   - Check database slow query logs
   - Profile hot code paths
   - Review cache efficiency

3. **Optimize** (2-4 hours per issue)
   - Add database indexes
   - Optimize N+1 queries
   - Improve cache strategy
   - Optimize ML inference
   - Reduce memory usage

4. **Retest** (30 minutes)
   - Run benchmarks again
   - Validate improvements
   - Update baselines

5. **Monitor** (Ongoing)
   - Setup APM (Application Performance Monitoring)
   - Configure alerts
   - Track trends

---

## Key Files Overview

### Test Files
- **locustfile.py** (378 lines)
  - Locust load testing configuration
  - 100 concurrent user simulation
  - 6 task categories

- **test_performance_load.py** (1,206 lines)
  - 16 test methods
  - Original 8 + 4 new tests
  - Covers full platform operations

- **test_ml_performance.py** (592 lines)
  - 8 test methods
  - ML inference validation
  - Memory and cache testing

### Execution
- **run_load_tests.sh** (430 lines)
  - Main test orchestration script
  - Automatic report generation
  - Flexible configuration

### Documentation
- **PERFORMANCE_BENCHMARKS.md** (18KB)
  - Comprehensive reference
  - All test details
  - Optimization guide

- **README_PERFORMANCE.md** (6KB)
  - Quick start
  - Common commands
  - Troubleshooting

---

## Important Metrics to Track

### After Each Test Run

Record these values:

1. **Locust Results**
   - Total requests
   - Failed requests
   - Median response time
   - P95 response time
   - P99 response time
   - Requests per second

2. **Benchmark Results**
   - 6000 stock processing time
   - Memory usage peak
   - Error rate
   - Throughput (stocks/second)

3. **ML Test Results**
   - Average inference latency
   - P95 inference latency
   - Cache hit rate
   - Memory per inference

4. **System Metrics**
   - Peak CPU usage
   - Peak memory usage
   - Sustained memory (no leaks)

### Create Spreadsheet for Trends

```
Date      | API P95 | Cache Hit | ML Latency | Notes
----------|---------|-----------|------------|----------
2026-01-27|  450ms  |   87%     |   8.2ms   | Baseline
2026-01-28|  420ms  |   89%     |   7.5ms   | After opt1
```

---

## Performance Optimization Roadmap

### Phase 4.3: Optimization (Next)
- Run tests and establish baseline
- Identify top 5 bottlenecks
- Implement fixes
- Retest and validate
- Document improvements

### Phase 4.4: Monitoring (After Optimization)
- Setup APM solution
- Configure alert thresholds
- Create dashboards
- Establish SLOs
- Implement trend tracking

---

## Command Cheat Sheet

```bash
# Full test suite
./scripts/run_load_tests.sh

# Just API tests
./scripts/run_load_tests.sh --api-only

# Just benchmarks
./scripts/run_load_tests.sh --benchmark-only

# Just ML tests
./scripts/run_load_tests.sh --ml-only

# Custom configuration
./scripts/run_load_tests.sh --users=50 --duration=600

# Single test class
pytest backend/tests/test_performance_load.py::TestRecommendationsUnderLoad -v

# Single test method
pytest backend/tests/test_ml_performance.py::TestMLModelInference::test_single_model_inference_latency -v

# With verbose output
pytest backend/tests/test_performance_load.py -v -s

# API server
python -m uvicorn backend.main:app --port 8000

# Locust with UI
locust -f backend/tests/locustfile.py --host=http://localhost:8000
```

---

## Support Resources

- **Main Documentation**: `/docs/PERFORMANCE_BENCHMARKS.md`
- **Quick Reference**: `/backend/tests/README_PERFORMANCE.md`
- **Implementation Details**: `/docs/PHASE_4.2_IMPLEMENTATION.md`
- **This Guide**: `/PHASE_4.2_QUICK_START.md`
- **Completion Summary**: `/PHASE_4.2_COMPLETION_SUMMARY.md`

---

## Summary

**Phase 4.2 is complete and ready to run.**

All performance tests are implemented, documented, and ready for execution. Use this guide to:

1. ✅ Run tests with minimal setup
2. ✅ Interpret results correctly
3. ✅ Identify bottlenecks
4. ✅ Plan optimizations
5. ✅ Track improvements

**To start**: Run `./scripts/run_load_tests.sh`

**Duration**: 30-45 minutes for complete test suite

**Expected Outcome**: Comprehensive performance baseline and bottleneck identification

---

**Ready to Begin?**

```bash
# Make sure services are running
docker-compose up -d postgres redis

# Start API server
python -m uvicorn backend.main:app --port 8000 &

# Execute all tests
./scripts/run_load_tests.sh

# View results
cat docs/PERFORMANCE_BENCHMARKS.md
open docs/performance_report.html
```

---

**Phase 4.2 Status**: ✅ Complete & Ready
**Next Phase**: 4.3 - Performance Optimization
