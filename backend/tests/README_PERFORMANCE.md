# Performance Testing Guide

Quick reference for running Phase 4.2 performance tests.

## Quick Start

### 1. Start the API Server

```bash
# Terminal 1: Start API
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run All Tests

```bash
# Terminal 2: Run comprehensive load tests
./scripts/run_load_tests.sh
```

## Individual Test Commands

### API Load Testing (Locust)

```bash
# Basic load test: 100 users, 300 seconds
locust -f backend/tests/locustfile.py --host=http://localhost:8000

# Advanced: CLI mode with specific parameters
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 100 \
    -r 10 \
    --run-time 300s \
    --headless \
    --csv=results/locust

# With custom timeout
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 100 \
    --timeout 60
```

### Performance Benchmarks

```bash
# All performance tests
cd backend
python -m pytest tests/test_performance_load.py -v -m performance

# Specific test class
python -m pytest tests/test_performance_load.py::TestLargeScaleProcessing -v

# Specific test
python -m pytest tests/test_performance_load.py::TestLargeScaleProcessing::test_6000_stock_analysis_performance -v

# With output capture (see print statements)
python -m pytest tests/test_performance_load.py -v -s -m performance
```

### ML Performance Tests

```bash
# All ML tests
cd backend
python -m pytest tests/test_ml_performance.py -v -m performance

# Specific class
python -m pytest tests/test_ml_performance.py::TestMLModelInference -v

# With verbose output
python -m pytest tests/test_ml_performance.py -v -s
```

### Specific Performance Scenarios

```bash
# Test 6000 stock processing
python -m pytest tests/test_performance_load.py::TestLargeScaleProcessing::test_6000_stock_analysis_performance -v -s

# Test recommendations under load
python -m pytest tests/test_performance_load.py::TestRecommendationsUnderLoad -v -s

# Test ML inference latency
python -m pytest tests/test_ml_performance.py::TestMLModelInference::test_single_model_inference_latency -v -s

# Test cache hit rates
python -m pytest tests/test_ml_performance.py::TestCacheHitRate -v -s
```

## Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| API p95 latency | <500ms | Locust stats |
| Cache hit rate | >85% | Cache metrics |
| Bulk query (6000 stocks) | <5min | Duration timing |
| ML inference | <100ms | Latency tracking |
| Error rate | <1% | Success/fail count |
| Concurrent users | 100 | Locust config |
| Memory peak | <2GB | psutil monitoring |

## Test Configuration

### Environment Variables

```bash
# API host
export API_HOST=http://localhost:8000

# Locust parameters
export NUM_USERS=100
export RAMP_UP_RATE=10  # users per second
export TEST_DURATION=300  # seconds

# Custom output
export OUTPUT_DIR=./docs
```

### pytest Configuration

File: `backend/pytest.ini`
```ini
[pytest]
markers =
    performance: mark test as performance test
asyncio_mode = auto
```

## Results Interpretation

### Locust Output Format

```
Type     | Name          | # requests | # failures | Median | 95%  | 99%
---------|---------------|------------|------------|--------|------|-----
GET      | /api/dashboard| 1500       | 0          | 120    | 450  | 800
GET      | /api/portfolio| 1200       | 0          | 100    | 400  | 750
POST     | /api/search   | 800        | 5          | 250    | 650  | 1200
```

**Key Metrics**:
- **Median**: Typical response time (50th percentile)
- **95%**: 95% of requests are faster than this
- **99%**: 99% of requests are faster than this
- **# failures**: Timeouts, errors, exceptions

### pytest Output Format

```
test_performance_load.py::TestLargeScaleProcessing::test_6000_stock_analysis_performance PASSED

Performance Metrics:
  Total time: 892.34s
  Avg time per stock: 0.137s
  Peak memory: 1840.5MB
  Throughput: 6.73 stocks/s
  Error rate: 0.021
  API calls: 18432
```

## Common Issues & Solutions

### Issue: "Connection refused" on localhost:8000

**Solution**: Start the API server first
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Issue: Tests fail with "psutil not found"

**Solution**: Install test dependencies
```bash
pip install psutil pytest pytest-asyncio memory-profiler
```

### Issue: Locust not found

**Solution**: Install Locust
```bash
pip install locust
```

### Issue: Tests timeout

**Solution**: Increase timeout or reduce load
```bash
# Increase test timeout in pytest.ini
timeout = 600

# Or run with fewer users
locust -u 50 -r 5
```

### Issue: Out of memory during tests

**Solution**: Reduce batch sizes or concurrent users
```bash
# Edit test parameters in the test file
batch_size = 100  # reduce from 500
concurrent_users = 50  # reduce from 100
```

## Advanced Usage

### Profile with cProfile

```bash
python -m cProfile -s cumtime -m pytest tests/test_performance_load.py::TestLargeScaleProcessing::test_6000_stock_analysis_performance
```

### Memory profiling with memory_profiler

```bash
python -m memory_profiler backend/tests/test_performance_load.py
```

### Stress test with high concurrency

```bash
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 500 \
    -r 50 \
    --run-time 600s
```

### Generate detailed HTML report (Locust)

```bash
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 100 \
    --headless \
    --html=reports/load_test.html
```

## Results Location

Test results are saved in:
- `/docs/PERFORMANCE_BENCHMARKS.md` - Main report
- `/docs/benchmark_results.log` - pytest output
- `/docs/ml_performance_results.log` - ML test output
- `/docs/locust_output.log` - Locust raw output
- `/docs/locust_results_stats.csv` - Locust statistics

## Next Steps

1. **Review Results**: Check `docs/PERFORMANCE_BENCHMARKS.md`
2. **Identify Bottlenecks**: Look at slowest endpoints/operations
3. **Optimize**: Implement improvements from analysis
4. **Retest**: Run benchmarks again to validate fixes
5. **Monitor**: Set up continuous performance monitoring

## Performance Optimization Checklist

After running tests, address bottlenecks:

- [ ] Database queries optimized (add indexes, batch operations)
- [ ] Cache hit rate >85% (review TTL, warming strategy)
- [ ] API latency p95 <500ms (optimize endpoint logic)
- [ ] ML inference <100ms (model optimization, batching)
- [ ] Memory usage reasonable (no leaks, batching)
- [ ] Error rate <1% (handle failures gracefully)
- [ ] Concurrent users >100 (no resource exhaustion)

## References

- Full documentation: `/docs/PERFORMANCE_BENCHMARKS.md`
- Locust docs: https://docs.locust.io/
- pytest docs: https://docs.pytest.org/
- Python profiling: https://docs.python.org/3/library/profile.html

---

**Last Updated**: 2026-01-27
**Phase**: 4.2 - Performance Load Testing
