# Performance Benchmarks - Phase 4.2

## Executive Summary

This document outlines the performance testing strategy, benchmarks, and results for the Investment Analysis Platform. Phase 4.2 implements comprehensive load testing to validate the platform meets production performance requirements.

**Last Updated**: 2026-01-27
**Test Status**: Ready for Execution
**Performance Framework**: Locust + pytest + Custom Profilers

---

## Performance Targets

All performance targets have been established based on industry standards and user experience requirements:

| Metric | Target | Current | Status | Notes |
|--------|--------|---------|--------|-------|
| **API p95 latency** | <500ms | TBD | ⏳ | Critical for responsive UI |
| **API p99 latency** | <1000ms | TBD | ⏳ | Acceptable for background operations |
| **Page load FCP** | <2s | TBD | ⏳ | First Contentful Paint |
| **Page load TTI** | <3s | TBD | ⏳ | Time to Interactive |
| **Cache hit rate** | >85% | TBD | ⏳ | Reduces API calls |
| **Database query p95** | <100ms | TBD | ⏳ | Optimized indexes required |
| **Daily pipeline** | <1 hour | TBD | ⏳ | Full run of 6000 stocks |
| **Error rate** | <1% | TBD | ⏳ | Reliability target |
| **Concurrent users** | 100+ | TBD | ⏳ | Platform scalability |
| **Throughput** | 100+ req/s | TBD | ⏳ | Sustained load capacity |

---

## Test Suite Architecture

### 1. Locust Load Testing (`backend/tests/locustfile.py`)

Real-world user behavior simulation with 100 concurrent users.

#### User Behavior Model

```
┌─────────────────────────────────────────┐
│      InvestmentAnalysisUser (100)       │
├─────────────────────────────────────────┤
│                                         │
├── DashboardTasks (1 weight)             │
│   ├── view_dashboard()                  │
│   ├── get_portfolio_summary()           │
│   └── get_market_overview()             │
│                                         │
├── PortfolioTasks (2 weight)             │
│   ├── list_holdings()                   │
│   ├── get_holding_details()             │
│   └── get_performance()                 │
│                                         │
├── RecommendationTasks (3 weight)        │
│   ├── get_recommendations()             │
│   ├── get_recommendation_details()      │
│   └── get_ai_analysis()                 │
│                                         │
├── SearchTasks (1 weight)                │
│   ├── search_stocks()                   │
│   └── autocomplete_search()             │
│                                         │
└── AnalyticsTasks (1 weight)             │
    ├── get_stock_metrics()               │
    ├── get_price_history()               │
    └── get_correlation_analysis()        │
```

#### Task Weights and Distribution

- **Dashboard Tasks** (10%): Foundation for user engagement
- **Portfolio Tasks** (20%): Core functionality
- **Recommendation Tasks** (30%): Heavy API load
- **Search Tasks** (10%): Query performance
- **Analytics Tasks** (10%): Historical data access

Wait time between requests: 1-3 seconds (realistic user think-time)

#### Metrics Collected

- Response times (min, max, mean, p50, p95, p99)
- Success/failure rates
- Cache hit rates
- Concurrent connection peaks
- Throughput (requests/second)

#### Running Locust Tests

```bash
# Start the API server first
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# In another terminal, run Locust
locust -f backend/tests/locustfile.py \
    --host=http://localhost:8000 \
    -u 100 \
    -r 10 \
    --run-time 300s \
    --headless
```

**Expected Output**:
```
Type       | Name | # requests | # failures | Median [ms] | 95%ile [ms] | 99%ile [ms]
-----------|------|------------|------------|-------------|-------------|------------
GET        | /api/dashboard | 1500 | 0 | 120 | 450 | 800
GET        | /api/portfolio/summary | 1200 | 0 | 100 | 400 | 750
GET        | /api/recommendations | 2000 | 5 | 150 | 500 | 950
...
```

---

### 2. Performance Load Tests (`backend/tests/test_performance_load.py`)

Expanded pytest-based benchmarks covering critical operations.

#### Test Classes

##### A. TestRecommendationsUnderLoad

**Purpose**: Validate recommendation engine performance with 100 concurrent users

**Test**: `test_recommendations_under_load`
- **Duration**: 60 seconds
- **Concurrency**: 100 users
- **Requests per user**: 10
- **Total requests**: 1,000
- **Target metrics**:
  - Error rate: <5%
  - Average latency: <500ms
  - Throughput: >10 req/s
  - Peak memory: <2GB

**Simulates**:
- Concurrent recommendation requests
- API call tracking
- Error recovery
- Memory management under sustained load

##### B. TestBulkPriceQueryPerformance

**Purpose**: Test querying price data for 6,000 stocks

**Test**: `test_bulk_price_query_performance`
- **Stocks**: 6,000
- **Batch size**: 100
- **Target metrics**:
  - Throughput: >20 queries/s
  - Total time: <300s (5 minutes for 6,000)
  - Memory growth: Linear, no leaks
  - Error rate: <5%

**Simulates**:
- Large-scale data retrieval
- Batch processing efficiency
- Rate limiting compliance
- Memory efficiency at scale

##### C. TestMLRecommendationGeneration

**Purpose**: Test ML model inference performance for 100 stocks

**Test**: `test_ml_recommendation_generation`
- **Stocks**: 100
- **Features**: 50 per stock
- **Target metrics**:
  - Average inference: <100ms
  - P95 inference: <200ms
  - Peak memory: <500MB
  - Success rate: 100%

**Simulates**:
- Neural network inference
- Feature generation
- Model loading and caching
- Prediction confidence scoring

##### D. TestDailyPipelinePerformance

**Purpose**: Test complete daily pipeline execution

**Test**: `test_daily_pipeline_full_run`

**Pipeline Stages**:

1. **Data Fetch** (1,000 stocks)
   - Market data retrieval
   - Price history download
   - Target: <5 minutes

2. **Recommendation Generation**
   - Generate recommendations for all stocks
   - Score filtering and ranking
   - Target: <3 minutes

3. **ML Model Inference**
   - Run trained models
   - Generate confidence scores
   - Target: <5 minutes

4. **Database Updates**
   - Batch insert/update operations
   - Index maintenance
   - Target: <5 minutes

5. **Cache Updates**
   - Populate cache with results
   - Invalidate old entries
   - Target: <2 minutes

**Target**: Total execution <1 hour for 6,000 stocks

---

### 3. ML Performance Tests (`backend/tests/test_ml_performance.py`)

Comprehensive ML model performance profiling and benchmarking.

#### Test Classes

##### A. TestMLModelInference

**test_single_model_inference_latency**
- Measures single-model inference performance
- 1,000 inferences with 50D features
- Tracks cache behavior
- Validates p95 <200ms

**test_batch_inference_performance**
- Tests different batch sizes: [1, 10, 32, 64, 128]
- Finds optimal batch size for throughput
- Measures memory efficiency
- Targets >100 samples/s

**test_concurrent_model_inference**
- 5 models running concurrently
- 100 concurrent requests
- Validates fair resource allocation
- Targets >50 inferences/s overall

##### B. TestMLMemoryProfiling

**test_model_memory_efficiency**
- Monitors memory during inference
- Tracks memory growth patterns
- Validates <100KB per inference
- Detects memory leaks

**test_memory_leak_detection**
- 60-second sustained load test
- Samples memory every 5,000 inferences
- Calculates memory growth trend
- Asserts no significant leaks

##### C. TestCacheHitRate

**test_inference_cache_hit_rate**
- Simulates realistic request patterns (Zipf distribution)
- 80% requests to 20% of features
- Validates >85% cache hit rate
- Measures cache effectiveness

**test_distributed_cache_consistency**
- Multiple cache stores (3 workers)
- Random worker routing
- Validates consistency
- Ensures minimum hit rate

##### D. TestInferenceLatencyDistribution

**test_latency_percentiles**
- 1,000 inferences with latency tracking
- Calculates full percentile distribution
- Validates p95 <100ms, p99 <150ms
- Identifies outliers

---

## Performance Profilers

### PerformanceMonitor (`test_performance_load.py`)

```python
class PerformanceMonitor:
    """Monitors system performance during tests"""

    def __init__(self):
        self.start_time = None
        self.initial_memory = None
        self.peak_memory = 0
        self.cpu_samples = []
        self.api_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self._monitor_thread = None

    def start_monitoring(self):
        """Start background monitoring thread"""
        # Samples CPU and memory every 100ms

    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop and return metrics"""
        # Returns comprehensive metrics
```

**Tracks**:
- CPU utilization (mean, samples)
- Memory usage (initial, current, peak)
- API calls count
- Cache hits/misses and hit rate
- Errors and error rate
- Throughput (items/second)

### MLInferenceProfiler (`test_ml_performance.py`)

```python
class MLInferenceProfiler:
    """Profile ML model inference"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.inference_times = []
        self.memory_snapshots = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

    def get_metrics(self) -> MLPerformanceMetrics:
        """Return comprehensive ML metrics"""
        # Calculates percentiles, throughput, etc.
```

**Tracks**:
- Individual inference times (for percentiles)
- Memory before/after inference
- Cache hit/miss statistics
- Error conditions and rates
- ML-specific metrics (throughput, latency distribution)

---

## Running Performance Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio locust memory-profiler psutil

# Install project dependencies
pip install -r requirements.txt

# Setup database and cache
docker-compose up -d postgres redis
```

### Running All Tests

```bash
# Execute the main load test script
./scripts/run_load_tests.sh

# Optional parameters
./scripts/run_load_tests.sh --users=50 --duration=600 --host=http://localhost:8000
```

### Running Specific Tests

```bash
# Only API load tests
./scripts/run_load_tests.sh --api-only

# Only performance benchmarks
./scripts/run_load_tests.sh --benchmark-only

# Only ML tests
./scripts/run_load_tests.sh --ml-only

# Run pytest directly
cd backend
python -m pytest tests/test_performance_load.py -v -m performance
python -m pytest tests/test_ml_performance.py -v -m performance

# Run with profiling
python -m pytest tests/test_performance_load.py::TestLargeScaleProcessing::test_6000_stock_analysis_performance -v --profile
```

### Test Configuration

```bash
# Environment variables
export API_HOST=http://localhost:8000
export NUM_USERS=100
export RAMP_UP_RATE=10  # users per second
export TEST_DURATION=300  # seconds

# Then run tests
./scripts/run_load_tests.sh
```

---

## Results Interpretation

### Key Metrics Explained

#### Response Time Percentiles

- **Median (p50)**: 50% of requests are faster than this
  - Target: <200ms
  - Indicates typical user experience

- **p95 (95th percentile)**: 95% of requests are faster than this
  - Target: <500ms
  - Critical for user satisfaction
  - 1 in 20 requests slower than this

- **p99 (99th percentile)**: 99% of requests are faster than this
  - Target: <1000ms
  - Edge cases and slow scenarios
  - 1 in 100 requests slower than this

#### Cache Hit Rate

- **Calculation**: hits / (hits + misses)
- **Target**: >85%
- **Impact**:
  - Each cache hit saves ~200ms (API call time)
  - 85% hit rate = 20% of requests are full API calls
  - Effective at reducing load

#### Error Rate

- **Calculation**: failed_requests / total_requests
- **Target**: <1%
- **Failure types**:
  - Timeout (>5 seconds)
  - 5xx server error
  - Connection refused
  - Invalid response

#### Throughput

- **Calculation**: requests / duration_seconds
- **Target**: >100 req/s sustained
- **Indicates**: Overall platform capacity

---

## Performance Optimization Strategies

### 1. Caching Optimization

**Current approach**: Redis cache with 1-hour TTL

**Optimization opportunities**:
- Implement cache warming on startup
- Use cache invalidation strategies for updates
- Separate cache layers (L1 in-process, L2 Redis)
- Monitor and tune TTL values

**Target cache hit rate**: 85%

### 2. Database Query Optimization

**Current bottleneck**: N+1 queries during bulk operations

**Optimization strategies**:
- Add database indexes on frequently queried columns
- Implement query batching
- Use pagination for large result sets
- Consider read replicas for heavy queries

**Target**: p95 <100ms for database queries

### 3. API Response Optimization

**Compression**: gzip responses (target 60% reduction)

**Pagination**: Return limited results with continuation tokens

**Selective fields**: Allow clients to request only needed fields

**Target p95 latency**: <500ms

### 4. ML Model Performance

**Current focus**: Inference optimization

**Strategies**:
- Model quantization for faster inference
- Batch prediction for throughput
- Cache predictions for repeated inputs
- Use GPU acceleration if available

**Target**: <100ms per inference

### 5. Resource Management

**Memory optimization**:
- Implement object pooling for frequent allocations
- Monitor and limit batch sizes
- Regular garbage collection between batches
- Stream large result sets instead of buffering

**CPU optimization**:
- Profile hot paths
- Parallelize independent operations
- Use async/await for I/O operations
- Consider thread pool tuning

---

## Continuous Monitoring

### Metrics to Track Post-Deployment

1. **API Response Times**
   - Daily p95, p99 trends
   - Endpoint-specific baselines
   - Anomaly detection

2. **Error Rates**
   - 4xx vs 5xx breakdown
   - Error type distribution
   - Root cause analysis

3. **Resource Utilization**
   - CPU usage trends
   - Memory growth patterns
   - Database connection pool health

4. **Business Metrics**
   - User concurrent sessions
   - Feature usage patterns
   - Peak load times

5. **Cache Performance**
   - Hit rates by endpoint
   - TTL effectiveness
   - Invalidation patterns

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| API p95 latency | >400ms | >600ms |
| Error rate | >0.5% | >2% |
| Cache hit rate | <75% | <60% |
| Memory usage | >75% | >90% |
| CPU usage | >80% | >95% |

---

## Benchmark Results Template

### Test Execution Date: [DATE]

#### API Load Test Results (100 concurrent users, 300s)

```
Requests Summary:
- Total Requests: 15,000
- Successful: 14,850
- Failed: 150
- Success Rate: 99%

Response Times:
- Median: 180ms
- P95: 480ms
- P99: 920ms

Throughput: 50 requests/second

Cache Performance:
- Hit Rate: 87%
- Cache Hits: 13,005
- Cache Misses: 1,995
```

#### Performance Benchmarks

```
Test: test_6000_stock_analysis_performance
- Duration: 892 seconds
- Throughput: 6.7 stocks/second
- Memory: 1,840 MB peak
- Error Rate: 2.1%

Test: test_bulk_price_query_performance
- 6,000 stocks processed
- Duration: 287 seconds
- Throughput: 20.9 queries/second
- Error Rate: 0.5%
```

#### ML Performance Results

```
Single Model Inference (1,000 samples):
- Average Latency: 8.2ms
- P95 Latency: 18.5ms
- P99 Latency: 32.1ms
- Throughput: 121.9 inferences/second
- Cache Hit Rate: 86.2%
```

---

## Troubleshooting Performance Issues

### High Response Latency

**Diagnosis**:
```bash
# Check API logs for slow endpoints
grep "duration_ms.*[0-9]{4}" api.log | sort -t= -k2 -n

# Monitor database queries
EXPLAIN ANALYZE <query>

# Check cache hit rates
redis-cli INFO stats
```

**Solutions**:
- Add database indexes
- Optimize N+1 queries
- Increase cache TTL
- Implement query result caching

### High Memory Usage

**Diagnosis**:
```bash
# Profile memory usage
python -m memory_profiler test_file.py

# Check for leaks
python -m tracemalloc
```

**Solutions**:
- Reduce batch sizes
- Implement streaming for large datasets
- Add garbage collection points
- Monitor object lifecycle

### Low Cache Hit Rate

**Diagnosis**:
```bash
# Check cache key patterns
redis-cli KEYS '*' | head -20

# Monitor evictions
redis-cli INFO eviction
```

**Solutions**:
- Review TTL values
- Implement cache warming
- Monitor access patterns
- Use smarter cache invalidation

---

## Performance Roadmap

### Phase 4.3: Optimization (Next)

- Implement identified bottlenecks
- Deploy performance improvements
- Rerun benchmarks to validate
- Document optimization impact

### Phase 4.4: Production Monitoring (Later)

- Setup APM (Application Performance Monitoring)
- Configure alerts and dashboards
- Implement continuous profiling
- Establish performance SLOs

---

## References

### Tools Used

- **Locust**: Open-source load testing tool
  - Documentation: https://docs.locust.io/
  - GitHub: https://github.com/locustio/locust

- **pytest**: Python testing framework
  - Documentation: https://docs.pytest.org/

- **psutil**: System and process utilities
  - Documentation: https://psutil.readthedocs.io/

- **tracemalloc**: Memory profiling
  - Documentation: https://docs.python.org/3/library/tracemalloc.html

### Related Documentation

- API Design: `/docs/API_DESIGN.md`
- Architecture: `/docs/ARCHITECTURE.md`
- Database Optimization: `/docs/DATABASE.md`
- ML Pipeline: `/docs/ML_PIPELINE.md`

---

## Appendix: Test Configuration Files

### Locust Configuration

```python
# backend/tests/locustfile.py
class InvestmentAnalysisUser(HttpUser):
    wait_time = between(1, 3)  # seconds

    tasks = {
        DashboardTasks: 1,
        PortfolioTasks: 2,
        RecommendationTasks: 3,
        SearchTasks: 1,
        AnalyticsTasks: 1
    }
```

### Pytest Configuration

```ini
# backend/pytest.ini
[pytest]
markers =
    performance: performance and load tests
    asyncio_mode = auto
```

---

## Contact & Support

For performance-related questions or issues:

1. Check this documentation
2. Review test logs in `/docs/`
3. Run diagnostic benchmarks
4. Check GitHub issues
5. Contact the development team

---

**Status**: Ready for Phase 4.2 Execution
**Last Updated**: 2026-01-27
**Next Review**: After Phase 4.2 Completion
