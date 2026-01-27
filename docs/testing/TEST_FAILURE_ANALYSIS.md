# Test Failure Analysis Framework - Phase 0.3
Prepared for Phase 0.4 Investigation

---

## Overview

This document provides a framework for analyzing and categorizing test failures when the full test suite is executed. It identifies the 7 most likely failure categories and provides debugging strategies for each.

---

## Identified Issues (#1-7)

### Issue #1: ML Model Loading Failures
**Likelihood**: HIGH (10-15% of test failures)
**Severity**: MEDIUM (localized to ML tests)

#### Root Causes to Investigate
1. **Missing Model Files**
   - Path: Check if `/backend/ml_models/` contains required models
   - Files to check: `*.pkl`, `*.h5`, `*.pt`, `*.joblib`
   - Command: `find backend -name "*.pkl" -o -name "*.h5" | head -20`

2. **Incompatible Dependencies**
   - Check: TensorFlow/PyTorch version compatibility
   - Command: `pip show tensorflow pytorch scikit-learn`
   - Issue: Version mismatches between test expectations and installed packages

3. **Model Serialization Issues**
   - Check: Model loading code in conftest.py
   - Issue: Pickle protocol version, binary mode handling
   - Tests affected: `test_ml_performance.py`, `test_recommendation_engine.py`

4. **Memory Constraints**
   - Check: Model size vs available memory
   - Issue: Large models may exceed test environment limits
   - Solution: Use smaller models for testing

#### Debugging Strategy
```bash
# Check which models are missing
python -c "from backend.ml.model_loader import ModelLoader; ml = ModelLoader(); ml.validate_models()"

# Test model loading in isolation
pytest backend/tests/test_ml_performance.py::test_model_initialization -vv

# Check TensorFlow/PyTorch status
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"

# List available models
ls -lah backend/ml_models/
```

#### Expected Failures
- `test_ml_performance.py::test_recommendation_accuracy` - Model file missing
- `test_recommendation_engine.py::test_portfolio_suggestions` - Model not initialized
- `test_financial_model_validation.py::test_ml_feature_engineering` - Features not computed

#### Fix Priority
**Priority**: 1 (Critical - blocks 3-5 test files)

---

### Issue #2: WebSocket Latency & Timing Issues
**Likelihood**: MEDIUM-HIGH (5-10% of test failures)
**Severity**: MEDIUM (flaky, intermittent)

#### Root Causes to Investigate
1. **Subscription Delivery Timing**
   - Check: WebSocket message handling in backend
   - Issue: Subscription ACK may be delayed >2 seconds
   - Tests: `test_websocket_integration.py::test_latency_requirement`

2. **Message Queue Backlog**
   - Check: Redis queue configuration
   - Issue: Message queue overflow during batch updates
   - Solution: Increase batch size or reduce message frequency

3. **Connection Pool Exhaustion**
   - Check: WebSocket pool size in config
   - Issue: Limited concurrent connections
   - Solution: Increase pool size or parallelize tests

4. **Async/Await Ordering**
   - Check: Event loop scheduling
   - Issue: Awaited subscriptions may be processed out of order
   - Solution: Add explicit sync points

#### Debugging Strategy
```bash
# Check WebSocket server status
netstat -an | grep LISTEN | grep 8000

# Monitor WebSocket connections
pytest backend/tests/test_websocket_integration.py::test_connection_establishment -vv -s

# Enable WebSocket debug logging
WEBSOCKET_DEBUG=1 pytest backend/tests/test_websocket_integration.py -vv

# Check message timing
pytest backend/tests/test_websocket_integration.py::test_latency_requirement -vv --durations=10

# Test with longer timeout
pytest backend/tests/test_websocket_integration.py -vv --timeout=30
```

#### Expected Failures
- `test_websocket_integration.py::test_latency_requirement` - Timeout >2s
- `test_websocket_integration.py::test_price_subscription_delivery` - Delayed message
- `test_error_scenarios.py::test_websocket_recovery` - Connection timeout

#### Fix Priority
**Priority**: 2 (High - affects real-time features)

---

### Issue #3: Database Connection Pool Exhaustion
**Likelihood**: MEDIUM (10-15% of test failures)
**Severity**: HIGH (blocks database-dependent tests)

#### Root Causes to Investigate
1. **Connection Pool Saturation**
   - Check: Pool size configuration in settings
   - Command: `grep -r "pool_size" backend/config/`
   - Issue: Pool may be too small for concurrent tests

2. **Connection Leaks**
   - Check: Session cleanup in conftest.py
   - Issue: Sessions not properly closed between tests
   - Solution: Add explicit session.close() in teardown

3. **Transaction Deadlocks**
   - Check: Query ordering and lock management
   - Issue: Concurrent tests may deadlock on resources
   - Solution: Serialize database tests or increase timeout

4. **Fixture Scope Issues**
   - Check: Database fixture scope (session vs function)
   - Issue: Session-scoped engine reused incorrectly
   - Solution: Use function-scoped sessions

#### Debugging Strategy
```bash
# Check current connections
python -c "from sqlalchemy import create_engine; e = create_engine('postgresql://...'); print(e.pool.checkedout())"

# Monitor connection usage
pytest backend/tests/test_performance_load.py -vv --capture=no 2>&1 | grep -i "pool\|connection"

# Enable SQLAlchemy logging
SQLALCHEMY_ECHO=1 pytest backend/tests/test_database_integration.py::test_concurrent_transactions -vv

# Test with serial execution
pytest backend/tests/ -n0 backend/tests/test_database_integration.py

# Check pool exhaustion point
for i in {1..100}; do pytest backend/tests/test_database_integration.py::test_basic_query -q || break; done
```

#### Expected Failures
- `test_performance_load.py::test_concurrent_users` - Connection timeout
- `test_database_integration.py::test_transaction_isolation` - Deadlock timeout
- `test_resilience_integration.py::test_recovery_parallel` - Pool exhausted

#### Fix Priority
**Priority**: 1 (Critical - blocks entire database layer)

---

### Issue #4: Cache Invalidation & State Leakage
**Likelihood**: MEDIUM (10-15% of test failures)
**Severity**: MEDIUM (test isolation issue)

#### Root Causes to Investigate
1. **Cache Not Cleared Between Tests**
   - Check: `cache_decorator.py` cleanup logic
   - Issue: Cache entries persist from previous tests
   - Solution: Add cache.clear() to fixture teardown

2. **Incomplete Mock Cleanup**
   - Check: `conftest.py` cleanup order
   - Issue: Mock patches not properly reset
   - Solution: Use @pytest.fixture autouse

3. **Redis State Persistence**
   - Check: Redis configuration in test mode
   - Issue: Redis container may have stale data
   - Solution: Use Redis FLUSHDB in setup/teardown

4. **Memory Cache Initialization**
   - Check: Singleton cache instance
   - Issue: Shared instance between test classes
   - Solution: Reinitialize cache per test

#### Debugging Strategy
```bash
# Check cache state between tests
pytest backend/tests/test_cache_decorator.py::test_cache_hit -vv -s
pytest backend/tests/test_cache_decorator.py::test_cache_miss -vv -s

# Monitor Redis state
redis-cli DBSIZE
redis-cli FLUSHDB

# Check fixture cleanup
pytest backend/tests/ --setup-show 2>&1 | grep -A2 -B2 "cache"

# Test cache isolation
pytest backend/tests/test_cache_decorator.py -vv --capture=no 2>&1 | grep -i "cache\|state"

# Force cache clear
pytest backend/tests/test_cache_decorator.py --cache-clear -vv
```

#### Expected Failures
- `test_cache_decorator.py::test_cache_hit_rate` - Stale cache entry
- `test_cache_decorator.py::test_cache_expiration` - Entry not expired
- `test_data_quality.py::test_validation_rules` - Cached invalid result

#### Fix Priority
**Priority**: 2 (High - affects test reliability)

---

### Issue #5: Async/Await Race Conditions
**Likelihood**: MEDIUM (5% of test failures)
**Severity**: HIGH (hard to debug, flaky)

#### Root Causes to Investigate
1. **Task Ordering Issues**
   - Check: asyncio.gather() vs asyncio.wait()
   - Issue: Tasks may complete in non-deterministic order
   - Solution: Use explicit ordering with asyncio.Semaphore

2. **Event Loop Shutdown**
   - Check: Event loop cleanup in conftest.py
   - Issue: Tasks may be interrupted mid-execution
   - Solution: Add proper async context managers

3. **Mock Timing Issues**
   - Check: AsyncMock side_effect timing
   - Issue: Mock may return before other tasks complete
   - Solution: Add explicit await/sleep in mocks

4. **Lock Contention**
   - Check: Asyncio locks/semaphores
   - Issue: Deadlock when multiple tasks wait
   - Solution: Use timeouts on lock acquisition

#### Debugging Strategy
```bash
# Enable asyncio debug mode
PYTHONASYNCDEBUG=1 pytest backend/tests/test_error_scenarios.py -vv

# Increase asyncio timeout
pytest backend/tests/test_error_scenarios.py --asyncio-mode=auto --timeout=30 -vv

# Test individual async functions
pytest backend/tests/test_error_scenarios.py::test_concurrent_updates -vv -s

# Check for race conditions with repeated runs
for i in {1..10}; do pytest backend/tests/test_error_scenarios.py::test_concurrent_updates -q || break; done

# Monitor task creation/completion
pytest backend/tests/ -vv -s 2>&1 | grep -i "task\|await\|gather"
```

#### Expected Failures
- `test_error_scenarios.py::test_concurrent_updates` - Race condition
- `test_resilience_integration.py::test_parallel_recovery` - Task timeout
- `test_websocket_integration.py::test_batch_updates` - Order dependency

#### Fix Priority
**Priority**: 3 (Medium - difficult to reproduce)

---

### Issue #6: External API Mock Inconsistency
**Likelihood**: LOW-MEDIUM (5% of test failures)
**Severity**: MEDIUM (breaks integration tests)

#### Root Causes to Investigate
1. **Mock Response Format Mismatch**
   - Check: Mock vs real API response structure
   - Issue: Fields may be in different format
   - Solution: Update mock to match actual API

2. **Incomplete Mock Implementation**
   - Check: All API endpoints covered
   - Issue: Some endpoints not mocked
   - Solution: Add missing mock responses

3. **Timeout Handling Difference**
   - Check: Mock vs real timeout behavior
   - Issue: Mock may not simulate timeouts
   - Solution: Add timeout simulation to mocks

4. **Rate Limit Mock**
   - Check: Mock rate limit headers
   - Issue: Mock may not include Retry-After
   - Solution: Add rate limit response headers

#### Debugging Strategy
```bash
# Compare mock and real responses
python -c "from backend.api.mock_responses import *; print(json.dumps(MOCK_PRICE_RESPONSE, indent=2))"

# Check actual API format
curl https://api.example.com/prices?symbol=AAPL | jq .

# Test mock consistency
pytest backend/tests/test_api_integration.py::test_mock_response_format -vv

# Compare response times
pytest backend/tests/test_api_integration.py -vv --durations=20 2>&1 | grep -i "api\|mock"

# Validate mock completeness
pytest backend/tests/ -k "external" -vv
```

#### Expected Failures
- `test_api_integration.py::test_price_endpoint_response` - Missing field
- `test_data_pipeline_integration.py::test_external_api_call` - Format mismatch
- `test_recommendation_engine.py::test_api_data_ingestion` - Type error

#### Fix Priority
**Priority**: 3 (Medium - isolated to API tests)

---

### Issue #7: Rate Limiter Shared State
**Likelihood**: LOW-MEDIUM (5% of test failures)
**Severity**: MEDIUM (shared state issue)

#### Root Causes to Investigate
1. **Shared Rate Limiter Instance**
   - Check: RateLimiter singleton pattern
   - Issue: State shared between test classes
   - Solution: Reset state per test class

2. **Token Bucket Timing**
   - Check: Token replenishment rate
   - Issue: Timing may be off between test runs
   - Solution: Use fixed time for testing

3. **Priority Queue State**
   - Check: Queue not emptied between tests
   - Issue: Requests from previous tests still in queue
   - Solution: Add queue.clear() to fixture

4. **Distributed Lock Issues**
   - Check: Redis lock handling
   - Issue: Lock may not release properly
   - Solution: Add lock timeout

#### Debugging Strategy
```bash
# Check rate limiter state
python -c "from backend.security.rate_limiter import get_limiter; rl = get_limiter(); print(rl.get_state())"

# Monitor token bucket
pytest backend/tests/test_rate_limiting.py::test_token_replenishment -vv -s

# Check queue state
pytest backend/tests/test_rate_limiting.py::test_priority_queue -vv -s

# Test concurrent rate limiting
pytest backend/tests/test_rate_limiting.py -vv --durations=20

# Reset and retry
pytest backend/tests/test_rate_limiting.py -vv --cache-clear
```

#### Expected Failures
- `test_rate_limiting.py::test_concurrent_request_limiting` - Quota exceeded
- `test_rate_limiting.py::test_priority_queue_ordering` - Order changed
- `test_error_scenarios.py::test_rate_limit_enforcement` - Not limited

#### Fix Priority
**Priority**: 2 (High - affects API protection)

---

## Failure Investigation Checklist

When a test fails, follow this process:

### Step 1: Categorize the Failure
- [ ] Identify which issue category (#1-7) or other
- [ ] Check test file and function name
- [ ] Record error message and stack trace
- [ ] Note whether failure is deterministic or flaky

### Step 2: Run Diagnostics
- [ ] Run test in isolation: `pytest file.py::test_name -vv`
- [ ] Run with increased verbosity: `-vv -s --tb=long`
- [ ] Check for timing issues: `--durations=20`
- [ ] Monitor resource usage: `--capture=no`

### Step 3: Debug Root Cause
- [ ] Check mocks and fixtures
- [ ] Verify dependencies are available
- [ ] Check for shared state/cleanup issues
- [ ] Review async/await handling
- [ ] Check database connections

### Step 4: Document Findings
- [ ] Update TEST_FAILURE_ANALYSIS.md
- [ ] Record root cause and fix applied
- [ ] Note if issue is deterministic or flaky
- [ ] Add to git for team visibility

### Step 5: Verify Fix
- [ ] Run test 3+ times to confirm
- [ ] Run related tests
- [ ] Run full suite to check for regressions

---

## Metrics to Capture

When running the full test suite, capture:

### Execution Metrics
- [ ] Total tests run
- [ ] Tests passed
- [ ] Tests failed (with count)
- [ ] Tests skipped
- [ ] Tests xfailed (expected failures)
- [ ] Total execution time
- [ ] Average test duration
- [ ] Slowest 10 tests

### Coverage Metrics
- [ ] Line coverage %
- [ ] Branch coverage %
- [ ] Functions coverage %
- [ ] Files with gaps
- [ ] Excluded lines count

### Failure Metrics
- [ ] Failure count by category
- [ ] Failure percentage
- [ ] Flaky test count
- [ ] Deterministic failure count
- [ ] Errors vs failures

### Performance Metrics
- [ ] P50 test duration
- [ ] P95 test duration
- [ ] P99 test duration
- [ ] Outliers (>30s)

---

## Storage & Reporting

### Phase 0.4 Deliverables
1. **Baseline Execution Report**
   - All 600+ test results
   - Failure list with stack traces
   - Coverage report (HTML)
   - Performance profile

2. **Issue Documentation**
   - One document per issue (#1-7)
   - Root cause analysis
   - Recommended fixes
   - Test cases that verify fix

3. **Metrics Summary**
   - Execution time trend
   - Pass rate by category
   - Coverage baseline
   - Slowest tests

### Files to Create
```
backend/tests/
├── .baseline/
│   ├── run_001_results.json
│   ├── run_002_results.json
│   ├── run_003_results.json
│   └── analysis.md
├── failures/
│   ├── issue_1_ml_models.md
│   ├── issue_2_websocket.md
│   ├── issue_3_database.md
│   ├── issue_4_cache.md
│   ├── issue_5_async.md
│   ├── issue_6_api_mocks.md
│   └── issue_7_rate_limit.md
└── metrics/
    ├── coverage.xml
    ├── coverage.html
    └── performance.json
```

---

## Success Criteria for Phase 0.4

- [ ] Full test suite executed 3 times
- [ ] 600+ tests logged with results
- [ ] Failures categorized to issues #1-7
- [ ] Root cause identified for each failure
- [ ] Flaky tests isolated
- [ ] Coverage metrics captured
- [ ] Performance baseline established
- [ ] Documentation updated
- [ ] Issues tracked for Phase 0.5

---

## References

- Test Suite Baseline: `TEST_BASELINE_REPORT.md`
- Test Configuration: `pytest.ini`
- Test Fixtures: `backend/tests/conftest.py`
- Investigation Tracking: `INVESTIGATION_LOG.md` (to be created Phase 0.4)
