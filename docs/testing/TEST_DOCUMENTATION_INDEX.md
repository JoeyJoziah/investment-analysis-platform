# Test Documentation Index
Quick reference guide to all testing documentation

---

## Documentation Files

### [TEST_BASELINE_REPORT.md](TEST_BASELINE_REPORT.md)
**Audience**: QA Engineers, Developers, Project Managers

Test suite structure, organization, and coverage areas:
- Test suite organization by category
- Configuration details (pytest.ini, conftest.py)
- Coverage areas and metrics
- Expected runtime (3-7 minutes)
- Test execution profile
- Critical user flows

**Read this for**: Understanding test suite structure, configuration, and coverage.

---

### [TESTING_GUIDE.md](TESTING_GUIDE.md)
**Audience**: All Developers, QA Engineers

Practical guide for writing and running tests:
- Quick start (running tests, organization, fixtures)
- Writing tests (unit, integration, async examples)
- Markers and filtering
- Mocking best practices
- Custom fixtures
- Coverage requirements (85% minimum)
- Debugging techniques
- Performance testing
- Common issues and solutions
- CI/CD integration
- Best practices checklist

**Use this when**: Writing new tests or running existing tests.

---

### [INTEGRATION_TESTS.md](INTEGRATION_TESTS.md)
**Audience**: Backend Developers, QA Engineers

Integration test patterns for cross-component flows:
- Stock-to-analysis flow
- Auth-to-portfolio flow
- Agents-to-recommendations flow
- GDPR lifecycle tests

**Use this when**: Writing integration tests spanning multiple components.

---

### [TEST_EXECUTION_CHECKLIST.md](TEST_EXECUTION_CHECKLIST.md)
**Audience**: All Developers, DevOps

Pre-deployment test execution procedures:
- Environment setup
- Test commands by category
- Frontend unit tests (Vitest)
- E2E tests (Playwright)
- Pre-deployment verification

**Use this when**: Running tests before a deployment.

---

## Project Test Files

### Configuration & Setup
- **pytest.ini** - Test configuration and markers
- **backend/tests/conftest.py** - Fixtures and test setup

### Test Layout

```
backend/tests/
├── unit/                          (28 files, unit tests)
│   ├── test_services_*.py
│   ├── test_utils_*.py
│   ├── test_ml_*.py
│   └── test_middleware_*.py
├── test_api_integration.py
├── test_websocket_integration.py
├── test_security_compliance.py
├── test_database_integration.py
├── test_performance_load.py
├── test_watchlist.py
├── test_recommendation_engine.py
├── test_error_scenarios.py
├── test_financial_model_validation.py
└── [40+ additional test files]
```

### Test Files by Category

#### Security & Compliance
- `test_security_compliance.py` - JWT, CSRF, SQL injection, GDPR, SEC
- `test_security_integration.py` - Auth flows, authorization, secrets
- `test_rate_limiting.py` - Token bucket, priority queue, batch handling

#### WebSocket & Real-time
- `test_websocket_integration.py` - Connection, subscriptions, latency, reconnection

#### Database & Integration
- `test_database_integration.py` - Transactions, connection pool, query performance
- `test_integration_comprehensive.py` - End-to-end workflows
- `test_data_pipeline_integration.py` - ETL, data ingestion, quality checks

#### Performance & ML
- `test_performance_load.py` - Load testing, benchmarks, resource utilization
- `test_performance_optimizations.py` - Query optimization, cache effectiveness
- `test_ml_performance.py` - Model inference, feature engineering

#### Business Logic
- `test_watchlist.py` - Watchlist management, stock tracking
- `test_recommendation_engine.py` - ML recommendations, portfolio suggestions
- `test_thesis_api.py` - Investment thesis, analysis endpoints

#### Error Handling & Resilience
- `test_error_scenarios.py` - Rate limiting, DB loss, circuit breaker activation
- `test_resilience_integration.py` - Error recovery, fault tolerance
- `test_circuit_breaker.py` - State transitions, failure thresholds, recovery

#### Caching & Data Quality
- `test_cache_decorator.py` - Cache operations, decorator functionality
- `test_bloom_filter.py` - Bloom filter operations
- `test_data_quality.py` - Data validation, quality checks
- `test_n1_query_fix.py` - N+1 query prevention

#### Financial Analysis
- `test_financial_model_validation.py` - DCF, financial calculations, model validation
- `test_dividend_analyzer.py` - Dividend analysis, historical data
- `test_cointegration.py` - Statistical analysis, correlation testing

---

## Quick Reference

### I want to...

#### Run all tests
```bash
pytest backend/tests/ -v
```

#### Run tests in a category
```bash
pytest backend/tests/ -m security   # Security tests
pytest backend/tests/ -m api        # API tests
pytest backend/tests/ -m "not slow" # Skip slow tests
```

#### Run unit tests only
```bash
pytest backend/tests/unit/ -v
```

#### Write a new test
See `TESTING_GUIDE.md`, "Writing Tests" section for unit, integration, and async examples.

#### Debug a failing test
```bash
pytest path/to/test_file.py::test_name -vv -s  # Verbose with output
pytest path/to/test_file.py::test_name --pdb   # Drop to debugger
```

#### Check test coverage
```bash
pytest backend/tests/ --cov=backend --cov-report=html
# Open htmlcov/index.html to view
```

#### View test configuration
Read `pytest.ini` and `backend/tests/conftest.py`.
Or see `TEST_BASELINE_REPORT.md` "Test Configuration" section.

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Tests Passing | 5020+ |
| Test Files | 71+ |
| Unit Test Files | 28 |
| Coverage Target | 85% |
| Expected Runtime | 3-7 min |
| Critical Path Flows | 5 |

---

## File Locations

```
docs/testing/
├── TEST_DOCUMENTATION_INDEX.md     (this file)
├── TEST_BASELINE_REPORT.md         (test suite structure and coverage)
├── TESTING_GUIDE.md                (developer guide)
├── INTEGRATION_TESTS.md            (integration test patterns)
└── TEST_EXECUTION_CHECKLIST.md     (pre-deployment checklist)

pytest.ini                          (test configuration)
backend/tests/conftest.py           (test fixtures)
backend/tests/unit/                 (28 unit test files)
backend/tests/test_*.py             (integration and e2e tests)
.claude/rules/testing.md            (project testing rules)
```

---

## Getting Started

**For Developers writing new tests**:
1. Read `TESTING_GUIDE.md` (20 min)
2. Look at examples in `backend/tests/unit/`
3. Reference `TESTING_GUIDE.md` when writing tests

**For QA Engineers**:
1. Read `TEST_BASELINE_REPORT.md` for coverage understanding
2. Use `TEST_EXECUTION_CHECKLIST.md` before deployments
3. See `INTEGRATION_TESTS.md` for cross-component test patterns

---

## Related Resources

- **Testing Best Practices**: `.claude/rules/testing.md`
- **Code Style Requirements**: `.claude/rules/coding-style.md`
- **Security Guidelines**: `.claude/rules/security.md`
- **Git Workflow**: `.claude/rules/git-workflow.md`

---

**Last Updated**: 2026-03-04
