# Integration Tests Documentation

## Overview

This document describes the comprehensive integration test suite for the Investment Analysis Platform. The tests validate end-to-end workflows across multiple modules, ensuring that components work correctly together in realistic scenarios.

## Test Organization

### Directory Structure

```
backend/tests/integration/
├── __init__.py
├── test_stock_to_analysis_flow.py          # 5 tests
├── test_auth_to_portfolio_flow.py          # 5 tests
├── test_agents_to_recommendations_flow.py  # 5 tests
└── test_gdpr_data_lifecycle.py             # 5 tests
```

## Test Suites

### 1. Stock to Analysis Flow (5 tests)

**File:** `test_stock_to_analysis_flow.py`

Tests the complete pipeline from stock data retrieval through to investment recommendations.

#### Tests:

1. **test_stock_lookup_to_recommendation**
   - **Purpose:** Validates end-to-end stock analysis pipeline
   - **Flow:** Stock lookup → Price history → Fundamentals → Recommendation generation
   - **Validates:** Data integration, analysis quality, recommendation accuracy
   - **Key Assertions:**
     - Stock data retrieved correctly
     - Price trends calculated accurately
     - Fundamental metrics available
     - Recommendations have high confidence (>80%)

2. **test_stock_data_caching**
   - **Purpose:** Tests cache hit/miss scenarios for performance
   - **Flow:** First request (cache miss) → Cache set → Second request (cache hit)
   - **Validates:** Caching mechanism, performance optimization
   - **Key Assertions:**
     - Cache miss triggers database query
     - Cache hit avoids database query
     - Cache key properly formatted

3. **test_stock_to_portfolio_addition**
   - **Purpose:** Tests adding analyzed stocks to portfolios
   - **Flow:** Stock analysis → Portfolio position creation → Portfolio value update
   - **Validates:** Cross-module integration, transaction handling
   - **Key Assertions:**
     - Position created with correct quantities
     - Portfolio cash balance updated
     - Stock-to-portfolio linkage established

4. **test_real_time_quote_to_alert**
   - **Purpose:** Tests price alert triggering
   - **Flow:** Alert creation → Real-time price update → Alert evaluation → Notification
   - **Validates:** Real-time data processing, alert system
   - **Key Assertions:**
     - Alerts trigger when thresholds crossed
     - Alert count incremented
     - Last triggered timestamp updated

5. **test_stock_fundamentals_to_thesis**
   - **Purpose:** Tests investment thesis generation
   - **Flow:** Fundamental data → Analysis → Bull/Bear case generation
   - **Validates:** LLM integration, thesis quality
   - **Key Assertions:**
     - Thesis includes bull/bear cases
     - Key metrics identified
     - Confidence score reasonable (>70%)

---

### 2. Auth to Portfolio Flow (5 tests)

**File:** `test_auth_to_portfolio_flow.py`

Tests authentication, authorization, and portfolio access workflows.

#### Tests:

1. **test_login_to_portfolio_access**
   - **Purpose:** Validates complete authentication flow
   - **Flow:** Login → JWT token issuance → Portfolio access with token
   - **Validates:** Authentication, authorization, token-based access
   - **Key Assertions:**
     - Login returns valid JWT tokens
     - Tokens grant access to user's portfolio
     - Portfolio data matches user ownership

2. **test_role_based_portfolio_limits**
   - **Purpose:** Tests tier-based quota enforcement
   - **Flow:** Free user actions → Premium user actions → Quota comparison
   - **Validates:** Role-based access control, subscription tiers
   - **Key Assertions:**
     - Free users limited to 5 positions
     - Premium users can add 10+ positions
     - Quota errors returned when limits exceeded

3. **test_session_expiry_during_portfolio**
   - **Purpose:** Tests token expiry and refresh handling
   - **Flow:** Expired token rejected → Refresh token → New access token → Retry
   - **Validates:** Session management, token lifecycle
   - **Key Assertions:**
     - Expired tokens rejected with 401
     - Refresh tokens generate new access tokens
     - New tokens grant access

4. **test_concurrent_portfolio_updates**
   - **Purpose:** Tests race condition handling
   - **Flow:** Multiple simultaneous portfolio updates → Conflict resolution
   - **Validates:** Concurrency control, data consistency
   - **Key Assertions:**
     - At least 2 of 3 concurrent updates succeed
     - Portfolio balance remains consistent
     - No data corruption from race conditions

5. **test_portfolio_rebalancing_with_locks**
   - **Purpose:** Tests row-level locking during rebalancing
   - **Flow:** Create positions → Calculate rebalance → Execute with locks
   - **Validates:** Transaction isolation, lock management
   - **Key Assertions:**
     - Rebalancing plan generated
     - Locks prevent concurrent modifications
     - Final allocation matches target (±5%)

---

### 3. Agents to Recommendations Flow (5 tests)

**File:** `test_agents_to_recommendations_flow.py`

Tests AI agent analysis, ML predictions, and recommendation generation.

#### Tests:

1. **test_agent_analysis_to_recommendation**
   - **Purpose:** Validates LLM agent → recommendation pipeline
   - **Flow:** Agent fundamental analysis → ML prediction → Combined recommendation
   - **Validates:** Agent integration, multi-source analysis
   - **Key Assertions:**
     - Agent analysis includes SWOT
     - ML predictions incorporated
     - Recommendation combines both sources
     - Reasoning mentions fundamental and technical factors

2. **test_ml_prediction_to_agent_analysis**
   - **Purpose:** Tests ML predictions feeding into agent interpretation
   - **Flow:** ML price prediction → Agent interpretation → Contextualized insights
   - **Validates:** Model-to-agent handoff, interpretation quality
   - **Key Assertions:**
     - ML predictions have high confidence (>70%)
     - Agent provides context and interpretation
     - Action recommendations generated

3. **test_recommendation_confidence_scoring**
   - **Purpose:** Tests multi-agent consensus mechanism
   - **Flow:** Multiple agents analyze → Aggregate opinions → Weighted confidence
   - **Validates:** Consensus algorithm, confidence calculation
   - **Key Assertions:**
     - 4 agents provide opinions
     - Consensus recommendation selected
     - Agreement level calculated
     - Dissenting opinions captured

4. **test_recommendation_to_portfolio_action**
   - **Purpose:** Tests auto-execution of high-confidence recommendations
   - **Flow:** High-confidence recommendation → Auto-trade check → Order execution
   - **Validates:** Automated trading, confidence thresholds
   - **Key Assertions:**
     - Recommendations >85% confidence trigger auto-trades
     - Orders submitted with correct quantities
     - Execution logged for audit

5. **test_agent_error_handling_cascade**
   - **Purpose:** Tests graceful degradation on failures
   - **Flow:** Agent failure → Fallback to available data → Degraded recommendation
   - **Validates:** Error handling, resilience
   - **Key Assertions:**
     - Partial failures still produce recommendations
     - Warnings indicate missing data
     - Complete failures return 503 with helpful message

---

### 4. GDPR Data Lifecycle (5 tests)

**File:** `test_gdpr_data_lifecycle.py`

Tests GDPR compliance including data export, deletion, consent, and audit trails.

#### Tests:

1. **test_user_registration_to_data_export**
   - **Purpose:** Validates complete data lifecycle
   - **Flow:** Registration → Usage → Export all data
   - **Validates:** GDPR Article 20 (Data Portability)
   - **Key Assertions:**
     - Export includes all user data categories
     - Personal data present in export
     - JSON format with metadata
     - Export timestamp recorded

2. **test_consent_affects_data_collection**
   - **Purpose:** Tests consent-based filtering
   - **Flow:** Disable analytics consent → Actions → Verify no analytics
   - **Validates:** GDPR Article 6 (Lawful Basis)
   - **Key Assertions:**
     - Analytics disabled when consent=False
     - Analytics enabled when consent=True
     - Consent changes take effect immediately

3. **test_data_deletion_cascades**
   - **Purpose:** Tests complete data deletion
   - **Flow:** Account deletion request → Cascade delete all related data
   - **Validates:** GDPR Article 17 (Right to Erasure)
   - **Key Assertions:**
     - User record deleted
     - Portfolios, alerts, watchlists deleted
     - Sessions deleted
     - Audit logs anonymized but preserved

4. **test_anonymization_in_analytics**
   - **Purpose:** Tests PII scrubbing in aggregated data
   - **Flow:** User actions → Analytics aggregation → PII removal
   - **Validates:** Data minimization, anonymization
   - **Key Assertions:**
     - Email, name, phone not in analytics
     - Anonymized user IDs used (32+ char hashes)
     - No reverse-identification possible

5. **test_gdpr_compliance_audit_trail**
   - **Purpose:** Tests comprehensive audit logging
   - **Flow:** GDPR operations → Audit log entries → Verify completeness
   - **Validates:** GDPR Article 30 (Records of Processing)
   - **Key Assertions:**
     - All operations logged (consent, export, delete)
     - Audit entries immutable
     - Sensitive data redacted in logs
     - Retention policy enforced (365+ days)

---

## Running the Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx sqlalchemy aiosqlite

# Set test environment
export TESTING=True
export DATABASE_URL="sqlite+aiosqlite:///:memory:"
```

### Run All Integration Tests

```bash
# From project root
pytest backend/tests/integration/ -v

# With coverage
pytest backend/tests/integration/ --cov=backend --cov-report=html

# Specific test file
pytest backend/tests/integration/test_stock_to_analysis_flow.py -v

# Specific test
pytest backend/tests/integration/test_gdpr_data_lifecycle.py::test_data_deletion_cascades -v
```

### Test Markers

Tests are marked for selective execution:

```bash
# Run only integration tests
pytest -m integration

# Run all except slow tests
pytest -m "not slow"

# Run database tests only
pytest -m database
```

## Test Patterns

### Database Isolation

All tests use isolated in-memory SQLite databases:

```python
@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    async with test_db_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
```

### Mock External Services

External APIs are mocked to ensure test reliability:

```python
@pytest.fixture
def mock_external_apis():
    with patch('httpx.AsyncClient.get', side_effect=mock_get):
        yield mock_responses
```

### Async Testing

All tests use pytest-asyncio for async/await support:

```python
@pytest.mark.asyncio
async def test_async_workflow(async_client: AsyncClient):
    response = await async_client.get("/api/endpoint")
    assert response.status_code == 200
```

## Success Criteria

### Coverage Requirements
- **Statements:** >80%
- **Branches:** >75%
- **Functions:** >80%
- **Integration paths:** >85%

### Performance Targets
- Individual tests: <5 seconds
- Full suite: <60 seconds
- Database operations: <100ms per query

### Quality Metrics
- All tests pass consistently
- No flaky tests
- Clear, descriptive test names
- Comprehensive assertions
- Proper cleanup/teardown

## Common Issues & Solutions

### Issue: Tests Timeout

**Solution:** Check for blocking operations, ensure async/await used correctly

```python
# Wrong
result = blocking_function()

# Correct
result = await async_function()
```

### Issue: Database Conflicts

**Solution:** Ensure proper session isolation and rollback

```python
# Always rollback after test
finally:
    await session.rollback()
    await session.close()
```

### Issue: Mock Not Working

**Solution:** Verify patch target path matches import path

```python
# Wrong - patches wrong location
@patch('backend.services.MyService')

# Correct - patches where it's imported
@patch('backend.api.routers.my_route.MyService')
```

## Best Practices

1. **Test Realistic Workflows**
   - Use actual user flows, not artificial test scenarios
   - Include edge cases and error paths
   - Test cross-module interactions

2. **Keep Tests Independent**
   - Each test should work in isolation
   - No dependencies between tests
   - Clean state before each test

3. **Use Descriptive Names**
   - Test name should explain what and why
   - Include expected outcome in name
   - Use underscores for readability

4. **Mock External Dependencies**
   - Don't rely on external APIs
   - Mock database for unit tests
   - Use in-memory DB for integration tests

5. **Assert Thoroughly**
   - Test both success and failure paths
   - Verify side effects and state changes
   - Check edge cases and boundaries

## Maintenance

### Adding New Tests

1. Identify integration point to test
2. Create fixture for required data
3. Write test following existing patterns
4. Add to appropriate test file
5. Update this documentation

### Updating Existing Tests

1. Check if changes affect multiple tests
2. Update fixtures if data model changed
3. Verify all affected tests still pass
4. Update documentation if behavior changed

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites)

## Contact

For questions about the integration tests:
- Check existing test examples
- Review test infrastructure in `conftest.py`
- Consult testing documentation
- Ask the development team
