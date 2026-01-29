# Integration Test Fix Checklist

**Quick Reference for Systematic Test Remediation**

---

## Week 1: Quick Wins (Days 1-5)

### Day 1: CSRF Configuration ⚡ CRITICAL

- [ ] **Task 1.1:** Add CSRF exemptions to conftest.py (2h)
  ```python
  # backend/tests/integration/conftest.py
  exempt_paths = ["/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/refresh"]
  for path in exempt_paths:
      app.state.csrf_exempt_paths.add(path)
  ```

- [ ] **Task 1.2:** Test CSRF configuration (1h)
  ```bash
  pytest backend/tests/integration/test_auth_to_portfolio_flow.py -v
  ```

- [ ] **Task 1.3:** Verify 5 auth tests now pass (1h)
  - test_login_to_portfolio_access
  - test_role_based_portfolio_limits
  - test_session_expiry_during_portfolio
  - test_concurrent_portfolio_updates
  - test_portfolio_rebalancing_with_locks

**Expected Result:** 10/38 → 15/38 tests passing (39%)

---

### Day 2: Critical Endpoints Part 1 (8h)

- [ ] **Task 2.1:** Implement POST /api/v1/agents/analysis (2h)
  - File: backend/api/routers/agents.py
  - Tests fixed: test_agent_analysis_to_recommendation
  - Model: AgentAnalysisRequest, AgentAnalysisResponse
  - Logic: Trigger multi-agent stock analysis

- [ ] **Task 2.2:** Implement POST /api/v1/ml/predictions (2h)
  - File: backend/api/routers/ml.py (NEW)
  - Tests fixed: test_ml_prediction_to_agent_analysis
  - Model: PredictionRequest, PredictionResponse
  - Logic: Generate ML-based predictions

- [ ] **Task 2.3:** Implement GET /api/v1/stocks/search (2h)
  - File: backend/api/routers/stocks.py
  - Tests fixed: test_stock_lookup_to_recommendation
  - Logic: Full-text search on symbols/names

- [ ] **Task 2.4:** Implement POST /api/v1/stocks/alerts (2h)
  - File: backend/api/routers/stocks.py
  - Tests fixed: test_real_time_quote_to_alert
  - Model: AlertRequest, AlertResponse
  - Logic: Price alert creation

**Expected Result:** 15/38 → 19/38 tests passing (50%)

---

### Day 3: Critical Endpoints Part 2 (8h)

- [ ] **Task 3.1:** Implement POST /api/v1/thesis/create (2h)
  - File: backend/api/routers/thesis.py
  - Tests fixed: test_stock_fundamentals_to_thesis
  - Model: ThesisRequest, ThesisResponse
  - Logic: Generate investment thesis from data

- [ ] **Task 3.2:** Implement POST /api/v1/portfolio/holdings (2h)
  - File: backend/api/routers/portfolio.py
  - Tests fixed: test_stock_to_portfolio_addition
  - Model: HoldingRequest, HoldingResponse
  - Logic: Add stock to portfolio

- [ ] **Task 3.3:** Implement POST /api/v1/gdpr/export (2h)
  - File: backend/api/routers/gdpr.py
  - Tests fixed: test_user_registration_to_data_export
  - Model: DataExportRequest, DataExportResponse
  - Logic: Export user data (GDPR compliance)

- [ ] **Task 3.4:** Implement PUT /api/v1/gdpr/consent (2h)
  - File: backend/api/routers/gdpr.py
  - Tests fixed: test_consent_affects_data_collection
  - Model: ConsentRequest, ConsentResponse
  - Logic: Update consent preferences

**Expected Result:** 19/38 → 23/38 tests passing (61%)

---

### Day 4: GDPR Completion (8h)

- [ ] **Task 4.1:** Implement DELETE /api/v1/gdpr/delete (3h)
  - File: backend/api/routers/gdpr.py
  - Tests fixed: test_data_deletion_cascades
  - Logic: Delete user data with cascades
  - **CRITICAL:** Test cascade logic thoroughly

- [ ] **Task 4.2:** Implement POST /api/v1/gdpr/anonymize (2h)
  - File: backend/api/routers/gdpr.py
  - Tests fixed: test_anonymization_in_analytics
  - Logic: Anonymize user data in analytics

- [ ] **Task 4.3:** Implement GET /api/v1/gdpr/audit (3h)
  - File: backend/api/routers/gdpr.py
  - Tests fixed: test_gdpr_compliance_audit_trail
  - Logic: Generate audit trail report

**Expected Result:** 23/38 → 26/38 tests passing (68%)

---

### Day 5: Service Layer Foundation (8h)

- [ ] **Task 5.1:** Create backend/services/recommendation_service.py (3h)
  ```python
  class RecommendationService:
      async def multi_agent_consensus(
          self, symbol: str, analyses: List[AgentAnalysis]
      ) -> ConsensusRecommendation:
          """Aggregate multiple agent analyses."""
          # Weight analyses by agent confidence
          # Calculate consensus score
          # Generate recommendation
  ```
  - Tests fixed: test_recommendation_confidence_scoring

- [ ] **Task 5.2:** Create backend/services/trading_service.py (3h)
  ```python
  class TradingService:
      async def execute_order(
          self, order: OrderRequest, portfolio_id: int
      ) -> OrderResult:
          """Execute order with validation."""
          # Validate portfolio permissions
          # Check balance/holdings
          # Execute order
          # Update portfolio
  ```
  - Tests fixed: test_recommendation_to_portfolio_action

- [ ] **Task 5.3:** Create backend/agents/analysis_agent.py (2h)
  ```python
  class AnalysisAgent:
      async def analyze_fundamentals(
          self, symbol: str, data: FundamentalData
      ) -> FundamentalAnalysis:
          """Perform fundamental analysis."""
          # Calculate ratios
          # Compare to sector
          # Generate insights
  ```
  - Tests fixed: test_agent_error_handling_cascade

**Expected Result:** 26/38 → 29/38 tests passing (76%)

---

## Week 2: Middleware & Infrastructure (Days 6-10)

### Day 6-7: Middleware Fixes (16h)

- [ ] **Task 6.1:** Fix middleware execution order (4h)
  - File: backend/api/main.py
  - Add priority-based registration
  - Test: test_middleware_stack_execution_order

- [ ] **Task 6.2:** Fix security headers + CORS (4h)
  - File: backend/middleware/security_headers.py
  - Implement header merging strategy
  - Test: test_security_headers_with_cors

- [ ] **Task 6.3:** Fix request size validation (4h)
  - File: backend/middleware/request_size_limiter.py
  - Enforce before routing
  - Test: test_request_size_limits_with_json_payload

- [ ] **Task 6.4:** Fix async client patterns (2h)
  - File: backend/tests/integration/conftest.py
  - Proper lifecycle management
  - Test: test_async_client_pattern_consistency

- [ ] **Task 6.5:** Optimize middleware performance (2h)
  - Add caching layer
  - Test: test_middleware_overhead_acceptable

**Expected Result:** 29/38 → 34/38 tests passing (89%)

---

### Day 8-10: Remaining Issues (24h)

- [ ] **Task 7.1:** Fix Pydantic model validation (4h)
  - Update all models to use ConfigDict
  - Test: test_pydantic_models_end_to_end

- [ ] **Task 7.2:** Fix portfolio endpoint completeness (8h)
  - Review all portfolio endpoints
  - Add missing routes
  - Test: test_existing_portfolio_endpoints_work

- [ ] **Task 7.3:** Fix stock endpoint completeness (8h)
  - Review all stock endpoints
  - Add missing routes
  - Test: test_existing_stock_endpoints_work

- [ ] **Task 7.4:** Implement stock caching (4h)
  - Add Redis caching layer
  - Test: test_stock_data_caching

**Expected Result:** 34/38 → 38/38 tests passing (100%) ✓

---

## Week 3+: Coverage Expansion

### Unit Test Coverage (40h)

- [ ] **Analytics Module Tests (20h)**
  - [ ] fundamental_analysis.py: Add 50+ unit tests
  - [ ] technical_analysis.py: Add 60+ unit tests
  - [ ] recommendation_engine.py: Add 40+ unit tests
  - [ ] sentiment_analysis.py: Add 25+ unit tests

- [ ] **Data Ingestion Tests (10h)**
  - [ ] Test all API clients (Alpha Vantage, Finnhub, Polygon)
  - [ ] Test error handling and retries
  - [ ] Test rate limiting

- [ ] **Router Tests (10h)**
  - [ ] Test all API endpoints
  - [ ] Test error cases
  - [ ] Test validation

**Expected Result:** Coverage 10% → 40%

---

### E2E Test Suite (16h)

- [ ] **Critical User Flows (16h)**
  - [ ] User registration → portfolio → first trade
  - [ ] Stock search → analysis → recommendation
  - [ ] Real-time data → alert generation
  - [ ] GDPR export → deletion workflow

**Expected Result:** E2E coverage for all critical paths

---

## Progress Tracking

Use this checklist to track daily progress:

```
Day 1:  [ ] CSRF Config        Expected: 15/38 tests  Actual: ___/38
Day 2:  [ ] Endpoints Pt1      Expected: 19/38 tests  Actual: ___/38
Day 3:  [ ] Endpoints Pt2      Expected: 23/38 tests  Actual: ___/38
Day 4:  [ ] GDPR Complete      Expected: 26/38 tests  Actual: ___/38
Day 5:  [ ] Service Layer      Expected: 29/38 tests  Actual: ___/38
Day 6-7: [ ] Middleware       Expected: 34/38 tests  Actual: ___/38
Day 8-10: [ ] Remaining       Expected: 38/38 tests  Actual: ___/38
```

---

## Daily Verification Commands

Run these after each day's work:

```bash
# Full test suite
pytest backend/tests/integration/ -v --cov=backend --cov-report=term-missing

# Quick status check
pytest backend/tests/integration/ -q --tb=no | tail -5

# Coverage summary
pytest backend/tests/integration/ --cov=backend --cov-report=term | grep "TOTAL"

# Specific test file
pytest backend/tests/integration/test_auth_to_portfolio_flow.py -v

# HTML coverage report
pytest backend/tests/integration/ --cov=backend --cov-report=html
open htmlcov/index.html
```

---

## Success Criteria

### Week 1
- [x] 0 schema errors (DONE)
- [ ] 0 CSRF failures
- [ ] 50%+ test pass rate
- [ ] All critical endpoints implemented

### Week 2
- [ ] 82%+ test pass rate
- [ ] Middleware stack stable
- [ ] Service layer complete

### Week 3+
- [ ] 100% integration test pass rate
- [ ] 40%+ overall coverage
- [ ] Production-ready infrastructure

---

## Blocked/Dependency Tracker

Use this section to track blockers:

```
BLOCKED ITEMS:
- None currently

WAITING ON:
- None currently

COMPLETED:
✓ Schema fixes (Wave 4.5)
```

---

## Notes & Observations

Track issues discovered during implementation:

```
ISSUES FOUND:
- [Date] [Issue description]
- [Date] [Resolution]

PERFORMANCE NOTES:
- [Date] [Observation]

SECURITY NOTES:
- [Date] [Concern/Fix]
```

---

## Quick Reference: Test Categories

**Category 1: Missing Endpoints (15 tests)**
- Agent/Recommendation Flow: 5 tests
- Stock Analysis Flow: 5 tests
- GDPR Compliance: 5 tests

**Category 2: Missing Services (3 tests)**
- recommendation_service
- trading_service
- analysis_agent

**Category 3: CSRF/Auth (5 tests)**
- All auth flow tests

**Category 4: Middleware (5 tests)**
- Execution order
- Header conflicts
- Size validation
- Client patterns
- Performance

---

**Last Updated:** 2026-01-28
**Next Review:** After each day's work
