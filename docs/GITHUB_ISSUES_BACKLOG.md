# GitHub Issues Backlog - Investment Analysis Platform

**Generated:** 2026-02-08
**Last Updated:** 2026-02-08 (post-Queen Audit)
**Status:** In Progress
**Total Issues:** 43 prioritized issues across 7 epics (3 done, 2 partially done, 38 remaining)

---

## Completed Work (Queen Audit - 2026-02-08)

The following items were completed across 5 commits in the 30-agent Queen Audit:

| Commit | Description | Issues Affected |
|--------|-------------|-----------------|
| `e3fada0` | Remove 28 orphan docs from root | Housekeeping (no issue) |
| `bca6756` | Security hardening: CSRF, rate limiting, auth gates, SSL | #1, #7 (partial), #16 (partial) |
| `d23c3e5` | Replace `datetime.utcnow()` with timezone-aware UTC across 100+ files | #41 (new, done) |
| `c156cc4` | Add 10 new integration and security test suites | #42 (new, done) |
| `1e3e73e` | CI/CD coverage reporting, TS strict audit, docs | #43 (new, done) |

---

## Quick Reference

| Epic | Issues | Priority | Total Effort | Done |
|------|--------|----------|--------------|------|
| 1. Testing Excellence | 9 issues | P0-P1 | 78 hours | #1 done |
| 2. Performance & Reliability | 6 issues | P1-P2 | 175 hours | 0 |
| 3. Security Hardening | 4 issues | P1-P2 | 32 hours | #16 partial |
| 4. Advanced Analytics | 6 issues | P2 | 95 hours | 0 |
| 5. User Experience | 5 issues | P2 | 68 hours | 0 |
| 6. Enterprise Features | 5 issues | P3 | 400 hours | 0 |
| 7. Market Expansion | 5 issues | P3 | 540 hours | 0 |
| Audit Follow-ups | 4 issues | P1-P2 | 28 hours | #41, #42, #43 done |
| **TOTAL** | **44** | **P0-P3** | **1,416 hours** | **3 done, 2 partial** |

---

## Epic 1: Testing Excellence

**Goal:** 80% test coverage, 100% integration tests passing
**Timeline:** 2 weeks
**Total Effort:** 78 hours

### Issue #1: CSRF Test Configuration Fix -- DONE
**Priority:** P0 | **Effort:** S (4h) | **Milestone:** Week 1
**Status:** DONE (2026-02-08, commit `bca6756` + `c156cc4`)

**Description:**
Integration tests were failing due to CSRF validation blocking test authentication endpoints. Fixed by hardening CSRF enforcement with timing-safe comparison and adding comprehensive CSRF test coverage.

**Acceptance Criteria:**
- [x] Add CSRF exempt paths for `/api/v1/auth/*` in test configuration
- [x] Update AsyncClient fixture with proper CSRF handling
- [x] Add `TESTING=true` environment detection
- [x] Verify auth-related tests pass with CSRF enforcement

**Completed in:**
- `backend/security/csrf_protection.py` -- Timing-safe CSRF token comparison
- `backend/tests/security/test_csrf_protection.py` -- Expanded from basic to comprehensive (540+ lines)
- `backend/tests/conftest.py` -- Updated test fixtures for CSRF handling

**Labels:** `bug`, `testing`, `p0`, `quick-win`, `csrf`, `status:done`

---

### Issue #2: Implement Agent Analysis Endpoint
**Priority:** P0 | **Effort:** S (2h) | **Milestone:** Week 1

**Description:**
Create `POST /api/v1/agents/analysis` endpoint to trigger AI agent-based stock analysis.

**Acceptance Criteria:**
- [ ] Create endpoint in `backend/api/routers/agents.py`
- [ ] Add `AgentAnalysisRequest` and `AgentAnalysisResponse` Pydantic models
- [ ] Implement agent orchestration logic
- [ ] Add endpoint to OpenAPI documentation
- [ ] Verify 2 integration tests pass

**API Contract:**
```python
# Request
{
  "ticker": "AAPL",
  "analysis_types": ["fundamental", "technical", "sentiment"],
  "depth": "standard" | "deep"
}

# Response
{
  "analysis_id": "uuid",
  "ticker": "AAPL",
  "results": {...},
  "confidence_score": 0.85,
  "timestamp": "2026-02-08T12:00:00Z"
}
```

**Labels:** `feature`, `api`, `p0`, `agents`
**Assignee:** Backend Team
**Dependencies:** None
**Estimate:** 2 hours

---

### Issue #3: Implement ML Predictions Endpoint
**Priority:** P0 | **Effort:** S (2h) | **Milestone:** Week 1

**Description:**
Create new ML router with `POST /api/v1/ml/predictions` endpoint for stock price predictions.

**Acceptance Criteria:**
- [ ] Create `backend/api/routers/ml.py` router
- [ ] Integrate with existing ML service
- [ ] Add prediction caching (Redis, 15min TTL)
- [ ] Include confidence intervals in response
- [ ] Verify 1 integration test passes

**API Contract:**
```python
# Request
{
  "ticker": "AAPL",
  "model": "lstm" | "xgboost" | "prophet" | "ensemble",
  "horizon_days": 30
}

# Response
{
  "prediction_id": "uuid",
  "ticker": "AAPL",
  "predictions": [
    {"date": "2026-02-09", "price": 175.50, "confidence": 0.82},
    ...
  ],
  "model_used": "ensemble",
  "accuracy_metrics": {...}
}
```

**Labels:** `feature`, `ml`, `p0`, `api`
**Assignee:** ML Team
**Dependencies:** ML models trained (complete)
**Estimate:** 2 hours

---

### Issue #4: Stock Search & Alerts Endpoints
**Priority:** P0 | **Effort:** S (4h) | **Milestone:** Week 1

**Description:**
Implement two critical stock endpoints: search functionality and price alerts.

**Acceptance Criteria:**
- [ ] `GET /api/v1/stocks/search` - Full-text search
  - Search by ticker symbol or company name
  - PostgreSQL `pg_trgm` for fuzzy matching
  - Return top 10 results, sorted by relevance
- [ ] `POST /api/v1/stocks/alerts` - Price alerts
  - Create price threshold alerts
  - Email/webhook notification support
  - Alert status tracking
- [ ] Verify 2 integration tests pass

**API Contracts:**
```python
# GET /api/v1/stocks/search?q=apple&limit=10
{
  "results": [
    {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    ...
  ],
  "total": 2
}

# POST /api/v1/stocks/alerts
{
  "ticker": "AAPL",
  "condition": "above" | "below",
  "price_threshold": 180.00,
  "notification_method": "email" | "webhook"
}
```

**Labels:** `feature`, `stocks`, `p0`, `search`, `alerts`
**Assignee:** Backend Team
**Dependencies:** None
**Estimate:** 4 hours

---

### Issue #5: Complete GDPR Endpoint Suite
**Priority:** P0 | **Effort:** M (8h) | **Milestone:** Week 1

**Description:**
Implement 5 GDPR compliance endpoints to ensure regulatory compliance and pass related integration tests.

**Acceptance Criteria:**
- [ ] `POST /api/v1/gdpr/export` - Export all user data
- [ ] `PUT /api/v1/gdpr/consent` - Update consent preferences
- [ ] `DELETE /api/v1/gdpr/delete` - Delete user account and data
- [ ] `POST /api/v1/gdpr/anonymize` - Anonymize user data
- [ ] `GET /api/v1/gdpr/audit` - Retrieve audit trail
- [ ] Add compliance validation logic
- [ ] Create comprehensive audit trail
- [ ] Verify 5 integration tests pass

**Compliance Requirements:**
- Data export within 30 days (GDPR Article 20)
- Right to be forgotten (GDPR Article 17)
- Consent tracking (GDPR Article 7)
- Audit trail 90-day retention

**Labels:** `feature`, `compliance`, `gdpr`, `p0`, `legal`
**Assignee:** Backend Team
**Dependencies:** None
**Estimate:** 8 hours

---

### Issue #6: Service Layer Implementation
**Priority:** P0 | **Effort:** L (24h) | **Milestone:** Week 2

**Description:**
Implement business logic layer with three critical services to support complex operations and pass integration tests.

**Acceptance Criteria:**
- [ ] Create `RecommendationService` - Multi-agent consensus
  - Aggregate analyses from multiple AI agents
  - Implement weighted voting based on agent expertise
  - Calculate confidence scores
  - Generate consolidated recommendations
- [ ] Create `TradingService` - Order execution
  - Validate trading orders
  - Execute portfolio actions
  - Track transaction history
  - Calculate portfolio impact
- [ ] Create `AnalysisAgent` - Fundamental analysis orchestration
  - Coordinate fundamental analysis workflow
  - Handle error cascades gracefully
  - Cache intermediate results
  - Return structured analysis
- [ ] Verify 3 integration tests pass

**Architecture:**
```
Controllers (API Routers)
         |
Services (Business Logic)
         |
Repositories (Data Access)
         |
Models (Database)
```

**Labels:** `architecture`, `refactor`, `p0`, `services`
**Assignee:** Backend Team
**Dependencies:** Issues #2, #3 complete
**Estimate:** 24 hours

---

### Issue #7: Middleware Stack Optimization -- PARTIALLY DONE
**Priority:** P0 | **Effort:** M (16h) | **Milestone:** Week 2
**Status:** PARTIALLY DONE (2026-02-08, commits `bca6756` + `d23c3e5`)

**Description:**
Fix middleware execution order conflicts, CORS + security header issues, and request validation problems causing integration test failures.

**Completed:**
- [x] CSRF middleware hardened with timing-safe token comparison
- [x] Security headers updated in `security_integration.py`
- [x] Dual database initialization cleaned up in `main.py`
- [x] Rate limiting added to `/auth/refresh` endpoint

**Remaining:**
- [ ] Implement priority-based middleware registration
- [ ] Resolve CORS + SecurityHeaders header conflicts
- [ ] Add request size validation before routing
- [ ] Fix AsyncClient event loop issues in tests
- [ ] Reduce middleware overhead to <5ms
- [ ] Verify 4 integration tests pass

**Technical Details:**
```python
# Priority order (1 = first)
middleware_stack = [
    (ErrorHandlerMiddleware, 1),
    (CSRFProtectionMiddleware, 2),
    (SecurityHeadersMiddleware, 3),
    (RequestSizeLimiterMiddleware, 4),
    (MonitoringMiddleware, 5),
]
```

**Labels:** `bug`, `middleware`, `p0`, `performance`, `status:in-progress`
**Assignee:** Backend Team
**Dependencies:** None
**Estimate:** 8 hours remaining

---

### Issue #8: Core Analytics Test Coverage
**Priority:** P1 | **Effort:** L (8h) | **Milestone:** Month 1

**Description:**
Increase test coverage for critical analytics modules from <15% to 50%.

**Acceptance Criteria:**
- [ ] `fundamental_analysis.py`: 8.56% -> 50% coverage
- [ ] `technical_analysis.py`: 7.03% -> 50% coverage
- [ ] `recommendation_engine.py`: 13.11% -> 50% coverage
- [ ] Add unit tests for all public methods
- [ ] Add integration tests for workflows
- [ ] Mock external API calls

**Note:** 10 new integration/security test suites were added in commit `c156cc4` covering routers (analysis, auth, health, news, recommendations, settings, stocks, websocket) and security modules (rate limiter, security modules). These improve overall coverage but do not yet target the analytics module internals listed above.

**Focus Areas:**
- DCF valuation calculations
- Technical indicator accuracy
- Recommendation confidence scoring
- Error handling and edge cases

**Labels:** `testing`, `coverage`, `p1`, `analytics`
**Assignee:** Backend Team
**Dependencies:** Issue #6 complete
**Estimate:** 8 hours

---

### Issue #9: E2E Test Suite Creation
**Priority:** P1 | **Effort:** M (10h) | **Milestone:** Month 1

**Description:**
Create Playwright-based end-to-end test suite for critical user journeys.

**Acceptance Criteria:**
- [ ] Set up Playwright framework
- [ ] Create 10 critical user journey tests:
  - User registration -> portfolio creation
  - Stock search -> analysis -> watchlist
  - Login -> real-time quotes -> alerts
  - Portfolio rebalancing flow
  - GDPR data export flow
- [ ] Integrate with CI/CD pipeline
- [ ] Configure parallel test execution
- [ ] Add visual regression testing

**Test Scenarios:**
```javascript
test('complete investment workflow', async ({ page }) => {
  await page.goto('/register');
  await registerUser(page);
  await searchStock(page, 'AAPL');
  await viewAnalysis(page, 'AAPL');
  await addToWatchlist(page, 'AAPL');
  await createPortfolio(page);
  await expect(page.locator('.portfolio-value')).toBeVisible();
});
```

**Labels:** `testing`, `e2e`, `p1`, `playwright`
**Assignee:** QA Team
**Dependencies:** Frontend stable
**Estimate:** 10 hours

---

## Epic 2: Performance & Reliability

**Goal:** 60-80% performance improvement, 99.9% uptime
**Timeline:** 1-3 months
**Total Effort:** 175 hours

### Issue #10: Database Query Optimization
**Priority:** P1 | **Effort:** M (10h) | **Milestone:** Month 1

**Description:**
Optimize database performance through indexing, query optimization, and connection pooling.

**Acceptance Criteria:**
- [ ] Add missing indexes (10+ identified)
- [ ] Optimize N+1 queries (8 instances found)
- [ ] Tune connection pool settings
- [ ] Implement query result caching
- [ ] Reduce p95 query time from 50ms to 25ms

**Critical Indexes:**
```sql
CREATE INDEX idx_stocks_ticker ON stocks(ticker);
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_recommendations_created_at ON recommendations(created_at DESC);
CREATE INDEX idx_watchlists_user_stock ON watchlists(user_id, stock_id);
```

**N+1 Query Fixes:**
- Stock list with latest prices: Use `selectinload()`
- Portfolio with positions: Use `joinedload()`
- Recommendations with stocks: Use `subqueryload()`

**Labels:** `performance`, `database`, `p1`, `optimization`
**Assignee:** Database Team
**Dependencies:** None
**Estimate:** 10 hours

---

### Issue #11: Multi-Layer Cache Tuning
**Priority:** P1 | **Effort:** M (10h) | **Milestone:** Month 1

**Description:**
Optimize caching strategy to improve hit rates and reduce API load.

**Acceptance Criteria:**
- [ ] Optimize TTL values based on data freshness requirements
- [ ] Implement cache warming for popular stocks (top 100)
- [ ] Add cache hit rate monitoring (Prometheus metrics)
- [ ] Increase cache hit rate from 78% to 90%
- [ ] Reduce cache memory usage by 20%

**Cache Strategy:**
```python
CACHE_TTL = {
    "stock_quote": 60,        # 1 minute (real-time)
    "stock_fundamentals": 3600 * 24,  # 24 hours
    "stock_analysis": 3600 * 6,       # 6 hours
    "ml_predictions": 3600 * 4,       # 4 hours
    "recommendations": 3600 * 12,     # 12 hours
}
```

**Labels:** `performance`, `cache`, `p1`, `redis`
**Assignee:** Backend Team
**Dependencies:** None
**Estimate:** 10 hours

---

### Issue #12: API Response Time Optimization
**Priority:** P1 | **Effort:** M (10h) | **Milestone:** Month 1

**Description:**
Reduce API response times through async optimization, compression, and payload reduction.

**Acceptance Criteria:**
- [ ] Optimize async/await patterns
- [ ] Implement response compression (gzip)
- [ ] Reduce payload sizes (remove unnecessary fields)
- [ ] Add HTTP/2 support
- [ ] Reduce p95 response time from 200ms to 100ms

**Optimizations:**
- Parallel async operations (stock + analysis + recommendations)
- JSON payload compression (30-40% size reduction)
- Field filtering (`?fields=ticker,price,change`)
- Response pagination for large results

**Labels:** `performance`, `api`, `p1`, `optimization`
**Assignee:** Backend Team
**Dependencies:** None
**Estimate:** 10 hours

---

### Issue #13: Background Job Optimization
**Priority:** P1 | **Effort:** M (10h) | **Milestone:** Month 1

**Description:**
Optimize Celery tasks and Airflow DAGs for better throughput and resource utilization.

**Acceptance Criteria:**
- [ ] Optimize Celery task priorities
- [ ] Tune Celery worker concurrency
- [ ] Optimize Airflow DAG schedules
- [ ] Implement task result caching
- [ ] Increase job throughput from 100/min to 150/min

**Celery Optimization:**
```python
# Priority queue routing
CELERY_TASK_ROUTES = {
    'tasks.ml_prediction': {'queue': 'high_priority'},
    'tasks.stock_analysis': {'queue': 'medium_priority'},
    'tasks.data_ingestion': {'queue': 'low_priority'},
}

# Worker concurrency
celery -A tasks worker --concurrency=8 --pool=eventlet
```

**Labels:** `performance`, `celery`, `airflow`, `p1`
**Assignee:** Backend Team
**Dependencies:** None
**Estimate:** 10 hours

---

### Issue #14: Multi-Region Deployment
**Priority:** P2 | **Effort:** XL (120h) | **Milestone:** Q2 2026

**Description:**
Deploy platform to multiple regions for improved reliability and performance.

**Acceptance Criteria:**
- [ ] Set up PostgreSQL streaming replication (3 regions)
- [ ] Configure Redis cluster (6 nodes, 2 per region)
- [ ] Implement AWS ALB with health checks
- [ ] Configure auto-scaling policies
- [ ] Set up CloudFront CDN
- [ ] Achieve 99.9% uptime SLA

**Regions:**
- US-East (primary): Virginia
- US-West (secondary): Oregon
- Europe (tertiary): Ireland

**Labels:** `infrastructure`, `deployment`, `p2`, `multi-region`
**Assignee:** DevOps Team
**Dependencies:** Production stable
**Estimate:** 120 hours

---

### Issue #15: APM Integration
**Priority:** P2 | **Effort:** M (15h) | **Milestone:** Q2 2026

**Description:**
Integrate Application Performance Monitoring for observability and faster debugging.

**Acceptance Criteria:**
- [ ] Integrate DataDog or New Relic APM
- [ ] Set up distributed tracing (OpenTelemetry)
- [ ] Configure error tracking and alerting
- [ ] Create custom dashboards
- [ ] Reduce MTTR from 15min to 5min

**Metrics to Track:**
- Request latency (p50, p95, p99)
- Error rates by endpoint
- Database query performance
- Cache hit rates
- Background job queue depth

**Labels:** `monitoring`, `observability`, `p2`, `apm`
**Assignee:** DevOps Team
**Dependencies:** None
**Estimate:** 15 hours

---

## Epic 3: Security Hardening

**Goal:** Pass security audit, automate compliance
**Timeline:** 1 month
**Total Effort:** 32 hours

### Issue #16: OWASP Top 10 Validation -- PARTIALLY DONE
**Priority:** P1 | **Effort:** M (8h) | **Milestone:** Month 1
**Status:** PARTIALLY DONE (2026-02-08, commits `bca6756` + `c156cc4`)

**Description:**
Run comprehensive security scans against OWASP Top 10 vulnerabilities and remediate findings.

**Acceptance Criteria:**
- [ ] Run automated security scan (OWASP ZAP, Burp Suite)
- [x] Fix all HIGH and CRITICAL vulnerabilities (CSRF, JWT, rate limiting hardened)
- [x] Document remediation for each finding (commit messages)
- [x] Add security regression tests (test_csrf_protection.py, test_rate_limiter.py, test_security_modules.py)
- [ ] Pass security audit with 0 critical issues (formal scan pending)

**OWASP Top 10 Checklist:**
- [x] A01:2021 - Broken Access Control (JWT implemented, portfolio auth gates strengthened)
- [x] A02:2021 - Cryptographic Failures (HTTPS, encrypted storage, SSL enforced for DB)
- [ ] A03:2021 - Injection (SQL injection testing needed)
- [x] A04:2021 - Insecure Design (security architecture review done)
- [x] A05:2021 - Security Misconfiguration (SSL enforcement, rate limiting, CSRF hardening)
- [x] A06:2021 - Vulnerable Components (Dependabot enabled)
- [x] A07:2021 - Authentication Failures (JWT hardened, token expiry reduced, auth flow tests added)
- [x] A08:2021 - Data Integrity Failures (input validation implemented, CSRF timing-safe)
- [ ] A09:2021 - Logging Failures (audit trail review needed)
- [ ] A10:2021 - SSRF (endpoint validation needed)

**Remaining:** Formal OWASP ZAP/Burp scan, A03 injection testing, A09 logging audit, A10 SSRF validation.

**Labels:** `security`, `audit`, `p1`, `owasp`, `status:in-progress`
**Assignee:** Security Team
**Dependencies:** None
**Estimate:** 4 hours remaining

---

### Issue #17: Secret Management with Vault
**Priority:** P1 | **Effort:** M (8h) | **Milestone:** Month 1

**Description:**
Implement HashiCorp Vault for centralized secret management and key rotation.

**Acceptance Criteria:**
- [ ] Deploy HashiCorp Vault (Docker)
- [ ] Migrate all secrets from `.env` to Vault
- [ ] Implement automatic key rotation
- [ ] Add secret access auditing
- [ ] Zero hardcoded secrets in codebase

**Secrets to Migrate:**
- Database credentials
- API keys (Alpha Vantage, Finnhub, Polygon)
- JWT secret keys
- Redis password
- Email credentials

**Labels:** `security`, `secrets`, `p1`, `vault`
**Assignee:** DevOps Team
**Dependencies:** None
**Estimate:** 8 hours

---

### Issue #18: GDPR Workflow Automation
**Priority:** P1 | **Effort:** M (8h) | **Milestone:** Month 1

**Description:**
Automate GDPR compliance workflows to reduce manual effort and ensure compliance.

**Acceptance Criteria:**
- [ ] Automate data export workflow (email delivery within 24h)
- [ ] Create consent management UI
- [ ] Automate data deletion workflow (7-day confirmation)
- [ ] Add GDPR compliance monitoring dashboard
- [ ] Generate monthly compliance reports

**Workflows:**
1. **Data Export:**
   - User request -> Queue job -> Generate export -> Email download link
2. **Data Deletion:**
   - User request -> 7-day grace period -> Anonymize -> Delete -> Confirmation email
3. **Consent Management:**
   - UI for granular consent preferences
   - Audit trail for consent changes

**Labels:** `compliance`, `gdpr`, `automation`, `p1`
**Assignee:** Backend Team
**Dependencies:** Issue #5 complete
**Estimate:** 8 hours

---

### Issue #19: Intrusion Detection System
**Priority:** P2 | **Effort:** M (8h) | **Milestone:** Q2 2026

**Description:**
Implement real-time intrusion detection and anomaly detection system.

**Acceptance Criteria:**
- [ ] Set up anomaly detection (Elastic SIEM or similar)
- [ ] Configure real-time alerts (Slack, PagerDuty)
- [ ] Create security dashboards (Grafana)
- [x] Add IP-based rate limiting (advanced rate limiter hardened in commit `bca6756`)
- [ ] Implement automated blocking for suspicious activity

**Detection Rules:**
- Failed login attempts (5 in 5 minutes)
- Unusual API access patterns
- Suspicious data exports
- SQL injection attempts
- CSRF bypass attempts

**Labels:** `security`, `monitoring`, `p2`, `ids`
**Assignee:** Security Team
**Dependencies:** APM integration helpful
**Estimate:** 8 hours

---

## Epic 4: Advanced Analytics

**Goal:** 5 new analysis types, 10% accuracy improvement
**Timeline:** 2 months
**Total Effort:** 95 hours

### Issue #20: Options Analysis Feature
**Priority:** P2 | **Effort:** L (20h) | **Milestone:** Month 2

**Description:**
Implement comprehensive options analysis including Greeks, implied volatility, and strategy recommendations.

**Acceptance Criteria:**
- [ ] Calculate options Greeks (Delta, Gamma, Theta, Vega, Rho)
- [ ] Compute implied volatility (Black-Scholes model)
- [ ] Generate options strategy recommendations
- [ ] Add risk/reward analysis
- [ ] Create options chain visualization

**Features:**
- Greeks calculator for individual options
- Strategy builder (covered call, protective put, straddle, etc.)
- Max profit/loss analysis
- Break-even point calculation
- Probability of profit estimation

**Labels:** `feature`, `analytics`, `options`, `p2`
**Assignee:** Quant Team
**Dependencies:** None
**Estimate:** 20 hours

---

### Issue #21: Sector Rotation Analysis
**Priority:** P2 | **Effort:** M (15h) | **Milestone:** Month 2

**Description:**
Implement sector rotation analysis based on economic cycle detection.

**Acceptance Criteria:**
- [ ] Detect current economic cycle phase
- [ ] Track sector performance relative to S&P 500
- [ ] Generate sector rotation recommendations
- [ ] Calculate sector correlation matrix
- [ ] Add historical sector performance charts

**Economic Cycle Phases:**
- Early Expansion: Technology, Consumer Discretionary
- Mid Expansion: Industrials, Materials
- Late Expansion: Energy, Financials
- Contraction: Consumer Staples, Healthcare, Utilities

**Labels:** `feature`, `analytics`, `sectors`, `p2`
**Assignee:** Quant Team
**Dependencies:** None
**Estimate:** 15 hours

---

### Issue #22: Alternative Data Integration
**Priority:** P2 | **Effort:** L (20h) | **Milestone:** Month 2

**Description:**
Integrate alternative data sources for enhanced analysis and alpha generation.

**Acceptance Criteria:**
- [ ] Social sentiment analysis (Twitter, Reddit)
- [ ] Web scraping (Glassdoor employee reviews, Indeed job postings)
- [ ] Credit card transaction data integration
- [ ] Satellite imagery analysis (parking lot traffic)
- [ ] Add alternative data dashboard

**Data Sources:**
- Twitter API (sentiment)
- Reddit API (r/wallstreetbets, r/stocks)
- Glassdoor (employee satisfaction)
- Indeed (job posting trends)
- SpaceKnow (satellite imagery)

**Labels:** `feature`, `data`, `alternative-data`, `p2`
**Assignee:** Data Team
**Dependencies:** None
**Estimate:** 20 hours

---

### Issue #23: ESG Scoring System
**Priority:** P2 | **Effort:** M (10h) | **Milestone:** Month 2

**Description:**
Implement Environmental, Social, and Governance (ESG) scoring for sustainable investing.

**Acceptance Criteria:**
- [ ] Environmental score (carbon emissions, renewable energy)
- [ ] Social score (diversity, labor practices)
- [ ] Governance score (board composition, executive compensation)
- [ ] Overall ESG rating (0-100 scale)
- [ ] ESG-aware portfolio recommendations

**Data Sources:**
- MSCI ESG Ratings
- Sustainalytics
- CDP (Carbon Disclosure Project)
- Company sustainability reports

**Labels:** `feature`, `analytics`, `esg`, `p2`
**Assignee:** Quant Team
**Dependencies:** None
**Estimate:** 10 hours

---

### Issue #24: ML Model Ensemble
**Priority:** P2 | **Effort:** L (20h) | **Milestone:** Month 2

**Description:**
Combine LSTM, XGBoost, and Prophet models into ensemble for improved prediction accuracy.

**Acceptance Criteria:**
- [ ] Implement weighted voting mechanism
- [ ] Add model confidence tracking
- [ ] Automatic model selection based on stock characteristics
- [ ] Ensemble performance monitoring
- [ ] Achieve 10% accuracy improvement over single models

**Ensemble Strategy:**
```python
ensemble_prediction = (
    0.4 * lstm_prediction +
    0.3 * xgboost_prediction +
    0.3 * prophet_prediction
)

# Adaptive weighting based on recent performance
weights = calculate_performance_weights(
    lookback_period=30,
    metric='mape'
)
```

**Labels:** `ml`, `ensemble`, `p2`
**Assignee:** ML Team
**Dependencies:** All 3 models trained (complete)
**Estimate:** 20 hours

---

### Issue #25: Model Explainability with SHAP
**Priority:** P2 | **Effort:** M (10h) | **Milestone:** Month 2

**Description:**
Add SHAP (SHapley Additive exPlanations) for ML model interpretability.

**Acceptance Criteria:**
- [ ] Integrate SHAP library
- [ ] Generate feature importance charts
- [ ] Create prediction explanation UI
- [ ] Add SHAP waterfall plots
- [ ] Generate summary plots for model debugging

**Features:**
- Global feature importance
- Local prediction explanations
- Force plots for individual predictions
- Dependence plots for feature relationships
- Model debugging tools

**Labels:** `ml`, `explainability`, `p2`, `shap`
**Assignee:** ML Team
**Dependencies:** ML models operational
**Estimate:** 10 hours

---

## Epic 5: User Experience

**Goal:** 90% user satisfaction, mobile app launch
**Timeline:** 3 months
**Total Effort:** 68 hours

### Issue #26: Dashboard Redesign (Material Design 3)
**Priority:** P2 | **Effort:** L (20h) | **Milestone:** Month 3

**Description:**
Redesign main dashboard with modern Material Design 3 components and customizable layout.

**Acceptance Criteria:**
- [ ] Implement Material Design 3 theme
- [ ] Create customizable widget system
- [ ] Add drag-and-drop layout builder
- [ ] Ensure mobile responsiveness
- [ ] Achieve <1.5s page load time

**Widgets:**
- Portfolio performance chart
- Watchlist with real-time quotes
- Top gainers/losers
- Market overview
- Recent recommendations
- News feed
- Sector performance heatmap

**Labels:** `frontend`, `ui`, `redesign`, `p2`
**Assignee:** Frontend Team
**Dependencies:** None
**Estimate:** 20 hours

---

### Issue #27: TradingView Chart Integration
**Priority:** P2 | **Effort:** M (15h) | **Milestone:** Month 3

**Description:**
Integrate TradingView advanced charting library for professional-grade charts.

**Acceptance Criteria:**
- [ ] Integrate TradingView Charting Library
- [ ] Add 50+ technical indicators
- [ ] Implement drawing tools (trendlines, Fibonacci, etc.)
- [ ] Create chart templates
- [ ] Add chart sharing functionality

**Features:**
- Multiple chart types (candlestick, line, bar, Heikin Ashi)
- 100+ technical indicators
- Drawing tools (trendlines, channels, Fibonacci retracements)
- Chart templates and layouts
- Real-time data updates via WebSocket

**Labels:** `frontend`, `charting`, `p2`, `tradingview`
**Assignee:** Frontend Team
**Dependencies:** None
**Estimate:** 15 hours

---

### Issue #28: Real-time Collaboration Features
**Priority:** P2 | **Effort:** M (15h) | **Milestone:** Month 3

**Description:**
Add real-time collaboration features for users to share insights and watchlists.

**Acceptance Criteria:**
- [ ] Shared watchlists with live updates
- [ ] Comments and annotations on stocks/charts
- [ ] Activity feed showing friend activity
- [ ] User mentions and notifications
- [ ] Privacy controls (public/private/friends)

**Features:**
- WebSocket-based real-time updates
- Collaborative watchlists (edit permissions)
- Stock discussion threads
- Chart annotations with sharing
- @mention notifications

**Labels:** `frontend`, `collaboration`, `p2`, `websocket`
**Assignee:** Frontend Team
**Dependencies:** WebSocket infrastructure (complete)
**Estimate:** 15 hours

---

### Issue #29: React Native Mobile App
**Priority:** P2 | **Effort:** M (10h) | **Milestone:** Month 3

**Description:**
Create React Native mobile app with core feature parity.

**Acceptance Criteria:**
- [ ] Set up React Native project structure
- [ ] Implement authentication flow
- [ ] Portfolio view with real-time updates
- [ ] Stock search and analysis
- [ ] Push notifications for alerts
- [ ] Offline mode for cached data

**Platforms:**
- iOS (App Store)
- Android (Play Store)

**Core Features:**
- Login/register
- Portfolio dashboard
- Stock search
- Real-time quotes
- Watchlists
- Push notifications
- Price alerts

**Labels:** `mobile`, `react-native`, `p2`
**Assignee:** Mobile Team
**Dependencies:** API stable
**Estimate:** 10 hours (foundation only, full parity is 80h)

---

### Issue #30: Interactive Onboarding Tutorial
**Priority:** P2 | **Effort:** M (8h) | **Milestone:** Month 3

**Description:**
Create interactive onboarding tutorial to improve user activation and retention.

**Acceptance Criteria:**
- [ ] Step-by-step guided tour
- [ ] Feature highlights with tooltips
- [ ] Video walkthroughs (2-3 minutes each)
- [ ] Progress tracking (50%, 75%, 100% completion)
- [ ] Achieve 80% completion rate

**Tutorial Steps:**
1. Welcome & account setup
2. Create first portfolio
3. Search for stocks
4. Add to watchlist
5. View analysis & recommendations
6. Set price alerts
7. Explore advanced features

**Labels:** `frontend`, `onboarding`, `p2`, `ux`
**Assignee:** Frontend Team
**Dependencies:** None
**Estimate:** 8 hours

---

## Epic 6: Enterprise Features

**Goal:** Multi-tenant ready, 10 enterprise clients
**Timeline:** 6 months
**Total Effort:** 400 hours

### Issue #31: Multi-Tenant Architecture
**Priority:** P3 | **Effort:** XL (200h) | **Milestone:** Q4 2026

**Description:**
Implement multi-tenant architecture for white-label enterprise deployments.

**Acceptance Criteria:**
- [ ] Database-per-tenant isolation
- [ ] Tenant-specific encryption keys
- [ ] Custom branding per tenant
- [ ] SSO integration (SAML, OAuth)
- [ ] Tenant admin console
- [ ] Support 100+ concurrent tenants

**Architecture:**
- Shared application layer
- Isolated database per tenant
- Tenant routing middleware
- Custom domain support
- Feature flags per tenant

**Labels:** `architecture`, `enterprise`, `multi-tenant`, `p3`
**Assignee:** Architecture Team
**Dependencies:** Production stable
**Estimate:** 200 hours

---

### Issue #32: Enterprise Admin Console
**Priority:** P3 | **Effort:** L (40h) | **Milestone:** Q4 2026

**Description:**
Build comprehensive admin console for enterprise user and usage management.

**Acceptance Criteria:**
- [ ] User management (create, edit, delete, roles)
- [ ] Usage analytics dashboard
- [ ] Billing and subscription management
- [ ] Feature flag controls
- [ ] Audit log viewer
- [ ] Support ticket system integration

**Features:**
- User provisioning (manual, CSV import, SCIM)
- Role-based access control (RBAC)
- Usage metrics (API calls, storage, compute)
- Cost allocation by department
- License management

**Labels:** `feature`, `enterprise`, `admin`, `p3`
**Assignee:** Backend Team
**Dependencies:** Multi-tenant architecture
**Estimate:** 40 hours

---

### Issue #33: SSO Integration (SAML, OAuth)
**Priority:** P3 | **Effort:** M (20h) | **Milestone:** Q4 2026

**Description:**
Implement Single Sign-On with enterprise identity providers.

**Acceptance Criteria:**
- [ ] SAML 2.0 support
- [ ] OAuth 2.0 / OpenID Connect
- [ ] Azure AD integration
- [ ] Okta integration
- [ ] Google Workspace integration
- [ ] Just-in-time (JIT) user provisioning

**Identity Providers:**
- Azure Active Directory
- Okta
- OneLogin
- Google Workspace
- PingFederate
- Generic SAML 2.0

**Labels:** `feature`, `auth`, `sso`, `p3`
**Assignee:** Backend Team
**Dependencies:** Multi-tenant architecture
**Estimate:** 20 hours

---

### Issue #34: Bloomberg Terminal Integration
**Priority:** P3 | **Effort:** L (60h) | **Milestone:** Q4 2026

**Description:**
Integrate with Bloomberg Terminal for institutional-grade data and analytics.

**Acceptance Criteria:**
- [ ] Bloomberg API integration
- [ ] Real-time data feed
- [ ] Historical data sync (10+ years)
- [ ] Bloomberg news integration
- [ ] Corporate actions data
- [ ] Earnings estimates

**Data Types:**
- Real-time quotes (Level 2 depth)
- Historical OHLCV data
- Fundamental data (income statement, balance sheet)
- Analyst estimates
- Corporate actions (splits, dividends)
- News and research

**Labels:** `feature`, `integration`, `bloomberg`, `p3`
**Assignee:** Data Team
**Dependencies:** Enterprise licensing
**Estimate:** 60 hours

---

### Issue #35: AI-Powered Natural Language Queries
**Priority:** P3 | **Effort:** XL (80h) | **Milestone:** Q4 2026

**Description:**
Integrate GPT-4 for natural language analysis queries and report generation.

**Acceptance Criteria:**
- [ ] GPT-4 API integration
- [ ] Natural language query parsing
- [ ] Automated research report generation
- [ ] Earnings call summarization
- [ ] Market commentary generation
- [ ] Conversational interface

**Use Cases:**
- "Show me undervalued tech stocks with strong fundamentals"
- "Summarize Apple's latest earnings call"
- "Compare Tesla and Rivian financials"
- "Generate investment thesis for NVDA"
- "What are the best dividend stocks in healthcare?"

**Labels:** `feature`, `ai`, `nlp`, `p3`, `gpt-4`
**Assignee:** ML Team
**Dependencies:** OpenAI API access
**Estimate:** 80 hours

---

## Epic 7: Market Expansion

**Goal:** 5 international exchanges, cryptocurrency support
**Timeline:** 9 months
**Total Effort:** 540 hours

### Issue #36: European Markets Integration
**Priority:** P3 | **Effort:** L (60h) | **Milestone:** Q1 2027

**Description:**
Integrate major European stock exchanges for international coverage.

**Acceptance Criteria:**
- [ ] LSE (London Stock Exchange) - 2,000+ stocks
- [ ] Euronext (Paris, Amsterdam, Brussels) - 1,500+ stocks
- [ ] Deutsche Borse (Frankfurt) - 1,000+ stocks
- [ ] SIX Swiss Exchange - 300+ stocks
- [ ] Currency conversion (EUR, GBP, CHF)
- [ ] Trading hours handling (different time zones)

**Technical Requirements:**
- Multi-currency support
- Exchange-specific trading rules
- Holiday calendars
- Market hours handling
- Regulatory compliance (MiFID II)

**Labels:** `feature`, `international`, `europe`, `p3`
**Assignee:** Data Team
**Dependencies:** None
**Estimate:** 60 hours

---

### Issue #37: Asian Markets Integration
**Priority:** P3 | **Effort:** XL (80h) | **Milestone:** Q1 2027

**Description:**
Integrate major Asian stock exchanges for global coverage.

**Acceptance Criteria:**
- [ ] Tokyo Stock Exchange - 3,800+ stocks
- [ ] Hong Kong Stock Exchange - 2,500+ stocks
- [ ] Shanghai Stock Exchange - 2,000+ stocks
- [ ] Singapore Exchange - 700+ stocks
- [ ] Language localization (Chinese, Japanese)
- [ ] Regulatory compliance per country

**Technical Requirements:**
- Multi-language support (Chinese characters, Japanese kanji)
- Currency conversion (JPY, HKD, CNY, SGD)
- Time zone handling
- Cultural considerations
- Local regulatory compliance

**Labels:** `feature`, `international`, `asia`, `p3`
**Assignee:** Data Team
**Dependencies:** None
**Estimate:** 80 hours

---

### Issue #38: Cryptocurrency Trading
**Priority:** P3 | **Effort:** L (40h) | **Milestone:** Q1 2027

**Description:**
Add cryptocurrency trading and portfolio tracking.

**Acceptance Criteria:**
- [ ] Bitcoin (BTC) and Ethereum (ETH) support
- [ ] 20+ top cryptocurrencies
- [ ] DeFi protocol integration (Uniswap, Aave, Compound)
- [ ] NFT tracking and valuation
- [ ] Crypto portfolio analytics
- [ ] Tax reporting (cost basis, realized gains)

**Features:**
- Real-time crypto prices
- Historical price charts
- On-chain analytics
- DeFi yield tracking
- NFT portfolio valuation
- Tax loss harvesting

**Data Sources:**
- CoinGecko API
- CoinMarketCap API
- Blockchain nodes (Infura)
- The Graph Protocol (DeFi data)
- OpenSea API (NFT data)

**Labels:** `feature`, `crypto`, `p3`
**Assignee:** Crypto Team
**Dependencies:** None
**Estimate:** 40 hours

---

### Issue #39: Algorithmic Trading Platform
**Priority:** P3 | **Effort:** XL (160h) | **Milestone:** Q3 2026

**Description:**
Build algorithmic trading platform with strategy builder, backtesting, and live trading.

**Acceptance Criteria:**
- [ ] Visual strategy builder (drag-and-drop)
- [ ] Backtesting framework (10+ years historical data)
- [ ] Paper trading mode
- [ ] Live trading with broker integration
- [ ] Risk management controls
- [ ] Strategy marketplace

**Strategy Builder Features:**
- Technical indicator library (50+)
- Fundamental filters
- Machine learning models
- Custom Python code
- Event-driven strategies
- Market microstructure strategies

**Broker Integrations:**
- Alpaca (commission-free)
- Interactive Brokers
- TD Ameritrade
- Robinhood

**Labels:** `feature`, `algo-trading`, `p3`
**Assignee:** Quant Team
**Dependencies:** Legal review, compliance
**Estimate:** 160 hours

---

### Issue #40: Institutional Portfolio Management
**Priority:** P3 | **Effort:** XL (200h) | **Milestone:** Q1 2027

**Description:**
Build institutional-grade portfolio management system for professional asset managers.

**Acceptance Criteria:**
- [ ] Multi-portfolio support (100+ portfolios per firm)
- [ ] Attribution analysis (Brinson-Fachler model)
- [ ] Risk analytics (VaR, CVaR, stress testing)
- [ ] Compliance reporting (SEC Form PF, Form ADV)
- [ ] Trade order management (OMS)
- [ ] Best execution analytics (TCA)

**Features:**
- Portfolio construction and rebalancing
- Performance attribution
- Risk decomposition
- Compliance monitoring
- Client reporting
- Custodian integration

**Target Users:**
- Registered Investment Advisors (RIAs)
- Hedge funds
- Family offices
- Wealth management firms

**Labels:** `feature`, `institutional`, `pms`, `p3`
**Assignee:** Enterprise Team
**Dependencies:** Multi-tenant architecture
**Estimate:** 200 hours

---

## Audit Follow-up Issues (Discovered 2026-02-08)

### Issue #41: Replace datetime.utcnow() with Timezone-Aware UTC -- DONE
**Priority:** P1 | **Effort:** S (4h) | **Milestone:** Week 1
**Status:** DONE (2026-02-08, commit `d23c3e5`)

**Description:**
`datetime.utcnow()` is deprecated in Python 3.12+. All 100+ backend modules using it were migrated to `datetime.now(timezone.utc)` for correct timezone-aware datetime handling.

**Acceptance Criteria:**
- [x] Replace all `datetime.utcnow()` calls with `datetime.now(timezone.utc)`
- [x] Update imports to include `timezone` from `datetime`
- [x] Verify no regressions in timestamp-dependent logic
- [x] Cover analytics, API, ML, utils, tasks, security, monitoring modules

**Files Changed:** 100+ across `backend/analytics/`, `backend/api/`, `backend/ml/`, `backend/utils/`, `backend/tasks/`, `backend/security/`, `backend/monitoring/`, `backend/repositories/`, `backend/compliance/`, `backend/streaming/`, `scripts/`, `data_pipelines/`

**Labels:** `refactor`, `p1`, `python`, `deprecation`, `status:done`

---

### Issue #42: Integration and Security Test Suites -- DONE
**Priority:** P1 | **Effort:** S (4h) | **Milestone:** Week 1
**Status:** DONE (2026-02-08, commit `c156cc4`)

**Description:**
Add 10 new test suites to improve integration and security test coverage across all major API routers and security modules.

**Acceptance Criteria:**
- [x] `test_analysis_router.py` - Analysis endpoint validation (590 lines)
- [x] `test_auth_flow_complete.py` - End-to-end auth flow testing (744 lines)
- [x] `test_health_router.py` - Health check endpoint tests (517 lines)
- [x] `test_news_router.py` - News API endpoint tests (378 lines)
- [x] `test_recommendations_router.py` - Recommendation engine tests (622 lines)
- [x] `test_settings_router.py` - Settings management tests (850 lines)
- [x] `test_stocks_router.py` - Stock data endpoint tests (836 lines)
- [x] `test_websocket_router.py` - WebSocket connection tests (596 lines)
- [x] `test_rate_limiter.py` - Rate limiting behavior tests (1036 lines)
- [x] `test_security_modules.py` - Security module unit tests (811 lines)
- [x] Update existing test fixtures for datetime and security changes

**Labels:** `testing`, `p1`, `integration`, `security`, `status:done`

---

### Issue #43: CI/CD Coverage Reporting and TypeScript Audit -- DONE
**Priority:** P1 | **Effort:** S (2h) | **Milestone:** Week 1
**Status:** DONE (2026-02-08, commit `1e3e73e`)

**Description:**
Add test coverage reporting to CI/CD pipelines and perform TypeScript strict mode audit for the frontend.

**Acceptance Criteria:**
- [x] Add coverage reporting to `.github/workflows/ci.yml`
- [x] Add coverage reporting to `.github/workflows/comprehensive-testing.yml`
- [x] Update `frontend/web/tsconfig.json` with stricter TypeScript checks
- [x] Audit TypeScript strict mode findings (31 errors in 10 files documented)
- [x] Update `pyproject.toml` and `pytest.ini` configuration
- [x] Update codebase architecture map

**Deliverables:**
- `docs/TYPESCRIPT_STRICT_MODE_AUDIT.md` - Detailed audit report
- Updated CI/CD workflows with coverage gates

**Labels:** `ci`, `testing`, `frontend`, `typescript`, `status:done`

---

### Issue #44: TypeScript Strict Mode Remediation
**Priority:** P2 | **Effort:** S (4h) | **Milestone:** Month 1

**Description:**
Fix the 31 TypeScript strict mode errors identified in the audit (see `docs/TYPESCRIPT_STRICT_MODE_AUDIT.md`). Currently `"strict": false` in `frontend/web/tsconfig.json`. Errors are concentrated in 10 of 73 TypeScript files (14% of codebase).

**Acceptance Criteria:**
- [ ] Fix 31 type errors across 10 files
- [ ] Enable `"strict": true` in `tsconfig.json`
- [ ] Add explicit types for all implicit `any` parameters
- [ ] Handle all nullable value cases properly
- [ ] Verify frontend builds cleanly with strict mode enabled

**Error Categories:**
- Implicit `any` types
- Nullable value mishandling
- Missing return type annotations

**Labels:** `frontend`, `typescript`, `p2`, `code-quality`
**Assignee:** Frontend Team
**Dependencies:** None
**Estimate:** 4 hours

---

## Issue Creation Instructions

### Creating Issues in GitHub

1. **Navigate to Repository:**
   ```
   https://github.com/your-org/investment-analysis-platform/issues
   ```

2. **Use Issue Templates:**
   - Create templates in `.github/ISSUE_TEMPLATE/`
   - Separate templates for: `bug.md`, `feature.md`, `epic.md`

3. **Issue Title Format:**
   ```
   [EPIC] Epic Name
   [FEATURE] Feature description
   [BUG] Bug description
   ```

4. **Required Fields:**
   - Title
   - Description
   - Acceptance Criteria
   - Priority (P0-P3)
   - Effort (S/M/L/XL)
   - Labels
   - Milestone
   - Estimate (hours)

5. **Link to Epic:**
   ```markdown
   Part of Epic #X: [Epic Name]
   ```

### Labels to Create

**Priority:**
- `p0` - Critical, blocking
- `p1` - High priority
- `p2` - Medium priority
- `p3` - Low priority

**Type:**
- `bug` - Something isn't working
- `feature` - New feature or request
- `enhancement` - Improvement to existing feature
- `refactor` - Code quality improvement

**Component:**
- `backend` - Backend/API
- `frontend` - UI/UX
- `ml` - Machine learning
- `infrastructure` - DevOps, deployment
- `testing` - Test coverage, QA
- `security` - Security-related
- `compliance` - GDPR, SEC compliance

**Effort:**
- `effort:S` - Small (1-4 hours)
- `effort:M` - Medium (5-15 hours)
- `effort:L` - Large (16-40 hours)
- `effort:XL` - Extra Large (40+ hours)

**Status:**
- `status:ready` - Ready to work on
- `status:in-progress` - Currently being worked on
- `status:blocked` - Blocked by dependencies
- `status:needs-review` - Needs code review
- `status:done` - Completed

---

## Roadmap View

### Sprint 1 (Week 1): Quick Wins -- PARTIALLY COMPLETE
**Focus:** Integration tests passing
- ~~Issue #1: CSRF Config (4h)~~ DONE
- Issue #2: Agent Analysis (2h)
- Issue #3: ML Predictions (2h)
- Issue #4: Stock Search & Alerts (4h)
- Issue #5: GDPR Endpoints (8h)
- ~~Issue #41: datetime refactor (4h)~~ DONE
- ~~Issue #42: Test suites (4h)~~ DONE
- ~~Issue #43: CI/CD coverage (2h)~~ DONE
**Remaining:** 16 hours | **Completed:** 14 hours

### Sprint 2 (Week 2): Service Layer
**Focus:** Business logic implementation
- Issue #6: Service Implementation (24h)
- Issue #7: Middleware Optimization (8h remaining)
**Total:** 32 hours

### Sprint 3 (Month 1): Coverage & Performance
**Focus:** Quality and speed
- Issue #8: Analytics Coverage (8h)
- Issue #9: E2E Tests (10h)
- Issue #10: DB Optimization (10h)
- Issue #11: Cache Tuning (10h)
- Issue #12: API Optimization (10h)
- Issue #13: Job Optimization (10h)
- Issue #44: TypeScript Strict Mode (4h)
**Total:** 62 hours

### Sprint 4 (Month 1): Security
**Focus:** Hardening and compliance
- Issue #16: OWASP Audit (4h remaining)
- Issue #17: Vault Setup (8h)
- Issue #18: GDPR Automation (8h)
**Total:** 20 hours

---

**Total Backlog:** 44 issues | 1,416 hours | 7 epics + audit follow-ups
**Completed:** Issues #1, #41, #42, #43 (14 hours)
**In Progress:** Issues #7, #16 (12 hours remaining)
**Immediate Priority:** Issues #2-6 (40 hours over 2 weeks)
**Next Quarter:** Issues #8-25, #44 (289 hours over 3 months)
**Next Half:** Issues #26-40 (1,045 hours over 6 months)

---

**Document Version:** 1.1.0
**Last Updated:** 2026-02-08
**Status:** UPDATED POST-AUDIT
