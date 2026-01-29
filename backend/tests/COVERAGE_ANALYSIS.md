# Test Coverage Analysis - API Standardization

**Assessment Date:** January 27, 2026
**Current Coverage:** 30% (will drop to 5% when tests fail)
**Target Coverage:** 80%+
**Gap:** 50+ percentage points

---

## Coverage by Router

### 1. admin.py
**Status:** ❌ NO TESTS
**Coverage:** 0%
**Risk:** CRITICAL

#### What's Tested:
- Nothing

#### What Should Be Tested (20+ tests needed):
1. Health endpoints (3 tests)
2. System status endpoints (3 tests)
3. User management endpoints (5 tests)
4. Configuration endpoints (4 tests)
5. Analytics endpoints (3 tests)
6. Error handling (2 tests)

#### Example Tests Needed:
```python
# Health and Status
- test_get_system_health
- test_get_service_status
- test_health_returns_valid_structure

# User Management
- test_create_user
- test_get_user
- test_list_users_pagination
- test_update_user
- test_delete_user

# Configuration
- test_get_config_section
- test_update_config_section
- test_config_requires_admin
- test_invalid_config_section

# Error Cases
- test_get_nonexistent_user
- test_unauthorized_config_access
```

**Effort:** 8-10 hours for full coverage

---

### 2. agents.py
**Status:** ❌ NO TESTS
**Coverage:** 0%
**Risk:** CRITICAL

#### What's Tested:
- Nothing

#### What Should Be Tested (15+ tests needed):
1. Stock analysis endpoints (4 tests)
2. Batch analysis endpoints (3 tests)
3. Budget status endpoints (3 tests)
4. Agent capabilities (2 tests)
5. Error handling (3 tests)

#### Example Tests Needed:
```python
# Agent Analysis
- test_analyze_stock_success
- test_analyze_stock_with_agents_forced
- test_analyze_stock_invalid_ticker
- test_analyze_stock_timeout

# Batch Analysis
- test_batch_analyze_success
- test_batch_analyze_concurrent_limit
- test_batch_analyze_invalid_tickers

# Budget Status
- test_get_budget_status
- test_get_budget_status_low_balance
- test_get_budget_status_no_balance

# Agent Capabilities
- test_get_agent_capabilities
- test_get_available_analysts

# Errors
- test_analyze_stock_cost_exceeded
- test_analyze_stock_engine_error
- test_analyze_stock_timeout_handling
```

**Effort:** 6-8 hours for full coverage

---

### 3. thesis.py
**Status:** ⚠️ PARTIAL TESTS (BROKEN)
**Coverage:** 50% (currently failing)
**Risk:** HIGH

#### Currently Tested (14 tests):
1. Create thesis success - **BROKEN**
2. Create thesis missing fields - **BROKEN**
3. Create thesis invalid stock - **BROKEN**
4. Create duplicate thesis - **BROKEN**
5. Get thesis by ID - **BROKEN**
6. Get thesis by stock ID - **BROKEN**
7. Get thesis not found - **BROKEN**
8. List user theses - **BROKEN**
9. List theses pagination - **BROKEN**
10. Update thesis - **BROKEN**
11. Update thesis not owned - **BROKEN**
12. Delete thesis - **BROKEN**
13. Delete thesis not owned - **BROKEN**
14. Thesis requires authentication - **BROKEN**

#### Additional Tests Needed (10+ tests):
```python
# Validation
- test_create_thesis_invalid_target_price
- test_create_thesis_missing_optional_fields
- test_update_thesis_missing_fields

# Authorization
- test_create_thesis_as_anonymous_user
- test_update_other_users_thesis
- test_delete_other_users_thesis

# Edge Cases
- test_thesis_with_very_long_content
- test_thesis_with_special_characters
- test_list_theses_empty_user

# Concurrency
- test_concurrent_thesis_creation
- test_concurrent_thesis_updates
```

**Effort:** 1 hour (fixes) + 2 hours (new tests)

---

### 4. gdpr.py
**Status:** ❌ NO TESTS
**Coverage:** 0%
**Risk:** CRITICAL

#### What's Tested:
- Nothing

#### What Should Be Tested (20+ tests needed):
1. Data export endpoints (3 tests)
2. Data deletion endpoints (4 tests)
3. Consent management (3 tests)
4. Right to be forgotten (3 tests)
5. Privacy reports (2 tests)
6. Error handling (5 tests)

#### Example Tests Needed:
```python
# Data Export
- test_export_personal_data
- test_export_data_pagination
- test_export_data_format_options

# Data Deletion
- test_delete_personal_data
- test_delete_with_confirmation_required
- test_delete_cannot_restore
- test_delete_anonymize_option

# Consent
- test_get_consent_status
- test_update_consent_preferences
- test_consent_audit_trail

# Right to be Forgotten
- test_request_deletion
- test_deletion_deadline
- test_deletion_status

# Privacy Reports
- test_get_privacy_report
- test_get_privacy_report_includes_all_data

# Errors
- test_export_nonexistent_user
- test_delete_with_valid_consent_only
- test_deletion_request_already_pending
- test_export_requires_authentication
- test_cannot_delete_if_disputed
```

**Effort:** 10-12 hours for full coverage

---

### 5. watchlist.py
**Status:** ⚠️ PARTIAL TESTS
**Coverage:** 60%
**Risk:** MEDIUM

#### Currently Tested:
1. Unit tests for repository (8 tests) - ✓ PASSING
2. API tests (mixed status)

#### API Tests Status:
- GET endpoints - May be broken
- POST endpoints - May be broken
- PUT endpoints - May be broken
- DELETE endpoints - May be broken

#### Additional Tests Needed (10+ tests):
```python
# Pagination
- test_list_watchlists_pagination
- test_list_items_pagination
- test_pagination_boundary_conditions

# Permissions
- test_view_public_watchlist
- test_cannot_modify_other_users_watchlist
- test_cannot_view_private_watchlist_of_others

# Item Management
- test_add_duplicate_item
- test_add_item_target_price_update
- test_remove_nonexistent_item

# Search and Filter
- test_search_watchlist_items
- test_filter_items_by_price_range
```

**Effort:** 1 hour (fixes) + 2 hours (new tests)

---

### 6. cache_management.py
**Status:** ⚠️ INDIRECT TESTS
**Coverage:** 40%
**Risk:** MEDIUM

#### Currently Tested:
- Indirectly through cache integration tests
- No dedicated endpoint tests

#### What Should Be Tested (15+ tests):
1. Cache statistics endpoints (3 tests)
2. Cache operation endpoints (4 tests)
3. Cache invalidation endpoints (3 tests)
4. Cache health endpoints (3 tests)
5. Error handling (2 tests)

#### Example Tests Needed:
```python
# Statistics
- test_get_cache_stats
- test_get_cache_stats_returns_metrics
- test_cache_stats_hit_rate_calculation

# Operations
- test_clear_cache
- test_invalidate_specific_key
- test_batch_invalidate_keys
- test_cache_operation_requires_admin

# Health
- test_cache_health_check
- test_cache_health_degraded
- test_cache_health_unavailable

# Invalidation
- test_invalidate_by_pattern
- test_invalidate_by_tag

# Errors
- test_clear_cache_redis_error
- test_cache_operation_timeout
```

**Effort:** 4-6 hours for full coverage

---

### 7. monitoring.py
**Status:** ❌ NO TESTS
**Coverage:** 0%
**Risk:** CRITICAL

#### What's Tested:
- Nothing

#### What Should Be Tested (15+ tests needed):
1. Metrics endpoints (4 tests)
2. Performance endpoints (3 tests)
3. Error tracking endpoints (3 tests)
4. Alert endpoints (3 tests)
5. Error handling (2 tests)

#### Example Tests Needed:
```python
# Metrics
- test_get_api_metrics
- test_get_database_metrics
- test_get_cache_metrics
- test_get_metrics_returns_structure

# Performance
- test_get_response_time_metrics
- test_get_throughput_metrics
- test_performance_thresholds

# Error Tracking
- test_get_error_logs
- test_get_error_logs_filtered
- test_get_error_rate

# Alerts
- test_create_alert
- test_list_alerts
- test_acknowledge_alert

# Errors
- test_metrics_requires_authentication
- test_invalid_metric_type
```

**Effort:** 6-8 hours for full coverage

---

## Coverage Summary Table

| Router | Current | Target | Gap | Tests Needed | Effort |
|--------|---------|--------|-----|--------------|--------|
| admin | 0% | 80% | 80% | 20 | 8-10h |
| agents | 0% | 80% | 80% | 15 | 6-8h |
| thesis | 50% | 80% | 30% | 10 | 1h + 2h |
| gdpr | 0% | 80% | 80% | 20 | 10-12h |
| watchlist | 60% | 80% | 20% | 10 | 1h + 2h |
| cache_mgmt | 40% | 80% | 40% | 15 | 4-6h |
| monitoring | 0% | 80% | 80% | 15 | 6-8h |
| **TOTAL** | **30%** | **80%** | **50%** | **105** | **38-50h** |

---

## Phased Implementation Plan

### Phase 1: CRITICAL (3-4 hours) - FIX BROKEN TESTS
**Goal:** Stop the bleeding - get tests passing again

**Tasks:**
1. Fix `test_thesis_api.py` assertions (1 hour)
2. Fix integration test assertions (1 hour)
3. Fix `test_watchlist.py` assertions (0.5 hour)
4. Add helper functions to conftest.py (0.5 hour)
5. Verify all Phase 1 tests pass (0.5 hour)

**Result:** 14 + 5-10 + 5 = 24 tests passing (+25%)

---

### Phase 2: HIGH PRIORITY (6.5 hours) - IMPROVE COVERAGE
**Goal:** Reach 50% coverage with focus on critical paths

**Routers in Priority Order:**
1. thesis.py - Add 10 tests (2 hours)
2. watchlist.py - Add 10 tests (2 hours)
3. admin.py - Add 10 core tests (2 hours)
4. agents.py - Add 8 tests (0.5 hour planning)

**Result:** +38 tests, coverage 50%

---

### Phase 3: MEDIUM PRIORITY (15+ hours) - ACHIEVE 80%
**Goal:** Comprehensive test coverage for all routers

**Routers:**
1. admin.py - Complete 20 tests (4 hours)
2. agents.py - Complete 15 tests (3 hours)
3. gdpr.py - Create 20 tests (5 hours)
4. cache_mgmt.py - Create 15 tests (3 hours)
5. monitoring.py - Create 15 tests (3 hours)

**Result:** +85 tests, coverage 80%

---

## Test Priority Matrix

### High Impact, Low Effort
- ✓ Fix thesis.py assertions (1 hour, +14 tests)
- ✓ Fix watchlist.py assertions (0.5 hour, +8 tests)
- ✓ Add thesis.py edge cases (2 hours, +10 tests)
- ✓ Add watchlist.py edge cases (2 hours, +10 tests)

### High Impact, Medium Effort
- ✓ Create admin.py tests (8 hours, +20 tests)
- ✓ Create agents.py tests (6 hours, +15 tests)

### High Impact, High Effort
- ✓ Create gdpr.py tests (10 hours, +20 tests)
- ✓ Create monitoring.py tests (6 hours, +15 tests)
- ✓ Create cache_mgmt.py tests (4 hours, +15 tests)

---

## Coverage Targets by Router

### Minimum (50% coverage)
```
admin:          10 tests
agents:         7 tests
thesis:         7 tests (currently have 14, fix + add 3)
gdpr:           10 tests
watchlist:      5 tests (currently have 13, fix + add 2)
cache_mgmt:     7 tests
monitoring:     7 tests
TOTAL:         53 tests
```

### Target (80% coverage)
```
admin:          20 tests
agents:         15 tests
thesis:         14 tests (fix + add 10)
gdpr:           20 tests
watchlist:      18 tests (fix + add 10)
cache_mgmt:     15 tests
monitoring:     15 tests
TOTAL:         117 tests
```

---

## Testing Best Practices for New Tests

### 1. Always Test Happy Path First
```python
# Test success case before error cases
@pytest.mark.asyncio
async def test_create_thesis_success(self, client, auth_headers):
    """Happy path - successful creation"""
    ...
```

### 2. Use Fixtures for Common Data
```python
@pytest.fixture
async def test_thesis(db_session):
    """Reusable thesis for multiple tests"""
    ...
```

### 3. Test One Thing Per Test
```python
# Good: Each test has one assertion focus
def test_create_thesis_requires_auth  # Auth only
def test_create_thesis_validates_price  # Validation only
def test_create_thesis_sets_version  # Business logic only
```

### 4. Use Parametrized Tests for Similar Cases
```python
@pytest.mark.parametrize("invalid_price", [-100, 0, None, "abc"])
@pytest.mark.asyncio
async def test_create_thesis_invalid_prices(self, client, auth_headers, invalid_price):
    """Test various invalid price values"""
    ...
```

### 5. Test Error Messages
```python
# Don't just check status code
assert response.status_code == 400

# Also verify the error message is helpful
response_json = response.json()
assert "stock_id" in response_json["error"].lower()
```

---

## Integration Test Coverage

### Database Layer
- [ ] Create/Read/Update/Delete operations
- [ ] Relationship handling
- [ ] Cascade operations
- [ ] Transaction rollback

### API Layer
- [ ] Request validation
- [ ] Response wrapper structure
- [ ] Status codes
- [ ] Error messages

### Authorization Layer
- [ ] Authentication required
- [ ] Role-based access
- [ ] Resource ownership
- [ ] Admin operations

### External Services
- [ ] Timeout handling
- [ ] Retry logic
- [ ] Fallback behavior
- [ ] Error propagation

---

## Success Criteria

✓ Phase 1: All 24 tests passing (within 4 hours)
✓ Phase 2: 50 tests total (within 10 hours)
✓ Phase 3: 80+ tests total with 80% coverage (within 48 hours)

---

## Effort Estimation Breakdown

### Phase 1: Fix Broken Tests
- thesis.py assertions: 1 hour
- integration tests: 1 hour
- watchlist.py: 0.5 hour
- conftest helpers: 0.5 hour
- **Total: 3 hours**

### Phase 2: Add Core Tests
- thesis.py new tests: 2 hours
- watchlist.py new tests: 2 hours
- admin.py core tests: 2 hours
- agents.py planning: 0.5 hour
- **Total: 6.5 hours**

### Phase 3: Full Coverage
- admin.py: 4 hours
- agents.py: 3 hours
- gdpr.py: 5 hours
- cache_mgmt.py: 3 hours
- monitoring.py: 3 hours
- **Total: 18 hours**

**Grand Total: ~27.5 hours for full 80% coverage**

---

## Metrics Dashboard

```
Coverage Timeline:
Phase 1 (3h):   0% → 25%
Phase 2 (6.5h): 25% → 50%
Phase 3 (18h):  50% → 80%

Tests Timeline:
Phase 1: 24 tests
Phase 2: 62 tests
Phase 3: 117 tests (+46% improvement)

ROI:
Phase 1: 3 hours blocks all development ❌ DO FIRST
Phase 2: 6.5 hours for 38 tests (good value)
Phase 3: 18 hours for 55 tests (slower but complete)
```

