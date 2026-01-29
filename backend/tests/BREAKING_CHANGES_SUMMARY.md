# API Standardization - Breaking Changes Summary

**Migration Date:** January 27, 2026
**Status:** REQUIRES TEST UPDATES
**Severity:** HIGH - All API response assertions will fail

---

## Quick Facts

- **7 routers migrated** to use `ApiResponse` wrapper
- **26+ tests broken** (need updates within assertions)
- **Fix time estimate:** 3-4 hours (Phase 1: CRITICAL)
- **Total time to 80% coverage:** 15+ hours
- **Rollback time if needed:** 2 hours

---

## What Changed

### Response Structure

```
BEFORE: Direct data returned
GET /api/v1/thesis/1
{
  "id": 1,
  "name": "Test",
  "stock_id": 100
}

AFTER: Wrapped in ApiResponse
GET /api/v1/thesis/1
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Test",
    "stock_id": 100
  },
  "error": null,
  "meta": null
}
```

---

## Affected Routers

| Router | Affected | Test Status |
|--------|----------|-------------|
| `admin.py` | Yes | No tests |
| `agents.py` | Yes | No tests |
| `thesis.py` | Yes | 14 broken tests |
| `gdpr.py` | Yes | No tests |
| `watchlist.py` | Yes | 8 tests (mixed) |
| `cache_management.py` | Yes | Indirect |
| `monitoring.py` | Yes | No tests |

---

## Breaking Changes

### 1. Single Item Endpoints
```python
# ❌ BROKEN
response = await client.get("/api/thesis/1")
data = response.json()
thesis_id = data["id"]

# ✓ FIXED
response = await client.get("/api/thesis/1")
json = response.json()
data = json["data"]
thesis_id = data["id"]
```

### 2. List Endpoints
```python
# ❌ BROKEN
response = await client.get("/api/theses/")
items = response.json()
count = len(items)

# ✓ FIXED
response = await client.get("/api/theses/")
json = response.json()
items = json["data"]
count = len(items)
total = json["meta"]["total"]
```

### 3. Create/Update Endpoints
```python
# ❌ BROKEN
response = await client.post("/api/thesis/", json=data)
created = response.json()
new_id = created["id"]

# ✓ FIXED
response = await client.post("/api/thesis/", json=data)
json = response.json()
created = json["data"]
new_id = created["id"]
```

### 4. Error Handling
```python
# ❌ BROKEN (FastAPI style)
response = await client.get("/api/thesis/999")
error = response.json()
msg = error["detail"]

# ✓ FIXED (ApiResponse style)
response = await client.get("/api/thesis/999")
json = response.json()
msg = json["error"]
```

---

## Files to Update

### CRITICAL (Do First)
1. `test_thesis_api.py` - 14 tests with broken assertions
2. Integration tests - Check all API endpoint tests

### HIGH (Do Next)
3. `test_watchlist.py` - 8+ tests
4. Create new tests for: `admin`, `agents`, `gdpr`, `cache_management`, `monitoring`

### MEDIUM (Complete Migration)
5. Add error scenario tests
6. Add pagination tests
7. Add authorization tests

---

## One-Minute Fix Template

```python
# For EVERY test that calls an API endpoint:

# STEP 1: Get the full response
response = await client.get("/api/v1/items/")

# STEP 2: Parse JSON
response_json = response.json()

# STEP 3: Check success
assert response_json["success"] is True  # For success cases

# STEP 4: Unwrap data
data = response_json["data"]  # or items = response_json["data"]

# STEP 5: Use data as before
assert data["name"] == "expected"
```

---

## Test Helper Functions

Add to `conftest.py`:

```python
def assert_success_response(json_response):
    """Verify response wrapper and return data"""
    assert json_response["success"] is True
    assert json_response["error"] is None
    return json_response["data"]

def assert_error_response(json_response):
    """Verify error wrapper and return error message"""
    assert json_response["success"] is False
    assert json_response["data"] is None
    return json_response["error"]

def assert_paginated_response(json_response):
    """Verify pagination and return data + meta"""
    assert json_response["success"] is True
    assert json_response["meta"] is not None
    return json_response["data"], json_response["meta"]
```

Then use in tests:

```python
response = await client.get("/api/items/")
json = response.json()
items, meta = assert_paginated_response(json)
assert len(items) > 0
```

---

## Priority Timeline

### Phase 1: CRITICAL (3-4 hours) - MUST DO TODAY
- [ ] Fix 14 tests in `test_thesis_api.py`
- [ ] Fix integration test assertions
- [ ] Verify tests pass

### Phase 2: HIGH (6.5 hours) - TOMORROW
- [ ] Create 20+ new tests for uncovered routers
- [ ] Reach 50% coverage

### Phase 3: MEDIUM (6.5 hours) - WITHIN 48 HOURS
- [ ] Add error scenario tests
- [ ] Add pagination tests
- [ ] Add authorization tests
- [ ] Reach 80% coverage target

---

## Validation Checklist

Before deploying:

- [ ] `test_thesis_api.py` - All 14 tests passing
- [ ] `test_watchlist.py` - All tests passing
- [ ] `test_integration.py` - All API tests passing
- [ ] `test_api_integration.py` - All API tests passing
- [ ] Coverage ≥ 80% for all migrated routers
- [ ] Error responses tested
- [ ] Pagination tested
- [ ] Authorization tested

---

## Documentation Files

1. **API_STANDARDIZATION_VALIDATION.md** - Comprehensive analysis
2. **TEST_FIX_EXAMPLES.md** - Detailed before/after examples
3. **BREAKING_CHANGES_SUMMARY.md** - This file

---

## Quick Commands

```bash
# Run thesis tests (currently failing)
pytest backend/tests/test_thesis_api.py -v

# Run watchlist tests
pytest backend/tests/test_watchlist.py -v

# Run all API tests
pytest backend/tests/ -k "test_api" -v

# Run with coverage
pytest backend/tests/ --cov=backend/api/routers --cov-report=html
```

---

## Key Insight

The wrapper is **complete and correct**. The `ApiResponse` model is properly implemented with helper functions. Tests just need to adjust how they access response data.

**It's a straightforward assertion fix, not a logic error.**

---

## Rollback Plan

If issues arise:
1. Revert routers back to direct returns (30 mins)
2. Keep database/logic layer intact (1.5 hours)
3. Re-plan as phased migration (2 hours)

**Total rollback time: 4 hours**

---

## Support

- See **API_STANDARDIZATION_VALIDATION.md** for detailed analysis
- See **TEST_FIX_EXAMPLES.md** for code examples
- Implementation in: `/backend/models/api_response.py`

