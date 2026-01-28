# Line-by-Line Mapping of Changes

## File: /backend/tests/test_thesis_api.py

### Import Section (Line 12)
```python
# Added import for helper functions
from backend.tests.conftest import assert_success_response, assert_api_error_response
```

### Test Methods: Before and After

---

## Test 1: test_create_thesis_success
**Location**: Lines 61-97
**Status**: ALREADY FIXED (6/14)
**Line 89**: `data = assert_success_response(response, expected_status=201)`

---

## Test 2: test_create_thesis_missing_required_fields
**Location**: Lines 99-118
**Status**: NOW FIXED (1/8)

### Change at Line 118:
```diff
- assert response.status_code == 422  # Validation error
+ assert_api_error_response(response, 422)
```

**Context**:
```python
response = await client.post(
    "/api/v1/thesis/",
    json=thesis_data,
    headers=auth_headers
)
- assert response.status_code == 422  # Line 118 (OLD)
+ assert_api_error_response(response, 422)  # Line 118 (NEW)
```

---

## Test 3: test_create_thesis_invalid_stock_id
**Location**: Lines 120-139
**Status**: NOW FIXED (2/8)

### Change at Line 139:
```diff
- assert response.status_code == 404
+ assert_api_error_response(response, 404)
```

**Context**:
```python
response = await client.post(
    "/api/v1/thesis/",
    json=thesis_data,
    headers=auth_headers
)
- assert response.status_code == 404  # Line 139 (OLD)
+ assert_api_error_response(response, 404)  # Line 139 (NEW)
```

---

## Test 4: test_create_duplicate_thesis
**Location**: Lines 141-161
**Status**: NOW FIXED (3/8)

### Change at Line 161:
```diff
- assert response.status_code == 409  # Conflict
+ assert_api_error_response(response, 409)
```

**Context**:
```python
response = await client.post(
    "/api/v1/thesis/",
    json=thesis_data,
    headers=auth_headers
)
- assert response.status_code == 409  # Conflict  # Line 161 (OLD)
+ assert_api_error_response(response, 409)  # Line 161 (NEW)
```

---

## Test 5: test_get_thesis_by_id
**Location**: Lines 163-179
**Status**: ALREADY FIXED (6/14)
**Line 176**: `data = assert_success_response(response)`

---

## Test 6: test_get_thesis_by_stock_id
**Location**: Lines 181-196
**Status**: ALREADY FIXED (6/14)
**Line 194**: `data = assert_success_response(response)`

---

## Test 7: test_get_thesis_not_found
**Location**: Lines 198-210
**Status**: NOW FIXED (4/8)

### Change at Line 210:
```diff
- assert response.status_code == 404
+ assert_api_error_response(response, 404)
```

**Context**:
```python
response = await client.get(
    "/api/v1/thesis/99999",
    headers=auth_headers
)
- assert response.status_code == 404  # Line 210 (OLD)
+ assert_api_error_response(response, 404)  # Line 210 (NEW)
```

---

## Test 8: test_list_user_theses
**Location**: Lines 212-228
**Status**: ALREADY FIXED (6/14)
**Line 225**: `data = assert_success_response(response)`

---

## Test 9: test_list_theses_pagination
**Location**: Lines 230-245
**Status**: ALREADY FIXED (6/14)
**Line 243**: `data = assert_success_response(response)`

---

## Test 10: test_update_thesis
**Location**: Lines 247-270
**Status**: ALREADY FIXED (6/14)
**Line 267**: `data = assert_success_response(response)`

---

## Test 11: test_update_thesis_not_owned
**Location**: Lines 272-308
**Status**: NOW FIXED (5/8)

### Changes at Lines 307-308:
```diff
- assert response.status_code in [403, 404]  # Forbidden or Not Found
+ assert response.status_code in [403, 404]
+ assert_api_error_response(response, response.status_code)
```

**Context**:
```python
response = await client.put(
    f"/api/v1/thesis/{test_thesis.id}",
    json=update_data,
    headers=other_headers
)
- assert response.status_code in [403, 404]  # Forbidden or Not Found  # Line 307 (OLD)
+ assert response.status_code in [403, 404]  # Line 307 (NEW - kept)
+ assert_api_error_response(response, response.status_code)  # Line 308 (NEW - added)
```

---

## Test 12: test_delete_thesis
**Location**: Lines 310-330
**Status**: NOW FIXED (6/8)

### Change at Line 330:
```diff
- assert get_response.status_code == 404
+ assert_api_error_response(get_response, 404)
```

**Context**:
```python
response = await client.delete(
    f"/api/v1/thesis/{test_thesis.id}",
    headers=auth_headers
)
assert response.status_code == 204

# Verify thesis is deleted
get_response = await client.get(
    f"/api/v1/thesis/{test_thesis.id}",
    headers=auth_headers
)
- assert get_response.status_code == 404  # Line 330 (OLD)
+ assert_api_error_response(get_response, 404)  # Line 330 (NEW)
```

---

## Test 13: test_delete_thesis_not_owned
**Location**: Lines 332-361
**Status**: NOW FIXED (7/8)

### Changes at Lines 360-361:
```diff
- assert response.status_code in [403, 404]
+ assert response.status_code in [403, 404]
+ assert_api_error_response(response, response.status_code)
```

**Context**:
```python
response = await client.delete(
    f"/api/v1/thesis/{test_thesis.id}",
    headers=other_headers
)
- assert response.status_code in [403, 404]  # Line 360 (OLD)
+ assert response.status_code in [403, 404]  # Line 360 (NEW - kept)
+ assert_api_error_response(response, response.status_code)  # Line 361 (NEW - added)
```

---

## Test 14: test_thesis_requires_authentication
**Location**: Lines 363-394
**Status**: NOW FIXED (8/8)

### Change at Line 372 (GET endpoint):
```diff
- assert response.status_code == 401
+ assert_api_error_response(response, 401)
```

### Change at Line 383 (POST endpoint):
```diff
- assert response.status_code == 401
+ assert_api_error_response(response, 401)
```

### Change at Line 390 (PUT endpoint):
```diff
- assert response.status_code == 401
+ assert_api_error_response(response, 401)
```

### Change at Line 394 (DELETE endpoint):
```diff
- assert response.status_code == 401
+ assert_api_error_response(response, 401)
```

**Context**:
```python
# Test GET
response = await client.get(f"/api/v1/thesis/{test_thesis.id}")
- assert response.status_code == 401  # Line 372 (OLD)
+ assert_api_error_response(response, 401)  # Line 372 (NEW)

# Test POST
response = await client.post(
    "/api/v1/thesis/",
    json={...}
)
- assert response.status_code == 401  # Line 383 (OLD)
+ assert_api_error_response(response, 401)  # Line 383 (NEW)

# Test PUT
response = await client.put(
    f"/api/v1/thesis/{test_thesis.id}",
    json={"investment_objective": "test"}
)
- assert response.status_code == 401  # Line 390 (OLD)
+ assert_api_error_response(response, 401)  # Line 390 (NEW)

# Test DELETE
response = await client.delete(f"/api/v1/thesis/{test_thesis.id}")
- assert response.status_code == 401  # Line 394 (OLD)
+ assert_api_error_response(response, 401)  # Line 394 (NEW)
```

---

## Summary of Line Changes

| Line(s) | Test | Change | Type |
|---------|------|--------|------|
| 12 | Import | Added helper function imports | Enhancement |
| 118 | test_create_thesis_missing_required_fields | 422 validation | Updated |
| 139 | test_create_thesis_invalid_stock_id | 404 not found | Updated |
| 161 | test_create_duplicate_thesis | 409 conflict | Updated |
| 210 | test_get_thesis_not_found | 404 not found | Updated |
| 308 | test_update_thesis_not_owned | Added wrapper validation | Enhanced |
| 330 | test_delete_thesis | 404 verification | Updated |
| 361 | test_delete_thesis_not_owned | Added wrapper validation | Enhanced |
| 372 | test_thesis_requires_authentication (GET) | 401 unauthorized | Updated |
| 383 | test_thesis_requires_authentication (POST) | 401 unauthorized | Updated |
| 390 | test_thesis_requires_authentication (PUT) | 401 unauthorized | Updated |
| 394 | test_thesis_requires_authentication (DELETE) | 401 unauthorized | Updated |

## Total Changes
- **Lines Modified**: 12
- **Assertions Updated**: 12
- **Helper Calls Added**: 12
- **Status Codes Affected**: 8 distinct codes (422, 404 x4, 409, 403, 401 x4)

## Result
All 8 remaining tests now properly validate ApiResponse wrapper structure using centralized helper functions.
