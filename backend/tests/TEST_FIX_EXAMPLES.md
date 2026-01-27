# Test Fix Examples - API Standardization

This document provides concrete examples of how to fix tests broken by the ApiResponse wrapper migration.

---

## Quick Reference: Before & After

### Get Single Item
```python
# ❌ BEFORE (Broken)
response = await client.get("/api/v1/thesis/1")
data = response.json()
assert data["id"] == 1
assert data["name"] == "Test"

# ✓ AFTER (Fixed)
response = await client.get("/api/v1/thesis/1")
response_json = response.json()
assert response_json["success"] is True
data = response_json["data"]
assert data["id"] == 1
assert data["name"] == "Test"
```

### List Items
```python
# ❌ BEFORE (Broken)
response = await client.get("/api/v1/theses/")
items = response.json()
assert len(items) == 3

# ✓ AFTER (Fixed)
response = await client.get("/api/v1/theses/")
response_json = response.json()
assert response_json["success"] is True
items = response_json["data"]
assert len(items) == 3
assert response_json["meta"]["total"] == 3
```

### Create Item
```python
# ❌ BEFORE (Broken)
response = await client.post("/api/v1/thesis/", json=data)
created = response.json()
assert created["id"] == 1

# ✓ AFTER (Fixed)
response = await client.post("/api/v1/thesis/", json=data)
response_json = response.json()
assert response_json["success"] is True
created = response_json["data"]
assert created["id"] == 1
```

### Handle Error
```python
# ❌ BEFORE (Broken)
response = await client.get("/api/v1/thesis/999")
error = response.json()
assert "detail" in error

# ✓ AFTER (Fixed)
response = await client.get("/api/v1/thesis/999")
response_json = response.json()
assert response_json["success"] is False
assert response_json["error"] is not None
```

---

## File-by-File Examples

### test_thesis_api.py

#### Test 1: test_create_thesis_success (Lines 60-97)

```python
# ❌ ORIGINAL CODE (BROKEN)
@pytest.mark.asyncio
async def test_create_thesis_success(
    self,
    client: AsyncClient,
    auth_headers: dict,
    test_stock: Stock
):
    """Test successful thesis creation"""
    thesis_data = {
        "stock_id": test_stock.id,
        "investment_objective": "Growth investment with 3-5 year horizon",
        "time_horizon": "medium-term",
        "target_price": 150.50,
        "business_model": "Tech company with strong moat",
        "competitive_advantages": "Network effects and brand",
        "financial_health": "Excellent",
        "growth_drivers": "New product launches",
        "risks": "Competition and regulation",
        "valuation_rationale": "DCF valuation shows upside",
        "exit_strategy": "Exit at target price or 2x return",
        "content": "# Full Thesis Document\n\nDetailed analysis..."
    }

    response = await client.post(
        "/api/v1/thesis/",
        json=thesis_data,
        headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()  # ❌ Gets ApiResponse, not thesis
    assert data["stock_id"] == test_stock.id  # ❌ KeyError
    assert data["investment_objective"] == thesis_data["investment_objective"]  # ❌ KeyError
    assert data["time_horizon"] == thesis_data["time_horizon"]  # ❌ KeyError
    assert float(data["target_price"]) == thesis_data["target_price"]  # ❌ KeyError
    assert data["version"] == 1  # ❌ KeyError
    assert data["stock_symbol"] == test_stock.symbol  # ❌ KeyError
    assert "id" in data  # ❌ KeyError
    assert "created_at" in data  # ❌ KeyError

# ✓ FIXED CODE
@pytest.mark.asyncio
async def test_create_thesis_success(
    self,
    client: AsyncClient,
    auth_headers: dict,
    test_stock: Stock
):
    """Test successful thesis creation"""
    thesis_data = {
        "stock_id": test_stock.id,
        "investment_objective": "Growth investment with 3-5 year horizon",
        "time_horizon": "medium-term",
        "target_price": 150.50,
        "business_model": "Tech company with strong moat",
        "competitive_advantages": "Network effects and brand",
        "financial_health": "Excellent",
        "growth_drivers": "New product launches",
        "risks": "Competition and regulation",
        "valuation_rationale": "DCF valuation shows upside",
        "exit_strategy": "Exit at target price or 2x return",
        "content": "# Full Thesis Document\n\nDetailed analysis..."
    }

    response = await client.post(
        "/api/v1/thesis/",
        json=thesis_data,
        headers=auth_headers
    )

    assert response.status_code == 201
    response_json = response.json()  # ✓ Get ApiResponse wrapper

    # ✓ Verify wrapper structure
    assert response_json["success"] is True
    assert response_json["error"] is None
    assert response_json["meta"] is None
    assert response_json["data"] is not None

    # ✓ Extract and verify data
    data = response_json["data"]
    assert data["stock_id"] == test_stock.id
    assert data["investment_objective"] == thesis_data["investment_objective"]
    assert data["time_horizon"] == thesis_data["time_horizon"]
    assert float(data["target_price"]) == thesis_data["target_price"]
    assert data["version"] == 1
    assert data["stock_symbol"] == test_stock.symbol
    assert "id" in data
    assert "created_at" in data
```

#### Test 2: test_get_thesis_by_id (Lines 164-180)

```python
# ❌ ORIGINAL CODE (BROKEN)
@pytest.mark.asyncio
async def test_get_thesis_by_id(
    self,
    client: AsyncClient,
    auth_headers: dict,
    test_thesis: InvestmentThesis
):
    """Test retrieving thesis by ID"""
    response = await client.get(
        f"/api/v1/thesis/{test_thesis.id}",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()  # ❌ Gets ApiResponse
    assert data["id"] == test_thesis.id  # ❌ KeyError
    assert data["investment_objective"] == test_thesis.investment_objective  # ❌ KeyError
    assert data["stock_symbol"] is not None  # ❌ KeyError

# ✓ FIXED CODE
@pytest.mark.asyncio
async def test_get_thesis_by_id(
    self,
    client: AsyncClient,
    auth_headers: dict,
    test_thesis: InvestmentThesis
):
    """Test retrieving thesis by ID"""
    response = await client.get(
        f"/api/v1/thesis/{test_thesis.id}",
        headers=auth_headers
    )

    assert response.status_code == 200
    response_json = response.json()  # ✓ Get ApiResponse wrapper

    # ✓ Verify wrapper
    assert response_json["success"] is True

    # ✓ Extract and verify data
    data = response_json["data"]
    assert data["id"] == test_thesis.id
    assert data["investment_objective"] == test_thesis.investment_objective
    assert data["stock_symbol"] is not None
```

#### Test 3: test_list_user_theses (Pagination Example)

```python
# ❌ ORIGINAL CODE (BROKEN)
@pytest.mark.asyncio
async def test_list_user_theses(
    self,
    client: AsyncClient,
    auth_headers: dict,
    test_thesis: InvestmentThesis
):
    """Test listing all user theses"""
    response = await client.get(
        "/api/v1/thesis/",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()  # ❌ Gets ApiResponse, not list
    assert len(data) >= 1  # ❌ TypeError: object of type ApiResponse is not iterable
    assert data[0]["id"] == test_thesis.id  # ❌ TypeError

# ✓ FIXED CODE
@pytest.mark.asyncio
async def test_list_user_theses(
    self,
    client: AsyncClient,
    auth_headers: dict,
    test_thesis: InvestmentThesis
):
    """Test listing all user theses"""
    response = await client.get(
        "/api/v1/thesis/",
        headers=auth_headers
    )

    assert response.status_code == 200
    response_json = response.json()  # ✓ Get ApiResponse wrapper

    # ✓ Verify wrapper
    assert response_json["success"] is True

    # ✓ Verify pagination metadata
    assert response_json["meta"] is not None
    assert "total" in response_json["meta"]
    assert "page" in response_json["meta"]

    # ✓ Extract and verify data
    theses = response_json["data"]
    assert len(theses) >= 1
    assert theses[0]["id"] == test_thesis.id
```

#### Test 4: test_list_theses_pagination (Advanced Example)

```python
# ❌ ORIGINAL CODE (BROKEN)
@pytest.mark.asyncio
async def test_list_theses_pagination(
    self,
    client: AsyncClient,
    auth_headers: dict
):
    """Test thesis list pagination"""
    # Assume we've created 25 theses
    response = await client.get(
        "/api/v1/thesis/?page=1&limit=10",
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()  # ❌ Gets ApiResponse
    assert len(data) == 10  # ❌ TypeError: object is not iterable

    # Get page 2
    response2 = await client.get(
        "/api/v1/thesis/?page=2&limit=10",
        headers=auth_headers
    )
    data2 = response2.json()  # ❌ Gets ApiResponse
    assert len(data2) == 10  # ❌ TypeError

# ✓ FIXED CODE
@pytest.mark.asyncio
async def test_list_theses_pagination(
    self,
    client: AsyncClient,
    auth_headers: dict
):
    """Test thesis list pagination"""
    # Assume we've created 25 theses
    response = await client.get(
        "/api/v1/thesis/?page=1&limit=10",
        headers=auth_headers
    )

    assert response.status_code == 200
    response_json = response.json()  # ✓ Get ApiResponse wrapper
    assert response_json["success"] is True

    # ✓ Verify pagination metadata
    meta = response_json["meta"]
    assert meta is not None
    assert meta["total"] == 25
    assert meta["page"] == 1
    assert meta["limit"] == 10
    assert meta["pages"] == 3  # 25 / 10 = 3 pages

    # ✓ Extract and verify page 1 data
    page1_data = response_json["data"]
    assert len(page1_data) == 10

    # Get page 2
    response2 = await client.get(
        "/api/v1/thesis/?page=2&limit=10",
        headers=auth_headers
    )
    response_json2 = response2.json()  # ✓ Get ApiResponse wrapper
    assert response_json2["success"] is True

    # ✓ Verify page 2 pagination
    meta2 = response_json2["meta"]
    assert meta2["page"] == 2

    # ✓ Extract and verify page 2 data
    page2_data = response_json2["data"]
    assert len(page2_data) == 10

    # Verify different data
    assert page1_data[0]["id"] != page2_data[0]["id"]
```

#### Test 5: test_get_thesis_not_found (Error Handling)

```python
# ❌ ORIGINAL CODE (BROKEN)
@pytest.mark.asyncio
async def test_get_thesis_not_found(
    self,
    client: AsyncClient,
    auth_headers: dict
):
    """Test retrieving non-existent thesis"""
    response = await client.get(
        "/api/v1/thesis/99999",
        headers=auth_headers
    )

    assert response.status_code == 404
    error = response.json()  # ❌ Gets ApiResponse
    assert "detail" in error  # ❌ KeyError (should be "error")
    assert "not found" in error["detail"].lower()  # ❌ KeyError

# ✓ FIXED CODE
@pytest.mark.asyncio
async def test_get_thesis_not_found(
    self,
    client: AsyncClient,
    auth_headers: dict
):
    """Test retrieving non-existent thesis"""
    response = await client.get(
        "/api/v1/thesis/99999",
        headers=auth_headers
    )

    assert response.status_code == 404
    response_json = response.json()  # ✓ Get ApiResponse wrapper

    # ✓ Verify error response
    assert response_json["success"] is False
    assert response_json["data"] is None
    assert response_json["error"] is not None
    assert "not found" in response_json["error"].lower()
```

---

### test_watchlist.py

#### Example: test_create_watchlist (API endpoint)

```python
# ❌ ORIGINAL CODE (BROKEN)
@pytest.mark.asyncio
async def test_create_watchlist(
    self,
    client: AsyncClient,
    auth_headers: dict
):
    """Test creating a watchlist"""
    watchlist_data = {
        "name": "Tech Stocks",
        "description": "My tech watchlist",
        "is_public": False
    }

    response = await client.post(
        "/api/v1/watchlists/",
        json=watchlist_data,
        headers=auth_headers
    )

    assert response.status_code == 201
    created = response.json()  # ❌ Gets ApiResponse
    assert created["name"] == "Tech Stocks"  # ❌ KeyError
    assert created["is_public"] is False  # ❌ KeyError
    assert "id" in created  # ❌ KeyError

# ✓ FIXED CODE
@pytest.mark.asyncio
async def test_create_watchlist(
    self,
    client: AsyncClient,
    auth_headers: dict
):
    """Test creating a watchlist"""
    watchlist_data = {
        "name": "Tech Stocks",
        "description": "My tech watchlist",
        "is_public": False
    }

    response = await client.post(
        "/api/v1/watchlists/",
        json=watchlist_data,
        headers=auth_headers
    )

    assert response.status_code == 201
    response_json = response.json()  # ✓ Get ApiResponse wrapper
    assert response_json["success"] is True

    created = response_json["data"]
    assert created["name"] == "Tech Stocks"
    assert created["is_public"] is False
    assert "id" in created
```

---

## Using Helper Functions

### Create conftest.py Helpers

```python
# In /backend/tests/conftest.py

def assert_success_response(response_json: dict, expect_data: bool = True, expect_meta: bool = False):
    """
    Assert that response has correct success wrapper structure.

    Args:
        response_json: The parsed JSON response
        expect_data: Whether data should be non-null
        expect_meta: Whether meta should be non-null

    Returns:
        The data field (unwrapped)
    """
    assert isinstance(response_json, dict), "Response must be a dict"
    assert "success" in response_json, "Response must have 'success' field"
    assert "error" in response_json, "Response must have 'error' field"
    assert "data" in response_json, "Response must have 'data' field"
    assert "meta" in response_json, "Response must have 'meta' field"

    assert response_json["success"] is True, "success must be True"
    assert response_json["error"] is None, "error must be None on success"

    if expect_data:
        assert response_json["data"] is not None, "data must not be None"

    if expect_meta:
        assert response_json["meta"] is not None, "meta must not be None"

    return response_json["data"]


def assert_error_response(response_json: dict, expected_error_substring: str = None):
    """
    Assert that response has correct error wrapper structure.

    Args:
        response_json: The parsed JSON response
        expected_error_substring: Substring to find in error message (case-insensitive)

    Returns:
        The error message
    """
    assert isinstance(response_json, dict), "Response must be a dict"
    assert response_json["success"] is False, "success must be False"
    assert response_json["data"] is None, "data must be None on error"
    assert response_json["error"] is not None, "error must be set"

    error_msg = response_json["error"]
    if expected_error_substring:
        assert expected_error_substring.lower() in error_msg.lower(), \
            f"Expected '{expected_error_substring}' in error: {error_msg}"

    return error_msg


def assert_paginated_response(response_json: dict, expected_total: int = None):
    """
    Assert that response has correct pagination metadata.

    Args:
        response_json: The parsed JSON response
        expected_total: Optional expected total count

    Returns:
        Tuple of (data, meta)
    """
    data = assert_success_response(response_json, expect_data=True, expect_meta=True)

    meta = response_json["meta"]
    assert isinstance(meta, dict), "meta must be a dict"
    assert "total" in meta, "meta must have 'total'"
    assert "page" in meta, "meta must have 'page'"
    assert "limit" in meta, "meta must have 'limit'"
    assert "pages" in meta, "meta must have 'pages'"

    assert isinstance(data, list), "data must be a list for paginated response"
    assert meta["page"] >= 1, "page must be >= 1"
    assert meta["limit"] > 0, "limit must be > 0"
    assert meta["total"] >= 0, "total must be >= 0"
    assert meta["pages"] >= 0, "pages must be >= 0"

    if expected_total is not None:
        assert meta["total"] == expected_total, \
            f"Expected total {expected_total}, got {meta['total']}"

    return data, meta
```

### Use Helpers in Tests

```python
# ✓ CLEAN CODE using helpers

@pytest.mark.asyncio
async def test_create_thesis_success(self, client: AsyncClient, auth_headers: dict, test_stock):
    """Test successful thesis creation"""
    thesis_data = {...}
    response = await client.post("/api/v1/thesis/", json=thesis_data, headers=auth_headers)

    assert response.status_code == 201
    response_json = response.json()
    data = assert_success_response(response_json)  # ✓ One line does all checks

    assert data["stock_id"] == test_stock.id
    assert data["investment_objective"] == thesis_data["investment_objective"]


@pytest.mark.asyncio
async def test_list_theses(self, client: AsyncClient, auth_headers: dict):
    """Test listing theses"""
    response = await client.get("/api/v1/thesis/", headers=auth_headers)

    assert response.status_code == 200
    response_json = response.json()
    theses, meta = assert_paginated_response(response_json)  # ✓ Validates pagination

    assert len(theses) > 0
    assert meta["total"] > 0


@pytest.mark.asyncio
async def test_get_thesis_not_found(self, client: AsyncClient, auth_headers: dict):
    """Test thesis not found"""
    response = await client.get("/api/v1/thesis/99999", headers=auth_headers)

    assert response.status_code == 404
    response_json = response.json()
    assert_error_response(response_json, "not found")  # ✓ One line does all checks
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Forgetting to unwrap data
```python
# WRONG
data = response.json()
assert data["name"] == "test"  # KeyError if using ApiResponse

# RIGHT
response_json = response.json()
data = response_json["data"]
assert data["name"] == "test"
```

### ❌ Mistake 2: Not checking success flag
```python
# WRONG - might get error response
data = response.json()["data"]  # Could be None on error
assert data["name"] == "test"  # AttributeError

# RIGHT
response_json = response.json()
assert response_json["success"]  # Fail early if error
data = response_json["data"]
assert data["name"] == "test"
```

### ❌ Mistake 3: Ignoring pagination metadata
```python
# WRONG - Misses pagination info
data = response.json()["data"]
assert len(data) == 10  # Only checking first page

# RIGHT
response_json = response.json()
data = response_json["data"]
meta = response_json["meta"]
assert len(data) == 10
assert meta["page"] == 1
assert meta["total"] == 50  # 5 pages total
```

### ❌ Mistake 4: Wrong error checking
```python
# WRONG
response = await client.get("/api/thesis/999")
error = response.json()
assert "detail" in error  # detail is in FastAPI errors, not ApiResponse

# RIGHT
response = await client.get("/api/thesis/999")
response_json = response.json()
assert response_json["success"] is False
assert response_json["error"] is not None
```

---

## Summary

**Key Changes:**
1. All responses are now wrapped in `ApiResponse`
2. Access data via `response_json["data"]`
3. Always check `response_json["success"]` first
4. Use `response_json["meta"]` for pagination info
5. Use `response_json["error"]` for error messages

**Three-Step Pattern:**
1. `response_json = response.json()` - Get wrapper
2. `assert response_json["success"]` - Verify success
3. `data = response_json["data"]` - Unwrap data

