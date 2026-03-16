# Phase 2 Quick Fixes

**Purpose:** Step-by-step guide to resolve identified issues
**Time Required:** ~30 minutes total
**Priority:** HIGH (blocking production deployment)

---

## 🚨 Critical Fix #1: Async Fixture Compatibility (5 minutes)

### Issue
```
pytest.PytestRemovedIn9Warning: 'test_analyze_stock_success' requested an async fixture
'test_client_with_engine', with no plugin or hook that handled it.
```

### Root Cause
The fixture is defined with `@pytest.fixture` but yields an async context manager. pytest-asyncio requires `@pytest_asyncio.fixture` for async fixtures.

### Fix

**File:** `backend/tests/test_agents_api.py`

**Line 175:** Change this:
```python
@pytest.fixture
async def test_client_with_engine(mock_hybrid_engine, mock_current_user):
    """Create test client with mocked dependencies"""
```

**To this:**
```python
@pytest_asyncio.fixture
async def test_client_with_engine(mock_hybrid_engine, mock_current_user):
    """Create test client with mocked dependencies"""
```

**Also add import at top of file (around line 6):**
```python
import pytest
import pytest_asyncio  # ADD THIS LINE
import asyncio
from datetime import datetime
```

### Verify Fix
```bash
# Run the specific tests that were failing
pytest backend/tests/test_agents_api.py::TestAgentAnalysis::test_analyze_stock_success -v
pytest backend/tests/test_agents_api.py::TestBatchAnalysis::test_batch_analyze_success -v

# Run all agents API tests
pytest backend/tests/test_agents_api.py -v

# Expected: All 19 tests should pass
```

---

## 🔧 Fix #2: Install mypy for Type Checking (2 minutes)

### Issue
```
Exit code 127: command not found: mypy
```

### Fix

**Install mypy:**
```bash
pip install mypy
```

**Or add to requirements:**
```bash
# requirements-dev.txt
mypy>=1.8.0
```

### Run Type Checking
```bash
# Check individual routers
mypy backend/api/routers/admin.py --show-error-codes
mypy backend/api/routers/agents.py --show-error-codes
mypy backend/api/routers/gdpr.py --show-error-codes

# Check all routers
mypy backend/api/routers/ --show-error-codes

# Expected: 0 errors (maybe some Pydantic warnings)
```

---

## 📊 Fix #3: Generate Coverage Report (5 minutes)

### Pre-requisite
Fix #1 must be completed first (async fixture fix)

### Commands

**Generate full coverage report:**
```bash
pytest backend/tests/ \
  --cov=backend/api/routers \
  --cov-report=term-missing \
  --cov-report=html \
  -v
```

**View HTML report:**
```bash
open htmlcov/index.html
```

**Check specific routers:**
```bash
pytest backend/tests/ \
  --cov=backend/api/routers/admin.py \
  --cov=backend/api/routers/agents.py \
  --cov=backend/api/routers/gdpr.py \
  --cov-report=term-missing
```

### Expected Results
```
admin.py:   40-60% coverage
agents.py:  60-80% coverage (has comprehensive tests)
gdpr.py:    30-50% coverage
```

---

## 📝 Fix #4: Pydantic V2 Deprecation Warnings (15 minutes)

### Issue
```
PydanticDeprecatedSince20: `min_items` is deprecated and will be removed,
use `min_length` instead.
```

### Files to Fix
- `backend/api/routers/agents.py`

### Changes

**File:** `backend/api/routers/agents.py`

**Line 71:** Change this:
```python
class BatchAnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers", min_items=1, max_items=50)
    max_concurrent: int = Field(5, description="Maximum concurrent analyses", gt=0, le=10)
    prioritize_by_tier: bool = Field(True, description="Prioritize analysis by stock tier")
```

**To this:**
```python
class BatchAnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers", min_length=1, max_length=50)
    max_concurrent: int = Field(5, description="Maximum concurrent analyses", gt=0, le=10)
    prioritize_by_tier: bool = Field(True, description="Prioritize analysis by stock tier")
```

**Note:** This is LOW priority - the code works fine with deprecated attributes

### Verify
```bash
# Run tests - should still pass with no deprecation warnings for this model
pytest backend/tests/test_agents_api.py -v -W ignore::DeprecationWarning
```

---

## 🧪 Optional: Create Missing Test Files (2-3 hours each)

### Test File #1: Admin API Tests

**Create:** `backend/tests/test_admin_api.py`

**Template Structure:**
```python
"""
Comprehensive tests for backend/api/routers/admin.py
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from backend.api.main import app
from backend.tests.conftest import assert_success_response, assert_api_error_response


@pytest.fixture
def mock_super_admin():
    """Mock super admin user"""
    return {
        "id": 1,
        "username": "superadmin",
        "email": "super@example.com",
        "is_active": True,
        "is_admin": True,
        "is_super_admin": True
    }


@pytest_asyncio.fixture
async def test_client(mock_super_admin):
    """Create test client with mocked dependencies"""
    from backend.utils.auth import get_current_user, require_super_admin

    app.dependency_overrides[get_current_user] = lambda: mock_super_admin
    app.dependency_overrides[require_super_admin] = lambda: mock_super_admin

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestSystemHealth:
    """Test system health endpoint"""

    async def test_get_system_health(self, test_client):
        """Test system health retrieval"""
        response = await test_client.get("/api/admin/system/health")
        data = assert_success_response(response, 200)

        assert "status" in data
        assert "uptime" in data
        assert "cpu_usage" in data


@pytest.mark.asyncio
class TestUserManagement:
    """Test user management endpoints"""

    async def test_list_users(self, test_client):
        """Test listing users"""
        response = await test_client.get("/api/admin/users")
        data = assert_success_response(response, 200)

        assert isinstance(data, list)

    async def test_get_user(self, test_client):
        """Test getting specific user"""
        response = await test_client.get("/api/admin/users/test-user-id")
        # Implement based on actual endpoint behavior


# Add more test classes for:
# - TestConfigManagement
# - TestSystemCommands
# - TestApiUsageStats
# - TestDataExport
```

---

### Test File #2: GDPR API Tests

**Create:** `backend/tests/test_gdpr_api.py`

**Template Structure:**
```python
"""
Comprehensive tests for backend/api/routers/gdpr.py
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from backend.api.main import app
from backend.tests.conftest import assert_success_response, assert_api_error_response


@pytest_asyncio.fixture
async def test_client():
    """Create test client with mocked dependencies"""
    mock_user = {"id": 1, "email": "test@example.com"}

    from backend.api.routers.gdpr import get_current_user_from_token
    from backend.config.database import get_async_db_session

    app.dependency_overrides[get_current_user_from_token] = lambda req, db: mock_user
    # Mock database session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestConsentManagement:
    """Test consent management endpoints"""

    async def test_record_consent(self, test_client):
        """Test recording user consent"""
        response = await test_client.post(
            "/api/gdpr/consent",
            json={
                "consent_type": "data_processing",
                "granted": True,
                "legal_basis": "explicit_consent"
            }
        )
        data = assert_success_response(response, 200)

        assert data["granted"] == True
        assert data["consent_type"] == "data_processing"


@pytest.mark.asyncio
class TestDataExport:
    """Test data export endpoints"""

    async def test_export_user_data(self, test_client):
        """Test GDPR data export"""
        response = await test_client.get("/api/gdpr/export")
        data = assert_success_response(response, 200)

        assert "export_id" in data
        assert "user_id" in data


# Add more test classes for:
# - TestDataDeletion
# - TestConsentHistory
# - TestRateLimiting
```

---

## ✅ Quick Validation Checklist

After completing fixes:

- [ ] Fix #1: Async fixture updated
- [ ] Verify: `pytest backend/tests/test_agents_api.py -v` → All pass
- [ ] Fix #2: mypy installed
- [ ] Verify: `mypy backend/api/routers/ --show-error-codes` → 0 errors
- [ ] Fix #3: Coverage report generated
- [ ] Verify: Coverage ≥50% for routers
- [ ] Optional: Create test files for admin and GDPR
- [ ] Optional: Fix Pydantic deprecations

---

## 🎯 Expected Results After Fixes

### Test Execution
```
backend/tests/test_agents_api.py ................... PASSED [ 100%]

19 passed in 2.5s
```

### Type Checking
```
mypy backend/api/routers/admin.py
Success: no issues found in 1 source file

mypy backend/api/routers/agents.py
Success: no issues found in 1 source file

mypy backend/api/routers/gdpr.py
Success: no issues found in 1 source file
```

### Coverage Report
```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
backend/api/routers/admin.py        450    250    44%   123-145, 200-250
backend/api/routers/agents.py       380    120    68%   300-350
backend/api/routers/gdpr.py         520    350    33%   100-200, 300-400
---------------------------------------------------------------
TOTAL                              1350    720    47%
```

---

## 🚀 Deploy Checklist

After all fixes completed:

1. **Run Full Test Suite**
   ```bash
   pytest backend/tests/ -v --tb=short
   ```

2. **Generate Final Coverage Report**
   ```bash
   pytest backend/tests/ \
     --cov=backend/api/routers \
     --cov-report=html \
     --cov-fail-under=50
   ```

3. **Type Check All Code**
   ```bash
   mypy backend/api/routers/ --show-error-codes
   ```

4. **Security Scan**
   ```bash
   # Check for secrets
   grep -r "password\|secret\|api_key" backend/api/routers/ | grep -v "Field\|description"

   # Check for dangerous patterns
   grep -r "eval\|exec\|os.system" backend/api/routers/
   ```

5. **Ready for Staging**
   - All tests passing
   - Coverage ≥50%
   - Type checking clean
   - No security issues
   - Documentation complete

---

**Last Updated:** 2026-01-27
**Maintainer:** Production Validation Agent
