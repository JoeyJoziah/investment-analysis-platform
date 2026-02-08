# Next Steps for Auth Flow Test Debugging

## Current Status

### Completed Fixes
✓ Request stream consumption issue resolved (middleware body reading)
✓ Test endpoint paths updated to match actual API routes
✓ Testing mode configuration verified

### Blocking Issue
✗ Context Manager Error in Dependency Injection
```
TypeError: '_GeneratorContextManager' object is not an iterator
```

## Debug Approach

### Step 1: Locate the Problematic Dependency (10 minutes)

Run this to find all uses of `get_db`:
```bash
grep -r "Depends(get_db" backend/api/routers/ --include="*.py"
```

Also check auth.py specifically:
```bash
grep -A 5 "async def login\|async def token" backend/api/routers/auth.py | head -30
```

Expected to find something like:
```python
# WRONG - sync generator in async function
async def login(credentials: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

# CORRECT - async session for async function
async def login(credentials: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_async_db_session)):
```

### Step 2: Check Available Database Session Getters

List all available options:
```bash
grep -r "^async def get.*session\|^def get_db" backend/config/database.py backend/utils/database*.py
```

Expected output should show:
- `get_db()` - SYNC generator (for sync routes only)
- `get_async_db_session()` - ASYNC for use in async routes
- `get_db_session()` - May be sync or async, check implementation

### Step 3: Identify All Route Issues

Search for potential problems in auth router:
```bash
grep -B 2 -A 10 "async def.*\(.*Depends(get_db" backend/api/routers/auth.py
```

For each async route found, check:
- Is `get_db` used? (WRONG)
- Should be `get_async_db_session` (CORRECT)

### Step 4: Review Dependency Pattern

Check if there's a general pattern issue:
```bash
grep -r "Session = Depends(get_db)" backend/api/routers/ --include="*.py" | wc -l
grep -r "AsyncSession = Depends(get_async_db_session)" backend/api/routers/ --include="*.py" | wc -l
```

If first number is high and second is low, systematic issue.

### Step 5: Inspect get_db Implementation

Check what it is:
```bash
grep -A 10 "^def get_db():" backend/utils/database.py
```

Should be a generator function, like:
```python
@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

This is sync-only and cannot be used in async context!

## Remediation Plan

### Option 1: Use Correct Async Session Getter (RECOMMENDED)

In **backend/api/routers/auth.py**:

**Before:**
```python
from backend.config.database import get_db

@router.post("/token")
async def login(
    credentials: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # ... auth logic with db.query() ...
```

**After:**
```python
from backend.config.database import get_async_db_session
from sqlalchemy.ext.asyncio import AsyncSession

@router.post("/token")
async def login(
    credentials: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_async_db_session)
):
    # ... auth logic with await session.execute() ...
```

### Option 2: Create Async Wrapper (TEMPORARY)

If immediate fix not possible, create async wrapper in a new file:

**backend/utils/async_db_compat.py:**
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config.database import get_async_db_session

@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async wrapper for dependency injection"""
    async for db in get_async_db_session():
        yield db
```

Then use in auth.py:
```python
from backend.utils.async_db_compat import get_async_db

@router.post("/token")
async def login(
    credentials: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    # ... use await db.execute() ...
```

## Validation Steps

After making fixes:

### Test 1: Single Login Test
```bash
pytest backend/tests/integration/test_auth_to_portfolio_flow.py::test_login_to_portfolio_access -xvs
```

Should see one of:
- ✓ Test passes
- ✗ Different error (progress - past context manager issue)
- ✓ 404 for endpoint (good - means dependency resolution worked)

### Test 2: All Auth Tests
```bash
pytest backend/tests/integration/test_auth_to_portfolio_flow.py -v
```

Expected: Fewer failures than before (ideally 0)

### Test 3: Check for Regression
```bash
pytest backend/tests/ -k "not test_auth_to_portfolio_flow" --co | wc -l
```

Run sample of other tests to ensure no regression:
```bash
pytest backend/tests/integration/test_integration_comprehensive.py -x
```

## Documentation to Create After Fix

Once tests pass, create:
1. **ASYNC_PATTERN_GUIDE.md** - How to use async database in FastAPI routes
2. **DATABASE_SESSION_REFERENCE.md** - All available session getters and when to use each
3. Update **ARCHITECTURAL_DECISIONS.md** with note about sync vs async patterns

## Expected Timeline

- **5 min**: Locate problematic dependency
- **5 min**: Understand the issue
- **10 min**: Make code changes
- **5 min**: Test and verify
- **Total**: ~25 minutes to resolution

## Key Learnings

1. **Async/Sync Context Managers**: Cannot mix sync generators with async contexts
2. **FastAPI Depends**: Must provide async dependencies for async routes
3. **Middleware Order**: Request stream consumption happens early, affects all downstream
4. **Testing Discipline**: Environment-based configuration (TESTING flag) helps isolate test behavior

## Questions to Ask if Stuck

1. Where is the auth token endpoint (`/api/auth/token`) defined?
2. What type of database session does it use? (Session vs AsyncSession)
3. Which get_db function is imported in that file?
4. Is get_async_db_session available in the same imports?
5. Are there type hints showing Session vs AsyncSession mismatch?

## Success Criteria for Next Phase

✓ No more context manager TypeError
✓ Auth endpoints return proper responses (not 500 errors)
✓ Test can reach `/api/auth/token` endpoint
✓ Test can reach `/api/portfolio/{id}` endpoint
✓ At least one test shows endpoint logic error (not middleware error)
