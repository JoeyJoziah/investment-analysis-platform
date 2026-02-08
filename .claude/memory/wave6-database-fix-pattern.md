# Wave 6 Phase 1: Database Configuration Fix Pattern

## Problem Discovered
**Date**: 2026-01-29
**Context**: Wave 6 Phase 1 - CSRF Configuration & Auth Test Fixes
**Issue**: Auth flow integration tests failing with database connection errors

## Root Cause
SQLite (aiosqlite driver) and PostgreSQL (asyncpg driver) have incompatible connection parameters:

### Error 1: `'server_settings' is an invalid keyword argument for Connection()`
- PostgreSQL-specific: `server_settings`, `command_timeout`, `statement_cache_size`
- SQLite doesn't support these parameters
- Error occurred during database initialization in `backend/config/database.py:136`

### Error 2: `Invalid value 'READ COMMITTED' for isolation_level`
- PostgreSQL supports: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE
- SQLite supports: READ UNCOMMITTED, SERIALIZABLE, AUTOCOMMIT
- READ COMMITTED is PostgreSQL-specific, causes error with SQLite

## Solution Pattern (REUSABLE)

### File: `backend/config/database.py`

```python
# Detect database type from connection URL
is_postgresql = "postgresql" in self.config.url

# Conditional connect_args based on driver
if is_postgresql:
    # PostgreSQL-specific parameters (asyncpg driver)
    connect_args = {
        "server_settings": {
            "application_name": "investment_analysis_app",
            "jit": "off",
        },
        "command_timeout": 60,
        "statement_cache_size": self.config.prepared_statement_cache_size,
    }
else:
    # SQLite-specific parameters (aiosqlite driver)
    connect_args = {
        "check_same_thread": False,  # Allow multi-threaded access
    }

# Conditional isolation_level based on database
if is_postgresql:
    isolation_level = self.config.isolation_level.value  # READ COMMITTED
else:
    # SQLite: Use SERIALIZABLE (closest to PostgreSQL's READ COMMITTED)
    isolation_level = "SERIALIZABLE"
```

## Test Configuration Context
Tests use SQLite `:memory:` database (from `backend/tests/conftest.py`):
```python
os.environ["TESTING"] = "True"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
```

## Key Learnings

1. **Driver-specific parameters**: Always check driver documentation when configuring database connections
2. **Test vs production**: Tests often use different databases (SQLite vs PostgreSQL)
3. **Conditional configuration**: Use URL detection (`"postgresql" in url`) for driver-specific settings
4. **Isolation level mapping**:
   - PostgreSQL READ COMMITTED → SQLite SERIALIZABLE (closest equivalent)
   - PostgreSQL REPEATABLE READ → SQLite SERIALIZABLE
   - PostgreSQL SERIALIZABLE → SQLite SERIALIZABLE

## Related Files Modified
- `backend/config/database.py` (lines 135-167)

## Status
✅ Fixed - Database configuration now supports both PostgreSQL (production) and SQLite (testing)
⚠️ Ongoing - Auth tests still hanging, likely middleware or async issues

## Reusability Score: HIGH
This pattern applies to any FastAPI/SQLAlchemy 2.0 project that:
- Uses different databases for testing vs production
- Supports both PostgreSQL and SQLite
- Uses async engines with driver-specific parameters

## Tags
`database`, `sqlalchemy`, `postgresql`, `sqlite`, `testing`, `configuration`, `wave6`
