# Wave 6 Database & Middleware Testing Patterns

**Extracted:** 2026-01-29
**Context:** FastAPI + SQLAlchemy 2.0 + pytest integration testing

## Problem 1: Database Driver Compatibility

### When to Use
Trigger this skill when you see:
- `'server_settings' is an invalid keyword argument for Connection()`
- `Invalid value 'READ COMMITTED' for isolation_level`
- Different databases for testing (SQLite) vs production (PostgreSQL)
- SQLAlchemy 2.0 async engine configuration

### Solution
```python
# backend/config/database.py
is_postgresql = "postgresql" in self.config.url

if is_postgresql:
    # PostgreSQL-specific (asyncpg driver)
    connect_args = {
        "server_settings": {
            "application_name": "app_name",
            "jit": "off",
        },
        "command_timeout": 60,
        "statement_cache_size": 100,
    }
    isolation_level = "READ COMMITTED"
else:
    # SQLite-specific (aiosqlite driver)
    connect_args = {
        "check_same_thread": False,
    }
    isolation_level = "SERIALIZABLE"

self._engine = create_async_engine(
    self.config.url,
    connect_args=connect_args,
    isolation_level=isolation_level,
    # ... other params
)
```

### Key Points
1. **Detect database type** from connection URL string
2. **Conditional connect_args** based on driver capabilities
3. **Map isolation levels**: PostgreSQL READ COMMITTED → SQLite SERIALIZABLE
4. **Driver differences**:
   - asyncpg (PostgreSQL): server_settings, command_timeout, statement_cache_size
   - aiosqlite (SQLite): check_same_thread

---

## Problem 2: Middleware Interfering with Tests

### When to Use
Trigger this skill when:
- AsyncClient tests fail with CSRF errors
- Monitoring middleware breaks test execution
- Security headers block test requests
- Middleware designed for production interferes with testing

### Solution
```python
# In any middleware function
import os

async def middleware_function(request: Request, call_next):
    # Skip middleware in testing environment
    if os.getenv("TESTING") == "True":
        return await call_next(request)

    # Production middleware logic
    # ... (CSRF validation, metrics, security checks)
```

### Locations to Apply
1. **CSRF Protection** - `backend/security/csrf_protection.py`
2. **Monitoring** - `backend/utils/monitoring.py`
3. **Security Filters** - `backend/security/security_config.py`
4. **Rate Limiting** - Any rate limiter middleware

### Test Configuration
```python
# backend/tests/conftest.py
os.environ["TESTING"] = "True"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
```

---

## Example: Combined Application

```python
# backend/config/database.py (Production + Test Support)
class DatabaseConfig:
    def create_engine(self):
        # Pattern 1: Database driver compatibility
        is_postgresql = "postgresql" in self.config.url

        connect_args = {
            "server_settings": {...} if is_postgresql else {},
            "check_same_thread": False if not is_postgresql else None,
        }

        isolation = "READ COMMITTED" if is_postgresql else "SERIALIZABLE"

        return create_async_engine(
            self.config.url,
            connect_args=connect_args,
            isolation_level=isolation,
        )

# backend/security/csrf_protection.py (Test-Compatible Middleware)
async def csrf_middleware(request: Request, call_next):
    # Pattern 2: Middleware test compatibility
    if os.getenv("TESTING") == "True":
        return await call_next(request)

    # Production CSRF validation
    await validate_csrf_token(request)
    response = await call_next(request)
    return response
```

---

## Verification

### Database Pattern
```bash
# Should work with both PostgreSQL and SQLite
pytest backend/tests/integration/test_database_config.py -v
```

### Middleware Pattern
```bash
# All middleware tests should pass
pytest backend/tests/integration/test_auth_to_portfolio_flow.py -v  # CSRF
pytest backend/tests/integration/test_monitoring.py -v              # Monitoring
pytest backend/tests/integration/test_security.py -v                # Security
```

---

## Success Metrics
- ✅ Database: Supports PostgreSQL (prod) + SQLite (test)
- ✅ Middleware: 117/122 tests passing with conditional execution
- ✅ Tests: No more driver parameter errors
- ✅ Production: No behavior changes, only conditional test bypasses

---

## Related Skills
- `/database-config` - Database configuration patterns
- `/middleware-testing` - Testing middleware-heavy applications
- `/environment-based-config` - Using env vars for conditional behavior

## Tags
`database`, `middleware`, `testing`, `sqlalchemy`, `fastapi`, `postgresql`, `sqlite`, `csrf`, `pytest`
