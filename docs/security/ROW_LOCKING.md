# Row Locking Implementation Guide

## Overview

This document describes the row locking implementation for the Investment Analysis Platform. Row locking prevents concurrent update issues and ensures data consistency in multi-user scenarios.

## Locking Strategies

### 1. Optimistic Locking (Version-Based)

**How it works:**
- Each record has a `version` column that increments on every update
- Before updating, client provides the expected version
- If current version doesn't match, a `StaleDataError` is raised
- Client must re-fetch data and retry

**When to use:**
- Low contention scenarios (most updates succeed)
- Read-heavy workloads
- When you want to avoid database locks

**Models with optimistic locking:**
- `Portfolio` - version column added
- `Position` - version column added
- `InvestmentThesis` - version column already existed

**Example:**
```python
from backend.repositories.portfolio_repository import PortfolioRepository
from backend.exceptions import StaleDataError

repo = PortfolioRepository()

# Read portfolio with current version
portfolio = await repo.get_by_id(portfolio_id)
current_version = portfolio.version

# Update with version check
try:
    updated = await repo.update_with_lock(
        id=portfolio_id,
        data={"name": "New Name"},
        expected_version=current_version
    )
    print(f"Updated to version {updated.version}")
except StaleDataError as e:
    print(f"Version conflict: expected {e.expected_version}, got {e.current_version}")
    # Re-fetch and retry
```

### 2. Pessimistic Locking (SELECT FOR UPDATE)

**How it works:**
- Uses database-level row locks via `SELECT ... FOR UPDATE`
- Blocks other transactions from reading/writing the locked row
- Lock is held until transaction commits or rolls back

**When to use:**
- High contention scenarios (many concurrent updates)
- Critical financial operations (balance updates, trades)
- When you must prevent concurrent modifications

**Example:**
```python
from backend.repositories.base import AsyncCRUDRepository
from backend.config.database import get_db_session

repo = AsyncCRUDRepository(Portfolio)

async with get_db_session() as session:
    # Lock portfolio for update
    portfolio = await repo.get_with_lock(
        id=portfolio_id,
        for_update=True,
        session=session
    )

    # Make changes - no one else can modify until commit
    portfolio.cash_balance -= 1000.00

    await session.commit()
```

### 3. Hybrid Approach (Both)

**Best practice:** Combine both for maximum safety

The `add_position()` method uses both:
1. **Pessimistic lock** - Locks portfolio and position rows
2. **Optimistic check** - Validates version if provided
3. **Business logic validation** - Checks balance and quantity

```python
# Lock portfolio and position
portfolio = await session.execute(
    select(Portfolio)
    .where(Portfolio.id == portfolio_id)
    .with_for_update()
)

# Check version
if expected_version and portfolio.version != expected_version:
    raise StaleDataError(...)

# Validate business rules
if portfolio.cash_balance < trade_amount:
    raise InsufficientBalanceError(...)

# Execute update with version increment
portfolio.version += 1
portfolio.cash_balance -= trade_amount
```

## Protected Models

### Portfolio

**Version column:** `version` (Integer)

**Critical operations protected:**
- `add_position()` - Uses SELECT FOR UPDATE + version check
- `update_with_lock()` - Optimistic locking available

**Race conditions prevented:**
- Concurrent balance updates
- Overdraft from parallel withdrawals
- Position updates from multiple clients

**Usage:**
```python
repo = PortfolioRepository()

# Add position with locking
await repo.add_position(
    portfolio_id=1,
    stock_id=5,
    quantity=Decimal("10"),
    price=Decimal("100.00"),
    transaction_type='buy',
    expected_portfolio_version=3  # Optional version check
)
```

### Position

**Version column:** `version` (Integer)

**Critical operations protected:**
- Buy/sell transactions via `add_position()`
- Quantity and average cost updates

**Race conditions prevented:**
- Overselling (selling more shares than owned)
- Incorrect average cost calculations from concurrent buys
- Lost updates to position quantities

**Behavior:**
- Version increments on every buy/sell
- SELECT FOR UPDATE ensures serialized access
- Position deleted when quantity reaches 0

### InvestmentThesis

**Version column:** `version` (Integer, pre-existing)

**Critical operations protected:**
- `update_thesis()` - Uses SELECT FOR UPDATE + version check

**Race conditions prevented:**
- Lost updates from concurrent thesis edits
- Overwriting changes from other users

**Usage:**
```python
repo = InvestmentThesisRepository()

# Update thesis with version check
try:
    updated = await repo.update_thesis(
        thesis_id=123,
        user_id=456,
        data={"investment_objective": "Growth"},
        expected_version=2
    )
except StaleDataError as e:
    print(f"Thesis was modified by someone else")
    # Re-fetch and merge changes
```

## Exception Handling

### StaleDataError

**When raised:**
- Optimistic lock version mismatch detected
- Another transaction modified the data

**Attributes:**
- `entity_type` - Portfolio, Position, InvestmentThesis
- `entity_id` - ID of the entity
- `expected_version` - Version client had
- `current_version` - Current version in database

**How to handle:**
```python
try:
    await repo.update_with_lock(
        id=portfolio_id,
        data=updates,
        expected_version=old_version
    )
except StaleDataError as e:
    # Option 1: Re-fetch and retry
    fresh_data = await repo.get_by_id(e.entity_id)
    # Re-apply changes and retry

    # Option 2: Notify user
    return {
        "error": "Data was modified by another user",
        "current_version": e.current_version,
        "message": "Please refresh and try again"
    }
```

### InsufficientBalanceError

**When raised:**
- Portfolio doesn't have enough cash for a buy order

**How to handle:**
```python
try:
    await repo.add_position(
        portfolio_id=1,
        stock_id=5,
        quantity=100,
        price=1000.00,
        transaction_type='buy'
    )
except InsufficientBalanceError as e:
    return {"error": str(e)}
```

### InvalidPositionError

**When raised:**
- Trying to sell more shares than owned

**How to handle:**
```python
try:
    await repo.add_position(
        portfolio_id=1,
        stock_id=5,
        quantity=100,
        price=100.00,
        transaction_type='sell'
    )
except InvalidPositionError as e:
    return {"error": str(e)}
```

## Lock Modes

### Standard Lock (Default)

```python
# Waits for lock to be available
portfolio = await repo.get_with_lock(
    id=portfolio_id,
    for_update=True
)
```

### NOWAIT

```python
# Raises exception immediately if locked
try:
    portfolio = await repo.get_with_lock(
        id=portfolio_id,
        for_update=True,
        nowait=True
    )
except OperationalError:
    return {"error": "Resource is locked by another user"}
```

### SKIP LOCKED

```python
# Returns None if locked (doesn't wait)
portfolio = await repo.get_with_lock(
    id=portfolio_id,
    for_update=True,
    skip_locked=True
)

if portfolio is None:
    return {"error": "Resource is currently locked"}
```

## Migration Guide

### Adding version column to existing models

```sql
-- Add version column with default value
ALTER TABLE portfolios ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE positions ADD COLUMN version INTEGER NOT NULL DEFAULT 1;

-- Note: investment_thesis already has version column
```

### Updating existing code

**Before:**
```python
# No version check
portfolio = await repo.get_by_id(portfolio_id)
portfolio.name = "New Name"
await session.commit()
```

**After:**
```python
# With optimistic locking
portfolio = await repo.get_by_id(portfolio_id)
current_version = portfolio.version

updated = await repo.update_with_lock(
    id=portfolio_id,
    data={"name": "New Name"},
    expected_version=current_version
)
```

## Testing Concurrent Updates

See `tests/security/test_row_locking.py` for comprehensive test suite:

- **12+ test scenarios** covering:
  - Optimistic lock version conflicts
  - Concurrent balance updates
  - Insufficient balance detection
  - Invalid position (oversell) prevention
  - Version increment verification
  - Lock modes (NOWAIT, SKIP LOCKED)
  - Concurrent buy/sell operations
  - Oversell prevention with locking

**Run tests:**
```bash
pytest tests/security/test_row_locking.py -v
```

## Performance Considerations

### Optimistic Locking
- **Pros:** No database locks, better concurrency
- **Cons:** Requires retry logic, can have conflicts
- **Best for:** Read-heavy workloads, low contention

### Pessimistic Locking
- **Pros:** Guaranteed consistency, no conflicts
- **Cons:** Blocks other transactions, potential deadlocks
- **Best for:** Critical financial operations, high contention

### Recommendations

1. **Portfolio balance updates** - Use pessimistic (SELECT FOR UPDATE)
2. **Position trades** - Use pessimistic (SELECT FOR UPDATE)
3. **Thesis updates** - Use optimistic (version check) with FOR UPDATE
4. **Read-only queries** - No locking needed

## Deadlock Prevention

### Best Practices

1. **Lock in consistent order** - Always lock Portfolio before Position
2. **Keep transactions short** - Acquire locks, update, commit quickly
3. **Use timeouts** - Don't hold locks indefinitely
4. **Retry logic** - Handle deadlock exceptions gracefully

**Example:**
```python
async def safe_trade(portfolio_id, stock_id, quantity, price):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with get_db_session() as session:
                # Lock portfolio FIRST (consistent order)
                portfolio = await session.execute(
                    select(Portfolio)
                    .where(Portfolio.id == portfolio_id)
                    .with_for_update()
                )

                # Then lock position
                position = await session.execute(
                    select(Position)
                    .where(...)
                    .with_for_update()
                )

                # Make updates
                # ...

                await session.commit()
                return True

        except OperationalError as e:
            if "deadlock" in str(e).lower():
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
            raise

    return False
```

## API Integration

### FastAPI Endpoint Example

```python
from fastapi import APIRouter, HTTPException
from backend.repositories.portfolio_repository import PortfolioRepository
from backend.exceptions import StaleDataError, InsufficientBalanceError

router = APIRouter()

@router.post("/portfolios/{portfolio_id}/positions")
async def add_position(
    portfolio_id: int,
    stock_id: int,
    quantity: float,
    price: float,
    expected_version: Optional[int] = None
):
    repo = PortfolioRepository()

    try:
        position = await repo.add_position(
            portfolio_id=portfolio_id,
            stock_id=stock_id,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            transaction_type='buy',
            expected_portfolio_version=expected_version
        )

        return {
            "success": True,
            "position": position,
            "portfolio_version": position.portfolio.version
        }

    except StaleDataError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "version_conflict",
                "message": str(e),
                "expected_version": e.expected_version,
                "current_version": e.current_version
            }
        )

    except InsufficientBalanceError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "insufficient_balance",
                "message": str(e)
            }
        )
```

## Monitoring

### Metrics to Track

1. **StaleDataError frequency** - High rate indicates contention
2. **Lock wait times** - Monitor FOR UPDATE blocking duration
3. **Deadlock occurrences** - Should be rare with proper ordering
4. **Transaction rollback rate** - Track conflicts and retries

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Log version conflicts
logger.warning(
    "Version conflict detected",
    extra={
        "entity_type": "Portfolio",
        "entity_id": portfolio_id,
        "expected_version": expected_version,
        "current_version": current_version
    }
)

# Log successful updates
logger.info(
    "Portfolio updated with lock",
    extra={
        "portfolio_id": portfolio_id,
        "old_version": old_version,
        "new_version": new_version,
        "operation": "add_position"
    }
)
```

## Security Implications

### OWASP Relevance

**A04:2021 – Insecure Design**
- Row locking prevents race conditions in financial operations
- Ensures atomic balance updates
- Prevents double-spending and overdrafts

**A01:2021 – Broken Access Control**
- Combined with user_id checks for authorization
- Prevents unauthorized modifications

### Compliance

**PCI-DSS Requirement 6.5.8:**
- Prevents race conditions in payment processing
- Ensures transactional integrity

**SOC 2 CC6.1:**
- Logical access controls to prevent concurrent update conflicts
- Audit trail of version changes

## Summary

Row locking implementation provides:
- ✅ **Optimistic locking** with version columns on Portfolio, Position, InvestmentThesis
- ✅ **Pessimistic locking** with SELECT FOR UPDATE for critical operations
- ✅ **Hybrid approach** combining both for maximum safety
- ✅ **Custom exceptions** for clear error handling (StaleDataError, InsufficientBalanceError, InvalidPositionError)
- ✅ **Comprehensive tests** - 12+ concurrent update scenarios
- ✅ **Production-ready** with proper logging, monitoring, and retry logic

**Result:** Zero lost updates, no race conditions in financial operations.
