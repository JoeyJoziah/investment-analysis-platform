# Type Annotation Guidelines

This document provides comprehensive guidelines for type annotations in the Investment Analysis Platform backend.

## Table of Contents
1. [Overview](#overview)
2. [Why Type Annotations Matter](#why-type-annotations-matter)
3. [Basic Type Annotations](#basic-type-annotations)
4. [Function Signatures](#function-signatures)
5. [Return Types](#return-types)
6. [Pydantic Models](#pydantic-models)
7. [Generic Types](#generic-types)
8. [Dict[str, Any] Usage](#dictstr-any-usage)
9. [Migration Guide](#migration-guide)
10. [Best Practices](#best-practices)
11. [Examples from Codebase](#examples-from-codebase)

## Overview

We use Python type hints with mypy for static type checking. Our target is **95%+ type coverage** across the backend codebase.

### Type Checking Stack
- **mypy**: Static type checker
- **Pydantic**: Runtime validation and type coercion
- **FastAPI**: Auto-generates OpenAPI specs from type hints
- **Pre-commit hooks**: Automatic type checking before commits

## Why Type Annotations Matter

1. **Early Error Detection**: Catch type errors before runtime
2. **Better IDE Support**: Autocomplete and inline documentation
3. **Self-Documenting Code**: Types serve as inline documentation
4. **Safer Refactoring**: Identify breaking changes automatically
5. **API Contract Clarity**: FastAPI uses types for request/response validation

## Basic Type Annotations

### Variables

```python
# Good: Explicit type annotations
name: str = "AAPL"
price: float = 150.25
volume: int = 1_000_000
is_active: bool = True

# Acceptable: Type inference is clear
name = "AAPL"  # str inferred
count = 42     # int inferred

# Bad: Ambiguous without annotation
data = get_stock_data()  # What type is this?

# Good: Clarify ambiguous types
data: Dict[str, Any] = get_stock_data()
```

### Collections

```python
from typing import List, Dict, Set, Tuple, Optional

# Lists
symbols: List[str] = ["AAPL", "GOOGL", "MSFT"]
prices: List[float] = [150.25, 2800.50, 350.75]

# Dictionaries
stock_data: Dict[str, float] = {"price": 150.25, "change": 2.50}
metadata: Dict[str, Any] = {"source": "api", "timestamp": 1234567890}

# Sets
unique_sectors: Set[str] = {"Technology", "Finance", "Healthcare"}

# Tuples (fixed size)
coordinates: Tuple[float, float] = (40.7128, -74.0060)
stock_info: Tuple[str, float, int] = ("AAPL", 150.25, 1_000_000)

# Optional values
description: Optional[str] = None
target_price: Optional[float] = 175.00
```

## Function Signatures

### Basic Functions

```python
# Good: Full type annotations
def calculate_return(
    initial_price: float,
    final_price: float,
    include_dividends: bool = False
) -> float:
    """Calculate investment return percentage."""
    return ((final_price - initial_price) / initial_price) * 100

# Good: Multiple return types
from typing import Union

def get_stock_price(
    symbol: str,
    as_string: bool = False
) -> Union[float, str]:
    """Get stock price as float or formatted string."""
    price = fetch_price(symbol)
    return f"${price:.2f}" if as_string else price

# Good: Complex return types
async def get_portfolio_summary(
    user_id: int
) -> Dict[str, Union[float, int, List[str]]]:
    """Get portfolio summary with mixed value types."""
    return {
        "total_value": 100000.50,
        "position_count": 15,
        "top_holdings": ["AAPL", "GOOGL", "MSFT"]
    }
```

### Async Functions

```python
from typing import List, Optional

async def fetch_stock_data(
    symbol: str,
    from_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Fetch stock data from external API."""
    # Implementation
    return data

async def get_multiple_stocks(
    symbols: List[str]
) -> List[Dict[str, Any]]:
    """Fetch data for multiple stocks concurrently."""
    tasks = [fetch_stock_data(symbol) for symbol in symbols]
    return await asyncio.gather(*tasks)
```

### Methods

```python
class StockAnalyzer:
    def __init__(self, symbol: str) -> None:
        self.symbol: str = symbol
        self.data: Optional[Dict[str, Any]] = None

    def analyze(self, period_days: int = 30) -> AnalysisResult:
        """Perform technical analysis."""
        # Implementation
        return result

    async def fetch_data(self) -> None:
        """Fetch stock data asynchronously."""
        self.data = await fetch_stock_data(self.symbol)

    @property
    def current_price(self) -> Optional[float]:
        """Get current stock price."""
        return self.data.get("price") if self.data else None
```

## Return Types

### API Endpoints (FastAPI)

```python
from fastapi import APIRouter
from backend.models.api_response import ApiResponse, success_response
from backend.models.schemas import Stock

router = APIRouter()

# Good: Explicit Pydantic response type
@router.get("/stocks/{symbol}")
async def get_stock(symbol: str) -> ApiResponse[Stock]:
    """Get stock details."""
    stock = await fetch_stock(symbol)
    return success_response(data=stock)

# Good: List response
@router.get("/stocks")
async def list_stocks(
    limit: int = 100
) -> ApiResponse[List[Stock]]:
    """List all stocks."""
    stocks = await fetch_all_stocks(limit)
    return success_response(data=stocks)

# Avoid: Dict[str, Any] when Pydantic model exists
@router.get("/stocks/{symbol}")
async def get_stock(symbol: str) -> ApiResponse[Dict[str, Any]]:  # Bad
    """This should use Stock model instead."""
    pass
```

### Service Layer

```python
class StockService:
    async def create_stock(
        self,
        stock_data: StockCreate
    ) -> Stock:
        """Create a new stock record."""
        # Implementation
        return stock

    async def get_stock_by_symbol(
        self,
        symbol: str
    ) -> Optional[Stock]:
        """Get stock by symbol, returns None if not found."""
        # Implementation
        return stock or None

    async def list_stocks(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Stock], int]:
        """List stocks with pagination.

        Returns:
            Tuple of (stocks, total_count)
        """
        # Implementation
        return stocks, total_count
```

## Pydantic Models

### Response Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class StockBase(BaseModel):
    """Base stock schema."""
    symbol: str = Field(..., min_length=1, max_length=10)
    name: str
    exchange: str

class StockCreate(StockBase):
    """Schema for creating a stock."""
    market_cap: Optional[float] = None
    sector: Optional[str] = None

class Stock(StockBase):
    """Full stock response schema."""
    id: int
    created_at: datetime
    current_price: Optional[float] = None

    class Config:
        from_attributes = True  # For SQLAlchemy models

# Using the models
async def create_stock(
    stock_data: StockCreate  # Type-safe input
) -> Stock:  # Type-safe output
    """Create stock with validated data."""
    # Pydantic validates stock_data automatically
    return Stock(**stock_data.dict(), id=1, created_at=datetime.now())
```

### Nested Models

```python
class Position(BaseModel):
    stock_id: int
    quantity: float
    average_cost: float

class Portfolio(BaseModel):
    id: int
    user_id: int
    cash_balance: float
    positions: List[Position]  # Nested model
    metadata: Dict[str, Any]  # Flexible metadata

# Forward references for circular dependencies
class PortfolioWithStocks(Portfolio):
    stocks: List['Stock']

# Update forward references
Portfolio.model_rebuild()
Stock.model_rebuild()
```

## Generic Types

### TypeVar for Generic Functions

```python
from typing import TypeVar, Generic, List, Callable

T = TypeVar('T')

def paginate(
    items: List[T],
    page: int,
    page_size: int
) -> List[T]:
    """Generic pagination that preserves type."""
    start = (page - 1) * page_size
    return items[start:start + page_size]

# Usage preserves type information
stocks: List[Stock] = [...]
page_stocks: List[Stock] = paginate(stocks, 1, 10)  # Still List[Stock]
```

### Generic Classes

```python
from typing import Generic, TypeVar, Optional

T = TypeVar('T')

class CacheManager(Generic[T]):
    """Generic cache manager."""

    def __init__(self) -> None:
        self._cache: Dict[str, T] = {}

    def get(self, key: str) -> Optional[T]:
        """Get cached value."""
        return self._cache.get(key)

    def set(self, key: str, value: T) -> None:
        """Set cached value."""
        self._cache[key] = value

# Usage with specific types
stock_cache: CacheManager[Stock] = CacheManager()
price_cache: CacheManager[float] = CacheManager()
```

## Dict[str, Any] Usage

### When to Use Dict[str, Any]

**Acceptable Use Cases:**

1. **External API Responses** (before parsing)
```python
async def fetch_from_api(url: str) -> Dict[str, Any]:
    """Raw API response before validation."""
    response = await client.get(url)
    return response.json()  # Unknown structure
```

2. **Flexible Configuration**
```python
class ServiceConfig(BaseModel):
    name: str
    version: str
    options: Dict[str, Any]  # Flexible options dict
```

3. **Metadata Fields**
```python
class Stock(BaseModel):
    symbol: str
    price: float
    metadata: Dict[str, Any]  # Additional flexible data
```

### When to Avoid Dict[str, Any]

**Create Pydantic Models Instead:**

```python
# Bad: Using Dict for structured data
@router.get("/stocks/{symbol}")
async def get_stock(symbol: str) -> ApiResponse[Dict[str, Any]]:
    return success_response(data={
        "symbol": "AAPL",
        "price": 150.25,
        "change": 2.50
    })

# Good: Use Pydantic model
class StockPrice(BaseModel):
    symbol: str
    price: float
    change: float

@router.get("/stocks/{symbol}")
async def get_stock(symbol: str) -> ApiResponse[StockPrice]:
    return success_response(data=StockPrice(
        symbol="AAPL",
        price=150.25,
        change=2.50
    ))
```

### Migrating from Dict[str, Any]

```python
# Step 1: Identify the structure
def analyze_stock(symbol: str) -> Dict[str, Any]:  # Current
    return {
        "symbol": symbol,
        "rsi": 65.5,
        "macd": {"value": 1.2, "signal": 1.0},
        "recommendation": "buy"
    }

# Step 2: Create Pydantic models
class MACDIndicator(BaseModel):
    value: float
    signal: float

class TechnicalAnalysis(BaseModel):
    symbol: str
    rsi: float
    macd: MACDIndicator
    recommendation: str

# Step 3: Update function signature
def analyze_stock(symbol: str) -> TechnicalAnalysis:  # Improved
    return TechnicalAnalysis(
        symbol=symbol,
        rsi=65.5,
        macd=MACDIndicator(value=1.2, signal=1.0),
        recommendation="buy"
    )
```

## Migration Guide

### Step-by-Step Migration Process

#### 1. Audit Current Code

```bash
# Find functions without return types
grep -r "def " backend/ | grep -v " -> "

# Find Dict[str, Any] usage
grep -r "Dict\[str, Any\]" backend/
```

#### 2. Create Pydantic Models

```python
# Before: Untyped dictionary
def get_user_stats(user_id: int):
    return {
        "total_value": 100000.50,
        "position_count": 15,
        "top_stocks": ["AAPL", "GOOGL"]
    }

# After: Typed model
class UserStats(BaseModel):
    total_value: float
    position_count: int
    top_stocks: List[str]

def get_user_stats(user_id: int) -> UserStats:
    return UserStats(
        total_value=100000.50,
        position_count=15,
        top_stocks=["AAPL", "GOOGL"]
    )
```

#### 3. Update Function Signatures

```python
# Before
async def fetch_data(symbol):
    result = await api.get(f"/stocks/{symbol}")
    return result

# After
async def fetch_data(symbol: str) -> Dict[str, Any]:
    result: Dict[str, Any] = await api.get(f"/stocks/{symbol}")
    return result
```

#### 4. Run mypy

```bash
# Check specific file
mypy backend/api/routers/stocks.py

# Check entire backend
mypy backend/

# Generate HTML report
mypy backend/ --html-report ./mypy-report
```

#### 5. Fix Type Errors

```python
# Common fixes:

# Error: Missing return type
def calculate(x, y):  # Error
    return x + y

def calculate(x: float, y: float) -> float:  # Fixed
    return x + y

# Error: Incompatible types
value: int = "123"  # Error
value: int = int("123")  # Fixed

# Error: Optional not handled
def get_name(user: User) -> str:
    return user.name  # Error if name is Optional[str]

def get_name(user: User) -> str:
    return user.name or "Unknown"  # Fixed
```

## Best Practices

### 1. Always Annotate Public APIs

```python
# Good: Public functions fully annotated
async def create_portfolio(
    user_id: int,
    name: str,
    initial_cash: float = 0.0
) -> Portfolio:
    """Create a new portfolio."""
    pass

# Acceptable: Private helper without annotations
def _calculate_helper(x, y):
    """Internal calculation helper."""
    return x * y + x / y
```

### 2. Use Specific Types Over Generic

```python
# Bad: Too generic
def process_data(data: Any) -> Any:
    pass

# Good: Specific types
def process_stock_data(data: Stock) -> AnalysisResult:
    pass
```

### 3. Leverage Union Types

```python
from typing import Union

def format_price(
    price: float,
    as_string: bool = False
) -> Union[float, str]:
    """Return price as float or formatted string."""
    return f"${price:.2f}" if as_string else price
```

### 4. Document Complex Types

```python
from typing import Dict, List, Tuple

def get_portfolio_breakdown(
    portfolio_id: int
) -> Dict[str, Tuple[float, List[str]]]:
    """Get portfolio breakdown by sector.

    Returns:
        Dictionary mapping sector name to tuple of (total_value, stock_symbols)

    Example:
        {
            "Technology": (50000.0, ["AAPL", "GOOGL"]),
            "Finance": (30000.0, ["JPM", "BAC"])
        }
    """
    pass
```

### 5. Use Type Aliases for Complex Types

```python
from typing import Dict, List, Tuple, TypeAlias

# Define reusable type aliases
StockSymbol: TypeAlias = str
Price: TypeAlias = float
Portfolio: TypeAlias = Dict[StockSymbol, Tuple[int, Price]]  # symbol -> (quantity, avg_price)

def calculate_portfolio_value(portfolio: Portfolio) -> float:
    """Calculate total portfolio value."""
    return sum(quantity * price for quantity, price in portfolio.values())
```

## Examples from Codebase

### Example 1: Monitoring Router (Before/After)

**Before:**
```python
@router.get("/health")
async def health_check() -> ApiResponse[Dict[str, Any]]:
    return success_response(data={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {...}
    })
```

**After:**
```python
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]

@router.get("/health")
async def health_check() -> ApiResponse[HealthCheckResponse]:
    return success_response(data=HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        services={...}
    ))
```

### Example 2: Cache Management (Existing Good Practice)

```python
class CacheMetricsResponse(BaseModel):
    """Cache metrics response"""
    timestamp: datetime
    hit_ratio: float
    total_requests: int
    api_calls_saved: int
    estimated_cost_savings: float
    storage_bytes: int

@router.get("/metrics")
async def get_cache_metrics() -> ApiResponse[Dict[str, Any]]:
    """Get comprehensive cache performance metrics"""
    cache_monitor = await get_cache_monitor()
    metrics = await cache_monitor.get_current_metrics()
    return success_response(data=metrics)
```

### Example 3: Admin Router (Already Well-Typed)

```python
class SystemHealth(BaseModel):
    status: SystemStatus
    uptime: int
    cpu_usage: float
    memory_usage: float
    # ... more fields

@router.get("/health")
async def get_system_health(
    current_user = Depends(check_admin_permission)
) -> ApiResponse[SystemHealth]:
    """Get comprehensive system health status"""
    return success_response(data=SystemHealth(...))
```

## Tools and Resources

### Running Type Checks

```bash
# Check single file
mypy backend/api/routers/stocks.py

# Check directory
mypy backend/api/routers/

# Check entire backend with config
mypy backend/ --config-file .mypy.ini

# Generate coverage report
mypy backend/ --html-report ./mypy-report

# Install pre-commit hook
pre-commit install
pre-commit run mypy --all-files
```

### IDE Integration

- **VS Code**: Install Pylance extension (automatic type checking)
- **PyCharm**: Built-in type checking
- **Vim/Neovim**: Use ALE or coc-pyright

### Additional Resources

- [mypy documentation](https://mypy.readthedocs.io/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)

## Checklist for New Code

- [ ] All function parameters have type annotations
- [ ] All functions have return type annotations
- [ ] Dict[str, Any] is only used for truly dynamic data
- [ ] Pydantic models are used for API responses
- [ ] Optional types are used for nullable values
- [ ] Complex types have documentation
- [ ] mypy passes without errors
- [ ] Pre-commit hooks pass

---

**Questions or Issues?**

Open an issue in the repository or contact the development team.
