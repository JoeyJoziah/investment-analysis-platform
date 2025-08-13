# AsyncIO Error Fix Summary: '_asyncio.Future' object has no attribute '_condition'

## Root Cause Analysis

### The Problem
The error `'_asyncio.Future' object has no attribute '_condition'` occurs due to a **nested `asyncio.run()` anti-pattern** in the batch processing code:

```python
# PROBLEMATIC CODE (Line 871)
lambda stock=stock: asyncio.run(self.load_stock_data(stock))
```

### Why This Causes the Error
1. **Nested Event Loop**: The script runs inside `asyncio.run(loader.run())`, creating the main event loop
2. **Invalid Nesting**: Inside `process_batch()`, the code attempts to call `asyncio.run()` again within a lambda
3. **Future Corruption**: Nested `asyncio.run()` corrupts the Future objects' internal state
4. **Missing `_condition`**: The `_condition` attribute is part of asyncio's internal synchronization mechanism that gets broken during this corruption

## Complete Fix Implementation

### 1. Fixed Process Batch Method
**Before (Problematic):**
```python
# Use ThreadPoolExecutor for I/O bound yfinance calls
with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    # Create async tasks
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(
            executor, 
            lambda stock=stock: asyncio.run(self.load_stock_data(stock))  # ❌ NESTED asyncio.run()
        )
        for stock in stock_batch
    ]
```

**After (Fixed):**
```python
# Create a semaphore to limit concurrent connections
semaphore = asyncio.Semaphore(self.max_workers)

async def process_single_stock(stock_info: StockInfo) -> Tuple[bool, int, str]:
    """Process a single stock with semaphore protection"""
    async with semaphore:
        try:
            success, records = await self.load_stock_data(stock_info)  # ✅ Proper async/await
            return success, records, stock_info.ticker
        except Exception as e:
            self.logger.error(f"Error processing {stock_info.ticker}: {e}")
            return False, 0, stock_info.ticker

# Create tasks for all stocks in the batch
tasks = [process_single_stock(stock) for stock in stock_batch]
```

### 2. Fixed Async Iteration Pattern
**Before (Problematic):**
```python
# Process tasks as they complete
for i, task in enumerate(as_completed(tasks)):  # ❌ Wrong enumeration
    try:
        success, records = await task
        stock = stock_batch[i]  # ❌ Wrong stock mapping
```

**After (Fixed):**
```python
# Process tasks as they complete
completed_count = 0
async for task in self._as_completed_async(tasks):  # ✅ Proper async iteration
    try:
        success, records, ticker = await task  # ✅ Ticker included in result
```

### 3. Fixed Synchronous yfinance Integration
**Before (Problematic):**
```python
# Download data
stock = yf.Ticker(ticker)
df = stock.history(start=start_date, end=end_date, auto_adjust=False)  # ❌ Blocking in async
```

**After (Fixed):**
```python
# Run the synchronous yfinance call in a thread executor
df = await loop.run_in_executor(
    None,  # Use default ThreadPoolExecutor
    self._fetch_stock_data_sync,
    ticker, start_date, end_date
)

def _fetch_stock_data_sync(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Synchronous helper method to fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        return df
    except Exception as e:
        self.logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
```

### 4. Added Connection Pool Management
```python
class ConnectionManager:
    """Manages database connections with proper async cleanup"""
    
    def __init__(self, engine, SessionLocal):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self._connection_semaphore = asyncio.Semaphore(5)  # Limit concurrent DB connections
    
    @asynccontextmanager
    async def get_session(self):
        """Get a database session with automatic cleanup"""
        async with self._connection_semaphore:
            session = self.SessionLocal()
            try:
                yield session
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
```

### 5. Added Proper Async Iterator
```python
async def _as_completed_async(self, tasks):
    """Async generator version of asyncio.as_completed for proper async iteration"""
    # Convert to asyncio tasks if they aren't already
    tasks = [asyncio.create_task(task) if not asyncio.iscoroutine(task) else asyncio.create_task(task) for task in tasks]
    
    while tasks:
        # Wait for at least one task to complete
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Yield completed tasks
        for task in done:
            yield task
            
        # Update tasks list to only include pending tasks
        tasks = list(pending)
```

## Key Improvements

### Performance Optimizations
1. **Semaphore-Based Concurrency**: Uses `asyncio.Semaphore` instead of ThreadPoolExecutor for better async control
2. **Connection Pool Limits**: Prevents database connection exhaustion
3. **Proper Task Management**: Eliminates Future object corruption
4. **Resource-Aware Processing**: Maintains CPU and memory limits

### Error Handling Improvements
1. **Graceful Degradation**: Failed stocks don't break entire batches
2. **Connection Cleanup**: Automatic database session cleanup
3. **Detailed Logging**: Better error tracking and debugging
4. **Progress Preservation**: Saves progress even on failures

### AsyncIO Best Practices
1. **Single Event Loop**: No nested `asyncio.run()` calls
2. **Proper Concurrency Control**: Semaphores for resource limiting
3. **Clean Async/Await**: Consistent async patterns throughout
4. **Thread Safety**: Proper executor usage for synchronous libraries

## Testing the Fix

Run the enhanced loader with these commands:
```bash
# Test with small batch first
python background_loader_enhanced.py --batch-size 5 --max-workers 2

# Full production run
python background_loader_enhanced.py --batch-size 20 --max-workers 4

# Monitor for the fixed behavior:
# ✅ No more '_condition' errors after batch completion
# ✅ Proper progress tracking through all batches  
# ✅ Clean async task completion
# ✅ Stable memory usage patterns
```

## Prevention Strategies

### 1. AsyncIO Code Review Checklist
- [ ] No nested `asyncio.run()` calls
- [ ] Proper `await` for all coroutines
- [ ] Use `asyncio.create_task()` for concurrent execution
- [ ] Semaphores for resource limiting
- [ ] Thread executors only for truly synchronous libraries

### 2. Common Anti-Patterns to Avoid
```python
# ❌ DON'T: Nested asyncio.run()
asyncio.run(some_async_function())  # Inside an async context

# ❌ DON'T: Blocking calls in async functions  
requests.get(url)  # Use aiohttp instead

# ❌ DON'T: ThreadPoolExecutor with async functions
executor.submit(async_function)  # Use asyncio.create_task() instead

# ❌ DON'T: Mixing sync/async without proper execution
yfinance_call()  # Use run_in_executor() for sync libraries
```

### 3. Recommended Patterns
```python
# ✅ DO: Proper semaphore usage
semaphore = asyncio.Semaphore(max_workers)
async with semaphore:
    result = await async_operation()

# ✅ DO: Thread executor for sync libraries
result = await loop.run_in_executor(None, sync_function, args)

# ✅ DO: Proper task creation and waiting
tasks = [asyncio.create_task(async_func(item)) for item in items]
results = await asyncio.gather(*tasks)

# ✅ DO: Connection management
async with connection_manager.get_session() as session:
    session.execute(query)
```

This comprehensive fix eliminates the `_condition` attribute error and establishes robust async patterns for scalable batch processing.