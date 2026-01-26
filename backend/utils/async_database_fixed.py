"""
Fixed Async Database Operations
Resolves asyncio Future object errors and connection pool issues
"""

import asyncio
import asyncpg
import logging
from contextlib import asynccontextmanager
from typing import Optional, Any, Dict, List
from datetime import datetime
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select
import traceback

logger = logging.getLogger(__name__)

class AsyncDatabaseManager:
    """
    Fixed async database manager that properly handles connection pools
    and resolves Future object errors
    """
    
    def __init__(self, database_url: str, max_connections: int = 20):
        self.database_url = database_url
        self.max_connections = max_connections
        self.engine = None
        self.async_session_factory = None
        self._connection_pool = None
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize async database engine with proper configuration"""
        try:
            # Detect database type from URL
            is_sqlite = 'sqlite' in self.database_url.lower()

            if is_sqlite:
                # SQLite with aiosqlite - StaticPool doesn't accept pool parameters
                engine_url = self.database_url
                if not engine_url.startswith('sqlite+aiosqlite://'):
                    engine_url = engine_url.replace('sqlite://', 'sqlite+aiosqlite://')

                self.engine = create_async_engine(
                    engine_url,
                    echo=False
                )
            else:
                # PostgreSQL with asyncpg - use connection pooling
                engine_url = self.database_url
                if engine_url.startswith('postgresql://'):
                    engine_url = engine_url.replace('postgresql://', 'postgresql+asyncpg://')
                elif engine_url.startswith('postgres://'):
                    engine_url = engine_url.replace('postgres://', 'postgresql+asyncpg://')

                self.engine = create_async_engine(
                    engine_url,
                    pool_size=10,
                    max_overflow=self.max_connections - 10,
                    pool_timeout=30,
                    pool_recycle=3600,
                    pool_pre_ping=True,
                    echo=False
                )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._is_initialized = True
            logger.info("Async database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async database: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def close(self):
        """Properly close all connections"""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get async session with proper error handling"""
        if not self._is_initialized:
            await self.initialize()
        
        session = None
        try:
            session = self.async_session_factory()
            yield session
            await session.commit()
        except Exception as e:
            if session:
                await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            if session:
                await session.close()
    
    async def execute_query(self, query: str, params: Dict = None):
        """Execute raw SQL query safely"""
        async with self.get_session() as session:
            try:
                result = await session.execute(text(query), params or {})
                return result.fetchall()
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise
    
    async def insert_stock_safe(self, stock_data: Dict) -> Optional[int]:
        """
        Safely insert stock with proper error handling
        Resolves the "Error creating stock record" issue
        """
        async with self.get_session() as session:
            try:
                # First, verify exchange exists
                exchange_query = select(text("id FROM exchanges WHERE code = :code"))
                result = await session.execute(exchange_query, {"code": stock_data.get("exchange", "NASDAQ")})
                exchange_row = result.fetchone()
                
                if not exchange_row:
                    logger.error(f"Exchange {stock_data.get('exchange')} not found")
                    return None
                
                exchange_id = exchange_row[0]
                
                # Insert stock record
                insert_query = text("""
                    INSERT INTO stocks (ticker, name, exchange_id, is_active, is_tradeable, last_updated)
                    VALUES (:ticker, :name, :exchange_id, :is_active, :is_tradeable, :last_updated)
                    ON CONFLICT (ticker) DO UPDATE SET
                        name = EXCLUDED.name,
                        last_updated = EXCLUDED.last_updated
                    RETURNING id
                """)
                
                result = await session.execute(insert_query, {
                    "ticker": stock_data["ticker"],
                    "name": stock_data.get("name", stock_data["ticker"]),
                    "exchange_id": exchange_id,
                    "is_active": stock_data.get("is_active", True),
                    "is_tradeable": stock_data.get("is_tradeable", True),
                    "last_updated": datetime.utcnow()
                })
                
                stock_row = result.fetchone()
                if stock_row:
                    stock_id = stock_row[0]
                    logger.info(f"Successfully processed stock {stock_data['ticker']} (ID: {stock_id})")
                    return stock_id
                else:
                    logger.error(f"Failed to get stock ID for {stock_data['ticker']}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error inserting stock {stock_data.get('ticker', 'UNKNOWN')}: {e}")
                logger.error(traceback.format_exc())
                return None

class BatchProcessor:
    """
    Fixed batch processor that resolves asyncio Future object errors
    """
    
    def __init__(self, db_manager: AsyncDatabaseManager, batch_size: int = 50):
        self.db_manager = db_manager
        self.batch_size = batch_size
        self.processed_count = 0
        self.error_count = 0
        self.errors = []
    
    async def process_stock_batch(self, stock_list: List[Dict]) -> Dict[str, Any]:
        """
        Process stocks in batches with proper error handling
        Fixes: '_asyncio.Future' object has no attribute '_condition'
        """
        try:
            batch_results = {
                "processed": 0,
                "errors": 0,
                "error_details": []
            }
            
            # Process stocks in smaller batches to avoid connection issues
            for i in range(0, len(stock_list), self.batch_size):
                batch = stock_list[i:i + self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}, stocks {i+1}-{min(i+self.batch_size, len(stock_list))}")
                
                # Process each stock in the batch
                batch_tasks = []
                for stock_data in batch:
                    task = self._process_single_stock(stock_data)
                    batch_tasks.append(task)
                
                # Execute batch with proper error handling
                try:
                    # Wait for all tasks in batch with timeout
                    batch_outcomes = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=300.0  # 5 minute timeout per batch
                    )
                    
                    # Process results
                    for j, outcome in enumerate(batch_outcomes):
                        if isinstance(outcome, Exception):
                            batch_results["errors"] += 1
                            error_detail = {
                                "ticker": batch[j].get("ticker", "UNKNOWN"),
                                "error": str(outcome),
                                "batch": i//self.batch_size + 1
                            }
                            batch_results["error_details"].append(error_detail)
                            logger.error(f"Error processing {batch[j].get('ticker', 'UNKNOWN')}: {outcome}")
                        elif outcome is not None:
                            batch_results["processed"] += 1
                        else:
                            batch_results["errors"] += 1
                            batch_results["error_details"].append({
                                "ticker": batch[j].get("ticker", "UNKNOWN"),
                                "error": "No result returned",
                                "batch": i//self.batch_size + 1
                            })
                    
                    logger.info(f"Batch {i//self.batch_size + 1} completed: {len([o for o in batch_outcomes if not isinstance(o, Exception)])} successful, {len([o for o in batch_outcomes if isinstance(o, Exception)])} errors")
                    
                    # Small delay between batches to prevent overwhelming the database
                    await asyncio.sleep(0.1)
                    
                except asyncio.TimeoutError:
                    logger.error(f"Batch {i//self.batch_size + 1} timed out")
                    batch_results["errors"] += len(batch)
                    for stock_data in batch:
                        batch_results["error_details"].append({
                            "ticker": stock_data.get("ticker", "UNKNOWN"),
                            "error": "Batch timeout",
                            "batch": i//self.batch_size + 1
                        })
                
                except Exception as e:
                    logger.error(f"Batch {i//self.batch_size + 1} failed with error: {e}")
                    logger.error(traceback.format_exc())
                    batch_results["errors"] += len(batch)
                    for stock_data in batch:
                        batch_results["error_details"].append({
                            "ticker": stock_data.get("ticker", "UNKNOWN"),
                            "error": f"Batch error: {str(e)}",
                            "batch": i//self.batch_size + 1
                        })
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Critical error in batch processing: {e}")
            logger.error(traceback.format_exc())
            return {
                "processed": 0,
                "errors": len(stock_list),
                "error_details": [{"error": f"Critical batch error: {str(e)}"}]
            }
    
    async def _process_single_stock(self, stock_data: Dict) -> Optional[int]:
        """Process a single stock with enhanced error handling"""
        try:
            # Validate required fields
            if not stock_data.get("ticker"):
                logger.error("Stock data missing ticker")
                return None
            
            # Clean and validate ticker
            ticker = str(stock_data["ticker"]).upper().strip()
            if not ticker or len(ticker) > 10:
                logger.error(f"Invalid ticker: {ticker}")
                return None
            
            # Prepare clean stock data
            clean_data = {
                "ticker": ticker,
                "name": stock_data.get("name", ticker),
                "exchange": stock_data.get("exchange", "NASDAQ"),
                "is_active": stock_data.get("is_active", True),
                "is_tradeable": stock_data.get("is_tradeable", True)
            }
            
            # Insert with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await self.db_manager.insert_stock_safe(clean_data)
                    if result is not None:
                        return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1} for {ticker}: {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing stock {stock_data.get('ticker', 'UNKNOWN')}: {e}")
            return None

class DataQualityValidator:
    """Enhanced data quality validation for price data"""
    
    @staticmethod
    def validate_price_data(price_data: Dict) -> tuple[bool, List[str]]:
        """
        Validate price data quality
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Required fields check
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in price_data or price_data[field] is None:
                    issues.append(f"Missing required field: {field}")
            
            if issues:
                return False, issues
            
            # Extract values
            open_price = float(price_data['open'])
            high_price = float(price_data['high'])
            low_price = float(price_data['low'])
            close_price = float(price_data['close'])
            volume = int(price_data['volume'])
            
            # Price positivity check
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                issues.append("Negative or zero prices detected")
            
            # Volume check
            if volume < 0:
                issues.append("Negative volume")
            
            # OHLC relationship checks
            if high_price < max(open_price, close_price):
                issues.append("High price less than open/close")
            
            if low_price > min(open_price, close_price):
                issues.append("Low price greater than open/close")
            
            if high_price < low_price:
                issues.append("High price less than low price")
            
            # Extreme price change check (>50%)
            if 'previous_close' in price_data and price_data['previous_close']:
                prev_close = float(price_data['previous_close'])
                if prev_close > 0:
                    change_pct = abs((close_price - prev_close) / prev_close) * 100
                    if change_pct > 50:
                        issues.append(f"Excessive price change detected ({change_pct:.1f}%)")
            
            # Price spike detection
            intraday_change = abs((high_price - low_price) / low_price) * 100
            if intraday_change > 100:  # 100% intraday range
                issues.append(f"Extreme intraday volatility ({intraday_change:.1f}%)")
            
            return len(issues) == 0, issues
            
        except (ValueError, TypeError) as e:
            issues.append(f"Data type error: {e}")
            return False, issues
        except Exception as e:
            issues.append(f"Validation error: {e}")
            return False, issues

# Example usage and testing functions
async def test_async_database():
    """Test the fixed async database functionality"""
    db_manager = AsyncDatabaseManager("postgresql://user:pass@localhost/investment_db")
    
    try:
        await db_manager.initialize()
        
        # Test basic query
        result = await db_manager.execute_query("SELECT COUNT(*) FROM exchanges")
        print(f"Exchange count: {result}")
        
        # Test stock insertion
        test_stock = {
            "ticker": "TEST",
            "name": "Test Company",
            "exchange": "NASDAQ"
        }
        
        stock_id = await db_manager.insert_stock_safe(test_stock)
        print(f"Test stock ID: {stock_id}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        await db_manager.close()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_async_database())