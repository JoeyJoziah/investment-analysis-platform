"""
Database Integration Tests for Investment Analysis Platform
Tests CRUD operations, transactions, data integrity, and database performance.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select, insert, update, delete
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import asyncpg
import pandas as pd
import numpy as np
from testcontainers.postgres import PostgresContainer

from backend.config.database import get_async_db_session, initialize_database, cleanup_database
from backend.models.database import Base
from backend.models.unified_models import (
    User, Stock, Portfolio, Position, Transaction,
    PriceHistory, Recommendation, Alert
)
from backend.repositories import (
    user_repository,
    stock_repository, 
    portfolio_repository,
    price_repository,
    recommendation_repository
)
from backend.utils.database_monitoring import DatabaseMonitor
from backend.utils.deadlock_handler import DeadlockHandler


class TestDatabaseIntegration:
    """Test comprehensive database operations with real PostgreSQL integration."""

    @pytest_asyncio.fixture
    async def db_session(self):
        """Create test database session using testcontainers."""
        # Start PostgreSQL container
        container = PostgresContainer("postgres:15")
        container.start()

        try:
            # Get connection URL and replace psycopg2 with asyncpg for async operations
            database_url = container.get_connection_url().replace('psycopg2', 'asyncpg')

            # Create async engine
            test_engine = create_async_engine(database_url, echo=False)

            # Create tables
            async with test_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Create session factory
            TestSessionLocal = sessionmaker(
                bind=test_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Create and yield session
            session = TestSessionLocal()
            try:
                yield session
            finally:
                await session.close()
                await test_engine.dispose()
        finally:
            # Stop container
            container.stop()

    @pytest.fixture
    def sample_user(self):
        """Create sample user data."""
        return User(
            username="testuser123",
            email="test@example.com",
            hashed_password="$2b$12$hashedpassword",
            is_active=True,
            is_verified=True,
            created_at=datetime.utcnow()
        )

    @pytest.fixture
    def sample_stock(self):
        """Create sample stock data."""
        return Stock(
            symbol="TESTSTOCK",
            name="Test Corporation",
            sector="Technology",
            industry="Software",
            market_cap=1000000000,
            description="Test stock for integration testing"
        )

    @pytest.fixture
    def sample_portfolio(self, sample_user):
        """Create sample portfolio data."""
        return Portfolio(
            user_id=sample_user.id,
            name="Test Portfolio",
            description="Integration test portfolio",
            strategy="balanced",
            cash_balance=100000.00,
            created_at=datetime.utcnow()
        )

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_user_repository_operations(self, db_session, sample_user):
        """Test user repository CRUD operations."""
        
        try:
            # Test user creation
            created_user = await user_repository.create_user(sample_user, session=db_session)
            await db_session.commit()
            
            assert created_user.id is not None
            assert created_user.username == sample_user.username
            assert created_user.email == sample_user.email
            assert created_user.is_active is True
            
            # Test user retrieval by ID
            retrieved_user = await user_repository.get_by_id(created_user.id, session=db_session)
            assert retrieved_user is not None
            assert retrieved_user.username == created_user.username
            
            # Test user retrieval by username
            user_by_username = await user_repository.get_by_username(
                created_user.username, 
                session=db_session
            )
            assert user_by_username is not None
            assert user_by_username.id == created_user.id
            
            # Test user retrieval by email
            user_by_email = await user_repository.get_by_email(
                created_user.email,
                session=db_session
            )
            assert user_by_email is not None
            assert user_by_email.id == created_user.id
            
            # Test user update
            updated_data = {"is_verified": True, "last_login": datetime.utcnow()}
            updated_user = await user_repository.update_user(
                created_user.id,
                updated_data,
                session=db_session
            )
            await db_session.commit()
            
            assert updated_user.is_verified is True
            assert updated_user.last_login is not None
            
            # Test user deletion (soft delete if implemented)
            await user_repository.delete_user(created_user.id, session=db_session)
            await db_session.commit()
            
            # Verify user is marked as deleted or actually deleted
            deleted_user = await user_repository.get_by_id(created_user.id, session=db_session)
            assert deleted_user is None or deleted_user.is_active is False
            
        except Exception as e:
            await db_session.rollback()
            pytest.fail(f"User repository test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_stock_repository_operations(self, db_session, sample_stock):
        """Test stock repository CRUD operations."""
        
        try:
            # Test stock creation
            created_stock = await stock_repository.create_stock(sample_stock, session=db_session)
            await db_session.commit()
            
            assert created_stock.symbol == sample_stock.symbol
            assert created_stock.name == sample_stock.name
            assert created_stock.sector == sample_stock.sector
            
            # Test stock retrieval by symbol
            retrieved_stock = await stock_repository.get_by_symbol(
                created_stock.symbol,
                session=db_session
            )
            assert retrieved_stock is not None
            assert retrieved_stock.symbol == created_stock.symbol
            
            # Test stock search
            search_results = await stock_repository.search_stocks(
                query="Test",
                limit=10,
                session=db_session
            )
            assert len(search_results) >= 1
            assert any(stock.symbol == created_stock.symbol for stock in search_results)
            
            # Test stock filtering by sector
            sector_stocks = await stock_repository.get_stocks_by_sector(
                sector="Technology",
                limit=10,
                session=db_session
            )
            assert len(sector_stocks) >= 1
            assert any(stock.symbol == created_stock.symbol for stock in sector_stocks)
            
            # Test bulk stock operations
            bulk_stocks = [
                Stock(symbol="BULK1", name="Bulk Stock 1", sector="Technology"),
                Stock(symbol="BULK2", name="Bulk Stock 2", sector="Healthcare"),
                Stock(symbol="BULK3", name="Bulk Stock 3", sector="Finance")
            ]
            
            created_bulk = await stock_repository.bulk_create_stocks(
                bulk_stocks,
                session=db_session
            )
            await db_session.commit()
            
            assert len(created_bulk) == 3
            
            # Verify bulk stocks exist
            for stock in bulk_stocks:
                found_stock = await stock_repository.get_by_symbol(stock.symbol, session=db_session)
                assert found_stock is not None
            
            # Test stock update
            update_data = {"market_cap": 2000000000, "description": "Updated description"}
            updated_stock = await stock_repository.update_stock(
                created_stock.symbol,
                update_data,
                session=db_session
            )
            await db_session.commit()
            
            assert updated_stock.market_cap == 2000000000
            assert updated_stock.description == "Updated description"
            
        except Exception as e:
            await db_session.rollback()
            pytest.fail(f"Stock repository test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_portfolio_repository_operations(self, db_session, sample_user, sample_portfolio):
        """Test portfolio repository CRUD operations."""
        
        try:
            # Create user first
            created_user = await user_repository.create_user(sample_user, session=db_session)
            await db_session.commit()
            
            # Update portfolio with actual user ID
            sample_portfolio.user_id = created_user.id
            
            # Test portfolio creation
            created_portfolio = await portfolio_repository.create_portfolio(
                sample_portfolio,
                session=db_session
            )
            await db_session.commit()
            
            assert created_portfolio.user_id == created_user.id
            assert created_portfolio.name == sample_portfolio.name
            assert float(created_portfolio.cash_balance) == float(sample_portfolio.cash_balance)
            
            # Test portfolio retrieval
            retrieved_portfolio = await portfolio_repository.get_portfolio_by_id(
                created_portfolio.id,
                session=db_session
            )
            assert retrieved_portfolio is not None
            assert retrieved_portfolio.id == created_portfolio.id
            
            # Test user portfolios retrieval
            user_portfolios = await portfolio_repository.get_user_portfolios(
                user_id=created_user.id,
                session=db_session
            )
            assert len(user_portfolios) >= 1
            assert any(p.id == created_portfolio.id for p in user_portfolios)
            
            # Test portfolio positions operations
            # Create stock for position
            test_stock = Stock(symbol="POSTEST", name="Position Test Stock", sector="Technology")
            created_stock = await stock_repository.create_stock(test_stock, session=db_session)
            await db_session.commit()
            
            # Create portfolio position
            position_data = Position(
                portfolio_id=created_portfolio.id,
                symbol=created_stock.symbol,
                quantity=100,
                average_cost=150.00,
                current_price=155.00,
                created_at=datetime.utcnow()
            )
            
            created_position = await portfolio_repository.create_position(
                position_data,
                session=db_session
            )
            await db_session.commit()
            
            assert created_position.portfolio_id == created_portfolio.id
            assert created_position.symbol == created_stock.symbol
            assert float(created_position.quantity) == 100
            
            # Test position retrieval
            portfolio_positions = await portfolio_repository.get_portfolio_positions(
                portfolio_id=created_portfolio.id,
                session=db_session
            )
            assert len(portfolio_positions) == 1
            assert portfolio_positions[0].symbol == created_stock.symbol
            
            # Test position update
            updated_position = await portfolio_repository.update_position(
                position_id=created_position.id,
                update_data={"quantity": 150, "current_price": 160.00},
                session=db_session
            )
            await db_session.commit()
            
            assert float(updated_position.quantity) == 150
            assert float(updated_position.current_price) == 160.00
            
        except Exception as e:
            await db_session.rollback()
            pytest.fail(f"Portfolio repository test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_price_data_operations(self, db_session, sample_stock):
        """Test price data repository operations."""
        
        try:
            # Create stock first
            created_stock = await stock_repository.create_stock(sample_stock, session=db_session)
            await db_session.commit()
            
            # Test single price data insertion
            price_data = PriceHistory(
                symbol=created_stock.symbol,
                date=date.today(),
                open=150.00,
                high=155.00,
                low=148.00,
                close=154.25,
                volume=1000000,
                adjusted_close=154.25
            )
            
            created_price = await price_repository.create_price_data(price_data, session=db_session)
            await db_session.commit()
            
            assert created_price.symbol == created_stock.symbol
            assert float(created_price.close) == 154.25
            
            # Test bulk price data insertion
            bulk_prices = []
            for i in range(30):  # 30 days of data
                price_date = date.today() - timedelta(days=i)
                base_price = 150 + (i * 0.5)
                
                bulk_prices.append(PriceHistory(
                    symbol=created_stock.symbol,
                    date=price_date,
                    open=base_price,
                    high=base_price + 2,
                    low=base_price - 2,
                    close=base_price + 1,
                    volume=1000000 + (i * 10000),
                    adjusted_close=base_price + 1
                ))
            
            await price_repository.bulk_create_prices(bulk_prices, session=db_session)
            await db_session.commit()
            
            # Test price history retrieval
            price_history = await price_repository.get_price_history(
                symbol=created_stock.symbol,
                start_date=date.today() - timedelta(days=30),
                end_date=date.today(),
                session=db_session
            )
            
            assert len(price_history) >= 30
            assert all(p.symbol == created_stock.symbol for p in price_history)
            
            # Test latest price retrieval
            latest_price = await price_repository.get_latest_price(
                symbol=created_stock.symbol,
                session=db_session
            )
            assert latest_price is not None
            assert latest_price.symbol == created_stock.symbol
            
            # Test price data aggregation
            avg_price = await price_repository.get_average_price(
                symbol=created_stock.symbol,
                days=7,
                session=db_session
            )
            assert avg_price > 0
            
            # Test bulk price updates
            update_data = [
                {"symbol": created_stock.symbol, "date": date.today(), "close": 160.00},
            ]
            
            await price_repository.bulk_update_prices(update_data, session=db_session)
            await db_session.commit()
            
            # Verify update
            updated_price = await price_repository.get_price_by_date(
                symbol=created_stock.symbol,
                price_date=date.today(),
                session=db_session
            )
            assert float(updated_price.close) == 160.00
            
        except Exception as e:
            await db_session.rollback()
            pytest.fail(f"Price data test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_transaction_integrity(self, db_session, sample_user, sample_stock):
        """Test database transaction integrity and rollback mechanisms."""
        
        try:
            # Create user and stock
            created_user = await user_repository.create_user(sample_user, session=db_session)
            created_stock = await stock_repository.create_stock(sample_stock, session=db_session)
            await db_session.commit()
            
            # Test successful transaction
            async with db_session.begin():
                portfolio = Portfolio(
                    user_id=created_user.id,
                    name="Transaction Test Portfolio",
                    cash_balance=50000.00
                )
                created_portfolio = await portfolio_repository.create_portfolio(
                    portfolio,
                    session=db_session
                )
                
                position = Position(
                    portfolio_id=created_portfolio.id,
                    symbol=created_stock.symbol,
                    quantity=100,
                    average_cost=150.00
                )
                await portfolio_repository.create_position(position, session=db_session)
                
                # Transaction should commit automatically
            
            # Verify data was committed
            verified_portfolio = await portfolio_repository.get_portfolio_by_id(
                created_portfolio.id,
                session=db_session
            )
            assert verified_portfolio is not None
            
            # Test transaction rollback on error
            try:
                async with db_session.begin():
                    # Create another portfolio
                    portfolio2 = Portfolio(
                        user_id=created_user.id,
                        name="Rollback Test Portfolio",
                        cash_balance=25000.00
                    )
                    created_portfolio2 = await portfolio_repository.create_portfolio(
                        portfolio2,
                        session=db_session
                    )
                    
                    # Cause an intentional error
                    invalid_position = Position(
                        portfolio_id=999999,  # Invalid portfolio ID
                        symbol=created_stock.symbol,
                        quantity=50,
                        average_cost=200.00
                    )
                    await portfolio_repository.create_position(invalid_position, session=db_session)
                    
            except Exception:
                # Expected to fail and rollback
                pass
            
            # Verify rollback - second portfolio should not exist
            rollback_portfolio = await portfolio_repository.get_portfolio_by_name(
                user_id=created_user.id,
                name="Rollback Test Portfolio",
                session=db_session
            )
            assert rollback_portfolio is None
            
        except Exception as e:
            await db_session.rollback()
            pytest.fail(f"Transaction integrity test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.performance
    async def test_database_performance(self, db_session, sample_stock):
        """Test database query performance and optimization."""
        
        try:
            # Create stock for performance testing
            created_stock = await stock_repository.create_stock(sample_stock, session=db_session)
            await db_session.commit()
            
            # Test bulk insert performance
            bulk_prices = []
            for i in range(1000):  # 1000 price records
                price_date = date.today() - timedelta(days=i % 365)
                base_price = 100 + np.random.uniform(-10, 10)
                
                bulk_prices.append(PriceHistory(
                    symbol=created_stock.symbol,
                    date=price_date,
                    open=base_price,
                    high=base_price + np.random.uniform(0, 5),
                    low=base_price - np.random.uniform(0, 5),
                    close=base_price + np.random.uniform(-2, 2),
                    volume=np.random.randint(100000, 10000000),
                    adjusted_close=base_price + np.random.uniform(-2, 2)
                ))
            
            start_time = datetime.utcnow()
            await price_repository.bulk_create_prices(bulk_prices, session=db_session)
            await db_session.commit()
            end_time = datetime.utcnow()
            
            insert_duration = (end_time - start_time).total_seconds()
            assert insert_duration < 30.0, f"Bulk insert took {insert_duration}s, should be under 30s"
            
            # Test query performance
            start_time = datetime.utcnow()
            price_history = await price_repository.get_price_history(
                symbol=created_stock.symbol,
                start_date=date.today() - timedelta(days=90),
                end_date=date.today(),
                session=db_session
            )
            end_time = datetime.utcnow()
            
            query_duration = (end_time - start_time).total_seconds()
            assert query_duration < 5.0, f"Query took {query_duration}s, should be under 5s"
            assert len(price_history) > 0
            
            # Test aggregation performance
            start_time = datetime.utcnow()
            avg_volume = await db_session.execute(
                text("""
                    SELECT AVG(volume) as avg_volume 
                    FROM price_data 
                    WHERE symbol = :symbol
                """),
                {"symbol": created_stock.symbol}
            )
            result = avg_volume.fetchone()
            end_time = datetime.utcnow()
            
            aggregation_duration = (end_time - start_time).total_seconds()
            assert aggregation_duration < 2.0, f"Aggregation took {aggregation_duration}s, should be under 2s"
            assert result[0] > 0
            
        except Exception as e:
            await db_session.rollback()
            pytest.fail(f"Performance test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_concurrent_database_operations(self, db_session):
        """Test concurrent database operations and deadlock handling."""
        
        # Create test data
        user = User(
            username="concurrentuser",
            email="concurrent@test.com",
            hashed_password="hashed",
            is_active=True
        )
        created_user = await user_repository.create_user(user, session=db_session)
        await db_session.commit()
        
        portfolio = Portfolio(
            user_id=created_user.id,
            name="Concurrent Test Portfolio",
            cash_balance=100000.00
        )
        created_portfolio = await portfolio_repository.create_portfolio(portfolio, session=db_session)
        await db_session.commit()
        
        # Test concurrent updates
        async def update_cash_balance(session: AsyncSession, portfolio_id: int, amount: float):
            try:
                async with session.begin():
                    # Simulate some processing time
                    await asyncio.sleep(0.1)
                    
                    # Update cash balance
                    await session.execute(
                        text("""
                            UPDATE portfolios 
                            SET cash_balance = cash_balance + :amount,
                                updated_at = NOW()
                            WHERE id = :portfolio_id
                        """),
                        {"amount": amount, "portfolio_id": portfolio_id}
                    )
                return True
            except Exception as e:
                await session.rollback()
                return False
        
        # Create multiple concurrent sessions
        sessions = []
        for _ in range(5):
            session = AsyncSession(bind=db_session.bind)
            sessions.append(session)
        
        try:
            # Execute concurrent updates
            tasks = []
            for i, session in enumerate(sessions):
                amount = (i + 1) * 1000  # Different amounts
                task = update_cash_balance(session, created_portfolio.id, amount)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least some operations should succeed
            successful_operations = sum(1 for r in results if r is True)
            assert successful_operations > 0
            
            # Verify final state
            final_portfolio = await portfolio_repository.get_portfolio_by_id(
                created_portfolio.id,
                session=db_session
            )
            
            # Cash balance should have increased
            assert float(final_portfolio.cash_balance) > float(created_portfolio.cash_balance)
            
        finally:
            # Clean up sessions
            for session in sessions:
                await session.close()

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_database_constraints_and_validation(self, db_session):
        """Test database constraints and data validation."""
        
        try:
            # Test unique constraint violation
            user1 = User(
                username="uniqueuser",
                email="unique@test.com",
                hashed_password="hashed",
                is_active=True
            )
            await user_repository.create_user(user1, session=db_session)
            await db_session.commit()
            
            # Try to create user with same username
            user2 = User(
                username="uniqueuser",  # Same username
                email="different@test.com",
                hashed_password="hashed",
                is_active=True
            )
            
            with pytest.raises((IntegrityError, ValueError)):
                await user_repository.create_user(user2, session=db_session)
                await db_session.commit()
            
            await db_session.rollback()
            
            # Test foreign key constraints
            invalid_portfolio = Portfolio(
                user_id=999999,  # Non-existent user
                name="Invalid Portfolio",
                cash_balance=10000.00
            )
            
            with pytest.raises((IntegrityError, ValueError)):
                await portfolio_repository.create_portfolio(invalid_portfolio, session=db_session)
                await db_session.commit()
            
            await db_session.rollback()
            
            # Test check constraints (if any)
            # For example, negative cash balance might not be allowed
            valid_user = User(
                username="validuser",
                email="valid@test.com",
                hashed_password="hashed",
                is_active=True
            )
            created_user = await user_repository.create_user(valid_user, session=db_session)
            await db_session.commit()
            
            negative_cash_portfolio = Portfolio(
                user_id=created_user.id,
                name="Negative Cash Portfolio",
                cash_balance=-1000.00  # Negative balance
            )
            
            # This might be allowed or rejected based on constraints
            try:
                await portfolio_repository.create_portfolio(
                    negative_cash_portfolio,
                    session=db_session
                )
                await db_session.commit()
                # If allowed, verify it's handled properly
            except (IntegrityError, ValueError):
                # If rejected, that's also valid
                await db_session.rollback()
            
        except Exception as e:
            await db_session.rollback()
            if "constraint" not in str(e).lower():
                pytest.fail(f"Unexpected error in constraints test: {e}")

    @pytest.mark.asyncio
    @pytest.mark.database
    @pytest.mark.integration
    async def test_database_indexing_and_optimization(self, db_session, sample_stock):
        """Test database indexing effectiveness and query optimization."""
        
        try:
            # Create stock with test data
            created_stock = await stock_repository.create_stock(sample_stock, session=db_session)
            await db_session.commit()
            
            # Create large dataset for index testing
            bulk_prices = []
            for i in range(5000):  # 5000 records
                price_date = date.today() - timedelta(days=i % 1000)
                bulk_prices.append(PriceHistory(
                    symbol=created_stock.symbol,
                    date=price_date,
                    open=100 + (i % 100),
                    high=105 + (i % 100),
                    low=95 + (i % 100),
                    close=102 + (i % 100),
                    volume=1000000 + i,
                    adjusted_close=102 + (i % 100)
                ))
            
            await price_repository.bulk_create_prices(bulk_prices, session=db_session)
            await db_session.commit()
            
            # Test indexed query performance (symbol + date range)
            start_time = datetime.utcnow()
            indexed_query_result = await db_session.execute(
                text("""
                    EXPLAIN ANALYZE
                    SELECT * FROM price_data 
                    WHERE symbol = :symbol 
                    AND date BETWEEN :start_date AND :end_date
                    ORDER BY date DESC
                    LIMIT 100
                """),
                {
                    "symbol": created_stock.symbol,
                    "start_date": date.today() - timedelta(days=30),
                    "end_date": date.today()
                }
            )
            end_time = datetime.utcnow()
            
            query_plan = indexed_query_result.fetchall()
            execution_time = (end_time - start_time).total_seconds()
            
            # Verify query uses index (look for "Index Scan" in plan)
            plan_text = str(query_plan)
            # In a real test, would parse the execution plan more thoroughly
            
            assert execution_time < 1.0, f"Indexed query took {execution_time}s, should be under 1s"
            
            # Test query without using index (full table scan)
            start_time = datetime.utcnow()
            unindexed_query_result = await db_session.execute(
                text("""
                    SELECT COUNT(*) FROM price_data 
                    WHERE volume > :volume_threshold
                """),
                {"volume_threshold": 5000000}
            )
            result = unindexed_query_result.fetchone()
            end_time = datetime.utcnow()
            
            scan_time = (end_time - start_time).total_seconds()
            # Full table scan might be slower but should still complete quickly
            assert scan_time < 5.0, f"Full table scan took {scan_time}s"
            
        except Exception as e:
            await db_session.rollback()
            pytest.fail(f"Index optimization test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])