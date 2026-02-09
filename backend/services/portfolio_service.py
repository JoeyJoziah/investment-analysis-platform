"""
Portfolio Service
Business logic for portfolio management operations.
"""

import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timezone

from backend.repositories.portfolio_repository import portfolio_repository

logger = logging.getLogger(__name__)


class PortfolioService:
    """
    Service for portfolio management operations.
    Provides business logic layer between API and repository.
    """

    def __init__(self):
        self.repository = portfolio_repository

    async def get_portfolio_summary(
        self,
        user_id: int,
        portfolio_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive portfolio summary with positions and current values.

        Args:
            user_id: User ID (for authorization)
            portfolio_id: Portfolio ID

        Returns:
            Dictionary containing portfolio summary or None if not found
        """
        try:
            # Get portfolio with positions
            portfolio = await self.repository.get_portfolio_with_positions(portfolio_id)

            if not portfolio:
                logger.warning(f"Portfolio {portfolio_id} not found")
                return None

            # Verify ownership
            if portfolio.user_id != user_id:
                logger.warning(f"User {user_id} not authorized for portfolio {portfolio_id}")
                return None

            # Calculate current portfolio value
            portfolio_value = await self.repository.calculate_portfolio_value(portfolio_id)

            if not portfolio_value:
                logger.error(f"Failed to calculate value for portfolio {portfolio_id}")
                return None

            # Get allocation breakdown
            allocation = await self.repository.get_portfolio_allocation(portfolio_id)

            return {
                'portfolio_id': portfolio.id,
                'name': portfolio.name,
                'description': portfolio.description,
                'cash_balance': float(portfolio.cash_balance),
                'is_default': portfolio.is_default,
                'created_at': portfolio.created_at.isoformat(),
                'updated_at': portfolio.updated_at.isoformat(),
                'value': portfolio_value,
                'allocation': allocation,
                'position_count': len(portfolio.positions)
            }

        except Exception as e:
            logger.error(f"Error getting portfolio summary for portfolio {portfolio_id}: {e}")
            return None

    async def add_position(
        self,
        portfolio_id: int,
        stock_symbol: str,
        quantity: float,
        cost: float,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add a stock position to portfolio.

        Args:
            portfolio_id: Portfolio ID
            stock_symbol: Stock ticker symbol
            quantity: Number of shares to buy
            cost: Cost per share
            user_id: Optional user ID for authorization

        Returns:
            Dictionary with operation result
        """
        try:
            # Verify portfolio ownership if user_id provided
            if user_id:
                portfolio = await self.repository.get_by_id(portfolio_id)
                if not portfolio or portfolio.user_id != user_id:
                    return {
                        'success': False,
                        'error': 'Portfolio not found or access denied'
                    }

            # Convert to Decimal for precision
            quantity_decimal = Decimal(str(quantity))
            cost_decimal = Decimal(str(cost))

            # Note: Would need to lookup stock_id from stock_symbol
            # For now, this is a placeholder
            logger.warning(f"Stock symbol lookup not implemented: {stock_symbol}")
            stock_id = 1  # Placeholder

            # Add position using repository
            position = await self.repository.add_position(
                portfolio_id=portfolio_id,
                stock_id=stock_id,
                quantity=quantity_decimal,
                price=cost_decimal,
                transaction_type='buy'
            )

            if not position:
                return {
                    'success': False,
                    'error': 'Failed to add position'
                }

            return {
                'success': True,
                'position_id': position.id,
                'portfolio_id': portfolio_id,
                'stock_symbol': stock_symbol,
                'quantity': float(quantity_decimal),
                'average_cost': float(position.avg_cost_basis),
                'total_value': float(quantity_decimal * position.avg_cost_basis)
            }

        except Exception as e:
            logger.error(f"Error adding position to portfolio {portfolio_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_allocation(
        self,
        portfolio_id: int
    ) -> Dict[str, Any]:
        """
        Get current portfolio allocation percentages.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Dictionary containing allocation breakdown by sector and stock
        """
        try:
            allocation = await self.repository.get_portfolio_allocation(portfolio_id)

            if not allocation:
                return {
                    'portfolio_id': portfolio_id,
                    'total_value': 0,
                    'cash_allocation_pct': 100,
                    'sector_allocation': {},
                    'stock_allocation': []
                }

            return allocation

        except Exception as e:
            logger.error(f"Error getting allocation for portfolio {portfolio_id}: {e}")
            return {
                'error': str(e)
            }

    async def get_performance(
        self,
        portfolio_id: int,
        timeframe: str = '1y'
    ) -> Optional[Dict[str, Any]]:
        """
        Get portfolio performance metrics.

        Args:
            portfolio_id: Portfolio ID
            timeframe: Time period (1w, 1m, 3m, 6m, 1y, ytd, all)

        Returns:
            Dictionary containing performance metrics
        """
        try:
            from datetime import date, timedelta

            # Map timeframe to date range
            end_date = date.today()

            if timeframe == '1w':
                start_date = end_date - timedelta(days=7)
            elif timeframe == '1m':
                start_date = end_date - timedelta(days=30)
            elif timeframe == '3m':
                start_date = end_date - timedelta(days=90)
            elif timeframe == '6m':
                start_date = end_date - timedelta(days=180)
            elif timeframe == 'ytd':
                start_date = date(end_date.year, 1, 1)
            elif timeframe == '1y':
                start_date = end_date - timedelta(days=365)
            else:  # 'all'
                start_date = None

            # Get performance from repository
            performance = await self.repository.calculate_portfolio_performance(
                portfolio_id=portfolio_id,
                start_date=start_date,
                end_date=end_date
            )

            return performance

        except Exception as e:
            logger.error(f"Error getting performance for portfolio {portfolio_id}: {e}")
            return None

    async def get_transactions(
        self,
        portfolio_id: int,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history for portfolio.

        Args:
            portfolio_id: Portfolio ID
            limit: Maximum number of transactions to return

        Returns:
            List of transaction records
        """
        try:
            transactions = await self.repository.get_portfolio_transactions(
                portfolio_id=portfolio_id,
                limit=limit
            )

            return [
                {
                    'id': txn.id,
                    'portfolio_id': txn.portfolio_id,
                    'stock_id': txn.stock_id,
                    'transaction_type': txn.transaction_type.value,
                    'quantity': float(txn.quantity),
                    'price': float(txn.price),
                    'total_amount': float(txn.total_amount),
                    'trade_date': txn.trade_date.isoformat(),
                    'executed_at': txn.executed_at.isoformat()
                }
                for txn in transactions
            ]

        except Exception as e:
            logger.error(f"Error getting transactions for portfolio {portfolio_id}: {e}")
            return []

    async def create_portfolio(
        self,
        user_id: int,
        name: str,
        description: Optional[str] = None,
        initial_cash: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Create a new portfolio for a user.

        Args:
            user_id: User ID
            name: Portfolio name
            description: Optional portfolio description
            initial_cash: Initial cash balance

        Returns:
            Dictionary with created portfolio details
        """
        try:
            portfolio_data = {
                'user_id': user_id,
                'name': name,
                'description': description or f'{name} portfolio',
                'cash_balance': Decimal(str(initial_cash)),
                'is_default': False
            }

            portfolio = await self.repository.create(portfolio_data)

            return {
                'success': True,
                'portfolio_id': portfolio.id,
                'name': portfolio.name,
                'cash_balance': float(portfolio.cash_balance),
                'created_at': portfolio.created_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating portfolio for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Create singleton instance
portfolio_service = PortfolioService()
