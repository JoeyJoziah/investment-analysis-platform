"""
Trading Service
Business logic for executing trades and validating orders.
"""

import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum

from backend.repositories.portfolio_repository import portfolio_repository
from backend.repositories import stock_repository
from backend.exceptions import InvalidPositionError

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order types for trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order side (buy or sell)"""
    BUY = "buy"
    SELL = "sell"


class TradingService:
    """
    Service for trading operations.
    Handles order validation, trade execution, and portfolio impact calculations.
    """

    def __init__(self):
        self.repository = portfolio_repository

    async def validate_order(
        self,
        order_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a trading order.

        Args:
            order_data: Dictionary containing order details
                - portfolio_id: int
                - symbol: str
                - side: OrderSide
                - order_type: OrderType
                - quantity: float
                - price: Optional[float] (required for limit orders)
                - stop_price: Optional[float] (required for stop orders)

        Returns:
            Dictionary with validation result and any errors
        """
        try:
            errors = []

            # Required fields
            required_fields = ['portfolio_id', 'symbol', 'side', 'order_type', 'quantity']
            for field in required_fields:
                if field not in order_data:
                    errors.append(f"Missing required field: {field}")

            if errors:
                return {
                    'valid': False,
                    'errors': errors
                }

            # Validate quantity
            quantity = float(order_data.get('quantity', 0))
            if quantity <= 0:
                errors.append("Quantity must be greater than 0")

            # Validate order type and price
            order_type = order_data.get('order_type')
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if 'price' not in order_data or not order_data['price']:
                    errors.append(f"{order_type} orders require a price")
                elif float(order_data['price']) <= 0:
                    errors.append("Price must be greater than 0")

            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if 'stop_price' not in order_data or not order_data['stop_price']:
                    errors.append(f"{order_type} orders require a stop_price")
                elif float(order_data['stop_price']) <= 0:
                    errors.append("Stop price must be greater than 0")

            # Validate symbol format (basic check)
            symbol = order_data.get('symbol', '')
            if not symbol or not symbol.isalpha() or len(symbol) > 5:
                errors.append("Invalid stock symbol format")

            # Validate portfolio exists and has sufficient funds/shares
            portfolio_id = order_data.get('portfolio_id')
            if portfolio_id:
                portfolio = await self.repository.get_by_id(portfolio_id)
                if not portfolio:
                    errors.append(f"Portfolio {portfolio_id} not found")
                else:
                    # For sell orders, validate sufficient shares
                    if order_data.get('side') == OrderSide.SELL:
                        # This would check actual positions
                        logger.info(f"Validating sell order for {symbol} in portfolio {portfolio_id}")

                    # For buy orders, validate sufficient cash
                    elif order_data.get('side') == OrderSide.BUY:
                        estimated_cost = quantity * float(order_data.get('price', 0))
                        if portfolio.cash_balance < Decimal(str(estimated_cost)):
                            errors.append("Insufficient cash balance for order")

            if errors:
                return {
                    'valid': False,
                    'errors': errors
                }

            return {
                'valid': True,
                'message': 'Order validation successful'
            }

        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return {
                'valid': False,
                'errors': [str(e)]
            }

    async def execute_trade(
        self,
        portfolio_id: int,
        order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a trade for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            order: Order details (validated)
                - symbol: str
                - side: OrderSide
                - quantity: float
                - price: float
                - order_type: OrderType

        Returns:
            Dictionary with execution result
        """
        try:
            # Validate order first
            validation = await self.validate_order({
                **order,
                'portfolio_id': portfolio_id
            })

            if not validation.get('valid'):
                return {
                    'success': False,
                    'error': 'Order validation failed',
                    'validation_errors': validation.get('errors', [])
                }

            symbol = order['symbol']
            side = order['side']
            quantity = Decimal(str(order['quantity']))
            price = Decimal(str(order.get('price', 0)))

            # Look up the stock_id from the ticker symbol
            stock = await stock_repository.get_by_symbol(symbol)
            if not stock:
                return {
                    'success': False,
                    'error': f"Stock symbol '{symbol}' not found"
                }
            stock_id = stock.id

            # Execute the trade using repository
            if side == OrderSide.BUY:
                position = await self.repository.add_position(
                    portfolio_id=portfolio_id,
                    stock_id=stock_id,
                    quantity=quantity,
                    price=price,
                    transaction_type='buy'
                )

                if not position:
                    return {
                        'success': False,
                        'error': 'Failed to execute buy trade'
                    }
            else:  # SELL
                try:
                    position = await self.repository.add_position(
                        portfolio_id=portfolio_id,
                        stock_id=stock_id,
                        quantity=quantity,
                        price=price,
                        transaction_type='sell'
                    )
                except InvalidPositionError as exc:
                    return {
                        'success': False,
                        'error': str(exc)
                    }

            return {
                'success': True,
                'trade_id': position.id if position else None,
                'portfolio_id': portfolio_id,
                'symbol': symbol,
                'side': side,
                'quantity': float(quantity),
                'price': float(price),
                'total_cost': float(quantity * price),
                'executed_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error executing trade for portfolio {portfolio_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def calculate_portfolio_impact(
        self,
        portfolio_id: int,
        trade: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the impact of a trade on portfolio metrics.

        Args:
            portfolio_id: Portfolio ID
            trade: Trade details
                - symbol: str
                - side: OrderSide
                - quantity: float
                - price: float

        Returns:
            Dictionary containing impact analysis
        """
        try:
            # Get current portfolio state
            portfolio = await self.repository.get_by_id(portfolio_id)
            if not portfolio:
                return {
                    'success': False,
                    'error': f'Portfolio {portfolio_id} not found'
                }

            # Get current allocation
            current_allocation = await self.repository.get_portfolio_allocation(portfolio_id)
            current_value = await self.repository.calculate_portfolio_value(portfolio_id)

            if not current_value:
                current_value = {'total_value': float(portfolio.cash_balance)}

            symbol = trade['symbol']
            side = trade['side']
            quantity = float(trade['quantity'])
            price = float(trade['price'])
            trade_value = quantity * price

            # Calculate new allocation
            total_value = current_value.get('total_value', 0)

            if side == OrderSide.BUY:
                new_total_value = total_value + trade_value
                new_cash = float(portfolio.cash_balance) - trade_value
                impact_type = "increase"
            else:  # SELL
                new_total_value = total_value - trade_value
                new_cash = float(portfolio.cash_balance) + trade_value
                impact_type = "decrease"

            # Calculate allocation change
            if total_value > 0:
                allocation_change = (trade_value / total_value) * 100
            else:
                allocation_change = 100.0

            return {
                'success': True,
                'portfolio_id': portfolio_id,
                'trade_impact': {
                    'symbol': symbol,
                    'side': side,
                    'trade_value': trade_value,
                    'impact_type': impact_type
                },
                'before': {
                    'total_value': total_value,
                    'cash_balance': float(portfolio.cash_balance),
                    'allocation': current_allocation
                },
                'after': {
                    'total_value': new_total_value,
                    'cash_balance': new_cash,
                    'allocation_change': allocation_change
                },
                'metrics': {
                    'value_change': new_total_value - total_value,
                    'value_change_percent': ((new_total_value - total_value) / total_value * 100) if total_value > 0 else 0,
                    'cash_utilization': ((float(portfolio.cash_balance) - new_cash) / float(portfolio.cash_balance) * 100) if portfolio.cash_balance > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio impact for portfolio {portfolio_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Create singleton instance
trading_service = TradingService()
