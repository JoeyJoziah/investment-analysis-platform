"""
Portfolio Repository
Specialized async repository for portfolio management operations.
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
import logging

from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import AsyncCRUDRepository, FilterCriteria, SortParams, PaginationParams
from backend.models.unified_models import Portfolio, Position, Transaction, User, Stock
from backend.models.tables import PortfolioPerformance
from backend.config.database import get_db_session
from backend.exceptions import StaleDataError, InsufficientBalanceError, InvalidPositionError

logger = logging.getLogger(__name__)


class PortfolioRepository(AsyncCRUDRepository[Portfolio]):
    """
    Specialized repository for Portfolio model with investment tracking operations.
    """
    
    def __init__(self):
        super().__init__(Portfolio)
    
    async def get_user_portfolios(
        self,
        user_id: int,
        *,
        include_positions: bool = False,
        session: Optional[AsyncSession] = None
    ) -> List[Portfolio]:
        """Get all portfolios for a user"""
        filters = [FilterCriteria(field='user_id', operator='eq', value=user_id)]
        
        load_relationships = []
        if include_positions:
            load_relationships.append('positions')
        
        return await self.get_multi(
            filters=filters,
            sort_params=[SortParams(field='created_at', direction='desc')],
            load_relationships=load_relationships,
            session=session
        )
    
    async def get_default_portfolio(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> Optional[Portfolio]:
        """Get user's default portfolio"""
        filters = [
            FilterCriteria(field='user_id', operator='eq', value=user_id),
            FilterCriteria(field='is_default', operator='eq', value=True)
        ]
        
        portfolios = await self.get_multi(filters=filters, session=session)
        return portfolios[0] if portfolios else None
    
    async def create_default_portfolio(
        self,
        user_id: int,
        initial_cash: Decimal = Decimal('10000'),
        session: Optional[AsyncSession] = None
    ) -> Portfolio:
        """Create default portfolio for a user"""
        portfolio_data = {
            'user_id': user_id,
            'name': 'Default Portfolio',
            'description': 'Default investment portfolio',
            'cash_balance': initial_cash,
            'is_default': True
        }
        
        return await self.create(portfolio_data, session=session)
    
    async def get_portfolio_with_positions(
        self,
        portfolio_id: int,
        session: Optional[AsyncSession] = None
    ) -> Optional[Portfolio]:
        """Get portfolio with all positions loaded"""
        return await self.get_by_id(
            portfolio_id,
            load_relationships=['positions'],
            session=session
        )
    
    async def calculate_portfolio_value(
        self,
        portfolio_id: int,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """Calculate current portfolio value and metrics"""
        async def _calculate_value(session: AsyncSession) -> Optional[Dict[str, Any]]:
            portfolio = await self.get_portfolio_with_positions(portfolio_id, session=session)
            
            if not portfolio:
                return None
            
            # Get latest prices for all positions
            positions_value = Decimal('0')
            position_details = []
            
            for position in portfolio.positions:
                # Get latest price for the stock
                latest_price_query = select(
                    func.max(PriceHistory.close)
                ).join(Stock).where(
                    Stock.id == position.stock_id
                )
                
                result = await session.execute(latest_price_query)
                latest_price = result.scalar() or Decimal('0')
                
                position_value = Decimal(str(latest_price)) * position.quantity
                positions_value += position_value
                
                # Calculate unrealized gain/loss
                cost_basis = position.average_cost * position.quantity
                unrealized_gain_loss = position_value - cost_basis
                unrealized_gain_loss_pct = (unrealized_gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                
                position_details.append({
                    'position_id': position.id,
                    'stock_id': position.stock_id,
                    'quantity': float(position.quantity),
                    'average_cost': float(position.average_cost),
                    'current_price': float(latest_price),
                    'market_value': float(position_value),
                    'cost_basis': float(cost_basis),
                    'unrealized_gain_loss': float(unrealized_gain_loss),
                    'unrealized_gain_loss_pct': float(unrealized_gain_loss_pct)
                })
            
            total_value = portfolio.cash_balance + positions_value
            
            return {
                'portfolio_id': portfolio_id,
                'cash_balance': float(portfolio.cash_balance),
                'positions_value': float(positions_value),
                'total_value': float(total_value),
                'positions': position_details,
                'calculated_at': datetime.utcnow()
            }
        
        if session:
            return await _calculate_value(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _calculate_value(session)
    
    async def add_position(
        self,
        portfolio_id: int,
        stock_id: int,
        quantity: Decimal,
        price: Decimal,
        transaction_type: str = 'buy',
        expected_portfolio_version: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[Position]:
        """
        Add or update position in portfolio with row locking for concurrent safety.

        Uses SELECT FOR UPDATE to prevent race conditions during balance updates.

        Args:
            portfolio_id: Portfolio ID
            stock_id: Stock ID
            quantity: Quantity to buy/sell
            price: Price per share
            transaction_type: 'buy' or 'sell'
            expected_portfolio_version: Expected version for optimistic locking
            session: Optional existing session

        Returns:
            Updated or created Position, or None if operation failed

        Raises:
            StaleDataError: If portfolio version mismatch
            InsufficientBalanceError: If insufficient cash for buy
            InvalidPositionError: If trying to sell more than owned
        """
        async def _add_position(session: AsyncSession) -> Optional[Position]:
            # Lock portfolio for update to prevent race conditions
            portfolio_query = select(Portfolio).where(
                Portfolio.id == portfolio_id
            ).with_for_update()

            portfolio_result = await session.execute(portfolio_query)
            portfolio = portfolio_result.scalar_one_or_none()

            if not portfolio:
                return None

            # Check optimistic lock version
            if expected_portfolio_version is not None:
                if portfolio.version != expected_portfolio_version:
                    raise StaleDataError(
                        entity_type='Portfolio',
                        entity_id=portfolio_id,
                        expected_version=expected_portfolio_version,
                        current_version=portfolio.version
                    )

            # Lock position for update if it exists
            position_query = select(Position).where(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.stock_id == stock_id
                )
            ).with_for_update()

            result = await session.execute(position_query)
            existing_position = result.scalar_one_or_none()

            # Validate transaction before executing
            trade_amount = quantity * price

            if transaction_type == 'buy':
                if portfolio.cash_balance < trade_amount:
                    raise InsufficientBalanceError(
                        f"Insufficient balance: ${portfolio.cash_balance} available, "
                        f"${trade_amount} required"
                    )
            elif transaction_type == 'sell':
                if not existing_position or existing_position.quantity < quantity:
                    owned_quantity = existing_position.quantity if existing_position else 0
                    raise InvalidPositionError(
                        f"Invalid sell: trying to sell {quantity} shares, "
                        f"but only {owned_quantity} owned"
                    )

            # Execute position update
            if existing_position:
                # Increment position version
                existing_position.version += 1

                if transaction_type == 'buy':
                    # Calculate new average cost
                    total_cost = (existing_position.quantity * existing_position.avg_cost_basis) + (quantity * price)
                    new_quantity = existing_position.quantity + quantity
                    new_average_cost = total_cost / new_quantity

                    existing_position.quantity = new_quantity
                    existing_position.avg_cost_basis = new_average_cost
                    existing_position.updated_at = datetime.utcnow()

                elif transaction_type == 'sell':
                    # Reduce position
                    existing_position.quantity -= quantity
                    existing_position.updated_at = datetime.utcnow()

                    # If quantity is zero or negative, remove position
                    if existing_position.quantity <= 0:
                        await session.delete(existing_position)
                        existing_position = None

                position = existing_position
            else:
                # Create new position (only for buy orders)
                if transaction_type == 'buy':
                    position = Position(
                        portfolio_id=portfolio_id,
                        stock_id=stock_id,
                        quantity=quantity,
                        avg_cost_basis=price,
                        version=1
                    )
                    session.add(position)
                    await session.flush()
                else:
                    # Cannot sell what we don't have
                    return None

            # Record transaction
            from backend.models.unified_models import Transaction, OrderSideEnum
            transaction = Transaction(
                portfolio_id=portfolio_id,
                stock_id=stock_id,
                transaction_type=OrderSideEnum.BUY if transaction_type == 'buy' else OrderSideEnum.SELL,
                quantity=quantity,
                price=price,
                trade_date=datetime.utcnow(),
                total_amount=trade_amount
            )
            session.add(transaction)

            # Update portfolio cash balance and version
            portfolio.version += 1
            if transaction_type == 'buy':
                portfolio.cash_balance -= trade_amount
            else:
                portfolio.cash_balance += trade_amount

            await session.flush()

            logger.info(
                f"Portfolio {portfolio_id} {transaction_type}: {quantity} shares "
                f"at ${price}, new balance: ${portfolio.cash_balance}, "
                f"version: {portfolio.version}"
            )

            return position

        if session:
            return await _add_position(session)
        else:
            async with get_db_session() as session:
                return await _add_position(session)
    
    async def get_portfolio_transactions(
        self,
        portfolio_id: int,
        *,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Transaction]:
        """Get transaction history for a portfolio"""
        async def _get_transactions(session: AsyncSession) -> List[Transaction]:
            query = select(Transaction).where(
                Transaction.portfolio_id == portfolio_id
            ).order_by(Transaction.executed_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
        
        if session:
            return await _get_transactions(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_transactions(session)
    
    async def calculate_portfolio_performance(
        self,
        portfolio_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """Calculate portfolio performance metrics"""
        async def _calculate_performance(session: AsyncSession) -> Optional[Dict[str, Any]]:
            if not start_date:
                start_date = date.today() - timedelta(days=365)
            if not end_date:
                end_date = date.today()
            
            # Get portfolio performance history
            perf_query = select(PortfolioPerformance).where(
                and_(
                    PortfolioPerformance.portfolio_id == portfolio_id,
                    PortfolioPerformance.date >= start_date,
                    PortfolioPerformance.date <= end_date
                )
            ).order_by(PortfolioPerformance.date)
            
            result = await session.execute(perf_query)
            performance_records = result.scalars().all()
            
            if not performance_records:
                return None
            
            first_record = performance_records[0]
            last_record = performance_records[-1]
            
            # Calculate metrics
            total_return = last_record.cumulative_return - first_record.cumulative_return
            days_diff = (end_date - start_date).days
            annualized_return = ((1 + total_return / 100) ** (365 / days_diff) - 1) * 100 if days_diff > 0 else 0
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = [record.daily_return for record in performance_records if record.daily_return is not None]
            
            if daily_returns:
                mean_return = sum(daily_returns) / len(daily_returns)
                variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
                volatility = (variance ** 0.5) * (252 ** 0.5)  # Annualized
            else:
                volatility = 0
            
            # Calculate Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (annualized_return / 100 - risk_free_rate) / (volatility / 100) if volatility > 0 else 0
            
            # Calculate max drawdown
            max_value = 0
            max_drawdown = 0
            
            for record in performance_records:
                if record.total_value > max_value:
                    max_value = record.total_value
                
                drawdown = (max_value - record.total_value) / max_value * 100 if max_value > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'portfolio_id': portfolio_id,
                'start_date': start_date,
                'end_date': end_date,
                'start_value': float(first_record.total_value),
                'end_value': float(last_record.total_value),
                'total_return_pct': total_return,
                'annualized_return_pct': annualized_return,
                'volatility_pct': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'trading_days': len(performance_records)
            }
        
        if session:
            return await _calculate_performance(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _calculate_performance(session)
    
    async def get_portfolio_allocation(
        self,
        portfolio_id: int,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """Get portfolio allocation by sector and stock"""
        async def _get_allocation(session: AsyncSession) -> Optional[Dict[str, Any]]:
            portfolio_value_data = await self.calculate_portfolio_value(portfolio_id, session=session)
            
            if not portfolio_value_data:
                return None
            
            total_value = portfolio_value_data['total_value']
            positions = portfolio_value_data['positions']
            
            # Get sector information for stocks
            sector_allocation = {}
            stock_allocation = []
            
            for position in positions:
                stock_query = select(Stock).where(Stock.id == position['stock_id'])
                result = await session.execute(stock_query)
                stock = result.scalar_one_or_none()
                
                if stock:
                    market_value = position['market_value']
                    allocation_pct = (market_value / total_value * 100) if total_value > 0 else 0
                    
                    # Sector allocation
                    sector = stock.sector or 'Unknown'
                    if sector in sector_allocation:
                        sector_allocation[sector] += allocation_pct
                    else:
                        sector_allocation[sector] = allocation_pct
                    
                    # Stock allocation
                    stock_allocation.append({
                        'stock_id': stock.id,
                        'symbol': stock.symbol,
                        'name': stock.name,
                        'sector': sector,
                        'market_value': market_value,
                        'allocation_pct': allocation_pct
                    })
            
            # Cash allocation
            cash_allocation_pct = (portfolio_value_data['cash_balance'] / total_value * 100) if total_value > 0 else 0
            
            return {
                'portfolio_id': portfolio_id,
                'total_value': total_value,
                'cash_allocation_pct': cash_allocation_pct,
                'sector_allocation': sector_allocation,
                'stock_allocation': sorted(stock_allocation, key=lambda x: x['allocation_pct'], reverse=True)
            }
        
        if session:
            return await _get_allocation(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_allocation(session)


# Create repository instance
portfolio_repository = PortfolioRepository()