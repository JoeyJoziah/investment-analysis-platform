"""
Price History Repository
Specialized async repository for price history operations with time-series optimizations.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import date, datetime, timedelta
import logging

from sqlalchemy import select, func, and_, desc, asc, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import AsyncCRUDRepository, FilterCriteria, SortParams, PaginationParams
from backend.models.unified_models import PriceHistory, Stock
from backend.config.database import get_db_session

logger = logging.getLogger(__name__)


class PriceHistoryRepository(AsyncCRUDRepository[PriceHistory]):
    """
    Specialized repository for PriceHistory model with time-series operations.
    """
    
    def __init__(self):
        super().__init__(PriceHistory)
    
    async def get_price_history(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        *,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> List[PriceHistory]:
        """
        Get price history for a stock by symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Optional limit on results
            session: Optional existing session
        
        Returns:
            List of price history records
        """
        async def _get_history(session: AsyncSession) -> List[PriceHistory]:
            query = select(PriceHistory).join(Stock).where(
                Stock.symbol == symbol.upper()
            )
            
            if start_date:
                query = query.where(PriceHistory.date >= start_date)
            
            if end_date:
                query = query.where(PriceHistory.date <= end_date)
            
            query = query.order_by(PriceHistory.date.desc())
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
        
        if session:
            return await _get_history(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_history(session)
    
    async def get_latest_price(
        self,
        symbol: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[PriceHistory]:
        """Get the latest price record for a stock"""
        price_history = await self.get_price_history(
            symbol, limit=1, session=session
        )
        return price_history[0] if price_history else None
    
    async def get_price_on_date(
        self,
        symbol: str,
        target_date: date,
        session: Optional[AsyncSession] = None
    ) -> Optional[PriceHistory]:
        """
        Get price on a specific date, falling back to nearest available date.
        
        Args:
            symbol: Stock symbol
            target_date: Target date
            session: Optional existing session
        
        Returns:
            Price history record or None
        """
        async def _get_price_on_date(session: AsyncSession) -> Optional[PriceHistory]:
            # First try exact date
            exact_query = select(PriceHistory).join(Stock).where(
                and_(
                    Stock.symbol == symbol.upper(),
                    PriceHistory.date == target_date
                )
            )
            
            result = await session.execute(exact_query)
            exact_match = result.scalar_one_or_none()
            
            if exact_match:
                return exact_match
            
            # If no exact match, find nearest previous date
            nearest_query = select(PriceHistory).join(Stock).where(
                and_(
                    Stock.symbol == symbol.upper(),
                    PriceHistory.date <= target_date
                )
            ).order_by(PriceHistory.date.desc()).limit(1)
            
            result = await session.execute(nearest_query)
            return result.scalar_one_or_none()
        
        if session:
            return await _get_price_on_date(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_price_on_date(session)
    
    async def calculate_returns(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, float]]:
        """
        Calculate returns for a stock over a period.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            session: Optional existing session
        
        Returns:
            Dictionary with return calculations or None
        """
        async def _calculate_returns(session: AsyncSession) -> Optional[Dict[str, float]]:
            start_price_record = await self.get_price_on_date(symbol, start_date, session)
            end_price_record = await self.get_price_on_date(symbol, end_date, session)
            
            if not start_price_record or not end_price_record:
                return None
            
            start_price = float(start_price_record.close)
            end_price = float(end_price_record.close)
            
            absolute_return = end_price - start_price
            percentage_return = (absolute_return / start_price) * 100
            
            return {
                'start_price': start_price,
                'end_price': end_price,
                'absolute_return': absolute_return,
                'percentage_return': percentage_return,
                'start_date': start_price_record.date,
                'end_date': end_price_record.date
            }
        
        if session:
            return await _calculate_returns(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _calculate_returns(session)
    
    async def get_volatility(
        self,
        symbol: str,
        days: int = 30,
        session: Optional[AsyncSession] = None
    ) -> Optional[float]:
        """
        Calculate historical volatility for a stock.
        
        Args:
            symbol: Stock symbol
            days: Number of days to calculate over
            session: Optional existing session
        
        Returns:
            Annualized volatility or None
        """
        async def _get_volatility(session: AsyncSession) -> Optional[float]:
            end_date = date.today()
            start_date = end_date - timedelta(days=days + 1)  # Extra day for returns calculation
            
            price_history = await self.get_price_history(
                symbol, start_date, end_date, session=session
            )
            
            if len(price_history) < 2:
                return None
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(price_history)):
                prev_price = float(price_history[i].close)
                curr_price = float(price_history[i-1].close)
                daily_return = (curr_price - prev_price) / prev_price
                daily_returns.append(daily_return)
            
            if not daily_returns:
                return None
            
            # Calculate standard deviation of returns
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
            daily_volatility = variance ** 0.5
            
            # Annualize volatility (assuming 252 trading days per year)
            annualized_volatility = daily_volatility * (252 ** 0.5)
            
            return annualized_volatility
        
        if session:
            return await _get_volatility(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_volatility(session)
    
    async def get_price_statistics(
        self,
        symbol: str,
        days: int = 252,  # 1 year
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive price statistics for a stock.
        
        Args:
            symbol: Stock symbol
            days: Number of days to analyze
            session: Optional existing session
        
        Returns:
            Dictionary with price statistics
        """
        async def _get_statistics(session: AsyncSession) -> Optional[Dict[str, Any]]:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            query = select(
                func.count(PriceHistory.id).label('trading_days'),
                func.min(PriceHistory.low).label('min_price'),
                func.max(PriceHistory.high).label('max_price'),
                func.avg(PriceHistory.close).label('avg_price'),
                func.sum(PriceHistory.volume).label('total_volume'),
                func.avg(PriceHistory.volume).label('avg_volume')
            ).join(Stock).where(
                and_(
                    Stock.symbol == symbol.upper(),
                    PriceHistory.date >= start_date,
                    PriceHistory.date <= end_date
                )
            )
            
            result = await session.execute(query)
            row = result.first()
            
            if not row or row.trading_days == 0:
                return None
            
            # Get first and last prices for period return
            first_price_query = select(PriceHistory.close).join(Stock).where(
                and_(
                    Stock.symbol == symbol.upper(),
                    PriceHistory.date >= start_date
                )
            ).order_by(PriceHistory.date.asc()).limit(1)
            
            last_price_query = select(PriceHistory.close).join(Stock).where(
                and_(
                    Stock.symbol == symbol.upper(),
                    PriceHistory.date <= end_date
                )
            ).order_by(PriceHistory.date.desc()).limit(1)
            
            first_result = await session.execute(first_price_query)
            last_result = await session.execute(last_price_query)
            
            first_price = first_result.scalar()
            last_price = last_result.scalar()
            
            period_return = 0.0
            if first_price and last_price and first_price > 0:
                period_return = ((float(last_price) - float(first_price)) / float(first_price)) * 100
            
            return {
                'trading_days': row.trading_days,
                'period_start': start_date,
                'period_end': end_date,
                'min_price': float(row.min_price) if row.min_price else None,
                'max_price': float(row.max_price) if row.max_price else None,
                'avg_price': float(row.avg_price) if row.avg_price else None,
                'first_price': float(first_price) if first_price else None,
                'last_price': float(last_price) if last_price else None,
                'period_return_pct': period_return,
                'total_volume': int(row.total_volume) if row.total_volume else 0,
                'avg_volume': int(row.avg_volume) if row.avg_volume else 0
            }
        
        if session:
            return await _get_statistics(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_statistics(session)
    
    async def bulk_upsert_prices(
        self,
        price_data: List[Dict[str, Any]],
        session: Optional[AsyncSession] = None
    ) -> int:
        """
        Bulk upsert price data for multiple stocks and dates.
        
        Args:
            price_data: List of price data dictionaries
            session: Optional existing session
        
        Returns:
            Number of records affected
        """
        if not price_data:
            return 0
        
        async def _bulk_upsert(session: AsyncSession) -> int:
            # Use database manager's bulk insert with conflict handling
            from backend.config.database import db_manager
            
            return await db_manager.bulk_insert_with_conflict_handling(
                PriceHistory,
                price_data,
                conflict_strategy="update"
            )
        
        if session:
            return await _bulk_upsert(session)
        else:
            async with get_db_session() as session:
                return await _bulk_upsert(session)
    
    async def get_missing_price_dates(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        session: Optional[AsyncSession] = None
    ) -> List[date]:
        """
        Find missing trading dates for a stock in a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            session: Optional existing session
        
        Returns:
            List of missing dates
        """
        async def _get_missing_dates(session: AsyncSession) -> List[date]:
            # Get all dates that have price data for this stock
            existing_dates_query = select(PriceHistory.date).join(Stock).where(
                and_(
                    Stock.symbol == symbol.upper(),
                    PriceHistory.date >= start_date,
                    PriceHistory.date <= end_date
                )
            ).order_by(PriceHistory.date)
            
            result = await session.execute(existing_dates_query)
            existing_dates = set(row.date for row in result)
            
            # Generate all business days in the range
            current_date = start_date
            all_business_days = []
            
            while current_date <= end_date:
                # Monday = 0, Sunday = 6
                if current_date.weekday() < 5:  # Monday to Friday
                    all_business_days.append(current_date)
                current_date += timedelta(days=1)
            
            # Find missing dates
            missing_dates = [d for d in all_business_days if d not in existing_dates]
            return missing_dates
        
        if session:
            return await _get_missing_dates(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_missing_dates(session)


# Create repository instance
price_repository = PriceHistoryRepository()