"""
Stock Repository
Specialized async repository for stock-related operations with advanced querying and caching.
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
import logging

from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import AsyncCRUDRepository, FilterCriteria, SortParams, PaginationParams
from backend.models.unified_models import Stock, PriceHistory, TechnicalIndicators, Recommendation, Fundamentals
from backend.config.database import get_db_session, TransactionIsolationLevel

logger = logging.getLogger(__name__)


class StockRepository(AsyncCRUDRepository[Stock]):
    """
    Specialized repository for Stock model with investment-specific operations.
    """
    
    def __init__(self):
        super().__init__(Stock)
    
    async def get_by_symbol(
        self,
        symbol: str,
        *,
        load_relationships: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[Stock]:
        """
        Get stock by symbol with optional relationship loading.
        
        Args:
            symbol: Stock symbol
            load_relationships: List of relationships to load
            session: Optional existing session
        
        Returns:
            Stock instance or None
        """
        return await self.get_by_field(
            'symbol', 
            symbol.upper(),
            load_relationships=load_relationships,
            session=session
        )
    
    async def get_by_sector(
        self,
        sector: str,
        *,
        pagination: Optional[PaginationParams] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Stock]:
        """Get all stocks in a specific sector"""
        filters = [FilterCriteria(field='sector', operator='eq', value=sector)]
        return await self.get_multi(
            filters=filters,
            pagination=pagination,
            session=session
        )
    
    async def get_by_market_cap_range(
        self,
        min_cap: Optional[float] = None,
        max_cap: Optional[float] = None,
        *,
        pagination: Optional[PaginationParams] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Stock]:
        """Get stocks within market cap range"""
        filters = []
        
        if min_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='gte', value=min_cap))
        
        if max_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='lte', value=max_cap))
        
        return await self.get_multi(
            filters=filters,
            pagination=pagination,
            session=session
        )
    
    async def search_stocks(
        self,
        query: str,
        *,
        limit: int = 50,
        session: Optional[AsyncSession] = None
    ) -> List[Stock]:
        """
        Search stocks by symbol or name using full-text search.
        
        Args:
            query: Search query
            limit: Maximum results
            session: Optional existing session
        
        Returns:
            List of matching stocks
        """
        async def _search(session: AsyncSession) -> List[Stock]:
            # Use PostgreSQL full-text search for better performance
            search_query = select(Stock).where(
                or_(
                    Stock.symbol.ilike(f'%{query.upper()}%'),
                    Stock.name.ilike(f'%{query}%')
                )
            ).order_by(
                # Prioritize exact symbol matches
                func.case(
                    (Stock.symbol == query.upper(), 1),
                    (Stock.symbol.ilike(f'{query.upper()}%'), 2),
                    else_=3
                ),
                Stock.market_cap.desc().nullslast()
            ).limit(limit)
            
            result = await session.execute(search_query)
            return result.scalars().all()
        
        if session:
            return await _search(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _search(session)
    
    async def get_top_performers(
        self,
        timeframe: str = "1d",
        limit: int = 100,
        *,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top performing stocks based on price change.
        
        Args:
            timeframe: Time period (1d, 1w, 1m, 3m, 6m, 1y)
            limit: Number of results
            session: Optional existing session
        
        Returns:
            List of stocks with performance data
        """
        async def _get_top_performers(session: AsyncSession) -> List[Dict[str, Any]]:
            # Calculate days based on timeframe
            timeframe_days = {
                '1d': 1, '1w': 7, '1m': 30, '3m': 90, '6m': 180, '1y': 365
            }
            days = timeframe_days.get(timeframe, 1)
            start_date = datetime.utcnow().date() - timedelta(days=days)
            
            # Complex query to calculate performance
            subquery = select(
                PriceHistory.stock_id,
                func.first_value(PriceHistory.close).over(
                    partition_by=PriceHistory.stock_id,
                    order_by=PriceHistory.date
                ).label('start_price'),
                func.last_value(PriceHistory.close).over(
                    partition_by=PriceHistory.stock_id,
                    order_by=PriceHistory.date,
                    range_=(None, None)
                ).label('end_price'),
                func.row_number().over(
                    partition_by=PriceHistory.stock_id,
                    order_by=PriceHistory.date.desc()
                ).label('rn')
            ).where(
                PriceHistory.date >= start_date
            ).subquery()
            
            performance_query = select(
                Stock,
                subquery.c.start_price,
                subquery.c.end_price,
                ((subquery.c.end_price - subquery.c.start_price) / 
                 subquery.c.start_price * 100).label('performance_pct')
            ).join(
                subquery, Stock.id == subquery.c.stock_id
            ).where(
                and_(
                    subquery.c.rn == 1,
                    Stock.is_active == True,
                    Stock.is_tradable == True
                )
            ).order_by(
                desc('performance_pct')
            ).limit(limit)
            
            result = await session.execute(performance_query)
            return [
                {
                    'stock': row.Stock,
                    'start_price': float(row.start_price),
                    'end_price': float(row.end_price),
                    'performance_pct': float(row.performance_pct)
                }
                for row in result
            ]
        
        if session:
            return await _get_top_performers(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_top_performers(session)
    
    async def get_stocks_with_latest_prices(
        self,
        symbols: Optional[List[str]] = None,
        *,
        limit: int = 1000,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Get stocks with their latest prices and basic metrics.
        
        Args:
            symbols: Optional list of symbols to filter by
            limit: Maximum results
            session: Optional existing session
        
        Returns:
            List of stocks with latest price data
        """
        async def _get_with_prices(session: AsyncSession) -> List[Dict[str, Any]]:
            # Subquery for latest price data
            latest_price_subq = select(
                PriceHistory.stock_id,
                PriceHistory.close.label('latest_price'),
                PriceHistory.date.label('price_date'),
                PriceHistory.volume,
                func.row_number().over(
                    partition_by=PriceHistory.stock_id,
                    order_by=PriceHistory.date.desc()
                ).label('rn')
            ).subquery()
            
            # Main query joining stock with latest prices
            query = select(
                Stock,
                latest_price_subq.c.latest_price,
                latest_price_subq.c.price_date,
                latest_price_subq.c.volume
            ).join(
                latest_price_subq,
                and_(
                    Stock.id == latest_price_subq.c.stock_id,
                    latest_price_subq.c.rn == 1
                )
            ).where(
                Stock.is_active == True
            )
            
            # Filter by symbols if provided
            if symbols:
                query = query.where(Stock.symbol.in_([s.upper() for s in symbols]))
            
            query = query.order_by(Stock.market_cap.desc().nullslast()).limit(limit)
            
            result = await session.execute(query)
            return [
                {
                    'stock': row.Stock,
                    'latest_price': float(row.latest_price) if row.latest_price else None,
                    'price_date': row.price_date,
                    'volume': row.volume
                }
                for row in result
            ]
        
        if session:
            return await _get_with_prices(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_with_prices(session)
    
    async def get_sector_summary(
        self,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """Get summary statistics by sector"""
        async def _get_sector_summary(session: AsyncSession) -> List[Dict[str, Any]]:
            query = select(
                Stock.sector,
                func.count(Stock.id).label('stock_count'),
                func.sum(Stock.market_cap).label('total_market_cap'),
                func.avg(Stock.market_cap).label('avg_market_cap')
            ).where(
                and_(
                    Stock.is_active == True,
                    Stock.sector.is_not(None)
                )
            ).group_by(
                Stock.sector
            ).order_by(
                desc('total_market_cap')
            )
            
            result = await session.execute(query)
            return [
                {
                    'sector': row.sector,
                    'stock_count': row.stock_count,
                    'total_market_cap': float(row.total_market_cap) if row.total_market_cap else 0,
                    'avg_market_cap': float(row.avg_market_cap) if row.avg_market_cap else 0
                }
                for row in result
            ]
        
        if session:
            return await _get_sector_summary(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_sector_summary(session)
    
    async def update_stock_metrics(
        self,
        symbol: str,
        metrics: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> Optional[Stock]:
        """
        Update stock metrics by symbol.
        
        Args:
            symbol: Stock symbol
            metrics: Dictionary of metrics to update
            session: Optional existing session
        
        Returns:
            Updated stock instance or None if not found
        """
        async def _update_metrics(session: AsyncSession) -> Optional[Stock]:
            stock = await self.get_by_symbol(symbol, session=session)
            if not stock:
                return None
            
            return await self.update(stock.id, metrics, session=session)
        
        if session:
            return await _update_metrics(session)
        else:
            async with get_db_session() as session:
                return await _update_metrics(session)
    
    async def bulk_update_market_caps(
        self,
        market_cap_data: List[Dict[str, Any]],
        session: Optional[AsyncSession] = None
    ) -> int:
        """
        Bulk update market cap data for multiple stocks.
        
        Args:
            market_cap_data: List of {'symbol': str, 'market_cap': float}
            session: Optional existing session
        
        Returns:
            Number of stocks updated
        """
        if not market_cap_data:
            return 0
        
        async def _bulk_update(session: AsyncSession) -> int:
            updated_count = 0
            
            for data in market_cap_data:
                symbol = data.get('symbol')
                market_cap = data.get('market_cap')
                
                if symbol and market_cap is not None:
                    stock = await self.get_by_symbol(symbol, session=session)
                    if stock:
                        await self.update(
                            stock.id,
                            {'market_cap': market_cap, 'last_price_update': datetime.utcnow()},
                            session=session
                        )
                        updated_count += 1
            
            logger.info(f"Bulk updated market caps for {updated_count} stocks")
            return updated_count
        
        if session:
            return await _bulk_update(session)
        else:
            async with get_db_session() as session:
                return await _bulk_update(session)
    
    async def get_stocks_for_analysis(
        self,
        tier: Optional[str] = None,
        exclude_inactive: bool = True,
        limit: Optional[int] = None,
        *,
        session: Optional[AsyncSession] = None
    ) -> List[Stock]:
        """
        Get stocks filtered for analysis tasks.
        
        Args:
            tier: Optional tier filter (if tier field exists)
            exclude_inactive: Whether to exclude inactive stocks
            limit: Optional limit
            session: Optional existing session
        
        Returns:
            List of stocks for analysis
        """
        filters = []
        
        if exclude_inactive:
            filters.extend([
                FilterCriteria(field='is_active', operator='eq', value=True),
                FilterCriteria(field='is_tradable', operator='eq', value=True)
            ])
        
        # Add tier filter if the field exists and tier is provided
        if tier and hasattr(Stock, 'tier'):
            filters.append(FilterCriteria(field='tier', operator='eq', value=tier))
        
        pagination = PaginationParams(limit=limit) if limit else None
        
        return await self.get_multi(
            filters=filters,
            sort_params=[SortParams(field='market_cap', direction='desc')],
            pagination=pagination,
            session=session
        )


    async def get_top_stocks(
        self,
        limit: int = 100,
        by_market_cap: bool = True,
        *,
        session: Optional[AsyncSession] = None
    ) -> List[Stock]:
        """
        Get top stocks ordered by market cap or other criteria.

        Args:
            limit: Maximum number of stocks to return
            by_market_cap: Whether to order by market cap (default True)
            session: Optional existing session

        Returns:
            List of top stocks
        """
        async def _get_top_stocks(session: AsyncSession) -> List[Stock]:
            query = select(Stock).where(
                and_(
                    Stock.is_active == True,
                    Stock.is_tradable == True,
                    Stock.market_cap.is_not(None)
                )
            )

            if by_market_cap:
                query = query.order_by(Stock.market_cap.desc())
            else:
                query = query.order_by(Stock.symbol.asc())

            query = query.limit(limit)

            result = await session.execute(query)
            return result.scalars().all()

        if session:
            return await _get_top_stocks(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_top_stocks(session)


# Create repository instance
stock_repository = StockRepository()