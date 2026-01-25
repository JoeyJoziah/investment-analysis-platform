"""
Recommendation Repository
Specialized async repository for investment recommendation operations.
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
import logging

from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import AsyncCRUDRepository, FilterCriteria, SortParams, PaginationParams
from backend.models.unified_models import Recommendation, Stock
from backend.models.tables import RecommendationPerformance
from backend.config.database import get_db_session

logger = logging.getLogger(__name__)


class RecommendationRepository(AsyncCRUDRepository[Recommendation]):
    """
    Specialized repository for Recommendation model with investment-specific operations.
    """
    
    def __init__(self):
        super().__init__(Recommendation)
    
    async def get_active_recommendations(
        self,
        *,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Recommendation]:
        """Get all active recommendations"""
        filters = [
            FilterCriteria(field='is_active', operator='eq', value=True),
            FilterCriteria(field='valid_until', operator='gt', value=datetime.utcnow())
        ]
        
        pagination = PaginationParams(limit=limit) if limit else None
        
        return await self.get_multi(
            filters=filters,
            sort_params=[SortParams(field='created_at', direction='desc')],
            pagination=pagination,
            load_relationships=['stock'],
            session=session
        )
    
    async def get_recommendations_by_symbol(
        self,
        symbol: str,
        *,
        active_only: bool = False,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Recommendation]:
        """Get recommendations for a specific stock symbol"""
        async def _get_by_symbol(session: AsyncSession) -> List[Recommendation]:
            query = select(Recommendation).join(Stock).where(
                Stock.symbol == symbol.upper()
            )
            
            if active_only:
                query = query.where(
                    and_(
                        Recommendation.is_active == True,
                        Recommendation.valid_until > datetime.utcnow()
                    )
                )
            
            query = query.order_by(Recommendation.created_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
        
        if session:
            return await _get_by_symbol(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_by_symbol(session)
    
    async def get_recommendations_by_type(
        self,
        recommendation_type: str,
        *,
        active_only: bool = False,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Recommendation]:
        """Get recommendations by type (strong_buy, buy, hold, sell, strong_sell)"""
        filters = [
            FilterCriteria(field='recommendation_type', operator='eq', value=recommendation_type)
        ]
        
        if active_only:
            filters.extend([
                FilterCriteria(field='is_active', operator='eq', value=True),
                FilterCriteria(field='valid_until', operator='gt', value=datetime.utcnow())
            ])
        
        pagination = PaginationParams(limit=limit) if limit else None
        
        return await self.get_multi(
            filters=filters,
            sort_params=[SortParams(field='confidence_score', direction='desc')],
            pagination=pagination,
            load_relationships=['stock'],
            session=session
        )
    
    async def get_top_recommendations(
        self,
        recommendation_types: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        limit: int = 50,
        session: Optional[AsyncSession] = None
    ) -> List[Recommendation]:
        """Get top recommendations by confidence score"""
        filters = [
            FilterCriteria(field='is_active', operator='eq', value=True),
            FilterCriteria(field='valid_until', operator='gt', value=datetime.utcnow()),
            FilterCriteria(field='confidence_score', operator='gte', value=min_confidence)
        ]
        
        if recommendation_types:
            filters.append(
                FilterCriteria(field='recommendation_type', operator='in', value=recommendation_types)
            )
        
        return await self.get_multi(
            filters=filters,
            sort_params=[SortParams(field='confidence_score', direction='desc')],
            pagination=PaginationParams(limit=limit),
            load_relationships=['stock'],
            session=session
        )
    
    async def get_recommendations_summary(
        self,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for recommendations"""
        async def _get_summary(session: AsyncSession) -> Dict[str, Any]:
            # Active recommendations by type
            active_by_type_query = select(
                Recommendation.recommendation_type,
                func.count(Recommendation.id).label('count')
            ).where(
                and_(
                    Recommendation.is_active == True,
                    Recommendation.valid_until > datetime.utcnow()
                )
            ).group_by(Recommendation.recommendation_type)
            
            result = await session.execute(active_by_type_query)
            active_by_type = {row.recommendation_type.value: row.count for row in result}
            
            # Overall statistics
            stats_query = select(
                func.count(Recommendation.id).label('total_recommendations'),
                func.count(Recommendation.id).filter(
                    and_(
                        Recommendation.is_active == True,
                        Recommendation.valid_until > datetime.utcnow()
                    )
                ).label('active_recommendations'),
                func.avg(Recommendation.confidence_score).label('avg_confidence'),
                func.avg(
                    Recommendation.confidence_score
                ).filter(
                    and_(
                        Recommendation.is_active == True,
                        Recommendation.valid_until > datetime.utcnow()
                    )
                ).label('avg_active_confidence')
            )
            
            result = await session.execute(stats_query)
            stats = result.first()
            
            return {
                'total_recommendations': stats.total_recommendations or 0,
                'active_recommendations': stats.active_recommendations or 0,
                'avg_confidence': float(stats.avg_confidence) if stats.avg_confidence else 0.0,
                'avg_active_confidence': float(stats.avg_active_confidence) if stats.avg_active_confidence else 0.0,
                'active_by_type': active_by_type
            }
        
        if session:
            return await _get_summary(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_summary(session)
    
    async def expire_old_recommendations(
        self,
        cutoff_date: Optional[datetime] = None,
        session: Optional[AsyncSession] = None
    ) -> int:
        """Expire old recommendations that are past their valid_until date"""
        if not cutoff_date:
            cutoff_date = datetime.utcnow()
        
        async def _expire_old(session: AsyncSession) -> int:
            from sqlalchemy import update
            
            stmt = update(Recommendation).where(
                and_(
                    Recommendation.is_active == True,
                    Recommendation.valid_until <= cutoff_date
                )
            ).values(is_active=False)
            
            result = await session.execute(stmt)
            expired_count = result.rowcount
            
            if expired_count > 0:
                logger.info(f"Expired {expired_count} old recommendations")
            
            return expired_count
        
        if session:
            return await _expire_old(session)
        else:
            async with get_db_session() as session:
                return await _expire_old(session)
    
    async def create_recommendation_with_performance(
        self,
        recommendation_data: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> Recommendation:
        """Create recommendation and initialize performance tracking"""
        async def _create_with_performance(session: AsyncSession) -> Recommendation:
            # Create recommendation
            recommendation = await self.create(recommendation_data, session=session)
            
            # Initialize performance tracking
            performance_data = {
                'recommendation_id': recommendation.id,
                'entry_price': recommendation_data.get('current_price', 0),
                'current_price': recommendation_data.get('current_price', 0),
                'highest_price': recommendation_data.get('current_price', 0),
                'lowest_price': recommendation_data.get('current_price', 0),
                'actual_return': 0.0,
                'max_return': 0.0,
                'max_drawdown': 0.0,
                'days_active': 0
            }
            
            performance = RecommendationPerformance(**performance_data)
            session.add(performance)
            await session.flush()
            
            return recommendation
        
        if session:
            return await _create_with_performance(session)
        else:
            async with get_db_session() as session:
                return await _create_with_performance(session)
    
    async def update_recommendation_performance(
        self,
        recommendation_id: int,
        current_price: float,
        session: Optional[AsyncSession] = None
    ) -> Optional[RecommendationPerformance]:
        """Update performance tracking for a recommendation"""
        async def _update_performance(session: AsyncSession) -> Optional[RecommendationPerformance]:
            # Get existing performance record
            performance_query = select(RecommendationPerformance).where(
                RecommendationPerformance.recommendation_id == recommendation_id
            )
            
            result = await session.execute(performance_query)
            performance = result.scalar_one_or_none()
            
            if not performance:
                logger.warning(f"No performance record found for recommendation {recommendation_id}")
                return None
            
            # Update performance metrics
            performance.current_price = current_price
            
            if current_price > performance.highest_price:
                performance.highest_price = current_price
            
            if current_price < performance.lowest_price:
                performance.lowest_price = current_price
            
            # Calculate returns and drawdown
            if performance.entry_price > 0:
                performance.actual_return = ((current_price - performance.entry_price) / 
                                           performance.entry_price) * 100
                
                performance.max_return = ((performance.highest_price - performance.entry_price) / 
                                        performance.entry_price) * 100
                
                performance.max_drawdown = ((performance.lowest_price - performance.entry_price) / 
                                          performance.entry_price) * 100
            
            # Update days active (simplified - could use actual creation date)
            performance.days_active = (performance.days_active or 0) + 1
            
            # Check if targets hit (requires recommendation data)
            recommendation_query = select(Recommendation).where(Recommendation.id == recommendation_id)
            rec_result = await session.execute(recommendation_query)
            recommendation = rec_result.scalar_one_or_none()
            
            if recommendation:
                if recommendation.target_price and current_price >= recommendation.target_price:
                    performance.target_hit = True
                
                if recommendation.stop_loss and current_price <= recommendation.stop_loss:
                    performance.stop_loss_hit = True
            
            performance.last_updated = datetime.utcnow()
            
            return performance
        
        if session:
            return await _update_performance(session)
        else:
            async with get_db_session() as session:
                return await _update_performance(session)
    
    async def get_performance_analytics(
        self,
        days_back: int = 30,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Get recommendation performance analytics"""
        async def _get_analytics(session: AsyncSession) -> Dict[str, Any]:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Performance statistics
            perf_query = select(
                func.count(RecommendationPerformance.id).label('total_tracked'),
                func.avg(RecommendationPerformance.actual_return).label('avg_return'),
                func.count(RecommendationPerformance.id).filter(
                    RecommendationPerformance.target_hit == True
                ).label('targets_hit'),
                func.count(RecommendationPerformance.id).filter(
                    RecommendationPerformance.stop_loss_hit == True
                ).label('stop_losses_hit'),
                func.avg(RecommendationPerformance.max_return).label('avg_max_return'),
                func.avg(RecommendationPerformance.max_drawdown).label('avg_max_drawdown')
            ).join(Recommendation).where(
                Recommendation.created_at >= cutoff_date
            )
            
            result = await session.execute(perf_query)
            perf_stats = result.first()
            
            success_rate = 0.0
            if perf_stats.total_tracked > 0:
                success_rate = (perf_stats.targets_hit / perf_stats.total_tracked) * 100
            
            return {
                'period_days': days_back,
                'total_tracked': perf_stats.total_tracked or 0,
                'avg_return_pct': float(perf_stats.avg_return) if perf_stats.avg_return else 0.0,
                'targets_hit': perf_stats.targets_hit or 0,
                'stop_losses_hit': perf_stats.stop_losses_hit or 0,
                'success_rate_pct': success_rate,
                'avg_max_return_pct': float(perf_stats.avg_max_return) if perf_stats.avg_max_return else 0.0,
                'avg_max_drawdown_pct': float(perf_stats.avg_max_drawdown) if perf_stats.avg_max_drawdown else 0.0
            }
        
        if session:
            return await _get_analytics(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_analytics(session)


# Create repository instance
recommendation_repository = RecommendationRepository()