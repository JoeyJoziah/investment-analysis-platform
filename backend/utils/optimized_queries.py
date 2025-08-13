"""
Optimized database queries to replace N+1 patterns with batch operations
"""

from sqlalchemy.orm import Session, joinedload, selectinload, contains_eager
from sqlalchemy import and_, or_, func, select, text
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from backend.models.unified_models import (
    Stock, PriceHistory, TechnicalIndicators, Fundamentals,
    NewsSentiment, Recommendation, Exchange, Sector, Industry
)

logger = logging.getLogger(__name__)


class OptimizedQueryManager:
    """Manager for optimized database queries"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_stocks_with_latest_data(
        self, 
        stock_ids: List[int], 
        include_price: bool = True,
        include_technical: bool = True,
        include_fundamentals: bool = True,
        days_back: int = 30
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get stocks with their latest data in a single optimized query
        Replaces N+1 pattern of fetching each stock's data separately
        """
        
        # Base query with eager loading
        query = (
            self.db.query(Stock)
            .options(
                joinedload(Stock.exchange),
                joinedload(Stock.sector),
                joinedload(Stock.industry)
            )
            .filter(Stock.id.in_(stock_ids))
        )
        
        stocks = query.all()
        result = {}
        
        # Get latest price data in batch
        if include_price and stock_ids:
            latest_prices = self._get_latest_prices_batch(stock_ids, days_back)
        else:
            latest_prices = {}
        
        # Get latest technical indicators in batch
        if include_technical and stock_ids:
            latest_technical = self._get_latest_technical_batch(stock_ids, days_back)
        else:
            latest_technical = {}
        
        # Get latest fundamentals in batch
        if include_fundamentals and stock_ids:
            latest_fundamentals = self._get_latest_fundamentals_batch(stock_ids)
        else:
            latest_fundamentals = {}
        
        # Combine results
        for stock in stocks:
            result[stock.id] = {
                'stock': stock,
                'latest_price': latest_prices.get(stock.id),
                'latest_technical': latest_technical.get(stock.id),
                'latest_fundamentals': latest_fundamentals.get(stock.id)
            }
        
        return result
    
    def _get_latest_prices_batch(self, stock_ids: List[int], days_back: int) -> Dict[int, PriceHistory]:
        """Get latest price data for multiple stocks in a single query"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Use window function to get latest price per stock
        subquery = (
            self.db.query(
                PriceHistory,
                func.row_number().over(
                    partition_by=PriceHistory.stock_id,
                    order_by=PriceHistory.date.desc()
                ).label('rn')
            )
            .filter(
                PriceHistory.stock_id.in_(stock_ids),
                PriceHistory.date >= cutoff_date
            )
            .subquery()
        )
        
        query = (
            self.db.query(PriceHistory)
            .select_from(subquery)
            .filter(subquery.c.rn == 1)
        )
        
        prices = query.all()
        return {price.stock_id: price for price in prices}
    
    def _get_latest_technical_batch(self, stock_ids: List[int], days_back: int) -> Dict[int, TechnicalIndicators]:
        """Get latest technical indicators for multiple stocks in a single query"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        subquery = (
            self.db.query(
                TechnicalIndicators,
                func.row_number().over(
                    partition_by=TechnicalIndicators.stock_id,
                    order_by=TechnicalIndicators.date.desc()
                ).label('rn')
            )
            .filter(
                TechnicalIndicators.stock_id.in_(stock_ids),
                TechnicalIndicators.date >= cutoff_date
            )
            .subquery()
        )
        
        query = (
            self.db.query(TechnicalIndicators)
            .select_from(subquery)
            .filter(subquery.c.rn == 1)
        )
        
        technical = query.all()
        return {tech.stock_id: tech for tech in technical}
    
    def _get_latest_fundamentals_batch(self, stock_ids: List[int]) -> Dict[int, Fundamentals]:
        """Get latest fundamental data for multiple stocks in a single query"""
        
        subquery = (
            self.db.query(
                Fundamentals,
                func.row_number().over(
                    partition_by=Fundamentals.stock_id,
                    order_by=Fundamentals.period_date.desc()
                ).label('rn')
            )
            .filter(Fundamentals.stock_id.in_(stock_ids))
            .subquery()
        )
        
        query = (
            self.db.query(Fundamentals)
            .select_from(subquery)
            .filter(subquery.c.rn == 1)
        )
        
        fundamentals = query.all()
        return {fund.stock_id: fund for fund in fundamentals}
    
    def get_price_history_batch(
        self, 
        stock_ids: List[int], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[int, List[PriceHistory]]:
        """Get price history for multiple stocks efficiently"""
        
        query = (
            self.db.query(PriceHistory)
            .filter(
                PriceHistory.stock_id.in_(stock_ids),
                PriceHistory.date >= start_date,
                PriceHistory.date <= end_date
            )
            .order_by(PriceHistory.stock_id, PriceHistory.date.desc())
        )
        
        prices = query.all()
        
        # Group by stock_id
        result = defaultdict(list)
        for price in prices:
            result[price.stock_id].append(price)
        
        return dict(result)
    
    def get_recommendations_with_context(
        self, 
        limit: int = 50,
        min_confidence: float = 0.6,
        actions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations with all related context in optimized queries
        Replaces multiple separate queries with joined queries
        """
        
        query = (
            self.db.query(Recommendation)
            .options(
                joinedload(Recommendation.stock)
                .joinedload(Stock.exchange),
                joinedload(Recommendation.stock)
                .joinedload(Stock.sector),
                joinedload(Recommendation.stock)
                .joinedload(Stock.industry)
            )
            .filter(
                Recommendation.is_active == True,
                Recommendation.confidence >= min_confidence
            )
        )
        
        if actions:
            query = query.filter(Recommendation.action.in_(actions))
        
        recommendations = query.order_by(
            Recommendation.confidence.desc(),
            Recommendation.created_at.desc()
        ).limit(limit).all()
        
        # Get latest prices for all recommended stocks
        stock_ids = [rec.stock_id for rec in recommendations]
        latest_prices = self._get_latest_prices_batch(stock_ids, days_back=1)
        
        # Get recent sentiment for all stocks
        recent_sentiment = self._get_recent_sentiment_batch(stock_ids, days_back=7)
        
        result = []
        for rec in recommendations:
            result.append({
                'recommendation': rec,
                'stock': rec.stock,
                'exchange': rec.stock.exchange,
                'sector': rec.stock.sector,
                'industry': rec.stock.industry,
                'latest_price': latest_prices.get(rec.stock_id),
                'recent_sentiment': recent_sentiment.get(rec.stock_id, [])
            })
        
        return result
    
    def _get_recent_sentiment_batch(self, stock_ids: List[int], days_back: int) -> Dict[int, List[NewsSentiment]]:
        """Get recent sentiment for multiple stocks"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        sentiment = (
            self.db.query(NewsSentiment)
            .filter(
                NewsSentiment.stock_id.in_(stock_ids),
                NewsSentiment.published_at >= cutoff_date
            )
            .order_by(
                NewsSentiment.stock_id,
                NewsSentiment.published_at.desc()
            )
            .all()
        )
        
        # Group by stock_id
        result = defaultdict(list)
        for sent in sentiment:
            result[sent.stock_id].append(sent)
        
        return dict(result)
    
    def get_sector_analysis_batch(self, sector_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get comprehensive sector analysis in optimized queries"""
        
        # Get all stocks in sectors with latest data
        stocks_query = (
            self.db.query(Stock)
            .filter(
                Stock.sector_id.in_(sector_ids),
                Stock.is_active == True
            )
            .options(joinedload(Stock.sector))
        )
        
        stocks = stocks_query.all()
        stock_ids = [stock.id for stock in stocks]
        
        if not stock_ids:
            return {}
        
        # Get aggregated metrics per sector using SQL
        sector_metrics = self._get_sector_metrics_aggregated(sector_ids)
        
        # Get top performers per sector
        top_performers = self._get_sector_top_performers(sector_ids, limit=5)
        
        # Get recent recommendations per sector
        recent_recs = self._get_sector_recent_recommendations(sector_ids, days_back=30)
        
        # Combine results
        result = {}
        for sector_id in sector_ids:
            sector_stocks = [s for s in stocks if s.sector_id == sector_id]
            
            result[sector_id] = {
                'sector': sector_stocks[0].sector if sector_stocks else None,
                'stock_count': len(sector_stocks),
                'metrics': sector_metrics.get(sector_id, {}),
                'top_performers': top_performers.get(sector_id, []),
                'recent_recommendations': recent_recs.get(sector_id, [])
            }
        
        return result
    
    def _get_sector_metrics_aggregated(self, sector_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get aggregated sector metrics using SQL"""
        
        # Use raw SQL for better performance on aggregations
        query = text("""
            WITH sector_data AS (
                SELECT 
                    s.sector_id,
                    ph.close,
                    ph.volume,
                    ph.date,
                    s.market_cap,
                    ROW_NUMBER() OVER (PARTITION BY s.id ORDER BY ph.date DESC) as rn
                FROM stocks s
                JOIN price_history ph ON s.id = ph.stock_id
                WHERE s.sector_id = ANY(:sector_ids)
                  AND s.is_active = true
                  AND ph.date >= CURRENT_DATE - INTERVAL '30 days'
            ),
            latest_prices AS (
                SELECT * FROM sector_data WHERE rn = 1
            )
            SELECT 
                sector_id,
                COUNT(*) as active_stocks,
                AVG(close) as avg_price,
                SUM(volume) as total_volume,
                SUM(market_cap) as total_market_cap,
                STDDEV(close) as price_volatility
            FROM latest_prices
            GROUP BY sector_id
        """)
        
        result = self.db.execute(query, {'sector_ids': sector_ids}).fetchall()
        
        return {
            row.sector_id: {
                'active_stocks': row.active_stocks,
                'avg_price': float(row.avg_price) if row.avg_price else 0,
                'total_volume': int(row.total_volume) if row.total_volume else 0,
                'total_market_cap': float(row.total_market_cap) if row.total_market_cap else 0,
                'price_volatility': float(row.price_volatility) if row.price_volatility else 0
            }
            for row in result
        }
    
    def _get_sector_top_performers(self, sector_ids: List[int], limit: int = 5) -> Dict[int, List[Stock]]:
        """Get top performing stocks per sector"""
        
        # Get top performers based on recent price performance
        query = text("""
            WITH performance AS (
                SELECT DISTINCT ON (s.id)
                    s.id as stock_id,
                    s.sector_id,
                    s.ticker,
                    s.name,
                    ph.close as current_price,
                    LAG(ph.close, 30) OVER (PARTITION BY s.id ORDER BY ph.date) as price_30d_ago,
                    ((ph.close - LAG(ph.close, 30) OVER (PARTITION BY s.id ORDER BY ph.date)) / 
                     LAG(ph.close, 30) OVER (PARTITION BY s.id ORDER BY ph.date)) * 100 as return_30d
                FROM stocks s
                JOIN price_history ph ON s.id = ph.stock_id
                WHERE s.sector_id = ANY(:sector_ids)
                  AND s.is_active = true
                  AND ph.date >= CURRENT_DATE - INTERVAL '45 days'
                ORDER BY s.id, ph.date DESC
            ),
            ranked_performers AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY sector_id ORDER BY return_30d DESC) as rank
                FROM performance
                WHERE return_30d IS NOT NULL
            )
            SELECT stock_id, sector_id, ticker, name, current_price, return_30d
            FROM ranked_performers
            WHERE rank <= :limit
            ORDER BY sector_id, rank
        """)
        
        result = self.db.execute(query, {'sector_ids': sector_ids, 'limit': limit}).fetchall()
        
        # Group by sector
        performers = defaultdict(list)
        for row in result:
            performers[row.sector_id].append({
                'stock_id': row.stock_id,
                'ticker': row.ticker,
                'name': row.name,
                'current_price': float(row.current_price),
                'return_30d': float(row.return_30d)
            })
        
        return dict(performers)
    
    def _get_sector_recent_recommendations(self, sector_ids: List[int], days_back: int = 30) -> Dict[int, List[Recommendation]]:
        """Get recent recommendations per sector"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        recommendations = (
            self.db.query(Recommendation)
            .join(Stock, Recommendation.stock_id == Stock.id)
            .filter(
                Stock.sector_id.in_(sector_ids),
                Recommendation.created_at >= cutoff_date,
                Recommendation.is_active == True
            )
            .options(contains_eager(Recommendation.stock))
            .order_by(
                Stock.sector_id,
                Recommendation.confidence.desc()
            )
            .all()
        )
        
        # Group by sector
        sector_recs = defaultdict(list)
        for rec in recommendations:
            sector_recs[rec.stock.sector_id].append(rec)
        
        return dict(sector_recs)
    
    def bulk_update_prices(self, price_data: List[Dict[str, Any]]) -> int:
        """Bulk update prices efficiently using UPSERT"""
        
        if not price_data:
            return 0
        
        # Use PostgreSQL UPSERT for efficient bulk updates
        upsert_query = text("""
            INSERT INTO price_history (stock_id, date, open, high, low, close, volume, adjusted_close, typical_price, vwap)
            VALUES (:stock_id, :date, :open, :high, :low, :close, :volume, :adjusted_close, :typical_price, :vwap)
            ON CONFLICT (stock_id, date) 
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adjusted_close = EXCLUDED.adjusted_close,
                typical_price = EXCLUDED.typical_price,
                vwap = EXCLUDED.vwap
        """)
        
        # Batch execute
        try:
            result = self.db.execute(upsert_query, price_data)
            self.db.commit()
            return result.rowcount
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in bulk update prices: {e}")
            raise
    
    def get_query_execution_stats(self) -> Dict[str, Any]:
        """Get database query execution statistics"""
        
        # PostgreSQL specific query stats
        stats_query = text("""
            SELECT 
                schemaname,
                tablename,
                seq_scan,
                seq_tup_read,
                idx_scan,
                idx_tup_fetch,
                n_tup_ins,
                n_tup_upd,
                n_tup_del,
                n_live_tup,
                n_dead_tup
            FROM pg_stat_user_tables 
            WHERE schemaname = 'public'
            ORDER BY seq_scan DESC, idx_scan DESC
        """)
        
        try:
            result = self.db.execute(stats_query).fetchall()
            return {
                'table_stats': [
                    {
                        'table': f"{row.schemaname}.{row.tablename}",
                        'sequential_scans': row.seq_scan,
                        'seq_tuples_read': row.seq_tup_read,
                        'index_scans': row.idx_scan,
                        'index_tuples_fetched': row.idx_tup_fetch,
                        'inserts': row.n_tup_ins,
                        'updates': row.n_tup_upd,
                        'deletes': row.n_tup_del,
                        'live_tuples': row.n_live_tup,
                        'dead_tuples': row.n_dead_tup
                    }
                    for row in result
                ]
            }
        except Exception as e:
            logger.error(f"Error getting query stats: {e}")
            return {'error': str(e)}