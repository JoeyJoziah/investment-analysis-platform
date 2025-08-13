"""
Materialized Views for Common Queries
Creates and manages materialized views to optimize frequent database queries.
"""

import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError
import os

logger = logging.getLogger(__name__)


class MaterializedViewManager:
    """Manager for creating and refreshing materialized views."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize materialized view manager.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:password@localhost:5432/investment_db'
        )
        self.engine = create_engine(self.database_url)
    
    def create_all_views(self) -> bool:
        """
        Create all materialized views for common queries.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.engine.begin() as conn:
                # Stock performance views
                self._create_stock_performance_views(conn)
                
                # Market overview views
                self._create_market_overview_views(conn)
                
                # Analysis views
                self._create_analysis_views(conn)
                
                # Portfolio views
                self._create_portfolio_views(conn)
                
                # Risk metrics views
                self._create_risk_views(conn)
                
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create materialized views: {e}")
            return False
    
    def _create_stock_performance_views(self, conn: Connection):
        """Create views for stock performance metrics."""
        
        # Top gainers view
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS top_gainers AS
            SELECT 
                s.symbol,
                s.company_name,
                ph_today.close as current_price,
                ph_yesterday.close as previous_close,
                ((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100) as daily_change_pct,
                ph_today.volume,
                s.market_cap,
                s.tier
            FROM stocks s
            INNER JOIN price_history ph_today ON s.id = ph_today.stock_id
            INNER JOIN price_history ph_yesterday ON s.id = ph_yesterday.stock_id
            WHERE ph_today.date = CURRENT_DATE
              AND ph_yesterday.date = CURRENT_DATE - INTERVAL '1 day'
              AND ((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100) > 0
            ORDER BY daily_change_pct DESC
            LIMIT 100
        """))
        
        # Top losers view
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS top_losers AS
            SELECT 
                s.symbol,
                s.company_name,
                ph_today.close as current_price,
                ph_yesterday.close as previous_close,
                ((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100) as daily_change_pct,
                ph_today.volume,
                s.market_cap,
                s.tier
            FROM stocks s
            INNER JOIN price_history ph_today ON s.id = ph_today.stock_id
            INNER JOIN price_history ph_yesterday ON s.id = ph_yesterday.stock_id
            WHERE ph_today.date = CURRENT_DATE
              AND ph_yesterday.date = CURRENT_DATE - INTERVAL '1 day'
              AND ((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100) < 0
            ORDER BY daily_change_pct ASC
            LIMIT 100
        """))
        
        # Most active stocks view
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS most_active_stocks AS
            SELECT 
                s.symbol,
                s.company_name,
                ph.close as current_price,
                ph.volume,
                ph.volume * ph.close as dollar_volume,
                s.avg_volume_30d,
                CASE 
                    WHEN s.avg_volume_30d > 0 
                    THEN ph.volume / s.avg_volume_30d 
                    ELSE 0 
                END as volume_ratio,
                s.tier
            FROM stocks s
            INNER JOIN price_history ph ON s.id = ph.stock_id
            WHERE ph.date = CURRENT_DATE
              AND ph.volume > 0
            ORDER BY ph.volume DESC
            LIMIT 100
        """))
        
        # 52-week highs and lows view
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS fifty_two_week_metrics AS
            SELECT 
                s.symbol,
                s.company_name,
                ph_current.close as current_price,
                ph_52w.max_high as fifty_two_week_high,
                ph_52w.min_low as fifty_two_week_low,
                ((ph_current.close - ph_52w.min_low) / (ph_52w.max_high - ph_52w.min_low) * 100) as range_position_pct,
                CASE 
                    WHEN ph_current.close >= ph_52w.max_high * 0.98 THEN 'Near 52W High'
                    WHEN ph_current.close <= ph_52w.min_low * 1.02 THEN 'Near 52W Low'
                    ELSE 'Mid Range'
                END as range_status
            FROM stocks s
            INNER JOIN price_history ph_current ON s.id = ph_current.stock_id
            INNER JOIN (
                SELECT 
                    stock_id,
                    MAX(high) as max_high,
                    MIN(low) as min_low
                FROM price_history 
                WHERE date >= CURRENT_DATE - INTERVAL '52 weeks'
                GROUP BY stock_id
            ) ph_52w ON s.id = ph_52w.stock_id
            WHERE ph_current.date = CURRENT_DATE
        """))
        
        logger.info("Created stock performance materialized views")
    
    def _create_market_overview_views(self, conn: Connection):
        """Create views for market overview metrics."""
        
        # Market sector summary
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS sector_performance AS
            SELECT 
                s.sector,
                COUNT(*) as stock_count,
                AVG(((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100)) as avg_daily_change,
                SUM(s.market_cap) as total_market_cap,
                SUM(ph_today.volume * ph_today.close) as total_dollar_volume,
                COUNT(CASE WHEN ((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100) > 0 THEN 1 END) as gainers,
                COUNT(CASE WHEN ((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100) < 0 THEN 1 END) as losers,
                COUNT(CASE WHEN ((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100) = 0 THEN 1 END) as unchanged
            FROM stocks s
            INNER JOIN price_history ph_today ON s.id = ph_today.stock_id
            INNER JOIN price_history ph_yesterday ON s.id = ph_yesterday.stock_id
            WHERE ph_today.date = CURRENT_DATE
              AND ph_yesterday.date = CURRENT_DATE - INTERVAL '1 day'
              AND s.sector IS NOT NULL
            GROUP BY s.sector
            ORDER BY avg_daily_change DESC
        """))
        
        # Market tier summary
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS tier_performance AS
            SELECT 
                s.tier,
                COUNT(*) as stock_count,
                AVG(((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100)) as avg_daily_change,
                SUM(s.market_cap) as total_market_cap,
                SUM(ph_today.volume * ph_today.close) as total_dollar_volume
            FROM stocks s
            INNER JOIN price_history ph_today ON s.id = ph_today.stock_id
            INNER JOIN price_history ph_yesterday ON s.id = ph_yesterday.stock_id
            WHERE ph_today.date = CURRENT_DATE
              AND ph_yesterday.date = CURRENT_DATE - INTERVAL '1 day'
              AND s.tier IS NOT NULL
            GROUP BY s.tier
            ORDER BY s.tier
        """))
        
        logger.info("Created market overview materialized views")
    
    def _create_analysis_views(self, conn: Connection):
        """Create views for analysis and recommendations."""
        
        # Latest recommendations view
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS latest_recommendations AS
            SELECT DISTINCT ON (r.stock_id)
                s.symbol,
                s.company_name,
                r.recommendation_type,
                r.confidence_score,
                r.target_price,
                r.current_price,
                r.reasoning,
                r.created_at,
                s.sector,
                s.tier
            FROM recommendations r
            INNER JOIN stocks s ON r.stock_id = s.id
            WHERE r.created_at >= CURRENT_DATE - INTERVAL '7 days'
              AND r.is_active = true
            ORDER BY r.stock_id, r.created_at DESC
        """))
        
        # High confidence recommendations
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS high_confidence_recommendations AS
            SELECT 
                s.symbol,
                s.company_name,
                r.recommendation_type,
                r.confidence_score,
                r.target_price,
                r.current_price,
                ((r.target_price - r.current_price) / r.current_price * 100) as potential_return,
                r.created_at,
                s.sector
            FROM recommendations r
            INNER JOIN stocks s ON r.stock_id = s.id
            WHERE r.confidence_score >= 0.8
              AND r.created_at >= CURRENT_DATE - INTERVAL '7 days'
              AND r.is_active = true
            ORDER BY r.confidence_score DESC, potential_return DESC
        """))
        
        # Technical analysis signals
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS technical_signals AS
            SELECT 
                s.symbol,
                s.company_name,
                ti.rsi,
                ti.macd,
                ti.bollinger_upper,
                ti.bollinger_lower,
                ph.close as current_price,
                CASE 
                    WHEN ti.rsi < 30 THEN 'Oversold'
                    WHEN ti.rsi > 70 THEN 'Overbought'
                    ELSE 'Neutral'
                END as rsi_signal,
                CASE 
                    WHEN ph.close > ti.bollinger_upper THEN 'Above Upper Band'
                    WHEN ph.close < ti.bollinger_lower THEN 'Below Lower Band'
                    ELSE 'Within Bands'
                END as bollinger_signal,
                ti.date
            FROM stocks s
            INNER JOIN technical_indicators ti ON s.id = ti.stock_id
            INNER JOIN price_history ph ON s.id = ph.stock_id AND ti.date = ph.date
            WHERE ti.date = CURRENT_DATE
              OR ti.date = (SELECT MAX(date) FROM technical_indicators WHERE stock_id = s.id)
        """))
        
        logger.info("Created analysis materialized views")
    
    def _create_portfolio_views(self, conn: Connection):
        """Create views for portfolio analysis."""
        
        # Portfolio performance summary (if portfolio tables exist)
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS portfolio_summary AS
            SELECT 
                'System Portfolio' as portfolio_name,
                COUNT(DISTINCT s.id) as total_positions,
                SUM(s.market_cap) as total_market_value,
                AVG(((ph_today.close - ph_yesterday.close) / ph_yesterday.close * 100)) as daily_return,
                COUNT(DISTINCT s.sector) as sector_diversification,
                COUNT(CASE WHEN s.tier = '1' THEN 1 END) as tier1_count,
                COUNT(CASE WHEN s.tier = '2' THEN 1 END) as tier2_count,
                COUNT(CASE WHEN s.tier = '3' THEN 1 END) as tier3_count
            FROM stocks s
            INNER JOIN price_history ph_today ON s.id = ph_today.stock_id
            INNER JOIN price_history ph_yesterday ON s.id = ph_yesterday.stock_id
            WHERE ph_today.date = CURRENT_DATE
              AND ph_yesterday.date = CURRENT_DATE - INTERVAL '1 day'
              AND s.is_active = true
        """))
        
        logger.info("Created portfolio materialized views")
    
    def _create_risk_views(self, conn: Connection):
        """Create views for risk metrics."""
        
        # High volatility stocks
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS high_volatility_stocks AS
            SELECT 
                s.symbol,
                s.company_name,
                ph_vol.volatility_30d,
                ph_current.close as current_price,
                s.market_cap,
                s.beta,
                CASE 
                    WHEN ph_vol.volatility_30d > 0.05 THEN 'Very High'
                    WHEN ph_vol.volatility_30d > 0.03 THEN 'High'
                    WHEN ph_vol.volatility_30d > 0.02 THEN 'Moderate'
                    ELSE 'Low'
                END as volatility_category
            FROM stocks s
            INNER JOIN price_history ph_current ON s.id = ph_current.stock_id
            INNER JOIN (
                SELECT 
                    stock_id,
                    STDDEV(((close - LAG(close) OVER (PARTITION BY stock_id ORDER BY date)) / LAG(close) OVER (PARTITION BY stock_id ORDER BY date))) as volatility_30d
                FROM price_history 
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY stock_id
                HAVING COUNT(*) >= 20  -- Ensure enough data points
            ) ph_vol ON s.id = ph_vol.stock_id
            WHERE ph_current.date = CURRENT_DATE
            ORDER BY ph_vol.volatility_30d DESC
        """))
        
        # Correlation analysis
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS sector_correlations AS
            WITH daily_returns AS (
                SELECT 
                    s.sector,
                    ph.date,
                    AVG(((ph.close - ph_prev.close) / ph_prev.close)) as sector_return
                FROM stocks s
                INNER JOIN price_history ph ON s.id = ph.stock_id
                INNER JOIN price_history ph_prev ON s.id = ph_prev.stock_id 
                    AND ph_prev.date = ph.date - INTERVAL '1 day'
                WHERE ph.date >= CURRENT_DATE - INTERVAL '90 days'
                  AND s.sector IS NOT NULL
                GROUP BY s.sector, ph.date
            )
            SELECT 
                sector,
                AVG(sector_return) as avg_return,
                STDDEV(sector_return) as volatility,
                COUNT(*) as data_points
            FROM daily_returns
            GROUP BY sector
            HAVING COUNT(*) >= 30  -- Minimum 30 days of data
            ORDER BY avg_return DESC
        """))
        
        logger.info("Created risk metrics materialized views")
    
    def refresh_all_views(self) -> bool:
        """
        Refresh all materialized views.
        
        Returns:
            bool: True if successful, False otherwise
        """
        view_names = [
            'top_gainers',
            'top_losers', 
            'most_active_stocks',
            'fifty_two_week_metrics',
            'sector_performance',
            'tier_performance',
            'latest_recommendations',
            'high_confidence_recommendations',
            'technical_signals',
            'portfolio_summary',
            'high_volatility_stocks',
            'sector_correlations'
        ]
        
        try:
            with self.engine.begin() as conn:
                for view_name in view_names:
                    try:
                        conn.execute(text(f"REFRESH MATERIALIZED VIEW {view_name}"))
                        logger.debug(f"Refreshed materialized view: {view_name}")
                    except Exception as e:
                        logger.warning(f"Failed to refresh view {view_name}: {e}")
                
                logger.info("Refreshed all materialized views")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to refresh materialized views: {e}")
            return False
    
    def refresh_view(self, view_name: str) -> bool:
        """
        Refresh a specific materialized view.
        
        Args:
            view_name: Name of the view to refresh
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(f"REFRESH MATERIALIZED VIEW {view_name}"))
                logger.info(f"Refreshed materialized view: {view_name}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to refresh view {view_name}: {e}")
            return False
    
    def drop_all_views(self) -> bool:
        """
        Drop all materialized views.
        
        Returns:
            bool: True if successful, False otherwise
        """
        view_names = [
            'sector_correlations',
            'high_volatility_stocks',
            'portfolio_summary',
            'technical_signals',
            'high_confidence_recommendations',
            'latest_recommendations',
            'tier_performance',
            'sector_performance',
            'fifty_two_week_metrics',
            'most_active_stocks',
            'top_losers',
            'top_gainers'
        ]
        
        try:
            with self.engine.begin() as conn:
                for view_name in view_names:
                    try:
                        conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {view_name}"))
                        logger.debug(f"Dropped materialized view: {view_name}")
                    except Exception as e:
                        logger.warning(f"Failed to drop view {view_name}: {e}")
                
                logger.info("Dropped all materialized views")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop materialized views: {e}")
            return False
    
    def get_view_stats(self) -> dict:
        """
        Get statistics about materialized views.
        
        Returns:
            dict: View statistics
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        matviewname,
                        definition,
                        ispopulated
                    FROM pg_matviews 
                    WHERE schemaname = 'public'
                    ORDER BY matviewname
                """))
                
                views = []
                for row in result:
                    views.append({
                        'schema': row[0],
                        'name': row[1],
                        'definition': row[2][:100] + '...' if len(row[2]) > 100 else row[2],
                        'is_populated': row[3]
                    })
                
                return {
                    'total_views': len(views),
                    'views': views
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get view stats: {e}")
            return {'total_views': 0, 'views': []}


# Global instance
materialized_view_manager = MaterializedViewManager()


def create_common_views():
    """Convenience function to create all common materialized views."""
    return materialized_view_manager.create_all_views()


def refresh_common_views():
    """Convenience function to refresh all materialized views."""
    return materialized_view_manager.refresh_all_views()


if __name__ == "__main__":
    # Command line interface
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create":
            success = create_common_views()
            print(f"Create views: {'Success' if success else 'Failed'}")
            
        elif command == "refresh":
            success = refresh_common_views()
            print(f"Refresh views: {'Success' if success else 'Failed'}")
            
        elif command == "stats":
            stats = materialized_view_manager.get_view_stats()
            print(f"Total views: {stats['total_views']}")
            for view in stats['views']:
                print(f"  - {view['name']} ({'populated' if view['is_populated'] else 'empty'})")
                
        else:
            print("Usage: python materialized_views.py [create|refresh|stats]")
    else:
        print("Usage: python materialized_views.py [create|refresh|stats]")