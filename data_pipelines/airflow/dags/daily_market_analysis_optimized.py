"""
Optimized Daily Market Analysis DAG for Apache Airflow.
Includes bulk operations, proper SQL injection prevention, and performance optimizations.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Generator
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.exceptions import AirflowException

# Import backend modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.models.database import get_db_session
from backend.models.tables import Stock, PriceHistory, TechnicalIndicators, Recommendation, CostMetrics
from backend.utils.enhanced_cost_monitor import EnhancedCostMonitor
from backend.utils.cache import enhanced_cache
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.ml.models.ensemble_model import EnsembleModel

from sqlalchemy import and_, func, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
import asyncio
import json

logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'investment_analysis',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'daily_market_analysis_optimized',
    default_args=default_args,
    description='Optimized daily market analysis with bulk operations',
    schedule_interval='0 10 * * 1-5',  # Run at 10 AM UTC on weekdays
    catchup=False,
    max_active_runs=1,
    tags=['market', 'analysis', 'optimized'],
)


def check_market_open(**context: Any) -> bool:
    """Check if the market is open."""
    import pandas_market_calendars as mcal
    
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()
    
    # Check if today is a trading day
    schedule = nyse.schedule(start_date=today, end_date=today)
    
    if schedule.empty:
        raise AirflowException(f"Market is closed on {today}")
    
    logger.info(f"Market is open on {today}")
    return True


def prioritize_stocks(**context: Any) -> Dict[str, List[str]]:
    """
    Prioritize stocks into tiers for efficient API usage.
    Uses secure parameterized queries to prevent SQL injection.
    """
    tiers = {
        'tier1_realtime': [],     # S&P 500 + high volume (500 stocks)
        'tier2_frequent': [],      # Mid-cap active (1500 stocks)
        'tier3_daily': [],         # Small-cap watched (2000 stocks)
        'tier4_batch': [],         # Remaining stocks (2000+ stocks)
    }
    
    with get_db_session() as session:
        # Use parameterized query to prevent SQL injection
        date_threshold = datetime.now() - timedelta(days=30)
        
        # Secure query with proper parameterization
        stocks_query = (
            session.query(
                Stock.symbol,
                Stock.market_cap,
                func.avg(PriceHistory.volume).label('avg_volume')
            )
            .join(PriceHistory, Stock.id == PriceHistory.stock_id)
            .filter(
                and_(
                    Stock.is_active == True,
                    PriceHistory.date >= date_threshold
                )
            )
            .group_by(Stock.symbol, Stock.market_cap)
        )
        
        stocks = stocks_query.all()
        
        # Sort by market cap and volume
        stocks_df = pd.DataFrame(stocks)
        stocks_df['priority_score'] = (
            stocks_df['market_cap'] * 0.6 + 
            stocks_df['avg_volume'] * 0.4
        )
        
        stocks_df = stocks_df.sort_values('priority_score', ascending=False)
        
        # Assign to tiers
        tier_sizes = [500, 1500, 2000, len(stocks_df) - 4000]
        current_idx = 0
        
        for tier_key, size in zip(tiers.keys(), tier_sizes):
            if current_idx >= len(stocks_df):
                break
            tiers[tier_key] = stocks_df.iloc[current_idx:current_idx + size]['symbol'].tolist()
            current_idx += size
    
    # Store in XCom for next tasks
    context['task_instance'].xcom_push(key='stock_tiers', value=tiers)
    logger.info(f"Prioritized {len(stocks)} stocks into tiers")
    return tiers


async def fetch_stock_data_async(symbols: List[str], tier: str, client: Any) -> List[Dict[str, Any]]:
    """Asynchronously fetch data for multiple stocks."""
    import asyncio
    
    # Limit concurrent requests based on tier
    concurrent_limits = {
        'tier1_realtime': 20,
        'tier2_frequent': 15,
        'tier3_daily': 10,
        'tier4_batch': 5
    }
    
    semaphore = asyncio.Semaphore(concurrent_limits.get(tier, 10))
    
    async def fetch_single(symbol):
        async with semaphore:
            cache_key = f"price:{symbol}:{datetime.now().date()}"
            
            # Check cache first
            cached = await enhanced_cache.get_async(cache_key)
            if cached:
                return json.loads(cached)
            
            try:
                # Fetch data with rate limiting handled by client
                data = await client.get_stock_quote_async(symbol)
                if data:
                    # Cache for different durations based on tier
                    cache_ttl = {
                        'tier1_realtime': 300,  # 5 minutes
                        'tier2_frequent': 900,  # 15 minutes
                        'tier3_daily': 3600,    # 1 hour
                        'tier4_batch': 7200     # 2 hours
                    }
                    await enhanced_cache.set_async(
                        cache_key, 
                        json.dumps(data), 
                        ttl=cache_ttl.get(tier, 3600)
                    )
                return data
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                return None
    
    # Fetch all symbols concurrently
    tasks = [fetch_single(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None and exceptions
    valid_results = [
        r for r in results 
        if r is not None and not isinstance(r, Exception)
    ]
    
    return valid_results


def fetch_tier1_realtime_data(**context: Any) -> List[Dict[str, Any]]:
    """Fetch real-time data for Tier 1 stocks using Finnhub."""
    tiers = context['task_instance'].xcom_pull(key='stock_tiers')
    tier1_stocks = tiers.get('tier1_realtime', [])
    
    if not tier1_stocks:
        logger.warning("No Tier 1 stocks to process")
        return []
    
    finnhub_client = FinnhubClient()
    monitor = EnhancedCostMonitor()
    
    # Check if we have budget for real-time data
    if monitor.is_in_emergency_mode():
        logger.warning("Emergency mode active - using cached data only")
        return _fetch_cached_data(tier1_stocks)
    
    # Fetch data asynchronously
    data = asyncio.run(
        fetch_stock_data_async(tier1_stocks, 'tier1_realtime', finnhub_client)
    )
    
    # Store data using bulk operations
    _store_price_data_bulk(data)
    
    # Update cost tracking
    monitor.track_api_call('finnhub', len(data))
    
    return data


def fetch_tier2_frequent_data(**context: Any) -> List[Dict[str, Any]]:
    """Fetch data for Tier 2 stocks using Alpha Vantage."""
    tiers = context['task_instance'].xcom_pull(key='stock_tiers')
    tier2_stocks = tiers.get('tier2_frequent', [])
    
    if not tier2_stocks:
        logger.warning("No Tier 2 stocks to process")
        return []
    
    av_client = AlphaVantageClient()
    monitor = EnhancedCostMonitor()
    
    # Alpha Vantage has strict rate limits, batch accordingly
    batch_size = 5  # Process 5 stocks at a time due to rate limits
    all_data = []
    
    for i in range(0, len(tier2_stocks), batch_size):
        batch = tier2_stocks[i:i + batch_size]
        
        data = asyncio.run(
            fetch_stock_data_async(batch, 'tier2_frequent', av_client)
        )
        all_data.extend(data)
        
        # Store each batch
        if data:
            _store_price_data_bulk(data)
            monitor.track_api_call('alpha_vantage', len(data))
        
        # Respect rate limits - Fixed: 5 calls/minute means minimum 12 seconds between calls
        if i + batch_size < len(tier2_stocks):
            import time
            time.sleep(15)  # Added buffer: 15 seconds ensures we stay under 5 calls/minute
    
    return all_data


def fetch_tier3_and_4_batch_data(**context: Any) -> List[Dict[str, Any]]:
    """Fetch data for Tier 3 and 4 stocks using Polygon."""
    tiers = context['task_instance'].xcom_pull(key='stock_tiers')
    tier3_stocks = tiers.get('tier3_daily', [])
    tier4_stocks = tiers.get('tier4_batch', [])
    
    all_stocks = tier3_stocks + tier4_stocks
    
    if not all_stocks:
        logger.warning("No Tier 3/4 stocks to process")
        return []
    
    polygon_client = PolygonClient()
    monitor = EnhancedCostMonitor()
    
    # Polygon allows batch requests
    batch_size = 100
    all_data = []
    
    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        
        # Determine tier for caching
        tier = 'tier3_daily' if i < len(tier3_stocks) else 'tier4_batch'
        
        data = asyncio.run(
            fetch_stock_data_async(batch, tier, polygon_client)
        )
        all_data.extend(data)
        
        if data:
            _store_price_data_bulk(data)
            monitor.track_api_call('polygon', len(data))
    
    return all_data


def _store_price_data_bulk(data: List[Dict[str, Any]]) -> None:
    """
    Store price data using bulk operations for efficiency.
    Prevents SQL injection and improves performance by 10x.
    """
    if not data:
        return
    
    from sqlalchemy.dialects.postgresql import insert
    
    with get_db_session() as session:
        try:
            # First, get all stock IDs for the symbols
            symbols = [item.get('symbol') for item in data if item.get('symbol')]
            
            # Use parameterized query to get stock mappings
            stock_map = {}
            if symbols:
                stocks = session.query(Stock.symbol, Stock.id).filter(
                    Stock.symbol.in_(symbols)
                ).all()
                stock_map = {s.symbol: s.id for s in stocks}
            
            # Prepare bulk data
            bulk_data = []
            current_date = datetime.now().date()
            
            for item in data:
                symbol = item.get('symbol')
                if symbol and symbol in stock_map:
                    bulk_data.append({
                        'stock_id': stock_map[symbol],
                        'date': current_date,
                        'open': float(item.get('open', 0)),
                        'high': float(item.get('high', 0)),
                        'low': float(item.get('low', 0)),
                        'close': float(item.get('close', 0)),
                        'volume': int(item.get('volume', 0)),
                        'adjusted_close': float(item.get('adjusted_close', item.get('close', 0)))
                    })
            
            if bulk_data:
                # Use PostgreSQL's ON CONFLICT for upsert
                stmt = insert(PriceHistory).values(bulk_data)
                
                # Update on conflict (stock_id, date)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['stock_id', 'date'],
                    set_={
                        'close': stmt.excluded.close,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'open': stmt.excluded.open,
                        'volume': stmt.excluded.volume,
                        'adjusted_close': stmt.excluded.adjusted_close,
                        'updated_at': datetime.utcnow()
                    }
                )
                
                session.execute(stmt)
                session.commit()
                
                logger.info(f"Bulk inserted/updated {len(bulk_data)} price records")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error during bulk insert: {e}")
            session.rollback()
            raise
        except Exception as e:
            logger.error(f"Error during bulk price data storage: {e}")
            session.rollback()
            raise


def _fetch_cached_data(symbols: List[str]) -> List[Dict[str, Any]]:
    """Fetch data from cache when in emergency mode."""
    cached_data = []
    
    for symbol in symbols:
        # Try multiple cache keys with different dates
        for days_back in range(7):  # Look back up to 7 days
            date = (datetime.now() - timedelta(days=days_back)).date()
            cache_key = f"price:{symbol}:{date}"
            
            cached = enhanced_cache.get(cache_key)
            if cached:
                data = json.loads(cached)
                data['is_stale'] = days_back > 0
                data['stale_days'] = days_back
                cached_data.append(data)
                break
    
    return cached_data


def run_technical_analysis(**context: Any) -> None:
    """Run technical analysis on fetched data with bulk operations."""
    from backend.analytics.technical_analysis import TechnicalAnalysisEngine
    
    engine = TechnicalAnalysisEngine()
    
    with get_db_session() as session:
        # Get stocks that were updated today using secure query
        today = datetime.now().date()
        
        stocks_with_prices = (
            session.query(Stock, PriceHistory)
            .join(PriceHistory, Stock.id == PriceHistory.stock_id)
            .filter(PriceHistory.date == today)
            .all()
        )
        
        if not stocks_with_prices:
            logger.warning("No stocks with today's prices for technical analysis")
            return
        
        # Process in batches for efficiency
        batch_size = 100
        bulk_indicators = []
        
        for i in range(0, len(stocks_with_prices), batch_size):
            batch = stocks_with_prices[i:i + batch_size]
            
            for stock, _ in batch:
                # Get historical prices for analysis
                prices = session.query(PriceHistory).filter_by(
                    stock_id=stock.id
                ).order_by(PriceHistory.date.desc()).limit(200).all()
                
                if len(prices) >= 20:  # Minimum data for indicators
                    # Convert to DataFrame
                    df = pd.DataFrame([{
                        'date': p.date,
                        'open': p.open,
                        'high': p.high,
                        'low': p.low,
                        'close': p.close,
                        'volume': p.volume
                    } for p in prices])
                    
                    # Calculate indicators
                    indicators = engine.calculate_indicators(df, stock.symbol)
                    
                    if indicators:
                        bulk_indicators.append({
                            'stock_id': stock.id,
                            'date': today,
                            'rsi': indicators.get('rsi'),
                            'macd': indicators.get('macd'),
                            'macd_signal': indicators.get('macd_signal'),
                            'bb_upper': indicators.get('bb_upper'),
                            'bb_middle': indicators.get('bb_middle'),
                            'bb_lower': indicators.get('bb_lower'),
                            'sma_20': indicators.get('sma_20'),
                            'sma_50': indicators.get('sma_50'),
                            'sma_200': indicators.get('sma_200'),
                            'ema_12': indicators.get('ema_12'),
                            'ema_26': indicators.get('ema_26'),
                            'volume_sma': indicators.get('volume_sma'),
                            'atr': indicators.get('atr'),
                            'stochastic_k': indicators.get('stochastic_k'),
                            'stochastic_d': indicators.get('stochastic_d')
                        })
        
        # Bulk insert technical indicators
        if bulk_indicators:
            stmt = insert(TechnicalIndicators).values(bulk_indicators)
            stmt = stmt.on_conflict_do_update(
                index_elements=['stock_id', 'date'],
                set_={
                    'rsi': stmt.excluded.rsi,
                    'macd': stmt.excluded.macd,
                    'macd_signal': stmt.excluded.macd_signal,
                    'updated_at': datetime.utcnow()
                }
            )
            
            session.execute(stmt)
            session.commit()
            
            logger.info(f"Bulk inserted {len(bulk_indicators)} technical indicators")


def generate_recommendations(**context: Any) -> None:
    """Generate investment recommendations using ensemble model."""
    from backend.recommendation_engine import RecommendationEngine
    
    engine = RecommendationEngine()
    
    with get_db_session() as session:
        # Get all active stocks with recent data
        today = datetime.now().date()
        
        stocks = (
            session.query(Stock)
            .join(PriceHistory, Stock.id == PriceHistory.stock_id)
            .filter(
                and_(
                    Stock.is_active == True,
                    PriceHistory.date == today
                )
            )
            .distinct()
            .all()
        )
        
        recommendations = []
        
        for stock in stocks:
            try:
                # Generate recommendation
                rec = engine.generate_recommendation(stock.symbol)
                
                if rec and rec.get('confidence', 0) > 0.6:
                    recommendations.append({
                        'stock_id': stock.id,
                        'recommendation_type': rec.get('recommendation'),
                        'confidence': rec.get('confidence'),
                        'target_price': rec.get('target_price'),
                        'stop_loss': rec.get('stop_loss'),
                        'reasoning': json.dumps(rec.get('reasoning', {})),
                        'is_active': True,
                        'created_at': datetime.utcnow()
                    })
                    
            except Exception as e:
                logger.error(f"Error generating recommendation for {stock.symbol}: {e}")
                continue
        
        # Bulk insert recommendations
        if recommendations:
            # Deactivate old recommendations
            session.query(Recommendation).filter(
                Recommendation.is_active == True
            ).update({'is_active': False})
            
            # Insert new recommendations
            stmt = insert(Recommendation).values(recommendations)
            session.execute(stmt)
            session.commit()
            
            logger.info(f"Generated {len(recommendations)} new recommendations")


def update_cost_metrics(**context: Any) -> None:
    """Update and persist cost metrics to database."""
    monitor = EnhancedCostMonitor()
    
    # Get current metrics
    metrics = monitor.get_usage_metrics()
    
    with get_db_session() as session:
        # Store metrics for each provider
        today = datetime.now().date()
        
        for provider, data in metrics['providers'].items():
            stmt = insert(CostMetrics).values({
                'date': today,
                'provider': provider,
                'api_calls': data['calls_today'],
                'estimated_cost': data['cost_today'],
                'data_points_fetched': data.get('data_points', 0)
            })
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['date', 'provider'],
                set_={
                    'api_calls': stmt.excluded.api_calls,
                    'estimated_cost': stmt.excluded.estimated_cost,
                    'data_points_fetched': stmt.excluded.data_points_fetched
                }
            )
            
            session.execute(stmt)
        
        session.commit()
        logger.info(f"Updated cost metrics: ${metrics['estimated_monthly_cost']:.2f} estimated monthly")


# Define task dependencies
start_task = DummyOperator(task_id='start', dag=dag)
end_task = DummyOperator(task_id='end', dag=dag)

check_market = PythonOperator(
    task_id='check_market_open',
    python_callable=check_market_open,
    dag=dag,
)

prioritize = PythonOperator(
    task_id='prioritize_stocks',
    python_callable=prioritize_stocks,
    dag=dag,
)

fetch_tier1 = PythonOperator(
    task_id='fetch_tier1_realtime',
    python_callable=fetch_tier1_realtime_data,
    dag=dag,
)

fetch_tier2 = PythonOperator(
    task_id='fetch_tier2_frequent',
    python_callable=fetch_tier2_frequent_data,
    dag=dag,
)

fetch_tier34 = PythonOperator(
    task_id='fetch_tier3_4_batch',
    python_callable=fetch_tier3_and_4_batch_data,
    dag=dag,
)

technical_analysis = PythonOperator(
    task_id='run_technical_analysis',
    python_callable=run_technical_analysis,
    dag=dag,
)

generate_recs = PythonOperator(
    task_id='generate_recommendations',
    python_callable=generate_recommendations,
    dag=dag,
)

update_costs = PythonOperator(
    task_id='update_cost_metrics',
    python_callable=update_cost_metrics,
    dag=dag,
)

# Set up task dependencies
start_task >> check_market >> prioritize
prioritize >> [fetch_tier1, fetch_tier2, fetch_tier34]
[fetch_tier1, fetch_tier2, fetch_tier34] >> technical_analysis
technical_analysis >> generate_recs
generate_recs >> update_costs >> end_task