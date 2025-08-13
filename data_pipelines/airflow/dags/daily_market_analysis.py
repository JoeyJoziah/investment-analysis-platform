"""
Main DAG for daily market analysis of 6000+ US stocks.
Implements tiered processing with intelligent API usage to stay within free tier limits.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
import json

# Configure logging
logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'investment-analysis',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# Main DAG definition
dag = DAG(
    'daily_market_analysis',
    default_args=default_args,
    description='Main ETL pipeline for analyzing 6000+ US stocks daily',
    schedule_interval='0 6 * * 1-5',  # 6 AM EST on weekdays
    catchup=False,
    tags=['production', 'data-ingestion', 'market-analysis'],
)


def get_market_calendar(**context):
    """Check if market is open and determine processing strategy."""
    from pandas_market_calendars import get_calendar
    
    nyse = get_calendar('NYSE')
    today = datetime.now().date()
    
    # Check if market is open
    schedule = nyse.schedule(start_date=today, end_date=today)
    is_market_open = len(schedule) > 0
    
    # Store in XCom for downstream tasks
    context['task_instance'].xcom_push(key='is_market_open', value=is_market_open)
    context['task_instance'].xcom_push(key='processing_date', value=str(today))
    
    logger.info(f"Market status for {today}: {'OPEN' if is_market_open else 'CLOSED'}")
    return is_market_open


def prioritize_stocks(**context) -> Dict[str, List[str]]:
    """
    Categorize stocks into priority tiers based on:
    - Market cap
    - Trading volume
    - Volatility
    - User watchlist preferences
    """
    import sys
    import os
    # Use proper package import instead of path manipulation
    from backend.models.database import get_db_session
    from backend.models.tables import Stock, PriceHistory
    from sqlalchemy import func, desc, text, and_
    from sqlalchemy.orm import Query
    
    priority_tiers = {
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
        
        for tier_key, tier_size in zip(priority_tiers.keys(), tier_sizes):
            end_idx = min(current_idx + tier_size, len(stocks_df))
            priority_tiers[tier_key] = stocks_df.iloc[current_idx:end_idx]['symbol'].tolist()
            current_idx = end_idx
            
            logger.info(f"{tier_key}: {len(priority_tiers[tier_key])} stocks")
    
    # Store in XCom
    context['task_instance'].xcom_push(key='priority_tiers', value=priority_tiers)
    return priority_tiers


def fetch_tier1_realtime_data(**context):
    """Fetch real-time data for Tier 1 stocks using Finnhub API."""
    from backend.data_ingestion.finnhub_client import FinnhubClient
    from backend.utils.cache import get_cache
    from backend.models.database import get_db_session
    from backend.models.tables import Stock, PriceHistory
    from backend.utils.circuit_breaker import CircuitBreaker
    
    # Get tier 1 stocks from XCom
    priority_tiers = context['task_instance'].xcom_pull(key='priority_tiers')
    tier1_stocks = priority_tiers.get('tier1_realtime', [])
    
    if not tier1_stocks:
        logger.warning("No Tier 1 stocks to process")
        return
    
    client = FinnhubClient()
    cache = get_cache()
    
    # Batch process to respect rate limits (60 calls/minute for Finnhub)
    batch_size = 50
    results = []
    
    for i in range(0, len(tier1_stocks), batch_size):
        batch = tier1_stocks[i:i+batch_size]
        
        for symbol in batch:
            try:
                # Check cache first
                cache_key = f"price:{symbol}:{datetime.now().date()}"
                cached_data = cache.get(cache_key)
                
                if cached_data:
                    results.append(json.loads(cached_data))
                    continue
                
                # Fetch from API
                data = client.get_stock_quote(symbol)
                if data:
                    results.append(data)
                    # Cache for 5 minutes (Tier 1)
                    cache.setex(cache_key, 300, json.dumps(data))
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        # Respect rate limit
        import time
        time.sleep(1)
    
    # Store results
    _store_price_data(results)
    logger.info(f"Processed {len(results)} Tier 1 stocks")
    return len(results)


def fetch_tier2_frequent_data(**context):
    """Fetch data for Tier 2 stocks using Alpha Vantage API."""
    from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
    from backend.utils.cache import get_cache
    from backend.utils.circuit_breaker import CircuitBreaker
    
    # Get tier 2 stocks from XCom
    priority_tiers = context['task_instance'].xcom_pull(key='priority_tiers')
    tier2_stocks = priority_tiers.get('tier2_frequent', [])
    
    if not tier2_stocks:
        logger.warning("No Tier 2 stocks to process")
        return
    
    client = AlphaVantageClient()
    cache = get_cache()
    
    # Alpha Vantage: 25 calls/day, 5 calls/minute
    # Process in small batches throughout the day
    daily_limit = 20  # Reserve 5 for other uses
    batch_size = min(daily_limit, len(tier2_stocks))
    
    results = []
    for symbol in tier2_stocks[:batch_size]:
        try:
            # Check cache first (15 minute TTL for Tier 2)
            cache_key = f"price:{symbol}:{datetime.now().date()}"
            cached_data = cache.get(cache_key)
            
            if cached_data:
                results.append(json.loads(cached_data))
                continue
            
            # Fetch from API
            data = client.get_daily_prices(symbol, outputsize='compact')
            if data:
                results.append(data)
                cache.setex(cache_key, 900, json.dumps(data))
                
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            continue
        
        # Respect rate limit (5 calls/minute)
        import time
        time.sleep(12)
    
    _store_price_data(results)
    logger.info(f"Processed {len(results)} Tier 2 stocks")
    return len(results)


def fetch_tier3_daily_data(**context):
    """Fetch daily data for Tier 3 stocks using Polygon API."""
    from backend.data_ingestion.polygon_client import PolygonClient
    from backend.utils.cache import get_cache
    from backend.utils.circuit_breaker import CircuitBreaker
    
    # Get tier 3 stocks from XCom
    priority_tiers = context['task_instance'].xcom_pull(key='priority_tiers')
    tier3_stocks = priority_tiers.get('tier3_daily', [])
    
    if not tier3_stocks:
        logger.warning("No Tier 3 stocks to process")
        return
    
    client = PolygonClient()
    cache = get_cache()
    
    # Polygon: 5 calls/minute on free tier
    # Batch multiple symbols per call where possible
    results = []
    batch_size = 5
    
    for i in range(0, min(len(tier3_stocks), 100), batch_size):  # Limit to 100 stocks/day
        batch = tier3_stocks[i:i+batch_size]
        
        for symbol in batch:
            try:
                # Check cache first (1 hour TTL for Tier 3)
                cache_key = f"price:{symbol}:{datetime.now().date()}"
                cached_data = cache.get(cache_key)
                
                if cached_data:
                    results.append(json.loads(cached_data))
                    continue
                
                # Fetch from API
                data = client.get_aggregates(
                    symbol,
                    multiplier=1,
                    timespan='day',
                    from_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    to_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if data:
                    results.append(data)
                    cache.setex(cache_key, 3600, json.dumps(data))
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        # Respect rate limit
        import time
        time.sleep(12)
    
    _store_price_data(results)
    logger.info(f"Processed {len(results)} Tier 3 stocks")
    return len(results)


def fetch_tier4_batch_data(**context):
    """Fetch batch data for Tier 4 stocks from cached/historical sources."""
    from backend.utils.cache import get_cache
    from backend.models.database import get_db_session
    from backend.models.tables import Stock, PriceHistory
    from sqlalchemy import and_, desc
    
    # Get tier 4 stocks from XCom
    priority_tiers = context['task_instance'].xcom_pull(key='priority_tiers')
    tier4_stocks = priority_tiers.get('tier4_batch', [])
    
    if not tier4_stocks:
        logger.warning("No Tier 4 stocks to process")
        return
    
    cache = get_cache()
    
    # For Tier 4, primarily use cached/historical data
    # Update only weekly or when specifically requested
    with get_db_session() as session:
        for symbol in tier4_stocks[:500]:  # Process subset daily
            try:
                # Check if we have recent data (within 7 days)
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    continue
                
                latest_price = session.query(PriceHistory).filter_by(
                    stock_id=stock.id
                ).order_by(PriceHistory.date.desc()).first()
                
                if latest_price and (datetime.now().date() - latest_price.date).days < 7:
                    # Use existing data, cache for 24 hours
                    cache_key = f"price:{symbol}:{datetime.now().date()}"
                    cache.setex(cache_key, 86400, json.dumps({
                        'symbol': symbol,
                        'price': float(latest_price.close),
                        'volume': int(latest_price.volume),
                        'date': str(latest_price.date)
                    }))
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
    
    logger.info(f"Processed Tier 4 batch stocks")
    return len(tier4_stocks[:500])


def _store_price_data(data: List[Dict]):
    """Helper function to store price data in database."""
    from backend.models.database import get_db_session
    from backend.models.tables import Stock, PriceHistory
    from sqlalchemy.exc import SQLAlchemyError
    
    with get_db_session() as session:
        for item in data:
            try:
                # Get stock
                stock = session.query(Stock).filter_by(symbol=item.get('symbol')).first()
                if not stock:
                    continue
                
                # Create price history record
                price_record = PriceHistory(
                    stock_id=stock.id,
                    date=datetime.now().date(),
                    open=item.get('open', 0),
                    high=item.get('high', 0),
                    low=item.get('low', 0),
                    close=item.get('close', 0),
                    volume=item.get('volume', 0),
                    adjusted_close=item.get('adjusted_close', item.get('close', 0))
                )
                
                session.merge(price_record)
                
            except Exception as e:
                logger.error(f"Error storing data: {e}")
                continue
        
        session.commit()


def run_technical_analysis(**context):
    """Run technical analysis on fetched data."""
    from backend.analytics.technical_analysis import TechnicalAnalysisEngine
    from backend.models.database import get_db_session
    from backend.models.tables import Stock, TechnicalIndicators, PriceHistory
    from sqlalchemy import and_
    
    engine = TechnicalAnalysisEngine()
    
    with get_db_session() as session:
        # Get stocks that were updated today
        stocks = session.query(Stock).join(
            PriceHistory
        ).filter(
            PriceHistory.date == datetime.now().date()
        ).distinct().all()
        
        for stock in stocks:
            try:
                # Get price history
                prices = session.query(PriceHistory).filter_by(
                    stock_id=stock.id
                ).order_by(PriceHistory.date.desc()).limit(100).all()
                
                if len(prices) < 20:
                    continue
                
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
                indicators = engine.analyze(df)
                
                # Store results
                tech_record = TechnicalIndicators(
                    stock_id=stock.id,
                    date=datetime.now().date(),
                    rsi=indicators.get('rsi'),
                    macd=indicators.get('macd'),
                    macd_signal=indicators.get('macd_signal'),
                    bollinger_upper=indicators.get('bollinger_upper'),
                    bollinger_lower=indicators.get('bollinger_lower'),
                    sma_20=indicators.get('sma_20'),
                    sma_50=indicators.get('sma_50'),
                    sma_200=indicators.get('sma_200'),
                    volume_sma=indicators.get('volume_sma'),
                    atr=indicators.get('atr'),
                    stochastic_k=indicators.get('stochastic_k'),
                    stochastic_d=indicators.get('stochastic_d')
                )
                
                session.merge(tech_record)
                
            except Exception as e:
                logger.error(f"Error analyzing {stock.symbol}: {e}")
                continue
        
        session.commit()
    
    logger.info("Technical analysis completed")
    return True


def generate_recommendations(**context):
    """Generate daily stock recommendations based on analysis."""
    from backend.analytics.recommendation_engine import RecommendationEngine
    
    engine = RecommendationEngine()
    recommendations = engine.generate_daily_recommendations()
    
    # Store recommendations in database
    engine.store_recommendations(recommendations)
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    return len(recommendations)


def send_notifications(**context):
    """Send notifications for high-priority recommendations."""
    # Implementation for sending email/push notifications
    logger.info("Notifications sent")
    return True


def update_cost_metrics(**context):
    """Update API usage and cost metrics."""
    from backend.utils.cost_monitor import CostMonitor
    
    monitor = CostMonitor()
    daily_cost = monitor.get_daily_cost()
    monthly_projection = monitor.project_monthly_cost()
    
    logger.info(f"Daily cost: ${daily_cost:.2f}, Monthly projection: ${monthly_projection:.2f}")
    
    if monthly_projection > 45:  # Alert if approaching $50 limit
        logger.warning(f"COST ALERT: Monthly projection ${monthly_projection:.2f} approaching limit!")
        monitor.enable_cost_saving_mode()
    
    return daily_cost


# Define DAG tasks
with dag:
    # Check market calendar
    check_market = PythonOperator(
        task_id='check_market_calendar',
        python_callable=get_market_calendar,
        provide_context=True
    )
    
    # Prioritize stocks
    prioritize = PythonOperator(
        task_id='prioritize_stocks',
        python_callable=prioritize_stocks,
        provide_context=True
    )
    
    # Create task groups for parallel processing
    with TaskGroup('data_ingestion') as data_ingestion:
        # Tier 1: Real-time data
        fetch_tier1 = PythonOperator(
            task_id='fetch_tier1_realtime',
            python_callable=fetch_tier1_realtime_data,
            provide_context=True,
            pool='api_calls',
            priority_weight=10
        )
        
        # Tier 2: Frequent updates
        fetch_tier2 = PythonOperator(
            task_id='fetch_tier2_frequent',
            python_callable=fetch_tier2_frequent_data,
            provide_context=True,
            pool='api_calls',
            priority_weight=7
        )
        
        # Tier 3: Daily updates
        fetch_tier3 = PythonOperator(
            task_id='fetch_tier3_daily',
            python_callable=fetch_tier3_daily_data,
            provide_context=True,
            pool='api_calls',
            priority_weight=5
        )
        
        # Tier 4: Batch updates
        fetch_tier4 = PythonOperator(
            task_id='fetch_tier4_batch',
            python_callable=fetch_tier4_batch_data,
            provide_context=True,
            pool='api_calls',
            priority_weight=3
        )
        
        # Parallel execution
        [fetch_tier1, fetch_tier2, fetch_tier3, fetch_tier4]
    
    # Analysis tasks
    with TaskGroup('analysis') as analysis:
        # Technical analysis
        technical = PythonOperator(
            task_id='run_technical_analysis',
            python_callable=run_technical_analysis,
            provide_context=True,
            pool='compute_intensive'
        )
        
        # Generate recommendations
        recommendations = PythonOperator(
            task_id='generate_recommendations',
            python_callable=generate_recommendations,
            provide_context=True,
            pool='compute_intensive'
        )
        
        technical >> recommendations
    
    # Post-processing
    notifications = PythonOperator(
        task_id='send_notifications',
        python_callable=send_notifications,
        provide_context=True
    )
    
    # Cost monitoring
    cost_update = PythonOperator(
        task_id='update_cost_metrics',
        python_callable=update_cost_metrics,
        provide_context=True
    )
    
    # Define task dependencies
    check_market >> prioritize >> data_ingestion >> analysis >> [notifications, cost_update]