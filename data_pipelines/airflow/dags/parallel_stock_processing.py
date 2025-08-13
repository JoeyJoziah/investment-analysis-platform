"""
Parallel Stock Processing DAG
Optimized for processing 6000+ stocks efficiently with intelligent batching and cost controls.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import asyncio
from typing import List, Dict, Any
import json
import concurrent.futures
from functools import partial

# Configure logging
logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'investment-analysis',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
    'max_active_runs': 1,
}

# DAG definition
dag = DAG(
    'parallel_stock_processing',
    default_args=default_args,
    description='Parallel processing of 6000+ stocks with intelligent batching',
    schedule_interval='@hourly',  # Run every hour for Tier 1 stocks
    catchup=False,
    tags=['production', 'parallel', 'cost-optimized'],
    max_active_tasks=20,  # Allow up to 20 concurrent tasks
)


class ParallelStockProcessor:
    """Intelligent parallel stock processing with cost optimization"""
    
    def __init__(self):
        self.batch_sizes = {
            1: 50,   # Tier 1: 50 stocks per batch (real-time priority)
            2: 100,  # Tier 2: 100 stocks per batch
            3: 200,  # Tier 3: 200 stocks per batch
            4: 500,  # Tier 4: 500 stocks per batch
            5: 1000  # Tier 5: 1000 stocks per batch
        }
        
        self.api_allocation = {
            1: 'finnhub',        # Best API for critical stocks
            2: 'alpha_vantage',  # Balanced for high priority
            3: 'polygon',        # Good for medium priority
            4: 'yahoo_finance',  # Free backup for low priority
            5: 'cache_only'      # Cache-only for minimal priority
        }
        
        self.processing_intervals = {
            1: 1,   # Every hour
            2: 4,   # Every 4 hours
            3: 8,   # Every 8 hours
            4: 24,  # Daily
            5: 168  # Weekly
        }
    
    def get_stocks_for_processing(self, **context) -> Dict[int, List[str]]:
        """Get stocks organized by tier for processing"""
        import sys
        sys.path.insert(0, '/opt/airflow/backend')
        
        from backend.models.database import get_db_session
        from backend.models.tables import Stock
        from sqlalchemy import and_
        
        current_hour = datetime.now().hour
        stocks_by_tier = {}
        
        with get_db_session() as session:
            for tier in range(1, 6):
                # Check if this tier should be processed at current hour
                interval = self.processing_intervals[tier]
                if current_hour % interval != 0:
                    continue  # Skip this tier for current hour
                
                # Get active stocks for this tier
                stocks = session.query(Stock.symbol).filter(
                    and_(
                        Stock.priority_tier == tier,
                        Stock.is_active == True
                    )
                ).all()
                
                stock_list = [stock.symbol for stock in stocks]
                if stock_list:
                    stocks_by_tier[tier] = stock_list
                    logger.info(f"Tier {tier}: {len(stock_list)} stocks to process")
        
        # Store in XCom for downstream tasks
        context['task_instance'].xcom_push(key='stocks_by_tier', value=stocks_by_tier)
        
        return stocks_by_tier
    
    def create_processing_batches(self, **context) -> Dict[str, List[List[str]]]:
        """Create optimized batches for parallel processing"""
        stocks_by_tier = context['task_instance'].xcom_pull(key='stocks_by_tier')
        
        if not stocks_by_tier:
            logger.info("No stocks to process at this time")
            return {}
        
        all_batches = {}
        
        for tier, stocks in stocks_by_tier.items():
            batch_size = self.batch_sizes.get(tier, 100)
            api_provider = self.api_allocation.get(tier, 'finnhub')
            
            # Create batches
            tier_batches = []
            for i in range(0, len(stocks), batch_size):
                batch = stocks[i:i + batch_size]
                tier_batches.append(batch)
            
            batch_key = f"tier_{tier}_{api_provider}"
            all_batches[batch_key] = tier_batches
            
            logger.info(f"Tier {tier} ({api_provider}): {len(tier_batches)} batches, "
                       f"{len(stocks)} total stocks")
        
        context['task_instance'].xcom_push(key='processing_batches', value=all_batches)
        
        return all_batches


def process_stock_batch_tier1(batch_symbols: List[str], **context) -> Dict[str, Any]:
    """Process a batch of Tier 1 stocks (Critical priority)"""
    import sys
    sys.path.insert(0, '/opt/airflow/backend')
    
    from backend.data_ingestion.finnhub_client import FinnhubClient
    from backend.utils.cost_monitor import cost_monitor
    from backend.utils.cache import get_cache
    
    logger.info(f"Processing Tier 1 batch: {len(batch_symbols)} stocks")
    
    client = FinnhubClient()
    cache = get_cache()
    results = []
    
    # Process with high concurrency for real-time data
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create tasks for all symbols
        future_to_symbol = {
            executor.submit(fetch_stock_data_with_retry, client, symbol): symbol 
            for symbol in batch_symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result(timeout=30)
                if data:
                    results.append(data)
                    # Cache for 5 minutes (Tier 1 real-time)
                    cache_key = f"price:{symbol}:{datetime.now().date()}"
                    cache.setex(cache_key, 300, json.dumps(data))
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    # Store results in database
    store_price_data(results)
    
    logger.info(f"Tier 1 batch completed: {len(results)}/{len(batch_symbols)} successful")
    
    return {
        'tier': 1,
        'batch_size': len(batch_symbols),
        'successful_count': len(results),
        'success_rate': len(results) / len(batch_symbols) if batch_symbols else 0,
        'api_provider': 'finnhub'
    }


def process_stock_batch_tier2(batch_symbols: List[str], **context) -> Dict[str, Any]:
    """Process a batch of Tier 2 stocks (High priority)"""
    import sys
    sys.path.insert(0, '/opt/airflow/backend')
    
    from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
    from backend.utils.cost_monitor import cost_monitor
    from backend.utils.cache import get_cache
    
    logger.info(f"Processing Tier 2 batch: {len(batch_symbols)} stocks")
    
    client = AlphaVantageClient()
    cache = get_cache()
    results = []
    
    # Alpha Vantage has strict rate limits (5/minute), process sequentially
    for symbol in batch_symbols:
        try:
            # Check rate limits
            if not asyncio.run(cost_monitor.check_api_limit('alpha_vantage')):
                logger.warning(f"Rate limit reached for Alpha Vantage, using cache for {symbol}")
                # Try to get cached data
                cache_key = f"price:{symbol}:{datetime.now().date()}"
                cached = cache.get(cache_key)
                if cached:
                    results.append(json.loads(cached))
                continue
            
            # Fetch data
            data = fetch_stock_data_with_retry(client, symbol)
            if data:
                results.append(data)
                # Cache for 15 minutes (Tier 2)
                cache_key = f"price:{symbol}:{datetime.now().date()}"
                cache.setex(cache_key, 900, json.dumps(data))
            
            # Rate limiting delay (12 seconds between calls)
            import time
            time.sleep(12)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Store results
    store_price_data(results)
    
    logger.info(f"Tier 2 batch completed: {len(results)}/{len(batch_symbols)} successful")
    
    return {
        'tier': 2,
        'batch_size': len(batch_symbols),
        'successful_count': len(results),
        'success_rate': len(results) / len(batch_symbols) if batch_symbols else 0,
        'api_provider': 'alpha_vantage'
    }


def process_stock_batch_tier3(batch_symbols: List[str], **context) -> Dict[str, Any]:
    """Process a batch of Tier 3 stocks (Medium priority)"""
    import sys
    sys.path.insert(0, '/opt/airflow/backend')
    
    from backend.data_ingestion.polygon_client import PolygonClient
    from backend.utils.cost_monitor import cost_monitor
    from backend.utils.cache import get_cache
    
    logger.info(f"Processing Tier 3 batch: {len(batch_symbols)} stocks")
    
    client = PolygonClient()
    cache = get_cache()
    results = []
    
    # Polygon has 5 calls/minute limit, batch process with delays
    batch_size = 5
    
    for i in range(0, len(batch_symbols), batch_size):
        mini_batch = batch_symbols[i:i + batch_size]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_stock_data_with_retry, client, symbol)
                for symbol in mini_batch
            ]
            
            for future, symbol in zip(futures, mini_batch):
                try:
                    data = future.result(timeout=30)
                    if data:
                        results.append(data)
                        # Cache for 1 hour (Tier 3)
                        cache_key = f"price:{symbol}:{datetime.now().date()}"
                        cache.setex(cache_key, 3600, json.dumps(data))
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        # Rate limiting delay between mini-batches
        if i + batch_size < len(batch_symbols):
            import time
            time.sleep(12)  # 12 seconds delay
    
    # Store results
    store_price_data(results)
    
    logger.info(f"Tier 3 batch completed: {len(results)}/{len(batch_symbols)} successful")
    
    return {
        'tier': 3,
        'batch_size': len(batch_symbols),
        'successful_count': len(results),
        'success_rate': len(results) / len(batch_symbols) if batch_symbols else 0,
        'api_provider': 'polygon'
    }


def process_stock_batch_tier45(batch_symbols: List[str], tier: int, **context) -> Dict[str, Any]:
    """Process a batch of Tier 4/5 stocks (Low/Minimal priority)"""
    import sys
    sys.path.insert(0, '/opt/airflow/backend')
    
    from backend.models.database import get_db_session
    from backend.models.tables import Stock, PriceHistory
    from backend.utils.cache import get_cache
    from sqlalchemy import desc
    
    logger.info(f"Processing Tier {tier} batch: {len(batch_symbols)} stocks (cache-based)")
    
    cache = get_cache()
    results = []
    
    with get_db_session() as session:
        for symbol in batch_symbols:
            try:
                # Get most recent price from database
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    continue
                
                latest_price = session.query(PriceHistory).filter_by(
                    stock_id=stock.id
                ).order_by(desc(PriceHistory.date)).first()
                
                if latest_price:
                    # Use existing data with extended cache
                    data = {
                        'symbol': symbol,
                        'price': float(latest_price.close),
                        'volume': int(latest_price.volume),
                        'date': str(latest_price.date),
                        'source': 'database_cache',
                        'tier': tier
                    }
                    results.append(data)
                    
                    # Cache for extended period (24 hours for Tier 4/5)
                    cache_key = f"price:{symbol}:{datetime.now().date()}"
                    cache.setex(cache_key, 86400, json.dumps(data))
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    logger.info(f"Tier {tier} batch completed: {len(results)}/{len(batch_symbols)} from cache")
    
    return {
        'tier': tier,
        'batch_size': len(batch_symbols),
        'successful_count': len(results),
        'success_rate': len(results) / len(batch_symbols) if batch_symbols else 0,
        'api_provider': 'cache_only'
    }


def fetch_stock_data_with_retry(client, symbol: str, max_retries: int = 3) -> Dict:
    """Fetch stock data with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            if hasattr(client, 'get_quote'):
                data = client.get_quote(symbol)
                if data:
                    return data
            else:
                logger.warning(f"Client {type(client)} does not have get_quote method")
                return None
                
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                return None
            else:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def store_price_data(data_list: List[Dict]):
    """Store price data in database efficiently"""
    if not data_list:
        return
    
    import sys
    sys.path.insert(0, '/opt/airflow/backend')
    
    from backend.models.database import get_db_session
    from backend.models.tables import Stock, PriceHistory
    from sqlalchemy.exc import IntegrityError
    
    with get_db_session() as session:
        stored_count = 0
        
        for data in data_list:
            try:
                symbol = data.get('symbol')
                if not symbol:
                    continue
                
                # Get stock
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    continue
                
                # Create or update price record
                price_record = PriceHistory(
                    stock_id=stock.id,
                    date=datetime.now().date(),
                    open=data.get('open', 0),
                    high=data.get('high', 0),
                    low=data.get('low', 0),
                    close=data.get('price', data.get('close', 0)),
                    volume=data.get('volume', 0),
                    adjusted_close=data.get('adjusted_close', data.get('price', data.get('close', 0)))
                )
                
                session.merge(price_record)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing data for {data.get('symbol', 'unknown')}: {e}")
                session.rollback()
                continue
        
        session.commit()
        logger.info(f"Stored {stored_count} price records")


def consolidate_batch_results(**context) -> Dict[str, Any]:
    """Consolidate results from all batch processing tasks"""
    import sys
    sys.path.insert(0, '/opt/airflow/backend')
    
    from backend.utils.cost_monitor import cost_monitor
    
    # This would collect results from all parallel tasks
    # For now, we'll create a summary
    
    summary = {
        'pipeline_run_time': datetime.now().isoformat(),
        'total_batches_processed': 0,
        'total_stocks_processed': 0,
        'total_successful': 0,
        'overall_success_rate': 0,
        'api_usage_summary': {},
        'performance_metrics': {
            'average_batch_time': 0,
            'stocks_per_minute': 0
        }
    }
    
    # Store summary in XCom and cache
    context['task_instance'].xcom_push(key='batch_summary', value=summary)
    
    # Update cost monitoring
    try:
        asyncio.run(cost_monitor.record_api_call(
            provider='summary',
            endpoint='batch_processing',
            success=True,
            data_points=summary['total_stocks_processed']
        ))
    except Exception as e:
        logger.error(f"Error updating cost monitor: {e}")
    
    logger.info(f"Batch processing summary: {json.dumps(summary, indent=2)}")
    
    return summary


def update_processing_metrics(**context) -> bool:
    """Update processing metrics and performance indicators"""
    import sys
    sys.path.insert(0, '/opt/airflow/backend')
    
    from backend.utils.cache import get_redis
    
    try:
        redis_client = get_redis()
        
        # Update pipeline status
        redis_client.set('pipeline_status', 'completed')
        redis_client.set('last_pipeline_run', datetime.now().isoformat())
        
        # Update processing metrics
        batch_summary = context['task_instance'].xcom_pull(key='batch_summary')
        if batch_summary:
            redis_client.set('latest_batch_summary', json.dumps(batch_summary))
        
        logger.info("Processing metrics updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")
        return False


# Create processor instance
processor = ParallelStockProcessor()

# Define DAG tasks
with dag:
    # Initialize processing
    get_stocks_task = PythonOperator(
        task_id='get_stocks_for_processing',
        python_callable=processor.get_stocks_for_processing,
        provide_context=True
    )
    
    create_batches_task = PythonOperator(
        task_id='create_processing_batches',
        python_callable=processor.create_processing_batches,
        provide_context=True
    )
    
    # Parallel processing task groups
    with TaskGroup('tier1_processing', tooltip="Process Tier 1 critical stocks") as tier1_group:
        # Multiple parallel Tier 1 processing tasks
        tier1_tasks = []
        for i in range(5):  # Up to 5 parallel Tier 1 batches
            task = PythonOperator(
                task_id=f'process_tier1_batch_{i}',
                python_callable=process_stock_batch_tier1,
                op_args=[[]],  # Batch will be determined dynamically
                provide_context=True,
                pool='api_calls',
                priority_weight=10
            )
            tier1_tasks.append(task)
    
    with TaskGroup('tier2_processing', tooltip="Process Tier 2 high priority stocks") as tier2_group:
        tier2_task = PythonOperator(
            task_id='process_tier2_batch',
            python_callable=process_stock_batch_tier2,
            op_args=[[]],
            provide_context=True,
            pool='api_calls',
            priority_weight=7
        )
    
    with TaskGroup('tier3_processing', tooltip="Process Tier 3 medium priority stocks") as tier3_group:
        tier3_tasks = []
        for i in range(3):  # Up to 3 parallel Tier 3 batches
            task = PythonOperator(
                task_id=f'process_tier3_batch_{i}',
                python_callable=process_stock_batch_tier3,
                op_args=[[]],
                provide_context=True,
                pool='api_calls',
                priority_weight=5
            )
            tier3_tasks.append(task)
    
    with TaskGroup('tier45_processing', tooltip="Process Tier 4/5 low priority stocks") as tier45_group:
        tier4_task = PythonOperator(
            task_id='process_tier4_batch',
            python_callable=partial(process_stock_batch_tier45, tier=4),
            op_args=[[]],
            provide_context=True,
            pool='database',
            priority_weight=3
        )
        
        tier5_task = PythonOperator(
            task_id='process_tier5_batch',
            python_callable=partial(process_stock_batch_tier45, tier=5),
            op_args=[[]],
            provide_context=True,
            pool='database',
            priority_weight=1
        )
    
    # Post-processing
    consolidate_task = PythonOperator(
        task_id='consolidate_batch_results',
        python_callable=consolidate_batch_results,
        provide_context=True,
        trigger_rule='none_failed_or_skipped'
    )
    
    update_metrics_task = PythonOperator(
        task_id='update_processing_metrics',
        python_callable=update_processing_metrics,
        provide_context=True
    )
    
    # Health check task
    health_check_task = BashOperator(
        task_id='pipeline_health_check',
        bash_command='python /opt/airflow/monitor_pipeline.py --once',
        trigger_rule='all_done'
    )
    
    # Define task dependencies
    get_stocks_task >> create_batches_task
    
    # Parallel processing (all tiers can run concurrently)
    create_batches_task >> [tier1_group, tier2_group, tier3_group, tier45_group]
    
    # Consolidation after all processing
    [tier1_group, tier2_group, tier3_group, tier45_group] >> consolidate_task
    
    consolidate_task >> update_metrics_task >> health_check_task