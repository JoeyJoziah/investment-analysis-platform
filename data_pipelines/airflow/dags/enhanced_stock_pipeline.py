"""
Enhanced Airflow DAG for Daily Stock Data Pipeline
Uses the new ETL orchestrator for improved data processing
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import asyncio
import logging
import json
import sys
import os

# Add project root to path
sys.path.insert(0, '/app')

from backend.etl.etl_orchestrator import ETLOrchestrator, ETLScheduler
from backend.etl.data_loader import DataLoader

# Default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'enhanced_stock_pipeline',
    default_args=default_args,
    description='Enhanced daily stock data ETL pipeline',
    schedule_interval='0 6 * * *',  # Run at 6 AM daily
    catchup=False,
    tags=['stocks', 'etl', 'production'],
)


def run_etl_pipeline(**context):
    """Run the complete ETL pipeline"""
    orchestrator = ETLOrchestrator()
    
    # Configure for production
    orchestrator.config.update({
        'batch_size': 20,
        'max_workers': 4,
        'enable_ml': True,
        'enable_sentiment': True,
        'enable_recommendations': True
    })
    
    # Run async pipeline
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(orchestrator.run_full_pipeline())
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='pipeline_metrics', value=result)
        
        logging.info(f"Pipeline completed: {result}")
        
        if result.get('errors'):
            logging.warning(f"Pipeline had {len(result['errors'])} errors")
        
        return result
        
    finally:
        loop.close()


def run_incremental_update(**context):
    """Run incremental update for high-priority stocks"""
    orchestrator = ETLOrchestrator()
    
    # Get high-priority tickers
    watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    # Configure for incremental update
    orchestrator.config.update({
        'batch_size': 5,
        'max_workers': 2,
        'enable_ml': True,
        'enable_sentiment': False,  # Skip sentiment for speed
        'enable_recommendations': False
    })
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            orchestrator.run_incremental_update(watchlist)
        )
        
        context['task_instance'].xcom_push(key='incremental_metrics', value=result)
        
        logging.info(f"Incremental update completed: {result}")
        return result
        
    finally:
        loop.close()


def validate_data_quality(**context):
    """Validate data quality after ETL"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    quality_checks = []
    
    # Check 1: Price data completeness
    sql = """
        SELECT COUNT(DISTINCT stock_id) as stocks_with_prices,
               COUNT(*) as total_price_records,
               MAX(date) as latest_date
        FROM price_history
        WHERE date >= CURRENT_DATE - INTERVAL '1 day'
    """
    result = pg_hook.get_records(sql)[0]
    quality_checks.append({
        'check': 'price_data_completeness',
        'stocks_with_prices': result[0],
        'total_records': result[1],
        'latest_date': str(result[2])
    })
    
    # Check 2: Technical indicators freshness
    sql = """
        SELECT COUNT(DISTINCT stock_id) as stocks_with_indicators,
               MAX(date) as latest_date
        FROM technical_indicators
        WHERE date >= CURRENT_DATE - INTERVAL '1 day'
    """
    result = pg_hook.get_records(sql)[0]
    quality_checks.append({
        'check': 'technical_indicators_freshness',
        'stocks_with_indicators': result[0],
        'latest_date': str(result[1])
    })
    
    # Check 3: Recommendations generated
    sql = """
        SELECT COUNT(*) as active_recommendations,
               AVG(confidence) as avg_confidence
        FROM recommendations
        WHERE is_active = true
        AND created_at >= CURRENT_DATE - INTERVAL '1 day'
    """
    result = pg_hook.get_records(sql)[0]
    quality_checks.append({
        'check': 'recommendations',
        'active_count': result[0],
        'avg_confidence': float(result[1]) if result[1] else 0
    })
    
    # Push to XCom
    context['task_instance'].xcom_push(key='quality_checks', value=quality_checks)
    
    logging.info(f"Data quality checks: {json.dumps(quality_checks, indent=2)}")
    
    # Fail if critical checks don't pass
    if quality_checks[0]['stocks_with_prices'] < 10:
        raise ValueError("Insufficient price data loaded")
    
    return quality_checks


def generate_daily_report(**context):
    """Generate daily pipeline report"""
    # Pull metrics from XCom
    pipeline_metrics = context['task_instance'].xcom_pull(
        task_ids='run_etl_pipeline', 
        key='pipeline_metrics'
    )
    
    quality_checks = context['task_instance'].xcom_pull(
        task_ids='validate_data_quality',
        key='quality_checks'
    )
    
    # Get database stats
    loader = DataLoader()
    db_stats = loader.get_loading_stats()
    
    # Create report
    report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'pipeline_metrics': pipeline_metrics,
        'quality_checks': quality_checks,
        'database_stats': db_stats,
        'status': 'SUCCESS' if pipeline_metrics.get('stocks_processed', 0) > 0 else 'FAILED'
    }
    
    # Log report
    logging.info("="*60)
    logging.info("DAILY ETL PIPELINE REPORT")
    logging.info("="*60)
    logging.info(json.dumps(report, indent=2, default=str))
    
    # Store report in database (optional)
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO pipeline_logs (run_date, status, metrics, report)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (run_date) DO UPDATE SET
            status = EXCLUDED.status,
            metrics = EXCLUDED.metrics,
            report = EXCLUDED.report,
            updated_at = CURRENT_TIMESTAMP
    """, (
        datetime.now().date(),
        report['status'],
        json.dumps(pipeline_metrics, default=str),
        json.dumps(report, default=str)
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return report


def cleanup_and_optimize(**context):
    """Cleanup old data and optimize database"""
    loader = DataLoader()
    
    # Cleanup old data
    loader.cleanup_old_data({
        'price_history': 730,  # 2 years
        'technical_indicators': 180,  # 6 months
        'news_sentiment': 90,  # 3 months
        'ml_predictions': 30,  # 1 month
        'recommendations': 30  # 1 month
    })
    
    # Additional cleanup via SQL
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    cleanup_queries = [
        # Archive old recommendations
        """
        UPDATE recommendations 
        SET is_active = false 
        WHERE created_at < CURRENT_DATE - INTERVAL '30 days'
        AND is_active = true
        """,
        
        # Clean up old pipeline logs
        """
        DELETE FROM pipeline_logs 
        WHERE run_date < CURRENT_DATE - INTERVAL '90 days'
        """,
        
        # Vacuum and analyze for performance
        "VACUUM ANALYZE price_history",
        "VACUUM ANALYZE technical_indicators",
        "VACUUM ANALYZE recommendations"
    ]
    
    for query in cleanup_queries:
        try:
            pg_hook.run(query)
            logging.info(f"Executed cleanup: {query[:50]}...")
        except Exception as e:
            logging.error(f"Cleanup query failed: {e}")
    
    logging.info("Cleanup and optimization completed")


# Define tasks
start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

# Main ETL pipeline
etl_task = PythonOperator(
    task_id='run_etl_pipeline',
    python_callable=run_etl_pipeline,
    provide_context=True,
    execution_timeout=timedelta(hours=2),
    dag=dag,
)

# Incremental update for high-priority stocks
incremental_task = PythonOperator(
    task_id='run_incremental_update',
    python_callable=run_incremental_update,
    provide_context=True,
    execution_timeout=timedelta(minutes=30),
    dag=dag,
)

# Data quality validation
validate_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    provide_context=True,
    dag=dag,
)

# Generate report
report_task = PythonOperator(
    task_id='generate_daily_report',
    python_callable=generate_daily_report,
    provide_context=True,
    trigger_rule='all_done',  # Run even if some tasks fail
    dag=dag,
)

# Cleanup and optimization
cleanup_task = PythonOperator(
    task_id='cleanup_and_optimize',
    python_callable=cleanup_and_optimize,
    provide_context=True,
    dag=dag,
)

# Create pipeline_logs table if not exists
create_log_table_task = PostgresOperator(
    task_id='create_log_table',
    postgres_conn_id='postgres_default',
    sql="""
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id SERIAL PRIMARY KEY,
            run_date DATE UNIQUE,
            status VARCHAR(50),
            metrics JSONB,
            report JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_pipeline_logs_date 
        ON pipeline_logs(run_date DESC);
    """,
    dag=dag,
)

end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define task dependencies
start_task >> create_log_table_task
create_log_table_task >> [etl_task, incremental_task]
[etl_task, incremental_task] >> validate_task
validate_task >> report_task
report_task >> cleanup_task >> end_task