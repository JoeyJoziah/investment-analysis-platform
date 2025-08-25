"""
Minimal Airflow DAG for Daily Stock Data Pipeline
This is a simplified version focusing on core functionality
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import yfinance as yf
import pandas as pd
import numpy as np
import logging

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
    'daily_stock_pipeline',
    default_args=default_args,
    description='Daily stock data ingestion and analysis pipeline',
    schedule_interval='0 6 * * *',  # Run at 6 AM daily
    catchup=False,
    tags=['stocks', 'daily', 'production'],
)

def get_active_stocks(**context):
    """Fetch list of active stocks from database"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    sql = "SELECT ticker FROM stocks WHERE is_active = true LIMIT 100"
    records = pg_hook.get_records(sql)
    tickers = [record[0] for record in records]
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='stock_tickers', value=tickers)
    logging.info(f"Found {len(tickers)} active stocks")
    return tickers

def fetch_stock_data(**context):
    """Fetch stock price data using yfinance"""
    tickers = context['task_instance'].xcom_pull(key='stock_tickers')
    
    if not tickers:
        logging.warning("No tickers to process")
        return
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    success_count = 0
    error_count = 0
    
    for ticker in tickers:
        try:
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if hist.empty:
                logging.warning(f"No data for {ticker}")
                error_count += 1
                continue
            
            # Get latest data
            latest = hist.iloc[-1]
            date = hist.index[-1].date()
            
            # Get stock_id
            cursor.execute("SELECT id FROM stocks WHERE ticker = %s", (ticker,))
            result = cursor.fetchone()
            
            if not result:
                logging.warning(f"Stock {ticker} not found in database")
                error_count += 1
                continue
            
            stock_id = result[0]
            
            # Insert price data
            insert_sql = """
                INSERT INTO price_history (stock_id, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """
            
            cursor.execute(insert_sql, (
                stock_id, date,
                float(latest['Open']), float(latest['High']),
                float(latest['Low']), float(latest['Close']),
                int(latest['Volume'])
            ))
            
            success_count += 1
            
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            error_count += 1
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logging.info(f"Processed {success_count} stocks successfully, {error_count} errors")
    context['task_instance'].xcom_push(key='fetch_stats', value={
        'success': success_count,
        'errors': error_count
    })

def calculate_indicators(**context):
    """Calculate technical indicators for all stocks"""
    tickers = context['task_instance'].xcom_pull(key='stock_tickers')
    
    if not tickers:
        return
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    for ticker in tickers[:50]:  # Process first 50 to avoid timeout
        try:
            # Get stock_id
            cursor.execute("SELECT id FROM stocks WHERE ticker = %s", (ticker,))
            result = cursor.fetchone()
            
            if not result:
                continue
            
            stock_id = result[0]
            
            # Fetch recent price history
            cursor.execute("""
                SELECT date, close FROM price_history 
                WHERE stock_id = %s 
                ORDER BY date DESC 
                LIMIT 50
            """, (stock_id,))
            
            prices = cursor.fetchall()
            
            if len(prices) < 20:
                logging.warning(f"Insufficient data for {ticker}")
                continue
            
            # Calculate indicators
            closes = [float(p[1]) for p in prices]
            closes.reverse()  # Oldest first
            
            # SMA calculations
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes) if len(closes) >= 50 else sma_20
            
            # RSI calculation (simplified)
            def calc_rsi(prices, period=14):
                if len(prices) < period:
                    return 50
                
                deltas = np.diff(prices[-period-1:])
                gains = deltas[deltas > 0].sum() / period
                losses = -deltas[deltas < 0].sum() / period
                
                if losses == 0:
                    return 100
                
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            rsi = calc_rsi(closes)
            
            # MACD (simplified)
            if len(closes) >= 26:
                ema_12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
                ema_26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
                macd = ema_12 - ema_26
                signal = pd.Series(closes).ewm(span=9).mean().iloc[-1]
            else:
                macd = 0
                signal = 0
            
            # Bollinger Bands
            std_20 = np.std(closes[-20:])
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            
            # Insert indicators
            insert_sql = """
                INSERT INTO technical_indicators 
                (stock_id, date, sma_20, sma_50, rsi_14, macd, macd_signal,
                 bollinger_upper, bollinger_middle, bollinger_lower)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    sma_20 = EXCLUDED.sma_20,
                    sma_50 = EXCLUDED.sma_50,
                    rsi_14 = EXCLUDED.rsi_14
            """
            
            cursor.execute(insert_sql, (
                stock_id, datetime.now(),
                round(sma_20, 2), round(sma_50, 2), round(rsi, 2),
                round(macd, 4), round(signal, 4),
                round(bb_upper, 2), round(sma_20, 2), round(bb_lower, 2)
            ))
            
        except Exception as e:
            logging.error(f"Error calculating indicators for {ticker}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logging.info("Technical indicators calculated successfully")

def generate_recommendations(**context):
    """Generate daily stock recommendations"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # Query stocks with good technical signals
    sql = """
        SELECT s.id, s.ticker, ti.rsi_14, ti.sma_20, ti.sma_50,
               ph.close, ti.macd, ti.macd_signal
        FROM stocks s
        JOIN technical_indicators ti ON s.id = ti.stock_id
        JOIN price_history ph ON s.id = ph.stock_id
        WHERE s.is_active = true
          AND ti.date >= CURRENT_DATE - INTERVAL '1 day'
          AND ph.date >= CURRENT_DATE - INTERVAL '1 day'
        ORDER BY ti.rsi_14 ASC
        LIMIT 20
    """
    
    cursor.execute(sql)
    stocks = cursor.fetchall()
    
    recommendations = []
    
    for stock in stocks:
        stock_id, ticker, rsi, sma_20, sma_50, close, macd, macd_signal = stock
        
        # Simple scoring system
        score = 0
        reasons = []
        
        # RSI signals
        if rsi and rsi < 30:
            score += 3
            reasons.append("Strong oversold signal")
        elif rsi and rsi < 40:
            score += 1
            reasons.append("Oversold")
        elif rsi and rsi > 70:
            score -= 2
            reasons.append("Overbought")
        
        # Moving average signals
        if sma_20 and sma_50 and sma_20 > sma_50:
            score += 2
            reasons.append("Bullish MA crossover")
        
        # MACD signals
        if macd and macd_signal and macd > macd_signal:
            score += 1
            reasons.append("Positive MACD signal")
        
        # Price vs MA
        if close and sma_20 and close > sma_20:
            score += 1
            reasons.append("Price above MA20")
        
        # Determine action
        if score >= 3:
            action = 'strong_buy'
            confidence = min(0.7 + (score * 0.05), 0.95)
        elif score >= 2:
            action = 'buy'
            confidence = 0.6 + (score * 0.05)
        elif score <= -2:
            action = 'sell'
            confidence = 0.6
        else:
            action = 'hold'
            confidence = 0.5
        
        if action in ['buy', 'strong_buy'] and len(recommendations) < 10:
            # Insert recommendation
            insert_sql = """
                INSERT INTO recommendations
                (stock_id, action, confidence, reasoning,
                 technical_score, is_active, created_at, priority,
                 target_price, stop_loss, time_horizon_days)
                VALUES (%s, %s, %s, %s, %s, true, %s, %s, %s, %s, %s)
            """
            
            target_price = close * 1.05 if close else 100  # 5% target
            stop_loss = close * 0.97 if close else 95  # 3% stop loss
            
            cursor.execute(insert_sql, (
                stock_id, action, confidence,
                '; '.join(reasons),
                score / 10.0,
                datetime.now(),
                int(confidence * 10),
                round(target_price, 2),
                round(stop_loss, 2),
                30  # 30-day horizon
            ))
            
            recommendations.append({
                'ticker': ticker,
                'action': action,
                'confidence': confidence,
                'reasons': reasons
            })
    
    # Deactivate old recommendations
    cursor.execute("""
        UPDATE recommendations 
        SET is_active = false 
        WHERE created_at < CURRENT_DATE - INTERVAL '7 days'
          AND is_active = true
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logging.info(f"Generated {len(recommendations)} recommendations")
    context['task_instance'].xcom_push(key='recommendations', value=recommendations)

def send_summary(**context):
    """Send pipeline summary"""
    fetch_stats = context['task_instance'].xcom_pull(key='fetch_stats')
    recommendations = context['task_instance'].xcom_pull(key='recommendations')
    
    summary = f"""
    Daily Pipeline Summary:
    - Stocks Processed: {fetch_stats.get('success', 0)}
    - Errors: {fetch_stats.get('errors', 0)}
    - Recommendations Generated: {len(recommendations) if recommendations else 0}
    
    Top Recommendations:
    """
    
    if recommendations:
        for rec in recommendations[:5]:
            summary += f"\n- {rec['ticker']}: {rec['action']} (confidence: {rec['confidence']:.2f})"
    
    logging.info(summary)
    
    # Here you could send email, Slack notification, etc.
    return summary

# Define tasks
start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

get_stocks_task = PythonOperator(
    task_id='get_active_stocks',
    python_callable=get_active_stocks,
    provide_context=True,
    dag=dag,
)

fetch_data_task = PythonOperator(
    task_id='fetch_stock_data',
    python_callable=fetch_stock_data,
    provide_context=True,
    dag=dag,
)

calculate_indicators_task = PythonOperator(
    task_id='calculate_indicators',
    python_callable=calculate_indicators,
    provide_context=True,
    dag=dag,
)

generate_recommendations_task = PythonOperator(
    task_id='generate_recommendations',
    python_callable=generate_recommendations,
    provide_context=True,
    dag=dag,
)

cleanup_task = PostgresOperator(
    task_id='cleanup_old_data',
    postgres_conn_id='postgres_default',
    sql="""
        -- Clean up old price data (keep last 2 years)
        DELETE FROM price_history 
        WHERE date < CURRENT_DATE - INTERVAL '2 years';
        
        -- Clean up old indicators
        DELETE FROM technical_indicators 
        WHERE date < CURRENT_DATE - INTERVAL '6 months';
        
        -- Archive old recommendations
        UPDATE recommendations 
        SET is_active = false 
        WHERE created_at < CURRENT_DATE - INTERVAL '30 days';
    """,
    dag=dag,
)

summary_task = PythonOperator(
    task_id='send_summary',
    python_callable=send_summary,
    provide_context=True,
    trigger_rule='all_done',  # Run even if some tasks fail
    dag=dag,
)

end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define task dependencies
start_task >> get_stocks_task >> fetch_data_task
fetch_data_task >> calculate_indicators_task >> generate_recommendations_task
[generate_recommendations_task, cleanup_task] >> summary_task >> end_task