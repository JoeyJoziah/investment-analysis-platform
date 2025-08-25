"""
Data Loader Module
Handles loading transformed data into PostgreSQL/TimescaleDB
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.pool import QueuePool
import psycopg2
from psycopg2.extras import execute_batch, execute_values
import os
from dotenv import load_dotenv
from contextlib import contextmanager
import json

load_dotenv()

logger = logging.getLogger(__name__)


class DataLoader:
    """Load transformed data into PostgreSQL/TimescaleDB"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'investment_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)
        
    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling"""
        connection_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        
        return create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False
        )
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_raw_connection(self):
        """Get raw psycopg2 connection for bulk operations"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            yield conn
            conn.commit()
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def ensure_stock_exists(self, ticker: str, company_info: Dict = None) -> int:
        """Ensure stock exists in database and return stock_id"""
        with self.get_connection() as conn:
            # Check if stock exists
            result = conn.execute(
                text("SELECT id FROM stocks WHERE ticker = :ticker"),
                {'ticker': ticker}
            ).fetchone()
            
            if result:
                return result[0]
            
            # Insert new stock
            insert_query = text("""
                INSERT INTO stocks (ticker, name, exchange, sector, industry, is_active)
                VALUES (:ticker, :name, :exchange, :sector, :industry, true)
                RETURNING id
            """)
            
            params = {
                'ticker': ticker,
                'name': company_info.get('name', ticker) if company_info else ticker,
                'exchange': company_info.get('exchange', 'UNKNOWN') if company_info else 'UNKNOWN',
                'sector': company_info.get('sector') if company_info else None,
                'industry': company_info.get('industry') if company_info else None
            }
            
            result = conn.execute(insert_query, params).fetchone()
            conn.commit()
            
            return result[0]
    
    def load_price_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """Load price data into price_history table"""
        try:
            if df.empty:
                logger.warning(f"No price data to load for {ticker}")
                return False
            
            stock_id = self.ensure_stock_exists(ticker)
            
            # Prepare data for insertion
            records = []
            for _, row in df.iterrows():
                records.append((
                    stock_id,
                    row['date'],
                    float(row['open']) if pd.notna(row['open']) else None,
                    float(row['high']) if pd.notna(row['high']) else None,
                    float(row['low']) if pd.notna(row['low']) else None,
                    float(row['close']) if pd.notna(row['close']) else None,
                    int(row['volume']) if pd.notna(row['volume']) else None,
                    float(row.get('adjusted_close', row['close'])) if 'adjusted_close' in row else None,
                    float(row.get('intraday_range', 0)) if 'intraday_range' in row else None,
                    float(row.get('vwap', 0)) if 'vwap' in row else None
                ))
            
            with self.get_raw_connection() as conn:
                cursor = conn.cursor()
                
                # Use COPY for bulk insertion (fastest method)
                insert_query = """
                    INSERT INTO price_history 
                    (stock_id, date, open, high, low, close, volume, 
                     adjusted_close, intraday_volatility, vwap)
                    VALUES %s
                    ON CONFLICT (stock_id, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        adjusted_close = EXCLUDED.adjusted_close,
                        intraday_volatility = EXCLUDED.intraday_volatility,
                        vwap = EXCLUDED.vwap,
                        updated_at = CURRENT_TIMESTAMP
                """
                
                execute_values(cursor, insert_query, records)
                
                logger.info(f"Loaded {len(records)} price records for {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {e}")
            return False
    
    def load_technical_indicators(self, df: pd.DataFrame, ticker: str) -> bool:
        """Load technical indicators into database"""
        try:
            if df.empty:
                logger.warning(f"No technical indicators to load for {ticker}")
                return False
            
            stock_id = self.ensure_stock_exists(ticker)
            
            # Get latest row with indicators
            latest = df.iloc[-1]
            
            with self.get_connection() as conn:
                insert_query = text("""
                    INSERT INTO technical_indicators
                    (stock_id, date, sma_20, sma_50, sma_200, ema_12, ema_26,
                     rsi_14, macd, macd_signal, macd_histogram,
                     bollinger_upper, bollinger_middle, bollinger_lower,
                     atr_14, adx, cci, mfi, obv, stoch_k, stoch_d,
                     williams_r, roc_10, momentum_10)
                    VALUES 
                    (:stock_id, :date, :sma_20, :sma_50, :sma_200, :ema_12, :ema_26,
                     :rsi_14, :macd, :macd_signal, :macd_histogram,
                     :bollinger_upper, :bollinger_middle, :bollinger_lower,
                     :atr_14, :adx, :cci, :mfi, :obv, :stoch_k, :stoch_d,
                     :williams_r, :roc_10, :momentum_10)
                    ON CONFLICT (stock_id, date) DO UPDATE SET
                        sma_20 = EXCLUDED.sma_20,
                        sma_50 = EXCLUDED.sma_50,
                        sma_200 = EXCLUDED.sma_200,
                        ema_12 = EXCLUDED.ema_12,
                        ema_26 = EXCLUDED.ema_26,
                        rsi_14 = EXCLUDED.rsi_14,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram,
                        bollinger_upper = EXCLUDED.bollinger_upper,
                        bollinger_middle = EXCLUDED.bollinger_middle,
                        bollinger_lower = EXCLUDED.bollinger_lower,
                        atr_14 = EXCLUDED.atr_14,
                        adx = EXCLUDED.adx,
                        cci = EXCLUDED.cci,
                        mfi = EXCLUDED.mfi,
                        obv = EXCLUDED.obv,
                        stoch_k = EXCLUDED.stoch_k,
                        stoch_d = EXCLUDED.stoch_d,
                        williams_r = EXCLUDED.williams_r,
                        roc_10 = EXCLUDED.roc_10,
                        momentum_10 = EXCLUDED.momentum_10,
                        updated_at = CURRENT_TIMESTAMP
                """)
                
                params = {
                    'stock_id': stock_id,
                    'date': datetime.now(),
                    'sma_20': float(latest.get('sma_20', 0)) if pd.notna(latest.get('sma_20')) else None,
                    'sma_50': float(latest.get('sma_50', 0)) if pd.notna(latest.get('sma_50')) else None,
                    'sma_200': float(latest.get('sma_200', 0)) if pd.notna(latest.get('sma_200')) else None,
                    'ema_12': float(latest.get('ema_12', 0)) if pd.notna(latest.get('ema_12')) else None,
                    'ema_26': float(latest.get('ema_26', 0)) if pd.notna(latest.get('ema_26')) else None,
                    'rsi_14': float(latest.get('rsi_14', 50)) if pd.notna(latest.get('rsi_14')) else None,
                    'macd': float(latest.get('macd', 0)) if pd.notna(latest.get('macd')) else None,
                    'macd_signal': float(latest.get('macd_signal', 0)) if pd.notna(latest.get('macd_signal')) else None,
                    'macd_histogram': float(latest.get('macd_hist', 0)) if pd.notna(latest.get('macd_hist')) else None,
                    'bollinger_upper': float(latest.get('bb_upper', 0)) if pd.notna(latest.get('bb_upper')) else None,
                    'bollinger_middle': float(latest.get('bb_middle', 0)) if pd.notna(latest.get('bb_middle')) else None,
                    'bollinger_lower': float(latest.get('bb_lower', 0)) if pd.notna(latest.get('bb_lower')) else None,
                    'atr_14': float(latest.get('atr_14', 0)) if pd.notna(latest.get('atr_14')) else None,
                    'adx': float(latest.get('adx', 0)) if pd.notna(latest.get('adx')) else None,
                    'cci': float(latest.get('cci', 0)) if pd.notna(latest.get('cci')) else None,
                    'mfi': float(latest.get('mfi', 50)) if pd.notna(latest.get('mfi')) else None,
                    'obv': float(latest.get('obv', 0)) if pd.notna(latest.get('obv')) else None,
                    'stoch_k': float(latest.get('stoch_k', 50)) if pd.notna(latest.get('stoch_k')) else None,
                    'stoch_d': float(latest.get('stoch_d', 50)) if pd.notna(latest.get('stoch_d')) else None,
                    'williams_r': float(latest.get('williams_r', -50)) if pd.notna(latest.get('williams_r')) else None,
                    'roc_10': float(latest.get('roc_10', 0)) if pd.notna(latest.get('roc_10')) else None,
                    'momentum_10': float(latest.get('momentum_10', 0)) if pd.notna(latest.get('momentum_10')) else None
                }
                
                conn.execute(insert_query, params)
                conn.commit()
                
                logger.info(f"Loaded technical indicators for {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading technical indicators for {ticker}: {e}")
            return False
    
    def load_sentiment_data(self, sentiment_data: Dict, ticker: str) -> bool:
        """Load sentiment data into database"""
        try:
            if not sentiment_data:
                logger.warning(f"No sentiment data to load for {ticker}")
                return False
            
            stock_id = self.ensure_stock_exists(ticker)
            
            with self.get_connection() as conn:
                # Load aggregated sentiment
                insert_query = text("""
                    INSERT INTO news_sentiment
                    (stock_id, date, sentiment_score, sentiment_confidence,
                     article_count, positive_count, negative_count, neutral_count,
                     source, raw_data)
                    VALUES
                    (:stock_id, :date, :sentiment_score, :sentiment_confidence,
                     :article_count, :positive_count, :negative_count, :neutral_count,
                     :source, :raw_data)
                    ON CONFLICT (stock_id, date, source) DO UPDATE SET
                        sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_confidence = EXCLUDED.sentiment_confidence,
                        article_count = EXCLUDED.article_count,
                        positive_count = EXCLUDED.positive_count,
                        negative_count = EXCLUDED.negative_count,
                        neutral_count = EXCLUDED.neutral_count,
                        raw_data = EXCLUDED.raw_data,
                        updated_at = CURRENT_TIMESTAMP
                """)
                
                # Calculate sentiment counts
                articles = sentiment_data.get('articles', [])
                positive_count = sum(1 for a in articles if 
                                   any(word in str(a).lower() for word in ['gain', 'rise', 'up']))
                negative_count = sum(1 for a in articles if 
                                   any(word in str(a).lower() for word in ['loss', 'fall', 'down']))
                neutral_count = len(articles) - positive_count - negative_count
                
                params = {
                    'stock_id': stock_id,
                    'date': datetime.now().date(),
                    'sentiment_score': float(sentiment_data.get('sentiment_score', 0)),
                    'sentiment_confidence': float(sentiment_data.get('sentiment_confidence', 0)),
                    'article_count': sentiment_data.get('article_count', 0),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'source': sentiment_data.get('source', 'mixed'),
                    'raw_data': json.dumps(sentiment_data.get('articles', [])[:5])  # Store first 5 articles
                }
                
                conn.execute(insert_query, params)
                conn.commit()
                
                logger.info(f"Loaded sentiment data for {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading sentiment data for {ticker}: {e}")
            return False
    
    def load_ml_predictions(self, predictions: Dict, ticker: str) -> bool:
        """Load ML model predictions into database"""
        try:
            stock_id = self.ensure_stock_exists(ticker)
            
            with self.get_connection() as conn:
                insert_query = text("""
                    INSERT INTO ml_predictions
                    (stock_id, prediction_date, model_name, model_version,
                     predicted_return, confidence_score, prediction_horizon_days,
                     features_used, model_params, is_active)
                    VALUES
                    (:stock_id, :prediction_date, :model_name, :model_version,
                     :predicted_return, :confidence_score, :prediction_horizon_days,
                     :features_used, :model_params, true)
                """)
                
                params = {
                    'stock_id': stock_id,
                    'prediction_date': datetime.now(),
                    'model_name': predictions.get('model_name', 'ensemble'),
                    'model_version': predictions.get('model_version', '1.0'),
                    'predicted_return': float(predictions.get('predicted_return', 0)),
                    'confidence_score': float(predictions.get('confidence', 0.5)),
                    'prediction_horizon_days': predictions.get('horizon_days', 5),
                    'features_used': json.dumps(predictions.get('features', [])),
                    'model_params': json.dumps(predictions.get('params', {}))
                }
                
                conn.execute(insert_query, params)
                conn.commit()
                
                logger.info(f"Loaded ML predictions for {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading ML predictions for {ticker}: {e}")
            return False
    
    def load_recommendations(self, recommendations: List[Dict]) -> bool:
        """Load investment recommendations into database"""
        try:
            if not recommendations:
                logger.warning("No recommendations to load")
                return False
            
            with self.get_raw_connection() as conn:
                cursor = conn.cursor()
                
                # Deactivate old recommendations
                cursor.execute("""
                    UPDATE recommendations 
                    SET is_active = false 
                    WHERE is_active = true 
                    AND created_at < CURRENT_TIMESTAMP - INTERVAL '7 days'
                """)
                
                # Prepare recommendation records
                records = []
                for rec in recommendations:
                    stock_id = self.ensure_stock_exists(rec['ticker'])
                    
                    records.append((
                        stock_id,
                        rec['action'],
                        float(rec['confidence']),
                        rec.get('reasoning', ''),
                        float(rec.get('technical_score', 0)),
                        float(rec.get('fundamental_score', 0)),
                        float(rec.get('sentiment_score', 0)),
                        float(rec.get('ml_score', 0)),
                        True,  # is_active
                        datetime.now(),
                        rec.get('priority', 5),
                        float(rec.get('target_price', 0)),
                        float(rec.get('stop_loss', 0)),
                        rec.get('time_horizon_days', 30)
                    ))
                
                # Bulk insert recommendations
                insert_query = """
                    INSERT INTO recommendations
                    (stock_id, action, confidence, reasoning,
                     technical_score, fundamental_score, sentiment_score, ml_score,
                     is_active, created_at, priority,
                     target_price, stop_loss, time_horizon_days)
                    VALUES %s
                """
                
                execute_values(cursor, insert_query, records)
                
                logger.info(f"Loaded {len(records)} recommendations")
                return True
                
        except Exception as e:
            logger.error(f"Error loading recommendations: {e}")
            return False
    
    def cleanup_old_data(self, retention_days: Dict = None) -> bool:
        """Clean up old data based on retention policies"""
        try:
            if retention_days is None:
                retention_days = {
                    'price_history': 730,  # 2 years
                    'technical_indicators': 180,  # 6 months
                    'news_sentiment': 90,  # 3 months
                    'ml_predictions': 30,  # 1 month
                    'recommendations': 30  # 1 month
                }
            
            with self.get_connection() as conn:
                for table, days in retention_days.items():
                    if table == 'recommendations':
                        # Archive old recommendations
                        query = text(f"""
                            UPDATE {table} 
                            SET is_active = false 
                            WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '{days} days'
                            AND is_active = true
                        """)
                    else:
                        # Delete old data
                        date_column = 'date' if table != 'ml_predictions' else 'prediction_date'
                        query = text(f"""
                            DELETE FROM {table} 
                            WHERE {date_column} < CURRENT_TIMESTAMP - INTERVAL '{days} days'
                        """)
                    
                    result = conn.execute(query)
                    conn.commit()
                    
                    logger.info(f"Cleaned up {result.rowcount} rows from {table}")
                
                # Vacuum analyze for performance
                conn.execute(text("VACUUM ANALYZE"))
                
                return True
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False
    
    def get_loading_stats(self) -> Dict:
        """Get statistics about loaded data"""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Get counts for each table
                tables = [
                    'stocks', 'price_history', 'technical_indicators',
                    'news_sentiment', 'ml_predictions', 'recommendations'
                ]
                
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    stats[f'{table}_count'] = result[0]
                
                # Get date ranges
                result = conn.execute(text("""
                    SELECT MIN(date) as min_date, MAX(date) as max_date 
                    FROM price_history
                """)).fetchone()
                
                if result:
                    stats['price_data_range'] = {
                        'start': result[0],
                        'end': result[1]
                    }
                
                # Get active recommendations count
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM recommendations 
                    WHERE is_active = true
                """)).fetchone()
                
                stats['active_recommendations'] = result[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting loading stats: {e}")
            return {}


class BatchLoader:
    """Optimized batch loading for large datasets"""
    
    def __init__(self, loader: DataLoader, batch_size: int = 1000):
        self.loader = loader
        self.batch_size = batch_size
    
    def load_dataframe_batch(self, df: pd.DataFrame, table_name: str, 
                            ticker: str = None) -> bool:
        """Load DataFrame in batches"""
        try:
            total_rows = len(df)
            num_batches = (total_rows + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Loading {total_rows} rows in {num_batches} batches")
            
            for i in range(0, total_rows, self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                
                if table_name == 'price_history':
                    success = self.loader.load_price_data(batch, ticker)
                elif table_name == 'technical_indicators':
                    success = self.loader.load_technical_indicators(batch, ticker)
                else:
                    logger.error(f"Unknown table: {table_name}")
                    return False
                
                if not success:
                    logger.error(f"Failed to load batch {i//self.batch_size + 1}")
                    return False
                
                logger.info(f"Loaded batch {i//self.batch_size + 1}/{num_batches}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in batch loading: {e}")
            return False


if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    
    # Test connection
    stats = loader.get_loading_stats()
    print("Database statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Test cleanup
    loader.cleanup_old_data()
    
    print("Data loader initialized successfully")