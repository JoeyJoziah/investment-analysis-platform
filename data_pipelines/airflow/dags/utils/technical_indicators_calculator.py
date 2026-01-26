"""
Optimized Technical Indicators Calculator

This module provides high-performance technical indicator calculation using PostgreSQL
window functions and bulk operations. Designed to process 6000+ stocks efficiently.

Key optimizations:
1. PostgreSQL window functions for SMA, EMA, RSI, MACD, Bollinger calculations
2. Single query to calculate all indicators (no N+1 pattern)
3. Bulk insert using psycopg2.extras.execute_values
4. Memory-efficient batching for large datasets
5. Transaction management for data integrity

Supported indicators:
- SMA (Simple Moving Average): 5, 10, 20, 50, 200 periods
- EMA (Exponential Moving Average): 12, 26 periods
- RSI (Relative Strength Index): 14 periods
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (20-period, 2 standard deviations)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import time

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class IndicatorStats:
    """Statistics for indicator calculation operations"""
    stocks_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0

    @property
    def throughput_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.stocks_processed / self.elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stocks_processed': self.stocks_processed,
            'records_inserted': self.records_inserted,
            'records_updated': self.records_updated,
            'errors': self.errors,
            'elapsed_seconds': self.elapsed_seconds,
            'throughput_per_second': self.throughput_per_second
        }


class TechnicalIndicatorsCalculator:
    """
    High-performance technical indicators calculator using PostgreSQL window functions.

    This class eliminates the N+1 query pattern by:
    1. Calculating all indicators in a single SQL query per batch
    2. Using PostgreSQL window functions for rolling calculations
    3. Bulk inserting results using execute_values

    Example usage:
        calculator = TechnicalIndicatorsCalculator(conn_string)
        stats = calculator.calculate_for_all_stocks()
        print(f"Processed {stats.stocks_processed} stocks")
    """

    # Batch size for processing stocks (memory-efficient)
    DEFAULT_BATCH_SIZE = 500

    # Minimum data points required for calculations
    MIN_DATA_POINTS = 26  # Needed for EMA-26

    def __init__(
        self,
        connection_string: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lookback_days: int = 252  # ~1 year of trading days
    ):
        """
        Initialize the calculator.

        Args:
            connection_string: PostgreSQL connection string
            batch_size: Number of stocks to process per batch
            lookback_days: Number of days of price history to consider
        """
        self.connection_string = connection_string
        self.batch_size = batch_size
        self.lookback_days = lookback_days
        self.stats = IndicatorStats()

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Create a new database connection"""
        return psycopg2.connect(self.connection_string)

    def _get_stock_batches(self, conn: psycopg2.extensions.connection) -> List[List[int]]:
        """
        Get all active stock IDs and split into batches.

        Returns list of lists, where each inner list is a batch of stock IDs.
        """
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id
                FROM stocks
                WHERE is_active = true
                ORDER BY id
            """)
            all_stock_ids = [row[0] for row in cur.fetchall()]

        # Split into batches
        batches = []
        for i in range(0, len(all_stock_ids), self.batch_size):
            batches.append(all_stock_ids[i:i + self.batch_size])

        logger.info(f"Created {len(batches)} batches from {len(all_stock_ids)} stocks")
        return batches

    def _calculate_indicators_sql(self) -> str:
        """
        Generate SQL query that calculates all technical indicators using window functions.

        This query calculates:
        - SMA: 5, 10, 20, 50, 200 periods
        - EMA: 12, 26 periods (approximation using weighted average)
        - RSI: 14 periods
        - MACD: 12/26 periods with 9-period signal
        - Bollinger Bands: 20-period with 2 std devs

        The query uses CTEs for clarity and efficient execution.
        """
        return """
        WITH price_data AS (
            -- Get price data for specified stocks with date filter
            SELECT
                ph.stock_id,
                ph.date,
                ph.close,
                ph.high,
                ph.low,
                ph.volume,
                -- Calculate price change for RSI
                ph.close - LAG(ph.close) OVER (PARTITION BY ph.stock_id ORDER BY ph.date) as price_change
            FROM price_history ph
            WHERE ph.stock_id = ANY(%(stock_ids)s)
              AND ph.date >= %(start_date)s
            ORDER BY ph.stock_id, ph.date
        ),

        sma_calculations AS (
            -- Calculate SMAs using window functions
            SELECT
                stock_id,
                date,
                close,
                high,
                low,
                volume,
                price_change,
                -- SMA calculations
                AVG(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) as sma_5,
                AVG(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                ) as sma_10,
                AVG(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) as sma_20,
                AVG(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                ) as sma_50,
                AVG(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
                ) as sma_200,
                -- Standard deviation for Bollinger Bands
                STDDEV(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) as std_20,
                -- Count for ensuring we have enough data
                COUNT(*) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
                ) as data_count,
                -- Row number for filtering
                ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date DESC) as rn
            FROM price_data
        ),

        rsi_base AS (
            -- Calculate gains and losses for RSI
            SELECT
                stock_id,
                date,
                close,
                sma_5, sma_10, sma_20, sma_50, sma_200,
                std_20,
                data_count,
                rn,
                -- Average gains over 14 periods
                AVG(CASE WHEN price_change > 0 THEN price_change ELSE 0 END) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) as avg_gain,
                -- Average losses over 14 periods
                AVG(CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) as avg_loss
            FROM sma_calculations
        ),

        ema_approximation AS (
            -- EMA approximation using weighted moving average
            -- True EMA requires recursive calculation, we use an approximation
            SELECT
                r.*,
                -- RSI calculation
                CASE
                    WHEN avg_loss = 0 THEN 100
                    WHEN avg_gain = 0 THEN 0
                    ELSE 100 - (100 / (1 + (avg_gain / NULLIF(avg_loss, 0))))
                END as rsi_14,
                -- EMA-12 approximation (weighted avg with decay)
                -- Using the formula: EMA = SMA for simplicity in SQL
                -- For more accurate EMA, use the recursive calculation below
                AVG(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ) as ema_12_approx,
                AVG(close) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 25 PRECEDING AND CURRENT ROW
                ) as ema_26_approx
            FROM rsi_base r
        ),

        final_indicators AS (
            SELECT
                stock_id,
                date,
                close,
                -- SMAs
                ROUND(sma_5::numeric, 4) as sma_5,
                ROUND(sma_10::numeric, 4) as sma_10,
                ROUND(sma_20::numeric, 4) as sma_20,
                ROUND(sma_50::numeric, 4) as sma_50,
                ROUND(sma_200::numeric, 4) as sma_200,
                -- EMAs (approximation)
                ROUND(ema_12_approx::numeric, 4) as ema_12,
                ROUND(ema_26_approx::numeric, 4) as ema_26,
                -- RSI
                ROUND(rsi_14::numeric, 2) as rsi_14,
                -- MACD (EMA12 - EMA26)
                ROUND((ema_12_approx - ema_26_approx)::numeric, 4) as macd,
                -- MACD Signal would require another window, simplified here
                ROUND(AVG(ema_12_approx - ema_26_approx) OVER (
                    PARTITION BY stock_id ORDER BY date
                    ROWS BETWEEN 8 PRECEDING AND CURRENT ROW
                )::numeric, 4) as macd_signal,
                -- Bollinger Bands
                ROUND((sma_20 + 2 * std_20)::numeric, 4) as bollinger_upper,
                ROUND(sma_20::numeric, 4) as bollinger_middle,
                ROUND((sma_20 - 2 * std_20)::numeric, 4) as bollinger_lower,
                -- Data quality flag
                data_count >= 20 as has_sufficient_data
            FROM ema_approximation
            WHERE rn = 1  -- Only get the latest date for each stock
              AND data_count >= 20  -- Ensure we have enough data
        )

        SELECT * FROM final_indicators
        """

    def _calculate_indicators_for_batch(
        self,
        conn: psycopg2.extensions.connection,
        stock_ids: List[int],
        target_date: Optional[date] = None
    ) -> List[Tuple]:
        """
        Calculate indicators for a batch of stocks.

        Args:
            conn: Database connection
            stock_ids: List of stock IDs to process
            target_date: Target date for calculations (defaults to today)

        Returns:
            List of tuples containing indicator values
        """
        if target_date is None:
            target_date = date.today()

        start_date = target_date - timedelta(days=self.lookback_days)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                self._calculate_indicators_sql(),
                {
                    'stock_ids': stock_ids,
                    'start_date': start_date
                }
            )
            results = cur.fetchall()

        return results

    def _bulk_upsert_indicators(
        self,
        conn: psycopg2.extensions.connection,
        indicators: List[Dict[str, Any]],
        target_date: Optional[date] = None
    ) -> int:
        """
        Bulk upsert indicator values using execute_values.

        Args:
            conn: Database connection
            indicators: List of indicator dictionaries
            target_date: Target date for the indicators

        Returns:
            Number of records affected
        """
        if not indicators:
            return 0

        if target_date is None:
            target_date = date.today()

        # Prepare data for bulk insert
        values = []
        for ind in indicators:
            values.append((
                ind['stock_id'],
                target_date,
                ind.get('sma_5'),
                ind.get('sma_10'),
                ind.get('sma_20'),
                ind.get('sma_50'),
                ind.get('sma_200'),
                ind.get('ema_12'),
                ind.get('ema_26'),
                ind.get('rsi_14'),
                ind.get('macd'),
                ind.get('macd_signal'),
                ind.get('bollinger_upper'),
                ind.get('bollinger_middle'),
                ind.get('bollinger_lower'),
            ))

        # Use execute_values for bulk insert with ON CONFLICT
        insert_sql = """
            INSERT INTO technical_indicators (
                stock_id, date,
                sma_5, sma_10, sma_20, sma_50, sma_200,
                ema_12, ema_26,
                rsi_14,
                macd, macd_signal,
                bollinger_upper, bollinger_middle, bollinger_lower
            )
            VALUES %s
            ON CONFLICT (stock_id, date) DO UPDATE SET
                sma_5 = EXCLUDED.sma_5,
                sma_10 = EXCLUDED.sma_10,
                sma_20 = EXCLUDED.sma_20,
                sma_50 = EXCLUDED.sma_50,
                sma_200 = EXCLUDED.sma_200,
                ema_12 = EXCLUDED.ema_12,
                ema_26 = EXCLUDED.ema_26,
                rsi_14 = EXCLUDED.rsi_14,
                macd = EXCLUDED.macd,
                macd_signal = EXCLUDED.macd_signal,
                bollinger_upper = EXCLUDED.bollinger_upper,
                bollinger_middle = EXCLUDED.bollinger_middle,
                bollinger_lower = EXCLUDED.bollinger_lower
        """

        with conn.cursor() as cur:
            execute_values(
                cur,
                insert_sql,
                values,
                template=None,
                page_size=1000  # Batch size for execute_values
            )
            affected = cur.rowcount

        return affected

    def calculate_for_stocks(
        self,
        stock_ids: List[int],
        target_date: Optional[date] = None
    ) -> IndicatorStats:
        """
        Calculate indicators for a specific list of stocks.

        Args:
            stock_ids: List of stock IDs to process
            target_date: Target date for calculations

        Returns:
            IndicatorStats with processing results
        """
        start_time = time.time()
        self.stats = IndicatorStats()

        if not stock_ids:
            logger.warning("No stock IDs provided")
            return self.stats

        logger.info(f"Calculating indicators for {len(stock_ids)} stocks")

        conn = self._get_connection()
        try:
            # Process in batches for memory efficiency
            batches = [stock_ids[i:i + self.batch_size]
                      for i in range(0, len(stock_ids), self.batch_size)]

            for batch_num, batch_ids in enumerate(batches):
                try:
                    # Calculate indicators
                    indicators = self._calculate_indicators_for_batch(
                        conn, batch_ids, target_date
                    )

                    # Bulk upsert
                    affected = self._bulk_upsert_indicators(conn, indicators, target_date)

                    self.stats.stocks_processed += len(indicators)
                    self.stats.records_inserted += affected

                    logger.debug(
                        f"Batch {batch_num + 1}/{len(batches)}: "
                        f"{len(indicators)} stocks, {affected} records"
                    )

                except Exception as e:
                    logger.error(f"Batch {batch_num + 1} failed: {e}")
                    self.stats.errors += len(batch_ids)
                    conn.rollback()
                    continue

            conn.commit()

        finally:
            conn.close()

        self.stats.elapsed_seconds = time.time() - start_time
        logger.info(
            f"Indicator calculation complete: {self.stats.stocks_processed} stocks, "
            f"{self.stats.records_inserted} records in {self.stats.elapsed_seconds:.2f}s"
        )

        return self.stats

    def calculate_for_all_stocks(
        self,
        target_date: Optional[date] = None
    ) -> IndicatorStats:
        """
        Calculate indicators for ALL active stocks.

        This is the main entry point for daily indicator calculation.
        It fetches all active stock IDs and processes them in batches.

        Args:
            target_date: Target date for calculations (defaults to today)

        Returns:
            IndicatorStats with processing results
        """
        start_time = time.time()
        self.stats = IndicatorStats()

        logger.info("Starting indicator calculation for all active stocks")

        conn = self._get_connection()
        try:
            # Get all stock batches
            batches = self._get_stock_batches(conn)
            total_stocks = sum(len(b) for b in batches)

            logger.info(f"Processing {total_stocks} stocks in {len(batches)} batches")

            for batch_num, batch_ids in enumerate(batches):
                batch_start = time.time()

                try:
                    # Calculate indicators for batch
                    indicators = self._calculate_indicators_for_batch(
                        conn, batch_ids, target_date
                    )

                    # Bulk upsert results
                    affected = self._bulk_upsert_indicators(conn, indicators, target_date)

                    self.stats.stocks_processed += len(indicators)
                    self.stats.records_inserted += affected

                    batch_elapsed = time.time() - batch_start
                    logger.info(
                        f"Batch {batch_num + 1}/{len(batches)}: "
                        f"{len(indicators)}/{len(batch_ids)} stocks processed in {batch_elapsed:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"Batch {batch_num + 1} failed: {e}")
                    self.stats.errors += len(batch_ids)
                    conn.rollback()
                    # Continue with next batch
                    continue

            # Final commit
            conn.commit()

        finally:
            conn.close()

        self.stats.elapsed_seconds = time.time() - start_time

        logger.info(
            f"All indicators calculated: "
            f"{self.stats.stocks_processed}/{total_stocks} stocks, "
            f"{self.stats.records_inserted} records, "
            f"{self.stats.errors} errors in {self.stats.elapsed_seconds:.2f}s "
            f"({self.stats.throughput_per_second:.1f} stocks/sec)"
        )

        return self.stats


def calculate_indicators_optimized(
    connection_string: str,
    stock_ids: Optional[List[int]] = None,
    target_date: Optional[date] = None,
    batch_size: int = 500
) -> Dict[str, Any]:
    """
    Convenience function for calculating technical indicators.

    This function is designed to be called from Airflow DAGs.

    Args:
        connection_string: PostgreSQL connection string
        stock_ids: Optional list of specific stock IDs (processes all if None)
        target_date: Target date for calculations
        batch_size: Number of stocks per batch

    Returns:
        Dictionary with processing statistics
    """
    calculator = TechnicalIndicatorsCalculator(
        connection_string=connection_string,
        batch_size=batch_size
    )

    if stock_ids:
        stats = calculator.calculate_for_stocks(stock_ids, target_date)
    else:
        stats = calculator.calculate_for_all_stocks(target_date)

    return stats.to_dict()


# More accurate EMA calculation using recursive CTE (PostgreSQL 9.4+)
# This is kept as a reference for when higher accuracy is needed
EMA_RECURSIVE_SQL = """
WITH RECURSIVE ema_calc AS (
    -- Base case: first row uses close price as EMA
    SELECT
        stock_id,
        date,
        close,
        close as ema_12,
        close as ema_26,
        1 as row_num
    FROM (
        SELECT stock_id, date, close,
               ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date) as rn
        FROM price_history
        WHERE stock_id = ANY(%(stock_ids)s)
          AND date >= %(start_date)s
    ) base
    WHERE rn = 1

    UNION ALL

    -- Recursive case: calculate EMA using previous EMA
    SELECT
        p.stock_id,
        p.date,
        p.close,
        -- EMA formula: EMA = (close * k) + (previous_EMA * (1 - k))
        -- where k = 2 / (period + 1)
        (p.close * 0.1538) + (e.ema_12 * 0.8462) as ema_12,  -- k for 12-period
        (p.close * 0.0741) + (e.ema_26 * 0.9259) as ema_26,  -- k for 26-period
        e.row_num + 1
    FROM (
        SELECT stock_id, date, close,
               ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY date) as rn
        FROM price_history
        WHERE stock_id = ANY(%(stock_ids)s)
          AND date >= %(start_date)s
    ) p
    JOIN ema_calc e ON p.stock_id = e.stock_id AND p.rn = e.row_num + 1
)
SELECT * FROM ema_calc WHERE row_num = (SELECT MAX(row_num) FROM ema_calc);
"""


if __name__ == "__main__":
    # Test the calculator
    import os

    logging.basicConfig(level=logging.INFO)

    # Get connection string from environment
    conn_string = os.environ.get(
        'DATABASE_URL',
        'postgresql://postgres:password@localhost:5432/investment_db'
    )

    calculator = TechnicalIndicatorsCalculator(
        connection_string=conn_string,
        batch_size=100
    )

    stats = calculator.calculate_for_all_stocks()
    print(f"\nFinal Statistics:")
    print(f"  Stocks processed: {stats.stocks_processed}")
    print(f"  Records inserted: {stats.records_inserted}")
    print(f"  Errors: {stats.errors}")
    print(f"  Elapsed time: {stats.elapsed_seconds:.2f}s")
    print(f"  Throughput: {stats.throughput_per_second:.1f} stocks/sec")
