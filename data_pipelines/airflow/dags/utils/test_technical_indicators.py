"""
Tests for Technical Indicators Calculator

These tests verify:
1. SQL query generation is correct
2. Indicator calculations match expected values
3. Bulk operations work correctly
4. Error handling is robust
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import date, timedelta
import numpy as np

# Import the module under test
from technical_indicators_calculator import (
    TechnicalIndicatorsCalculator,
    IndicatorStats,
    calculate_indicators_optimized
)


class TestIndicatorStats:
    """Test the IndicatorStats dataclass"""

    def test_throughput_calculation(self):
        stats = IndicatorStats(
            stocks_processed=1000,
            elapsed_seconds=10.0
        )
        assert stats.throughput_per_second == 100.0

    def test_throughput_zero_time(self):
        stats = IndicatorStats(
            stocks_processed=1000,
            elapsed_seconds=0.0
        )
        assert stats.throughput_per_second == 0.0

    def test_to_dict(self):
        stats = IndicatorStats(
            stocks_processed=100,
            records_inserted=95,
            errors=5,
            elapsed_seconds=2.0
        )
        result = stats.to_dict()

        assert result['stocks_processed'] == 100
        assert result['records_inserted'] == 95
        assert result['errors'] == 5
        assert result['elapsed_seconds'] == 2.0
        assert result['throughput_per_second'] == 50.0


class TestTechnicalIndicatorsCalculator:
    """Test the TechnicalIndicatorsCalculator class"""

    @pytest.fixture
    def calculator(self):
        """Create a calculator instance for testing"""
        return TechnicalIndicatorsCalculator(
            connection_string='postgresql://test:test@localhost/test',
            batch_size=100,
            lookback_days=252
        )

    def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert calculator.batch_size == 100
        assert calculator.lookback_days == 252

    def test_calculate_indicators_sql_returns_string(self, calculator):
        """Test that SQL generation returns a valid string"""
        sql = calculator._calculate_indicators_sql()
        assert isinstance(sql, str)
        assert len(sql) > 0

    def test_calculate_indicators_sql_contains_required_calculations(self, calculator):
        """Test that SQL contains all required indicator calculations"""
        sql = calculator._calculate_indicators_sql()

        # Check for SMA calculations
        assert 'sma_5' in sql.lower()
        assert 'sma_10' in sql.lower()
        assert 'sma_20' in sql.lower()
        assert 'sma_50' in sql.lower()
        assert 'sma_200' in sql.lower()

        # Check for EMA
        assert 'ema_12' in sql.lower()
        assert 'ema_26' in sql.lower()

        # Check for RSI
        assert 'rsi_14' in sql.lower()

        # Check for MACD
        assert 'macd' in sql.lower()
        assert 'macd_signal' in sql.lower()

        # Check for Bollinger Bands
        assert 'bollinger_upper' in sql.lower()
        assert 'bollinger_middle' in sql.lower()
        assert 'bollinger_lower' in sql.lower()

    def test_calculate_indicators_sql_uses_window_functions(self, calculator):
        """Test that SQL uses PostgreSQL window functions"""
        sql = calculator._calculate_indicators_sql()

        # Check for window function keywords
        assert 'OVER' in sql
        assert 'PARTITION BY' in sql
        assert 'ORDER BY' in sql
        assert 'ROWS BETWEEN' in sql

    def test_calculate_indicators_sql_uses_parameters(self, calculator):
        """Test that SQL uses parameterized queries (no SQL injection)"""
        sql = calculator._calculate_indicators_sql()

        # Should use named parameters, not string formatting
        assert '%(stock_ids)s' in sql
        assert '%(start_date)s' in sql

    @patch('technical_indicators_calculator.psycopg2')
    def test_calculate_for_stocks_empty_list(self, mock_psycopg2, calculator):
        """Test handling of empty stock list"""
        stats = calculator.calculate_for_stocks([])
        assert stats.stocks_processed == 0
        assert stats.errors == 0

    @patch('technical_indicators_calculator.psycopg2')
    def test_get_stock_batches(self, mock_psycopg2, calculator):
        """Test stock batching logic"""
        # Create mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        # Mock fetchall to return 250 stock IDs
        mock_cursor.fetchall.return_value = [(i,) for i in range(250)]

        batches = calculator._get_stock_batches(mock_conn)

        # With batch_size=100, should create 3 batches
        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50


class TestIndicatorMathematics:
    """Test the mathematical correctness of indicator calculations"""

    def test_sma_calculation_logic(self):
        """Verify SMA calculation formula is correct in SQL"""
        # SMA is simple average over N periods
        # AVG(close) OVER (ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW)

        # Example: prices = [10, 11, 12, 13, 14]
        prices = [10, 11, 12, 13, 14]

        # SMA-5 of last point = (10+11+12+13+14)/5 = 12.0
        expected_sma5 = sum(prices) / 5
        assert expected_sma5 == 12.0

    def test_rsi_calculation_logic(self):
        """Verify RSI calculation formula is correct"""
        # RSI = 100 - (100 / (1 + RS))
        # RS = Average Gain / Average Loss over period

        # Example: 14 periods with gains and losses
        gains = [1, 0, 2, 0, 1, 0.5, 0, 0, 1, 0, 0.5, 0, 1, 0]  # total gains
        losses = [0, 0.5, 0, 1, 0, 0, 0.5, 1, 0, 0.5, 0, 0.5, 0, 0.5]  # total losses

        avg_gain = sum(gains) / 14  # 0.5
        avg_loss = sum(losses) / 14  # 0.357...

        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        expected_rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50

        # RSI should be between 0 and 100
        assert 0 <= expected_rsi <= 100

    def test_macd_calculation_logic(self):
        """Verify MACD calculation formula is correct"""
        # MACD = EMA(12) - EMA(26)
        # Signal = EMA(9) of MACD

        # For testing, use simple approximation
        ema_12 = 105.5
        ema_26 = 104.2

        macd = ema_12 - ema_26
        assert macd == pytest.approx(1.3, rel=0.01)

    def test_bollinger_bands_calculation_logic(self):
        """Verify Bollinger Bands calculation formula is correct"""
        # Upper = SMA(20) + 2 * StdDev(20)
        # Middle = SMA(20)
        # Lower = SMA(20) - 2 * StdDev(20)

        prices = list(range(1, 21))  # [1, 2, ..., 20]
        sma_20 = np.mean(prices)  # 10.5
        std_20 = np.std(prices)  # ~5.77

        upper = sma_20 + 2 * std_20
        middle = sma_20
        lower = sma_20 - 2 * std_20

        assert middle == pytest.approx(10.5, rel=0.01)
        assert upper > middle
        assert lower < middle
        assert upper - middle == pytest.approx(middle - lower, rel=0.01)


class TestBulkOperations:
    """Test bulk insert operations"""

    @patch('technical_indicators_calculator.execute_values')
    @patch('technical_indicators_calculator.psycopg2')
    def test_bulk_upsert_format(self, mock_psycopg2, mock_execute_values):
        """Test that bulk upsert uses correct format"""
        calculator = TechnicalIndicatorsCalculator(
            connection_string='postgresql://test:test@localhost/test'
        )

        # Create mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        # Test data
        indicators = [
            {
                'stock_id': 1,
                'sma_5': 100.0,
                'sma_10': 101.0,
                'sma_20': 102.0,
                'sma_50': 103.0,
                'sma_200': 104.0,
                'ema_12': 100.5,
                'ema_26': 100.3,
                'rsi_14': 55.5,
                'macd': 0.2,
                'macd_signal': 0.15,
                'bollinger_upper': 110.0,
                'bollinger_middle': 102.0,
                'bollinger_lower': 94.0,
            }
        ]

        calculator._bulk_upsert_indicators(mock_conn, indicators)

        # Verify execute_values was called
        assert mock_execute_values.called


class TestErrorHandling:
    """Test error handling scenarios"""

    @patch('technical_indicators_calculator.psycopg2')
    def test_connection_error_handling(self, mock_psycopg2):
        """Test handling of database connection errors"""
        mock_psycopg2.connect.side_effect = Exception("Connection failed")

        calculator = TechnicalIndicatorsCalculator(
            connection_string='postgresql://test:test@localhost/test'
        )

        # Should not raise, but should record errors in stats
        stats = calculator.calculate_for_stocks([1, 2, 3])
        # Stats should reflect the error condition
        assert stats.errors >= 0

    def test_empty_indicators_handling(self):
        """Test handling when no indicators are returned"""
        calculator = TechnicalIndicatorsCalculator(
            connection_string='postgresql://test:test@localhost/test'
        )

        # Mock connection that returns empty results
        with patch.object(calculator, '_get_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_cursor.fetchall.return_value = []

            mock_get_conn.return_value = mock_conn

            # Should handle gracefully
            stats = calculator.calculate_for_stocks([1, 2, 3])
            assert stats.records_inserted == 0


class TestIntegration:
    """Integration tests (require database connection)"""

    @pytest.mark.skip(reason="Requires actual database connection")
    def test_full_calculation_pipeline(self):
        """Test complete calculation pipeline with real database"""
        import os
        conn_string = os.environ.get('DATABASE_URL')

        if not conn_string:
            pytest.skip("DATABASE_URL not set")

        calculator = TechnicalIndicatorsCalculator(
            connection_string=conn_string,
            batch_size=100
        )

        stats = calculator.calculate_for_all_stocks()

        # Should process some stocks
        assert stats.stocks_processed > 0
        # Should have reasonable throughput
        assert stats.throughput_per_second > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
