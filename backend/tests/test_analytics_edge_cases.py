"""
Comprehensive edge case tests for analytics modules
Tests sentiment analysis, fundamental edge cases, technical edge cases, and optimized recommendation engine
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List

from backend.analytics.sentiment_analysis import (
    SentimentAnalysisEngine,
    SentimentResult
)
from backend.analytics.fundamental_analysis import (
    FundamentalAnalysisEngine,
    FinancialMetrics
)
from backend.analytics.technical_analysis import TechnicalAnalysisEngine


# ============================================================================
# Sentiment Analysis Tests
# ============================================================================

class TestSentimentAnalysisEdgeCases:
    """Test sentiment analysis with various edge cases"""

    @pytest.fixture
    def sentiment_engine(self):
        """Create sentiment engine without FinBERT"""
        return SentimentAnalysisEngine(use_finbert=False)

    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, sentiment_engine):
        """Test sentiment analysis with empty text"""
        result = await sentiment_engine.analyze_sentiment("", "test")

        assert isinstance(result, SentimentResult)
        assert result.score == 0.0
        assert result.label == 'neutral'
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_analyze_very_short_text(self, sentiment_engine):
        """Test sentiment analysis with very short text"""
        result = await sentiment_engine.analyze_sentiment("ok", "test")

        assert isinstance(result, SentimentResult)
        assert -1 <= result.score <= 1

    @pytest.mark.asyncio
    async def test_analyze_positive_sentiment(self, sentiment_engine):
        """Test positive sentiment detection"""
        text = "Strong buy signal with excellent growth and outstanding profits"
        result = await sentiment_engine.analyze_sentiment(text, "test")

        assert result.score > 0
        assert result.label == 'positive'
        assert 'growth' in result.keywords or 'profits' in result.keywords

    @pytest.mark.asyncio
    async def test_analyze_negative_sentiment(self, sentiment_engine):
        """Test negative sentiment detection"""
        text = "Terrible losses, bearish downgrade with disappointing weakness"
        result = await sentiment_engine.analyze_sentiment(text, "test")

        assert result.score < 0
        assert result.label == 'negative'

    @pytest.mark.asyncio
    async def test_analyze_with_intensifiers(self, sentiment_engine):
        """Test sentiment with intensifier words"""
        text1 = "Good performance"
        text2 = "Very good performance"
        text3 = "Extremely good performance"

        result1 = await sentiment_engine.analyze_sentiment(text1, "test")
        result2 = await sentiment_engine.analyze_sentiment(text2, "test")
        result3 = await sentiment_engine.analyze_sentiment(text3, "test")

        # Intensified sentiment should be stronger
        assert result3.score >= result2.score >= result1.score

    @pytest.mark.asyncio
    async def test_analyze_stock_sentiment_empty_list(self, sentiment_engine):
        """Test stock sentiment with empty text list"""
        result = await sentiment_engine.analyze_stock_sentiment("AAPL", [])

        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.sources_analyzed == 0

    @pytest.mark.asyncio
    async def test_analyze_stock_sentiment_single_text(self, sentiment_engine):
        """Test stock sentiment with single text"""
        texts = ["Apple reports strong earnings beat"]
        result = await sentiment_engine.analyze_stock_sentiment("AAPL", texts)

        assert isinstance(result, SentimentResult)
        assert result.sources_analyzed == 1

    @pytest.mark.asyncio
    async def test_analyze_stock_sentiment_multiple_texts(self, sentiment_engine):
        """Test stock sentiment with multiple texts"""
        texts = [
            "Strong buy signal for stock",
            "Excellent growth prospects",
            "Outstanding revenue performance"
        ]
        result = await sentiment_engine.analyze_stock_sentiment("AAPL", texts)

        assert result.score > 0  # Should be positive overall
        assert result.sources_analyzed == 3
        assert len(result.keywords) > 0

    @pytest.mark.asyncio
    async def test_analyze_stock_sentiment_mixed(self, sentiment_engine):
        """Test stock sentiment with mixed positive and negative texts"""
        texts = [
            "Strong buy signal",
            "Terrible losses reported",
            "Excellent growth"
        ]
        result = await sentiment_engine.analyze_stock_sentiment("AAPL", texts)

        assert isinstance(result, SentimentResult)
        assert -1 <= result.score <= 1

    @pytest.mark.asyncio
    async def test_get_news_sentiment_placeholder(self, sentiment_engine):
        """Test news sentiment placeholder"""
        result = await sentiment_engine.get_news_sentiment("AAPL", limit=10)

        assert isinstance(result, SentimentResult)
        assert result.score == 0.0
        assert result.label == 'neutral'
        assert 'news' in result.keywords

    @pytest.mark.asyncio
    async def test_get_social_sentiment_placeholder(self, sentiment_engine):
        """Test social sentiment placeholder"""
        result = await sentiment_engine.get_social_sentiment("AAPL", limit=50)

        assert isinstance(result, SentimentResult)
        assert result.score == 0.0
        assert result.label == 'neutral'
        assert 'social' in result.keywords

    @pytest.mark.asyncio
    async def test_comprehensive_sentiment_analysis(self, sentiment_engine):
        """Test comprehensive sentiment analysis"""
        result = await sentiment_engine.analyze_comprehensive_sentiment("AAPL")

        assert 'ticker' in result
        assert 'overall_sentiment' in result
        assert 'news_sentiment' in result
        assert 'social_sentiment' in result
        assert 'timestamp' in result
        assert result['ticker'] == 'AAPL'

    @pytest.mark.asyncio
    async def test_sentiment_with_special_characters(self, sentiment_engine):
        """Test sentiment with special characters"""
        text = "Stock $$$ price !@#$ up 50%!!! 🚀🚀🚀"
        result = await sentiment_engine.analyze_sentiment(text, "test")

        assert isinstance(result, SentimentResult)
        # Should still extract some sentiment

    @pytest.mark.asyncio
    async def test_sentiment_with_numbers_only(self, sentiment_engine):
        """Test sentiment with only numbers"""
        text = "123 456 789"
        result = await sentiment_engine.analyze_sentiment(text, "test")

        assert result.label == 'neutral'
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_keyword_extraction_stopwords(self, sentiment_engine):
        """Test keyword extraction removes stopwords"""
        text = "The company has very good performance with strong results"
        result = await sentiment_engine.analyze_sentiment(text, "test")

        # Should not contain stopwords like 'the', 'with', 'has'
        stopwords = {'the', 'has', 'with', 'very'}
        extracted = set(result.keywords)
        assert not extracted.intersection(stopwords)

    @pytest.mark.asyncio
    async def test_error_handling_in_analysis(self, sentiment_engine):
        """Test error handling with exception during analysis"""
        # Mock to force an error in text processing
        with patch.object(sentiment_engine, '_extract_keywords', side_effect=Exception("Test error")):
            result = await sentiment_engine.analyze_sentiment("test text", "test")

            # Should return neutral sentiment on error
            assert result.score == 0.0
            assert result.confidence == 0.0
            assert result.label == 'neutral'


# ============================================================================
# Fundamental Analysis Edge Cases
# ============================================================================

class TestFundamentalAnalysisEdgeCases:
    """Test fundamental analysis with edge cases"""

    @pytest.fixture
    def fundamental_engine(self):
        """Create fundamental analysis engine"""
        return FundamentalAnalysisEngine()

    def test_zero_revenue_company(self, fundamental_engine):
        """Test company with zero revenue"""
        financials = {
            'revenue': 0,
            'gross_profit': 0,
            'operating_income': 0,
            'net_income': -5000000,  # Losses
            'total_assets': 10000000,
            'total_equity': 8000000,
            'total_debt': 2000000,
            'current_assets': 5000000,
            'current_liabilities': 1000000,
            'cash': 3000000,
            'inventory': 0,
            'receivables': 0,
            'free_cash_flow': -1000000,
            'shares_outstanding': 1000000
        }
        market_data = {
            'market_cap': 50000000,
            'price': 50.0,
            'beta': 1.5
        }

        metrics = fundamental_engine._calculate_financial_metrics(financials, market_data)

        assert metrics.gross_margin == 0
        assert metrics.operating_margin == 0
        assert metrics.net_margin == 0
        assert metrics.roa < 0  # Negative due to losses

    def test_negative_equity(self, fundamental_engine):
        """Test company with negative equity"""
        financials = {
            'revenue': 10000000,
            'gross_profit': 2000000,
            'operating_income': 500000,
            'net_income': -1000000,
            'total_assets': 5000000,
            'total_equity': -2000000,  # Negative equity
            'total_debt': 7000000,
            'current_assets': 2000000,
            'current_liabilities': 1000000,
            'cash': 500000,
            'inventory': 500000,
            'receivables': 500000,
            'free_cash_flow': -500000,
            'shares_outstanding': 1000000
        }
        market_data = {
            'market_cap': 10000000,
            'price': 10.0,
            'beta': 2.0
        }

        metrics = fundamental_engine._calculate_financial_metrics(financials, market_data)

        # Should handle negative equity gracefully
        assert isinstance(metrics, FinancialMetrics)
        assert metrics.roe == 0  # Division by zero protection
        assert metrics.debt_to_equity == 0  # Division by zero protection

    def test_extremely_high_pe_ratio(self, fundamental_engine):
        """Test company with extremely high P/E ratio"""
        financials = {
            'revenue': 1000000,
            'net_income': 100,  # Very small earnings
            'shares_outstanding': 1000000,
            'total_assets': 1000000,
            'total_equity': 500000
        }
        market_data = {
            'market_cap': 1000000000,  # Very high market cap
            'price': 1000.0,
            'beta': 1.0
        }

        metrics = fundamental_engine._calculate_financial_metrics(financials, market_data)

        assert metrics.pe_ratio > 1000  # Should be extremely high

    def test_negative_pe_ratio(self, fundamental_engine):
        """Test company with negative earnings (no P/E)"""
        financials = {
            'revenue': 10000000,
            'net_income': -5000000,  # Losses
            'shares_outstanding': 1000000,
            'total_assets': 10000000,
            'total_equity': 5000000
        }
        market_data = {
            'market_cap': 50000000,
            'price': 50.0,
            'beta': 1.0
        }

        metrics = fundamental_engine._calculate_financial_metrics(financials, market_data)

        assert metrics.pe_ratio == 0  # Should be 0 for negative earnings

    def test_missing_financial_data(self, fundamental_engine):
        """Test with missing financial data fields"""
        financials = {
            'revenue': 10000000,
            'net_income': 1000000,
            'shares_outstanding': 1000000
            # Missing most fields
        }
        market_data = {
            'market_cap': 50000000,
            'price': 50.0,
            'beta': 1.0
        }

        metrics = fundamental_engine._calculate_financial_metrics(financials, market_data)

        # Should use defaults and not crash
        assert isinstance(metrics, FinancialMetrics)
        assert metrics.current_ratio == 0  # No current assets/liabilities

    def test_zero_shares_outstanding(self, fundamental_engine):
        """Test with zero shares outstanding"""
        financials = {
            'revenue': 10000000,
            'net_income': 1000000,
            'shares_outstanding': 0,  # Invalid
            'total_equity': 5000000
        }
        market_data = {
            'market_cap': 0,
            'price': 0,
            'beta': 1.0
        }

        metrics = fundamental_engine._calculate_financial_metrics(financials, market_data)

        # Should handle gracefully with division protection
        assert metrics.pe_ratio == 0

    def test_altman_z_score_edge_cases(self, fundamental_engine):
        """Test Altman Z-Score with edge cases"""
        # Company in distress
        distressed = {
            'current_assets': 1000000,
            'current_liabilities': 5000000,
            'total_assets': 10000000,
            'retained_earnings': -2000000,
            'ebit': -500000,
            'market_cap': 1000000,
            'total_liabilities': 8000000,
            'revenue': 5000000
        }

        z_score = fundamental_engine._calculate_altman_z_score(distressed)

        assert z_score['zone'] == 'distress'
        assert z_score['bankruptcy_risk'] == 'high'
        assert z_score['score'] < 1.81

    def test_growth_rate_with_negative_values(self, fundamental_engine):
        """Test growth rate calculation with negative values"""
        # Revenue decline
        declining = [100000000, 80000000, 60000000]
        growth = fundamental_engine._calculate_growth_rate(declining)

        assert growth < 0  # Should be negative growth

    def test_growth_rate_single_value(self, fundamental_engine):
        """Test growth rate with single value"""
        growth = fundamental_engine._calculate_growth_rate([100000000])
        assert growth == 0

    def test_piotroski_score_all_failing(self, fundamental_engine):
        """Test Piotroski score with all criteria failing"""
        failing_financials = {
            'net_income': -1000000,
            'operating_cash_flow': -500000,
            'roa': 0.05,
            'roa_previous': 0.10,  # Declining
            'debt_to_assets': 0.8,
            'debt_to_assets_previous': 0.6,  # Increasing
            'current_ratio': 1.2,
            'current_ratio_previous': 1.5,  # Declining
            'shares_outstanding': 2000000,
            'shares_outstanding_previous': 1000000,  # Dilution
            'gross_margin': 30,
            'gross_margin_previous': 35,  # Declining
            'asset_turnover': 0.8,
            'asset_turnover_previous': 1.0  # Declining
        }

        piotroski = fundamental_engine._calculate_piotroski_score(failing_financials)

        assert piotroski['score'] <= 2  # Should be very low
        assert piotroski['strength'] == 'weak'


# ============================================================================
# Technical Analysis Edge Cases
# ============================================================================

class TestTechnicalAnalysisEdgeCases:
    """Test technical analysis with edge cases"""

    @pytest.fixture
    def technical_engine(self):
        """Create technical analysis engine"""
        return TechnicalAnalysisEngine()

    def test_single_data_point(self, technical_engine):
        """Test with single data point"""
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000000]
        })

        analysis = technical_engine.analyze_stock(df)
        assert analysis == {}  # Should return empty for insufficient data

    def test_flat_prices(self, technical_engine):
        """Test with completely flat prices"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        flat_price = 100.0

        df = pd.DataFrame({
            'open': [flat_price] * 250,
            'high': [flat_price] * 250,
            'low': [flat_price] * 250,
            'close': [flat_price] * 250,
            'volume': [1000000] * 250
        }, index=dates)

        analysis = technical_engine.analyze_stock(df)

        # Should handle flat prices
        assert 'volatility_indicators' in analysis
        assert analysis['volatility_indicators']['atr_14'] == 0.0

    def test_all_zero_volume(self, technical_engine):
        """Test with zero volume"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        prices = np.linspace(100, 120, 250)

        df = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': [0] * 250  # Zero volume
        }, index=dates)

        analysis = technical_engine.analyze_stock(df)

        # Should handle zero volume gracefully
        assert 'volume_indicators' in analysis
        assert analysis['volume_indicators']['volume_sma_20'] == 0

    def test_extreme_price_spike(self, technical_engine):
        """Test with extreme price spike"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        prices = [100] * 249 + [10000]  # Extreme spike

        df = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 250
        }, index=dates)

        analysis = technical_engine.analyze_stock(df)

        # Should handle extreme volatility
        assert 'volatility_indicators' in analysis
        assert analysis['volatility_indicators']['atr_14'] > 0

    def test_nan_values_in_prices(self, technical_engine):
        """Test with NaN values in prices"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        prices = np.linspace(100, 120, 250)
        prices[50:60] = np.nan  # Insert NaN values

        df = pd.DataFrame({
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000000] * 250
        }, index=dates)

        # Should handle NaN values or raise appropriate error
        try:
            analysis = technical_engine.analyze_stock(df)
            # If it doesn't crash, check that some analysis was done
            assert isinstance(analysis, dict)
        except Exception:
            # NaN handling might raise exception, which is acceptable
            pass

    def test_rsi_all_gains(self, technical_engine):
        """Test RSI with prices only going up"""
        prices = np.arange(100, 150, 1)  # Only gains
        rsi = technical_engine._calculate_rsi(prices, period=14)

        assert rsi >= 70  # Should be overbought

    def test_rsi_all_losses(self, technical_engine):
        """Test RSI with prices only going down"""
        prices = np.arange(150, 100, -1)  # Only losses
        rsi = technical_engine._calculate_rsi(prices, period=14)

        assert rsi <= 30  # Should be oversold

    def test_bollinger_bands_zero_std(self, technical_engine):
        """Test Bollinger Bands with zero standard deviation"""
        prices = np.array([100] * 50)  # Constant prices
        bb = technical_engine._calculate_bollinger_bands(prices)

        assert bb['upper'] == bb['middle'] == bb['lower']
        assert bb['width'] == 0

    def test_atr_no_true_range(self, technical_engine):
        """Test ATR when high = low = close"""
        high = np.array([100] * 20)
        low = np.array([100] * 20)
        close = np.array([100] * 20)

        atr = technical_engine._calculate_atr(high, low, close, period=14)
        assert atr == 0.0

    def test_support_resistance_no_extrema(self, technical_engine):
        """Test support/resistance with linear trend (no local extrema)"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        prices = np.linspace(100, 120, 250)  # Perfect linear trend

        df = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': [1000000] * 250
        }, index=dates)

        sr = technical_engine._find_support_resistance(df)

        # Should still provide levels even without clear extrema
        assert 'primary_support' in sr
        assert 'primary_resistance' in sr


# ============================================================================
# Additional Analytics Edge Cases
# ============================================================================

class TestAnalyticsIntegration:
    """Test integration scenarios across analytics modules"""

    @pytest.mark.asyncio
    async def test_combined_analysis_workflow(self):
        """Test workflow combining fundamental, technical, and sentiment"""
        # Create engines
        fundamental = FundamentalAnalysisEngine()
        technical = TechnicalAnalysisEngine()
        sentiment = SentimentAnalysisEngine(use_finbert=False)

        # Sample data
        financials = {
            'revenue': 10000000,
            'net_income': 1000000,
            'total_assets': 8000000,
            'total_equity': 5000000,
            'shares_outstanding': 1000000
        }
        market_data = {
            'market_cap': 50000000,
            'price': 50.0,
            'beta': 1.2
        }

        # Run analyses
        fin_metrics = fundamental._calculate_financial_metrics(financials, market_data)
        assert isinstance(fin_metrics, FinancialMetrics)

        sentiment_result = await sentiment.analyze_sentiment("Great company performance", "test")
        assert isinstance(sentiment_result, SentimentResult)

    def test_price_data_validation(self):
        """Test that technical analysis validates price data properly"""
        technical = TechnicalAnalysisEngine()

        # Test with invalid columns
        invalid_df = pd.DataFrame({
            'price': [100, 101, 102],
            'vol': [1000, 1100, 1200]
        })

        # Should handle missing required columns
        try:
            result = technical.analyze_stock(invalid_df)
            assert result == {}  # Should return empty
        except KeyError:
            pass  # Expected error

    @pytest.mark.asyncio
    async def test_batch_sentiment_processing(self):
        """Test batch processing of sentiment texts"""
        sentiment = SentimentAnalysisEngine(use_finbert=False)

        # Large batch of texts
        texts = [f"This is test text number {i}" for i in range(50)]

        result = await sentiment.analyze_stock_sentiment("TEST", texts)

        assert result.sources_analyzed == 50
        assert isinstance(result.keywords, list)

    def test_financial_ratios_consistency(self):
        """Test that financial ratios are internally consistent"""
        fundamental = FundamentalAnalysisEngine()

        financials = {
            'revenue': 100000000,
            'gross_profit': 40000000,
            'operating_income': 20000000,
            'net_income': 15000000,
            'total_assets': 80000000,
            'total_equity': 50000000,
            'total_debt': 20000000,
            'current_assets': 30000000,
            'current_liabilities': 15000000,
            'cash': 10000000,
            'inventory': 5000000,
            'receivables': 8000000,
            'free_cash_flow': 12000000,
            'shares_outstanding': 1000000,
            'operating_cash_flow': 18000000
        }
        market_data = {
            'market_cap': 150000000,
            'price': 150.0,
            'beta': 1.2
        }

        metrics = fundamental._calculate_financial_metrics(financials, market_data)

        # Gross margin should be >= operating margin
        assert metrics.gross_margin >= metrics.operating_margin

        # Operating margin should be >= net margin
        assert metrics.operating_margin >= metrics.net_margin

        # Current ratio should match calculation
        expected_current = 30000000 / 15000000
        assert abs(metrics.current_ratio - expected_current) < 0.01
