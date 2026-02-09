"""
Comprehensive test coverage for analytics modules
Tests for fundamental_analysis.py, technical_analysis.py, and recommendation_engine.py
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock

from backend.analytics.fundamental_analysis import (
    FundamentalAnalysisEngine,
    FinancialMetrics
)
from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.recommendation_engine import (
    RecommendationEngine,
    RecommendationAction
)


# ============================================================================
# Fundamental Analysis Tests
# ============================================================================

class TestFundamentalAnalysis:
    """Test fundamental analysis calculations"""

    @pytest.fixture
    def fundamental_engine(self):
        """Create fundamental analysis engine"""
        return FundamentalAnalysisEngine()

    @pytest.fixture
    def sample_financials(self):
        """Sample financial data"""
        return {
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
            'operating_cash_flow': 18000000,
            'revenue_history': [80000000, 90000000, 100000000],
            'earnings_history': [10000000, 12000000, 15000000],
            'fcf_history': [8000000, 10000000, 12000000]
        }

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data"""
        return {
            'market_cap': 150000000,
            'price': 150.0,
            'beta': 1.2,
            'enterprise_value': 160000000
        }

    def test_calculate_financial_metrics(self, fundamental_engine, sample_financials, sample_market_data):
        """Test financial metrics calculation"""
        metrics = fundamental_engine._calculate_financial_metrics(
            sample_financials,
            sample_market_data
        )

        assert isinstance(metrics, FinancialMetrics)
        assert metrics.gross_margin == 40.0  # (40M / 100M) * 100
        assert metrics.operating_margin == 20.0  # (20M / 100M) * 100
        assert metrics.net_margin == 15.0  # (15M / 100M) * 100
        assert metrics.roe > 0
        assert metrics.roa > 0
        assert metrics.current_ratio == 2.0  # 30M / 15M

    def test_dcf_valuation(self, fundamental_engine, sample_financials, sample_market_data):
        """Test DCF valuation calculation"""
        dcf = fundamental_engine._calculate_dcf(sample_financials, sample_market_data)

        assert 'value' in dcf
        assert 'enterprise_value' in dcf
        assert 'wacc' in dcf
        assert dcf['value'] > 0
        assert dcf['confidence'] == 0.8

    def test_wacc_calculation(self, fundamental_engine, sample_financials, sample_market_data):
        """Test WACC calculation"""
        wacc = fundamental_engine._calculate_wacc(sample_financials, sample_market_data)

        assert 0 < wacc < 1  # Should be between 0 and 100%
        assert isinstance(wacc, float)

    def test_wacc_edge_case_zero_values(self, fundamental_engine):
        """Test WACC with zero market cap"""
        financials = {'total_debt': 0, 'interest_expense': 0}
        market_data = {'market_cap': 0, 'beta': 1.0}

        wacc = fundamental_engine._calculate_wacc(financials, market_data)
        assert wacc == 0.10  # Should return default 10%

    def test_quality_score_calculation(self, fundamental_engine, sample_financials):
        """Test quality score calculation"""
        quality = fundamental_engine._calculate_quality_score(sample_financials)

        assert 'overall_score' in quality
        assert 'scores' in quality
        assert 'grade' in quality
        assert 0 <= quality['overall_score'] <= 100
        assert quality['grade'] in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']

    def test_profitability_scoring(self, fundamental_engine, sample_financials):
        """Test profitability scoring"""
        score = fundamental_engine._score_profitability(sample_financials)

        assert 0 <= score <= 100
        assert isinstance(score, (int, float))

    def test_altman_z_score(self, fundamental_engine, sample_financials):
        """Test Altman Z-Score calculation"""
        sample_financials.update({
            'retained_earnings': 30000000,
            'ebit': 20000000,
            'market_cap': 150000000,
            'total_liabilities': 30000000
        })

        z_score = fundamental_engine._calculate_altman_z_score(sample_financials)

        assert 'score' in z_score
        assert 'zone' in z_score
        assert 'bankruptcy_risk' in z_score
        assert z_score['zone'] in ['safe', 'grey', 'distress']
        assert z_score['bankruptcy_risk'] in ['low', 'medium', 'high']

    def test_piotroski_score(self, fundamental_engine, sample_financials):
        """Test Piotroski F-Score calculation"""
        piotroski = fundamental_engine._calculate_piotroski_score(sample_financials)

        assert 'score' in piotroski
        assert 'criteria' in piotroski
        assert 'strength' in piotroski
        assert 0 <= piotroski['score'] <= 9
        assert piotroski['strength'] in ['strong', 'moderate', 'weak']

    def test_quality_grade_conversion(self, fundamental_engine):
        """Test quality grade conversion"""
        assert fundamental_engine._get_quality_grade(95) == 'A+'
        assert fundamental_engine._get_quality_grade(85) == 'A'
        assert fundamental_engine._get_quality_grade(75) == 'B+'
        assert fundamental_engine._get_quality_grade(65) == 'B-'
        assert fundamental_engine._get_quality_grade(45) == 'D'

    def test_growth_rate_calculation(self, fundamental_engine):
        """Test CAGR calculation"""
        values = [100, 110, 121]  # 10% annual growth
        cagr = fundamental_engine._calculate_growth_rate(values)

        assert 9.5 <= cagr <= 10.5  # Should be close to 10%

    def test_growth_rate_edge_cases(self, fundamental_engine):
        """Test growth rate with edge cases"""
        # Empty list
        assert fundamental_engine._calculate_growth_rate([]) == 0

        # Single value
        assert fundamental_engine._calculate_growth_rate([100]) == 0

        # Zero starting value
        assert fundamental_engine._calculate_growth_rate([0, 100]) == 0

        # Negative growth
        values = [100, 90, 81]
        cagr = fundamental_engine._calculate_growth_rate(values)
        assert cagr < 0

    def test_financial_metrics_zero_denominator(self, fundamental_engine):
        """Test financial metrics with zero denominators"""
        financials = {
            'revenue': 0,
            'gross_profit': 0,
            'operating_income': 0,
            'net_income': 0,
            'total_assets': 0,
            'total_equity': 0,
            'total_debt': 0,
            'current_assets': 0,
            'current_liabilities': 0,
            'cash': 0,
            'inventory': 0,
            'receivables': 0,
            'free_cash_flow': 0,
            'shares_outstanding': 1
        }
        market_data = {
            'market_cap': 0,
            'price': 0,
            'beta': 1.0
        }

        metrics = fundamental_engine._calculate_financial_metrics(financials, market_data)

        # Should not raise errors, should return zeros or defaults
        assert metrics.gross_margin == 0
        assert metrics.operating_margin == 0
        assert metrics.net_margin == 0


# ============================================================================
# Technical Analysis Tests
# ============================================================================

class TestTechnicalAnalysis:
    """Test technical analysis calculations"""

    @pytest.fixture
    def technical_engine(self):
        """Create technical analysis engine"""
        return TechnicalAnalysisEngine()

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data with 250 days"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        np.random.seed(42)  # For reproducibility

        # Generate trending price data
        base_price = 100
        trend = np.linspace(0, 20, 250)
        noise = np.random.normal(0, 2, 250)
        prices = base_price + trend + noise

        return pd.DataFrame({
            'open': prices + np.random.uniform(-1, 1, 250),
            'high': prices + np.random.uniform(0, 2, 250),
            'low': prices - np.random.uniform(0, 2, 250),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 250)
        }, index=dates)

    def test_analyze_stock_basic(self, technical_engine, sample_price_data):
        """Test basic stock analysis"""
        analysis = technical_engine.analyze_stock(sample_price_data)

        assert 'trend_indicators' in analysis
        assert 'momentum_indicators' in analysis
        assert 'volatility_indicators' in analysis
        assert 'volume_indicators' in analysis
        assert 'composite_score' in analysis
        assert -1 <= analysis['composite_score'] <= 1

    def test_insufficient_data_handling(self, technical_engine):
        """Test handling of insufficient data"""
        # Only 50 days of data (need 200)
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        small_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(95, 105, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)

        analysis = technical_engine.analyze_stock(small_df)
        assert analysis == {}  # Should return empty dict

    def test_rsi_calculation(self, technical_engine):
        """Test RSI calculation"""
        # Create data with clear uptrend for testing
        prices = np.array([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116])

        rsi = technical_engine._calculate_rsi(prices, period=14)

        assert 0 <= rsi <= 100
        assert isinstance(rsi, float)

    def test_rsi_edge_cases(self, technical_engine):
        """Test RSI with edge cases"""
        # Insufficient data
        short_prices = np.array([100, 101, 102])
        rsi = technical_engine._calculate_rsi(short_prices, period=14)
        assert rsi == 50.0  # Should return neutral

        # All prices going up (should be near 100)
        up_prices = np.array(range(100, 120))
        rsi = technical_engine._calculate_rsi(up_prices, period=14)
        assert rsi > 70  # Should be overbought

        # All prices going down (should be near 0)
        down_prices = np.array(range(120, 100, -1))
        rsi = technical_engine._calculate_rsi(down_prices, period=14)
        assert rsi < 30  # Should be oversold

    def test_macd_calculation(self, technical_engine):
        """Test MACD calculation"""
        prices = np.linspace(100, 120, 50)  # Uptrend

        macd = technical_engine._calculate_macd(prices)

        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
        assert isinstance(macd['macd'], float)

    def test_bollinger_bands(self, technical_engine):
        """Test Bollinger Bands calculation"""
        prices = np.random.normal(100, 5, 50)

        bb = technical_engine._calculate_bollinger_bands(prices, period=20, std_dev=2)

        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert 'width' in bb
        assert bb['upper'] > bb['middle'] > bb['lower']
        assert 0 <= bb['percent'] <= 1

    def test_bollinger_bands_edge_case(self, technical_engine):
        """Test Bollinger Bands with insufficient data"""
        prices = np.array([100, 101, 102])

        bb = technical_engine._calculate_bollinger_bands(prices, period=20)

        # Should return reasonable defaults
        assert bb['middle'] > 0
        assert bb['upper'] > bb['lower']

    def test_atr_calculation(self, technical_engine):
        """Test ATR calculation"""
        high = np.array([105, 106, 107, 108, 109])
        low = np.array([95, 96, 97, 98, 99])
        close = np.array([100, 101, 102, 103, 104])

        atr = technical_engine._calculate_atr(high, low, close, period=14)

        assert atr > 0
        assert isinstance(atr, float)

    def test_atr_edge_case(self, technical_engine):
        """Test ATR with minimal data"""
        high = np.array([105])
        low = np.array([95])
        close = np.array([100])

        atr = technical_engine._calculate_atr(high, low, close, period=14)
        assert atr == 0.0  # Should return 0 with insufficient data

    def test_support_resistance_detection(self, technical_engine, sample_price_data):
        """Test support and resistance level detection"""
        sr = technical_engine._find_support_resistance(sample_price_data)

        assert 'primary_support' in sr
        assert 'primary_resistance' in sr
        assert 'support_levels' in sr
        assert 'resistance_levels' in sr
        assert sr['primary_resistance'] > sr['primary_support']

    def test_moving_average_calculations(self, technical_engine):
        """Test EMA calculation"""
        values = np.array([100, 102, 104, 106, 108, 110, 112])

        ema = technical_engine._calculate_ema(values, period=5)

        assert ema > 0
        assert 100 < ema < 112  # Should be within range

    def test_ema_edge_case(self, technical_engine):
        """Test EMA with insufficient data"""
        values = np.array([100, 101])

        ema = technical_engine._calculate_ema(values, period=10)
        assert ema > 0  # Should return mean

    def test_composite_score_calculation(self, technical_engine, sample_price_data):
        """Test composite technical score calculation"""
        analysis = technical_engine.analyze_stock(sample_price_data)

        score = technical_engine._calculate_composite_score(analysis)

        assert -1 <= score <= 1
        assert isinstance(score, float)

    def test_pattern_detection(self, technical_engine, sample_price_data):
        """Test candlestick pattern detection"""
        patterns = technical_engine._detect_patterns(sample_price_data)

        assert 'candlestick_patterns' in patterns
        assert 'chart_patterns' in patterns


# ============================================================================
# Recommendation Engine Tests
# ============================================================================

class TestRecommendationEngine:
    """Test recommendation engine functionality"""

    @pytest.fixture
    def recommendation_engine(self):
        """Create recommendation engine"""
        return RecommendationEngine()

    @pytest.fixture
    def mock_analysis_results(self):
        """Mock analysis results"""
        return {
            'technical': {
                'composite_score': 0.7,
                'support_resistance': {
                    'primary_support': 145,
                    'primary_resistance': 155,
                    'current_price': 150
                },
                'momentum_indicators': {'rsi_14': 55},
                'signals': []
            },
            'fundamental': {
                'composite_score': 75,
                'valuation_models': {'upside_potential': 20},
                'quality_score': {'overall_score': 80}
            },
            'sentiment': {
                'overall_sentiment': {'score': 0.6, 'confidence': 0.8}
            },
            'ml_predictions': {
                'horizon_5': Mock(
                    predicted_price=155,
                    predicted_return=0.033,
                    model_confidence=0.75
                )
            },
            'risk_metrics': {
                'risk_score': 0.4,
                'volatility': 0.25,
                'beta': 1.2,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.15,
                'var_95': -0.02,
                'cvar_95': -0.03
            }
        }

    def test_normalize_score(self, recommendation_engine):
        """Test score normalization"""
        # Test normal range
        assert recommendation_engine._normalize_score(0.5, 0, 1) == 0.5
        assert recommendation_engine._normalize_score(75, 0, 100) == 0.75

        # Test edge cases
        assert recommendation_engine._normalize_score(0, 0, 100) == 0
        assert recommendation_engine._normalize_score(100, 0, 100) == 1

        # Test clamping
        assert recommendation_engine._normalize_score(-10, 0, 100) == 0
        assert recommendation_engine._normalize_score(110, 0, 100) == 1

        # Test equal min/max
        assert recommendation_engine._normalize_score(50, 50, 50) == 0.5

    def test_determine_action(self, recommendation_engine):
        """Test action determination from score"""
        assert recommendation_engine._determine_action(0.85) == RecommendationAction.STRONG_BUY
        assert recommendation_engine._determine_action(0.65) == RecommendationAction.BUY
        assert recommendation_engine._determine_action(0.45) == RecommendationAction.HOLD
        assert recommendation_engine._determine_action(0.25) == RecommendationAction.SELL
        assert recommendation_engine._determine_action(0.1) == RecommendationAction.STRONG_SELL

    def test_calculate_confidence(self, recommendation_engine, mock_analysis_results):
        """Test confidence calculation"""
        confidence = recommendation_engine._calculate_confidence(
            mock_analysis_results['technical'],
            mock_analysis_results['fundamental'],
            mock_analysis_results['sentiment'],
            mock_analysis_results['ml_predictions'],
            mock_analysis_results['risk_metrics']
        )

        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)

    def test_confidence_with_empty_analysis(self, recommendation_engine):
        """Test confidence with empty analysis"""
        confidence = recommendation_engine._calculate_confidence(
            {}, {}, {}, {}, {'risk_score': 0.5}
        )

        assert confidence == 0.5  # Should return default

    def test_calculate_price_targets(self, recommendation_engine, mock_analysis_results):
        """Test price target calculation"""
        targets = recommendation_engine._calculate_price_targets(
            current_price=150.0,
            ml_predictions=mock_analysis_results['ml_predictions'],
            technical_analysis=mock_analysis_results['technical'],
            risk_metrics=mock_analysis_results['risk_metrics']
        )

        assert targets['target'] > 150.0
        assert targets['stop_loss'] < 150.0
        assert targets['expected_return'] > 0

    def test_position_sizing(self, recommendation_engine, mock_analysis_results):
        """Test position sizing calculation"""
        sizing = recommendation_engine._calculate_position_sizing(
            confidence=0.7,
            risk_metrics=mock_analysis_results['risk_metrics'],
            action=RecommendationAction.BUY
        )

        assert 0 <= sizing['allocation'] <= 0.1
        assert sizing['max_size'] > 0
        assert sizing['kelly_fraction'] >= 0

    def test_position_sizing_sell_action(self, recommendation_engine, mock_analysis_results):
        """Test position sizing for sell action"""
        sizing = recommendation_engine._calculate_position_sizing(
            confidence=0.7,
            risk_metrics=mock_analysis_results['risk_metrics'],
            action=RecommendationAction.SELL
        )

        assert sizing['allocation'] == 0  # No new position on sell

    def test_should_recommend_filtering(self, recommendation_engine):
        """Test recommendation filtering"""
        # Create mock recommendation
        good_rec = Mock()
        good_rec.risk_score = 0.4
        good_rec.confidence = 0.7
        good_rec.action = RecommendationAction.BUY
        good_rec.expected_return = 0.15

        assert recommendation_engine._should_recommend(good_rec, 'moderate') is True

        # High risk for conservative
        risky_rec = Mock()
        risky_rec.risk_score = 0.8
        risky_rec.confidence = 0.7
        risky_rec.action = RecommendationAction.BUY
        risky_rec.expected_return = 0.15

        assert recommendation_engine._should_recommend(risky_rec, 'conservative') is False
        assert recommendation_engine._should_recommend(risky_rec, 'aggressive') is True

    def test_extract_key_factors(self, recommendation_engine, mock_analysis_results):
        """Test key factor extraction"""
        factors = recommendation_engine._extract_key_factors(
            mock_analysis_results['technical'],
            mock_analysis_results['fundamental'],
            mock_analysis_results['sentiment'],
            mock_analysis_results['ml_predictions']
        )

        assert isinstance(factors, list)
        assert len(factors) <= 5
        assert all(isinstance(f, str) for f in factors)

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, recommendation_engine):
        """Test risk metrics calculation"""
        # Create sample price history
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = pd.DataFrame({
            'close': np.random.normal(100, 10, 100)
        }, index=dates)

        stock_data = {
            'price_history': prices,
            'beta': 1.2
        }

        ml_predictions = {
            'horizon_5': Mock(predicted_return=0.05)
        }

        risk_metrics = await recommendation_engine._calculate_risk_metrics(
            stock_data,
            ml_predictions
        )

        assert 'risk_score' in risk_metrics
        assert 'volatility' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 0 <= risk_metrics['risk_score'] <= 1

    @pytest.mark.asyncio
    async def test_risk_metrics_insufficient_data(self, recommendation_engine):
        """Test risk metrics with insufficient data"""
        stock_data = {
            'price_history': None
        }

        risk_metrics = await recommendation_engine._calculate_risk_metrics(
            stock_data,
            {}
        )

        # Should return defaults
        assert risk_metrics['risk_score'] == 0.5
        assert risk_metrics['volatility'] == 0.0

    def test_calculate_priority(self, recommendation_engine):
        """Test priority calculation"""
        priority = recommendation_engine._calculate_priority(
            score=0.8,
            confidence=0.85,
            opportunities=['opp1', 'opp2', 'opp3']
        )

        assert 1 <= priority <= 10

    def test_identify_risks(self, recommendation_engine, mock_analysis_results):
        """Test risk identification"""
        risks = recommendation_engine._identify_risks(
            mock_analysis_results['fundamental'],
            mock_analysis_results['risk_metrics'],
            mock_analysis_results['sentiment']
        )

        assert isinstance(risks, list)
        assert len(risks) <= 4

    def test_identify_opportunities(self, recommendation_engine, mock_analysis_results):
        """Test opportunity identification"""
        opportunities = recommendation_engine._identify_opportunities(
            mock_analysis_results['fundamental'],
            mock_analysis_results['technical'],
            mock_analysis_results['sentiment']
        )

        assert isinstance(opportunities, list)
        assert len(opportunities) <= 3
