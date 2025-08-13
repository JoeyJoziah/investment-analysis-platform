"""
Market Data Testing Fixtures
Comprehensive fixtures for financial market data testing scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal
import random
from dataclasses import dataclass

from backend.models.unified_models import Stock, PriceHistory


@dataclass
class MarketScenario:
    """Represents a specific market scenario for testing"""
    name: str
    description: str
    market_trend: str  # 'bull', 'bear', 'sideways', 'volatile'
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    duration_days: int
    base_price: float
    expected_patterns: List[str]


class MarketDataGenerator:
    """Generate realistic market data for various testing scenarios"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible tests"""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def generate_price_series(
        self,
        scenario: MarketScenario,
        start_date: date,
        ticker: str = "TEST"
    ) -> pd.DataFrame:
        """Generate price series based on market scenario"""
        dates = pd.date_range(
            start=start_date,
            periods=scenario.duration_days,
            freq='D'
        )
        
        # Generate returns based on scenario
        returns = self._generate_returns(scenario)
        
        # Calculate prices
        prices = [scenario.base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (date_val, close_price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC based on daily volatility
            daily_vol = self._get_daily_volatility(scenario.volatility_level)
            
            open_price = close_price * (1 + self.rng.normal(0, daily_vol * 0.3))
            high_price = max(open_price, close_price) * (1 + abs(self.rng.normal(0, daily_vol * 0.5)))
            low_price = min(open_price, close_price) * (1 - abs(self.rng.normal(0, daily_vol * 0.5)))
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume based on price movement
            volume_base = 1000000
            price_change = abs((close_price - open_price) / open_price) if open_price > 0 else 0
            volume_multiplier = 1 + price_change * 2
            volume = int(volume_base * volume_multiplier * self.rng.lognormal(0, 0.5))
            
            data.append({
                'date': date_val.date(),
                'ticker': ticker,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'adj_close': round(close_price, 2)  # Simplified - no adjustments
            })
        
        return pd.DataFrame(data)
    
    def _generate_returns(self, scenario: MarketScenario) -> np.ndarray:
        """Generate return series based on scenario parameters"""
        n_days = scenario.duration_days
        
        # Base parameters
        vol_mapping = {
            'low': 0.01,      # 1% daily vol
            'medium': 0.02,   # 2% daily vol
            'high': 0.03,     # 3% daily vol
            'extreme': 0.05   # 5% daily vol
        }
        
        daily_vol = vol_mapping[scenario.volatility_level]
        
        # Trend parameters
        trend_mapping = {
            'bull': 0.0008,    # ~20% annual drift
            'bear': -0.0008,   # ~-20% annual drift
            'sideways': 0,     # No drift
            'volatile': 0      # No trend, high vol
        }
        
        drift = trend_mapping[scenario.market_trend]
        
        # Generate base returns
        returns = self.rng.normal(drift, daily_vol, n_days)
        
        # Add scenario-specific patterns
        if scenario.market_trend == 'volatile':
            # Add volatility clustering
            for i in range(1, n_days):
                if abs(returns[i-1]) > daily_vol:
                    returns[i] *= 1.5  # Amplify following volatility
        
        elif 'crash' in scenario.name.lower():
            # Add crash event
            crash_day = n_days // 3
            returns[crash_day] = -0.15  # 15% drop
            returns[crash_day + 1] = -0.08  # 8% drop next day
        
        elif 'rally' in scenario.name.lower():
            # Add strong rally
            rally_start = n_days // 4
            for i in range(rally_start, rally_start + 5):
                if i < n_days:
                    returns[i] += 0.03  # 3% extra gain per day
        
        return returns
    
    def _get_daily_volatility(self, vol_level: str) -> float:
        """Get daily volatility for scenario"""
        mapping = {
            'low': 0.01,
            'medium': 0.02, 
            'high': 0.03,
            'extreme': 0.05
        }
        return mapping.get(vol_level, 0.02)
    
    def generate_fundamental_data(
        self,
        ticker: str,
        market_cap_tier: str = 'large'
    ) -> Dict[str, Any]:
        """Generate realistic fundamental data"""
        
        # Market cap tiers
        cap_ranges = {
            'large': (50_000_000_000, 500_000_000_000),
            'mid': (2_000_000_000, 50_000_000_000),
            'small': (300_000_000, 2_000_000_000),
            'micro': (50_000_000, 300_000_000)
        }
        
        market_cap = self.rng.uniform(*cap_ranges[market_cap_tier])
        
        # Generate correlated metrics
        revenue = market_cap * self.rng.uniform(0.8, 2.5)  # P/S ratio 0.4-1.25
        net_margin = self.rng.uniform(0.05, 0.25)  # 5-25% net margin
        net_income = revenue * net_margin
        
        # Balance sheet items
        total_assets = revenue * self.rng.uniform(0.8, 3.0)
        equity_ratio = self.rng.uniform(0.3, 0.7)
        total_equity = total_assets * equity_ratio
        total_debt = total_assets * self.rng.uniform(0.1, 0.4)
        
        # Financial ratios
        shares_outstanding = market_cap / self.rng.uniform(20, 200)  # Price $20-200
        eps = net_income / shares_outstanding
        book_value_per_share = total_equity / shares_outstanding
        current_ratio = self.rng.uniform(1.0, 3.0)
        
        return {
            'market_cap': market_cap,
            'revenue': revenue,
            'net_income': net_income,
            'total_assets': total_assets,
            'total_equity': total_equity,
            'total_debt': total_debt,
            'shares_outstanding': shares_outstanding,
            'eps': eps,
            'book_value_per_share': book_value_per_share,
            'current_ratio': current_ratio,
            'pe_ratio': (market_cap / shares_outstanding) / eps if eps > 0 else None,
            'pb_ratio': (market_cap / shares_outstanding) / book_value_per_share if book_value_per_share > 0 else None,
            'roe': net_income / total_equity if total_equity > 0 else None,
            'debt_to_equity': total_debt / total_equity if total_equity > 0 else None,
            'gross_margin': self.rng.uniform(0.2, 0.7),
            'operating_margin': net_margin * self.rng.uniform(1.2, 1.8),
            'net_margin': net_margin
        }
    
    def generate_news_data(
        self,
        ticker: str,
        num_articles: int = 10,
        sentiment_bias: str = 'neutral'
    ) -> List[Dict[str, Any]]:
        """Generate realistic news articles with sentiment"""
        
        # Sentiment mapping
        sentiment_scores = {
            'positive': (0.3, 0.8),
            'negative': (-0.8, -0.3),
            'neutral': (-0.2, 0.2),
            'mixed': (-0.5, 0.5)
        }
        
        sentiment_range = sentiment_scores.get(sentiment_bias, (-0.2, 0.2))
        
        # News templates
        positive_headlines = [
            f"{ticker} beats earnings expectations",
            f"{ticker} announces major partnership",
            f"Analyst upgrades {ticker} rating",
            f"{ticker} reports strong quarterly growth",
            f"{ticker} launches innovative product line"
        ]
        
        negative_headlines = [
            f"{ticker} misses earnings estimates",
            f"Regulatory concerns for {ticker}",
            f"Analyst downgrades {ticker}",
            f"{ticker} faces supply chain challenges",
            f"Leadership changes at {ticker}"
        ]
        
        neutral_headlines = [
            f"{ticker} announces quarterly results",
            f"{ticker} updates business strategy",
            f"Industry analysis includes {ticker}",
            f"{ticker} participates in conference",
            f"Market update on {ticker}"
        ]
        
        articles = []
        for i in range(num_articles):
            sentiment_score = self.rng.uniform(*sentiment_range)
            
            if sentiment_score > 0.1:
                headline = random.choice(positive_headlines)
                summary = f"Positive developments for {ticker} continue to drive investor optimism."
            elif sentiment_score < -0.1:
                headline = random.choice(negative_headlines)
                summary = f"Challenges facing {ticker} raise concerns among market participants."
            else:
                headline = random.choice(neutral_headlines)
                summary = f"Regular business updates from {ticker} provide market transparency."
            
            articles.append({
                'headline': headline,
                'summary': summary,
                'sentiment_score': sentiment_score,
                'confidence': self.rng.uniform(0.6, 0.9),
                'published_at': datetime.now() - timedelta(days=self.rng.randint(0, 30)),
                'source': random.choice(['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance']),
                'url': f"https://example.com/news/{ticker.lower()}-{i}"
            })
        
        return articles


# Predefined market scenarios for testing
MARKET_SCENARIOS = {
    'bull_market': MarketScenario(
        name='Bull Market',
        description='Strong upward trend with low volatility',
        market_trend='bull',
        volatility_level='low',
        duration_days=90,
        base_price=100.0,
        expected_patterns=['uptrend', 'higher_highs', 'higher_lows']
    ),
    
    'bear_market': MarketScenario(
        name='Bear Market',
        description='Declining market with increasing volatility',
        market_trend='bear',
        volatility_level='medium',
        duration_days=120,
        base_price=100.0,
        expected_patterns=['downtrend', 'lower_highs', 'lower_lows']
    ),
    
    'volatile_market': MarketScenario(
        name='Volatile Market',
        description='High volatility with no clear trend',
        market_trend='volatile',
        volatility_level='high',
        duration_days=60,
        base_price=100.0,
        expected_patterns=['high_volatility', 'no_trend', 'whipsaws']
    ),
    
    'market_crash': MarketScenario(
        name='Market Crash',
        description='Sudden severe decline followed by volatility',
        market_trend='bear',
        volatility_level='extreme',
        duration_days=45,
        base_price=100.0,
        expected_patterns=['crash', 'panic_selling', 'extreme_volatility']
    ),
    
    'strong_rally': MarketScenario(
        name='Strong Rally',
        description='Rapid price appreciation with momentum',
        market_trend='bull',
        volatility_level='medium',
        duration_days=30,
        base_price=100.0,
        expected_patterns=['rally', 'momentum', 'breakout']
    ),
    
    'sideways_market': MarketScenario(
        name='Sideways Market',
        description='Range-bound trading with low volatility',
        market_trend='sideways',
        volatility_level='low',
        duration_days=180,
        base_price=100.0,
        expected_patterns=['range_bound', 'support_resistance', 'consolidation']
    )
}


# Pytest fixtures
@pytest.fixture
def market_data_generator():
    """Fixture providing market data generator"""
    return MarketDataGenerator()


@pytest.fixture(params=list(MARKET_SCENARIOS.keys()))
def market_scenario(request):
    """Parametrized fixture for different market scenarios"""
    return MARKET_SCENARIOS[request.param]


@pytest.fixture
def bull_market_data(market_data_generator):
    """Generate bull market data"""
    scenario = MARKET_SCENARIOS['bull_market']
    return market_data_generator.generate_price_series(
        scenario,
        start_date=date(2024, 1, 1),
        ticker="BULL"
    )


@pytest.fixture
def bear_market_data(market_data_generator):
    """Generate bear market data"""
    scenario = MARKET_SCENARIOS['bear_market']
    return market_data_generator.generate_price_series(
        scenario,
        start_date=date(2024, 1, 1),
        ticker="BEAR"
    )


@pytest.fixture
def volatile_market_data(market_data_generator):
    """Generate volatile market data"""
    scenario = MARKET_SCENARIOS['volatile_market']
    return market_data_generator.generate_price_series(
        scenario,
        start_date=date(2024, 1, 1),
        ticker="VOLT"
    )


@pytest.fixture
def multi_scenario_data(market_data_generator):
    """Generate data for multiple market scenarios"""
    data = {}
    for name, scenario in MARKET_SCENARIOS.items():
        data[name] = market_data_generator.generate_price_series(
            scenario,
            start_date=date(2024, 1, 1),
            ticker=name.upper()[:4]
        )
    return data


@pytest.fixture
def sample_stocks_with_fundamentals(market_data_generator):
    """Generate stocks with fundamental data"""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    cap_tiers = ['large', 'large', 'large', 'large', 'large']
    
    stocks_data = []
    for ticker, tier in zip(tickers, cap_tiers):
        fundamental_data = market_data_generator.generate_fundamental_data(ticker, tier)
        price_data = market_data_generator.generate_price_series(
            MARKET_SCENARIOS['bull_market'],
            start_date=date(2024, 1, 1),
            ticker=ticker
        )
        
        stocks_data.append({
            'ticker': ticker,
            'fundamentals': fundamental_data,
            'price_history': price_data,
            'market_cap_tier': tier
        })
    
    return stocks_data


@pytest.fixture
def news_data_samples(market_data_generator):
    """Generate sample news data with different sentiments"""
    return {
        'positive': market_data_generator.generate_news_data('AAPL', 5, 'positive'),
        'negative': market_data_generator.generate_news_data('TSLA', 5, 'negative'),
        'neutral': market_data_generator.generate_news_data('MSFT', 5, 'neutral'),
        'mixed': market_data_generator.generate_news_data('GOOGL', 10, 'mixed')
    }


@pytest.fixture
def options_data_sample():
    """Generate sample options data"""
    return {
        'AAPL': {
            'calls': [
                {'strike': 150, 'expiry': '2024-03-15', 'bid': 5.20, 'ask': 5.40, 'volume': 1000, 'open_interest': 5000},
                {'strike': 155, 'expiry': '2024-03-15', 'bid': 2.80, 'ask': 3.00, 'volume': 800, 'open_interest': 3000},
                {'strike': 160, 'expiry': '2024-03-15', 'bid': 1.20, 'ask': 1.40, 'volume': 500, 'open_interest': 2000}
            ],
            'puts': [
                {'strike': 145, 'expiry': '2024-03-15', 'bid': 2.10, 'ask': 2.30, 'volume': 600, 'open_interest': 4000},
                {'strike': 140, 'expiry': '2024-03-15', 'bid': 0.80, 'ask': 1.00, 'volume': 300, 'open_interest': 2500},
                {'strike': 135, 'expiry': '2024-03-15', 'bid': 0.30, 'ask': 0.50, 'volume': 200, 'open_interest': 1500}
            ]
        }
    }


@pytest.fixture
def economic_indicators_sample():
    """Generate sample economic indicators"""
    return {
        'interest_rates': {
            'federal_funds_rate': 5.25,
            '10_year_treasury': 4.20,
            '2_year_treasury': 4.80
        },
        'inflation': {
            'cpi_yoy': 3.2,
            'core_cpi_yoy': 2.8,
            'pce_yoy': 2.9
        },
        'employment': {
            'unemployment_rate': 3.8,
            'non_farm_payrolls_change': 180000,
            'participation_rate': 62.9
        },
        'gdp': {
            'quarterly_growth_rate': 0.6,
            'annual_growth_rate': 2.4
        },
        'market_indicators': {
            'vix': 18.5,
            'dollar_index': 103.2,
            'oil_price': 85.4,
            'gold_price': 2020.5
        }
    }


# Utility functions for test assertions
def assert_price_data_valid(price_df: pd.DataFrame):
    """Assert that price data is valid"""
    assert not price_df.empty, "Price data should not be empty"
    assert all(col in price_df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Check OHLC consistency
    assert (price_df['high'] >= price_df['low']).all(), "High should be >= Low"
    assert (price_df['high'] >= price_df['open']).all(), "High should be >= Open"
    assert (price_df['high'] >= price_df['close']).all(), "High should be >= Close"
    assert (price_df['low'] <= price_df['open']).all(), "Low should be <= Open"
    assert (price_df['low'] <= price_df['close']).all(), "Low should be <= Close"
    
    # Check for reasonable values
    assert (price_df['volume'] > 0).all(), "Volume should be positive"
    assert (price_df[['open', 'high', 'low', 'close']] > 0).all().all(), "Prices should be positive"


def assert_fundamental_data_valid(fundamentals: Dict[str, Any]):
    """Assert that fundamental data is valid"""
    required_fields = [
        'market_cap', 'revenue', 'net_income', 'total_assets',
        'total_equity', 'total_debt', 'pe_ratio', 'roe'
    ]
    
    for field in required_fields:
        assert field in fundamentals, f"Missing fundamental field: {field}"
    
    # Check logical relationships
    assert fundamentals['total_assets'] > 0, "Total assets should be positive"
    assert fundamentals['market_cap'] > 0, "Market cap should be positive"
    
    if fundamentals['pe_ratio']:
        assert fundamentals['pe_ratio'] > 0, "P/E ratio should be positive"