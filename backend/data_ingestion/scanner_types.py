"""
Market Scanner Types

Defines the DataProvider enum and all dataclasses used by MarketScanner:
- DataProvider: enum of available data providers
- ProviderHealth: health/availability tracking per provider
- StockQuote: real-time quote snapshot
- StockFundamentals: fundamental financial metrics
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class DataProvider(Enum):
    """Available data providers"""
    YFINANCE = "yfinance"
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    SEC_EDGAR = "sec_edgar"
    FMP = "fmp"
    NEWS_API = "news_api"
    FRED = "fred"


@dataclass
class ProviderHealth:
    """Health status for a data provider"""
    name: str
    is_available: bool = True
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 1.0

    @property
    def is_healthy(self) -> bool:
        return self.is_available and self.consecutive_failures < 5


@dataclass
class StockQuote:
    """Real-time stock quote data"""
    ticker: str
    current_price: float
    open: float
    high: float
    low: float
    previous_close: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    source: str


@dataclass
class StockFundamentals:
    """Fundamental data for a stock"""
    ticker: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = [
    "DataProvider",
    "ProviderHealth",
    "StockQuote",
    "StockFundamentals",
]
