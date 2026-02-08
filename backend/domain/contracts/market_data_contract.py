"""
Market Data Domain Contract (Domain 1).

Defines the interface for accessing market data including stocks, prices,
exchanges, sectors, and industries. This is the core reference data domain
that other domains depend on.
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import ContractResult, DomainContract


class AssetType(Enum):
    """Types of tradeable assets."""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    OPTION = "option"
    FUTURE = "future"


@dataclass(frozen=True)
class ExchangeDTO:
    """Data transfer object for exchange information."""
    id: int
    code: str
    name: str
    timezone: str
    country: str
    currency: str
    market_open: Optional[str] = None
    market_close: Optional[str] = None


@dataclass(frozen=True)
class SectorDTO:
    """Data transfer object for sector information."""
    id: int
    name: str
    description: Optional[str] = None


@dataclass(frozen=True)
class IndustryDTO:
    """Data transfer object for industry information."""
    id: int
    name: str
    sector_id: int
    description: Optional[str] = None


@dataclass(frozen=True)
class StockDTO:
    """Data transfer object for stock information."""
    id: int
    symbol: str
    name: str
    exchange_id: int
    exchange_code: Optional[str] = None
    sector_id: Optional[int] = None
    sector_name: Optional[str] = None
    industry_id: Optional[int] = None
    industry_name: Optional[str] = None
    asset_type: AssetType = AssetType.STOCK
    market_cap: Optional[float] = None
    shares_outstanding: Optional[int] = None
    country: str = "US"
    currency: str = "USD"
    is_active: bool = True
    is_tradeable: bool = True


@dataclass(frozen=True)
class PriceHistoryDTO:
    """Data transfer object for historical price data."""
    stock_id: int
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_close: Optional[Decimal] = None
    volume: int = 0
    daily_return: Optional[float] = None
    intraday_volatility: Optional[float] = None


@dataclass(frozen=True)
class QuoteDTO:
    """Data transfer object for real-time quotes."""
    symbol: str
    price: Decimal
    change: Decimal
    change_percent: float
    volume: int
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    previous_close: Optional[Decimal] = None


@dataclass(frozen=True)
class TechnicalIndicatorDTO:
    """Data transfer object for technical indicators."""
    stock_id: int
    date: date
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None


class MarketDataContract(DomainContract):
    """
    Contract for Domain 1: Market Data.

    Provides access to stock, price, exchange, sector, and industry data.
    This is the foundational domain that other domains depend on for
    reference data.
    """

    @property
    def domain_name(self) -> str:
        return "MarketData"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "get_stock",
            "search_stocks",
            "list_stocks_by_sector",
            "get_exchange",
            "list_exchanges",
            "get_sector",
            "list_sectors",
            "get_industry",
            "get_price_history",
            "get_latest_price",
            "get_quote",
            "get_technical_indicators",
        ]

    # Stock Operations

    @abstractmethod
    async def get_stock(self, symbol: str) -> ContractResult[StockDTO]:
        """
        Get stock by symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            ContractResult containing StockDTO if found
        """
        pass

    @abstractmethod
    async def get_stock_by_id(self, stock_id: int) -> ContractResult[StockDTO]:
        """
        Get stock by database ID.

        Args:
            stock_id: Internal stock ID

        Returns:
            ContractResult containing StockDTO if found
        """
        pass

    @abstractmethod
    async def search_stocks(
        self,
        query: str,
        limit: int = 20,
        asset_types: Optional[List[AssetType]] = None,
        active_only: bool = True
    ) -> ContractResult[List[StockDTO]]:
        """
        Search stocks by name or symbol.

        Args:
            query: Search term
            limit: Maximum results to return
            asset_types: Filter by asset types
            active_only: Only return active stocks

        Returns:
            ContractResult containing list of matching stocks
        """
        pass

    @abstractmethod
    async def list_stocks_by_sector(
        self,
        sector_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> ContractResult[List[StockDTO]]:
        """
        List stocks in a given sector.

        Args:
            sector_id: Sector ID to filter by
            limit: Maximum results
            offset: Pagination offset

        Returns:
            ContractResult containing list of stocks
        """
        pass

    @abstractmethod
    async def list_stocks_by_exchange(
        self,
        exchange_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> ContractResult[List[StockDTO]]:
        """
        List stocks on a given exchange.

        Args:
            exchange_id: Exchange ID to filter by
            limit: Maximum results
            offset: Pagination offset

        Returns:
            ContractResult containing list of stocks
        """
        pass

    # Exchange Operations

    @abstractmethod
    async def get_exchange(self, code: str) -> ContractResult[ExchangeDTO]:
        """
        Get exchange by code.

        Args:
            code: Exchange code (e.g., "NASDAQ")

        Returns:
            ContractResult containing ExchangeDTO if found
        """
        pass

    @abstractmethod
    async def list_exchanges(self) -> ContractResult[List[ExchangeDTO]]:
        """
        List all exchanges.

        Returns:
            ContractResult containing list of exchanges
        """
        pass

    # Sector/Industry Operations

    @abstractmethod
    async def get_sector(self, sector_id: int) -> ContractResult[SectorDTO]:
        """
        Get sector by ID.

        Args:
            sector_id: Sector ID

        Returns:
            ContractResult containing SectorDTO if found
        """
        pass

    @abstractmethod
    async def list_sectors(self) -> ContractResult[List[SectorDTO]]:
        """
        List all sectors.

        Returns:
            ContractResult containing list of sectors
        """
        pass

    @abstractmethod
    async def get_industry(self, industry_id: int) -> ContractResult[IndustryDTO]:
        """
        Get industry by ID.

        Args:
            industry_id: Industry ID

        Returns:
            ContractResult containing IndustryDTO if found
        """
        pass

    @abstractmethod
    async def list_industries_by_sector(
        self,
        sector_id: int
    ) -> ContractResult[List[IndustryDTO]]:
        """
        List industries in a sector.

        Args:
            sector_id: Sector ID

        Returns:
            ContractResult containing list of industries
        """
        pass

    # Price Operations

    @abstractmethod
    async def get_price_history(
        self,
        stock_id: int,
        start_date: date,
        end_date: date
    ) -> ContractResult[List[PriceHistoryDTO]]:
        """
        Get historical price data for a stock.

        Args:
            stock_id: Stock ID
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            ContractResult containing list of price history entries
        """
        pass

    @abstractmethod
    async def get_latest_price(self, stock_id: int) -> ContractResult[PriceHistoryDTO]:
        """
        Get the most recent price for a stock.

        Args:
            stock_id: Stock ID

        Returns:
            ContractResult containing latest price data
        """
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> ContractResult[QuoteDTO]:
        """
        Get real-time quote for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            ContractResult containing real-time quote
        """
        pass

    # Technical Indicators

    @abstractmethod
    async def get_technical_indicators(
        self,
        stock_id: int,
        date: date
    ) -> ContractResult[TechnicalIndicatorDTO]:
        """
        Get technical indicators for a stock on a specific date.

        Args:
            stock_id: Stock ID
            date: Date for indicators

        Returns:
            ContractResult containing technical indicators
        """
        pass

    @abstractmethod
    async def get_technical_indicators_history(
        self,
        stock_id: int,
        start_date: date,
        end_date: date
    ) -> ContractResult[List[TechnicalIndicatorDTO]]:
        """
        Get technical indicators history.

        Args:
            stock_id: Stock ID
            start_date: Start date
            end_date: End date

        Returns:
            ContractResult containing list of technical indicators
        """
        pass

    # Health Check

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        """Check market data service health."""
        return ContractResult.ok({
            "domain": self.domain_name,
            "version": self.version,
            "status": "healthy",
            "capabilities": self.capabilities,
        })
