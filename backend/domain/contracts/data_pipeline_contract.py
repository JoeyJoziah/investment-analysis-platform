"""
Data Pipeline Domain Contract (Domain 3).

Defines the interface for data ingestion, ETL operations, and data quality
management. This domain handles fetching data from external sources,
transforming it, and loading it into the system.
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import ContractResult, DomainContract


class DataSource(Enum):
    """Supported data sources."""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yfinance"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    SEC_EDGAR = "sec_edgar"
    NEWS_API = "newsapi"
    REDDIT = "reddit"
    INTERNAL = "internal"


class DataType(Enum):
    """Types of data that can be fetched."""
    PRICE_HISTORY = "price_history"
    QUOTE = "quote"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    EARNINGS = "earnings"
    DIVIDENDS = "dividends"
    SPLITS = "splits"
    INSIDER = "insider"


class JobStatus(Enum):
    """Status of an ETL job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class DataQualityLevel(Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"  # 95%+ quality
    GOOD = "good"           # 85-95% quality
    FAIR = "fair"           # 70-85% quality
    POOR = "poor"           # Below 70% quality


@dataclass(frozen=True)
class DataSourceStatusDTO:
    """Status of a data source."""
    source: DataSource
    is_available: bool
    last_successful_fetch: Optional[datetime] = None
    error_count_24h: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


@dataclass(frozen=True)
class ETLJobDTO:
    """Data transfer object for ETL job."""
    id: str
    name: str
    data_type: DataType
    source: DataSource
    status: JobStatus
    symbols: List[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None


@dataclass(frozen=True)
class DataQualityReportDTO:
    """Data quality assessment report."""
    stock_id: int
    symbol: str
    data_type: DataType
    quality_level: DataQualityLevel
    quality_score: float  # 0-100
    completeness: float   # % of expected records present
    accuracy: float       # % of records passing validation
    timeliness: float     # % of records within expected time window
    consistency: float    # % of records passing consistency checks
    issues: List[str]
    assessed_at: datetime


@dataclass(frozen=True)
class IngestionResultDTO:
    """Result of a data ingestion operation."""
    job_id: str
    data_type: DataType
    source: DataSource
    symbols_requested: int
    symbols_processed: int
    records_inserted: int
    records_updated: int
    records_failed: int
    duration_seconds: float
    quality_report: Optional[DataQualityReportDTO] = None


@dataclass(frozen=True)
class DataCoverageDTO:
    """Data coverage information for a symbol."""
    symbol: str
    stock_id: int
    price_history_start: Optional[date] = None
    price_history_end: Optional[date] = None
    fundamentals_quarters: int = 0
    fundamentals_latest: Optional[date] = None
    news_count_30d: int = 0
    technical_indicators_latest: Optional[date] = None
    last_updated: Optional[datetime] = None


class DataPipelineContract(DomainContract):
    """
    Contract for Domain 3: Data Pipeline.

    Provides data ingestion, ETL operations, and data quality management.
    Responsible for fetching data from external sources and maintaining
    data freshness and quality.
    """

    @property
    def domain_name(self) -> str:
        return "DataPipeline"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "ingest_price_data",
            "ingest_fundamentals",
            "ingest_news",
            "run_etl_job",
            "get_job_status",
            "cancel_job",
            "get_data_source_status",
            "get_data_quality_report",
            "get_data_coverage",
            "validate_data",
            "schedule_refresh",
        ]

    # Data Ingestion Operations

    @abstractmethod
    async def ingest_price_data(
        self,
        symbols: List[str],
        source: DataSource = DataSource.YAHOO_FINANCE,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        validate: bool = True
    ) -> ContractResult[IngestionResultDTO]:
        """
        Ingest historical price data for symbols.

        Args:
            symbols: List of stock symbols
            source: Data source to use
            start_date: Start date (None = max historical)
            end_date: End date (None = today)
            validate: Whether to run validation

        Returns:
            ContractResult containing ingestion result
        """
        pass

    @abstractmethod
    async def ingest_fundamentals(
        self,
        symbols: List[str],
        source: DataSource = DataSource.SEC_EDGAR,
        quarters: int = 8,
        validate: bool = True
    ) -> ContractResult[IngestionResultDTO]:
        """
        Ingest fundamental data for symbols.

        Args:
            symbols: List of stock symbols
            source: Data source to use
            quarters: Number of quarters to fetch
            validate: Whether to run validation

        Returns:
            ContractResult containing ingestion result
        """
        pass

    @abstractmethod
    async def ingest_news(
        self,
        symbols: List[str],
        source: DataSource = DataSource.NEWS_API,
        days: int = 30,
        analyze_sentiment: bool = True
    ) -> ContractResult[IngestionResultDTO]:
        """
        Ingest news articles for symbols.

        Args:
            symbols: List of stock symbols
            source: Data source to use
            days: Number of days of news to fetch
            analyze_sentiment: Whether to analyze sentiment

        Returns:
            ContractResult containing ingestion result
        """
        pass

    @abstractmethod
    async def ingest_quotes(
        self,
        symbols: List[str],
        source: DataSource = DataSource.FINNHUB
    ) -> ContractResult[IngestionResultDTO]:
        """
        Ingest real-time quotes for symbols.

        Args:
            symbols: List of stock symbols
            source: Data source to use

        Returns:
            ContractResult containing ingestion result
        """
        pass

    # ETL Job Management

    @abstractmethod
    async def run_etl_job(
        self,
        name: str,
        data_type: DataType,
        source: DataSource,
        symbols: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> ContractResult[ETLJobDTO]:
        """
        Start an ETL job.

        Args:
            name: Job name
            data_type: Type of data to process
            source: Data source
            symbols: Symbols to process
            config: Additional job configuration

        Returns:
            ContractResult containing job info
        """
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> ContractResult[ETLJobDTO]:
        """
        Get status of an ETL job.

        Args:
            job_id: Job ID

        Returns:
            ContractResult containing job status
        """
        pass

    @abstractmethod
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        data_type: Optional[DataType] = None,
        limit: int = 50
    ) -> ContractResult[List[ETLJobDTO]]:
        """
        List ETL jobs.

        Args:
            status: Filter by status
            data_type: Filter by data type
            limit: Maximum results

        Returns:
            ContractResult containing list of jobs
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> ContractResult[bool]:
        """
        Cancel a running job.

        Args:
            job_id: Job ID

        Returns:
            ContractResult indicating success
        """
        pass

    # Data Source Management

    @abstractmethod
    async def get_data_source_status(
        self,
        source: DataSource
    ) -> ContractResult[DataSourceStatusDTO]:
        """
        Get status of a data source.

        Args:
            source: Data source

        Returns:
            ContractResult containing source status
        """
        pass

    @abstractmethod
    async def list_data_sources(self) -> ContractResult[List[DataSourceStatusDTO]]:
        """
        List all data sources and their status.

        Returns:
            ContractResult containing list of source statuses
        """
        pass

    # Data Quality Operations

    @abstractmethod
    async def get_data_quality_report(
        self,
        stock_id: int,
        data_type: DataType
    ) -> ContractResult[DataQualityReportDTO]:
        """
        Get data quality report for a stock.

        Args:
            stock_id: Stock ID
            data_type: Type of data to assess

        Returns:
            ContractResult containing quality report
        """
        pass

    @abstractmethod
    async def validate_data(
        self,
        stock_id: int,
        data_type: DataType,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> ContractResult[DataQualityReportDTO]:
        """
        Validate data quality for a stock.

        Args:
            stock_id: Stock ID
            data_type: Type of data to validate
            start_date: Start date
            end_date: End date

        Returns:
            ContractResult containing validation report
        """
        pass

    @abstractmethod
    async def get_data_coverage(
        self,
        symbol: str
    ) -> ContractResult[DataCoverageDTO]:
        """
        Get data coverage information for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            ContractResult containing coverage info
        """
        pass

    # Scheduling

    @abstractmethod
    async def schedule_refresh(
        self,
        symbols: List[str],
        data_types: List[DataType],
        cron_expression: str
    ) -> ContractResult[str]:
        """
        Schedule periodic data refresh.

        Args:
            symbols: Symbols to refresh
            data_types: Types of data to refresh
            cron_expression: Cron schedule

        Returns:
            ContractResult containing schedule ID
        """
        pass

    # Health Check

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        """Check data pipeline health."""
        return ContractResult.ok({
            "domain": self.domain_name,
            "version": self.version,
            "status": "healthy",
            "capabilities": self.capabilities,
        })
