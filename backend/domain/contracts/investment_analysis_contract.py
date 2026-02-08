"""
Investment Analysis Domain Contract (Domain 5).

Defines the interface for investment analysis including technical analysis,
fundamental analysis, recommendations, and investment thesis generation.

This domain integrates with:
- Domain 1 (Market Data): Stock prices and technical indicators
- Domain 2 (Portfolio): Portfolio context for recommendations
- Domain 3 (Data Pipeline): Data freshness and quality
- Domain 4 (ML): Prediction signals for recommendations
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import ContractResult, DomainContract


class RecommendationAction(Enum):
    """Investment recommendation actions."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TrendDirection(Enum):
    """Price trend direction."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class ValuationAssessment(Enum):
    """Stock valuation assessment."""
    SIGNIFICANTLY_UNDERVALUED = "significantly_undervalued"
    UNDERVALUED = "undervalued"
    FAIRLY_VALUED = "fairly_valued"
    OVERVALUED = "overvalued"
    SIGNIFICANTLY_OVERVALUED = "significantly_overvalued"


class MoatRating(Enum):
    """Competitive moat rating."""
    WIDE = "wide"
    NARROW = "narrow"
    NONE = "none"


class QualityGrade(Enum):
    """Business quality grade."""
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    F = "F"


@dataclass(frozen=True)
class TechnicalAnalysisDTO:
    """Data transfer object for technical analysis."""
    stock_id: int
    symbol: str
    analysis_date: datetime
    trend: TrendDirection
    trend_strength: float  # 0-1
    rsi: float
    macd: float
    macd_signal: float
    bollinger_percent: float
    support_level: Decimal
    resistance_level: Decimal
    volume_trend: str
    pattern_signals: List[str]
    overall_score: float  # -1 to 1
    signals: List[Dict[str, Any]]


@dataclass(frozen=True)
class FundamentalAnalysisDTO:
    """Data transfer object for fundamental analysis."""
    stock_id: int
    symbol: str
    analysis_date: datetime
    valuation: ValuationAssessment
    quality_grade: QualityGrade
    moat_rating: MoatRating

    # Valuation metrics
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None

    # Profitability metrics
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None

    # Growth metrics
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    fcf_growth: Optional[float] = None

    # Financial health
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None

    # Scores
    piotroski_score: Optional[int] = None
    altman_z_score: Optional[float] = None
    beneish_m_score: Optional[float] = None
    overall_score: float = 0.0

    # DCF valuation
    dcf_value: Optional[Decimal] = None
    dcf_upside: Optional[float] = None


@dataclass(frozen=True)
class SentimentAnalysisDTO:
    """Data transfer object for sentiment analysis."""
    stock_id: int
    symbol: str
    analysis_date: datetime
    overall_sentiment: float  # -1 to 1
    news_sentiment: float
    social_sentiment: float
    analyst_sentiment: float
    news_count: int
    positive_mentions: int
    negative_mentions: int
    key_topics: List[str]
    recent_headlines: List[str]


@dataclass(frozen=True)
class RecommendationDTO:
    """Data transfer object for investment recommendation."""
    id: int
    recommendation_id: str
    stock_id: int
    symbol: str
    created_at: datetime
    valid_until: Optional[datetime] = None

    # Recommendation
    action: RecommendationAction = RecommendationAction.HOLD
    confidence: float = 0.0
    priority: int = 5  # 1-10

    # Price targets
    entry_price: Optional[Decimal] = None
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    expected_return: Optional[float] = None
    time_horizon_days: int = 90

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: Optional[float] = None
    volatility_percentile: Optional[float] = None
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Component scores
    technical_score: Optional[float] = None
    fundamental_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    ml_score: Optional[float] = None
    overall_score: float = 0.0

    # Explanation
    reasoning: Optional[str] = None
    key_factors: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    actual_return: Optional[float] = None
    outcome: Optional[str] = None


@dataclass(frozen=True)
class InvestmentThesisDTO:
    """Data transfer object for investment thesis."""
    stock_id: int
    symbol: str
    created_at: datetime

    # Thesis
    title: str
    summary: str
    bull_case: List[str]
    bear_case: List[str]
    catalysts: List[str]
    risks: List[str]

    # Assessment
    valuation_assessment: ValuationAssessment
    quality_assessment: QualityGrade
    moat_assessment: MoatRating

    # Key metrics
    key_metrics: Dict[str, Any]

    # Confidence
    confidence: float = 0.0
    time_horizon: str = "medium_term"


@dataclass(frozen=True)
class SectorAnalysisDTO:
    """Data transfer object for sector analysis."""
    sector_id: int
    sector_name: str
    analysis_date: datetime
    trend: TrendDirection
    relative_strength: float
    top_performers: List[str]
    underperformers: List[str]
    sector_rotation_signal: str
    recommended_allocation: float


@dataclass(frozen=True)
class ScreenerResultDTO:
    """Data transfer object for stock screener result."""
    stock_id: int
    symbol: str
    name: str
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    price: Optional[Decimal] = None
    change_pct: Optional[float] = None
    volume: Optional[int] = None
    match_score: float = 0.0
    matched_criteria: List[str] = field(default_factory=list)


class InvestmentAnalysisContract(DomainContract):
    """
    Contract for Domain 5: Investment Analysis.

    Provides technical analysis, fundamental analysis, recommendations,
    and investment thesis generation. This domain synthesizes data from
    all other domains to provide actionable investment insights.

    Cross-domain integrations:
    - Market Data (Domain 1): Price history, quotes, technical indicators
    - Portfolio (Domain 2): Portfolio context for personalized recommendations
    - Data Pipeline (Domain 3): Data quality checks for analysis reliability
    - ML (Domain 4): Prediction signals for recommendation generation
    """

    @property
    def domain_name(self) -> str:
        return "InvestmentAnalysis"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "get_technical_analysis",
            "get_fundamental_analysis",
            "get_sentiment_analysis",
            "generate_recommendation",
            "get_recommendation",
            "list_recommendations",
            "generate_thesis",
            "get_thesis",
            "get_sector_analysis",
            "run_screener",
            "compare_stocks",
        ]

    # Technical Analysis

    @abstractmethod
    async def get_technical_analysis(
        self,
        stock_id: int,
        as_of_date: Optional[date] = None
    ) -> ContractResult[TechnicalAnalysisDTO]:
        """
        Get technical analysis for a stock.

        Integration: Uses Market Data (Domain 1) for price history
        and technical indicators.

        Args:
            stock_id: Stock ID
            as_of_date: Date for analysis (None = latest)

        Returns:
            ContractResult containing technical analysis
        """
        pass

    # Fundamental Analysis

    @abstractmethod
    async def get_fundamental_analysis(
        self,
        stock_id: int,
        include_peers: bool = True
    ) -> ContractResult[FundamentalAnalysisDTO]:
        """
        Get fundamental analysis for a stock.

        Integration: Uses Market Data (Domain 1) for fundamentals data
        and Data Pipeline (Domain 3) for data quality validation.

        Args:
            stock_id: Stock ID
            include_peers: Include peer comparison

        Returns:
            ContractResult containing fundamental analysis
        """
        pass

    # Sentiment Analysis

    @abstractmethod
    async def get_sentiment_analysis(
        self,
        stock_id: int,
        days: int = 30
    ) -> ContractResult[SentimentAnalysisDTO]:
        """
        Get sentiment analysis for a stock.

        Integration: Uses Data Pipeline (Domain 3) for news data.

        Args:
            stock_id: Stock ID
            days: Number of days of sentiment data

        Returns:
            ContractResult containing sentiment analysis
        """
        pass

    # Recommendations

    @abstractmethod
    async def generate_recommendation(
        self,
        stock_id: int,
        user_id: Optional[int] = None,
        risk_tolerance: Optional[str] = None
    ) -> ContractResult[RecommendationDTO]:
        """
        Generate investment recommendation for a stock.

        Integration: Uses all other domains:
        - Market Data (Domain 1): Price data and indicators
        - Portfolio (Domain 2): User portfolio context if user_id provided
        - Data Pipeline (Domain 3): Data quality validation
        - ML (Domain 4): Prediction signals

        Args:
            stock_id: Stock ID
            user_id: Optional user ID for personalization
            risk_tolerance: Risk tolerance level

        Returns:
            ContractResult containing recommendation
        """
        pass

    @abstractmethod
    async def get_recommendation(
        self,
        recommendation_id: str
    ) -> ContractResult[RecommendationDTO]:
        """
        Get a specific recommendation.

        Args:
            recommendation_id: Recommendation UUID

        Returns:
            ContractResult containing recommendation
        """
        pass

    @abstractmethod
    async def list_recommendations(
        self,
        stock_id: Optional[int] = None,
        action: Optional[RecommendationAction] = None,
        min_confidence: float = 0.0,
        active_only: bool = True,
        limit: int = 20
    ) -> ContractResult[List[RecommendationDTO]]:
        """
        List recommendations.

        Args:
            stock_id: Filter by stock
            action: Filter by action
            min_confidence: Minimum confidence threshold
            active_only: Only active recommendations
            limit: Maximum results

        Returns:
            ContractResult containing list of recommendations
        """
        pass

    @abstractmethod
    async def update_recommendation_outcome(
        self,
        recommendation_id: str,
        actual_return: float,
        outcome: str
    ) -> ContractResult[RecommendationDTO]:
        """
        Update recommendation with actual outcome.

        Args:
            recommendation_id: Recommendation UUID
            actual_return: Actual return achieved
            outcome: Outcome (success, partial, failure)

        Returns:
            ContractResult containing updated recommendation
        """
        pass

    # Investment Thesis

    @abstractmethod
    async def generate_thesis(
        self,
        stock_id: int
    ) -> ContractResult[InvestmentThesisDTO]:
        """
        Generate investment thesis for a stock.

        Integration: Uses fundamental analysis and compiles
        a comprehensive investment case.

        Args:
            stock_id: Stock ID

        Returns:
            ContractResult containing investment thesis
        """
        pass

    @abstractmethod
    async def get_thesis(
        self,
        stock_id: int
    ) -> ContractResult[InvestmentThesisDTO]:
        """
        Get existing investment thesis.

        Args:
            stock_id: Stock ID

        Returns:
            ContractResult containing investment thesis
        """
        pass

    # Sector Analysis

    @abstractmethod
    async def get_sector_analysis(
        self,
        sector_id: int
    ) -> ContractResult[SectorAnalysisDTO]:
        """
        Get sector analysis.

        Integration: Uses Market Data (Domain 1) for sector stocks
        and price data.

        Args:
            sector_id: Sector ID

        Returns:
            ContractResult containing sector analysis
        """
        pass

    # Screener

    @abstractmethod
    async def run_screener(
        self,
        criteria: Dict[str, Any],
        limit: int = 50
    ) -> ContractResult[List[ScreenerResultDTO]]:
        """
        Run stock screener with given criteria.

        Integration: Uses Market Data (Domain 1) for filtering stocks.

        Args:
            criteria: Screening criteria dictionary
            limit: Maximum results

        Returns:
            ContractResult containing screener results
        """
        pass

    # Comparison

    @abstractmethod
    async def compare_stocks(
        self,
        stock_ids: List[int]
    ) -> ContractResult[Dict[str, Any]]:
        """
        Compare multiple stocks.

        Args:
            stock_ids: List of stock IDs to compare

        Returns:
            ContractResult containing comparison data
        """
        pass

    # Health Check

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        """Check investment analysis service health."""
        return ContractResult.ok({
            "domain": self.domain_name,
            "version": self.version,
            "status": "healthy",
            "capabilities": self.capabilities,
        })
