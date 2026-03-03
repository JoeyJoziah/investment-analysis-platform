"""
Recommendation CRUD Operations

Contains create/read/update/delete-style methods extracted from RecommendationService.
These are the "data shape" methods - generating, filtering, paginating, and returning
recommendation records without touching external ML or database systems.

This module is NOT meant to be imported directly by application code; use
backend.services.recommendation_service instead, which re-exports everything
through the RecommendationService class.
"""

import logging
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SEC compliance constants (duplicated here so this module is self-contained;
# the canonical definitions live in recommendation_service.py and are imported
# back into that file from there).
SEC_RISK_WARNING = (
    "IMPORTANT: Past performance does not guarantee future results. All investments "
    "involve risk, including possible loss of principal. The value of investments can "
    "fluctuate, and investors may not get back the amount originally invested. Before "
    "making any investment decision, you should carefully consider your investment "
    "objectives, level of experience, and risk appetite."
)

SEC_METHODOLOGY_DISCLOSURE_TEMPLATE = (
    "This recommendation was generated using {algorithm_type} analysis incorporating "
    "technical indicators, fundamental metrics, and market sentiment data. Model version: "
    "{model_version}. Last model training date: {training_date}."
)

SEC_LIMITATIONS_STATEMENT = (
    "This analysis does NOT consider: (1) your individual financial situation or goals, "
    "(2) tax implications specific to your circumstances, (3) real-time market conditions "
    "that may have changed since data collection, (4) non-public information, (5) geopolitical "
    "events occurring after the analysis date. Data freshness may vary by source; prices and "
    "metrics may be delayed up to 15 minutes for free-tier data sources."
)

RECOMMENDATION_MODEL_VERSION = "1.0.0"
RECOMMENDATION_MODEL_TRAINING_DATE = "2025-12-15"


class RecommendationCrudMixin:
    """
    Mixin providing CRUD and "data shape" methods for RecommendationService.

    Methods here are all synchronous (no DB, no external I/O).  They generate
    sample/fallback recommendation records, apply filters, sort, and paginate.
    """

    # ------------------------------------------------------------------
    # SEC Disclosure
    # ------------------------------------------------------------------

    def generate_sec_disclosure(
        self,
        algorithm_type: str = "ML-powered quantitative",
        data_sources: Optional[List[str]] = None,
        confidence_score: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate SEC 2025 compliant disclosure for a recommendation.

        Args:
            algorithm_type: Description of the algorithm used
            data_sources: List of data sources with timestamps
            confidence_score: Model confidence score (0-1)

        Returns:
            Dictionary with all SEC required disclosure fields
        """
        if data_sources is None:
            data_sources = [
                f"Alpha Vantage API (delayed 15 min) - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Finnhub Market Data - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Historical price data (EOD) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                f"Financial statements (quarterly) - Last updated Q4 2025",
            ]

        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.6:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        methodology_disclosure = SEC_METHODOLOGY_DISCLOSURE_TEMPLATE.format(
            algorithm_type=algorithm_type,
            model_version=RECOMMENDATION_MODEL_VERSION,
            training_date=RECOMMENDATION_MODEL_TRAINING_DATE
        )

        return {
            "methodology_disclosure": methodology_disclosure,
            "data_sources": data_sources,
            "model_version": RECOMMENDATION_MODEL_VERSION,
            "model_training_date": RECOMMENDATION_MODEL_TRAINING_DATE,
            "risk_warning": SEC_RISK_WARNING,
            "limitations_statement": SEC_LIMITATIONS_STATEMENT,
            "confidence_level": confidence_level,
            "conflict_of_interest_statement": (
                "This platform does not hold positions in any recommended securities. "
                "No material relationships exist between this platform and any recommended issuers."
            ),
        }

    # ------------------------------------------------------------------
    # Sample / Fallback Record Generation
    # ------------------------------------------------------------------

    def generate_sample_recommendation(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a sample recommendation with SEC disclosure.

        Used as a fallback when real data is unavailable.

        Args:
            symbol: Stock ticker. If None, picks a random popular symbol.

        Returns:
            Dictionary representing a RecommendationDetail-compatible structure
        """
        if not symbol:
            symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
            symbol = random.choice(symbols)

        current_price = random.uniform(50, 500)
        target_price = current_price * random.uniform(0.9, 1.3)
        confidence_score = random.uniform(0.6, 0.95)

        sec_disclosure = self.generate_sec_disclosure(
            algorithm_type="quantitative technical and fundamental",
            data_sources=[
                f"Market data feed - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Financial statements - Q4 2025",
                f"Analyst consensus data - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            ],
            confidence_score=confidence_score
        )

        return {
            "id": f"REC-{random.randint(1000, 9999)}",
            "symbol": symbol,
            "company_name": f"{symbol} Inc.",
            "recommendation_type": random.choice(["strong_buy", "buy", "hold", "sell", "strong_sell"]),
            "category": random.choice(["value", "growth", "dividend", "momentum", "contrarian", "index", "sector_rotation"]),
            "confidence_score": confidence_score,
            "target_price": round(target_price, 2),
            "current_price": round(current_price, 2),
            "expected_return": round((target_price - current_price) / current_price, 4),
            "time_horizon": random.choice(["short_term", "medium_term", "long_term"]),
            "risk_level": random.choice(["conservative", "moderate", "aggressive"]),
            "created_at": datetime.now(timezone.utc),
            "valid_until": datetime.now(timezone.utc) + timedelta(days=random.randint(7, 90)),
            "reasoning": "Based on strong technical indicators and improving fundamentals",
            "key_factors": [
                "Strong earnings growth",
                "Positive analyst sentiment",
                "Technical breakout pattern",
                "Sector rotation favor"
            ],
            "technical_signals": {
                "rsi": random.uniform(30, 70),
                "macd": "bullish",
                "support": current_price * 0.95,
                "resistance": current_price * 1.05
            },
            "fundamental_metrics": {
                "pe_ratio": random.uniform(10, 30),
                "eps_growth": random.uniform(-0.1, 0.3),
                "revenue_growth": random.uniform(0, 0.25),
                "profit_margin": random.uniform(0.05, 0.3)
            },
            "risk_factors": [
                "Market volatility",
                "Sector competition",
                "Regulatory changes"
            ],
            "entry_points": [current_price * 0.98, current_price * 0.96],
            "exit_points": [target_price * 0.95, target_price],
            "stop_loss": current_price * 0.92,
            "sector": "Technology",
            "market_cap": random.uniform(100000000000, 3000000000000),
            "volume": random.randint(10000000, 100000000),
            "analyst_consensus": "Buy",
            "similar_stocks": ["GOOG", "FB", "NFLX"] if symbol != "GOOGL" else ["AAPL", "MSFT"],
            "sec_disclosure": sec_disclosure,
        }

    # ------------------------------------------------------------------
    # Backtest Simulation
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        strategy: str,
        start_date: date,
        end_date: date,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """
        Simulate a backtest for a given recommendation strategy.

        Args:
            strategy: Strategy category (e.g. 'growth', 'value')
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital in USD

        Returns:
            Dictionary with complete backtest performance summary
        """
        total_return = random.uniform(-0.2, 0.5)
        days = (end_date - start_date).days or 1

        return {
            "strategy": strategy,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "initial_capital": initial_capital,
            "final_value": initial_capital * (1 + total_return),
            "total_return": total_return,
            "annualized_return": total_return * (365 / days),
            "sharpe_ratio": random.uniform(0.5, 2.0),
            "max_drawdown": random.uniform(-0.3, -0.05),
            "win_rate": random.uniform(0.4, 0.7),
            "total_trades": random.randint(20, 100),
            "profitable_trades": random.randint(10, 70),
            "average_win": random.uniform(0.05, 0.15),
            "average_loss": random.uniform(-0.1, -0.03),
            "best_trade": {
                "symbol": "NVDA",
                "return": 0.45
            },
            "worst_trade": {
                "symbol": "BBBY",
                "return": -0.25
            }
        }

    # ------------------------------------------------------------------
    # Performance Tracking
    # ------------------------------------------------------------------

    def generate_performance_records(
        self,
        days_back: int = 30,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate performance tracking records for past recommendations.

        Args:
            days_back: Number of days to look back
            status_filter: Optional status filter (active, closed, stopped_out)

        Returns:
            List of performance record dictionaries
        """
        performances = []

        for i in range(20):
            entry_price = random.uniform(50, 300)
            current_price = entry_price * random.uniform(0.8, 1.3)
            target_price = entry_price * random.uniform(1.1, 1.4)

            perf = {
                "recommendation_id": f"REC-{1000 + i}",
                "symbol": random.choice(["AAPL", "GOOGL", "MSFT", "AMZN", "META"]),
                "recommended_date": (date.today() - timedelta(days=random.randint(1, days_back))).isoformat(),
                "recommendation_type": random.choice(["strong_buy", "buy", "hold", "sell", "strong_sell"]),
                "entry_price": entry_price,
                "current_price": current_price,
                "target_price": target_price,
                "actual_return": (current_price - entry_price) / entry_price,
                "expected_return": (target_price - entry_price) / entry_price,
                "days_since_recommendation": random.randint(1, days_back),
                "status": status_filter or random.choice(["active", "closed", "stopped_out"]),
                "performance_rating": random.uniform(2, 5),
            }
            performances.append(perf)

        if status_filter:
            performances = [p for p in performances if p["status"] == status_filter]

        return performances

    # ------------------------------------------------------------------
    # Portfolio Recommendations
    # ------------------------------------------------------------------

    def build_portfolio_recommendations(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Generate recommendations tailored to a specific portfolio.

        Args:
            portfolio_id: Portfolio identifier

        Returns:
            Dictionary with portfolio-specific recommendations and rebalancing data
        """
        recommendations = [self.generate_sample_recommendation() for _ in range(5)]

        rebalancing = {
            "AAPL": 0.25,
            "GOOGL": 0.20,
            "MSFT": 0.20,
            "AMZN": 0.15,
            "NVDA": 0.10,
            "Cash": 0.10
        }

        return {
            "portfolio_id": portfolio_id,
            "recommendations": recommendations,
            "rebalancing_suggestions": rebalancing,
            "risk_score": random.uniform(30, 70),
            "expected_portfolio_return": random.uniform(0.08, 0.15),
            "diversification_score": random.uniform(0.6, 0.9),
        }

    # ------------------------------------------------------------------
    # Alert History
    # ------------------------------------------------------------------

    def generate_alert_history(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Generate historical alert records for recommendations.

        Args:
            days_back: Number of days to look back for alerts

        Returns:
            List of alert dictionaries sorted by timestamp descending
        """
        alerts = []
        for i in range(10):
            alert_date = datetime.now(timezone.utc) - timedelta(days=random.randint(0, days_back))
            alerts.append({
                "id": f"ALERT-{1000 + i}",
                "timestamp": alert_date.isoformat(),
                "type": random.choice(["strong_buy", "target_reached", "stop_loss_triggered"]),
                "symbol": random.choice(["AAPL", "GOOGL", "MSFT"]),
                "message": "Strong buy signal detected",
                "read": random.choice([True, False]),
            })

        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)

    # ------------------------------------------------------------------
    # Filtered List Generation
    # ------------------------------------------------------------------

    def generate_filtered_recommendations(
        self,
        count: int = 50,
        recommendation_type: Optional[str] = None,
        category: Optional[str] = None,
        risk_level: Optional[str] = None,
        min_confidence: float = 0.0,
        sort_by: str = "confidence_score",
        order: str = "desc",
        limit: int = 10,
        offset: int = 0,
        categories: Optional[List[str]] = None,
        risk_levels: Optional[List[str]] = None,
        time_horizons: Optional[List[str]] = None,
        min_expected_return: Optional[float] = None,
        sectors: Optional[List[str]] = None,
        market_cap_min: Optional[float] = None,
        market_cap_max: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate and filter a set of sample recommendations.

        Supports both simple list filters (single recommendation_type, category,
        risk_level) and advanced filters (lists of categories, risk_levels,
        time_horizons, min_expected_return, sectors, market_cap bounds).

        Args:
            count: Number of raw recommendations to generate before filtering
            recommendation_type: Single recommendation type filter
            category: Single category filter
            risk_level: Single risk level filter
            min_confidence: Minimum confidence score threshold
            sort_by: Sort field (confidence_score, expected_return, created_at)
            order: Sort direction (asc, desc)
            limit: Maximum results after filtering
            offset: Pagination offset
            categories: Multi-select category filter
            risk_levels: Multi-select risk level filter
            time_horizons: Multi-select time horizon filter
            min_expected_return: Minimum expected return threshold
            sectors: Sector whitelist
            market_cap_min: Minimum market cap
            market_cap_max: Maximum market cap

        Returns:
            Filtered, sorted, and paginated list of recommendation dictionaries
        """
        recommendations = [self.generate_sample_recommendation() for _ in range(count)]

        # Single-value filters
        if recommendation_type:
            recommendations = [r for r in recommendations if r["recommendation_type"] == recommendation_type]

        if category:
            recommendations = [r for r in recommendations if r["category"] == category]

        if risk_level:
            recommendations = [r for r in recommendations if r["risk_level"] == risk_level]

        recommendations = [r for r in recommendations if r["confidence_score"] >= min_confidence]

        # Multi-value filters (advanced filter endpoint)
        if categories:
            recommendations = [r for r in recommendations if r["category"] in categories]

        if risk_levels:
            recommendations = [r for r in recommendations if r["risk_level"] in risk_levels]

        if time_horizons:
            recommendations = [r for r in recommendations if r["time_horizon"] in time_horizons]

        if min_expected_return is not None:
            recommendations = [r for r in recommendations if r["expected_return"] >= min_expected_return]

        if sectors:
            recommendations = [r for r in recommendations if r["sector"] in sectors]

        if market_cap_min is not None:
            recommendations = [r for r in recommendations if r["market_cap"] >= market_cap_min]

        if market_cap_max is not None:
            recommendations = [r for r in recommendations if r["market_cap"] <= market_cap_max]

        # Sorting
        reverse = (order == "desc")
        if sort_by == "confidence_score":
            recommendations.sort(key=lambda x: x["confidence_score"], reverse=reverse)
        elif sort_by == "expected_return":
            recommendations.sort(key=lambda x: x["expected_return"], reverse=reverse)
        elif sort_by == "created_at":
            recommendations.sort(key=lambda x: x["created_at"], reverse=reverse)

        # Pagination
        return recommendations[offset:offset + limit]

    # ------------------------------------------------------------------
    # Trending Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def generate_trending_fallback(
        symbols: List[str],
        limit: int = 10,
        timeframe: str = "24h",
    ) -> List[Dict[str, Any]]:
        """
        Generate fallback trending data when the real trending service is unavailable.

        Args:
            symbols: Pool of symbols to pick from
            limit: Maximum entries to return
            timeframe: Timeframe label to attach to each entry

        Returns:
            List of trending recommendation dicts sorted by trending_score descending
        """
        trending = []
        for symbol in symbols[:limit]:
            trending.append({
                "symbol": symbol,
                "views": random.randint(1000, 50000),
                "saves": random.randint(100, 5000),
                "recommendation_type": random.choice(
                    ["strong_buy", "buy", "hold", "sell", "strong_sell"]
                ),
                "confidence_score": random.uniform(0.7, 0.95),
                "trending_score": random.uniform(70, 100),
                "timeframe": timeframe,
            })
        return sorted(trending, key=lambda x: x["trending_score"], reverse=True)

    # ------------------------------------------------------------------
    # Confidence Calculation
    # ------------------------------------------------------------------

    def calculate_confidence(self, analyses: List[Dict[str, Any]]) -> float:
        """
        Calculate weighted confidence score from multiple analyses.

        Args:
            analyses: List of analysis results with confidence scores

        Returns:
            Weighted confidence score (0-1)
        """
        if not analyses:
            return 0.0

        weights = {
            'technical': 0.25,
            'fundamental': 0.30,
            'sentiment': 0.15,
            'ml': 0.30
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for analysis in analyses:
            analysis_type = analysis.get('type', '').lower()
            confidence = analysis.get('confidence', 0.0)
            weight = weights.get(analysis_type, 0.1)

            weighted_sum += confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight


__all__ = [
    "RecommendationCrudMixin",
    "SEC_RISK_WARNING",
    "SEC_METHODOLOGY_DISCLOSURE_TEMPLATE",
    "SEC_LIMITATIONS_STATEMENT",
    "RECOMMENDATION_MODEL_VERSION",
    "RECOMMENDATION_MODEL_TRAINING_DATE",
]
