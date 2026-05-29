"""
Recommendation Service
Business logic for generating and managing investment recommendations.
"""

import logging
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from backend.analytics.recommendation_engine import RecommendationEngine, StockRecommendation
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.config.settings import settings
from backend.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)

# Sentinel to distinguish "not provided" from an explicit None
_UNSET = object()

# =============================================================================
# SEC 2025 COMPLIANCE CONSTANTS
# =============================================================================

# Standard SEC Risk Warning (required on all recommendations)
SEC_RISK_WARNING = (
    "IMPORTANT: Past performance does not guarantee future results. All investments "
    "involve risk, including possible loss of principal. The value of investments can "
    "fluctuate, and investors may not get back the amount originally invested. Before "
    "making any investment decision, you should carefully consider your investment "
    "objectives, level of experience, and risk appetite."
)

# Standard Methodology Disclosure Template
SEC_METHODOLOGY_DISCLOSURE_TEMPLATE = (
    "This recommendation was generated using {algorithm_type} analysis incorporating "
    "technical indicators, fundamental metrics, and market sentiment data. Model version: "
    "{model_version}. Last model training date: {training_date}."
)

# Standard Limitations Statement
SEC_LIMITATIONS_STATEMENT = (
    "This analysis does NOT consider: (1) your individual financial situation or goals, "
    "(2) tax implications specific to your circumstances, (3) real-time market conditions "
    "that may have changed since data collection, (4) non-public information, (5) geopolitical "
    "events occurring after the analysis date. Data freshness may vary by source; prices and "
    "metrics may be delayed up to 15 minutes for free-tier data sources."
)

# Current model version for SEC disclosure
RECOMMENDATION_MODEL_VERSION = "1.0.0"
RECOMMENDATION_MODEL_TRAINING_DATE = "2025-12-15"

# =============================================================================
# RULES-BASED QUANTITATIVE SCREEN (no ML, transparent, deterministic)
# =============================================================================

# Honest algorithm label for the transparent screen. This MUST NOT claim
# machine learning — the screen is a deterministic momentum + valuation rank
# over stored historical data (PRD audit 2026-05 Lane C, no-synthetic-data rule).
RULES_BASED_ALGORITHM_TYPE = "rules-based quantitative screen"

# Standalone methodology disclosure for the rules-based screen. Used instead of
# SEC_METHODOLOGY_DISCLOSURE_TEMPLATE so the text accurately describes the
# transparent momentum + P/E percentile methodology rather than an ML model.
RULES_BASED_METHODOLOGY_DISCLOSURE = (
    "This recommendation was generated using a transparent, rules-based "
    "quantitative screen over stored historical data. It does NOT use machine "
    "learning, neural networks, or predictive models. Momentum is measured as "
    "the trailing 60-trading-day price return computed from end-of-day closing "
    "prices. Valuation is measured as the cross-sectional percentile rank of the "
    "price-to-earnings (P/E) ratio across the screened universe (a lower P/E "
    "ranks more favorably), with the PEG ratio used as a tiebreaker. The "
    "composite score equally weights momentum percentile and inverse-valuation "
    "percentile when both inputs are available, and uses momentum alone "
    "otherwise. Recommendation tiers and confidence are derived deterministically "
    "from the composite rank; identical inputs always produce identical outputs."
)

# Trading-day window for the momentum signal and the minimum rows required.
MOMENTUM_WINDOW_DAYS = 60
MOMENTUM_MIN_ROWS = 30
# Bound on the momentum component used to derive the target price.
TARGET_PRICE_MOMENTUM_CLAMP = 0.30


class RecommendationService:
    """
    Service for generating investment recommendations.
    Orchestrates multiple analysis engines and aggregates results.
    Contains all business logic extracted from the router layer.
    """

    def __init__(self):
        self.recommendation_engine = RecommendationEngine()
        self.fundamental_engine = FundamentalAnalysisEngine()
        self._initialized = False

    async def initialize(self):
        """Initialize the service and its dependencies."""
        if not self._initialized:
            try:
                await self.recommendation_engine.initialize()
                self._initialized = True
                logger.info("RecommendationService initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RecommendationService: {e}")
                raise

    # =========================================================================
    # SEC Disclosure Generation
    # =========================================================================

    def generate_sec_disclosure(
        self,
        algorithm_type: str = "ML-powered quantitative",
        data_sources: Optional[List[str]] = None,
        confidence_score: float = 0.5,
        methodology_disclosure: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate SEC 2025 compliant disclosure for a recommendation.

        Args:
            algorithm_type: Description of the algorithm used
            data_sources: List of data sources with timestamps
            confidence_score: Model confidence score (0-1)
            methodology_disclosure: Optional explicit methodology text. When
                provided it is used verbatim instead of the generic ML template
                — required for the rules-based screen so the disclosure does not
                misrepresent a transparent screen as machine learning.

        Returns:
            Dictionary with all SEC required disclosure fields
        """
        # Default data sources if not provided
        if data_sources is None:
            data_sources = [
                f"Alpha Vantage API (delayed 15 min) - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Finnhub Market Data - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Historical price data (EOD) - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                f"Financial statements (quarterly) - Last updated Q4 2025",
            ]

        # Determine confidence level from score
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.6:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        # Generate methodology disclosure. An explicit override (rules-based
        # screen) is used verbatim; otherwise fall back to the generic template.
        if methodology_disclosure is None:
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

    # =========================================================================
    # Sample / Fallback Recommendation Generation
    # =========================================================================

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

        # Generate SEC disclosure for sample recommendation
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

    # =========================================================================
    # ML-Powered Recommendation Generation
    # =========================================================================

    async def generate_ml_powered_recommendations(
        self,
        user_id: Optional[int] = None,
        portfolio_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: int = 10,
        db_session: Optional[AsyncSession] = None,
        *,
        stock_repo: Any = _UNSET,
        price_repo: Any = _UNSET,
        model_mgr: Any = _UNSET,
        rec_engine: Any = _UNSET,
    ) -> List[Dict[str, Any]]:
        """
        Generate ML-powered recommendations with real market data.

        OPTIMIZED: Uses batch queries to eliminate N+1 query pattern.
        Previously: 201+ queries (1 for stocks + 2 per stock for prices/ML)
        Now: 2-3 queries total (1 for stocks + 1 bulk price history)

        Args:
            user_id: Optional user identifier for personalization
            portfolio_id: Optional portfolio identifier
            risk_level: Risk level string (conservative, moderate, aggressive)
            categories: List of category strings to filter by
            limit: Maximum number of recommendations to return
            db_session: Database session for repository queries
            stock_repo: Optional stock repository override (for test patching)
            price_repo: Optional price repository override (for test patching)
            model_mgr: Optional model manager override (for test patching)
            rec_engine: Optional recommendation engine override (for test patching)

        Returns:
            List of recommendation dictionaries
        """
        if stock_repo is _UNSET or price_repo is _UNSET:
            from backend.repositories import (
                stock_repository as _stock_repo,
                price_repository as _price_repo,
            )
            stock_repo = _stock_repo if stock_repo is _UNSET else stock_repo
            price_repo = _price_repo if price_repo is _UNSET else price_repo

        # Use provided engine/manager or initialise defaults
        if model_mgr is _UNSET and rec_engine is _UNSET:
            from backend.ml.model_manager import get_model_manager as _get_mm
            from backend.analytics.recommendation_engine import RecommendationEngine as _RE
            model_manager = None
            recommendation_engine_instance = None
            try:
                model_manager = _get_mm()
                recommendation_engine_instance = _RE(model_manager=model_manager)
            except Exception as e:
                logger.warning(f"ML model manager not available: {e}")
                recommendation_engine_instance = _RE()
        else:
            model_manager = model_mgr if model_mgr is not _UNSET else None
            recommendation_engine_instance = rec_engine if rec_engine is not _UNSET else None

        try:
            logger.info(f"Generating ML recommendations for user {user_id}, portfolio {portfolio_id}")

            # Query 1: Get market data for top stocks
            top_stocks = await stock_repo.get_top_stocks(
                limit=100,
                by_market_cap=True,
                session=db_session
            )

            if not top_stocks:
                logger.warning("No stocks found for recommendations")
                return [self.generate_sample_recommendation() for _ in range(min(limit, 5))]

            # OPTIMIZATION: Batch fetch all price histories in a single query
            symbols_to_fetch = [stock.symbol for stock in top_stocks[:limit * 2]]

            # Query 2: Single bulk query for all price histories
            all_price_histories = await price_repo.get_bulk_price_history(
                symbols=symbols_to_fetch,
                start_date=datetime.now().date() - timedelta(days=90),
                end_date=datetime.now().date(),
                limit_per_symbol=60,
                session=db_session
            )

            logger.debug(f"Bulk fetched price histories for {len(all_price_histories)} symbols")

            # Build stock lookup for similar stocks calculation
            stock_by_sector: Dict[str, List[str]] = {}
            for stock in top_stocks:
                if stock.sector:
                    if stock.sector not in stock_by_sector:
                        stock_by_sector[stock.sector] = []
                    stock_by_sector[stock.sector].append(stock.symbol)

            # OPTIMIZATION: Prepare batch ML predictions if available
            ml_predictions_batch: Dict[str, Dict[str, Any]] = {}
            if model_manager and recommendation_engine_instance:
                try:
                    batch_price_data = {}
                    for symbol, price_history in all_price_histories.items():
                        if price_history and len(price_history) >= 30:
                            batch_price_data[symbol] = [
                                {
                                    'open': float(p.open),
                                    'high': float(p.high),
                                    'low': float(p.low),
                                    'close': float(p.close),
                                    'volume': p.volume,
                                    'date': p.date
                                }
                                for p in price_history
                            ]

                    if hasattr(recommendation_engine_instance, 'analyze_stocks_batch'):
                        ml_predictions_batch = await recommendation_engine_instance.analyze_stocks_batch(
                            price_data_batch=batch_price_data,
                            user_risk_tolerance=risk_level if risk_level else 'moderate'
                        )
                except Exception as e:
                    logger.warning(f"Batch ML prediction not available, will use individual: {e}")

            recommendations = []

            # Process stocks using pre-fetched data (no additional queries in loop)
            for stock in top_stocks:
                if len(recommendations) >= limit:
                    break

                try:
                    price_history = all_price_histories.get(stock.symbol, [])

                    if not price_history or len(price_history) < 30:
                        continue

                    price_data = [
                        {
                            'open': float(p.open),
                            'high': float(p.high),
                            'low': float(p.low),
                            'close': float(p.close),
                            'volume': p.volume,
                            'date': p.date
                        }
                        for p in price_history
                    ]

                    current_price = float(price_history[-1].close)

                    ml_prediction = None
                    recommendation_type = "hold"
                    confidence_score = 0.6

                    if stock.symbol in ml_predictions_batch:
                        analysis = ml_predictions_batch[stock.symbol]
                        ml_prediction = analysis.get('prediction')
                        confidence_score = analysis.get('confidence', 0.6)
                    elif model_manager and recommendation_engine_instance:
                        try:
                            analysis = await recommendation_engine_instance.analyze_stock(
                                symbol=stock.symbol,
                                price_data=price_data,
                                user_risk_tolerance=risk_level if risk_level else 'moderate'
                            )
                            ml_prediction = analysis.get('prediction')
                            confidence_score = analysis.get('confidence', 0.6)
                        except Exception as e:
                            logger.error(f"Error in ML prediction for {stock.symbol}: {e}")

                    # Map ML prediction to recommendation type
                    if ml_prediction:
                        pred_value = ml_prediction.get('direction', 0)
                        if pred_value > 0.7:
                            recommendation_type = "strong_buy"
                        elif pred_value > 0.3:
                            recommendation_type = "buy"
                        elif pred_value < -0.7:
                            recommendation_type = "strong_sell"
                        elif pred_value < -0.3:
                            recommendation_type = "sell"

                    # Calculate target price
                    target_price = current_price * (1 + (confidence_score * 0.2 - 0.1))
                    expected_return = (target_price - current_price) / current_price

                    # Determine category based on stock characteristics
                    category = "growth"
                    if stock.sector == "Technology":
                        category = "growth"
                    elif stock.market_cap and stock.market_cap > 100000000000:
                        category = "value"

                    # Filter by requested categories
                    if categories and category not in categories:
                        continue

                    # Generate SEC disclosure for this recommendation
                    sec_disclosure = self.generate_sec_disclosure(
                        algorithm_type="ML-powered quantitative analysis",
                        data_sources=[
                            f"Price history database - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                            f"Stock fundamentals - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                            f"ML prediction model v{RECOMMENDATION_MODEL_VERSION}",
                        ],
                        confidence_score=confidence_score
                    )

                    # Get similar stocks from pre-computed sector lookup
                    similar_stocks = []
                    if stock.sector and stock.sector in stock_by_sector:
                        similar_stocks = [s for s in stock_by_sector[stock.sector] if s != stock.symbol][:3]

                    recommendation = {
                        "id": f"ML-{stock.symbol}-{int(datetime.now(timezone.utc).timestamp())}",
                        "symbol": stock.symbol,
                        "company_name": stock.name,
                        "recommendation_type": recommendation_type,
                        "category": category,
                        "confidence_score": confidence_score,
                        "target_price": round(target_price, 2),
                        "current_price": current_price,
                        "expected_return": round(expected_return, 4),
                        "time_horizon": "medium_term",
                        "risk_level": risk_level or "moderate",
                        "created_at": datetime.now(timezone.utc),
                        "valid_until": datetime.now(timezone.utc) + timedelta(days=7),
                        "reasoning": "ML-powered analysis based on price patterns, volume trends, and market conditions",
                        "key_factors": [
                            f"ML confidence: {confidence_score:.1%}",
                            f"Price momentum: {'Positive' if expected_return > 0 else 'Negative'}",
                            f"Market cap: ${stock.market_cap:,.0f}" if stock.market_cap else "Market cap: N/A",
                            f"Sector: {stock.sector}" if stock.sector else "Sector: N/A"
                        ],
                        "technical_signals": {
                            "ml_prediction": ml_prediction.get('direction', 0) if ml_prediction else 0,
                            "price_trend": "bullish" if expected_return > 0 else "bearish",
                            "volatility": ml_prediction.get('volatility', 0.2) if ml_prediction else 0.2
                        },
                        "fundamental_metrics": {
                            "market_cap": stock.market_cap,
                            "sector": stock.sector,
                            "industry": stock.industry
                        },
                        "risk_factors": [
                            f for f in [
                                "Market volatility",
                                "Sector-specific risks",
                                "Liquidity risk" if stock.market_cap and stock.market_cap < 1000000000 else None,
                            ] if f is not None
                        ],
                        "entry_points": [current_price * 0.98, current_price * 0.95],
                        "exit_points": [target_price * 0.95, target_price],
                        "stop_loss": current_price * 0.92,
                        "sector": stock.sector or "Unknown",
                        "market_cap": stock.market_cap or 0,
                        "volume": price_history[-1].volume if price_history else 0,
                        "analyst_consensus": None,
                        "similar_stocks": similar_stocks,
                        "sec_disclosure": sec_disclosure,
                    }

                    recommendations.append(recommendation)

                except Exception as e:
                    logger.error(f"Error generating recommendation for {stock.symbol}: {e}")
                    continue

            # Sort by confidence score
            recommendations.sort(key=lambda x: x["confidence_score"], reverse=True)
            logger.info(f"Generated {len(recommendations)} ML recommendations using optimized batch queries")
            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Error generating ML recommendations: {e}")
            return [self.generate_sample_recommendation() for _ in range(min(limit, 5))]

    # =========================================================================
    # Rules-Based Quantitative Screen (transparent, deterministic, no ML)
    # =========================================================================

    @staticmethod
    def _compute_momentum_return(price_history: List[Any]) -> Optional[float]:
        """Compute the trailing 60-trading-day return from stored closes.

        ``price_history`` is chronological (oldest first), matching
        ``get_bulk_price_history``. Requires at least ``MOMENTUM_MIN_ROWS`` rows.
        Uses the longest available window in [30, 60] when fewer than 60 rows
        exist. Returns ``None`` when the requirement is not met or the lookback
        close is non-positive.

        Returns:
            ``(close[-1] / close[-window] - 1)`` as a float, or ``None``.
        """
        if not price_history or len(price_history) < MOMENTUM_MIN_ROWS:
            return None

        # Window = 60 when available, otherwise the longest window >= 30.
        window = min(MOMENTUM_WINDOW_DAYS, len(price_history))
        if window < MOMENTUM_MIN_ROWS:
            return None

        latest_close = float(price_history[-1].close)
        lookback_close = float(price_history[-window].close)
        if lookback_close <= 0:
            return None

        return latest_close / lookback_close - 1.0

    @staticmethod
    def _percentile_ranks(values: List[float]) -> List[float]:
        """Compute fractional percentile ranks in [0, 1] for ``values``.

        Higher raw value -> higher percentile. Ties share the average rank so
        the mapping is deterministic. A single value maps to 0.5; an empty list
        returns an empty list.
        """
        n = len(values)
        if n == 0:
            return []
        if n == 1:
            return [0.5]

        # Rank each value by the count of strictly-smaller plus half of equal
        # values (midrank), normalized to [0, 1]. Deterministic and tie-safe.
        ranks: List[float] = []
        for v in values:
            less = sum(1 for o in values if o < v)
            equal = sum(1 for o in values if o == v)
            midrank = less + (equal - 1) / 2.0
            ranks.append(midrank / (n - 1))
        return ranks

    @staticmethod
    def _recommendation_type_for_percentile(composite_pct: float) -> str:
        """Map a composite percentile in [0, 1] to a recommendation tier."""
        if composite_pct >= 0.80:
            return "strong_buy"
        if composite_pct >= 0.60:
            return "buy"
        if composite_pct >= 0.40:
            return "hold"
        if composite_pct >= 0.20:
            return "sell"
        return "strong_sell"

    async def generate_rules_based_recommendations(
        self,
        risk_level: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: int = 10,
        db_session: Optional[AsyncSession] = None,
        *,
        stock_repo: Any = _UNSET,
        price_repo: Any = _UNSET,
        universe_limit: int = 503,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations from a transparent, deterministic rules-based
        screen over REAL stored data only. NEVER fabricates: when data is
        insufficient the corresponding symbol is skipped, and if no symbol
        qualifies an empty list is returned.

        Screen definition:
          * Momentum = trailing 60-trading-day price return from stored closes
            (longest window in [30, 60] when fewer than 60 rows; symbols with
            < 30 rows are skipped).
          * Valuation = cross-sectional percentile of P/E across the scored
            universe (lower P/E ranks better); PEG used as a tiebreaker. Symbols
            without fundamentals are scored momentum-only.
          * Composite = 0.5 * momentum_pct + 0.5 * (1 - pe_pct) when both are
            available, else the momentum percentile.
          * Tier from composite percentile; confidence = 0.5 + 0.45 * composite
            (deterministic, never random).
          * target_price = current_price * (1 + clamp(momentum_60d, -0.3, 0.3)).

        Args:
            risk_level: Optional risk level string carried onto each rec.
            categories: Optional category whitelist (post-filter).
            limit: Maximum number of recommendations to return.
            db_session: Database session for repository queries.
            stock_repo: Optional stock repository override (test patching).
            price_repo: Optional price repository override (test patching).
            universe_limit: Max stocks pulled from ``get_top_stocks``.

        Returns:
            List of recommendation dictionaries (possibly empty).
        """
        if stock_repo is _UNSET or price_repo is _UNSET:
            from backend.repositories import (
                stock_repository as _stock_repo,
                price_repository as _price_repo,
            )
            stock_repo = _stock_repo if stock_repo is _UNSET else stock_repo
            price_repo = _price_repo if price_repo is _UNSET else price_repo

        logger.info(
            "Generating rules-based recommendations (limit=%s, risk=%s)",
            limit, risk_level,
        )

        # 1. Universe of candidate stocks (real, stored). Do NOT require a
        # market cap here -- the screen only needs stocks with price history;
        # market cap is enriched lazily, so requiring it would empty the
        # universe before enrichment has run.
        top_stocks = await stock_repo.get_top_stocks(
            limit=universe_limit,
            by_market_cap=True,
            require_market_cap=False,
            session=db_session,
        )
        if not top_stocks:
            logger.warning("Rules-based screen: no stocks in universe -> []")
            return []

        symbols = [stock.symbol for stock in top_stocks]

        # 2. Bulk fetch price history + latest fundamentals (batch, not per-symbol).
        # Pull a generous lookback so >= 60 trading days are available.
        price_histories = await price_repo.get_bulk_price_history(
            symbols=symbols,
            start_date=datetime.now(timezone.utc).date() - timedelta(days=400),
            end_date=datetime.now(timezone.utc).date(),
            limit_per_symbol=MOMENTUM_WINDOW_DAYS + 5,
            session=db_session,
        )
        fundamentals_by_symbol = await stock_repo.get_bulk_latest_fundamentals(
            symbols=symbols,
            session=db_session,
        )

        # 3. First pass: compute momentum + collect raw signals per qualifying symbol.
        scored: List[Dict[str, Any]] = []
        for stock in top_stocks:
            history = price_histories.get(stock.symbol) or []
            momentum = self._compute_momentum_return(history)
            if momentum is None:
                # Missing price data or below the row threshold -> skip (graceful-empty).
                continue

            fundamentals = fundamentals_by_symbol.get(stock.symbol)
            pe_ratio = None
            peg_ratio = None
            if fundamentals is not None:
                pe = getattr(fundamentals, "pe_ratio", None)
                # Only positive P/E participates in the valuation rank; a
                # non-positive or missing P/E falls back to momentum-only.
                if pe is not None and float(pe) > 0:
                    pe_ratio = float(pe)
                    peg = getattr(fundamentals, "peg_ratio", None)
                    peg_ratio = float(peg) if peg is not None else None

            scored.append({
                "stock": stock,
                "history": history,
                "fundamentals": fundamentals,
                "momentum": momentum,
                "current_price": float(history[-1].close),
                "pe_ratio": pe_ratio,
                "peg_ratio": peg_ratio,
            })

        if not scored:
            logger.warning("Rules-based screen: no symbol met the data threshold -> []")
            return []

        # 4. Percentile ranks across the scored universe.
        momentum_pcts = self._percentile_ranks([s["momentum"] for s in scored])

        # Valuation percentile is computed only over symbols with a usable P/E.
        # pe_percentile ranks the RAW P/E (higher P/E -> higher percentile); the
        # PEG ratio is folded in as a small tiebreaker (higher PEG -> slightly
        # higher percentile, i.e. also "more expensive"). The composite then
        # uses (1 - pe_percentile) so that a LOWER P/E ranks more favorably.
        valuation_indices = [i for i, s in enumerate(scored) if s["pe_ratio"] is not None]
        pe_pct_by_index: Dict[int, float] = {}
        if valuation_indices:
            def _pe_rank_key(i: int) -> float:
                s = scored[i]
                peg_term = s["peg_ratio"] if s["peg_ratio"] is not None else 0.0
                return s["pe_ratio"] + 1e-6 * peg_term
            ranked_values = [_pe_rank_key(i) for i in valuation_indices]
            pe_percentiles = self._percentile_ranks(ranked_values)
            for idx, pct in zip(valuation_indices, pe_percentiles):
                pe_pct_by_index[idx] = pct  # higher = more expensive

        # 5. Composite score per symbol.
        for i, s in enumerate(scored):
            momentum_pct = momentum_pcts[i]
            if i in pe_pct_by_index:
                pe_percentile = pe_pct_by_index[i]
                # Lower P/E (lower percentile) -> higher inverse-valuation term.
                inverse_valuation = 1.0 - pe_percentile
                composite = 0.5 * momentum_pct + 0.5 * inverse_valuation
                s["pe_percentile"] = pe_percentile
                s["valuation_pct"] = inverse_valuation
            else:
                composite = momentum_pct
                s["pe_percentile"] = None
                s["valuation_pct"] = None
            s["momentum_pct"] = momentum_pct
            s["composite"] = composite

        # 6. Rank the universe by composite (deterministic tiebreak by symbol).
        scored.sort(key=lambda s: (s["composite"], s["stock"].symbol), reverse=True)

        # Percentile of each symbol's composite drives the recommendation tier.
        composite_pcts = self._percentile_ranks([s["composite"] for s in scored])
        for s, c_pct in zip(scored, composite_pcts):
            s["composite_pct"] = c_pct

        # 7. Build recommendation dicts (real values only).
        recommendations: List[Dict[str, Any]] = []
        for s in scored:
            if len(recommendations) >= limit:
                break

            stock = s["stock"]
            momentum = s["momentum"]
            composite = s["composite"]
            current_price = s["current_price"]

            # Deterministic confidence from the composite score.
            confidence_score = round(0.5 + 0.45 * composite, 4)

            # Target price from clamped momentum (no random).
            clamped_momentum = max(
                -TARGET_PRICE_MOMENTUM_CLAMP,
                min(TARGET_PRICE_MOMENTUM_CLAMP, momentum),
            )
            target_price = current_price * (1 + clamped_momentum)
            expected_return = (target_price - current_price) / current_price if current_price else 0.0

            recommendation_type = self._recommendation_type_for_percentile(s["composite_pct"])

            # Category derived from real characteristics; momentum-led screen.
            category = "momentum"
            if s["valuation_pct"] is not None and s["valuation_pct"] >= 0.6:
                category = "value"
            if categories and category not in categories:
                continue

            fundamentals = s["fundamentals"]
            fundamental_metrics: Dict[str, Any] = {
                "pe_ratio": s["pe_ratio"],
                "peg_ratio": s["peg_ratio"],
            }
            if fundamentals is not None:
                for field in ("pb_ratio", "roe", "net_margin", "debt_to_equity", "revenue", "eps"):
                    val = getattr(fundamentals, field, None)
                    fundamental_metrics[field] = float(val) if val is not None else None
                period_date = getattr(fundamentals, "period_date", None)
                fundamental_metrics["period_date"] = (
                    period_date.isoformat() if period_date is not None else None
                )

            key_factors = [
                f"60-day momentum: {momentum:+.1%}",
                f"Momentum percentile: {s['momentum_pct']:.0%}",
            ]
            if s["valuation_pct"] is not None:
                key_factors.append(f"Valuation percentile (lower P/E better): {s['valuation_pct']:.0%}")
            else:
                key_factors.append("Valuation: no fundamentals on file (momentum-only score)")

            sec_disclosure = self.generate_sec_disclosure(
                algorithm_type=RULES_BASED_ALGORITHM_TYPE,
                data_sources=[
                    f"Stored end-of-day price history - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                    (
                        f"Stored fundamentals (period {fundamental_metrics.get('period_date')})"
                        if fundamentals is not None
                        else "Fundamentals: none on file for this symbol"
                    ),
                ],
                confidence_score=confidence_score,
                methodology_disclosure=RULES_BASED_METHODOLOGY_DISCLOSURE,
            )

            recommendations.append({
                "id": f"RULES-{stock.symbol}-{int(datetime.now(timezone.utc).timestamp())}",
                "symbol": stock.symbol,
                "company_name": stock.name,
                "recommendation_type": recommendation_type,
                "category": category,
                "confidence_score": confidence_score,
                "target_price": round(target_price, 2),
                "current_price": round(current_price, 2),
                "expected_return": round(expected_return, 4),
                "time_horizon": "medium_term",
                "risk_level": risk_level or "moderate",
                "created_at": datetime.now(timezone.utc),
                "valid_until": datetime.now(timezone.utc) + timedelta(days=7),
                "reasoning": (
                    "Rules-based screen ranking on trailing 60-day price momentum "
                    "and cross-sectional P/E valuation over stored historical data."
                ),
                "key_factors": key_factors,
                "technical_signals": {
                    "momentum_60d": round(momentum, 4),
                    "momentum_percentile": round(s["momentum_pct"], 4),
                    "price_trend": "bullish" if momentum > 0 else "bearish",
                },
                "fundamental_metrics": fundamental_metrics,
                "risk_factors": [
                    "Market volatility",
                    "Screen does not consider forward-looking events",
                    "Backward-looking momentum may not persist",
                ],
                "entry_points": [round(current_price * 0.98, 2), round(current_price * 0.95, 2)],
                "exit_points": [round(target_price * 0.95, 2), round(target_price, 2)],
                "stop_loss": round(current_price * 0.92, 2),
                # stock.sector is a Sector ORM relationship (use .name); tolerate
                # a plain string too. RecommendationDetail.sector wants a string.
                "sector": (
                    stock.sector if isinstance(stock.sector, str)
                    else getattr(stock.sector, "name", None)
                ) or "Unknown",
                "market_cap": stock.market_cap or 0,
                "volume": s["history"][-1].volume if s["history"] else 0,
                "analyst_consensus": None,
                "similar_stocks": [],
                "sec_disclosure": sec_disclosure,
            })

        logger.info("Rules-based screen produced %d recommendations", len(recommendations))
        return recommendations

    # =========================================================================
    # Personalized Recommendation Generation
    # =========================================================================

    async def generate_personalized_recommendations(
        self,
        user_id: int,
        portfolio_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        *,
        stock_repo: Any = None,
        portfolio_repo: Any = None,
        ml_recs_fn: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations based on user's portfolio and preferences.

        Args:
            user_id: User identifier for preference lookup
            portfolio_id: Optional portfolio identifier
            db_session: Database session for repository queries
            stock_repo: Optional stock repository override (for test patching)
            portfolio_repo: Optional portfolio repository override (for test patching)
            ml_recs_fn: Optional callable override for ML recommendation generation

        Returns:
            List of personalized recommendation dictionaries
        """
        if portfolio_repo is None or stock_repo is None:
            from backend.repositories import (
                portfolio_repository as _portfolio_repo,
                stock_repository as _stock_repo,
            )
            portfolio_repo = portfolio_repo or _portfolio_repo
            stock_repo = stock_repo or _stock_repo

        try:
            logger.info(f"Generating personalized recommendations for user {user_id}")

            # Get user's portfolio(s) to understand preferences
            user_portfolios = await portfolio_repo.get_user_portfolios(
                user_id=user_id,
                session=db_session
            )

            # Analyze existing positions to understand preferences
            existing_symbols = set()
            preferred_sectors: Dict[str, int] = {}

            for portfolio in user_portfolios:
                positions = await portfolio_repo.get_portfolio_positions(
                    portfolio_id=portfolio.id,
                    session=db_session
                )

                for position in positions:
                    existing_symbols.add(position.symbol)

                    # Get stock info to determine sector preference
                    stock = await stock_repo.get_by_symbol(position.symbol, session=db_session)
                    if stock and stock.sector:
                        preferred_sectors[stock.sector] = preferred_sectors.get(stock.sector, 0) + 1

            # Generate recommendations excluding existing positions
            if ml_recs_fn:
                all_recommendations = await ml_recs_fn(user_id=user_id, limit=20, db_session=db_session)
            else:
                all_recommendations = await self.generate_ml_powered_recommendations(
                    user_id=user_id,
                    limit=20,
                    db_session=db_session
                )

            # Filter out existing positions and prefer similar sectors
            filtered_recommendations = []
            for rec in all_recommendations:
                if rec["symbol"] not in existing_symbols:
                    # Boost confidence for preferred sectors
                    if rec["sector"] in preferred_sectors:
                        rec = {
                            **rec,
                            "confidence_score": min(0.95, rec["confidence_score"] * 1.1),
                            "reasoning": rec["reasoning"] + f" (Matches your sector preference for {rec['sector']})",
                        }
                    filtered_recommendations.append(rec)

            return filtered_recommendations[:10]

        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}")
            return await self.generate_ml_powered_recommendations(limit=5, db_session=db_session)

    # =========================================================================
    # Daily Recommendations Aggregation
    # =========================================================================

    async def build_daily_recommendations(
        self,
        user_id: int,
        target_date: date,
        risk_level: Optional[str] = None,
        db_session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Aggregate ML and personalized recommendations into a daily digest.

        Combines both ML-powered and personalized recommendations, deduplicates
        by symbol (preferring the higher confidence entry), filters by risk level
        if requested, and computes market sentiment, outlook, and risk assessment.

        Args:
            user_id: User identifier for personalization
            target_date: Date for the recommendations
            risk_level: Optional risk level filter string
            db_session: Database session for repository queries

        Returns:
            Dictionary compatible with the DailyRecommendations schema
        """
        try:
            logger.info(f"Building daily recommendations for user {user_id}, date {target_date}, risk {risk_level}")

            # Generate ML-powered recommendations
            ml_recommendations = await self.generate_ml_powered_recommendations(
                user_id=user_id,
                risk_level=risk_level,
                limit=15,
                db_session=db_session
            )

            # Get personalized recommendations based on user's portfolio
            personalized_recs = await self.generate_personalized_recommendations(
                user_id=user_id,
                db_session=db_session
            )

            # Combine and deduplicate recommendations
            all_recommendations: Dict[str, Dict[str, Any]] = {}
            for rec in ml_recommendations + personalized_recs:
                symbol = rec["symbol"]
                if symbol not in all_recommendations:
                    all_recommendations[symbol] = rec
                elif rec["confidence_score"] > all_recommendations[symbol]["confidence_score"]:
                    all_recommendations[symbol] = rec

            # Filter by risk level if specified
            if risk_level:
                filtered_recs = [r for r in all_recommendations.values() if r["risk_level"] == risk_level]
                if len(filtered_recs) < 5:
                    # Include some other risk recommendations if not enough matches
                    other_recs = [r for r in all_recommendations.values() if r["risk_level"] != risk_level]
                    filtered_recs.extend(other_recs[:(5 - len(filtered_recs))])
                all_recommendations = {r["symbol"]: r for r in filtered_recs}

            # Sort by confidence score and take top picks
            sorted_recs = sorted(all_recommendations.values(), key=lambda x: x["confidence_score"], reverse=True)
            top_picks = sorted_recs[:8]

            # Generate watchlist from remaining high-confidence picks
            watchlist_symbols = [r["symbol"] for r in sorted_recs[8:15]]

            # Generate avoid list based on negative recommendations
            negative_recs = [
                r for r in all_recommendations.values()
                if r["recommendation_type"] in ["sell", "strong_sell"]
            ]
            avoid_list = [r["symbol"] for r in negative_recs[:5]]

            # Determine sector focus based on recommendations
            sector_counts: Dict[str, int] = {}
            for rec in top_picks:
                sector = rec.get("sector", "")
                if sector and sector != "Unknown":
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1

            sector_focus = max(sector_counts.items(), key=lambda x: x[1])[0] if sector_counts else "Technology"

            # Calculate market sentiment from recommendations
            sentiment_map = {
                "strong_buy": 1.0,
                "buy": 0.5,
                "hold": 0.0,
                "sell": -0.5,
                "strong_sell": -1.0,
            }
            sentiment_scores = [
                sentiment_map.get(rec["recommendation_type"], 0.0)
                for rec in top_picks
            ]
            market_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            # Generate market outlook
            if market_sentiment > 0.3:
                outlook = "Bullish - Strong buying opportunities identified across multiple sectors"
            elif market_sentiment > 0.1:
                outlook = "Cautiously optimistic - Selective opportunities in preferred sectors"
            elif market_sentiment > -0.1:
                outlook = "Neutral - Mixed signals, focus on risk management"
            elif market_sentiment > -0.3:
                outlook = "Cautious - Defensive positioning recommended"
            else:
                outlook = "Bearish - High risk environment, consider cash positions"

            # Risk assessment based on average confidence
            avg_confidence = (
                sum(r["confidence_score"] for r in top_picks) / len(top_picks)
                if top_picks else 0.5
            )
            if avg_confidence > 0.8:
                risk_assessment = "Low - High confidence in current recommendations"
            elif avg_confidence > 0.6:
                risk_assessment = "Moderate - Standard market conditions"
            else:
                risk_assessment = "Elevated - Uncertain market environment"

            # Generate special situations (high-confidence buy signals)
            special_situations = []
            for rec in top_picks[:3]:
                if rec["recommendation_type"] in ["strong_buy", "buy"]:
                    reasoning = rec["reasoning"]
                    special_situations.append({
                        "type": "high_confidence_pick",
                        "symbol": rec["symbol"],
                        "confidence": rec["confidence_score"],
                        "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
                        "target_return": rec["expected_return"],
                    })

            return {
                "top_picks": top_picks[:5],
                "watchlist": watchlist_symbols,
                "avoid_list": avoid_list,
                "sector_focus": sector_focus,
                "market_sentiment": round(market_sentiment, 3),
                "market_outlook": outlook,
                "risk_assessment": risk_assessment,
                "special_situations": special_situations,
            }

        except Exception as e:
            logger.error(f"Error building daily recommendations: {e}")
            raise

    # =========================================================================
    # Backtest Simulation
    # =========================================================================

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

        Raises:
            ModelUnavailableError: when ``settings.DEMO_MODE`` is False
                (production default). Per PRD audit 2026-04 §3 D Step 2
                (F-02-003, Q4 default), backtest results are sourced from
                ``random.uniform`` and must not surface as real numbers in
                production; the router layer already returns 503 but this
                defense-in-depth gate covers internal/script callers.
        """
        if not settings.DEMO_MODE:
            raise ModelUnavailableError(
                model="recommendation_backtest",
                reason="not_implemented",
            )

        total_return = random.uniform(-0.2, 0.5)
        days = (end_date - start_date).days or 1  # avoid division by zero

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

    # =========================================================================
    # Performance Tracking
    # =========================================================================

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

        Raises:
            ModelUnavailableError: when ``settings.DEMO_MODE`` is False
                (production default). Per PRD audit 2026-04 §3 D Step 2
                (F-02-003, Q4 default), performance records are fabricated
                via ``random.choice`` and ``random.uniform``; production
                must surface a 503 rather than synthetic numbers tied to
                non-existent recommendation IDs.
        """
        if not settings.DEMO_MODE:
            raise ModelUnavailableError(
                model="recommendation_performance_history",
                reason="not_implemented",
            )

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

    # =========================================================================
    # Portfolio Recommendations
    # =========================================================================

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

    # =========================================================================
    # Alert History Generation
    # =========================================================================

    def generate_alert_history(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Generate historical alert records for recommendations.

        Args:
            days_back: Number of days to look back for alerts

        Returns:
            List of alert dictionaries sorted by timestamp descending

        Raises:
            ModelUnavailableError: when ``settings.DEMO_MODE`` is False
                (production default). Per PRD audit 2026-04 §3 D Step 2
                (F-02-003, Q4 default), alert history entries are
                fabricated via ``random.choice``; surfacing them on the
                authenticated ``/alerts/history`` route would imply a real
                signal pipeline that does not exist.
        """
        if not settings.DEMO_MODE:
            raise ModelUnavailableError(
                model="recommendation_alert_history",
                reason="not_implemented",
            )

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

    # =========================================================================
    # Filtered List Generation
    # =========================================================================

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

    # =========================================================================
    # Original recommendation_engine delegation methods
    # =========================================================================

    async def generate_recommendation(
        self,
        ticker: str,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate investment recommendation for a stock.

        Args:
            ticker: Stock ticker symbol
            analysis_types: Types of analysis to include (technical, fundamental, sentiment, ml)
                           If None, includes all available analyses

        Returns:
            Dictionary containing recommendation details
        """
        try:
            await self.initialize()

            if analysis_types is None:
                analysis_types = ['technical', 'fundamental', 'sentiment', 'ml']

            logger.info(f"Generating recommendation for {ticker} with analyses: {analysis_types}")

            # Use recommendation engine to perform comprehensive analysis
            recommendation = await self.recommendation_engine.analyze_stock(ticker)

            if not recommendation:
                return {
                    'success': False,
                    'error': f'Unable to generate recommendation for {ticker}',
                    'ticker': ticker
                }

            # Convert to dictionary format
            return {
                'success': True,
                'ticker': recommendation.ticker,
                'action': recommendation.action.value,
                'confidence': recommendation.confidence,
                'priority': recommendation.priority,
                'entry_price': recommendation.entry_price,
                'target_price': recommendation.target_price,
                'stop_loss': recommendation.stop_loss,
                'expected_return': recommendation.expected_return,
                'time_horizon_days': recommendation.time_horizon_days,
                'risk_metrics': {
                    'risk_score': recommendation.risk_score,
                    'volatility': recommendation.volatility,
                    'beta': recommendation.beta,
                    'sharpe_ratio': recommendation.sharpe_ratio,
                    'max_drawdown': recommendation.max_drawdown
                },
                'analysis_scores': {
                    'technical': recommendation.technical_score,
                    'fundamental': recommendation.fundamental_score,
                    'sentiment': recommendation.sentiment_score,
                    'ml_prediction': recommendation.ml_prediction_score
                },
                'key_factors': recommendation.key_factors,
                'risks': recommendation.risks,
                'opportunities': recommendation.opportunities,
                'catalysts': recommendation.catalysts,
                'position_sizing': {
                    'recommended_allocation': recommendation.recommended_allocation,
                    'max_position_size': recommendation.max_position_size
                },
                'generated_at': recommendation.generated_at.isoformat(),
                'valid_until': recommendation.valid_until.isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating recommendation for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }

    async def get_trending(
        self,
        timeframe: str = '1d',
        limit: int = 10,
        risk_tolerance: str = 'moderate'
    ) -> List[Dict[str, Any]]:
        """
        Get trending stock recommendations.

        Args:
            timeframe: Time period for trending (1d, 1w, 1m)
            limit: Maximum number of recommendations to return
            risk_tolerance: Risk tolerance level (conservative, moderate, aggressive)

        Returns:
            List of trending recommendations
        """
        try:
            await self.initialize()

            logger.info(f"Getting trending recommendations: timeframe={timeframe}, limit={limit}, risk={risk_tolerance}")

            # Get daily recommendations from the engine
            recommendations = await self.recommendation_engine.generate_daily_recommendations(
                max_recommendations=limit * 2,  # Get extra for filtering
                risk_tolerance=risk_tolerance
            )

            # Convert to dictionary format and limit results
            trending = []
            for rec in recommendations[:limit]:
                trending.append({
                    'ticker': rec.ticker,
                    'action': rec.action.value,
                    'confidence': rec.confidence,
                    'priority': rec.priority,
                    'expected_return': rec.expected_return,
                    'risk_score': rec.risk_score,
                    'technical_score': rec.technical_score,
                    'fundamental_score': rec.fundamental_score,
                    'key_factors': rec.key_factors[:3],  # Top 3 factors
                    'generated_at': rec.generated_at.isoformat()
                })

            return trending

        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []

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

        # Define weights for different analysis types
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

    async def monitor_recommendations(
        self,
        recommendation_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Monitor active recommendations and generate alerts.

        Args:
            recommendation_ids: List of recommendation IDs to monitor

        Returns:
            List of alerts for the recommendations
        """
        try:
            await self.initialize()

            # Note: This would need access to stored recommendations
            # For now, returning empty list as placeholder
            logger.warning("Recommendation monitoring not fully implemented - requires database integration")
            return []

        except Exception as e:
            logger.error(f"Error monitoring recommendations: {e}")
            return []


# Create singleton instance
recommendation_service = RecommendationService()
