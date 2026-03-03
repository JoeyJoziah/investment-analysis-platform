"""
Recommendation Analysis Methods

Contains all async analysis and scoring methods extracted from RecommendationService:
- generate_ml_powered_recommendations
- generate_personalized_recommendations
- build_daily_recommendations
- generate_recommendation  (single-ticker engine delegation)
- get_trending             (engine delegation)
- monitor_recommendations

This module is NOT meant to be imported directly by application code; use
backend.services.recommendation_service instead, which re-exports everything
through the RecommendationService class.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.recommendation_crud import RECOMMENDATION_MODEL_VERSION

logger = logging.getLogger(__name__)

# Sentinel to distinguish "not provided" from an explicit None
_UNSET = object()


class RecommendationAnalysisMixin:
    """
    Mixin providing async analysis methods for RecommendationService.

    Expects the host class to expose:
        self.recommendation_engine  - RecommendationEngine instance
        self.generate_sample_recommendation()
        self.generate_sec_disclosure()
        self.initialize()
    """

    # ------------------------------------------------------------------
    # ML-Powered Recommendation Generation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Personalized Recommendation Generation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Daily Recommendations Aggregation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Engine Delegation Methods
    # ------------------------------------------------------------------

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

            recommendation = await self.recommendation_engine.analyze_stock(ticker)

            if not recommendation:
                return {
                    'success': False,
                    'error': f'Unable to generate recommendation for {ticker}',
                    'ticker': ticker
                }

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

            recommendations = await self.recommendation_engine.generate_daily_recommendations(
                max_recommendations=limit * 2,
                risk_tolerance=risk_tolerance
            )

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
                    'key_factors': rec.key_factors[:3],
                    'generated_at': rec.generated_at.isoformat()
                })

            return trending

        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []

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

            logger.warning("Recommendation monitoring not fully implemented - requires database integration")
            return []

        except Exception as e:
            logger.error(f"Error monitoring recommendations: {e}")
            return []


__all__ = [
    "RecommendationAnalysisMixin",
]
