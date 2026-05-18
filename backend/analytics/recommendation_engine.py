"""
World-Class Investment Recommendation Engine
Combines all analysis types to generate actionable recommendations

This module is the public surface of the recommendation subsystem.  All
implementation details have been extracted into focused sub-modules:

  recommendation_types.py    - RecommendationAction, StockRecommendation
  recommendation_scoring.py  - scoring helpers, price targets, confidence
  recommendation_ranking.py  - filtering, ranking, portfolio optimisation
  recommendation_optimized.py - OptimizedRecommendationEngine + singleton

Every name that was previously defined here is still importable from here,
so all existing ``from backend.analytics.recommendation_engine import X``
statements continue to work without modification.
"""

import asyncio
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
import logging

# ---------------------------------------------------------------------------
# Re-export types so callers remain unaffected
# ---------------------------------------------------------------------------
from backend.analytics.recommendation_types import (   # noqa: F401
    RecommendationAction,
    StockRecommendation,
)

# ---------------------------------------------------------------------------
# Re-export optimised engine so callers remain unaffected
# ---------------------------------------------------------------------------
from backend.analytics.recommendation_optimized import (  # noqa: F401
    OptimizedRecommendationEngine,
    get_optimized_recommendation_engine,
)

# ---------------------------------------------------------------------------
# Internal helpers (used inside RecommendationEngine methods)
# ---------------------------------------------------------------------------
from backend.analytics.recommendation_scoring import (
    normalize_score,
    determine_action,
    calculate_confidence,
    calculate_price_targets,
    determine_time_horizon,
    extract_key_factors,
    identify_risks,
    identify_opportunities,
    find_catalysts,
    calculate_position_sizing,
    calculate_priority,
)
from backend.analytics.recommendation_ranking import (
    should_recommend,
    rank_recommendations,
    optimize_recommendations,
    generate_summary,
    get_top_sectors,
)

from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.models.ml_models import ModelManager, PredictionResult
from backend.data_ingestion.market_scanner import MarketScanner
from backend.utils.risk_manager import RiskManager
from backend.utils.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Master recommendation engine that orchestrates all analysis.

    Heavy lifting (scoring, ranking, portfolio optimisation) is delegated to
    the sub-modules imported above; this class owns the async coordination
    and the sub-engine lifecycle.
    """

    def __init__(self):
        self.technical_engine = TechnicalAnalysisEngine()
        self.fundamental_engine = FundamentalAnalysisEngine()
        self.sentiment_engine = SentimentAnalysisEngine()
        self.model_manager = ModelManager()
        self.market_scanner = MarketScanner()
        self.risk_manager = RiskManager()
        self.portfolio_optimizer = PortfolioOptimizer()

        # F-09-008: source from the fundamental engine so the two
        # stay in lockstep instead of drifting silently.
        # F-09-009: portfolio size is now operator-controlled.
        self.risk_free_rate = self.fundamental_engine.risk_free_rate
        self.portfolio_size = float(os.getenv("DEFAULT_PORTFOLIO_SIZE", "100000"))

        self.thresholds = {
            'strong_buy':  0.8,
            'buy':         0.6,
            'hold':        0.4,
            'sell':        0.2,
            'strong_sell': 0.0,
        }

    async def initialize(self):
        """Initialize all components."""
        await self.model_manager.load_models()
        await self.market_scanner.initialize()
        logger.info("Recommendation engine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_daily_recommendations(
        self,
        max_recommendations: int = 50,
        risk_tolerance: str = 'moderate',
        sectors: Optional[List[str]] = None,
        market_cap_range: Optional[Tuple[float, float]] = None,
    ) -> List[StockRecommendation]:
        """Generate daily recommendations for all stocks."""
        logger.info("Starting daily recommendation generation...")

        candidates = await self.market_scanner.scan_market(
            sectors=sectors,
            market_cap_range=market_cap_range,
            max_stocks=500,
        )
        logger.info(f"Found {len(candidates)} candidate stocks")

        recommendations = []
        batch_size = 20
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_tasks = [self.analyze_stock(stock['ticker'], stock) for stock in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing stock: {result}")
                    continue
                if result and self._should_recommend(result, risk_tolerance):
                    recommendations.append(result)

        ranked_recommendations = self._rank_recommendations(recommendations)

        optimized_recommendations = await self._optimize_recommendations(
            ranked_recommendations[: max_recommendations * 2],
            risk_tolerance,
        )

        final_recommendations = optimized_recommendations[:max_recommendations]
        logger.info(f"Generated {len(final_recommendations)} recommendations")
        return final_recommendations

    async def analyze_stock(
        self,
        ticker: str,
        market_data: Optional[Dict] = None,
    ) -> Optional[StockRecommendation]:
        """Comprehensive analysis of a single stock."""
        try:
            logger.info(f"Analyzing {ticker}...")

            stock_data = await self._fetch_stock_data(ticker, market_data)
            if not stock_data:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            analysis_tasks = [
                self._run_technical_analysis(stock_data),
                self._run_fundamental_analysis(stock_data),
                self._run_sentiment_analysis(ticker, stock_data),
                self._run_ml_predictions(ticker, stock_data),
            ]
            results = await asyncio.gather(*analysis_tasks)

            technical_analysis  = results[0]
            fundamental_analysis = results[1]
            sentiment_analysis   = results[2]
            ml_predictions       = results[3]

            risk_metrics = await self._calculate_risk_metrics(stock_data, ml_predictions)

            return self._generate_recommendation(
                ticker=ticker,
                stock_data=stock_data,
                technical_analysis=technical_analysis,
                fundamental_analysis=fundamental_analysis,
                sentiment_analysis=sentiment_analysis,
                ml_predictions=ml_predictions,
                risk_metrics=risk_metrics,
            )

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    async def monitor_recommendations(
        self,
        active_recommendations: List[StockRecommendation],
    ) -> List[Dict]:
        """Monitor active recommendations and generate alerts."""
        alerts = []

        for rec in active_recommendations:
            current_data = await self.market_scanner.get_stock_data(rec.ticker)
            if not current_data:
                continue

            current_price = current_data.get('current_price', 0)

            if current_price <= rec.stop_loss:
                alerts.append({
                    'type':    'stop_loss',
                    'ticker':  rec.ticker,
                    'message': f"{rec.ticker} hit stop loss at ${current_price:.2f}",
                    'action':  'sell',
                    'urgency': 'high',
                })
            elif current_price >= rec.target_price:
                alerts.append({
                    'type':    'target_reached',
                    'ticker':  rec.ticker,
                    'message': f"{rec.ticker} reached target at ${current_price:.2f}",
                    'action':  'consider_profit_taking',
                    'urgency': 'medium',
                })
            elif datetime.now(timezone.utc) > rec.valid_until:
                alerts.append({
                    'type':    'recommendation_expired',
                    'ticker':  rec.ticker,
                    'message': f"{rec.ticker} recommendation needs refresh",
                    'action':  'reanalyze',
                    'urgency': 'low',
                })

            price_change = (current_price - rec.entry_price) / rec.entry_price
            if abs(price_change) > 0.1:
                alerts.append({
                    'type':    'significant_move',
                    'ticker':  rec.ticker,
                    'message': f"{rec.ticker} moved {price_change * 100:.1f}% since recommendation",
                    'action':  'review_position',
                    'urgency': 'medium',
                })

        return alerts

    async def generate_report(
        self,
        recommendations: List[StockRecommendation],
        format: str = 'json',
    ) -> Any:
        """Generate recommendation report in various formats."""
        if format == 'json':
            return {
                'generated_at':         datetime.now(timezone.utc).isoformat(),
                'recommendation_count': len(recommendations),
                'recommendations':      [rec.to_dict() for rec in recommendations],
                'summary':              generate_summary(recommendations),
            }
        elif format == 'pdf':
            pass
        elif format == 'excel':
            pass
        return None

    # ------------------------------------------------------------------
    # Internal - data fetching
    # ------------------------------------------------------------------

    async def _fetch_stock_data(
        self,
        ticker: str,
        market_data: Optional[Dict],
    ) -> Optional[Dict]:
        """Fetch all required data for analysis."""
        if not market_data:
            market_data = await self.market_scanner.get_stock_data(ticker)

        if not market_data:
            return None

        required_fields = ['price_history', 'fundamentals', 'market_cap']
        for field in required_fields:
            if field not in market_data:
                logger.warning(f"Missing {field} for {ticker}")
                return None

        return market_data

    # ------------------------------------------------------------------
    # Internal - per-analysis runners
    # ------------------------------------------------------------------

    async def _run_technical_analysis(self, stock_data: Dict) -> Dict:
        """Run technical analysis."""
        price_df = stock_data.get('price_history')
        if price_df is None or len(price_df) < 200:
            return {}
        return self.technical_engine.analyze_stock(price_df)

    async def _run_fundamental_analysis(self, stock_data: Dict) -> Dict:
        """Run fundamental analysis."""
        financials = stock_data.get('fundamentals', {})
        market_data = {
            'market_cap': stock_data.get('market_cap', 0),
            'price':      stock_data.get('current_price', 0),
            'beta':       stock_data.get('beta', 1.0),
        }
        peer_data = stock_data.get('peer_data')
        return await self.fundamental_engine.analyze_company(
            ticker=stock_data.get('ticker'),
            financials=financials,
            market_data=market_data,
            peer_data=peer_data,
        )

    async def _run_sentiment_analysis(self, ticker: str, stock_data: Dict) -> Dict:
        """Run sentiment analysis."""
        text_data = []

        if 'news' in stock_data:
            for article in stock_data['news'][:50]:
                text_data.append({
                    'text':      f"{article.get('headline', '')} {article.get('summary', '')}",
                    'source':    'news',
                    'timestamp': article.get('datetime', datetime.now(timezone.utc)),
                })

        if 'social_mentions' in stock_data:
            for mention in stock_data['social_mentions'][:100]:
                text_data.append({
                    'text':      mention.get('text', ''),
                    'source':    mention.get('platform', 'social'),
                    'timestamp': mention.get('timestamp', datetime.now(timezone.utc)),
                })

        if 'analyst_opinions' in stock_data:
            for opinion in stock_data['analyst_opinions']:
                text_data.append({
                    'text':      opinion.get('summary', ''),
                    'source':    'analyst',
                    'timestamp': opinion.get('date', datetime.now(timezone.utc)),
                })

        if not text_data:
            return {'overall_sentiment': {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}}

        # F-09-001: ``analyze_sentiment(text: str, source: str)`` is the
        # per-text entrypoint and rejects the list-of-dicts shape this
        # method assembles. The batch entrypoint is
        # ``analyze_stock_sentiment(ticker, texts: List[str])``.
        # Adapter pattern: feed it the extracted ``text`` field, then
        # map the SentimentResult back to the legacy dict shape that
        # downstream consumers (line ~480) read via
        # ``sentiment_analysis.get('overall_sentiment', ...)``.
        result = await self.sentiment_engine.analyze_stock_sentiment(
            ticker, [item['text'] for item in text_data if item.get('text')]
        )
        return {
            'overall_sentiment': {
                'score': result.score,
                'label': result.label,
                'confidence': result.confidence,
            },
            'breakdown': result.breakdown,
            'keywords': result.keywords,
            'sources_analyzed': result.sources_analyzed,
            'timestamp': result.timestamp,
        }

    async def _run_ml_predictions(
        self, ticker: str, stock_data: Dict
    ) -> Dict[str, PredictionResult]:
        """Run ML predictions."""
        price_df = stock_data.get('price_history')

        if price_df is None or len(price_df) < 60:
            return {}

        if 'fundamentals' in stock_data:
            for key, value in stock_data['fundamentals'].items():
                if isinstance(value, (int, float)):
                    price_df[f'fundamental_{key}'] = value

        if 'sentiment_history' in stock_data:
            price_df['sentiment_score'] = stock_data['sentiment_history']

        predictions = {}
        for horizon in [5, 20, 60]:
            horizon_predictions = await self.model_manager.predict(
                ticker=ticker,
                current_data=price_df,
                horizon=horizon,
            )
            if 'ensemble' in horizon_predictions:
                predictions[f'horizon_{horizon}'] = horizon_predictions['ensemble']

        return predictions

    # ------------------------------------------------------------------
    # Internal - risk metrics
    # ------------------------------------------------------------------

    async def _calculate_risk_metrics(
        self,
        stock_data: Dict,
        ml_predictions: Dict,
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        price_history = stock_data.get('price_history')

        if price_history is None or len(price_history) < 30:
            return {
                'risk_score':  0.5,
                'volatility':  0.0,
                'beta':        1.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95':      0.0,
                'cvar_95':     0.0,
            }

        returns = price_history['close'].pct_change().dropna()

        volatility = returns.std() * np.sqrt(252)
        beta = stock_data.get('beta', 1.0)

        # F-09-008: use the engine-level risk-free rate (synced with
        # FundamentalAnalysisEngine.risk_free_rate) instead of a local
        # 0.045 constant.
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = (
            (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
            if returns.std() > 0
            else 0
        )

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        if ml_predictions:
            prediction_std = np.std([
                pred.predicted_return for pred in ml_predictions.values()
                if hasattr(pred, 'predicted_return')
            ])
        else:
            prediction_std = 0.0

        risk_components = [
            min(volatility / 0.5, 1.0),
            min(abs(max_drawdown) / 0.3, 1.0),
            min(prediction_std / 0.1, 1.0),
            max(0, 1 - sharpe_ratio / 2),
        ]
        risk_score = np.mean(risk_components)

        return {
            'risk_score':   risk_score,
            'volatility':   volatility,
            'beta':         beta,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95':       var_95,
            'cvar_95':      cvar_95,
        }

    # ------------------------------------------------------------------
    # Internal - recommendation assembly
    # ------------------------------------------------------------------

    def _generate_recommendation(
        self,
        ticker: str,
        stock_data: Dict,
        technical_analysis: Dict,
        fundamental_analysis: Dict,
        sentiment_analysis: Dict,
        ml_predictions: Dict[str, PredictionResult],
        risk_metrics: Dict[str, float],
    ) -> StockRecommendation:
        """Generate final recommendation based on all analysis."""
        current_price = stock_data.get('current_price', 0)
        if current_price == 0:
            price_history = stock_data.get('price_history')
            if price_history is not None and len(price_history) > 0:
                current_price = price_history['close'].iloc[-1]

        technical_score = normalize_score(
            technical_analysis.get('composite_score', 0), -1, 1
        )
        fundamental_score = (
            normalize_score(fundamental_analysis.get('composite_score', 50), 0, 100)
            if fundamental_analysis
            else 0.5
        )
        sentiment_score = normalize_score(
            sentiment_analysis.get('overall_sentiment', {}).get('score', 0), -1, 1
        )

        ml_scores = [
            normalize_score(pred.predicted_return, -0.2, 0.2)
            for pred in ml_predictions.values()
            if hasattr(pred, 'predicted_return')
        ]
        ml_prediction_score = float(np.mean(ml_scores)) if ml_scores else 0.5

        weights = {'technical': 0.25, 'fundamental': 0.30, 'sentiment': 0.15, 'ml_prediction': 0.30}
        overall_score = (
            weights['technical']     * technical_score
            + weights['fundamental'] * fundamental_score
            + weights['sentiment']   * sentiment_score
            + weights['ml_prediction'] * ml_prediction_score
        )

        risk_adjusted_score = overall_score * (1 - risk_metrics['risk_score'] * 0.3)

        action = determine_action(risk_adjusted_score, self.thresholds)

        confidence = calculate_confidence(
            technical_analysis, fundamental_analysis, sentiment_analysis,
            ml_predictions, risk_metrics,
        )

        price_targets = calculate_price_targets(
            current_price, ml_predictions, technical_analysis, risk_metrics
        )

        time_horizon = determine_time_horizon(action, technical_analysis, ml_predictions)

        key_factors = extract_key_factors(
            technical_analysis, fundamental_analysis, sentiment_analysis, ml_predictions
        )
        risks = identify_risks(fundamental_analysis, risk_metrics, sentiment_analysis)
        opportunities = identify_opportunities(
            fundamental_analysis, technical_analysis, sentiment_analysis
        )
        catalysts = find_catalysts(stock_data, sentiment_analysis, fundamental_analysis)

        position_sizing = calculate_position_sizing(
            confidence, risk_metrics, action,
            portfolio_size=self.portfolio_size,  # F-09-009
        )

        priority = calculate_priority(risk_adjusted_score, confidence, opportunities)

        return StockRecommendation(
            ticker=ticker,
            action=action,
            confidence=confidence,
            priority=priority,
            entry_price=current_price,
            target_price=price_targets['target'],
            stop_loss=price_targets['stop_loss'],
            expected_return=price_targets['expected_return'],
            time_horizon_days=time_horizon,
            risk_score=risk_metrics['risk_score'],
            volatility=risk_metrics['volatility'],
            beta=risk_metrics['beta'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            ml_prediction_score=ml_prediction_score,
            technical_analysis=technical_analysis,
            fundamental_analysis=fundamental_analysis,
            sentiment_analysis=sentiment_analysis,
            ml_predictions=ml_predictions,
            key_factors=key_factors,
            risks=risks,
            opportunities=opportunities,
            catalysts=catalysts,
            generated_at=datetime.now(timezone.utc),
            valid_until=datetime.now(timezone.utc) + timedelta(days=1),
            recommended_allocation=position_sizing['allocation'],
            max_position_size=position_sizing['max_size'],
        )

    # ------------------------------------------------------------------
    # Thin private shims kept for any code that calls them on the engine
    # instance directly (e.g. unit tests patching internals).
    # ------------------------------------------------------------------

    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        return normalize_score(value, min_val, max_val)

    def _determine_action(self, score: float) -> RecommendationAction:
        return determine_action(score, self.thresholds)

    def _calculate_confidence(self, technical, fundamental, sentiment, ml_predictions, risk_metrics):
        return calculate_confidence(technical, fundamental, sentiment, ml_predictions, risk_metrics)

    def _calculate_price_targets(self, current_price, ml_predictions, technical_analysis, risk_metrics):
        return calculate_price_targets(current_price, ml_predictions, technical_analysis, risk_metrics)

    def _determine_time_horizon(self, action, technical_analysis, ml_predictions):
        return determine_time_horizon(action, technical_analysis, ml_predictions)

    def _extract_key_factors(self, technical, fundamental, sentiment, ml_predictions):
        return extract_key_factors(technical, fundamental, sentiment, ml_predictions)

    def _identify_risks(self, fundamental, risk_metrics, sentiment):
        return identify_risks(fundamental, risk_metrics, sentiment)

    def _identify_opportunities(self, fundamental, technical, sentiment):
        return identify_opportunities(fundamental, technical, sentiment)

    def _find_catalysts(self, stock_data, sentiment, fundamental):
        return find_catalysts(stock_data, sentiment, fundamental)

    def _calculate_position_sizing(self, confidence, risk_metrics, action):
        return calculate_position_sizing(
            confidence, risk_metrics, action,
            portfolio_size=self.portfolio_size,  # F-09-009
        )

    def _calculate_priority(self, score, confidence, opportunities):
        return calculate_priority(score, confidence, opportunities)

    def _should_recommend(self, recommendation, risk_tolerance):
        return should_recommend(recommendation, risk_tolerance)

    def _rank_recommendations(self, recommendations):
        return rank_recommendations(recommendations)

    async def _optimize_recommendations(self, recommendations, risk_tolerance):
        return await optimize_recommendations(
            recommendations, risk_tolerance, self.portfolio_optimizer,
            portfolio_size=self.portfolio_size,  # F-09-009
        )

    # Also keep the generate_report helper using summary directly


    def _generate_summary(self, recommendations):
        return generate_summary(recommendations)

    def _get_top_sectors(self, recommendations):
        return get_top_sectors(recommendations)
