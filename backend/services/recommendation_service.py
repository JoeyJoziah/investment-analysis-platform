"""
Recommendation Service
Business logic for generating and managing investment recommendations.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from backend.analytics.recommendation_engine import RecommendationEngine, StockRecommendation
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Service for generating investment recommendations.
    Orchestrates multiple analysis engines and aggregates results.
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
