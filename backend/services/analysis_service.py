"""
Analysis Service
Business logic for orchestrating multiple analysis types.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for orchestrating multi-type stock analysis.
    Coordinates technical, fundamental, and sentiment analysis.
    """

    def __init__(self):
        self.fundamental_engine = FundamentalAnalysisEngine()
        self.technical_engine = TechnicalAnalysisEngine()
        self.sentiment_engine = SentimentAnalysisEngine()
        self._cache = {}  # Simple in-memory cache

    async def run_analysis(
        self,
        ticker: str,
        types: Optional[List[str]] = None,
        depth: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis on a stock.

        Args:
            ticker: Stock ticker symbol
            types: Types of analysis to run (technical, fundamental, sentiment)
                   If None, runs all available analyses
            depth: Analysis depth (quick, standard, deep)

        Returns:
            Dictionary containing all analysis results
        """
        try:
            if types is None:
                types = ['technical', 'fundamental', 'sentiment']

            logger.info(f"Running analysis for {ticker}: types={types}, depth={depth}")

            # Check cache first
            cache_key = self._get_cache_key(ticker, types, depth)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                age_minutes = (datetime.now(timezone.utc) - cached_result['cached_at']).seconds / 60

                if age_minutes < 15:  # Cache for 15 minutes
                    logger.info(f"Returning cached analysis for {ticker} (age: {age_minutes:.1f}m)")
                    return cached_result['data']

            # Prepare result container
            result = {
                'ticker': ticker,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'depth': depth,
                'analyses': {}
            }

            # Note: These would need actual stock data
            # For now, adding placeholders for the structure

            # Run technical analysis
            if 'technical' in types:
                try:
                    # Would need price history DataFrame
                    # technical_result = self.technical_engine.analyze_stock(price_data)
                    result['analyses']['technical'] = {
                        'available': False,
                        'message': 'Price data required for technical analysis'
                    }
                except Exception as e:
                    logger.error(f"Technical analysis failed for {ticker}: {e}")
                    result['analyses']['technical'] = {'error': str(e)}

            # Run fundamental analysis
            if 'fundamental' in types:
                try:
                    # Would need financial data and market data
                    # fundamental_result = await self.fundamental_engine.analyze_company(
                    #     ticker=ticker,
                    #     financials=financials,
                    #     market_data=market_data
                    # )
                    result['analyses']['fundamental'] = {
                        'available': False,
                        'message': 'Financial data required for fundamental analysis'
                    }
                except Exception as e:
                    logger.error(f"Fundamental analysis failed for {ticker}: {e}")
                    result['analyses']['fundamental'] = {'error': str(e)}

            # Run sentiment analysis
            if 'sentiment' in types:
                try:
                    # Would need news/social media text data
                    # sentiment_result = await self.sentiment_engine.analyze_sentiment(
                    #     ticker=ticker,
                    #     text_data=text_data
                    # )
                    result['analyses']['sentiment'] = {
                        'available': False,
                        'message': 'Text data required for sentiment analysis'
                    }
                except Exception as e:
                    logger.error(f"Sentiment analysis failed for {ticker}: {e}")
                    result['analyses']['sentiment'] = {'error': str(e)}

            # Calculate composite score
            result['composite_score'] = self._calculate_composite_score(result['analyses'])

            # Cache the result
            self._cache[cache_key] = {
                'data': result,
                'cached_at': datetime.now(timezone.utc)
            }

            return result

        except Exception as e:
            logger.error(f"Error running analysis for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    async def get_cached_analysis(
        self,
        ticker: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if cached analysis exists for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Cached analysis if available and not stale, None otherwise
        """
        try:
            # Look for any cached analysis for this ticker
            for key, cached in self._cache.items():
                if ticker.upper() in key:
                    age_minutes = (datetime.now(timezone.utc) - cached['cached_at']).seconds / 60

                    if age_minutes < 15:  # Cache valid for 15 minutes
                        logger.info(f"Found cached analysis for {ticker} (age: {age_minutes:.1f}m)")
                        return cached['data']

            return None

        except Exception as e:
            logger.error(f"Error checking cache for {ticker}: {e}")
            return None

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear analysis cache.

        Args:
            ticker: Optional ticker to clear specific cache. If None, clears all.
        """
        if ticker:
            # Remove specific ticker from cache
            keys_to_remove = [k for k in self._cache.keys() if ticker.upper() in k]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for {ticker}")
        else:
            # Clear all cache
            self._cache.clear()
            logger.info("Cleared all analysis cache")

    def _get_cache_key(
        self,
        ticker: str,
        types: List[str],
        depth: str
    ) -> str:
        """Generate cache key from parameters."""
        return f"{ticker.upper()}:{'_'.join(sorted(types))}:{depth}"

    def _calculate_composite_score(
        self,
        analyses: Dict[str, Any]
    ) -> float:
        """
        Calculate composite score from multiple analyses.

        Args:
            analyses: Dictionary of analysis results

        Returns:
            Composite score (0-100)
        """
        scores = []
        weights = {
            'technical': 0.3,
            'fundamental': 0.4,
            'sentiment': 0.3
        }

        for analysis_type, weight in weights.items():
            if analysis_type in analyses:
                analysis = analyses[analysis_type]

                # Extract score if available
                if isinstance(analysis, dict) and 'composite_score' in analysis:
                    scores.append(analysis['composite_score'] * weight)

        if not scores:
            return 0.0

        return sum(scores)

    async def compare_stocks(
        self,
        tickers: List[str],
        analysis_type: str = 'fundamental'
    ) -> Dict[str, Any]:
        """
        Compare multiple stocks using specified analysis type.

        Args:
            tickers: List of stock ticker symbols
            analysis_type: Type of analysis for comparison

        Returns:
            Comparison results
        """
        try:
            logger.info(f"Comparing {len(tickers)} stocks: {tickers}")

            results = []
            for ticker in tickers:
                analysis = await self.run_analysis(
                    ticker=ticker,
                    types=[analysis_type],
                    depth='quick'
                )
                results.append(analysis)

            return {
                'comparison_type': analysis_type,
                'stocks': results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error comparing stocks {tickers}: {e}")
            return {
                'error': str(e)
            }


# Create singleton instance
analysis_service = AnalysisService()
