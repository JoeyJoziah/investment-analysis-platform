"""
Memory-optimized recommendation engine.

Extracted from recommendation_engine.py (originally merged from
recommendation_engine_optimized.py).  This module owns:

  - OptimizedRecommendationEngine
  - get_optimized_recommendation_engine() factory

All names are re-exported from recommendation_engine.py so existing import
paths continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import weakref
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.ml.runtime_models import ModelManager
from backend.data_ingestion.market_scanner import MarketScanner
from backend.utils.risk_manager import RiskManager
from backend.utils.portfolio_optimizer import PortfolioOptimizer

from backend.analytics.recommendation_types import RecommendationAction, StockRecommendation
from backend.analytics.recommendation_ranking import (
    rank_recommendations_optimized,
    optimize_recommendations_streaming,
)

try:
    from backend.utils.memory_manager import (
        get_memory_manager,
        memory_efficient,
        BoundedDict,
        BoundedList,
        MemoryPressureLevel,
        GCStrategy,
    )
    from backend.utils.adaptive_batch_processor import get_batch_processor, BatchConfiguration
    _OPTIMIZED_DEPS_AVAILABLE = True
except ImportError:
    _OPTIMIZED_DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizedRecommendationEngine:
    """
    Memory-optimized recommendation engine with performance improvements.
    Uses weak references, bounded caches, and adaptive batch processing.
    """

    def __init__(self):
        # Weak references prevent circular references to sub-engines
        self._technical_engine_ref = None
        self._fundamental_engine_ref = None
        self._sentiment_engine_ref = None
        self._model_manager_ref = None
        self._market_scanner_ref = None
        self._risk_manager_ref = None
        self._portfolio_optimizer_ref = None

        # Memory-optimized caches with bounds
        if _OPTIMIZED_DEPS_AVAILABLE:
            self._analysis_cache = BoundedDict(max_size=1000)
            self._stock_data_cache = BoundedDict(max_size=500)
            self._recommendation_history = BoundedList(max_size=10000)
            self._processing_locks = BoundedDict(max_size=500)
        else:
            self._analysis_cache: Dict = {}
            self._stock_data_cache: Dict = {}
            self._recommendation_history: List = []
            self._processing_locks: Dict = {}

        # Performance metrics with bounded storage
        self._processing_metrics: deque = deque(maxlen=1000)
        self._memory_usage_history: deque = deque(maxlen=100)

        # Batch processing configuration
        self._batch_processor = None

        # Memory manager
        self._memory_manager = None

        # Recommendation thresholds
        self.thresholds = {
            'strong_buy':  0.8,
            'buy':         0.6,
            'hold':        0.4,
            'sell':        0.2,
            'strong_sell': 0.0,
        }

    # ------------------------------------------------------------------
    # Lazy-loaded engine properties (weak references)
    # ------------------------------------------------------------------

    @property
    def technical_engine(self):
        """Lazy-loaded technical engine with weak reference."""
        if self._technical_engine_ref is None or self._technical_engine_ref() is None:
            engine = TechnicalAnalysisEngine()
            self._technical_engine_ref = weakref.ref(engine)
            return engine
        return self._technical_engine_ref()

    @property
    def fundamental_engine(self):
        """Lazy-loaded fundamental engine with weak reference."""
        if self._fundamental_engine_ref is None or self._fundamental_engine_ref() is None:
            engine = FundamentalAnalysisEngine()
            self._fundamental_engine_ref = weakref.ref(engine)
            return engine
        return self._fundamental_engine_ref()

    @property
    def sentiment_engine(self):
        """Lazy-loaded sentiment engine with weak reference."""
        if self._sentiment_engine_ref is None or self._sentiment_engine_ref() is None:
            engine = SentimentAnalysisEngine()
            self._sentiment_engine_ref = weakref.ref(engine)
            return engine
        return self._sentiment_engine_ref()

    @property
    def model_manager(self):
        """Lazy-loaded model manager with weak reference."""
        if self._model_manager_ref is None or self._model_manager_ref() is None:
            manager = ModelManager()
            self._model_manager_ref = weakref.ref(manager)
            return manager
        return self._model_manager_ref()

    @property
    def market_scanner(self):
        """Lazy-loaded market scanner with weak reference."""
        if self._market_scanner_ref is None or self._market_scanner_ref() is None:
            scanner = MarketScanner()
            self._market_scanner_ref = weakref.ref(scanner)
            return scanner
        return self._market_scanner_ref()

    @property
    def risk_manager(self):
        """Lazy-loaded risk manager with weak reference."""
        if self._risk_manager_ref is None or self._risk_manager_ref() is None:
            manager = RiskManager()
            self._risk_manager_ref = weakref.ref(manager)
            return manager
        return self._risk_manager_ref()

    @property
    def portfolio_optimizer(self):
        """Lazy-loaded portfolio optimizer with weak reference."""
        if self._portfolio_optimizer_ref is None or self._portfolio_optimizer_ref() is None:
            optimizer = PortfolioOptimizer()
            self._portfolio_optimizer_ref = weakref.ref(optimizer)
            return optimizer
        return self._portfolio_optimizer_ref()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self):
        """Initialize all components with memory optimization."""
        if not _OPTIMIZED_DEPS_AVAILABLE:
            logger.warning("Optimized deps unavailable; running without memory manager")
            return

        self._memory_manager = await get_memory_manager(
            gc_strategy=GCStrategy.ADAPTIVE,
            memory_threshold_mb=4096,
        )

        self._memory_manager.register_bounded_collection("analysis_cache", self._analysis_cache)
        self._memory_manager.register_bounded_collection("stock_data_cache", self._stock_data_cache)
        self._memory_manager.register_bounded_collection(
            "recommendation_history", self._recommendation_history
        )

        batch_config = BatchConfiguration(
            min_batch_size=20,
            max_batch_size=200,
            initial_batch_size=50,
            target_processing_time_ms=2000,
            max_memory_mb=1024,
            max_cpu_percent=70,
            adjustment_factor=0.3,
            stability_window=5,
        )
        self._batch_processor = await get_batch_processor(batch_config)

        await self._memory_manager.optimize_for_batch_processing()

        logger.info("Optimized recommendation engine initialized")

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
        """Generate daily recommendations with memory optimization."""
        logger.info("Starting memory-optimized daily recommendation generation...")
        start_time = datetime.now(timezone.utc)

        try:
            candidates = await self._scan_market_optimized(
                sectors=sectors,
                market_cap_range=market_cap_range,
                max_stocks=min(1000, max_recommendations * 20),
            )

            logger.info(f"Found {len(candidates)} candidate stocks")

            recommendations = await self._process_candidates_batched(
                candidates,
                max_recommendations * 2,
            )

            ranked_recommendations = rank_recommendations_optimized(recommendations)

            optimized_recommendations = await optimize_recommendations_streaming(
                ranked_recommendations[: max_recommendations * 2],
                risk_tolerance,
            )

            final_recommendations = optimized_recommendations[:max_recommendations]

            if isinstance(self._recommendation_history, list):
                self._recommendation_history.extend(final_recommendations)
            else:
                self._recommendation_history.extend(final_recommendations)

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            memory_mb = 0.0
            if self._memory_manager:
                memory_mb = (await self._memory_manager.collect_metrics()).process_memory_mb

            self._processing_metrics.append({
                'timestamp': datetime.now(timezone.utc),
                'processing_time_s': elapsed,
                'candidates_processed': len(candidates),
                'recommendations_generated': len(final_recommendations),
                'memory_usage_mb': memory_mb,
            })

            await self._post_processing_cleanup()

            logger.info(
                f"Generated {len(final_recommendations)} optimized recommendations in {elapsed:.2f}s"
            )

            return final_recommendations

        except Exception as e:
            logger.error(f"Error in optimized recommendation generation: {e}")
            if self._memory_manager:
                await self._memory_manager.emergency_cleanup()
            raise

    # ------------------------------------------------------------------
    # Internal - market scanning
    # ------------------------------------------------------------------

    async def _scan_market_optimized(
        self,
        sectors: Optional[List[str]] = None,
        market_cap_range: Optional[Tuple[float, float]] = None,
        max_stocks: int = 1000,
    ) -> List[Dict]:
        """Optimized market scanning with memory limits."""
        scanner = self.market_scanner
        candidates: List[Dict] = []

        chunk_size = 100
        processed = 0

        async for stock_chunk in scanner.scan_market_streaming(
            sectors=sectors,
            market_cap_range=market_cap_range,
            chunk_size=chunk_size,
        ):
            candidates.extend(stock_chunk)
            processed += len(stock_chunk)

            if processed >= max_stocks:
                candidates = candidates[:max_stocks]
                break

            if self._memory_manager and processed % (chunk_size * 5) == 0:
                metrics = await self._memory_manager.collect_metrics()
                if metrics.pressure_level == MemoryPressureLevel.HIGH:
                    logger.warning("High memory pressure during market scan, reducing candidates")
                    break

        return candidates

    # ------------------------------------------------------------------
    # Internal - batch processing
    # ------------------------------------------------------------------

    async def _process_candidates_batched(
        self,
        candidates: List[Dict],
        max_results: int,
    ) -> List[StockRecommendation]:
        """Process candidates using optimized batching."""
        recommendations: List[StockRecommendation] = []

        async def process_batch(batch):
            batch_recommendations = []
            tasks = [
                self._analyze_stock_optimized(candidate['ticker'], candidate)
                for candidate in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for candidate, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing {candidate['ticker']}: {result}")
                    continue
                if result and isinstance(result, StockRecommendation):
                    batch_recommendations.append(result)
            return batch_recommendations

        if self._batch_processor:
            batch_results = await self._batch_processor.process_adaptive_batch(
                candidates, process_batch
            )
            for batch_result, _metrics in batch_results:
                if batch_result:
                    recommendations.extend(batch_result)
                if len(recommendations) >= max_results:
                    recommendations = recommendations[:max_results]
                    break
        else:
            batch_size = 50
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                batch_result = await process_batch(batch)
                recommendations.extend(batch_result)
                if len(recommendations) >= max_results:
                    recommendations = recommendations[:max_results]
                    break

        return recommendations

    # ------------------------------------------------------------------
    # Internal - single-stock analysis
    # ------------------------------------------------------------------

    async def _analyze_stock_optimized(
        self,
        ticker: str,
        market_data: Optional[Dict] = None,
    ) -> Optional[StockRecommendation]:
        """Memory-optimized stock analysis."""
        if ticker in self._processing_locks:
            return None

        self._processing_locks[ticker] = datetime.now(timezone.utc)

        try:
            cache_key = f"analysis_{ticker}_{datetime.now(timezone.utc).date()}"
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]

            logger.debug(f"Analyzing {ticker} with memory optimization...")

            stock_data = await self._fetch_stock_data_minimal(ticker, market_data)

            if not stock_data:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            analysis_summary = await self._run_lightweight_analysis(ticker, stock_data)

            if not analysis_summary:
                return None

            recommendation = self._generate_recommendation_optimized(
                ticker=ticker,
                stock_data=stock_data,
                analysis_summary=analysis_summary,
            )

            self._analysis_cache[cache_key] = recommendation
            return recommendation

        except Exception as e:
            logger.error(f"Error in optimized analysis of {ticker}: {e}")
            return None
        finally:
            self._processing_locks.pop(ticker, None)
            gc.collect()

    async def _fetch_stock_data_minimal(
        self,
        ticker: str,
        market_data: Optional[Dict],
    ) -> Optional[Dict]:
        """Fetch minimal required data to reduce memory usage."""
        cache_key = f"stock_data_{ticker}_{datetime.now(timezone.utc).date()}"
        if cache_key in self._stock_data_cache:
            return self._stock_data_cache[cache_key]

        if not market_data:
            market_data = await self.market_scanner.get_stock_data_minimal(ticker)

        if not market_data:
            return None

        essential_data = {
            'ticker':        ticker,
            'current_price': market_data.get('current_price'),
            'price_history': market_data.get('price_history'),
            'volume':        market_data.get('volume'),
            'market_cap':    market_data.get('market_cap'),
            'beta':          market_data.get('beta', 1.0),
        }

        if essential_data['price_history'] is not None:
            if len(essential_data['price_history']) > 100:
                essential_data['price_history'] = essential_data['price_history'].tail(100)

        self._stock_data_cache[cache_key] = essential_data
        return essential_data

    async def _run_lightweight_analysis(
        self,
        ticker: str,
        stock_data: Dict,
    ) -> Optional[Dict]:
        """Run lightweight analysis instead of full analysis."""
        if stock_data.get('price_history') is None:
            return None

        price_df = stock_data['price_history']

        if len(price_df) < 30:
            return None

        returns = price_df['close'].pct_change().dropna()

        sma_20 = price_df['close'].rolling(20).mean().iloc[-1]
        current_price = price_df['close'].iloc[-1]

        momentum_5d = (current_price - price_df['close'].iloc[-6]) / price_df['close'].iloc[-6]
        momentum_20d = (current_price - sma_20) / sma_20

        volatility = returns.std() * np.sqrt(252)

        avg_volume = price_df['volume'].rolling(20).mean().iloc[-1]
        current_volume = price_df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        max_drawdown = self._calculate_simple_drawdown(price_df['close'])

        return {
            'technical_score': self._calculate_technical_score_simple(
                momentum_5d, momentum_20d, volume_ratio
            ),
            'risk_score':   min(volatility / 0.5, 1.0),
            'volatility':   volatility,
            'max_drawdown': max_drawdown,
            'momentum_5d':  momentum_5d,
            'momentum_20d': momentum_20d,
            'volume_ratio': volume_ratio,
            'key_signals':  self._extract_key_signals_simple(momentum_5d, momentum_20d, volume_ratio),
            'risk_factors': self._extract_risk_factors_simple(volatility, max_drawdown),
            'opportunities': self._extract_opportunities_simple(momentum_5d, volume_ratio),
        }

    # ------------------------------------------------------------------
    # Internal - lightweight score helpers
    # ------------------------------------------------------------------

    def _calculate_simple_drawdown(self, prices: "Any") -> float:
        """Calculate maximum drawdown efficiently."""
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    def _calculate_technical_score_simple(
        self,
        momentum_5d: float,
        momentum_20d: float,
        volume_ratio: float,
    ) -> float:
        """Simple technical score calculation."""
        momentum_score = (momentum_5d + momentum_20d) / 2
        volume_score = min(volume_ratio / 2, 1.0) if volume_ratio > 1 else 0.5
        technical_score = momentum_score * 0.7 + (volume_score - 0.5) * 0.3
        return max(0.0, min(1.0, (technical_score + 0.2) / 0.4))

    def _extract_key_signals_simple(
        self,
        momentum_5d: float,
        momentum_20d: float,
        volume_ratio: float,
    ) -> List[str]:
        """Extract key signals efficiently."""
        signals: List[str] = []
        if momentum_5d > 0.05:
            signals.append("Strong 5-day momentum")
        elif momentum_5d < -0.05:
            signals.append("Negative 5-day momentum")
        if momentum_20d > 0.1:
            signals.append("Strong 20-day trend")
        elif momentum_20d < -0.1:
            signals.append("Bearish 20-day trend")
        if volume_ratio > 1.5:
            signals.append("High volume activity")
        return signals[:3]

    def _extract_risk_factors_simple(self, volatility: float, max_drawdown: float) -> List[str]:
        """Extract risk factors efficiently."""
        risks: List[str] = []
        if volatility > 0.4:
            risks.append(f"High volatility ({volatility * 100:.0f}%)")
        if max_drawdown < -0.2:
            risks.append(f"Significant drawdown risk ({max_drawdown * 100:.0f}%)")
        return risks[:2]

    def _extract_opportunities_simple(
        self, momentum_5d: float, volume_ratio: float
    ) -> List[str]:
        """Extract opportunities efficiently."""
        opportunities: List[str] = []
        if momentum_5d > 0.03 and volume_ratio > 1.2:
            opportunities.append("Momentum with volume confirmation")
        if momentum_5d < -0.05:
            opportunities.append("Potential oversold condition")
        return opportunities[:2]

    # ------------------------------------------------------------------
    # Internal - recommendation generation
    # ------------------------------------------------------------------

    def _determine_action(self, score: float) -> RecommendationAction:
        """Determine recommendation action based on score."""
        if score >= self.thresholds['strong_buy']:
            return RecommendationAction.STRONG_BUY
        elif score >= self.thresholds['buy']:
            return RecommendationAction.BUY
        elif score >= self.thresholds['hold']:
            return RecommendationAction.HOLD
        elif score >= self.thresholds['sell']:
            return RecommendationAction.SELL
        else:
            return RecommendationAction.STRONG_SELL

    def _calculate_priority_simple(self, score: float, confidence: float) -> int:
        """Calculate priority efficiently."""
        base_priority = int(score * 10)
        if confidence > 0.8:
            base_priority += 1
        return max(1, min(10, base_priority))

    def _calculate_allocation_simple(self, confidence: float, risk_score: float) -> float:
        """Calculate allocation efficiently."""
        base_allocation = confidence * 0.1
        risk_adjustment = 1 - risk_score * 0.5
        return max(0.01, base_allocation * risk_adjustment)

    def _generate_recommendation_optimized(
        self,
        ticker: str,
        stock_data: Dict,
        analysis_summary: Dict,
    ) -> StockRecommendation:
        """Generate recommendation with optimized memory usage."""
        current_price = stock_data.get('current_price', 0)
        if current_price == 0 and stock_data.get('price_history') is not None:
            current_price = stock_data['price_history']['close'].iloc[-1]

        technical_score = analysis_summary.get('technical_score', 0.5)
        risk_score = analysis_summary.get('risk_score', 0.5)

        overall_score = technical_score * (1 - risk_score * 0.3)

        action = self._determine_action(overall_score)

        confidence = min(technical_score + 0.2, 1.0)

        momentum_20d = analysis_summary.get('momentum_20d', 0)
        target_price = current_price * (1 + max(0.05, abs(momentum_20d)))
        stop_loss = current_price * 0.95

        expected_return = (target_price - current_price) / current_price if current_price else 0.0

        volatility = analysis_summary.get('volatility', 0.2)
        max_drawdown = analysis_summary.get('max_drawdown', -0.1)

        return StockRecommendation(
            ticker=ticker,
            action=action,
            confidence=confidence,
            priority=self._calculate_priority_simple(overall_score, confidence),
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return=expected_return,
            time_horizon_days=30,
            risk_score=risk_score,
            volatility=volatility,
            beta=stock_data.get('beta', 1.0),
            sharpe_ratio=max(0.0, expected_return / volatility) if volatility > 0 else 0.0,
            max_drawdown=max_drawdown,
            technical_score=technical_score,
            fundamental_score=0.5,
            sentiment_score=0.5,
            ml_prediction_score=0.5,
            technical_analysis={},
            fundamental_analysis={},
            sentiment_analysis={},
            ml_predictions={},
            key_factors=analysis_summary.get('key_signals', []),
            risks=analysis_summary.get('risk_factors', []),
            opportunities=analysis_summary.get('opportunities', []),
            catalysts=[],
            generated_at=datetime.now(timezone.utc),
            valid_until=datetime.now(timezone.utc) + timedelta(days=1),
            recommended_allocation=self._calculate_allocation_simple(confidence, risk_score),
            max_position_size=self._calculate_allocation_simple(confidence, risk_score) * 100_000,
        )

    # ------------------------------------------------------------------
    # Internal - cleanup and housekeeping
    # ------------------------------------------------------------------

    async def _post_processing_cleanup(self):
        """Cleanup after processing a batch."""
        collected = gc.collect()

        if len(self._analysis_cache) > 800:
            keys_to_remove = list(self._analysis_cache.keys())[:200]
            for key in keys_to_remove:
                self._analysis_cache.pop(key, None)

        if len(self._stock_data_cache) > 400:
            keys_to_remove = list(self._stock_data_cache.keys())[:100]
            for key in keys_to_remove:
                self._stock_data_cache.pop(key, None)

        logger.debug(f"Post-processing cleanup collected {collected} objects")

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._processing_metrics:
            return {}

        recent_metrics = list(self._processing_metrics)[-10:]

        return {
            'recent_processing_time_avg_s': np.mean(
                [m['processing_time_s'] for m in recent_metrics]
            ),
            'recent_candidates_avg': np.mean(
                [m['candidates_processed'] for m in recent_metrics]
            ),
            'recent_recommendations_avg': np.mean(
                [m['recommendations_generated'] for m in recent_metrics]
            ),
            'cache_sizes': {
                'analysis_cache':         len(self._analysis_cache),
                'stock_data_cache':       len(self._stock_data_cache),
                'recommendation_history': len(self._recommendation_history),
            },
            'memory_optimization': 'enabled',
            'batch_processor_stats': (
                self._batch_processor.get_statistics() if self._batch_processor else {}
            ),
        }

    async def shutdown(self):
        """Shutdown with comprehensive cleanup."""
        try:
            self._analysis_cache.clear()
            self._stock_data_cache.clear()
            self._recommendation_history.clear()
            self._processing_metrics.clear()
            self._memory_usage_history.clear()
            self._processing_locks.clear()

            self._technical_engine_ref = None
            self._fundamental_engine_ref = None
            self._sentiment_engine_ref = None
            self._model_manager_ref = None
            self._market_scanner_ref = None
            self._risk_manager_ref = None
            self._portfolio_optimizer_ref = None

            if self._memory_manager:
                await self._memory_manager.restore_default_settings()

            gc.collect()

            logger.info("Optimized recommendation engine shutdown complete")

        except Exception as e:
            logger.error(f"Error during optimized engine shutdown: {e}")


# ---------------------------------------------------------------------------
# Global singleton factory
# ---------------------------------------------------------------------------

_optimized_engine: Optional[OptimizedRecommendationEngine] = None


async def get_optimized_recommendation_engine() -> OptimizedRecommendationEngine:
    """Get or create the global optimized recommendation engine."""
    global _optimized_engine
    if _optimized_engine is None:
        _optimized_engine = OptimizedRecommendationEngine()
        await _optimized_engine.initialize()
    return _optimized_engine
