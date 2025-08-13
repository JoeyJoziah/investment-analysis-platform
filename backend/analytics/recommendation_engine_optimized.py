"""
Memory-Optimized Investment Recommendation Engine
Fixes memory leaks and implements performance optimizations
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import json
import weakref
import gc
from collections import deque

from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.models.ml_models import ModelManager, PredictionResult
from backend.data_ingestion.market_scanner import MarketScanner
from backend.utils.risk_manager import RiskManager
from backend.utils.portfolio_optimizer import PortfolioOptimizer
from backend.utils.memory_manager import (
    get_memory_manager, memory_efficient, BoundedDict, BoundedList,
    MemoryPressureLevel, GCStrategy
)
from backend.utils.adaptive_batch_processor import get_batch_processor, BatchConfiguration

logger = logging.getLogger(__name__)


class RecommendationAction(Enum):
    """Recommendation actions"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class StockRecommendation:
    """Optimized stock recommendation with memory management"""
    ticker: str
    action: RecommendationAction
    confidence: float
    priority: int  # 1-10
    
    # Price targets
    entry_price: float
    target_price: float
    stop_loss: float
    expected_return: float
    time_horizon_days: int
    
    # Risk metrics
    risk_score: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Analysis scores
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    ml_prediction_score: float
    
    # Lightweight analysis summaries (instead of full objects)
    key_signals: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=1))
    
    # Position sizing
    recommended_allocation: float = 0.0  # Percentage of portfolio
    max_position_size: float = 0.0  # Dollar amount
    
    def __post_init__(self):
        """Post-initialization cleanup"""
        # Ensure lists are bounded
        self.key_signals = self.key_signals[:5]  # Max 5 signals
        self.risk_factors = self.risk_factors[:4]  # Max 4 risks
        self.opportunities = self.opportunities[:3]  # Max 3 opportunities
        self.catalysts = self.catalysts[:3]  # Max 3 catalysts
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage with memory optimization"""
        return {
            'ticker': self.ticker,
            'action': self.action.value,
            'confidence': round(self.confidence, 3),
            'priority': self.priority,
            'entry_price': round(self.entry_price, 2),
            'target_price': round(self.target_price, 2),
            'stop_loss': round(self.stop_loss, 2),
            'expected_return': round(self.expected_return, 4),
            'time_horizon_days': self.time_horizon_days,
            'risk_score': round(self.risk_score, 3),
            'volatility': round(self.volatility, 4),
            'beta': round(self.beta, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 4),
            'technical_score': round(self.technical_score, 3),
            'fundamental_score': round(self.fundamental_score, 3),
            'sentiment_score': round(self.sentiment_score, 3),
            'ml_prediction_score': round(self.ml_prediction_score, 3),
            'key_signals': self.key_signals,
            'risk_factors': self.risk_factors,
            'opportunities': self.opportunities,
            'catalysts': self.catalysts,
            'generated_at': self.generated_at.isoformat(),
            'valid_until': self.valid_until.isoformat(),
            'recommended_allocation': round(self.recommended_allocation, 4),
            'max_position_size': round(self.max_position_size, 2)
        }
    
    def __del__(self):
        """Destructor for cleanup"""
        # Clear references
        if hasattr(self, 'key_signals'):
            self.key_signals.clear()
        if hasattr(self, 'risk_factors'):
            self.risk_factors.clear()
        if hasattr(self, 'opportunities'):
            self.opportunities.clear()
        if hasattr(self, 'catalysts'):
            self.catalysts.clear()


class OptimizedRecommendationEngine:
    """
    Memory-optimized recommendation engine with performance improvements
    """
    
    def __init__(self):
        # Use weak references for engines to prevent circular references
        self._technical_engine_ref = None
        self._fundamental_engine_ref = None
        self._sentiment_engine_ref = None
        self._model_manager_ref = None
        self._market_scanner_ref = None
        self._risk_manager_ref = None
        self._portfolio_optimizer_ref = None
        
        # Memory-optimized caches with bounds
        self._analysis_cache = BoundedDict(max_size=1000)
        self._stock_data_cache = BoundedDict(max_size=500)
        self._recommendation_history = BoundedList(max_size=10000)
        
        # Performance metrics with bounded storage
        self._processing_metrics = deque(maxlen=1000)
        self._memory_usage_history = deque(maxlen=100)
        
        # Batch processing configuration
        self._batch_processor = None
        
        # Memory manager
        self._memory_manager = None
        
        # Recommendation thresholds
        self.thresholds = {
            'strong_buy': 0.8,
            'buy': 0.6,
            'hold': 0.4,
            'sell': 0.2,
            'strong_sell': 0.0
        }
        
        # Processing locks to prevent concurrent analysis of same stock
        self._processing_locks = BoundedDict(max_size=500)
    
    @property
    def technical_engine(self):
        """Lazy-loaded technical engine with weak reference"""
        if self._technical_engine_ref is None or self._technical_engine_ref() is None:
            engine = TechnicalAnalysisEngine()
            self._technical_engine_ref = weakref.ref(engine)
            return engine
        return self._technical_engine_ref()
    
    @property
    def fundamental_engine(self):
        """Lazy-loaded fundamental engine with weak reference"""
        if self._fundamental_engine_ref is None or self._fundamental_engine_ref() is None:
            engine = FundamentalAnalysisEngine()
            self._fundamental_engine_ref = weakref.ref(engine)
            return engine
        return self._fundamental_engine_ref()
    
    @property
    def sentiment_engine(self):
        """Lazy-loaded sentiment engine with weak reference"""
        if self._sentiment_engine_ref is None or self._sentiment_engine_ref() is None:
            engine = SentimentAnalysisEngine()
            self._sentiment_engine_ref = weakref.ref(engine)
            return engine
        return self._sentiment_engine_ref()
    
    @property
    def model_manager(self):
        """Lazy-loaded model manager with weak reference"""
        if self._model_manager_ref is None or self._model_manager_ref() is None:
            manager = ModelManager()
            self._model_manager_ref = weakref.ref(manager)
            return manager
        return self._model_manager_ref()
    
    @property
    def market_scanner(self):
        """Lazy-loaded market scanner with weak reference"""
        if self._market_scanner_ref is None or self._market_scanner_ref() is None:
            scanner = MarketScanner()
            self._market_scanner_ref = weakref.ref(scanner)
            return scanner
        return self._market_scanner_ref()
    
    @property
    def risk_manager(self):
        """Lazy-loaded risk manager with weak reference"""
        if self._risk_manager_ref is None or self._risk_manager_ref() is None:
            manager = RiskManager()
            self._risk_manager_ref = weakref.ref(manager)
            return manager
        return self._risk_manager_ref()
    
    @property
    def portfolio_optimizer(self):
        """Lazy-loaded portfolio optimizer with weak reference"""
        if self._portfolio_optimizer_ref is None or self._portfolio_optimizer_ref() is None:
            optimizer = PortfolioOptimizer()
            self._portfolio_optimizer_ref = weakref.ref(optimizer)
            return optimizer
        return self._portfolio_optimizer_ref()
    
    async def initialize(self):
        """Initialize all components with memory optimization"""
        # Get memory manager
        self._memory_manager = await get_memory_manager(
            gc_strategy=GCStrategy.ADAPTIVE,
            memory_threshold_mb=4096
        )
        
        # Register bounded collections for automatic cleanup
        self._memory_manager.register_bounded_collection("analysis_cache", self._analysis_cache)
        self._memory_manager.register_bounded_collection("stock_data_cache", self._stock_data_cache)
        self._memory_manager.register_bounded_collection("recommendation_history", self._recommendation_history)
        
        # Get batch processor with optimized configuration
        batch_config = BatchConfiguration(
            min_batch_size=20,  # Increased from default
            max_batch_size=200,  # Increased from default
            initial_batch_size=50,  # Increased from default
            target_processing_time_ms=2000,  # Increased for better throughput
            max_memory_mb=1024,
            max_cpu_percent=70,
            adjustment_factor=0.3,
            stability_window=5
        )
        self._batch_processor = await get_batch_processor(batch_config)
        
        # Optimize memory for batch processing
        await self._memory_manager.optimize_for_batch_processing()
        
        logger.info("Optimized recommendation engine initialized")
    
    @memory_efficient
    async def generate_daily_recommendations(
        self,
        max_recommendations: int = 50,
        risk_tolerance: str = 'moderate',
        sectors: Optional[List[str]] = None,
        market_cap_range: Optional[Tuple[float, float]] = None
    ) -> List[StockRecommendation]:
        """
        Generate daily recommendations with memory optimization
        """
        logger.info("Starting memory-optimized daily recommendation generation...")
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Scan market for candidates with memory limits
            candidates = await self._scan_market_optimized(
                sectors=sectors,
                market_cap_range=market_cap_range,
                max_stocks=min(1000, max_recommendations * 20)  # Limit candidates
            )
            
            logger.info(f"Found {len(candidates)} candidate stocks")
            
            # Step 2: Process candidates in optimized batches
            recommendations = await self._process_candidates_batched(
                candidates,
                max_recommendations * 2  # Get 2x for filtering
            )
            
            # Step 3: Apply memory-efficient ranking
            ranked_recommendations = await self._rank_recommendations_optimized(recommendations)
            
            # Step 4: Portfolio optimization with streaming
            optimized_recommendations = await self._optimize_recommendations_streaming(
                ranked_recommendations[:max_recommendations * 2],
                risk_tolerance
            )
            
            # Step 5: Final selection
            final_recommendations = optimized_recommendations[:max_recommendations]
            
            # Store in bounded history
            self._recommendation_history.extend(final_recommendations)
            
            # Record processing metrics
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self._processing_metrics.append({
                'timestamp': datetime.utcnow(),
                'processing_time_s': elapsed,
                'candidates_processed': len(candidates),
                'recommendations_generated': len(final_recommendations),
                'memory_usage_mb': (await self._memory_manager.collect_metrics()).process_memory_mb
            })
            
            # Force cleanup after processing
            await self._post_processing_cleanup()
            
            logger.info(
                f"Generated {len(final_recommendations)} optimized recommendations in {elapsed:.2f}s"
            )
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in optimized recommendation generation: {e}")
            # Emergency cleanup on error
            await self._memory_manager.emergency_cleanup()
            raise
    
    async def _scan_market_optimized(
        self,
        sectors: Optional[List[str]] = None,
        market_cap_range: Optional[Tuple[float, float]] = None,
        max_stocks: int = 1000
    ) -> List[Dict]:
        """Optimized market scanning with memory limits"""
        # Use streaming approach to avoid loading all stocks at once
        scanner = self.market_scanner
        candidates = []
        
        # Process in chunks to limit memory usage
        chunk_size = 100
        processed = 0
        
        async for stock_chunk in scanner.scan_market_streaming(
            sectors=sectors,
            market_cap_range=market_cap_range,
            chunk_size=chunk_size
        ):
            candidates.extend(stock_chunk)
            processed += len(stock_chunk)
            
            # Limit total candidates
            if processed >= max_stocks:
                candidates = candidates[:max_stocks]
                break
            
            # Periodic memory check
            if processed % (chunk_size * 5) == 0:
                metrics = await self._memory_manager.collect_metrics()
                if metrics.pressure_level == MemoryPressureLevel.HIGH:
                    logger.warning("High memory pressure during market scan, reducing candidates")
                    break
        
        return candidates
    
    async def _process_candidates_batched(
        self,
        candidates: List[Dict],
        max_results: int
    ) -> List[StockRecommendation]:
        """Process candidates using optimized batching"""
        recommendations = []
        
        # Process using adaptive batch processor
        async def process_batch(batch):
            batch_recommendations = []
            
            # Process each stock in the batch
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
        
        # Use adaptive batch processor
        batch_results = await self._batch_processor.process_adaptive_batch(
            candidates, process_batch
        )
        
        # Collect all recommendations
        for batch_result, metrics in batch_results:
            if batch_result:
                recommendations.extend(batch_result)
            
            # Early exit if we have enough high-quality recommendations
            if len(recommendations) >= max_results:
                recommendations = recommendations[:max_results]
                break
        
        return recommendations
    
    @memory_efficient
    async def _analyze_stock_optimized(
        self,
        ticker: str,
        market_data: Optional[Dict] = None
    ) -> Optional[StockRecommendation]:
        """
        Memory-optimized stock analysis
        """
        # Check if already processing this ticker
        if ticker in self._processing_locks:
            return None
        
        # Set processing lock
        self._processing_locks[ticker] = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = f"analysis_{ticker}_{datetime.utcnow().date()}"
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
            
            logger.debug(f"Analyzing {ticker} with memory optimization...")
            
            # Fetch minimal required data
            stock_data = await self._fetch_stock_data_minimal(ticker, market_data)
            
            if not stock_data:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            # Run lightweight analysis instead of full analysis
            analysis_summary = await self._run_lightweight_analysis(ticker, stock_data)
            
            if not analysis_summary:
                return None
            
            # Generate optimized recommendation
            recommendation = self._generate_recommendation_optimized(
                ticker=ticker,
                stock_data=stock_data,
                analysis_summary=analysis_summary
            )
            
            # Cache the result
            self._analysis_cache[cache_key] = recommendation
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in optimized analysis of {ticker}: {e}")
            return None
        finally:
            # Remove processing lock
            self._processing_locks.pop(ticker, None)
            # Force cleanup of local variables
            stock_data = None
            analysis_summary = None
            gc.collect()
    
    async def _fetch_stock_data_minimal(
        self,
        ticker: str,
        market_data: Optional[Dict]
    ) -> Optional[Dict]:
        """Fetch minimal required data to reduce memory usage"""
        
        # Check cache first
        cache_key = f"stock_data_{ticker}_{datetime.utcnow().date()}"
        if cache_key in self._stock_data_cache:
            return self._stock_data_cache[cache_key]
        
        # Get from market scanner if not provided
        if not market_data:
            market_data = await self.market_scanner.get_stock_data_minimal(ticker)
        
        if not market_data:
            return None
        
        # Extract only essential fields to minimize memory
        essential_data = {
            'ticker': ticker,
            'current_price': market_data.get('current_price'),
            'price_history': market_data.get('price_history'),  # Keep last 100 days only
            'volume': market_data.get('volume'),
            'market_cap': market_data.get('market_cap'),
            'beta': market_data.get('beta', 1.0)
        }
        
        # Limit price history to reduce memory
        if essential_data['price_history'] is not None:
            if len(essential_data['price_history']) > 100:
                essential_data['price_history'] = essential_data['price_history'].tail(100)
        
        # Cache minimal data
        self._stock_data_cache[cache_key] = essential_data
        
        return essential_data
    
    async def _run_lightweight_analysis(
        self,
        ticker: str,
        stock_data: Dict
    ) -> Optional[Dict]:
        """Run lightweight analysis instead of full analysis"""
        
        if not stock_data.get('price_history') is not None:
            return None
        
        price_df = stock_data['price_history']
        
        if len(price_df) < 30:  # Need minimum data
            return None
        
        # Calculate essential indicators only
        returns = price_df['close'].pct_change().dropna()
        
        # Technical indicators (lightweight)
        sma_20 = price_df['close'].rolling(20).mean().iloc[-1]
        current_price = price_df['close'].iloc[-1]
        
        # Simple momentum
        momentum_5d = (current_price - price_df['close'].iloc[-6]) / price_df['close'].iloc[-6]
        momentum_20d = (current_price - sma_20) / sma_20
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Volume trend
        avg_volume = price_df['volume'].rolling(20).mean().iloc[-1]
        current_volume = price_df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Risk metrics (simplified)
        max_drawdown = self._calculate_simple_drawdown(price_df['close'])
        
        return {
            'technical_score': self._calculate_technical_score_simple(
                momentum_5d, momentum_20d, volume_ratio
            ),
            'risk_score': min(volatility / 0.5, 1.0),  # Normalize to 50% vol
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'volume_ratio': volume_ratio,
            'key_signals': self._extract_key_signals_simple(momentum_5d, momentum_20d, volume_ratio),
            'risk_factors': self._extract_risk_factors_simple(volatility, max_drawdown),
            'opportunities': self._extract_opportunities_simple(momentum_5d, volume_ratio)
        }
    
    def _calculate_simple_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown efficiently"""
        cumulative = (prices / prices.iloc[0])
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_technical_score_simple(
        self,
        momentum_5d: float,
        momentum_20d: float,
        volume_ratio: float
    ) -> float:
        """Simple technical score calculation"""
        # Momentum score
        momentum_score = (momentum_5d + momentum_20d) / 2
        
        # Volume score
        volume_score = min(volume_ratio / 2, 1.0) if volume_ratio > 1 else 0.5
        
        # Combined score
        technical_score = (momentum_score * 0.7 + (volume_score - 0.5) * 0.3)
        
        # Normalize to 0-1
        return max(0, min(1, (technical_score + 0.2) / 0.4))
    
    def _extract_key_signals_simple(self, momentum_5d, momentum_20d, volume_ratio) -> List[str]:
        """Extract key signals efficiently"""
        signals = []
        
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
        
        return signals[:3]  # Limit to 3 signals
    
    def _extract_risk_factors_simple(self, volatility, max_drawdown) -> List[str]:
        """Extract risk factors efficiently"""
        risks = []
        
        if volatility > 0.4:
            risks.append(f"High volatility ({volatility*100:.0f}%)")
        
        if max_drawdown < -0.2:
            risks.append(f"Significant drawdown risk ({max_drawdown*100:.0f}%)")
        
        return risks[:2]  # Limit to 2 risks
    
    def _extract_opportunities_simple(self, momentum_5d, volume_ratio) -> List[str]:
        """Extract opportunities efficiently"""
        opportunities = []
        
        if momentum_5d > 0.03 and volume_ratio > 1.2:
            opportunities.append("Momentum with volume confirmation")
        
        if momentum_5d < -0.05:
            opportunities.append("Potential oversold condition")
        
        return opportunities[:2]  # Limit to 2 opportunities
    
    def _generate_recommendation_optimized(
        self,
        ticker: str,
        stock_data: Dict,
        analysis_summary: Dict
    ) -> StockRecommendation:
        """Generate recommendation with optimized memory usage"""
        
        current_price = stock_data.get('current_price', 0)
        if current_price == 0 and stock_data.get('price_history') is not None:
            current_price = stock_data['price_history']['close'].iloc[-1]
        
        # Simple scoring
        technical_score = analysis_summary.get('technical_score', 0.5)
        risk_score = analysis_summary.get('risk_score', 0.5)
        
        # Overall score (simplified)
        overall_score = technical_score * (1 - risk_score * 0.3)
        
        # Determine action
        action = self._determine_action(overall_score)
        
        # Simple confidence calculation
        confidence = min(technical_score + 0.2, 1.0)
        
        # Basic price targets
        momentum_20d = analysis_summary.get('momentum_20d', 0)
        target_price = current_price * (1 + max(0.05, abs(momentum_20d)))
        stop_loss = current_price * 0.95  # Simple 5% stop loss
        
        expected_return = (target_price - current_price) / current_price
        
        # Risk metrics
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
            time_horizon_days=30,  # Default 30 days
            risk_score=risk_score,
            volatility=volatility,
            beta=stock_data.get('beta', 1.0),
            sharpe_ratio=max(0, expected_return / volatility) if volatility > 0 else 0,
            max_drawdown=max_drawdown,
            technical_score=technical_score,
            fundamental_score=0.5,  # Default
            sentiment_score=0.5,  # Default
            ml_prediction_score=0.5,  # Default
            key_signals=analysis_summary.get('key_signals', []),
            risk_factors=analysis_summary.get('risk_factors', []),
            opportunities=analysis_summary.get('opportunities', []),
            catalysts=[],  # Empty for performance
            recommended_allocation=self._calculate_allocation_simple(confidence, risk_score),
            max_position_size=self._calculate_allocation_simple(confidence, risk_score) * 100000
        )
    
    def _determine_action(self, score: float) -> RecommendationAction:
        """Determine recommendation action based on score"""
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
        """Calculate priority efficiently"""
        base_priority = int(score * 10)
        if confidence > 0.8:
            base_priority += 1
        return max(1, min(10, base_priority))
    
    def _calculate_allocation_simple(self, confidence: float, risk_score: float) -> float:
        """Calculate allocation efficiently"""
        base_allocation = confidence * 0.1  # Max 10%
        risk_adjustment = 1 - risk_score * 0.5
        return max(0.01, base_allocation * risk_adjustment)
    
    async def _rank_recommendations_optimized(
        self,
        recommendations: List[StockRecommendation]
    ) -> List[StockRecommendation]:
        """Memory-efficient ranking"""
        
        # Simple ranking based on key metrics
        for rec in recommendations:
            # Lightweight ranking score
            rec.ranking_score = (
                rec.technical_score * 0.4 +
                rec.confidence * 0.3 +
                (1 - rec.risk_score) * 0.3
            )
        
        # Sort by ranking score
        ranked = sorted(
            recommendations,
            key=lambda x: x.ranking_score,
            reverse=True
        )
        
        # Update priorities
        for i, rec in enumerate(ranked):
            rec.priority = min(10, max(1, 10 - i // 5))
        
        return ranked
    
    async def _optimize_recommendations_streaming(
        self,
        recommendations: List[StockRecommendation],
        risk_tolerance: str
    ) -> List[StockRecommendation]:
        """Streaming portfolio optimization to reduce memory usage"""
        
        if len(recommendations) < 2:
            return recommendations
        
        # Simple filtering based on risk tolerance
        risk_thresholds = {
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.8
        }
        
        max_risk = risk_thresholds.get(risk_tolerance, 0.5)
        
        # Filter and limit allocations
        optimized = []
        total_allocation = 0
        
        for rec in recommendations:
            if rec.risk_score <= max_risk and total_allocation < 0.8:  # Max 80% allocation
                # Adjust allocation based on remaining capacity
                remaining_capacity = 0.8 - total_allocation
                rec.recommended_allocation = min(rec.recommended_allocation, remaining_capacity)
                
                if rec.recommended_allocation > 0.01:  # Min 1%
                    optimized.append(rec)
                    total_allocation += rec.recommended_allocation
        
        return optimized
    
    async def _post_processing_cleanup(self):
        """Cleanup after processing batch"""
        # Force garbage collection
        collected = gc.collect()
        
        # Clear temporary caches if they're getting large
        if len(self._analysis_cache) > 800:
            # Remove oldest 20% of entries
            keys_to_remove = list(self._analysis_cache.keys())[:200]
            for key in keys_to_remove:
                self._analysis_cache.pop(key, None)
        
        if len(self._stock_data_cache) > 400:
            keys_to_remove = list(self._stock_data_cache.keys())[:100]
            for key in keys_to_remove:
                self._stock_data_cache.pop(key, None)
        
        logger.debug(f"Post-processing cleanup collected {collected} objects")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._processing_metrics:
            return {}
        
        recent_metrics = list(self._processing_metrics)[-10:]
        
        return {
            'recent_processing_time_avg_s': np.mean([m['processing_time_s'] for m in recent_metrics]),
            'recent_candidates_avg': np.mean([m['candidates_processed'] for m in recent_metrics]),
            'recent_recommendations_avg': np.mean([m['recommendations_generated'] for m in recent_metrics]),
            'cache_sizes': {
                'analysis_cache': len(self._analysis_cache),
                'stock_data_cache': len(self._stock_data_cache),
                'recommendation_history': len(self._recommendation_history)
            },
            'memory_optimization': 'enabled',
            'batch_processor_stats': self._batch_processor.get_statistics() if self._batch_processor else {}
        }
    
    async def shutdown(self):
        """Shutdown with comprehensive cleanup"""
        try:
            # Clear all caches
            self._analysis_cache.clear()
            self._stock_data_cache.clear()
            self._recommendation_history.clear()
            self._processing_metrics.clear()
            self._memory_usage_history.clear()
            self._processing_locks.clear()
            
            # Clear weak references
            self._technical_engine_ref = None
            self._fundamental_engine_ref = None
            self._sentiment_engine_ref = None
            self._model_manager_ref = None
            self._market_scanner_ref = None
            self._risk_manager_ref = None
            self._portfolio_optimizer_ref = None
            
            # Restore default memory settings
            if self._memory_manager:
                await self._memory_manager.restore_default_settings()
            
            # Force final garbage collection
            gc.collect()
            
            logger.info("Optimized recommendation engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during optimized engine shutdown: {e}")


# Global optimized engine instance
_optimized_engine: Optional[OptimizedRecommendationEngine] = None


async def get_optimized_recommendation_engine() -> OptimizedRecommendationEngine:
    """Get or create the global optimized recommendation engine"""
    global _optimized_engine
    if _optimized_engine is None:
        _optimized_engine = OptimizedRecommendationEngine()
        await _optimized_engine.initialize()
    return _optimized_engine