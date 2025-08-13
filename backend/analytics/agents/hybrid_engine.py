import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from backend.analytics.recommendation_engine import RecommendationEngine, StockRecommendation
from backend.utils.llm_budget_manager import LLMBudgetManager, BudgetExceededException, LLMCircuitBreaker
from backend.utils.cache_manager import CacheManager
from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

from .cache_aware_agents import CacheAwareTradingAgents
from .selective_orchestrator import SelectiveAgentOrchestrator, AnalysisComplexity

logger = logging.getLogger(__name__)


@dataclass
class EnhancedStockRecommendation(StockRecommendation):
    """Enhanced recommendation that includes LLM agent insights"""
    
    # Agent-specific fields
    agent_analysis: Optional[Dict] = None
    agent_confidence: Optional[float] = None
    agent_reasoning: Optional[str] = None
    agent_consensus: Optional[str] = None
    
    # Cost and performance metadata
    analysis_cost: float = 0.0
    analysis_duration: float = 0.0
    agents_used: List[str] = None
    complexity_level: str = "traditional"
    
    # Hybrid scoring
    hybrid_score: Optional[float] = None
    confidence_boost: float = 0.0
    
    def __post_init__(self):
        if self.agents_used is None:
            self.agents_used = []
        
        # Calculate hybrid score if agent analysis available
        if self.agent_analysis and hasattr(self, 'overall_score'):
            self.hybrid_score = self._calculate_hybrid_score()
    
    def _calculate_hybrid_score(self) -> float:
        """Calculate hybrid score combining traditional and agent analysis"""
        if not self.agent_analysis:
            return self.overall_score
        
        # Get agent confidence (0-1 scale)
        agent_conf = self.agent_confidence or 0.5
        
        # Weight traditional vs agent analysis based on agent confidence
        agent_weight = min(agent_conf, 0.4)  # Max 40% weight to agents
        traditional_weight = 1.0 - agent_weight
        
        # Assume agent provides a score (this would need to be extracted from actual agent output)
        agent_score = self._extract_agent_score()
        
        hybrid = (traditional_weight * self.overall_score + 
                 agent_weight * agent_score)
        
        return max(0.0, min(1.0, hybrid))  # Clamp to [0,1]
    
    def _extract_agent_score(self) -> float:
        """Extract numerical score from agent analysis"""
        if not self.agent_analysis:
            return 0.5
            
        # This would need to parse actual agent output format
        # For now, assume agent provides a recommendation mapping
        agent_rec = self.agent_analysis.get('recommendation', 'HOLD').upper()
        
        score_mapping = {
            'STRONG_BUY': 0.9,
            'BUY': 0.7,
            'HOLD': 0.5,
            'SELL': 0.3,
            'STRONG_SELL': 0.1
        }
        
        return score_mapping.get(agent_rec, 0.5)


class AnalysisMode(Enum):
    """Analysis execution modes"""
    TRADITIONAL_ONLY = "traditional_only"
    SELECTIVE_HYBRID = "selective_hybrid"
    ALWAYS_HYBRID = "always_hybrid"
    AGENT_ONLY = "agent_only"


class HybridAnalysisEngine:
    """
    Intelligent orchestrator that combines traditional ML with selective LLM agents
    """
    
    def __init__(
        self,
        traditional_engine: RecommendationEngine,
        smart_fetcher: SmartDataFetcher,
        cache_manager: CacheManager,
        budget_manager: LLMBudgetManager,
        analysis_mode: AnalysisMode = AnalysisMode.SELECTIVE_HYBRID
    ):
        self.traditional_engine = traditional_engine
        self.smart_fetcher = smart_fetcher
        self.cache_manager = cache_manager
        self.budget_manager = budget_manager
        self.analysis_mode = analysis_mode
        
        # Initialize LLM components
        self.trading_agents = CacheAwareTradingAgents(
            smart_fetcher=smart_fetcher,
            cache_manager=cache_manager,
            budget_manager=budget_manager
        )
        
        self.agent_orchestrator = SelectiveAgentOrchestrator(self.trading_agents)
        self.circuit_breaker = LLMCircuitBreaker(budget_manager)
        
        # Performance tracking
        self.stats = {
            'total_analyses': 0,
            'agent_analyses': 0,
            'budget_blocks': 0,
            'circuit_breaker_trips': 0,
            'avg_cost_per_analysis': 0.0,
            'avg_duration': 0.0
        }
        
        logger.info(f"HybridAnalysisEngine initialized in {analysis_mode.value} mode")
    
    async def analyze_stock(
        self, 
        ticker: str,
        force_agents: bool = False,
        analysis_timeout: float = 120.0
    ) -> EnhancedStockRecommendation:
        """
        Analyze stock using hybrid traditional + LLM approach
        
        Args:
            ticker: Stock ticker symbol
            force_agents: Force use of LLM agents regardless of selection criteria
            analysis_timeout: Maximum time to spend on analysis (seconds)
            
        Returns:
            Enhanced stock recommendation with potential agent insights
        """
        start_time = datetime.utcnow()
        analysis_cost = 0.0
        agents_used = []
        agent_analysis = None
        
        try:
            logger.info(f"Starting hybrid analysis for {ticker} in {self.analysis_mode.value} mode")
            
            # Step 1: Always run traditional analysis (it's free and fast)
            traditional_result = await asyncio.wait_for(
                self.traditional_engine.analyze_stock(ticker),
                timeout=analysis_timeout * 0.6  # Reserve 60% time for traditional
            )
            
            logger.debug(f"Traditional analysis complete for {ticker} - "
                        f"Score: {traditional_result.overall_score:.2f}, "
                        f"Confidence: {traditional_result.confidence:.2f}")
            
            # Step 2: Determine if we should enhance with agents
            should_use_agents = await self._should_use_agents(
                traditional_result, ticker, force_agents
            )
            
            if should_use_agents and self.analysis_mode != AnalysisMode.TRADITIONAL_ONLY:
                # Step 3: Run agent analysis with circuit breaker protection
                remaining_time = analysis_timeout * 0.4  # Reserve 40% time for agents
                agent_analysis, analysis_cost, agents_used = await self._run_agent_analysis(
                    ticker, traditional_result, remaining_time
                )
            
            # Step 4: Create enhanced recommendation
            enhanced_result = await self._create_enhanced_recommendation(
                traditional_result, agent_analysis, analysis_cost, agents_used, start_time
            )
            
            # Step 5: Update statistics
            await self._update_stats(analysis_cost, start_time, len(agents_used) > 0)
            
            logger.info(f"Hybrid analysis complete for {ticker} - "
                       f"Duration: {enhanced_result.analysis_duration:.1f}s, "
                       f"Cost: ${analysis_cost:.4f}, "
                       f"Agents: {agents_used}")
            
            return enhanced_result
            
        except asyncio.TimeoutError:
            logger.warning(f"Analysis timeout for {ticker} after {analysis_timeout}s")
            return await self._create_fallback_recommendation(traditional_result, start_time)
            
        except BudgetExceededException as e:
            logger.warning(f"Budget exceeded for {ticker}: {e}")
            self.stats['budget_blocks'] += 1
            return await self._create_fallback_recommendation(traditional_result, start_time)
            
        except Exception as e:
            logger.error(f"Hybrid analysis failed for {ticker}: {e}")
            # Fall back to traditional analysis
            if 'traditional_result' in locals():
                return await self._create_fallback_recommendation(traditional_result, start_time)
            else:
                # Even traditional failed - return minimal recommendation
                return self._create_error_recommendation(ticker, str(e), start_time)
    
    async def _should_use_agents(
        self, 
        traditional_result: StockRecommendation,
        ticker: str,
        force_agents: bool = False
    ) -> bool:
        """Determine if LLM agents should be used"""
        if force_agents:
            return True
            
        if self.analysis_mode == AnalysisMode.TRADITIONAL_ONLY:
            return False
        elif self.analysis_mode == AnalysisMode.ALWAYS_HYBRID:
            # Still check budget
            can_afford, _ = await self.budget_manager.can_afford_analysis('single_agent')
            return can_afford
        elif self.analysis_mode == AnalysisMode.AGENT_ONLY:
            can_afford, _ = await self.budget_manager.can_afford_analysis('single_agent')
            return can_afford
        else:  # SELECTIVE_HYBRID
            # Get market context for decision
            market_context = await self._get_market_context(ticker)
            
            should_use, reason, complexity = await self.agent_orchestrator.should_use_agents(
                traditional_result, ticker, market_context
            )
            
            if should_use:
                logger.debug(f"Agents selected for {ticker}: {reason}")
            
            return should_use
    
    async def _run_agent_analysis(
        self, 
        ticker: str, 
        traditional_result: StockRecommendation,
        timeout: float
    ) -> tuple[Optional[Dict], float, List[str]]:
        """Run agent analysis with error handling"""
        try:
            # Get market context
            market_context = await self._get_market_context(ticker)
            
            # Run with circuit breaker protection
            agent_result = await asyncio.wait_for(
                self.circuit_breaker.call_with_circuit_breaker(
                    self.agent_orchestrator.run_contextual_analysis,
                    ticker=ticker,
                    traditional_result=traditional_result,
                    market_context=market_context
                ),
                timeout=timeout
            )
            
            if agent_result:
                return (
                    agent_result.get('agent_analysis'),
                    agent_result.get('metadata', {}).get('cost_incurred', 0.0),
                    agent_result.get('metadata', {}).get('selected_agents', [])
                )
            
        except asyncio.TimeoutError:
            logger.warning(f"Agent analysis timeout for {ticker}")
        except BudgetExceededException as e:
            logger.warning(f"Agent analysis budget exceeded for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Agent analysis error for {ticker}: {e}")
            if "Circuit breaker" in str(e):
                self.stats['circuit_breaker_trips'] += 1
        
        return None, 0.0, []
    
    async def _get_market_context(self, ticker: str) -> Dict[str, Any]:
        """Get market context for agent decision making"""
        try:
            # This would typically gather various market indicators
            # For now, return basic context
            context = {
                'tier': 1,  # Assume tier 1 stock for now
                'market_hours': True,
                'volatility': 0.02,  # Placeholder
                'volume_ratio': 1.0,  # Placeholder
                'news_count': 0,  # Would be fetched from news API
                'days_to_earnings': 999,  # Would be calculated from earnings calendar
                'social_activity': 0  # Would be fetched from social APIs
            }
            
            # In a real implementation, you'd fetch:
            # - Recent volatility from price data
            # - Volume ratios from market data
            # - News article counts
            # - Earnings calendar
            # - Social media activity levels
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get market context for {ticker}: {e}")
            return {}
    
    async def _create_enhanced_recommendation(
        self,
        traditional_result: StockRecommendation,
        agent_analysis: Optional[Dict],
        analysis_cost: float,
        agents_used: List[str],
        start_time: datetime
    ) -> EnhancedStockRecommendation:
        """Create enhanced recommendation combining traditional and agent analysis"""
        
        analysis_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Convert traditional result to enhanced format
        enhanced_data = asdict(traditional_result)
        
        # Add agent-specific fields
        enhanced_data.update({
            'agent_analysis': agent_analysis,
            'agent_confidence': self._extract_agent_confidence(agent_analysis),
            'agent_reasoning': self._extract_agent_reasoning(agent_analysis),
            'agent_consensus': self._extract_agent_consensus(agent_analysis),
            'analysis_cost': analysis_cost,
            'analysis_duration': analysis_duration,
            'agents_used': agents_used,
            'complexity_level': "hybrid" if agent_analysis else "traditional"
        })
        
        enhanced_result = EnhancedStockRecommendation(**enhanced_data)
        
        # Calculate confidence boost from agent analysis
        if agent_analysis and enhanced_result.agent_confidence:
            confidence_boost = self._calculate_confidence_boost(
                traditional_result.confidence,
                enhanced_result.agent_confidence
            )
            enhanced_result.confidence_boost = confidence_boost
            
            # Apply boost to overall confidence
            enhanced_result.confidence = min(1.0, traditional_result.confidence + confidence_boost)
        
        return enhanced_result
    
    def _extract_agent_confidence(self, agent_analysis: Optional[Dict]) -> Optional[float]:
        """Extract confidence score from agent analysis"""
        if not agent_analysis:
            return None
        
        # This would parse the actual agent output format
        # For now, assume agents provide confidence scores
        return agent_analysis.get('confidence', 0.5)
    
    def _extract_agent_reasoning(self, agent_analysis: Optional[Dict]) -> Optional[str]:
        """Extract reasoning from agent analysis"""
        if not agent_analysis:
            return None
        
        return agent_analysis.get('reasoning', 'No agent reasoning available')
    
    def _extract_agent_consensus(self, agent_analysis: Optional[Dict]) -> Optional[str]:
        """Extract consensus recommendation from agent analysis"""
        if not agent_analysis:
            return None
            
        return agent_analysis.get('recommendation', 'HOLD')
    
    def _calculate_confidence_boost(
        self, 
        traditional_confidence: float, 
        agent_confidence: float
    ) -> float:
        """Calculate confidence boost from agent analysis"""
        # If agents agree with traditional analysis and are confident, boost confidence
        # If agents disagree, don't boost (or even reduce)
        
        # For now, simple heuristic: boost if agent confidence > traditional
        if agent_confidence > traditional_confidence:
            boost = (agent_confidence - traditional_confidence) * 0.2  # Max 20% boost
            return max(0.0, min(0.2, boost))
        
        return 0.0
    
    async def _create_fallback_recommendation(
        self, 
        traditional_result: StockRecommendation,
        start_time: datetime
    ) -> EnhancedStockRecommendation:
        """Create fallback recommendation using only traditional analysis"""
        analysis_duration = (datetime.utcnow() - start_time).total_seconds()
        
        enhanced_data = asdict(traditional_result)
        enhanced_data.update({
            'analysis_duration': analysis_duration,
            'complexity_level': "traditional_fallback",
            'agents_used': []
        })
        
        return EnhancedStockRecommendation(**enhanced_data)
    
    def _create_error_recommendation(
        self, 
        ticker: str, 
        error: str,
        start_time: datetime
    ) -> EnhancedStockRecommendation:
        """Create minimal recommendation when analysis fails"""
        analysis_duration = (datetime.utcnow() - start_time).total_seconds()
        
        return EnhancedStockRecommendation(
            ticker=ticker,
            recommendation="HOLD",
            overall_score=0.5,
            confidence=0.1,
            target_price=0.0,
            risks=["Analysis failed: " + error],
            opportunities=[],
            analysis_duration=analysis_duration,
            complexity_level="error",
            agents_used=[]
        )
    
    async def _update_stats(self, cost: float, start_time: datetime, used_agents: bool):
        """Update performance statistics"""
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        self.stats['total_analyses'] += 1
        if used_agents:
            self.stats['agent_analyses'] += 1
        
        # Update running averages
        total = self.stats['total_analyses']
        self.stats['avg_cost_per_analysis'] = (
            (self.stats['avg_cost_per_analysis'] * (total - 1) + cost) / total
        )
        self.stats['avg_duration'] = (
            (self.stats['avg_duration'] * (total - 1) + duration) / total
        )
    
    async def batch_analyze_stocks(
        self, 
        tickers: List[str],
        max_concurrent: int = 5,
        prioritize_by_tier: bool = True
    ) -> Dict[str, EnhancedStockRecommendation]:
        """
        Analyze multiple stocks efficiently with resource management
        """
        if prioritize_by_tier:
            # Sort tickers by tier (this would need tier data)
            # For now, analyze in provided order
            pass
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(ticker: str) -> tuple[str, EnhancedStockRecommendation]:
            async with semaphore:
                try:
                    result = await self.analyze_stock(ticker)
                    return ticker, result
                except Exception as e:
                    logger.error(f"Batch analysis failed for {ticker}: {e}")
                    return ticker, self._create_error_recommendation(
                        ticker, str(e), datetime.utcnow()
                    )
        
        # Run analyses concurrently
        tasks = [analyze_with_semaphore(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary
        analysis_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch analysis exception: {result}")
                continue
                
            ticker, recommendation = result
            analysis_results[ticker] = recommendation
        
        logger.info(f"Batch analysis complete: {len(analysis_results)}/{len(tickers)} successful")
        return analysis_results
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status and statistics"""
        budget_status = await self.budget_manager.get_budget_status()
        agent_capabilities = await self.trading_agents.get_agent_capabilities()
        
        return {
            'analysis_mode': self.analysis_mode.value,
            'performance_stats': self.stats.copy(),
            'budget_status': budget_status,
            'agent_capabilities': agent_capabilities,
            'circuit_breaker_state': self.circuit_breaker.state,
            'uptime': datetime.utcnow().isoformat()
        }
    
    def set_analysis_mode(self, mode: AnalysisMode):
        """Change analysis mode"""
        self.analysis_mode = mode
        logger.info(f"Analysis mode changed to: {mode.value}")
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.trading_agents.cleanup()
        except Exception as e:
            logger.error(f"Error during hybrid engine cleanup: {e}")