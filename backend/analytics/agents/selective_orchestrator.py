import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from backend.analytics.recommendation_engine import StockRecommendation
from .cache_aware_agents import CacheAwareTradingAgents

logger = logging.getLogger(__name__)


class AnalysisComplexity(Enum):
    """Analysis complexity levels"""
    BASIC = "basic"      # Single agent, minimal analysis
    ENHANCED = "enhanced"  # Multiple agents, moderate analysis  
    PREMIUM = "premium"    # Full debate system, comprehensive analysis


@dataclass
class AgentSelectionCriteria:
    """Criteria for selecting which agents to run"""
    confidence_threshold: float = 0.5
    sentiment_threshold: float = 0.3
    volatility_threshold: float = 0.02
    volume_spike_threshold: float = 2.0
    news_volume_threshold: int = 5
    earnings_window_days: int = 7
    tier_level: int = 1


class SelectiveAgentOrchestrator:
    """
    Orchestrates selective activation of TradingAgents based on context and budget
    """
    
    def __init__(self, trading_agents: CacheAwareTradingAgents):
        self.trading_agents = trading_agents
        self.selection_criteria = AgentSelectionCriteria()
        
        # Agent priority mapping (higher priority = run first)
        self.agent_priorities = {
            'news': 10,         # High priority - market moving
            'fundamentals': 8,  # High priority - reliable
            'market': 6,        # Medium priority - technical
            'social': 4         # Lower priority - noisy
        }
        
    async def should_use_agents(
        self, 
        traditional_result: StockRecommendation,
        ticker: str,
        market_context: Dict = None
    ) -> Tuple[bool, str, AnalysisComplexity]:
        """
        Determine if LLM agents should be used and at what complexity level
        
        Returns:
            Tuple of (should_use, reason, complexity_level)
        """
        reasons = []
        complexity_factors = 0
        
        # Check budget first
        can_afford, budget_reason = await self.trading_agents.budget_manager.can_afford_analysis('single_agent')
        if not can_afford:
            return False, f"Budget constraint: {budget_reason}", AnalysisComplexity.BASIC
            
        market_context = market_context or {}
        
        # Factor 1: Traditional analysis confidence (lower confidence = need LLM help)
        if traditional_result.confidence < self.selection_criteria.confidence_threshold:
            reasons.append(f"Low confidence ({traditional_result.confidence:.2f})")
            complexity_factors += 2
            
        # Factor 2: Conflicting signals between analysis engines
        if self._has_conflicting_signals(traditional_result):
            reasons.append("Conflicting analysis signals")
            complexity_factors += 3
            
        # Factor 3: High sentiment uncertainty
        if hasattr(traditional_result, 'sentiment_score'):
            if traditional_result.sentiment_score < self.selection_criteria.sentiment_threshold:
                reasons.append(f"Weak sentiment data ({traditional_result.sentiment_score:.2f})")
                complexity_factors += 1
                
        # Factor 4: High volatility periods
        if market_context.get('volatility', 0) > self.selection_criteria.volatility_threshold:
            reasons.append(f"High volatility ({market_context['volatility']:.3f})")
            complexity_factors += 2
            
        # Factor 5: Volume spikes (unusual market activity)
        if market_context.get('volume_ratio', 1.0) > self.selection_criteria.volume_spike_threshold:
            reasons.append(f"Volume spike ({market_context['volume_ratio']:.1f}x)")
            complexity_factors += 1
            
        # Factor 6: High news volume
        if market_context.get('news_count', 0) > self.selection_criteria.news_volume_threshold:
            reasons.append(f"High news volume ({market_context['news_count']} articles)")
            complexity_factors += 2
            
        # Factor 7: Earnings period
        if market_context.get('days_to_earnings', 999) <= self.selection_criteria.earnings_window_days:
            reasons.append(f"Earnings period ({market_context['days_to_earnings']} days)")
            complexity_factors += 3
            
        # Factor 8: Stock tier (only use agents for important stocks)
        stock_tier = market_context.get('tier', 5)
        if stock_tier > 2:  # Only tier 1-2 stocks get agent analysis
            return False, f"Stock tier too low ({stock_tier})", AnalysisComplexity.BASIC
        elif stock_tier == 1:
            complexity_factors += 1
            
        # Factor 9: Market hours vs after-hours
        if market_context.get('market_hours', True):
            complexity_factors += 1
            
        # Decision logic
        if complexity_factors == 0:
            return False, "No complexity factors detected", AnalysisComplexity.BASIC
        elif complexity_factors <= 3:
            complexity = AnalysisComplexity.BASIC
        elif complexity_factors <= 6:
            complexity = AnalysisComplexity.ENHANCED  
        else:
            complexity = AnalysisComplexity.PREMIUM
            
        reason = f"Complexity factors ({complexity_factors}): {', '.join(reasons)}"
        return True, reason, complexity
        
    async def select_agents_for_context(
        self, 
        traditional_result: StockRecommendation,
        complexity: AnalysisComplexity,
        market_context: Dict = None
    ) -> List[str]:
        """
        Select specific agents based on analysis needs and context
        """
        selected_agents = []
        market_context = market_context or {}
        
        if complexity == AnalysisComplexity.BASIC:
            # Single most relevant agent
            selected_agents = [self._get_most_relevant_agent(traditional_result, market_context)]
            
        elif complexity == AnalysisComplexity.ENHANCED:
            # Multiple complementary agents
            agents = self._get_complementary_agents(traditional_result, market_context)
            selected_agents = agents[:2]  # Limit to 2 agents for cost control
            
        else:  # PREMIUM
            # Full agent suite with debate
            agents = self._get_comprehensive_agents(traditional_result, market_context)
            selected_agents = agents[:3]  # Limit to 3 agents + debate
            
        logger.info(f"Selected agents for {complexity.value} analysis: {selected_agents}")
        return selected_agents
    
    def _has_conflicting_signals(self, traditional_result: StockRecommendation) -> bool:
        """Check if traditional analysis engines have conflicting signals"""
        try:
            # Get individual engine scores
            technical_score = getattr(traditional_result, 'technical_score', 0.5)
            fundamental_score = getattr(traditional_result, 'fundamental_score', 0.5) 
            sentiment_score = getattr(traditional_result, 'sentiment_score', 0.5)
            
            scores = [technical_score, fundamental_score, sentiment_score]
            
            # Check for significant disagreement (>0.3 spread between min/max)
            score_spread = max(scores) - min(scores)
            conflicting = score_spread > 0.3
            
            if conflicting:
                logger.debug(f"Conflicting signals detected - spread: {score_spread:.2f}")
                
            return conflicting
            
        except Exception as e:
            logger.warning(f"Error checking conflicting signals: {e}")
            return False
    
    def _get_most_relevant_agent(
        self, 
        traditional_result: StockRecommendation,
        market_context: Dict
    ) -> str:
        """Get single most relevant agent based on context"""
        
        # Priority logic for single agent selection
        if market_context.get('news_count', 0) > 3:
            return 'news'
        elif market_context.get('days_to_earnings', 999) <= 7:
            return 'fundamentals' 
        elif market_context.get('volume_ratio', 1.0) > 1.5:
            return 'market'
        elif hasattr(traditional_result, 'sentiment_score') and traditional_result.sentiment_score < 0.3:
            return 'social'
        else:
            # Default to fundamentals for stable analysis
            return 'fundamentals'
    
    def _get_complementary_agents(
        self, 
        traditional_result: StockRecommendation,
        market_context: Dict
    ) -> List[str]:
        """Get complementary agents that address analysis gaps"""
        agents = []
        
        # Always include news if there's significant news volume
        if market_context.get('news_count', 0) > 2:
            agents.append('news')
            
        # Include fundamentals for earnings or valuation questions
        if (market_context.get('days_to_earnings', 999) <= 14 or 
            getattr(traditional_result, 'fundamental_score', 0.5) > 0.7):
            agents.append('fundamentals')
            
        # Include market/technical for high volatility
        if market_context.get('volatility', 0) > 0.015:
            agents.append('market')
            
        # Include social for sentiment gaps
        if (hasattr(traditional_result, 'sentiment_score') and 
            traditional_result.sentiment_score < 0.4):
            agents.append('social')
        
        # Sort by priority and return top 2
        agents = sorted(agents, key=lambda x: self.agent_priorities.get(x, 0), reverse=True)
        
        # Ensure we have at least 2 agents
        if len(agents) < 2:
            remaining = [a for a in ['news', 'fundamentals', 'market'] if a not in agents]
            agents.extend(remaining[:2-len(agents)])
            
        return agents[:2]
    
    def _get_comprehensive_agents(
        self, 
        traditional_result: StockRecommendation,
        market_context: Dict
    ) -> List[str]:
        """Get comprehensive agent set for premium analysis"""
        agents = []
        
        # Include most relevant agents based on context
        if market_context.get('news_count', 0) > 0:
            agents.append('news')
            
        # Always include fundamentals for comprehensive analysis
        agents.append('fundamentals')
        
        # Include market analysis for technical perspective
        agents.append('market')
        
        # Include social if sentiment is relevant
        if market_context.get('social_activity', 0) > 0:
            agents.append('social')
            
        return agents[:3]  # Limit to 3 for cost control
    
    async def run_contextual_analysis(
        self,
        ticker: str,
        traditional_result: StockRecommendation,
        market_context: Dict = None
    ) -> Optional[Dict]:
        """
        Run contextual LLM analysis based on traditional analysis gaps
        """
        try:
            # Determine if we should use agents and at what level
            should_use, reason, complexity = await self.should_use_agents(
                traditional_result, ticker, market_context
            )
            
            if not should_use:
                logger.info(f"Skipping agents for {ticker}: {reason}")
                return None
                
            logger.info(f"Running {complexity.value} agent analysis for {ticker}: {reason}")
            
            # Select appropriate agents
            selected_agents = await self.select_agents_for_context(
                traditional_result, complexity, market_context
            )
            
            # Determine analysis type for cost tracking
            analysis_type = self._get_analysis_type(complexity, selected_agents)
            
            # Run the analysis with selected agents
            analysis_result, cost = await self.trading_agents.analyze_stock_with_budget(
                ticker=ticker,
                analysis_type=analysis_type,
                context={
                    'traditional_result': traditional_result.__dict__,
                    'market_context': market_context,
                    'selected_agents': selected_agents,
                    'complexity': complexity.value
                }
            )
            
            # Enhance result with metadata
            enhanced_result = {
                'agent_analysis': analysis_result,
                'metadata': {
                    'complexity_level': complexity.value,
                    'selected_agents': selected_agents,
                    'selection_reason': reason,
                    'cost_incurred': cost,
                    'analysis_type': analysis_type,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            logger.info(f"Completed agent analysis for {ticker} - "
                       f"Cost: ${cost:.4f}, Agents: {selected_agents}")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Contextual analysis failed for {ticker}: {e}")
            return None
    
    def _get_analysis_type(self, complexity: AnalysisComplexity, agents: List[str]) -> str:
        """Get analysis type string for cost tracking"""
        if complexity == AnalysisComplexity.BASIC:
            if len(agents) == 1:
                agent_map = {
                    'news': 'news_analysis',
                    'fundamentals': 'fundamentals_analysis', 
                    'market': 'technical_analysis',
                    'social': 'sentiment_analysis'
                }
                return agent_map.get(agents[0], 'single_agent')
            else:
                return 'single_agent'
        elif complexity == AnalysisComplexity.ENHANCED:
            return 'bull_bear_debate' if len(agents) >= 2 else 'single_agent'
        else:  # PREMIUM
            return 'full_analysis'
    
    async def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about agent selection patterns"""
        # This would typically query Redis or a database for historical data
        # For now, return current configuration
        return {
            'selection_criteria': {
                'confidence_threshold': self.selection_criteria.confidence_threshold,
                'sentiment_threshold': self.selection_criteria.sentiment_threshold,
                'volatility_threshold': self.selection_criteria.volatility_threshold,
                'volume_spike_threshold': self.selection_criteria.volume_spike_threshold,
                'news_volume_threshold': self.selection_criteria.news_volume_threshold,
                'earnings_window_days': self.selection_criteria.earnings_window_days
            },
            'agent_priorities': self.agent_priorities,
            'complexity_levels': {level.value: level.name for level in AnalysisComplexity}
        }
        
    def update_selection_criteria(self, **kwargs):
        """Update agent selection criteria"""
        for key, value in kwargs.items():
            if hasattr(self.selection_criteria, key):
                setattr(self.selection_criteria, key, value)
                logger.info(f"Updated selection criteria: {key} = {value}")
            else:
                logger.warning(f"Unknown selection criteria: {key}")
                
    def update_agent_priorities(self, priorities: Dict[str, int]):
        """Update agent priority mapping"""
        for agent, priority in priorities.items():
            if agent in self.agent_priorities:
                self.agent_priorities[agent] = priority
                logger.info(f"Updated agent priority: {agent} = {priority}")
            else:
                logger.warning(f"Unknown agent: {agent}")