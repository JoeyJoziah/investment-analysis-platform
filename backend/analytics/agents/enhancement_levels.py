"""
Progressive enhancement system for LLM agent analysis
"""

import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from backend.analytics.recommendation_engine import StockRecommendation

logger = logging.getLogger(__name__)


class EnhancementLevel(Enum):
    """Enhancement complexity levels"""
    NONE = "none"                    # No agent enhancement
    BASIC = "basic"                  # Single agent, minimal cost
    STANDARD = "standard"            # Multiple agents, moderate cost  
    PREMIUM = "premium"              # Full analysis with debate
    COMPREHENSIVE = "comprehensive"   # Maximum analysis depth


@dataclass
class EnhancementCriteria:
    """Criteria for determining enhancement level"""
    
    # Confidence thresholds
    low_confidence_threshold: float = 0.3
    medium_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    
    # Signal strength thresholds
    weak_signal_threshold: float = 0.4
    strong_signal_threshold: float = 0.7
    
    # Market context thresholds
    high_volatility_threshold: float = 0.025
    volume_spike_threshold: float = 2.0
    news_significance_threshold: int = 3
    
    # Stock tier requirements
    tier_1_stocks: List[str] = None  # S&P 500, major indices
    tier_2_stocks: List[str] = None  # Mid-cap active stocks
    
    # Cost constraints
    max_daily_agent_budget: float = 8.33  # $250/month / 30 days
    max_single_analysis_cost: float = 2.50
    
    def __post_init__(self):
        if self.tier_1_stocks is None:
            self.tier_1_stocks = []
        if self.tier_2_stocks is None:
            self.tier_2_stocks = []


class ProgressiveEnhancement:
    """
    Determines appropriate enhancement level based on analysis context
    """
    
    def __init__(self, criteria: Optional[EnhancementCriteria] = None):
        self.criteria = criteria or EnhancementCriteria()
        
        # Enhancement level costs (USD)
        self.level_costs = {
            EnhancementLevel.NONE: 0.0,
            EnhancementLevel.BASIC: 0.15,      # Single agent
            EnhancementLevel.STANDARD: 0.45,   # 2-3 agents
            EnhancementLevel.PREMIUM: 1.20,    # Full analysis + debate
            EnhancementLevel.COMPREHENSIVE: 2.50  # Maximum depth
        }
        
        # Agent combinations for each level
        self.level_agents = {
            EnhancementLevel.NONE: [],
            EnhancementLevel.BASIC: ["news"],  # Most cost-effective
            EnhancementLevel.STANDARD: ["news", "fundamentals"],
            EnhancementLevel.PREMIUM: ["news", "fundamentals", "market"],
            EnhancementLevel.COMPREHENSIVE: ["news", "fundamentals", "market", "social"]
        }
        
    def determine_enhancement_level(
        self, 
        traditional_result: StockRecommendation,
        market_context: Dict,
        budget_available: float,
        ticker: str
    ) -> Tuple[EnhancementLevel, str, Dict]:
        """
        Determine the appropriate enhancement level
        
        Returns:
            Tuple of (enhancement_level, reasoning, enhancement_config)
        """
        
        reasoning_factors = []
        enhancement_score = 0
        
        # Factor 1: Traditional analysis confidence
        confidence = traditional_result.confidence
        if confidence < self.criteria.low_confidence_threshold:
            enhancement_score += 3
            reasoning_factors.append(f"Very low confidence ({confidence:.2f})")
        elif confidence < self.criteria.medium_confidence_threshold:
            enhancement_score += 2
            reasoning_factors.append(f"Low confidence ({confidence:.2f})")
        elif confidence > self.criteria.high_confidence_threshold:
            enhancement_score += 1
            reasoning_factors.append(f"High confidence needs validation ({confidence:.2f})")
        
        # Factor 2: Signal strength consistency
        signal_strength = self._assess_signal_strength(traditional_result)
        if signal_strength < self.criteria.weak_signal_threshold:
            enhancement_score += 2
            reasoning_factors.append(f"Weak signals ({signal_strength:.2f})")
        elif signal_strength > self.criteria.strong_signal_threshold:
            enhancement_score += 1
            reasoning_factors.append(f"Strong signals need confirmation ({signal_strength:.2f})")
        
        # Factor 3: Market volatility
        volatility = market_context.get('volatility', 0.0)
        if volatility > self.criteria.high_volatility_threshold:
            enhancement_score += 2
            reasoning_factors.append(f"High volatility ({volatility:.3f})")
        
        # Factor 4: Volume anomalies
        volume_ratio = market_context.get('volume_ratio', 1.0)
        if volume_ratio > self.criteria.volume_spike_threshold:
            enhancement_score += 1
            reasoning_factors.append(f"Volume spike ({volume_ratio:.1f}x)")
        
        # Factor 5: News significance
        news_count = market_context.get('news_count', 0)
        news_sentiment_volatility = market_context.get('news_sentiment_volatility', 0.0)
        if news_count >= self.criteria.news_significance_threshold:
            enhancement_score += 2
            reasoning_factors.append(f"Significant news volume ({news_count} articles)")
        if news_sentiment_volatility > 0.3:
            enhancement_score += 1
            reasoning_factors.append(f"News sentiment volatility ({news_sentiment_volatility:.2f})")
        
        # Factor 6: Earnings proximity
        days_to_earnings = market_context.get('days_to_earnings', 999)
        if days_to_earnings <= 3:
            enhancement_score += 3
            reasoning_factors.append(f"Earnings imminent ({days_to_earnings} days)")
        elif days_to_earnings <= 7:
            enhancement_score += 2
            reasoning_factors.append(f"Earnings week ({days_to_earnings} days)")
        elif days_to_earnings <= 14:
            enhancement_score += 1
            reasoning_factors.append(f"Pre-earnings period ({days_to_earnings} days)")
        
        # Factor 7: Stock tier and importance
        tier_bonus = self._get_tier_bonus(ticker)
        enhancement_score += tier_bonus
        if tier_bonus > 0:
            reasoning_factors.append(f"High-tier stock (tier {tier_bonus})")
        
        # Factor 8: Conflicting engine signals
        if self._has_conflicting_signals(traditional_result):
            enhancement_score += 2
            reasoning_factors.append("Conflicting analysis signals")
        
        # Factor 9: Market conditions (after hours, holidays, etc.)
        market_conditions = market_context.get('market_conditions', 'normal')
        if market_conditions in ['pre_market', 'after_hours']:
            enhancement_score += 1
            reasoning_factors.append(f"Extended hours trading ({market_conditions})")
        
        # Determine level based on score and budget
        level = self._score_to_level(enhancement_score, budget_available)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(level, enhancement_score, reasoning_factors, budget_available)
        
        # Create enhancement configuration
        config = self._create_enhancement_config(level, traditional_result, market_context)
        
        logger.debug(f"Enhancement level for {ticker}: {level.value} "
                    f"(score: {enhancement_score}, factors: {len(reasoning_factors)})")
        
        return level, reasoning, config
    
    def _assess_signal_strength(self, result: StockRecommendation) -> float:
        """Assess overall signal strength from traditional analysis"""
        try:
            # Get individual engine scores
            technical = getattr(result, 'technical_score', 0.5)
            fundamental = getattr(result, 'fundamental_score', 0.5)
            sentiment = getattr(result, 'sentiment_score', 0.5)
            
            scores = [technical, fundamental, sentiment]
            
            # Signal strength is the consistency and magnitude
            avg_score = sum(scores) / len(scores)
            score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            
            # Strong signals: high average, low variance
            # Weak signals: low average or high variance
            strength = avg_score * (1 - score_variance)
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            logger.warning(f"Error assessing signal strength: {e}")
            return 0.5
    
    def _get_tier_bonus(self, ticker: str) -> int:
        """Get tier bonus for enhancement scoring"""
        if ticker.upper() in self.criteria.tier_1_stocks:
            return 2  # S&P 500, major stocks
        elif ticker.upper() in self.criteria.tier_2_stocks:
            return 1  # Mid-cap active stocks
        else:
            return 0  # Lower tier stocks
    
    def _has_conflicting_signals(self, result: StockRecommendation) -> bool:
        """Check for conflicting signals between analysis engines"""
        try:
            technical = getattr(result, 'technical_score', 0.5)
            fundamental = getattr(result, 'fundamental_score', 0.5)
            sentiment = getattr(result, 'sentiment_score', 0.5)
            
            scores = [technical, fundamental, sentiment]
            
            # Significant disagreement if range > 0.4
            score_range = max(scores) - min(scores)
            return score_range > 0.4
            
        except Exception:
            return False
    
    def _score_to_level(self, score: int, budget_available: float) -> EnhancementLevel:
        """Convert enhancement score to level based on budget constraints"""
        
        # Budget constraints override scoring
        if budget_available < self.level_costs[EnhancementLevel.BASIC]:
            return EnhancementLevel.NONE
        
        # Score to level mapping
        if score == 0:
            return EnhancementLevel.NONE
        elif score <= 2:
            level = EnhancementLevel.BASIC
        elif score <= 4:
            level = EnhancementLevel.STANDARD
        elif score <= 7:
            level = EnhancementLevel.PREMIUM
        else:
            level = EnhancementLevel.COMPREHENSIVE
        
        # Downgrade if budget insufficient
        while self.level_costs[level] > budget_available:
            if level == EnhancementLevel.COMPREHENSIVE:
                level = EnhancementLevel.PREMIUM
            elif level == EnhancementLevel.PREMIUM:
                level = EnhancementLevel.STANDARD
            elif level == EnhancementLevel.STANDARD:
                level = EnhancementLevel.BASIC
            elif level == EnhancementLevel.BASIC:
                level = EnhancementLevel.NONE
                break
            else:
                break
        
        return level
    
    def _generate_reasoning(
        self, 
        level: EnhancementLevel,
        score: int,
        factors: List[str],
        budget: float
    ) -> str:
        """Generate human-readable reasoning for enhancement decision"""
        
        if level == EnhancementLevel.NONE:
            if budget < self.level_costs[EnhancementLevel.BASIC]:
                return f"No enhancement: insufficient budget (${budget:.2f} < ${self.level_costs[EnhancementLevel.BASIC]:.2f})"
            else:
                return f"No enhancement needed: low complexity (score: {score})"
        
        cost = self.level_costs[level]
        reasoning = f"{level.value.title()} enhancement (${cost:.2f}) - Score: {score}"
        
        if factors:
            reasoning += f" - Factors: {', '.join(factors[:3])}"  # Show top 3 factors
            if len(factors) > 3:
                reasoning += f" (+{len(factors)-3} more)"
        
        return reasoning
    
    def _create_enhancement_config(
        self, 
        level: EnhancementLevel,
        result: StockRecommendation,
        context: Dict
    ) -> Dict:
        """Create configuration for the selected enhancement level"""
        
        config = {
            'level': level.value,
            'estimated_cost': self.level_costs[level],
            'agents': self.level_agents[level].copy(),
            'analysis_type': self._get_analysis_type(level),
            'timeout': self._get_timeout(level),
            'parameters': {}
        }
        
        # Level-specific configuration
        if level == EnhancementLevel.BASIC:
            config['parameters'] = {
                'max_tokens': 1000,
                'temperature': 0.1,  # More deterministic
                'focus_areas': ['news_impact']
            }
        
        elif level == EnhancementLevel.STANDARD:
            config['parameters'] = {
                'max_tokens': 2000,
                'temperature': 0.2,
                'focus_areas': ['news_impact', 'valuation']
            }
        
        elif level == EnhancementLevel.PREMIUM:
            config['parameters'] = {
                'max_tokens': 3000,
                'temperature': 0.3,
                'focus_areas': ['news_impact', 'valuation', 'technical_patterns'],
                'enable_debate': True
            }
        
        elif level == EnhancementLevel.COMPREHENSIVE:
            config['parameters'] = {
                'max_tokens': 5000,
                'temperature': 0.3,
                'focus_areas': ['news_impact', 'valuation', 'technical_patterns', 'sentiment'],
                'enable_debate': True,
                'debate_rounds': 2
            }
        
        return config
    
    def _get_analysis_type(self, level: EnhancementLevel) -> str:
        """Get analysis type string for cost tracking"""
        mapping = {
            EnhancementLevel.NONE: 'traditional_only',
            EnhancementLevel.BASIC: 'single_agent',
            EnhancementLevel.STANDARD: 'multi_agent',
            EnhancementLevel.PREMIUM: 'bull_bear_debate',
            EnhancementLevel.COMPREHENSIVE: 'full_analysis'
        }
        return mapping.get(level, 'single_agent')
    
    def _get_timeout(self, level: EnhancementLevel) -> float:
        """Get timeout for analysis level"""
        timeouts = {
            EnhancementLevel.NONE: 0.0,
            EnhancementLevel.BASIC: 30.0,
            EnhancementLevel.STANDARD: 60.0,
            EnhancementLevel.PREMIUM: 120.0,
            EnhancementLevel.COMPREHENSIVE: 180.0
        }
        return timeouts.get(level, 30.0)
    
    def get_level_info(self) -> Dict[str, Dict]:
        """Get information about all enhancement levels"""
        return {
            level.value: {
                'cost': self.level_costs[level],
                'agents': self.level_agents[level],
                'analysis_type': self._get_analysis_type(level),
                'timeout': self._get_timeout(level),
                'description': self._get_level_description(level)
            }
            for level in EnhancementLevel
        }
    
    def _get_level_description(self, level: EnhancementLevel) -> str:
        """Get description for enhancement level"""
        descriptions = {
            EnhancementLevel.NONE: "Traditional analysis only, no LLM enhancement",
            EnhancementLevel.BASIC: "Single agent analysis for critical insights",
            EnhancementLevel.STANDARD: "Multi-agent analysis with complementary perspectives",
            EnhancementLevel.PREMIUM: "Comprehensive analysis with agent debate",
            EnhancementLevel.COMPREHENSIVE: "Maximum depth analysis with full agent suite"
        }
        return descriptions.get(level, "Unknown enhancement level")
    
    def update_criteria(self, **kwargs):
        """Update enhancement criteria"""
        for key, value in kwargs.items():
            if hasattr(self.criteria, key):
                setattr(self.criteria, key, value)
                logger.info(f"Updated enhancement criteria: {key} = {value}")
            else:
                logger.warning(f"Unknown enhancement criteria: {key}")
    
    def update_costs(self, cost_updates: Dict[EnhancementLevel, float]):
        """Update level costs"""
        for level, cost in cost_updates.items():
            if level in self.level_costs:
                self.level_costs[level] = cost
                logger.info(f"Updated {level.value} cost to ${cost:.2f}")
            else:
                logger.warning(f"Unknown enhancement level: {level}")
    
    def get_cost_estimate(self, level: EnhancementLevel) -> float:
        """Get cost estimate for enhancement level"""
        return self.level_costs.get(level, 0.0)
    
    def get_agents_for_level(self, level: EnhancementLevel) -> List[str]:
        """Get agents used for enhancement level"""
        return self.level_agents.get(level, []).copy()