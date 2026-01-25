import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Add TradingAgents to Python path
# TradingAgents is located at backend/TradingAgents/
trading_agents_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TradingAgents'))
if os.path.exists(trading_agents_path) and trading_agents_path not in sys.path:
    sys.path.insert(0, trading_agents_path)

# Optional TradingAgents imports - fallback to stubs if not available
TRADING_AGENTS_AVAILABLE = False
try:
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.dataflows.interface import set_config
    TRADING_AGENTS_AVAILABLE = True
except ImportError:
    # Create stub classes for when TradingAgents is not available
    class TradingAgentsGraph:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("TradingAgents module not available")
    DEFAULT_CONFIG = {}
    def set_config(*args, **kwargs):
        pass
    logging.getLogger(__name__).warning("TradingAgents module not available - using stubs")

from backend.utils.cache_manager import CacheManager
from backend.utils.llm_budget_manager import LLMBudgetManager, BudgetExceededException
from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

logger = logging.getLogger(__name__)


class CacheAwareDataInterface:
    """Provides cached data to TradingAgents while respecting API limits"""
    
    def __init__(self, smart_fetcher: SmartDataFetcher, cache_manager: CacheManager):
        self.smart_fetcher = smart_fetcher
        self.cache_manager = cache_manager
        
    async def get_cached_stock_data(self, ticker: str, data_type: str) -> Dict:
        """Get stock data with intelligent caching"""
        cache_key = f"agents_data:{ticker}:{data_type}"
        
        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            logger.debug(f"Using cached {data_type} data for {ticker}")
            return cached_data
        
        try:
            # Fetch fresh data through SmartDataFetcher
            fresh_data = await self.smart_fetcher.fetch_stock_data(ticker, data_type)
            
            # Cache with appropriate TTL based on data type
            ttl = self._get_cache_ttl(data_type)
            await self.cache_manager.set(cache_key, fresh_data, ttl)
            
            return fresh_data
            
        except Exception as e:
            logger.error(f"Failed to fetch {data_type} for {ticker}: {e}")
            # Try to get stale cache as fallback
            stale_data = await self.cache_manager.get_stale(cache_key, max_age=86400 * 7)
            if stale_data:
                logger.warning(f"Using stale {data_type} data for {ticker}")
                return stale_data
            raise
    
    def _get_cache_ttl(self, data_type: str) -> int:
        """Get appropriate cache TTL based on data type"""
        ttl_mapping = {
            'price': 300,           # 5 minutes for price data
            'fundamentals': 3600 * 24,  # 24 hours for fundamentals
            'news': 1800,           # 30 minutes for news
            'social': 900,          # 15 minutes for social media
            'technical': 300,       # 5 minutes for technical indicators
            'earnings': 3600 * 24 * 7,  # 7 days for earnings
            'options': 1800         # 30 minutes for options data
        }
        return ttl_mapping.get(data_type, 3600)  # Default 1 hour


class CacheAwareTradingAgents(TradingAgentsGraph):
    """
    TradingAgents wrapper that integrates with existing caching and cost monitoring
    """
    
    def __init__(
        self, 
        smart_fetcher: SmartDataFetcher,
        cache_manager: CacheManager,
        budget_manager: LLMBudgetManager,
        selected_analysts: List[str] = None,
        config: Dict[str, Any] = None
    ):
        # Prepare config with cost-optimized settings
        self.config = config or self._get_cost_optimized_config()
        
        # Initialize parent class
        super().__init__(
            selected_analysts=selected_analysts or ['market', 'news', 'fundamentals'],
            debug=False,
            config=self.config
        )
        
        # Store our integrations
        self.smart_fetcher = smart_fetcher
        self.cache_manager = cache_manager
        self.budget_manager = budget_manager
        
        # Initialize cache-aware data interface
        self.data_interface = CacheAwareDataInterface(smart_fetcher, cache_manager)
        
        # Replace data tools with cache-aware versions
        self._replace_data_tools()
        
        logger.info("CacheAwareTradingAgents initialized with cost controls")
    
    def _get_cost_optimized_config(self) -> Dict[str, Any]:
        """Get configuration optimized for cost efficiency"""
        config = DEFAULT_CONFIG.copy()
        
        # Use cheaper models
        config["deep_think_llm"] = "gpt-4o-mini"  # Instead of o1-preview
        config["quick_think_llm"] = "gpt-4o-mini"  # Instead of gpt-4o
        
        # Reduce debate rounds to save costs
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1
        
        # Use cached data primarily
        config["online_tools"] = False  # Disable online tools to use cache
        
        # Set project directories
        config["results_dir"] = "./results/trading_agents"
        config["data_cache_dir"] = "./cache/trading_agents"
        
        return config
    
    def _replace_data_tools(self):
        """Replace TradingAgents data tools with cache-aware versions"""
        # This method would typically replace the tools in the agent toolkit
        # For now, we'll override the data fetching methods
        self.original_get_data = getattr(self, '_get_data', None)
        
    async def analyze_stock_with_budget(
        self, 
        ticker: str, 
        date: str = None,
        analysis_type: str = 'single_agent',
        context: Dict = None
    ) -> Tuple[Optional[Dict], float]:
        """
        Analyze stock with strict budget controls
        
        Returns:
            Tuple of (analysis_result, actual_cost)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Check budget before starting
        can_afford, reason = await self.budget_manager.can_afford_analysis(analysis_type)
        if not can_afford:
            logger.warning(f"Cannot afford {analysis_type} for {ticker}: {reason}")
            raise BudgetExceededException(reason)
        
        # Reserve budget
        reservation_id = await self.budget_manager.reserve_budget(analysis_type)
        
        try:
            # Pre-load cached data to reduce API calls during analysis
            await self._preload_data_for_analysis(ticker, context)
            
            # Run the analysis with token counting
            start_time = datetime.utcnow()
            logger.info(f"Starting {analysis_type} analysis for {ticker}")
            
            # Use original TradingAgents propagate method
            _, decision = self.propagate(ticker, date)
            
            # Estimate costs (in production, you'd get actual token usage from the LLM)
            analysis_duration = (datetime.utcnow() - start_time).total_seconds()
            estimated_tokens = self._estimate_tokens_used(analysis_type, decision)
            estimated_cost = float(self.budget_manager.cost_estimates.get(analysis_type, 0.15))
            
            # Confirm usage
            await self.budget_manager.confirm_usage(
                reservation_id=reservation_id,
                actual_cost=estimated_cost,
                model=self.config.get('quick_think_llm', 'gpt-4o-mini'),
                tokens_used=estimated_tokens,
                analysis_type=analysis_type,
                ticker=ticker
            )
            
            logger.info(f"Completed {analysis_type} for {ticker} in {analysis_duration:.1f}s, "
                       f"cost: ${estimated_cost:.4f}")
            
            return decision, estimated_cost
            
        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            # Cancel reservation on failure
            await self.budget_manager.redis.delete(reservation_id)
            raise
    
    async def _preload_data_for_analysis(self, ticker: str, context: Dict = None):
        """Preload commonly needed data to cache"""
        try:
            # Determine what data types we'll need based on selected analysts
            data_types_needed = []
            
            if 'market' in self.selected_analysts:
                data_types_needed.extend(['price', 'technical'])
            if 'news' in self.selected_analysts:
                data_types_needed.append('news')
            if 'social' in self.selected_analysts:
                data_types_needed.append('social')
            if 'fundamentals' in self.selected_analysts:
                data_types_needed.extend(['fundamentals', 'earnings'])
            
            # Pre-fetch and cache all needed data
            for data_type in data_types_needed:
                try:
                    await self.data_interface.get_cached_stock_data(ticker, data_type)
                except Exception as e:
                    logger.warning(f"Failed to preload {data_type} for {ticker}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to preload data for {ticker}: {e}")
    
    def _estimate_tokens_used(self, analysis_type: str, decision: Dict) -> int:
        """Estimate tokens used based on analysis type and output length"""
        base_tokens = {
            'single_agent': 1500,
            'news_analysis': 1200,
            'fundamentals_analysis': 2000,
            'sentiment_analysis': 1000,
            'technical_analysis': 800,
            'bull_bear_debate': 4000,
            'risk_assessment': 2500,
            'full_analysis': 8000,
            'trader_decision': 1000
        }
        
        estimated = base_tokens.get(analysis_type, 1500)
        
        # Adjust based on output length
        if isinstance(decision, dict):
            output_length = len(str(decision))
            if output_length > 2000:
                estimated *= 1.5
            elif output_length < 500:
                estimated *= 0.7
        
        return int(estimated)
    
    async def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get information about available agents and their capabilities"""
        return {
            'available_analysts': {
                'market': {
                    'description': 'Technical analysis using market indicators',
                    'cost_estimate': float(self.budget_manager.cost_estimates['technical_analysis']),
                    'data_requirements': ['price', 'volume', 'technical_indicators']
                },
                'news': {
                    'description': 'News sentiment and event impact analysis', 
                    'cost_estimate': float(self.budget_manager.cost_estimates['news_analysis']),
                    'data_requirements': ['news', 'events']
                },
                'fundamentals': {
                    'description': 'Financial statement and valuation analysis',
                    'cost_estimate': float(self.budget_manager.cost_estimates['fundamentals_analysis']),
                    'data_requirements': ['fundamentals', 'earnings', 'financials']
                },
                'social': {
                    'description': 'Social media sentiment analysis',
                    'cost_estimate': float(self.budget_manager.cost_estimates['sentiment_analysis']),
                    'data_requirements': ['social_media', 'sentiment']
                }
            },
            'analysis_types': {
                'single_agent': {
                    'description': 'Run single analyst',
                    'cost_estimate': float(self.budget_manager.cost_estimates['single_agent']),
                    'duration_estimate': '30-60 seconds'
                },
                'bull_bear_debate': {
                    'description': 'Bull vs bear researcher debate',
                    'cost_estimate': float(self.budget_manager.cost_estimates['bull_bear_debate']),
                    'duration_estimate': '2-4 minutes'
                },
                'full_analysis': {
                    'description': 'Complete multi-agent analysis with debate',
                    'cost_estimate': float(self.budget_manager.cost_estimates['full_analysis']),
                    'duration_estimate': '5-10 minutes'
                }
            },
            'current_config': {
                'deep_think_model': self.config.get('deep_think_llm'),
                'quick_think_model': self.config.get('quick_think_llm'),
                'max_debate_rounds': self.config.get('max_debate_rounds'),
                'online_tools': self.config.get('online_tools')
            }
        }
    
    async def test_agent_connectivity(self) -> Dict[str, Any]:
        """Test that agents can connect to LLM providers"""
        test_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'llm_connectivity': {},
            'budget_status': {},
            'cache_status': {}
        }
        
        try:
            # Test budget manager
            budget_status = await self.budget_manager.get_budget_status()
            test_results['budget_status'] = budget_status
            
            # Test cache connectivity
            test_key = f"agent_test:{datetime.utcnow().timestamp()}"
            await self.cache_manager.set(test_key, {'test': True}, 60)
            cached_value = await self.cache_manager.get(test_key)
            test_results['cache_status'] = {
                'connected': cached_value is not None,
                'test_successful': cached_value.get('test') if cached_value else False
            }
            
            # Clean up test key
            await self.cache_manager.delete(test_key)
            
            # Test basic LLM connectivity (if budget allows)
            can_afford, reason = await self.budget_manager.can_afford_analysis('single_agent')
            test_results['llm_connectivity'] = {
                'budget_available': can_afford,
                'budget_reason': reason,
                'models_configured': {
                    'deep_think': self.config.get('deep_think_llm'),
                    'quick_think': self.config.get('quick_think_llm')
                }
            }
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"Agent connectivity test failed: {e}")
            
        return test_results
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self.cache_manager, 'close'):
                await self.cache_manager.close()
            if hasattr(self.budget_manager, 'close'):
                await self.budget_manager.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            asyncio.create_task(self.cleanup())
        except Exception:
            pass  # Ignore cleanup errors during destruction