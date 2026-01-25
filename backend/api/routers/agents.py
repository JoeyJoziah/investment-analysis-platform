"""
API routes for LLM agents and hybrid analysis functionality
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from backend.analytics.agents import HybridAnalysisEngine, AnalysisMode
from backend.utils.auth import get_current_user, require_admin
from backend.utils.rate_limiter import rate_limit
from backend.utils.llm_budget_manager import BudgetExceededException

logger = logging.getLogger(__name__)
router = APIRouter()

# Global hybrid engine instance (would be properly injected in production)
hybrid_engine: Optional[HybridAnalysisEngine] = None


async def get_hybrid_engine():
    """Dependency to get hybrid analysis engine"""
    global hybrid_engine
    if hybrid_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Hybrid analysis engine not initialized"
        )
    return hybrid_engine


# Request/Response Models
class AgentAnalysisRequest(BaseModel):
    """Request for agent-enhanced stock analysis"""
    ticker: str = Field(..., description="Stock ticker symbol", min_length=1, max_length=10)
    force_agents: bool = Field(False, description="Force use of LLM agents regardless of selection criteria")
    analysis_timeout: float = Field(120.0, description="Maximum analysis time in seconds", gt=0, le=300)


class AgentAnalysisResponse(BaseModel):
    """Response from agent-enhanced stock analysis"""
    ticker: str
    recommendation: str
    overall_score: float
    confidence: float
    hybrid_score: Optional[float] = None
    
    # Agent-specific fields
    agent_analysis: Optional[Dict[str, Any]] = None
    agent_confidence: Optional[float] = None
    agent_reasoning: Optional[str] = None
    agents_used: List[str] = []
    
    # Cost and performance metadata
    analysis_cost: float
    analysis_duration: float
    complexity_level: str
    
    # Traditional analysis fields
    target_price: float
    risks: List[str]
    opportunities: List[str]


class BatchAnalysisRequest(BaseModel):
    """Request for batch stock analysis"""
    tickers: List[str] = Field(..., description="List of stock tickers", min_items=1, max_items=50)
    max_concurrent: int = Field(5, description="Maximum concurrent analyses", gt=0, le=10)
    prioritize_by_tier: bool = Field(True, description="Prioritize analysis by stock tier")


class BudgetStatusResponse(BaseModel):
    """Response with LLM budget status"""
    monthly_budget: float
    monthly_used: float
    monthly_remaining: float
    daily_used: float
    hourly_used: float
    cost_health: str
    recommended_actions: List[str]
    usage_stats: Dict[str, Any]


class AgentCapabilitiesResponse(BaseModel):
    """Response with agent capabilities information"""
    available_analysts: Dict[str, Dict[str, Any]]
    analysis_types: Dict[str, Dict[str, Any]]
    current_config: Dict[str, Any]


# Routes

@router.post("/analyze", response_model=AgentAnalysisResponse)
@rate_limit(requests_per_minute=10)  # 10 calls per minute
async def analyze_stock_with_agents(
    request: AgentAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Analyze a stock using hybrid traditional + LLM agent approach
    """
    try:
        logger.info(f"Agent analysis requested for {request.ticker} by {current_user.get('username', 'anonymous')}")
        
        # Run hybrid analysis
        result = await engine.analyze_stock(
            ticker=request.ticker,
            force_agents=request.force_agents,
            analysis_timeout=request.analysis_timeout
        )
        
        # Convert to response model
        response = AgentAnalysisResponse(
            ticker=result.ticker,
            recommendation=result.recommendation,
            overall_score=result.overall_score,
            confidence=result.confidence,
            hybrid_score=result.hybrid_score,
            agent_analysis=result.agent_analysis,
            agent_confidence=result.agent_confidence,
            agent_reasoning=result.agent_reasoning,
            agents_used=result.agents_used or [],
            analysis_cost=result.analysis_cost,
            analysis_duration=result.analysis_duration,
            complexity_level=result.complexity_level,
            target_price=result.target_price,
            risks=result.risks or [],
            opportunities=result.opportunities or []
        )
        
        # Log analysis for monitoring
        background_tasks.add_task(
            log_analysis_metrics,
            request.ticker,
            result.complexity_level,
            result.analysis_cost,
            result.analysis_duration,
            len(result.agents_used or [])
        )
        
        return response
        
    except BudgetExceededException as e:
        logger.warning(f"Budget exceeded for {request.ticker}: {e}")
        raise HTTPException(
            status_code=429,
            detail=f"LLM budget exceeded: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Agent analysis failed for {request.ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/batch-analyze")
@rate_limit(requests_per_minute=2)  # 2 calls per 5 minutes (limited)
async def batch_analyze_stocks(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Analyze multiple stocks in batch with resource management
    """
    try:
        logger.info(f"Batch analysis requested for {len(request.tickers)} stocks by {current_user.get('username', 'anonymous')}")
        
        # Run batch analysis
        results = await engine.batch_analyze_stocks(
            tickers=request.tickers,
            max_concurrent=request.max_concurrent,
            prioritize_by_tier=request.prioritize_by_tier
        )
        
        # Convert results to response format
        response_data = {}
        total_cost = 0.0
        total_duration = 0.0
        agents_used_count = 0
        
        for ticker, result in results.items():
            response_data[ticker] = AgentAnalysisResponse(
                ticker=result.ticker,
                recommendation=result.recommendation,
                overall_score=result.overall_score,
                confidence=result.confidence,
                hybrid_score=result.hybrid_score,
                agent_analysis=result.agent_analysis,
                agent_confidence=result.agent_confidence,
                agent_reasoning=result.agent_reasoning,
                agents_used=result.agents_used or [],
                analysis_cost=result.analysis_cost,
                analysis_duration=result.analysis_duration,
                complexity_level=result.complexity_level,
                target_price=result.target_price,
                risks=result.risks or [],
                opportunities=result.opportunities or []
            )
            
            total_cost += result.analysis_cost
            total_duration += result.analysis_duration
            if result.agents_used:
                agents_used_count += 1
        
        # Log batch metrics
        background_tasks.add_task(
            log_batch_analysis_metrics,
            len(request.tickers),
            len(results),
            total_cost,
            total_duration,
            agents_used_count
        )
        
        return {
            "results": response_data,
            "summary": {
                "requested": len(request.tickers),
                "completed": len(results),
                "total_cost": total_cost,
                "avg_duration": total_duration / len(results) if results else 0,
                "agents_used_count": agents_used_count
            }
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get("/budget-status", response_model=BudgetStatusResponse)
async def get_budget_status(
    current_user = Depends(get_current_user),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Get current LLM budget status and usage statistics
    """
    try:
        status = await engine.budget_manager.get_budget_status()
        
        return BudgetStatusResponse(
            monthly_budget=status['budget']['monthly_budget'],
            monthly_used=status['budget']['monthly_used'],
            monthly_remaining=status['budget']['monthly_remaining'],
            daily_used=status['budget']['daily_used'],
            hourly_used=status['budget']['hourly_used'],
            cost_health=status['cost_health'],
            recommended_actions=status['recommended_actions'],
            usage_stats=status['usage_stats']
        )
        
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get budget status: {str(e)}"
        )


@router.get("/capabilities", response_model=AgentCapabilitiesResponse)
async def get_agent_capabilities(
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Get information about available agents and their capabilities
    """
    try:
        capabilities = await engine.trading_agents.get_agent_capabilities()
        
        return AgentCapabilitiesResponse(
            available_analysts=capabilities['available_analysts'],
            analysis_types=capabilities['analysis_types'],
            current_config=capabilities['current_config']
        )
        
    except Exception as e:
        logger.error(f"Failed to get agent capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent capabilities: {str(e)}"
        )


@router.get("/status")
async def get_engine_status(
    current_user = Depends(get_current_user),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Get comprehensive engine status and statistics
    """
    try:
        status = await engine.get_engine_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get engine status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get engine status: {str(e)}"
        )


@router.post("/test-connectivity")
async def test_agent_connectivity(
    current_user = Depends(require_admin),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Test agent connectivity and system health (admin only)
    """
    try:
        test_results = await engine.trading_agents.test_agent_connectivity()
        return {
            "status": "success",
            "test_results": test_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent connectivity test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Connectivity test failed: {str(e)}"
        )


@router.post("/set-analysis-mode")
async def set_analysis_mode(
    mode: str,
    current_user = Depends(require_admin),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Set analysis mode (admin only)
    """
    try:
        # Validate mode
        valid_modes = [mode.value for mode in AnalysisMode]
        if mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Valid modes: {valid_modes}"
            )
        
        engine.set_analysis_mode(AnalysisMode(mode))
        
        return {
            "status": "success",
            "new_mode": mode,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to set analysis mode: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set analysis mode: {str(e)}"
        )


@router.get("/selection-stats")
async def get_agent_selection_stats(
    current_user = Depends(get_current_user),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
):
    """
    Get agent selection statistics and criteria
    """
    try:
        stats = await engine.agent_orchestrator.get_selection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get selection stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get selection stats: {str(e)}"
        )


# Background task functions

async def log_analysis_metrics(
    ticker: str,
    complexity_level: str,
    cost: float,
    duration: float,
    agent_count: int
):
    """Log analysis metrics for monitoring"""
    try:
        # This would typically write to a metrics database
        logger.info(f"Analysis metrics - Ticker: {ticker}, "
                   f"Complexity: {complexity_level}, "
                   f"Cost: ${cost:.4f}, "
                   f"Duration: {duration:.1f}s, "
                   f"Agents: {agent_count}")
    except Exception as e:
        logger.error(f"Failed to log analysis metrics: {e}")


async def log_batch_analysis_metrics(
    requested: int,
    completed: int,
    total_cost: float,
    total_duration: float,
    agents_used_count: int
):
    """Log batch analysis metrics for monitoring"""
    try:
        logger.info(f"Batch analysis metrics - Requested: {requested}, "
                   f"Completed: {completed}, "
                   f"Total cost: ${total_cost:.4f}, "
                   f"Total duration: {total_duration:.1f}s, "
                   f"Agent analyses: {agents_used_count}")
    except Exception as e:
        logger.error(f"Failed to log batch analysis metrics: {e}")


# Initialize hybrid engine (this would be done in app startup)
async def initialize_hybrid_engine():
    """Initialize the hybrid analysis engine"""
    global hybrid_engine
    try:
        from backend.analytics.recommendation_engine import RecommendationEngine
        from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher
        from backend.utils.cache_manager import CacheManager
        from backend.utils.llm_budget_manager import LLMBudgetManager
        
        # Initialize components (these would be properly injected in production)
        traditional_engine = RecommendationEngine()
        smart_fetcher = SmartDataFetcher()
        cache_manager = CacheManager()
        budget_manager = LLMBudgetManager()
        
        # Create hybrid engine
        hybrid_engine = HybridAnalysisEngine(
            traditional_engine=traditional_engine,
            smart_fetcher=smart_fetcher,
            cache_manager=cache_manager,
            budget_manager=budget_manager,
            analysis_mode=AnalysisMode.SELECTIVE_HYBRID
        )
        
        logger.info("Hybrid analysis engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize hybrid engine: {e}")
        hybrid_engine = None