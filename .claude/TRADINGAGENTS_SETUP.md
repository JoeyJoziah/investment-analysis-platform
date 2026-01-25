# TradingAgents Integration Setup Guide

## Overview

TradingAgents is a multi-agent LLM trading framework integrated into the investment analysis platform. It provides advanced trading analysis through specialized agents (fundamental, sentiment, technical analysts, traders, and risk management).

## Location

**Primary Implementation**: `backend/TradingAgents/`

**Integration Layer**: `backend/analytics/agents/`
- `cache_aware_agents.py` - Manages TradingAgents with cost controls
- `hybrid_engine.py` - Combines traditional ML with TradingAgents
- `selective_orchestrator.py` - Intelligently routes analysis tasks
- `enhancement_levels.py` - Progressive enhancement strategies

## Architecture

```
backend/
├── TradingAgents/                    # Main TradingAgents framework
│   ├── tradingagents/
│   │   ├── agents/                   # Analyst, trader, risk mgmt agents
│   │   ├── graph/                    # LangGraph orchestration
│   │   ├── dataflows/                # Data interfaces
│   │   └── default_config.py         # Configuration
│   └── requirements.txt              # TradingAgents dependencies
│
└── analytics/
    └── agents/                       # Integration layer
        ├── cache_aware_agents.py     # Cost-aware wrapper
        ├── hybrid_engine.py          # ML + LLM hybrid
        ├── selective_orchestrator.py # Smart routing
        └── enhancement_levels.py     # Progressive features
```

## Required Environment Variables

Add these to your `.env` file:

```bash
# LLM Provider (for TradingAgents)
OPENAI_API_KEY=your_openai_api_key_here
# OR use Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# OR use Google
GOOGLE_API_KEY=your_google_api_key_here

# Financial Data (already configured)
FINNHUB_API_KEY=your_finnhub_api_key_here
```

## Dependencies

All TradingAgents dependencies are included in the main `requirements.txt`:
- langchain-openai
- langchain-anthropic
- langchain-google-genai
- langgraph
- finnhub-python
- And more...

Run `pip install -r requirements.txt` to install all dependencies.

## Usage

### 1. Direct TradingAgents Usage

```python
import sys
import os

# Add TradingAgents to path
sys.path.insert(0, 'backend/TradingAgents')

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Configure
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"  # or "anthropic" or "google"
config["deep_think_llm"] = "gpt-4o-mini"
config["quick_think_llm"] = "gpt-4o-mini"
config["online_tools"] = True

# Initialize
ta = TradingAgentsGraph(debug=True, config=config)

# Analyze a stock
_, decision = ta.propagate("AAPL", "2024-05-10")
print(decision)
```

### 2. Cost-Aware Integration (Recommended)

```python
from backend.analytics.agents import CacheAwareTradingAgents
from backend.utils.llm_budget_manager import LLMBudgetManager
from backend.utils.cache_manager import CacheManager
from backend.data_ingestion.smart_data_fetcher import SmartDataFetcher

# Initialize dependencies
budget_mgr = LLMBudgetManager(daily_limit=5.0, monthly_limit=150.0)
cache_mgr = CacheManager()
data_fetcher = SmartDataFetcher()

# Create cost-aware wrapper
ta = CacheAwareTradingAgents(
    budget_manager=budget_mgr,
    cache_manager=cache_mgr,
    smart_fetcher=data_fetcher
)

# Analyze with cost controls
result = await ta.analyze_stock("AAPL", enable_agents=True)
```

### 3. Hybrid Engine (Best for Production)

```python
from backend.analytics.agents import HybridAnalysisEngine, AnalysisMode

# Initialize hybrid engine
engine = HybridAnalysisEngine(
    recommendation_engine=traditional_engine,
    budget_manager=budget_mgr,
    cache_manager=cache_mgr,
    smart_fetcher=data_fetcher
)

# Smart routing - automatically chooses best approach
recommendations = await engine.generate_recommendations(
    tickers=["AAPL", "MSFT", "GOOGL"],
    mode=AnalysisMode.SMART  # or TRADITIONAL, AGENT_ENHANCED, FULL_AGENT
)
```

## Configuration

### TradingAgents Config Options

Edit `backend/TradingAgents/tradingagents/default_config.py`:

```python
DEFAULT_CONFIG = {
    "llm_provider": "openai",        # openai, anthropic, google
    "deep_think_llm": "gpt-4o-mini", # For complex analysis
    "quick_think_llm": "gpt-4o-mini",# For simple tasks
    "backend_url": "https://api.openai.com/v1",
    "max_debate_rounds": 1,          # Agent debate rounds
    "online_tools": True,            # Use real-time data
}
```

### Cost Control Settings

In `backend/analytics/agents/cache_aware_agents.py`:

```python
# Default budget limits
DAILY_LLM_BUDGET = 5.0    # $5/day
MONTHLY_LLM_BUDGET = 150.0  # $150/month
CACHE_TTL = 3600  # 1 hour cache
```

## Features

### 1. Multi-Agent Analysis
- **Fundamental Analyst**: Company financials, metrics, red flags
- **Sentiment Analyst**: Social media, public sentiment
- **News Analyst**: Global news, macroeconomic indicators
- **Technical Analyst**: MACD, RSI, trading patterns
- **Researchers**: Bull/bear debate
- **Trader**: Final trading decision
- **Risk Management**: Portfolio risk assessment

### 2. Cost Optimization
- Smart caching (avoids duplicate API calls)
- Budget tracking and circuit breakers
- Progressive enhancement (simple → advanced)
- Selective agent activation

### 3. Hybrid Intelligence
- Combines traditional ML with LLM insights
- Falls back to ML if budget exceeded
- Confidence-based routing

## Testing

```bash
# Test TradingAgents import
python3 -c "import sys; sys.path.insert(0, 'backend/TradingAgents'); from tradingagents.graph.trading_graph import TradingAgentsGraph; print('✓ OK')"

# Run integration tests
pytest backend/tests/test_tradingagents_integration.py -v

# Test hybrid engine
pytest backend/tests/test_hybrid_engine.py -v
```

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'langchain_openai'`:
```bash
pip install -r requirements.txt
```

### API Key Issues
If you see authentication errors:
1. Check `.env` file has `OPENAI_API_KEY` (or equivalent)
2. Verify the key is valid and has credits
3. Check `FINNHUB_API_KEY` is configured

### Path Issues
If TradingAgents can't be imported:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../TradingAgents'))
```

### Budget Exceeded
If analysis is skipped due to budget:
1. Check budget limits in LLMBudgetManager
2. Use `AnalysisMode.TRADITIONAL` for cost-free analysis
3. Increase daily/monthly limits if needed

## Cost Estimates

### TradingAgents Analysis (per stock)
- **Quick Analysis**: ~$0.01 - 0.05 (gpt-4o-mini)
- **Deep Analysis**: ~$0.10 - 0.50 (gpt-4o)
- **Full Debate**: ~$0.50 - 2.00 (with o1 models)

### Daily Budget Recommendations
- **Development**: $1-5/day (testing, limited stocks)
- **Production Light**: $5-10/day (top picks only)
- **Production Full**: $20-50/day (comprehensive analysis)

## Integration Points

### With Recommendation Engine
`backend/analytics/recommendation_engine.py` can use TradingAgents through the hybrid engine for enhanced recommendations.

### With API Endpoints
`backend/api/routers/` can expose TradingAgents analysis via REST endpoints.

### With Data Pipeline
`backend/data_ingestion/smart_data_fetcher.py` provides cached data to TradingAgents.

## Best Practices

1. **Use Hybrid Engine**: Automatically balances cost vs. insight
2. **Cache Aggressively**: 1-hour TTL for stock analysis
3. **Monitor Budgets**: Set alerts at 80% budget consumption
4. **Progressive Enhancement**: Start with traditional ML, add agents for high-confidence trades
5. **Selective Activation**: Only use TradingAgents for complex/uncertain stocks

## References

- TradingAgents Paper: https://arxiv.org/abs/2412.20138
- TradingAgents GitHub: https://github.com/TauricResearch/TradingAgents
- Integration Source: `backend/analytics/agents/`

## Notes

- TradingAgents is for **research purposes** - not financial advice
- Performance varies with model choice, temperature, data quality
- Designed to complement (not replace) traditional ML analysis
- Cost optimization is critical for daily operation under $50/month budget
