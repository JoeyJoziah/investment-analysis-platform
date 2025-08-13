# TradingAgents Integration Guide

This document provides a comprehensive guide for the TradingAgents integration with the investment analysis application.

## Overview

The TradingAgents framework has been successfully integrated into the investment analysis platform, providing LLM-powered agents that complement traditional ML-based analysis engines. The integration follows a hybrid approach that maintains strict cost controls while leveraging advanced reasoning capabilities.

## Architecture Components

### 1. Budget Management System
- **File**: `backend/utils/llm_budget_manager.py`
- **Purpose**: Strict cost controls for LLM usage
- **Budget**: $25/month (50% of total budget)
- **Features**:
  - Real-time cost tracking
  - Budget reservations
  - Circuit breaker protection
  - Daily/hourly limits

### 2. Cache-Aware TradingAgents
- **File**: `backend/analytics/agents/cache_aware_agents.py`
- **Purpose**: Integrates TradingAgents with existing caching system
- **Features**:
  - Respects API rate limits
  - Uses cached data when available
  - Cost-optimized model selection
  - Token usage estimation

### 3. Selective Agent Orchestrator
- **File**: `backend/analytics/agents/selective_orchestrator.py`
- **Purpose**: Smart agent selection based on analysis context
- **Features**:
  - Contextual agent activation
  - Complexity level determination
  - Analysis gap identification
  - Priority-based agent selection

### 4. Hybrid Analysis Engine
- **File**: `backend/analytics/agents/hybrid_engine.py`
- **Purpose**: Orchestrates traditional ML + LLM agent analysis
- **Features**:
  - Multiple analysis modes
  - Intelligent fallback
  - Performance tracking
  - Batch processing

### 5. Progressive Enhancement
- **File**: `backend/analytics/agents/enhancement_levels.py`
- **Purpose**: Determines appropriate enhancement level
- **Levels**:
  - **Basic**: Single agent ($0.15)
  - **Standard**: Multi-agent ($0.45)
  - **Premium**: Full debate ($1.20)
  - **Comprehensive**: Maximum depth ($2.50)

### 6. API Integration
- **File**: `backend/api/routers/agents.py`
- **Purpose**: REST API endpoints for agent functionality
- **Endpoints**:
  - `POST /api/agents/analyze` - Single stock analysis
  - `POST /api/agents/batch-analyze` - Batch analysis
  - `GET /api/agents/budget-status` - Budget monitoring
  - `GET /api/agents/capabilities` - Agent information

## Configuration

### Environment Variables

```bash
# LLM API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Budget Management
LLM_MONTHLY_BUDGET=25.0
LLM_DAILY_LIMIT=0.83
LLM_HOURLY_LIMIT=0.35

# TradingAgents Settings
TRADINGAGENTS_DEEP_THINK_MODEL=gpt-4o-mini
TRADINGAGENTS_QUICK_THINK_MODEL=gpt-4o-mini
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1
TRADINGAGENTS_ONLINE_TOOLS=false

# Analysis Configuration
ENABLE_AGENT_ANALYSIS=true
AGENT_ANALYSIS_MODE=selective_hybrid
DEFAULT_ANALYSIS_TIMEOUT=120
```

### Dependencies Added

```txt
# LLM and TradingAgents Dependencies
langchain-anthropic==0.3.15
langchain-openai==0.3.23
langgraph==0.4.8
typing-extensions==4.14.0
```

## Usage Examples

### 1. Basic Stock Analysis with Agents

```python
from backend.analytics.agents import HybridAnalysisEngine

# Initialize hybrid engine
hybrid_engine = HybridAnalysisEngine(
    traditional_engine=recommendation_engine,
    smart_fetcher=smart_data_fetcher,
    cache_manager=cache_manager,
    budget_manager=budget_manager
)

# Analyze stock with intelligent agent selection
result = await hybrid_engine.analyze_stock("AAPL")
```

### 2. Forced Agent Analysis

```python
# Force use of agents regardless of selection criteria
result = await hybrid_engine.analyze_stock(
    ticker="NVDA",
    force_agents=True,
    analysis_timeout=180.0
)
```

### 3. Batch Analysis

```python
# Analyze multiple stocks efficiently
results = await hybrid_engine.batch_analyze_stocks(
    tickers=["AAPL", "GOOGL", "MSFT", "TSLA"],
    max_concurrent=3,
    prioritize_by_tier=True
)
```

### 4. Budget Monitoring

```python
# Check budget status
budget_status = await budget_manager.get_budget_status()
print(f"Monthly used: ${budget_status['budget']['monthly_used']:.2f}")
print(f"Remaining: ${budget_status['budget']['monthly_remaining']:.2f}")
```

## API Usage

### Single Stock Analysis

```bash
curl -X POST "http://localhost:8000/api/agents/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "ticker": "AAPL",
    "force_agents": false,
    "analysis_timeout": 120
  }'
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8000/api/agents/batch-analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "max_concurrent": 3,
    "prioritize_by_tier": true
  }'
```

### Budget Status

```bash
curl -X GET "http://localhost:8000/api/agents/budget-status" \
  -H "Authorization: Bearer your_token"
```

## Cost Control Features

### 1. Automatic Budget Enforcement
- Monthly budget: $25 (50% of total)
- Daily limits: $0.83
- Hourly burst protection: $0.35
- Automatic fallback to traditional analysis

### 2. Smart Model Selection
- Default: `gpt-4o-mini` (cost-effective)
- Task-based optimization
- Token usage estimation
- Cost per analysis tracking

### 3. Circuit Breaker Protection
- Prevents cost overruns
- Automatic recovery
- Failure threshold: 5 budget failures
- Recovery timeout: 5 minutes

### 4. Progressive Enhancement
- Basic: $0.15 (single agent)
- Standard: $0.45 (multi-agent)
- Premium: $1.20 (with debate)
- Comprehensive: $2.50 (full analysis)

## Monitoring and Alerting

### 1. Cost Monitoring
```python
# Real-time cost tracking
cost_health = budget_status['cost_health']  # healthy, moderate, high, critical

# Recommendations
actions = budget_status['recommended_actions']
```

### 2. Performance Metrics
```python
# Engine statistics
stats = await hybrid_engine.get_engine_status()
print(f"Agent analyses: {stats['performance_stats']['agent_analyses']}")
print(f"Average cost: ${stats['performance_stats']['avg_cost_per_analysis']:.4f}")
```

### 3. Agent Selection Statistics
```python
# Selection patterns
selection_stats = await agent_orchestrator.get_selection_stats()
```

## Troubleshooting

### Common Issues

1. **Budget Exceeded**
   ```python
   # Check budget status
   status = await budget_manager.get_budget_status()
   
   # Wait for budget reset or increase limits
   if status['cost_health'] == 'critical':
       # Use traditional analysis only
       engine.set_analysis_mode(AnalysisMode.TRADITIONAL_ONLY)
   ```

2. **Agent Connectivity Issues**
   ```python
   # Test connectivity
   test_results = await trading_agents.test_agent_connectivity()
   
   if not test_results['llm_connectivity']['budget_available']:
       # Budget issue
       print("Budget constraint:", test_results['llm_connectivity']['budget_reason'])
   ```

3. **Performance Issues**
   ```python
   # Check circuit breaker status
   if circuit_breaker.state == 'OPEN':
       # Too many failures, wait for recovery
       print("Circuit breaker open, using traditional analysis")
   ```

### Error Handling

The system includes comprehensive error handling:
- Budget exceeded → Fallback to traditional analysis
- LLM timeout → Use cached results
- API rate limits → Automatic provider switching
- Agent failures → Graceful degradation

## Security Considerations

1. **API Key Management**
   - Store keys in environment variables
   - Never commit keys to version control
   - Use different keys for different environments

2. **Rate Limiting**
   - Agent analysis: 10 calls/minute
   - Batch analysis: 2 calls/5 minutes
   - Budget monitoring: No limits

3. **Authentication**
   - All agent endpoints require authentication
   - Admin-only endpoints for configuration
   - Role-based access control

## Performance Optimization

### 1. Caching Strategy
- Pre-load data for analysis
- Three-tier caching system
- Intelligent cache TTLs
- Stale data fallback

### 2. Async Processing
- Parallel agent execution
- Non-blocking operations
- Semaphore-based concurrency control
- Timeout management

### 3. Resource Management
- Connection pooling
- Memory optimization
- Cleanup on errors
- Graceful shutdown

## Deployment Notes

### 1. Docker Integration
The TradingAgents are integrated into the existing Docker setup:
- No additional containers needed
- Uses existing Redis and PostgreSQL
- Shared volume mounts for cache

### 2. Environment Setup
```bash
# Create necessary directories
mkdir -p results/trading_agents
mkdir -p cache/trading_agents

# Set permissions
chmod 755 results/trading_agents
chmod 755 cache/trading_agents
```

### 3. Health Checks
```bash
# Check agent status
curl http://localhost:8000/api/agents/status

# Test connectivity
curl -X POST http://localhost:8000/api/agents/test-connectivity
```

## Future Enhancements

### 1. Additional Models
- Support for other LLM providers
- Model performance comparison
- Auto-selection of best model

### 2. Advanced Features
- Agent learning from feedback
- Custom agent training
- Multi-language support

### 3. Optimization
- Model fine-tuning for financial data
- Reduced token usage
- Better caching strategies

## Support and Maintenance

### 1. Monitoring
- Set up alerts for budget thresholds
- Monitor agent performance metrics
- Track cost trends

### 2. Regular Maintenance
- Review budget allocation monthly
- Update model configurations
- Optimize selection criteria

### 3. Scaling
- Adjust concurrent limits based on usage
- Scale Redis for increased caching
- Consider model hosting options

This integration provides a powerful hybrid approach that combines the reliability of traditional analysis with the advanced reasoning capabilities of LLM agents, all while maintaining strict cost controls and operational efficiency.