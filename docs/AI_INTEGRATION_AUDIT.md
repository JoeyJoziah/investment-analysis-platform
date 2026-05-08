# AI / LLM Integration Audit

Last updated: 2026-05-08
Status: TradingAgents (LangGraph multi-agent framework) installable; runtime requires LLM provider keys.

## TL;DR

```bash
# Provider keys (set at least one)
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
# or
export GOOGLE_API_KEY=...

# Verify wiring
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('backend/TradingAgents'))
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
print('OK', DEFAULT_CONFIG['llm_provider'])
"
```

## Components

| Component | Path | Purpose |
|---|---|---|
| TradingAgents (vendored) | `backend/TradingAgents/` | Multi-agent LangGraph framework — analyst, researcher, trader, risk roles |
| Cache-aware wrapper | `backend/analytics/agents/cache_aware_agents.py` | Wraps TradingAgents with cache + LLM budget controls |
| LLM budget manager | `backend/utils/llm_budget_manager.py` | Per-call cost tracking, daily/monthly limits, circuit breaker |
| FinBERT analyzer | `backend/analytics/finbert_analyzer.py` | Local sentiment via HuggingFace ProsusAI/finbert |
| Default config | `backend/TradingAgents/tradingagents/default_config.py` | Model selection, deep vs quick think, backend URL |

## Provider matrix

The framework supports multiple providers — pick one or more by setting env vars:

| Provider | Env var | Models used | Notes |
|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | claude-opus, claude-sonnet, claude-haiku | Recommended for deep_think |
| OpenAI | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini | Recommended for quick_think |
| Google | `GOOGLE_API_KEY` | gemini-2.0-flash, gemini-1.5-pro | Cost-effective alternative |

Override the default provider in `DEFAULT_CONFIG`:
```python
DEFAULT_CONFIG = {
    "llm_provider": "anthropic",          # "openai" | "anthropic" | "google"
    "deep_think_llm": "claude-opus",
    "quick_think_llm": "claude-haiku",
    "backend_url": "https://api.anthropic.com",
    ...
}
```

## Cost controls

The cache-aware wrapper enforces budgets via `LLMBudgetManager`:

| Setting | Env var | Default |
|---|---|---|
| Daily budget USD | `LLM_DAILY_BUDGET_USD` | 5.00 |
| Monthly budget USD | `LLM_MONTHLY_BUDGET_USD` | 100.00 |
| Per-call max tokens | `LLM_MAX_TOKENS` | 4096 |
| Cache TTL seconds | `LLM_CACHE_TTL_S` | 3600 |
| Circuit breaker after N errors | `LLM_CB_THRESHOLD` | 5 |

Status: live and integrated with Prometheus metrics (`llm_calls_total`, `llm_cost_usd_total`, `llm_tokens_total`, `llm_cache_hit_rate`).

## Operational checklist

- [ ] Set at least one provider key in environment / secrets manager
- [ ] Set `LLM_DAILY_BUDGET_USD` and `LLM_MONTHLY_BUDGET_USD` to actual budget
- [ ] Verify cost monitoring dashboard in Grafana shows live data
- [ ] Run a test analysis: `POST /api/v1/agents/analyze {"symbol": "AAPL"}` and confirm response
- [ ] Confirm circuit breaker fires when daily budget exceeded
- [ ] Confirm Slack alert fires when monthly budget reaches 80%

## Smoke test

```bash
# With keys set, run the agents test suite (will skip live calls if keys absent)
pytest backend/tests/unit/test_trading_agents.py -v

# Trigger an end-to-end analysis (requires backend running)
curl -X POST http://localhost:8000/api/v1/agents/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "depth": "quick"}'
```

## Caveats

- **Vendored framework**: `backend/TradingAgents/` is a fork. Patches to upstream require manual reconciliation.
- **stockstats / akshare**: TradingAgents pulls Chinese market data unless you set `data_dir` to a US-only feed. Disable with `TRADING_AGENTS_DISABLE_AKSHARE=1` in env.
- **chromadb**: TradingAgents uses chromadb for vector memory. First run will create `data/chroma/` — ensure it's in `.gitignore` (it already is).
- **httpx version conflict**: `mcp-server-fetch` pins httpx<0.28; langchain-openai requires >=0.28. Last install resolved by upgrading httpx; mcp-server-fetch may need pinning if you use it.

## See also
- `INVESTMENT_THESIS_FEATURE.md` — agent-driven investment thesis flow
- `architecture/MULTI_SOURCE_ETL_SOLUTION.md` — data sources fed to agents
- `RUNBOOK.md` — LLM circuit-breaker recovery procedure