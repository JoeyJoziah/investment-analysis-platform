# Data Flow Architecture Codemap

## Data Sources

### Financial APIs

| API | Rate Limit | Data Type | File |
|-----|------------|-----------|------|
| Alpha Vantage | 25/day | Historical prices, fundamentals | `multi_source_extractor.py` |
| Finnhub | 60/min | Real-time quotes, news | `multi_source_extractor.py` |
| Polygon.io | 5/min | Aggregates, tickers | `multi_source_extractor.py` |
| NewsAPI | 100/day | News articles | `news_collector.py` |
| FMP | 250/day | Financial statements | `fundamentals_extractor.py` |
| MarketAux | 100/day | Market news | `news_collector.py` |
| FRED | 120/min | Economic indicators | `economic_data.py` |

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     EXTERNAL DATA SOURCES                     │
├──────────────┬──────────────┬────────────────┬───────────────┤
│ Alpha Vantage│   Finnhub    │   Polygon.io   │   NewsAPI     │
└──────┬───────┴──────┬───────┴───────┬────────┴───────┬───────┘
       │              │               │                │
       ▼              ▼               ▼                ▼
┌──────────────────────────────────────────────────────────────┐
│              MULTI-SOURCE EXTRACTOR (ETL Layer)              │
│           backend/etl/multi_source_extractor.py              │
│  • Rate limiting per API                                     │
│  • Request batching                                          │
│  • Error handling & retries                                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    3-LAYER CACHE SYSTEM                      │
├─────────────────┬─────────────────┬──────────────────────────┤
│   L1: Memory    │   L2: Redis     │   L3: PostgreSQL         │
│   (In-process)  │   (512MB LRU)   │   (TimescaleDB)          │
│   TTL: 60s      │   TTL: 5-30min  │   Persistent             │
└─────────────────┴────────┬────────┴──────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    AIRFLOW DAG SCHEDULER                     │
│         data_pipelines/airflow/dags/daily_stock_pipeline.py  │
│  • Daily data ingestion (6 AM UTC)                           │
│  • Technical indicator calculation                           │
│  • ML model inference                                        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      ML PIPELINE                             │
│                    backend/ml/                               │
│  • Feature engineering                                       │
│  • LSTM, XGBoost, Prophet predictions                        │
│  • Sentiment analysis (FinBERT)                              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION ENGINE                     │
│           backend/services/recommendation_service.py         │
│  • Score aggregation                                         │
│  • Risk assessment                                           │
│  • SEC compliance validation                                 │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                       API LAYER                              │
│                 backend/api/routers/                         │
│  • REST endpoints                                            │
│  • WebSocket real-time                                       │
│  • Cache-decorated responses                                 │
└──────────────────────────────────────────────────────────────┘
```

## Caching Architecture

### L1: Memory Cache (In-Process)
- **Implementation**: Python dict with TTL
- **TTL**: 60 seconds
- **Scope**: Per-process, frequently accessed data
- **Location**: `backend/etl/intelligent_cache_system.py`

### L2: Redis Cache (512MB LRU)
- **Implementation**: Redis 7.0 with `allkeys-lru`
- **TTL**: 5-30 minutes depending on data type
- **Scope**: Cross-process, API response caching
- **Location**: `backend/utils/cache.py:205-300`

**Quick Win Implementation**:
```python
# backend/utils/cache.py
@cache_with_ttl(ttl=300)
async def get_analysis(symbol: str) -> dict:
    # Cached for 5 minutes
    pass
```

### L3: PostgreSQL (TimescaleDB)
- **Implementation**: TimescaleDB hypertables
- **Scope**: Persistent historical data
- **Location**: `backend/repositories/`

## Pipeline Timing

### Daily Schedule (UTC)

| Time | Pipeline | Duration |
|------|----------|----------|
| 06:00 | Stock data ingestion | ~45 min |
| 07:00 | Technical indicators | ~30 min |
| 07:30 | Fundamental updates | ~15 min |
| 08:00 | ML predictions | ~30 min |
| 08:30 | Recommendations | ~15 min |
| 09:00 | Cache warming | ~10 min |

### Airflow DAGs

| DAG | File | Schedule |
|-----|------|----------|
| daily_stock_pipeline | `daily_stock_pipeline.py` | 0 6 * * * |
| ml_training_pipeline | `ml_training_pipeline_dag.py` | 0 2 * * 0 |
| data_quality_check | `data_quality_dag.py` | 0 5 * * * |

## Database Schema (Key Tables)

| Table | Type | Purpose |
|-------|------|---------|
| stocks | Regular | Stock metadata |
| price_history | TimescaleDB | OHLCV data (hypertable) |
| technical_indicators | TimescaleDB | Calculated indicators |
| recommendations | Regular | AI recommendations |
| ml_predictions | Regular | Model predictions |
| news_sentiment | Regular | Sentiment scores |

### Indexes (Quick Win #4)

45 indexes added in `008_add_missing_query_indexes.py`:
- Covering indexes for common queries
- Partial indexes for active records
- B-tree indexes for sorting
- GIN indexes for full-text search (pg_trgm)

## API Response Flow

### Analysis Endpoint (Optimized)

```
GET /api/analysis/{symbol}
       │
       ▼
┌─────────────────────────────┐
│   Check Redis Cache (L2)    │
│   cache_with_ttl decorator  │
└──────────────┬──────────────┘
               │ MISS
               ▼
┌─────────────────────────────┐
│   asyncio.gather() PARALLEL │ ◄── Quick Win #2
│   • fetch_technical_indicators()
│   • price_repository.get_price_history()
│   • fetch_fundamental_data()
│   • fetch_sentiment_data()
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Aggregate & Transform     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Store in Redis (L2)       │
│   TTL: 300 seconds          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Return JSON Response      │
└─────────────────────────────┘
```

**Performance**: 4-6s → 1.5-2s (60% improvement)

## Error Handling

| Component | Strategy |
|-----------|----------|
| API calls | Retry with exponential backoff |
| Cache misses | Fallback to database |
| ML predictions | Return last known good value |
| WebSocket | Auto-reconnect with exponential backoff |

**Last Updated**: 2026-01-26
