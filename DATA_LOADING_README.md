# Investment Analysis Platform - Data Loading System

## Overview

This data loading system populates the PostgreSQL database with historical stock market data for analysis. The system is designed to be cost-effective, reliable, and scalable while staying within the $50/month budget constraint.

## Key Features

- **Free Data Sources**: Uses Yahoo Finance (yfinance library) for bulk historical data
- **Smart Rate Limiting**: Prevents API blocks with built-in delays and retry logic
- **Progress Monitoring**: Real-time progress tracking with resume capability
- **Data Validation**: Comprehensive quality checks and error detection
- **Background Execution**: Non-blocking operation with monitoring
- **Cost Monitoring**: Tracks API usage to stay under budget
- **Error Recovery**: Graceful handling of network issues and data problems

## Quick Start

### 1. Basic Usage (Load 10 stocks)
```bash
# Simple start - loads top 10 S&P 500 stocks
./start_data_loading.sh
```

### 2. Load More Stocks in Background
```bash
# Load 100 stocks in background with monitoring
./start_data_loading.sh --stocks 100 --background
```

### 3. Resume Previous Session
```bash
# Continue from where you left off
./start_data_loading.sh --resume --background
```

### 4. Validate Data Quality
```bash
# Check loaded data quality
./start_data_loading.sh --validate-only
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--stocks N` | Number of stocks to load (default: 10) | `--stocks 100` |
| `--background` | Run with real-time monitoring | `--background` |
| `--resume` | Continue from previous progress | `--resume` |
| `--validate-only` | Only validate existing data | `--validate-only` |
| `--monitor-only` | Monitor existing pipeline | `--monitor-only` |
| `--help` | Show help message | `--help` |

## Architecture

### Components

1. **DatabaseManager**: Handles PostgreSQL connections and schema management
2. **StockDataLoader**: Manages parallel data loading with progress tracking
3. **CostMonitor**: Tracks API usage and prevents budget overruns
4. **DataValidator**: Validates data quality and detects issues
5. **CircuitBreaker**: Prevents cascading failures during API issues

### Data Flow

```
Yahoo Finance API → Data Loader → Data Validation → PostgreSQL
                     ↓
              Progress Tracking → Cache Files
                     ↓
              Cost Monitoring → Usage Statistics
```

## File Structure

```
scripts/
├── load_historical_data.py     # Core data loading logic
├── start_data_pipeline.py      # Pipeline orchestration
├── validate_data.py            # Data quality validation
├── start_data_loading.sh       # Easy-to-use startup script
└── data/
    └── cache/
        ├── loading_progress.json    # Loading progress state
        └── pipeline_status.json     # Real-time status
```

## Database Schema

The system populates these key tables:

- **stocks**: Master stock records (ticker, name, sector, etc.)
- **exchanges**: Stock exchanges (NYSE, NASDAQ, AMEX)
- **sectors**: Market sectors (Technology, Healthcare, etc.)
- **price_history**: OHLCV data with calculated fields
- **api_usage**: API call tracking for cost monitoring

## Data Loading Process

### Phase 1: Initialization
1. Connect to PostgreSQL database
2. Create/verify table structure
3. Setup exchanges and sectors
4. Initialize cost monitoring

### Phase 2: Stock Processing
1. Get S&P 500 stock list (prioritized by market cap)
2. Check for existing data (incremental loading)
3. Fetch historical data via Yahoo Finance
4. Validate and clean data
5. Save to database with calculated metrics

### Phase 3: Monitoring & Validation
1. Real-time progress tracking
2. Error logging and recovery
3. Data quality validation
4. Cost monitoring and alerts

## Data Validation

The system performs comprehensive validation:

### Structure Validation
- ✅ Table existence and record counts
- ✅ Database connectivity
- ✅ Schema integrity

### Data Quality Validation
- ✅ OHLC data consistency (High ≥ Low, etc.)
- ✅ Volume data validation
- ✅ Price change reasonableness
- ✅ Date coverage and gaps

### Completeness Validation
- ✅ Stock coverage percentage
- ✅ Data recency (within 7 days)
- ✅ Missing fields detection

## Cost Monitoring

The system tracks API usage to stay under the $50/month budget:

- **Yahoo Finance**: Free (unlimited for reasonable use)
- **Rate Limiting**: Built-in delays between requests
- **Usage Tracking**: All API calls logged to database
- **Fallback Strategies**: Multiple data sources available
- **Budget Alerts**: Warnings when approaching limits

## Error Handling

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Database connection failed | PostgreSQL not running | Start: `docker-compose up postgres -d` |
| Rate limit exceeded | Too many requests | System auto-throttles, wait and retry |
| Invalid stock data | Yahoo Finance data issue | System skips and logs issue |
| Network timeout | Internet connectivity | Circuit breaker activates, auto-retry |
| Disk space low | Large data volumes | Monitor `logs/` and `data/` directories |

### Recovery Mechanisms

1. **Progress Persistence**: Resume from last successful stock
2. **Circuit Breaker**: Fail fast on repeated errors
3. **Incremental Loading**: Skip existing data
4. **Error Logging**: Detailed logs in `logs/data_pipeline.log`

## Monitoring

### Real-Time Monitoring
```bash
# Monitor active pipeline
./start_data_loading.sh --monitor-only
```

### Progress Files
- `scripts/data/cache/loading_progress.json`: Individual stock progress
- `scripts/data/cache/pipeline_status.json`: Overall system status
- `logs/data_pipeline.log`: Detailed execution logs

### Status Information
- Stocks completed/failed/skipped
- Total records loaded
- API calls made
- Data quality score
- Budget utilization

## Performance Optimization

### Parallel Processing
- **Default**: 5 concurrent workers
- **Configurable**: Adjust via `--max-workers`
- **Rate Limited**: Respects API constraints
- **Memory Efficient**: Streams data to database

### Caching Strategy
- **Progress Caching**: Resume capability
- **Data Deduplication**: Skip existing records
- **Error Caching**: Avoid repeated failures

## Data Quality Metrics

The system calculates a quality score (0-100) based on:

- **Structure**: Tables exist and have data (20 points)
- **Coverage**: Percentage of stocks with data (20 points)  
- **Validity**: OHLC data consistency (20 points)
- **Completeness**: Missing fields and gaps (20 points)
- **Recency**: Data freshness (20 points)

### Quality Thresholds
- **90-100**: Excellent quality ✅
- **75-89**: Good quality ✅
- **60-74**: Fair quality ⚠️
- **0-59**: Poor quality ❌

## Troubleshooting

### Check System Status
```bash
# Validate current data
python scripts/validate_data.py --detailed

# Check database connection
psql "postgresql://postgres:password@localhost:5432/investment_db" -c "SELECT COUNT(*) FROM stocks;"

# View recent logs
tail -f logs/data_pipeline.log
```

### Common Problems

**Problem**: "Database connection failed"
```bash
# Solution: Start PostgreSQL
docker-compose up postgres -d
# Or check connection string in .env file
```

**Problem**: "No data loaded after running"
```bash
# Solution: Check for network issues or API problems
python scripts/validate_data.py
tail -n 50 logs/data_pipeline.log
```

**Problem**: "Rate limit exceeded"
```bash
# Solution: Wait and resume (system auto-throttles)
./start_data_loading.sh --resume
```

## Next Steps After Data Loading

1. **Validate Data Quality**
   ```bash
   ./start_data_loading.sh --validate-only
   ```

2. **Start API Server**
   ```bash
   uvicorn backend.api.main:app --reload --port 8000
   ```

3. **Access API Documentation**
   - Open http://localhost:8000/docs
   - Test endpoints with loaded data

4. **Set Up Daily Updates**
   - Configure Airflow DAGs
   - Schedule incremental updates
   - Monitor data freshness

5. **Enable Analytics**
   - Start ML model training
   - Configure technical indicators
   - Set up recommendation engine

## Support

For issues or questions:

1. Check logs in `logs/data_pipeline.log`
2. Run validation: `./start_data_loading.sh --validate-only`
3. Review this README for common solutions
4. Check the main project documentation

---

**Budget Status**: The data loading system is designed to operate within the $50/month budget by using free data sources and efficient processing. Monitor API usage via the cost monitoring system.