# Portfolio Real-Time Analytics - Quick Reference Guide

## Files at a Glance

### Backend (Python)
```
backend/
├── services/
│   └── realtime_price_service.py          (17KB) - Price service with Finnhub WebSocket
└── api/routers/
    └── portfolio.py                        (Modified) - Updated endpoints with real-time prices
```

### Frontend (TypeScript/React)
```
frontend/web/src/
├── hooks/
│   └── usePortfolioWebSocket.ts           (7.4KB) - WebSocket connection hook
├── components/
│   ├── CorrelationMatrix.tsx              (5.6KB) - Asset correlation heatmap
│   ├── EfficientFrontier.tsx              (7.8KB) - ML-based efficient frontier
│   └── RiskDecomposition.tsx              (9.2KB) - Risk analysis chart
└── pages/
    └── Portfolio.tsx                       (Modified) - Integrated with WebSocket
```

### Documentation
```
├── WEBSOCKET_IMPLEMENTATION.md             - Complete architecture & specs
├── PHASE_3_2_IMPLEMENTATION_SUMMARY.md     - Implementation overview
└── QUICK_REFERENCE.md                      - This file
```

## Key Imports

### Python
```python
# Get the price service
from backend.services.realtime_price_service import get_realtime_price_service

# Use it
price_service = await get_realtime_price_service()
prices = await price_service.get_latest_prices_bulk(['AAPL', 'GOOGL'], db)
```

### TypeScript
```typescript
// Use the WebSocket hook
import { usePortfolioWebSocket } from '../hooks/usePortfolioWebSocket';

const { isConnected, priceUpdates } = usePortfolioWebSocket(
  portfolioId,
  symbols,
  true  // enabled
);
```

### React Components
```typescript
import CorrelationMatrix from '../components/CorrelationMatrix';
import EfficientFrontier from '../components/EfficientFrontier';
import RiskDecomposition from '../components/RiskDecomposition';

// Use in JSX
<CorrelationMatrix correlations={metrics.correlationMatrix} />
<EfficientFrontier frontier={frontier} currentPortfolio={current} />
<RiskDecomposition components={riskComponents} totalRisk={risk} />
```

## API Endpoints

### Portfolio Endpoints (Updated)
```
GET /portfolio/summary
  - Real-time portfolio values
  - Uses RealtimePriceService for bulk price fetch
  - Cache: 1 minute

GET /portfolio/{portfolio_id}
  - Detailed portfolio with real-time prices
  - Includes allocations and metrics
  - Cache: 30 seconds
```

### WebSocket Endpoints (Existing)
```
WS /ws/stream?client_id={id}
  - Main real-time stream
  - Subscribe/unsubscribe to symbols
  - Heartbeat every 30 seconds

WS /ws/portfolio/{portfolio_id}
  - Portfolio-specific updates
  - Aggregated position data
```

## Configuration

### Environment Variables
```bash
# Required
FINNHUB_API_KEY=sk_...              # For real market data

# Optional
REDIS_URL=redis://localhost:6379    # For price caching
```

### Startup Code (main.py)
```python
@app.on_event("startup")
async def startup():
    # Initialize WebSocket cleanup
    from backend.api.routers.websocket import start_cleanup_task
    start_cleanup_task()

    # Initialize price service
    await get_realtime_price_service()

@app.on_event("shutdown")
async def shutdown():
    from backend.services.realtime_price_service import shutdown_realtime_price_service
    await shutdown_realtime_price_service()
```

## Common Tasks

### Get Latest Prices
```python
service = await get_realtime_price_service()

# Single price
price = await service.get_latest_price('AAPL', db)
print(f"AAPL: ${price.price}")

# Multiple prices (recommended)
prices = await service.get_latest_prices_bulk(['AAPL', 'GOOGL', 'MSFT'], db)
for symbol, price_update in prices.items():
    print(f"{symbol}: ${price_update.price}")
```

### Subscribe to Price Updates
```python
# Define callback
async def on_price_update(update):
    print(f"{update.symbol}: ${update.price}")

# Subscribe
await service.subscribe_to_symbol('AAPL', on_price_update, db)

# Later: unsubscribe
await service.unsubscribe_from_symbol('AAPL')
```

### Use WebSocket in Component
```typescript
const Portfolio = () => {
  const { isConnected, priceUpdates, latency } = usePortfolioWebSocket(
    'portfolio-123',
    ['AAPL', 'GOOGL'],
    true
  );

  // Real-time price from WebSocket
  const aaplPrice = priceUpdates.get('AAPL')?.price;

  return (
    <>
      {isConnected && <span>LIVE ({latency}ms)</span>}
      <p>AAPL: ${aaplPrice}</p>
    </>
  );
};
```

## Troubleshooting

### WebSocket Shows "OFFLINE"
1. Check FINNHUB_API_KEY environment variable
2. Check network connectivity
3. Look at browser console for errors
4. Verify Finnhub API status

### Prices Not Updating
1. Verify WebSocket is connected (LIVE badge)
2. Check symbols are subscribed
3. Look at server logs: `grep realtime_price_service /var/log/app.log`
4. Check Redis: `redis-cli KEYS "price:*"`

### High Latency (>1s)
1. Check network conditions (DevTools Network tab)
2. Check server CPU usage
3. Verify Finnhub API performance
4. Check database connection pool

### Memory Leak
1. Ensure components clean up WebSocket hook on unmount
2. Verify subscriptions are unsubscribed
3. Check Redis TTL is set (5 minutes default)
4. Monitor Redis memory: `redis-cli INFO memory`

## Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| Price Update Latency | <1s | From Finnhub to UI |
| Dashboard Load | <2s | With cached data |
| WebSocket Reconnect | <30s | Exponential backoff |
| Memory per Connection | ~5KB | In-memory subscription |
| Concurrent Connections | ~1000 | Per FastAPI process |

## Testing

### Manual Testing Checklist
- [ ] Dashboard loads in <2 seconds
- [ ] WebSocket shows "LIVE" badge
- [ ] Prices update real-time in table
- [ ] Latency displays correct value
- [ ] Metrics recalculate on price changes
- [ ] Disconnect -> "OFFLINE" badge appears
- [ ] Reconnect works after 1-2 seconds
- [ ] Correlation matrix displays
- [ ] Efficient frontier renders
- [ ] Risk decomposition shows data

### Load Testing
```bash
# Monitor connections
while true; do
  echo "=== $(date) ==="
  curl http://localhost:8000/api/ws/health | jq .
  sleep 5
done
```

## Monitoring

### Health Check Endpoint
```bash
curl http://localhost:8000/api/ws/health
# Returns:
{
  "active_connections": 5,
  "subscriptions": {"client1": ["AAPL", "GOOGL"]},
  "active_streams": ["AAPL", "GOOGL"],
  "price_service_initialized": true
}
```

### Key Metrics to Monitor
- Active WebSocket connections
- Price update latency
- Finnhub API usage
- Redis cache hit rate
- Database query performance
- Memory usage per connection

## Documentation Links

- **Full Implementation**: `WEBSOCKET_IMPLEMENTATION.md`
- **Phase Summary**: `PHASE_3_2_IMPLEMENTATION_SUMMARY.md`
- **This Guide**: `QUICK_REFERENCE.md`

## Support

For issues or questions:
1. Check the full implementation docs
2. Review the code comments
3. Check server logs for errors
4. Verify environment configuration
5. Test with manual curl/WebSocket client

---

**Last Updated**: 2026-01-27
**Status**: Production Ready
**Version**: 1.0
