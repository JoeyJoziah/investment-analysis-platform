# API V1 to V2 Migration Guide

## Overview

API V1 has been **sunset as of July 1, 2025**. All clients must migrate to API V2 or V3 to continue using the Investment Analysis Platform API.

This guide provides detailed instructions for migrating from V1 to V2, including endpoint mappings, parameter changes, authentication updates, and code examples.

## Timeline

| Date | Status | Action Required |
|------|--------|-----------------|
| January 1, 2025 | V1 Deprecated | Begin migration planning |
| July 1, 2025 | V1 Sunset | **V1 no longer operational** |
| August 1, 2025 | Grace Period End | All V1 requests return 410 Gone |

## Quick Reference: Breaking Changes

### 1. Authentication Changes
- **V1**: Simple API key authentication via `X-API-Key` header
- **V2**: OAuth2 Bearer token authentication

### 2. Parameter Renames
| V1 Parameter | V2 Parameter |
|--------------|--------------|
| `ticker` | `symbol` |
| `stock_id` | `symbol` |
| `page_size` | `limit` |
| `page_num` | `offset` |

### 3. Response Structure Changes
- Stock list responses now use `symbol` field instead of `ticker`
- Pagination uses `offset/limit` instead of `page_num/page_size`
- Error responses have new structured format

## Endpoint Migration Map

### Stock Endpoints

| V1 Endpoint | V2 Endpoint | Notes |
|------------|-------------|-------|
| `GET /api/v1/stocks` | `GET /api/stocks` | Response structure changed |
| `GET /api/v1/stocks/search` | `GET /api/stocks/search` | Same functionality |
| `GET /api/v1/stocks/sectors` | `GET /api/stocks/sectors` | Same functionality |
| `GET /api/v1/stock/{ticker}` | `GET /api/stocks/{symbol}` | Path parameter renamed |
| `GET /api/v1/stock/{ticker}/quote` | `GET /api/stocks/{symbol}/quote` | Enhanced response |
| `GET /api/v1/stock/{ticker}/history` | `GET /api/stocks/{symbol}/history` | Same functionality |
| `GET /api/v1/stock/{ticker}/statistics` | `GET /api/stocks/{symbol}/statistics` | Same functionality |

### Analysis Endpoints

| V1 Endpoint | V2 Endpoint | Notes |
|------------|-------------|-------|
| `POST /api/v1/analysis/analyze` | `POST /api/analysis/analyze` | Same functionality |
| `GET /api/v1/analysis/{symbol}` | `POST /api/analysis/analyze` | Changed to POST |
| `POST /api/v1/analysis/batch` | `POST /api/analysis/batch` | Same functionality |
| `POST /api/v1/analysis/compare` | `POST /api/analysis/compare` | Same functionality |
| `GET /api/v1/analysis/indicators/{symbol}` | `GET /api/analysis/indicators/{symbol}` | Same functionality |
| `GET /api/v1/analysis/sentiment/{symbol}` | `GET /api/analysis/sentiment/{symbol}` | Same functionality |

### Portfolio Endpoints

| V1 Endpoint | V2 Endpoint | Notes |
|------------|-------------|-------|
| `GET /api/v1/portfolio` | `GET /api/portfolio` | Same functionality |
| `GET /api/v1/portfolio/{id}` | `GET /api/portfolio/{id}` | Same functionality |
| `GET /api/v1/portfolio/{id}/holdings` | `GET /api/portfolio/{id}/holdings` | Same functionality |
| `GET /api/v1/portfolio/{id}/performance` | `GET /api/portfolio/{id}/performance` | Same functionality |

### Authentication Endpoints

| V1 Endpoint | V2 Endpoint | Notes |
|------------|-------------|-------|
| `POST /api/v1/auth/login` | `POST /api/auth/login` | OAuth2 format |
| `POST /api/v1/auth/register` | `POST /api/auth/register` | Same functionality |
| `POST /api/v1/auth/token` | `POST /api/auth/token` | OAuth2 format |
| `POST /api/v1/auth/refresh` | `POST /api/auth/refresh` | Same functionality |
| `GET /api/v1/auth/me` | `GET /api/auth/me` | Same functionality |

### Watchlist Endpoints

| V1 Endpoint | V2 Endpoint | Notes |
|------------|-------------|-------|
| `GET /api/v1/watchlist` | `GET /api/watchlists/default` | New URL structure |
| `POST /api/v1/watchlist/add/{ticker}` | `POST /api/watchlists/default/symbols/{symbol}` | New URL structure |
| `DELETE /api/v1/watchlist/remove/{ticker}` | `DELETE /api/watchlists/default/symbols/{symbol}` | New URL structure |

### Recommendations Endpoints

| V1 Endpoint | V2 Endpoint | Notes |
|------------|-------------|-------|
| `GET /api/v1/recommendations` | `GET /api/recommendations` | Same functionality |
| `GET /api/v1/recommendations/{symbol}` | `GET /api/recommendations/{symbol}` | Same functionality |

## Authentication Migration

### V1 Authentication (Deprecated)

```bash
# V1: Simple API key
curl -X GET "https://api.example.com/api/v1/stocks" \
  -H "X-API-Key: your-api-key"
```

### V2 Authentication (Current)

```bash
# Step 1: Get OAuth2 token
curl -X POST "https://api.example.com/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your-email@example.com&password=your-password"

# Response:
# {"access_token": "eyJ...", "token_type": "bearer"}

# Step 2: Use bearer token for requests
curl -X GET "https://api.example.com/api/stocks" \
  -H "Authorization: Bearer eyJ..."
```

## Parameter Migration Examples

### Pagination

**V1 (Deprecated):**
```bash
GET /api/v1/stocks?page_num=2&page_size=50
```

**V2 (Current):**
```bash
# offset = (page_num - 1) * page_size
# For page 2 with 50 items: offset = (2-1) * 50 = 50
GET /api/stocks?offset=50&limit=50
```

### Stock Symbol Parameter

**V1 (Deprecated):**
```bash
GET /api/v1/stock/AAPL/quote
```

**V2 (Current):**
```bash
GET /api/stocks/AAPL/quote
```

## Response Structure Changes

### Stock List Response

**V1 Response (Deprecated):**
```json
{
  "stocks": [
    {
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "sector": "Technology",
      "market_cap": 3000000000000
    }
  ],
  "page_num": 1,
  "page_size": 50,
  "total": 500
}
```

**V2 Response (Current):**
```json
{
  "stocks": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "sector": "Technology",
      "market_cap": 3000000000000
    }
  ],
  "total_count": 500,
  "page": 1,
  "per_page": 50
}
```

### Error Response

**V1 Error Response (Deprecated):**
```json
{
  "error": "Stock not found",
  "error_code": "ERR002"
}
```

**V2 Error Response (Current):**
```json
{
  "error": "Stock not found",
  "timestamp": "2025-07-15T10:30:00Z",
  "path": "/api/stocks/INVALID"
}
```

## Code Migration Examples

### Python (requests library)

**V1 Code (Deprecated):**
```python
import requests

# V1 Authentication
headers = {"X-API-Key": "your-api-key"}

# V1 Get stocks
response = requests.get(
    "https://api.example.com/api/v1/stocks",
    headers=headers,
    params={"page_num": 1, "page_size": 50}
)
stocks = response.json()["stocks"]

# V1 Get stock by ticker
response = requests.get(
    f"https://api.example.com/api/v1/stock/AAPL/quote",
    headers=headers
)
```

**V2 Code (Current):**
```python
import requests

# V2 Authentication - Get token first
auth_response = requests.post(
    "https://api.example.com/api/auth/token",
    data={"username": "email@example.com", "password": "password"}
)
token = auth_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# V2 Get stocks (note: offset/limit instead of page_num/page_size)
response = requests.get(
    "https://api.example.com/api/stocks",
    headers=headers,
    params={"offset": 0, "limit": 50}
)
stocks = response.json()["stocks"]

# V2 Get stock by symbol (note: /stocks/ instead of /stock/)
response = requests.get(
    f"https://api.example.com/api/stocks/AAPL/quote",
    headers=headers
)
```

### JavaScript (fetch)

**V1 Code (Deprecated):**
```javascript
// V1 Authentication
const headers = { "X-API-Key": "your-api-key" };

// V1 Get stocks
const response = await fetch(
  "https://api.example.com/api/v1/stocks?page_num=1&page_size=50",
  { headers }
);
const { stocks } = await response.json();

// V1 Access ticker field
stocks.forEach(stock => console.log(stock.ticker));
```

**V2 Code (Current):**
```javascript
// V2 Authentication
const authResponse = await fetch("https://api.example.com/api/auth/token", {
  method: "POST",
  headers: { "Content-Type": "application/x-www-form-urlencoded" },
  body: "username=email@example.com&password=password"
});
const { access_token } = await authResponse.json();
const headers = { "Authorization": `Bearer ${access_token}` };

// V2 Get stocks (note: offset/limit)
const response = await fetch(
  "https://api.example.com/api/stocks?offset=0&limit=50",
  { headers }
);
const { stocks } = await response.json();

// V2 Access symbol field (not ticker)
stocks.forEach(stock => console.log(stock.symbol));
```

## Deprecation Headers

When making requests during the deprecation period, V1 responses included these headers:

| Header | Description | Example |
|--------|-------------|---------|
| `Sunset` | RFC 8594 sunset date | `Tue, 01 Jul 2025 00:00:00 GMT` |
| `Deprecation` | Deprecation announcement date | `Wed, 01 Jan 2025 00:00:00 GMT` |
| `Link` | Successor version | `</api/v3>; rel="successor-version"` |
| `Warning` | Human-readable warning | `299 - "API V1 is deprecated..."` |
| `X-API-Version` | Current API version | `v1` |
| `X-API-Status` | Version status | `deprecated` or `sunset` |
| `X-Migration-Guide` | Link to this guide | `/api/docs/migration/v1-to-v2` |

## Monitoring Your Migration

### Check Migration Metrics

Administrators can monitor V1 usage via:

```bash
GET /api/admin/v1-migration/metrics
```

Response:
```json
{
  "status": "success",
  "data": {
    "total_requests": 150000,
    "v1_requests": 5000,
    "v2_requests": 100000,
    "v3_requests": 45000,
    "v1_percentage": 3.33,
    "migration_complete": false,
    "top_v1_endpoints": [
      ["/api/v1/stocks", 2500],
      ["/api/v1/stock/AAPL/quote", 1200]
    ]
  }
}
```

### Check Client Usage

```bash
GET /api/admin/v1-migration/clients
```

## Troubleshooting

### Error: 410 Gone

If you receive a `410 Gone` response:

```json
{
  "error": "API version no longer supported",
  "code": "API_VERSION_SUNSET",
  "message": "API V1 was sunset on 2025-07-01. Please migrate to API V2 or V3.",
  "migration": {
    "current_endpoint": "/api/v1/stocks",
    "suggested_endpoint": "/api/stocks",
    "migration_guide": "/api/docs/migration/v1-to-v2"
  }
}
```

**Solution:** Update your API calls to use V2 endpoints as described in this guide.

### Error: 401 Unauthorized (after migration)

If you receive `401 Unauthorized` after migrating:

**Solution:** Ensure you're using OAuth2 Bearer token authentication instead of the old API key method.

### Error: Field 'ticker' not found

If your code expects a `ticker` field in responses:

**Solution:** Update your code to use `symbol` instead of `ticker`.

## Support

If you encounter issues during migration:

1. Check this migration guide for common solutions
2. Review the API documentation at `/api/docs`
3. Contact support with your client ID and specific error messages

## V3 Preview

Consider migrating directly to V3 for access to:

- GraphQL support
- Real-time streaming via WebSockets
- Advanced analytics endpoints
- Machine learning predictions
- Enhanced pagination format

V3 documentation: `/api/docs#v3`
