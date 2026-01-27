# Investment Thesis Documentation Feature

## Overview

The Investment Thesis feature provides comprehensive documentation and analysis tools for investment decisions. It allows users to create, manage, and version structured investment theses for stocks in their portfolio or watchlist.

## Features

### Backend
- **SQLAlchemy Model**: `InvestmentThesis` with comprehensive fields
- **RESTful API**: Full CRUD operations with authentication
- **Repository Pattern**: Async repository for efficient database operations
- **User Scoping**: Each thesis is scoped to a specific user and stock
- **Version Tracking**: Automatic version incrementation on updates
- **Validation**: Pydantic schemas with field validation

### Frontend
- **React Component**: `InvestmentThesis.tsx` page
- **Markdown Editor**: Rich text editing for thesis content
- **Template Loading**: Pre-built comprehensive thesis template
- **Export Options**: Markdown and PDF export capabilities
- **Real-time Saving**: Auto-save and version management

## Database Schema

### Table: `investment_thesis`

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| user_id | Integer | Foreign key to users table |
| stock_id | Integer | Foreign key to stocks table |
| investment_objective | Text | Primary investment goal |
| time_horizon | String(50) | Expected holding period (short/medium/long-term) |
| target_price | Decimal(10,2) | Price target based on valuation |
| business_model | Text | Description of how company makes money |
| competitive_advantages | Text | Moats and competitive positioning |
| financial_health | Text | Balance sheet and cash flow analysis |
| growth_drivers | Text | Key factors driving future growth |
| risks | Text | Risk assessment and mitigation |
| valuation_rationale | Text | Valuation methodology |
| exit_strategy | Text | Conditions for selling |
| content | Text | Complete thesis in markdown format |
| version | Integer | Version number (auto-incremented) |
| created_at | DateTime | Creation timestamp |
| updated_at | DateTime | Last update timestamp |

### Indexes
- `idx_thesis_user_stock` - Composite index on (user_id, stock_id)
- `idx_thesis_updated_at` - Index on updated_at for sorting
- `idx_thesis_user_updated` - Composite index on (user_id, updated_at)

## API Endpoints

All endpoints require authentication via Bearer token.

### Create Thesis
```
POST /api/v1/thesis/
```

**Request Body:**
```json
{
  "stock_id": 123,
  "investment_objective": "Long-term growth investment",
  "time_horizon": "long-term",
  "target_price": 200.00,
  "business_model": "Tech company...",
  "competitive_advantages": "Strong brand...",
  "financial_health": "Excellent balance sheet...",
  "growth_drivers": "New markets...",
  "risks": "Market competition...",
  "valuation_rationale": "DCF analysis...",
  "exit_strategy": "Sell at target or 2x return",
  "content": "# Full Thesis\n\nDetailed analysis..."
}
```

**Response:** `201 Created`
```json
{
  "id": 456,
  "user_id": 789,
  "stock_id": 123,
  "investment_objective": "Long-term growth investment",
  "time_horizon": "long-term",
  "target_price": 200.00,
  "version": 1,
  "created_at": "2026-01-27T18:00:00Z",
  "updated_at": "2026-01-27T18:00:00Z",
  "stock_symbol": "AAPL",
  "stock_name": "Apple Inc."
}
```

### Get Thesis by ID
```
GET /api/v1/thesis/{thesis_id}
```

**Response:** `200 OK`

### Get Thesis by Stock ID
```
GET /api/v1/thesis/stock/{stock_id}
```

**Response:** `200 OK` or `404 Not Found`

### List User's Theses
```
GET /api/v1/thesis/?limit=50&offset=0
```

**Response:** `200 OK` - Array of theses

### Update Thesis
```
PUT /api/v1/thesis/{thesis_id}
```

**Request Body:** (all fields optional)
```json
{
  "investment_objective": "Updated objective",
  "target_price": 250.00,
  "content": "# Updated Content..."
}
```

**Response:** `200 OK` - Version automatically incremented

### Delete Thesis
```
DELETE /api/v1/thesis/{thesis_id}
```

**Response:** `204 No Content`

## Frontend Usage

### Accessing the Page
Navigate to `/thesis/{stockId}` where `stockId` is the numeric ID of the stock.

### Creating a New Thesis
1. Navigate to the thesis page for a stock
2. Fill in the required fields:
   - Investment Objective (required)
   - Time Horizon (required)
   - Target Price (optional)
3. Click "Load Template" to start with a comprehensive structure
4. Edit the markdown content in the editor
5. Click "Create Thesis" to save

### Editing an Existing Thesis
1. The page automatically loads the existing thesis
2. Make changes to any fields or the markdown content
3. Click "Update Thesis" to save
4. Version is automatically incremented

### Exporting
- **Export as Markdown**: Download the thesis as a `.md` file
- **Export as PDF**: (TODO) Download as formatted PDF

## Template Structure

The investment thesis template includes:

1. **Executive Summary** - 3-5 sentence overview
2. **Company Overview** - Business model and market position
3. **Competitive Advantages (Moats)** - Sustainable competitive advantages
4. **Financial Health** - Income statement, balance sheet, cash flow
5. **Growth Drivers** - Factors driving future growth
6. **Risk Assessment** - Bear/Base/Bull cases with mitigation
7. **Valuation Analysis** - Multiple valuation methods
8. **Investment Strategy** - Entry/exit strategy and monitoring
9. **Catalysts** - Near-term events that could move the stock
10. **Decision Log** - Track actions and outcomes over time
11. **Supporting Research** - Sources and competitor comparison

## Installation & Setup

### Backend Migration
```bash
# Run the migration
cd backend
alembic upgrade head
```

### Frontend Dependencies
```bash
# Install Monaco Editor for rich markdown editing
cd frontend/web
npm install @monaco-editor/react

# The basic implementation uses a TextField as fallback
# For the full Monaco Editor experience, update InvestmentThesis.tsx
# to import and use the Editor component
```

### Router Registration
The thesis router is automatically registered in `backend/api/main.py`:
```python
from backend.api.routers import thesis
app.include_router(thesis.router, prefix="/api/v1", tags=["investment-thesis"])
```

## Testing

### Run Backend Tests
```bash
cd backend
pytest tests/test_thesis_api.py -v
```

### Test Coverage
The test suite includes:
- ✅ Create thesis (success and validation errors)
- ✅ Get thesis by ID
- ✅ Get thesis by stock ID
- ✅ List user theses with pagination
- ✅ Update thesis (with version increment)
- ✅ Delete thesis
- ✅ Authorization checks (users can't access others' theses)
- ✅ Authentication requirements
- ✅ Duplicate prevention

## Security Considerations

1. **Authentication Required**: All endpoints require valid JWT token
2. **User Scoping**: Users can only access their own theses
3. **Input Validation**: All fields validated via Pydantic schemas
4. **SQL Injection Prevention**: SQLAlchemy ORM with parameterized queries
5. **Foreign Key Constraints**: CASCADE delete on user/stock deletion

## Performance Optimizations

1. **Composite Indexes**: Fast lookups by user+stock
2. **Async Operations**: Non-blocking database queries
3. **Pagination**: Limit query results to prevent large payloads
4. **Selective Loading**: Only load required fields

## Future Enhancements

- [ ] PDF export implementation (using jsPDF or html2pdf)
- [ ] Rich Monaco Editor integration with syntax highlighting
- [ ] Thesis comparison across different versions
- [ ] Sharing theses with other users (read-only)
- [ ] AI-powered thesis suggestions and analysis
- [ ] Thesis templates for different investment styles
- [ ] Automatic thesis updates based on market events
- [ ] Integration with portfolio decisions (link thesis to transactions)

## File Structure

```
backend/
├── models/
│   ├── thesis.py                      # SQLAlchemy model
│   └── schemas.py                     # Pydantic schemas (updated)
├── repositories/
│   └── thesis_repository.py           # Async repository
├── api/routers/
│   └── thesis.py                      # API endpoints
├── migrations/versions/
│   └── 010_add_investment_thesis.py   # Database migration
└── tests/
    └── test_thesis_api.py             # API tests

frontend/web/src/
└── pages/
    └── InvestmentThesis.tsx           # React component

docs/
├── templates/
│   └── investment_thesis_template.md  # Comprehensive template
└── INVESTMENT_THESIS_FEATURE.md       # This file
```

## Maintenance

- **Database Backups**: Ensure theses are included in backup strategy
- **Monitoring**: Track API usage and performance metrics
- **User Feedback**: Collect feedback on template structure and features

## Support

For issues or questions:
1. Check the API documentation at `/api/docs` (when DEBUG=True)
2. Review test cases for usage examples
3. Consult the template for structure guidance

---

**Version:** 1.0.0
**Last Updated:** 2026-01-27
**Author:** Investment Analysis Platform Team
