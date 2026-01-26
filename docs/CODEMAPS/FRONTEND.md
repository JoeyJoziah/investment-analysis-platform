# Frontend Architecture Codemap

## Pages (`frontend/web/src/pages/`)

| Page | File | Purpose |
|------|------|---------|
| Dashboard | `Dashboard.tsx` | Main dashboard with widgets |
| Analysis | `Analysis.tsx` | Stock analysis view |
| Portfolio | `Portfolio.tsx` | Portfolio management |
| Recommendations | `Recommendations.tsx` | AI recommendations |
| Watchlist | `Watchlist.tsx` | Custom watchlists |
| MarketOverview | `MarketOverview.tsx` | Market summary |
| Alerts | `Alerts.tsx` | Price/event alerts |
| Reports | `Reports.tsx` | Analytics reports |
| Settings | `Settings.tsx` | User preferences |
| Login | `Login.tsx` | Authentication |
| Help | `Help.tsx` | Documentation |

## Components (`frontend/web/src/components/`)

| Directory | Purpose |
|-----------|---------|
| `charts/` | Plotly/Recharts visualizations |
| `cards/` | Information cards |
| `panels/` | Dashboard panels |
| `tables/` | Data tables |
| `forms/` | Input forms |
| `alerts/` | Notification components |
| `dashboard/` | Dashboard-specific widgets |
| `navigation/` | Nav bars, sidebars |
| `common/` | Shared utilities |
| `modals/` | Modal dialogs |
| `indicators/` | Status indicators |
| `layout/` | Page layouts |

## Redux Store (`frontend/web/src/store/`)

| Slice | File | Purpose |
|-------|------|---------|
| stocks | `stocksSlice.ts` | Stock data state |
| portfolio | `portfolioSlice.ts` | Portfolio state |
| recommendations | `recommendationsSlice.ts` | Recommendation state |
| watchlist | `watchlistSlice.ts` | Watchlist state |
| auth | `authSlice.ts` | Authentication state |
| ui | `uiSlice.ts` | UI state (loading, errors) |

## Custom Hooks (`frontend/web/src/hooks/`)

| Hook | Purpose |
|------|---------|
| `useRealTimeData.ts` | WebSocket data subscription |
| `usePerformanceMonitor.ts` | Performance tracking |
| `useStockData.ts` | Stock data fetching |
| `usePortfolio.ts` | Portfolio operations |
| `useAuth.ts` | Authentication helpers |
| `useDebounce.ts` | Input debouncing |

## Services (`frontend/web/src/services/`)

| Service | Purpose |
|---------|---------|
| `api.ts` | Base API client |
| `stockService.ts` | Stock API calls |
| `portfolioService.ts` | Portfolio API calls |
| `authService.ts` | Auth API calls |
| `websocketService.ts` | WebSocket connection |

## Types (`frontend/web/src/types/`)

| File | Purpose |
|------|---------|
| `stock.ts` | Stock type definitions |
| `portfolio.ts` | Portfolio types |
| `recommendation.ts` | Recommendation types |
| `user.ts` | User types |
| `api.ts` | API response types |

## Key Component Paths

### Dashboard
```
Dashboard.tsx
├── DashboardHeader
├── MarketSummaryWidget
├── TopMoversWidget
├── RecommendationsWidget
├── PortfolioSummaryWidget
└── AlertsWidget
```

### Analysis Page
```
Analysis.tsx
├── StockSearchBar
├── PriceChart
├── TechnicalIndicators
├── FundamentalsPanel
├── SentimentGauge
└── PredictionsPanel
```

### Portfolio Page
```
Portfolio.tsx
├── PortfolioSummary
├── HoldingsTable
├── PerformanceChart
├── AllocationPie
└── TransactionHistory
```

## Styling

| Approach | Usage |
|----------|-------|
| Material-UI 5.14 | Component library |
| CSS Modules | Component-specific styles |
| Theme Provider | Global theming |

## Build Configuration

| File | Purpose |
|------|---------|
| `vite.config.ts` | Vite bundler config |
| `tsconfig.json` | TypeScript config |
| `.eslintrc.js` | ESLint rules |
| `jest.config.js` | Test configuration |

## Testing

| Type | Coverage | Command |
|------|----------|---------|
| Unit Tests | 84 tests | `npm test` |
| Integration | Included | `npm test` |
| E2E | Playwright | `npm run e2e` |

**Last Updated**: 2026-01-26
