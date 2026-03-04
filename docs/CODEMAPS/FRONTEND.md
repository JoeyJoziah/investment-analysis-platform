# Frontend Architecture Codemap

**Last Updated:** 2026-03-04

## Pages (`frontend/web/src/pages/`)

14 pages, all lazy-loaded with `React.lazy` and typed skeleton fallbacks.

| Page | File | Purpose |
|------|------|---------|
| Dashboard | `Dashboard.tsx` | Main dashboard with widgets |
| Analysis | `Analysis.tsx` | Stock analysis view (optional ticker param) |
| Portfolio | `Portfolio.tsx` | Portfolio management (positions, transactions, analysis tabs) |
| Recommendations | `Recommendations.tsx` | AI recommendations (filterable list) |
| Watchlist | `Watchlist.tsx` | Custom watchlists (CRUD) |
| MarketOverview | `MarketOverview.tsx` | Market summary (charts, tickers) |
| Alerts | `Alerts.tsx` | Price/event alerts (form + list) |
| Reports | `Reports.tsx` | Analytics reports |
| Settings | `Settings.tsx` | User preferences (tabbed form) |
| InvestmentThesis | `InvestmentThesis.tsx` | Per-stock thesis view |
| Login | `Login.tsx` | Authentication (demo credentials support) |
| Register | `Register.tsx` | User registration (full validation) |
| ForgotPassword | `ForgotPassword.tsx` | Password reset (prevents email enumeration) |
| Help | `Help.tsx` | Documentation |

## Components (`frontend/web/src/components/`)

54 active component files across 15+ subdirectories (EnhancedDashboard.tsx deleted as dead code).

| Directory | Components |
|-----------|-----------|
| `alerts/` | AlertForm, AlertsList |
| `analysis/` | AnalysisCharts, AnalysisFilters, AnalysisTable |
| `cards/` | EnhancedRecommendationCard, PortfolioSummary, NewsFeedCard |
| `charts/` | MarketHeatmap, Sparkline, StockChart |
| `common/` | ErrorBoundary, LoadingSpinner, PageSkeleton |
| `dashboard/` | DashboardLayout, HoldingsSection, MetricsSection, PerformanceSection |
| `market/` | MarketCharts, MarketSummary, MarketTickers |
| `monitoring/` | CostMonitor |
| `panels/` | AllocationPanel, MarketOverviewPanel, NewsFeedPanel, RecommendationsPanel |
| `portfolio/` | PortfolioActions, PortfolioChart, PortfolioTabs, CorrelationMatrix, EfficientFrontier, RiskDecomposition |
| `recommendations/` | RecommendationFilter, RecommendationList |
| `settings/` | SettingsForm, SettingsTabs |
| `watchlist/` | WatchlistActions, WatchlistTable |
| `Layout/` | App shell, navigation |
| `NotificationPanel/` | Notification panel |
| `SearchModal/` | Global search |
| `WebSocketIndicator/` | Connection status indicator |

Note: `CorrelationMatrix.tsx`, `EfficientFrontier.tsx`, and `RiskDecomposition.tsx`
were relocated from root components/ into `portfolio/` subdirectory (commit 46e7986).

## Redux Store (`frontend/web/src/store/slices/`)

| Slice | File | Purpose |
|-------|------|---------|
| app | `appSlice.ts` | Global app state |
| dashboard | `dashboardSlice.ts` | Dashboard widgets state |
| recommendations | `recommendationsSlice.ts` | Recommendation state |
| portfolio | `portfolioSlice.ts` | Portfolio state |
| market | `marketSlice.ts` | Market overview state |
| stock | `stockSlice.ts` | Stock data state |

## Custom Hooks (`frontend/web/src/hooks/`)

13 performance-oriented hooks.

| Hook | Purpose |
|------|---------|
| `useRealTimePrices.ts` | WebSocket price subscription |
| `usePortfolioWebSocket.ts` | Portfolio real-time updates |
| `useVirtualScroll.ts` | Virtual scrolling for large lists |
| `useDebounce.ts` | Input debouncing |
| `useThrottle.ts` | Event throttling |
| `useLazyLoad.ts` | Lazy loading for images/content |
| `useWebWorker.ts` | Offload computation to Web Worker |
| `usePrefetch.ts` | Route-based prefetching |
| `useStockData.ts` | Stock data fetching |
| `usePortfolio.ts` | Portfolio operations |
| `useAuth.ts` | Authentication helpers |
| `usePerformanceMonitor.ts` | Performance tracking |

## Services (`frontend/web/src/services/`)

| Service | Purpose |
|---------|---------|
| `api.service.ts` | Base Axios client with JWT refresh |
| `stockService.ts` | Stock API calls |
| `portfolioService.ts` | Portfolio API calls |
| `authService.ts` | Auth API calls |
| `websocketService.ts` | Native WebSocket connection |
| `socketioService.ts` | Socket.IO real-time layer |

Note: Typed methods for watchlist/alerts/settings are pending in `api.service.ts`.

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
├── DashboardLayout
├── MarketHeatmap
├── PerformanceSection
├── HoldingsSection
└── MetricsSection
```

### Analysis Page
```
Analysis.tsx
├── StockSearchBar
├── AnalysisCharts
├── AnalysisFilters
├── AnalysisTable
└── SentimentGauge
```

### Portfolio Page
```
Portfolio.tsx
├── PortfolioTabs
├── PortfolioActions
├── PortfolioChart
├── PortfolioSummary
├── CorrelationMatrix    (in portfolio/ subdirectory)
├── EfficientFrontier    (in portfolio/ subdirectory)
└── RiskDecomposition    (in portfolio/ subdirectory)
```

## Styling

| Approach | Usage |
|----------|-------|
| Material-UI 5.14 | Component library with full theming |
| Design tokens | `theme/` directory |
| CSS-in-JS | MUI sx prop + styled components |

Note: `style-src` CSP allows `'unsafe-inline'` specifically for MUI CSS-in-JS runtime.

## Build Configuration

| File | Purpose |
|------|---------|
| `vite.config.ts` | Vite 7.3.1 bundler (18 manual vendor chunks) |
| `tsconfig.json` | TypeScript 5.3.3 strict mode |
| `.eslintrc.js` | ESLint rules |
| `vitest.config.ts` | Vitest 4.0.16 test configuration |

Note: Add `exclude: ['**/tests/e2e/**']` to Vitest config to prevent Playwright
spec collection errors (Vitest/Playwright collision issue).

## Testing

| Type | Files | Tests | Command |
|------|-------|-------|---------|
| Unit/Integration | 13 `.test.tsx` files | 201 tests | `npm test` |
| E2E | 2 Playwright specs | Not in CI | `npm run e2e` |

Frontend test files are co-located with source in `src/`:
- `pages/auth.test.tsx` — 30 tests: Login, Register, ForgotPassword
- `pages/Dashboard.test.tsx` — Dashboard rendering
- `pages/Portfolio.test.tsx` — Portfolio interactions
- `pages/Analysis.test.tsx` — Analysis page
- `pages/Recommendations.test.tsx` — Recommendations view
- `pages/Watchlist.test.tsx` — Watchlist CRUD
- `pages/Alerts.test.tsx` — Alerts management
- `pages/Settings.test.tsx` — Settings form
- `pages/MarketOverview.test.tsx` — Market overview
- `components/cards/EnhancedRecommendationCard.test.tsx`
- `components/cards/PortfolioSummary.test.tsx`
- `components/dashboard/HoldingsSection.test.tsx`
- `components/monitoring/CostMonitor.test.tsx`

**Coverage gaps**: Redux slices (0%), custom hooks (0%), API service layer (0%)

## Technology Stack

| Technology | Version | Notes |
|------------|---------|-------|
| React | 18.2.0 | 14 lazy-loaded pages |
| TypeScript | 5.3.3 | Strict mode, zero @ts-ignore suppressions |
| Vite | 7.3.1 | 18 manual vendor chunks |
| Redux Toolkit | Latest | 6 domain slices with typed hooks |
| Material-UI | 5.14 | Full theming + design tokens |
| Recharts | Latest | Primary charting library |
| Plotly | Latest | Advanced charts |
| Chart.js | Latest | Additional charts |
| Lightweight Charts | Latest | Financial candlestick charts |
| Vitest | 4.0.16 | 13 test files, 201 tests |
| Playwright | 1.40 | 2 E2E specs (not yet in CI) |

**Last Updated**: 2026-03-04
