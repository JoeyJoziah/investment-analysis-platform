---
scope_id: "12-frontend"
scope_name: "Frontend (React)"
agent_type: "react-specialist"
date: "2026-04-27"
files_in_scope: 120
files_reviewed: 48
files_skipped:
  - "frontend/web/tests/** — owned by scope 15"
  - "frontend/web/playwright.config.ts — owned by scope 15"
  - "frontend/web/node_modules/ — excluded"
  - "frontend/web/dist/ — excluded"
  - "Low-priority page components not directly related to critical findings (Help.tsx, Reports.tsx, Register.tsx, ForgotPassword.tsx)"
prior_reports_validated:
  - path: "docs/FRONTEND_AUDIT_REPORT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/FRONTEND_AUDIT_REPORT.archived.md"
    claims_validated: 22
    claims_still_valid: 14
    claims_stale: 8
  - path: "docs/TYPESCRIPT_STRICT_MODE_AUDIT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/TYPESCRIPT_STRICT_MODE_AUDIT.archived.md"
    claims_validated: 10
    claims_still_valid: 5
    claims_stale: 5
  - path: "frontend/web/UI_DESIGN_ANALYSIS.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/UI_DESIGN_ANALYSIS.archived.md"
    claims_validated: 14
    claims_still_valid: 8
    claims_stale: 6
findings_summary:
  critical: 3
  high: 7
  medium: 8
  low: 4
  total: 22
estimated_remediation_effort_days: 8
agent_status: "complete"
agent_token_usage: 13800
---

# Frontend (React) — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- **CRITICAL (F-12-001)**: Every API call from `api.config.ts` targets `/api/auth/*`, `/api/stocks/*` etc. but all backend routers are mounted at `/api/v1/*` — 100% of app API traffic yields 404 in any environment that does not have the Vite proxy rewriting paths. The one page that works correctly (`InvestmentThesis.tsx`) hard-codes `/api/v1` directly, confirming the mismatch.
- **CRITICAL (F-12-002)**: `appSlice.ts:68` reads `response.data.token` to store the access token after login, but the backend `Token` schema field is `access_token` (verified in `auth.py:44`); the ApiResponse wrapper nests it as `response.data.data.access_token`. Every login silently stores `undefined` as the token, making all subsequent authenticated API calls fail.
- **HIGH (F-12-005)**: `InvestmentThesis.tsx` bypasses the central API service entirely — it imports raw `axios`, reads `localStorage` directly for the token, and constructs URLs from a locally-defined `API_BASE_URL` constant; this page skips auth interceptors, token refresh, and error normalization.
- **HIGH (F-12-007)**: Three analytics components are fully duplicated — `CorrelationMatrix`, `EfficientFrontier`, and `RiskDecomposition` exist byte-for-byte identically at both `components/*.tsx` and `components/portfolio/*.tsx` (confirmed via `diff`); the top-level copies are unreferenced dead code inflating the bundle.
- **MEDIUM (F-12-012)**: `PortfolioMetrics` interface in `portfolioSlice.ts` does not define `correlationMatrix`, `efficientFrontier`, or `diversificationScore`, yet `Portfolio.tsx:443` and `PortfolioChart.tsx:102-112` access them via `(metrics as any)?.*` — the TypeScript strict-mode flag is set in tsconfig but these casts bypass it, masking schema drift between backend analytics response and frontend type definitions.

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

**Path globs covered:**
- `frontend/web/src/**` — 120 source files (TypeScript/TSX, CSS, config)
- `frontend/web/vite.config.ts`
- `frontend/web/tsconfig.json` (and tsconfig.node.json)
- `frontend/web/package.json`
- `frontend/web/index.html`

**Files explicitly excluded with reason:**
- `frontend/web/tests/**` — owned by scope 15 (test suite)
- `frontend/web/playwright.config.ts` — owned by scope 15
- `frontend/web/node_modules/` — vendor, out of scope
- `frontend/web/dist/` — build output, out of scope

**Wave 3 cross-scope context read:**
- `docs/audits/2026-04/reports/01-backend-api.md` — TL;DR for API contract
- `docs/audits/2026-04/reports/08-auth-security-compliance.md` — F-08-003 CSP unsafe-inline cross-link

## 2. Prior Report Reconciliation

### `docs/FRONTEND_AUDIT_REPORT.md` — status: `partially_stale`

**Validation method:** Direct file reads of `tsconfig.json`, `package.json`, `App.tsx`, `api.service.ts`, `appSlice.ts`, `websocket.service.ts`, `serviceWorkerRegistration.ts`, test file enumeration, grep for token keys, memoization counts.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/FRONTEND_AUDIT_REPORT.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | TypeScript strict mode disabled (`"strict": false`) | §13 | `fully_stale` | `tsconfig.json:19` — `"strict": true` is now enabled; comment says "GitHub Issue #44 — TypeScript Strict Mode Remediation — Status: ✅ STRICT MODE ENABLED" |
| 2 | Only 4 test files exist | §12 | `fully_stale` | `find src -name "*.test.*"` returns 16 test files (Dashboard, Portfolio, Analysis, Alerts, MarketOverview, Settings, Watchlist, Recommendations, auth, slices, hooks, components) |
| 3 | Vite version `^7.3.1` with Terser minification and manual chunks | §4 | `current` | `package.json:117` — `"vite": "^7.3.1"`; `vite.config.ts:39` — `minify: 'terser'`; manual chunks at lines 55-159 |
| 4 | Plotly.js in dependencies (`"plotly.js": "^2.27.1"`) | §1 | `current` | `package.json:32` — `"plotly.js": "^2.27.1"` present; however no `import … from 'plotly'` found in src — package is a dead dependency |
| 5 | Lazy loading all pages with Suspense + skeleton loaders | §3 | `current` | `App.tsx:28-49` — all 12 pages use `lazy(() => import(...))` with `<SuspenseWrapper>` wrapping each route |
| 6 | Inconsistent token naming (`authToken` vs `access_token`) | §7 | `partially_stale` | `authToken` key no longer appears in src grep; `access_token` is consistently used — however `appSlice.ts:68` uses `response.data.token` (wrong field name vs backend) |
| 7 | No Redux DevTools integration configured | §5 | `current` | `store/index.ts` — `configureStore` with no devTools config explicitly set; Redux DevTools works via browser extension but no explicit config |
| 8 | WebSocket service uses Socket.IO with reconnect | §8 | `current` | `websocket.service.ts:22` — `io(wsUrl, { reconnection: true, reconnectionAttempts: this.maxReconnectAttempts })` confirmed |
| 9 | No request deduplication or abort controller | §6 | `current` | `api.service.ts` — no AbortController, no request deduplication logic found |
| 10 | `MetricCard` defined inline in Dashboard | §17 | `fully_stale` | `components/dashboard/MetricCard.tsx` exists as separate extracted file; `Dashboard.tsx` imports it from there |
| 11 | `serviceWorkerRegistration.ts` undefined `PUBLIC_URL` | §15 | `fully_stale` | `serviceWorkerRegistration.ts:19` — `const publicUrl = new URL(process.env.PUBLIC_URL \|\| window.location.origin, window.location.href)` — fallback added |
| 12 | No DOMPurify for XSS sanitization | §19 | `current` | `grep -rn "DOMPurify\|sanitize\|dangerouslySetInnerHTML"` — zero results; no XSS sanitization library present |
| 13 | Error boundary catches chunk loading errors | §14 | `current` | `ErrorBoundary.tsx:61-62` — `isChunkError = error?.message?.includes('Loading chunk') \|\| error?.message?.includes('Failed to fetch')` |
| 14 | Theme hardcoded to `lightTheme` | §3 | `fully_stale` | `App.tsx:152` — `const theme = useMemo(() => createAppTheme(themeMode), [themeMode])` — theme is now dynamic from Redux state |
| 15 | No `@types/lodash` installed | §TS Audit | `fully_stale` | `package.json:98` — `"@types/lodash": "^4.17.24"` in devDependencies |
| 16 | `alerts = []` implicit any in websocket.service.ts | §TS Audit | `current` | `websocket.service.ts:80-84` — `stock:trade` event handler accesses trade data without typed Alert interface; no typed alerts array found — still untyped |
| 17 | 14 vendor chunks in vite.config | §4 | `current` | `vite.config.ts:55-158` — 15 named chunk groups confirmed (react-core, react-router, emotion, mui-system, mui-core, mui-icons, mui-x, redux, d3, recharts, plotly, lightweight-charts, chartjs, date-fns, framer-motion, network, react-utils) |
| 18 | `recharts` used throughout | §1 | `current` | `grep` returns 13 files importing from recharts |
| 19 | `framer-motion` imported | §1 | `current` | `EnhancedRecommendationCard.tsx:31` — `import { motion } from 'framer-motion'` confirmed |
| 20 | No CSRF protection visible in requests | §19 | `current` | `api.service.ts` — no CSRF token header; cross-referenced with F-08-003 (CSP) from scope 08 audit |
| 21 | `console.error` in ErrorBoundary production path | §17 | `current` | `ErrorBoundary.tsx:40` — `console.error('ErrorBoundary caught an error:…')` behind `NODE_ENV === 'production'` check — note: Vite build uses `import.meta.env.MODE`, not `process.env.NODE_ENV`; this check may not work |
| 22 | `react-beautiful-dnd` in dependencies but no import found | §1 | `partially_stale` | `package.json:29,30` — `react-beautiful-dnd` and `@types/react-beautiful-dnd` present; `grep -rn "react-beautiful-dnd" src` — zero imports in source |

---

### `docs/TYPESCRIPT_STRICT_MODE_AUDIT.md` — status: `partially_stale`

**Validation method:** Read `tsconfig.json`, grep for remaining `as any` casts, check `@types/lodash` in `package.json`.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/TYPESCRIPT_STRICT_MODE_AUDIT.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | `"strict": false` in tsconfig.json | §Config | `fully_stale` | `tsconfig.json:19` — `"strict": true` now enabled with comment "Status: ✅ STRICT MODE ENABLED — Completion Date: 2026-02-08" |
| 2 | 31 type errors across 10 files | §ErrorAnalysis | `partially_stale` | Strict mode enabled but 5 `as any` casts remain in source: `Portfolio.tsx:443`, `PortfolioChart.tsx:102,110,111,112`, `recommendationsSlice.ts:163`, `InvestmentThesis.tsx:291` — errors suppressed not fixed |
| 3 | Missing `@types/lodash` | §Phase1 | `fully_stale` | `package.json:98` — `"@types/lodash": "^4.17.24"` installed in devDependencies |
| 4 | `PortfolioMetrics` interface missing `correlationMatrix`, `efficientFrontier`, `diversificationScore` | §Phase2 | `partially_stale` | `portfolioSlice.ts:31-56` — interface still lacks these fields; workaround is `(metrics as any)?.correlationMatrix` in PortfolioChart.tsx:102 |
| 5 | `serviceWorkerRegistration.ts` undefined `PUBLIC_URL` | §Phase1 | `fully_stale` | `serviceWorkerRegistration.ts:19` — `process.env.PUBLIC_URL \|\| window.location.origin` fallback added |
| 6 | Implicit `any[]` for alerts in websocket.service.ts | §Phase1 | `current` | `websocket.service.ts` — no typed `Alert` interface for the `stock:trade` handler; trade data passed to `handleTradeUpdate(data)` without type annotation |
| 7 | `null` vs `undefined` mismatch in Dashboard.tsx | §Phase2 | `partially_stale` | Strict mode enabled but dashboard data access is now via Redux selectors; direct null/undefined mismatch in Dashboard not reproduced in current code |
| 8 | React element null vs undefined in EnhancedRecommendationCard | §Phase3 | `current` | `EnhancedRecommendationCard.tsx:291` — `setTimeHorizon(e.target.value as any)` confirms cast is still needed; icon null issue not directly visible without full tsc run |
| 9 | `noUnusedLocals: false` | §Config | `fully_stale` | `tsconfig.json:20-21` — `"noUnusedLocals": true, "noUnusedParameters": true` now enabled |
| 10 | Total remediation effort ~3.25 hours | §Effort | `fully_stale` | Work was completed (strict mode enabled, types/lodash installed) but `as any` escapes indicate incomplete remediation — remaining effort ~2h |

---

### `frontend/web/UI_DESIGN_ANALYSIS.md` — status: `partially_stale`

**Validation method:** Read `ErrorBoundary.tsx`, `PageSkeleton.tsx`, `utils/accessibility.tsx`, grep for keyboard nav hooks, check virtual scrolling.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/UI_DESIGN_ANALYSIS.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "No loading skeletons or progressive enhancement" | §CurrentState | `fully_stale` | `PageSkeleton.tsx` exists with types: dashboard, portfolio, analysis, list, default; used in `App.tsx:125` for all lazy routes |
| 2 | "Missing error boundaries and recovery mechanisms" | §CurrentState | `fully_stale` | `ErrorBoundary.tsx` fully implemented with chunk error detection, reload, reset, and go-back recovery actions |
| 3 | Accessibility utilities implemented | §1 | `current` | `utils/accessibility.tsx` exists with `useFocusTrap`, `announceToScreenReader`, `useKeyboardNavigation`; `styles/accessibility.css` present |
| 4 | "No virtual scrolling for large datasets" | §CurrentState | `current` | No `react-window`, `react-virtual`, or `@tanstack/react-virtual` in package.json or imports — virtual scrolling still absent |
| 5 | `EnhancedDashboard.tsx` with full ARIA support | §1 | `unverifiable` | No `EnhancedDashboard.tsx` file exists in src — either renamed or never committed |
| 6 | Undo/redo functionality | §2 | `unverifiable` | No history/undo pattern found in any source file |
| 7 | Design tokens via `theme/tokens.ts` | §Usability | `current` | `theme/tokens.ts` exists and is imported in components |
| 8 | Skeleton loading implemented via `useMemoizedAsync` | §Usability | `unverifiable` | `useMemoizedAsync` not found in any source file — the code snippet in the prior was aspirational |
| 9 | `EmptyState` component | §Usability | `unverifiable` | No `EmptyState` component found in src — snippet was aspirational |
| 10 | Focus trap management | §1 | `current` | `utils/accessibility.tsx` — `useFocusTrap` hook exists |
| 11 | Color contrast checking utility | §1 | `current` | `utils/accessibility.tsx` — `checkColorContrast` function present |
| 12 | ARIA live regions | §1 | `current` | `utils/accessibility.tsx` — `announceToScreenReader` creates live region element |
| 13 | Reduced motion support | §1 | `current` | `utils/accessibility.tsx` — `prefersReducedMotion` exported |
| 14 | Screen reader-friendly data presentation | §1 | `partially_stale` | Accessibility utilities exist but adoption is partial — only `EnhancedRecommendationCard.tsx` and `SearchModal` demonstrably use them |

---

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-12-001 | critical | schema_mismatch | frontend/web/src/config/api.config.ts:19-139 | Frontend API base paths omit `/v1` — all 40+ endpoints yield 404 | All `apiConfig.endpoints.*` paths use `/api/auth/login`, `/api/stocks`, etc. Backend registers ALL routers under `/api/v1/` prefix (`main.py:333-348`). The Vite dev proxy rewrites `/api/` to backend but does not insert `/v1/`. Every non-thesis API call is 404 in all envs. | Add `v1` segment to all paths in `api.config.ts` (e.g. `/api/v1/auth/login`) OR configure Vite proxy to rewrite `/api/` to `/api/v1/` at the proxy level. Prefer explicit paths. | Login succeeds and `localStorage.getItem('access_token')` is non-null after submit | 4 | true | ["01-backend-api"] |
| F-12-002 | critical | schema_mismatch | frontend/web/src/store/slices/appSlice.ts:68 | Login thunk reads `response.data.token` but backend field is `access_token` | `appSlice.ts:68` — `localStorage.setItem('access_token', response.data.token)`. Backend `Token` schema at `auth.py:44` defines `access_token: str`. ApiResponse wraps it as `{ success, data: { access_token } }`. Frontend reads `response.data.token` — evaluates to `undefined`. Every login stores `undefined` as the auth token; all subsequent requests send `Bearer undefined`. | Change line 68 to `localStorage.setItem('access_token', response.data.data?.access_token)` — or update the login thunk to destructure `{ data: { access_token } } = response`. Also log a warning if the value is undefined before storing. | `appSlice` login thunk test: mock response with `{ data: { access_token: 'tok' } }` and assert `localStorage.getItem('access_token') === 'tok'` | 1 | true | ["01-backend-api"] |
| F-12-003 | critical | security | frontend/web/src/index.html:1 + vite.config.ts | No CSP meta tag and Vite build uses eval-adjacent features | `index.html` has no Content-Security-Policy meta tag. Vite's React plugin in dev uses `eval` for HMR. Production build uses Terser (safe) but no CSP is set at the HTML layer. Scope 08 audit F-08-003 found backend CSP hardcodes `unsafe-inline unsafe-eval` — the frontend does not add a compensating stricter CSP at the HTML level either. Any XSS in the app has no CSP mitigation. | Add `<meta http-equiv="Content-Security-Policy">` to `index.html` with `script-src 'self'`; use nonces for any inline scripts. Remove `unsafe-eval` from nginx/backend CSP (scope 08 F-08-003 fix). Configure Vite to emit nonce-based CSP header. | Lighthouse security audit shows CSP present; browser DevTools shows no CSP violation on page load | 6 | true | ["08-auth-security-compliance", "13-infra-deployment"] |
| F-12-004 | high | broken_import | frontend/web/src/components/CorrelationMatrix.tsx:1 | Root-level CorrelationMatrix/EfficientFrontier/RiskDecomposition are dead duplicate code | `components/CorrelationMatrix.tsx`, `components/EfficientFrontier.tsx`, `components/RiskDecomposition.tsx` are byte-for-byte identical to `components/portfolio/CorrelationMatrix.tsx` etc. (confirmed via `diff`). Only the `portfolio/` versions are imported (from `PortfolioChart.tsx:14-16`). Root-level copies are dead code bundled into the vendor chunk increasing build output. | Delete `src/components/CorrelationMatrix.tsx`, `src/components/EfficientFrontier.tsx`, `src/components/RiskDecomposition.tsx`. Run `npm run build:typecheck` to confirm no broken imports. | `vite build` completes with no import errors; bundle size decreases by ~3 deduplicated chunks | 0.5 | true | [] |
| F-12-005 | high | architecture | frontend/web/src/pages/InvestmentThesis.tsx:30,53,87,137 | InvestmentThesis bypasses central API service — raw axios, hardcoded URL, manual token | `InvestmentThesis.tsx:30` imports `axios` directly. Line 53 defines `const API_BASE_URL = import.meta.env.VITE_API_URL \|\| 'http://localhost:8000/api/v1'`. Lines 87 and 137 call `localStorage.getItem('access_token')` and manually set Authorization header. This bypasses: auth interceptor, token refresh (401 handling), error normalization, timeout config, and centralized baseURL. | Refactor `InvestmentThesis.tsx` to use `api` from `services/api.service.ts`. Add thesis endpoints to `api.config.ts` (e.g. `thesis: { byStock: (id) => /api/v1/thesis/stock/${id}` }). Remove raw axios import and localStorage calls. | All thesis CRUD operations work in integration; token refresh on 401 is exercised automatically | 3 | true | [] |
| F-12-006 | high | schema_mismatch | frontend/web/src/store/slices/portfolioSlice.ts:31-56 + frontend/web/src/components/portfolio/PortfolioChart.tsx:102-112 | PortfolioMetrics interface missing analytics fields — escaped with `as any` | `PortfolioMetrics` interface does not declare `correlationMatrix`, `efficientFrontier`, or `diversificationScore`. `PortfolioChart.tsx:102` uses `(metrics as any)?.correlationMatrix`, line 110 `(metrics as any)?.efficientFrontier?.points`, line 112 `(metrics as any)?.efficientFrontier?.optimalPosition`. `Portfolio.tsx:443` uses `(metrics as any)?.diversificationScore`. These `as any` casts silently hide schema drift if the backend changes field names. | Extend `PortfolioMetrics` in `portfolioSlice.ts` with optional typed fields: `correlationMatrix?: Record<string, number>; efficientFrontier?: { points: {risk: number; return: number}[]; currentPosition: {risk: number; return: number}; optimalPosition?: {risk: number; return: number} }; diversificationScore?: number`. Remove all `as any` casts. | `tsc --noEmit` passes with no errors in Portfolio.tsx and PortfolioChart.tsx | 2 | true | ["09-analytics"] |
| F-12-007 | high | dead_code | frontend/web/package.json:29-30,32 | Three dead production dependencies: `react-beautiful-dnd`, `plotly.js`, `react-plotly.js` | `grep -rn "react-beautiful-dnd" src` — 0 results. `grep -rn "from 'plotly" src` — 0 results. `grep -rn "from 'react-plotly" src` — 0 results. Yet `package.json` lists all three as production `dependencies`. Plotly alone is ~2.8MB minified; it inflates `npm install` time and may be included in vendor chunks if the manualChunks guard is encountered unexpectedly. | Remove `react-beautiful-dnd`, `@types/react-beautiful-dnd`, `plotly.js`, `react-plotly.js`, `@types/react-plotly.js` from package.json. Run `npm install` then `npm run build` to verify no broken imports. | `npm run build` completes; `du -sh dist/assets/plotly*` returns no files | 0.5 | true | [] |
| F-12-008 | high | security | frontend/web/src/pages/InvestmentThesis.tsx:169 | Catch-block uses `err: any` and leaks raw API error detail to UI | `InvestmentThesis.tsx:169` — `catch (err: any) { setError(err.response?.data?.detail \|\| 'Failed…') }`. Displaying `err.response?.data?.detail` verbatim can leak internal server error messages, stack traces, or database field names to the user. This is an XSS-adjacent pattern if the backend ever returns user-controlled content in `detail`. | Replace `err: any` with `err: unknown`. Use a typed error handler: `const message = isAxiosError(err) && err.response?.status !== 500 ? err.response?.data?.detail : 'An error occurred'`. Never display 500-level detail to users. | Unit test: mock a 500 response with `detail: 'SQL error: ...'` and assert the displayed message is the generic fallback string | 1 | true | [] |
| F-12-009 | high | architecture | frontend/web/src/services/websocket.service.ts:144 | WebSocket subscribes to `state.portfolio.watchlist?.items` — property path not in PortfolioState | `websocket.service.ts:144` — `state.portfolio.watchlist?.items`. Looking at `portfolioSlice.ts`, the state structure contains `watchlists: Watchlist[]` and `selectedWatchlist: Watchlist | null` but no `.watchlist.items` direct path. This access silently returns `undefined`, so watchlist stock subscriptions are never sent to the server — no real-time price updates for watchlisted items. | Fix the state access to match the actual Redux state shape: `state.portfolio.selectedWatchlist?.items \|\| []` or iterate `state.portfolio.watchlists`. Add an integration test for `subscribeToChannels`. | Manual test: add a stock to watchlist, observe WebSocket `subscribe` emit in browser DevTools Network | 2 | true | [] |
| F-12-010 | high | security | frontend/web/src/components/common/ErrorBoundary.tsx:38 | ErrorBoundary uses `process.env.NODE_ENV` — unreliable in Vite production | `ErrorBoundary.tsx:38` — `if (process.env.NODE_ENV === 'production')`. Vite replaces `import.meta.env.MODE` not `process.env.NODE_ENV`. In a Vite build, `process.env.NODE_ENV` may or may not be injected depending on the plugin config. If it evaluates falsy in production, the error detail block renders for all users, leaking stack traces. | Replace `process.env.NODE_ENV === 'production'` with `import.meta.env.PROD` (Vite's canonical boolean). Also replace the duplicate `process.env.NODE_ENV !== 'production'` guard at line 99. | `npm run build` and serve; trigger an error in the built app; confirm error detail block is NOT rendered | 0.5 | true | [] |
| F-12-011 | medium | code_quality | frontend/web/src/store/slices/appSlice.ts:68 | Login thunk stores `response.data.token` but refresh interceptor stores `response.data.access_token` | Login path: `appSlice.ts:68` sets `localStorage.setItem('access_token', response.data.token)`. Refresh path in `api.service.ts:57-58`: `const { access_token } = response.data; localStorage.setItem('access_token', access_token)`. Even if fixed for the backend response structure, these two code paths use different destructuring patterns and must remain in sync. | Centralise token persistence in a `tokenStorage.ts` utility (`setAccessToken`, `getAccessToken`, `clearTokens`) used by both `appSlice.ts` and `api.service.ts`. This eliminates the sync problem. | Unit tests for `tokenStorage` module; both login and refresh paths call the same setter | 2 | true | [] |
| F-12-012 | medium | incomplete_code | frontend/web/src/services/websocket.service.ts:80-84 | `stock:trade` event handler delegates to unimplemented `handleTradeUpdate` | `websocket.service.ts:81-84` — `this.socket.on('stock:trade', (data) => { this.handleTradeUpdate(data); })`. No `handleTradeUpdate` method is defined in the class. TypeScript strict mode does not catch this because it appears to be callable but `this.handleTradeUpdate` will throw `TypeError: this.handleTradeUpdate is not a function` at runtime on every trade event. | Implement `handleTradeUpdate(data: unknown): void` or replace with an inline dispatch. Add a `TradeUpdate` interface. | `websocket.service.ts` compiles without error; unit test: mock socket, emit `stock:trade`, verify no TypeError | 1 | true | [] |
| F-12-013 | medium | performance | frontend/web/src/App.tsx:96-103 | Route prefetch `setTimeout` leaks memory if component unmounts before timeout fires | `App.tsx:96-103` — `const timeoutId = setTimeout(() => { routesToPrefetch.forEach(prefetchRoute); }, 1000); return () => clearTimeout(timeoutId)`. The cleanup is correct. However `prefetchRoute` calls `routeModules[route]()` which triggers a dynamic import — if the import resolves after the component has unmounted, React will log a state update on unmounted component warning. The imported modules are side-effect free so this is low risk, but the pattern is not idiomatic. | The existing `clearTimeout` cleanup is correct. No urgent fix needed. Consider replacing with `requestIdleCallback` for non-essential prefetching to avoid competing with visible page resources. | No "Can't perform state update on unmounted component" warnings in console after rapid navigation | 1 | false | [] |
| F-12-014 | medium | testing_gap | frontend/web/src | No test coverage for `api.service.ts`, `websocket.service.ts`, or `api.config.ts` — core integration layer | `find src -name "*.test.*"` — no test file for `services/api.service.ts`, `services/websocket.service.ts`, `config/api.config.ts`. These files implement the auth interceptor, token refresh, WebSocket reconnection — the most business-critical client logic. With the schema mismatches found (F-12-001, F-12-002), tests would have caught these issues. | Add vitest unit tests for: (1) `api.service.ts` interceptor behavior (401→refresh→retry); (2) `websocket.service.ts` connection/reconnect lifecycle; (3) `api.config.ts` URL construction helpers. | `npm run test:coverage` shows api.service.ts and websocket.service.ts at >80% branch coverage | 8 | true | ["15-test-suite"] |
| F-12-015 | medium | testing_gap | frontend/web/src/store/slices/appSlice.ts | Login thunk `access_token` extraction not unit-tested | `slices.test.ts` tests `login.fulfilled` (line 219) but does not verify `localStorage.setItem` was called with the correct token value. The F-12-002 bug would not be caught by the existing test. | Add assertion: `expect(localStorageMock.setItem).toHaveBeenCalledWith('access_token', 'expected-token')` in the login thunk test. | Login test fails before fix and passes after fix | 0.5 | true | ["15-test-suite"] |
| F-12-016 | medium | architecture | frontend/web/src/pages/InvestmentThesis.tsx:53 | `InvestmentThesis` hard-codes `/api/v1` base — will break if API prefix changes | `InvestmentThesis.tsx:53` — `const API_BASE_URL = import.meta.env.VITE_API_URL \|\| 'http://localhost:8000/api/v1'`. When F-12-001 is fixed by updating `api.config.ts`, this file will still target `/api/v1` while the rest of the app moves to a new path, creating divergence again. | Subsumed by F-12-005 fix (migrating to central api service). Tracking separately because F-12-005 and F-12-001 may be fixed in separate PRs. | `InvestmentThesis` does not define `API_BASE_URL` locally | 0 | true | [] |
| F-12-017 | medium | stale_code | frontend/web/src/serviceWorkerRegistration.ts:1 | Service worker file registered and exported but never called — dead runtime code | `serviceWorkerRegistration.ts` exports `register()` and `unregister()`. `index.tsx` imports `reportWebVitals` but does NOT import or call `serviceWorkerRegistration.register()`. The file is never invoked; no service worker is active. The prior audit noted "Missing PWA service worker registration" as a gap. | Either remove the file if PWA is not a goal, or call `register()` in `index.tsx` after deciding on a caching strategy. Leaving the unreferenced file is misleading. | If removed: `npm run build:typecheck` passes. If activated: Lighthouse PWA score improves. | 0.5 | true | [] |
| F-12-018 | medium | better_pattern | frontend/web/src/store | No RTK Query — all data fetching uses hand-rolled thunks without request deduplication or cache | `store/slices/dashboardSlice.ts`, `portfolioSlice.ts`, `marketSlice.ts` all use `createAsyncThunk` for data fetching with manual loading/error state. No cache TTL, no deduplication, no automatic refetch on focus. With 40+ endpoints, the boilerplate volume is high and inconsistent (some slices have `isLoading`, others use `status`). | Evaluate migrating to RTK Query for server state; keep slice reducers only for client/UI state. This would eliminate ~60% of thunk boilerplate and add automatic caching, deduplication, and refetch. | RTK Query baseApi established; at least 2 feature APIs migrated; bundle size neutral or better | 12 | false | [] |
| F-12-019 | low | code_quality | frontend/web/src/store/slices/recommendationsSlice.ts:163 | `state.sortBy = action.payload.sortBy as any` — type escape in reducer | `recommendationsSlice.ts:163` — `state.sortBy = action.payload.sortBy as any`. This cast bypasses Immer's draft type checking and allows any value to be assigned to `sortBy`. | Define a `SortBy` union type and type the payload properly: `action: PayloadAction<{ sortBy: SortBy }>`. Remove `as any`. | `tsc --noEmit` passes without `as any` in recommendationsSlice | 0.5 | true | [] |
| F-12-020 | low | doc_drift | frontend/web/src/design/PORTFOLIO_DASHBOARD_DESIGN.md | Design doc in `src/design/` belongs in `docs/` not in the source tree | `src/design/PORTFOLIO_DASHBOARD_DESIGN.md` is a Markdown design document inside the TypeScript source directory. It will be included in `tsc` file indexing and may confuse documentation tools. | Move to `docs/design/PORTFOLIO_DASHBOARD_DESIGN.md` or delete if superseded. | `find src -name "*.md"` returns no results | 0.25 | true | [] |
| F-12-021 | low | performance | frontend/web/src | No `React.memo` on any page-level component — all pages re-render on any store change | `grep -rn "React.memo\|export default memo" src/pages/` — 0 results. Pages re-render whenever any Redux state changes because they are not wrapped in `memo()`. Layout component triggers all child page renders on sidebar toggle, search open, or notification add. | Wrap page components in `memo()` where `props` are stable. Identify selectors in pages that pull large slices (e.g. `state.portfolio`) and use `createSelector` memoized selectors. | React DevTools profiler shows page components do not re-render on sidebar toggle | 4 | true | [] |
| F-12-022 | low | better_pattern | frontend/web/src/components/SearchModal/index.tsx:67,105 | `recentSearches` persisted to localStorage without TTL or size limit | `SearchModal:67` — `localStorage.getItem('recentSearches')`, `SearchModal:105` — `localStorage.setItem('recentSearches', JSON.stringify(updated))`. No max age, no cap on number of entries (no limit on the `updated` array). Over time this can grow large. | Add a max of 10 entries (slice to 10 before storing) and optionally a `savedAt` timestamp to expire old entries after 30 days. | Unit test: add 15 searches; assert stored array length is ≤ 10 | 0.5 | true | [] |

## 4. Cross-Scope Linkages

- **F-12-001** → `01-backend-api` — All backend API routers registered at `/api/v1/` prefix (confirmed `main.py:333-348`). Frontend `api.config.ts` must match exactly. If backend versioning strategy changes (adding `/api/v2/` routes), frontend must be updated in lockstep.
- **F-12-002** → `01-backend-api` — Backend `Token` model (`auth.py:44`) defines `access_token: str`, wrapped in `ApiResponse`. Frontend reads `response.data.token`. Fix requires knowing the exact shape of `ApiResponse<Token>`.
- **F-12-003** → `08-auth-security-compliance` (F-08-003), `13-infra-deployment` — Backend CSP header (`security_config.py:85-92`) uses `unsafe-inline unsafe-eval`. Frontend needs to coordinate with nginx/backend CSP so that the stricter policy does not break Vite's React plugin in dev. Production build is eval-free (Terser), so a strict CSP is achievable in prod.
- **F-12-006** → `09-analytics` — `PortfolioMetrics` extended fields (`correlationMatrix`, `efficientFrontier`) presumably come from backend analytics endpoints. The analytics scope should define the canonical response schema; the frontend type should be derived from it.
- **F-12-009** → no cross-scope root cause, but the fix may reveal that backend WebSocket emits `portfolio:position:update` events keyed to a user's portfolio positions, not watchlist items — worth confirming with `01-backend-api` WebSocket event schema.
- **F-12-014**, **F-12-015** → `15-test-suite` — Frontend test coverage gaps will need to be addressed in scope 15 (test suite). The vitest config (`vite.config.ts:194-204`) excludes `tests/e2e/**` correctly and is ready for expanded unit tests.

## 5. Risk-Prioritized Punch List (top 10)

1. **F-12-001** — API path mismatch (`/api/` vs `/api/v1/`) — This makes the entire app non-functional in production without the Vite dev proxy. Highest-risk finding.
2. **F-12-002** — Login token field mismatch (`response.data.token` vs `response.data.data?.access_token`) — Every authenticated session is broken even if F-12-001 is fixed. Single-line fix.
3. **F-12-003** — No CSP in HTML — Directly cross-links to F-08-003 from scope 08; XSS has no mitigation layer. Requires coordination with infra/backend.
4. **F-12-012** — `handleTradeUpdate` not defined — Runtime `TypeError` on every trade WebSocket event; silently swallowed.
5. **F-12-009** — WebSocket watchlist subscription uses wrong Redux state path — Watchlist real-time prices never subscribe; all watchlist price cards show stale data.
6. **F-12-005** — InvestmentThesis raw axios bypass — Auth interceptor and token refresh skipped; manual token retrieval could break if storage key changes.
7. **F-12-010** — `process.env.NODE_ENV` in Vite context — Error stack traces may leak to production users.
8. **F-12-007** — Dead production dependencies (plotly.js, react-beautiful-dnd) — Unnecessary `npm install` weight; plotly manualChunk guard in vite.config could produce a 2.8MB chunk if ever accidentally imported.
9. **F-12-004** — Duplicate analytics components — Dead code increasing maintainability surface.
10. **F-12-006** — `PortfolioMetrics` type gaps with `as any` — Masks any backend schema drift for 3 analytics fields.

## 6. Open Questions

- Q1: Does the production nginx/Caddy config rewrite `/api/` to `/api/v1/` at the reverse proxy layer? If yes, F-12-001 may not cause failures in production behind the proxy — but it still causes failures in `npm run preview` and in CI integration tests that call the real backend.
- Q2: What is the intended API path structure going forward? The `versioning.py` migration map shows `/api/v1/` as the "old" paths mapping to `/api/` as "new" — but no `/api/` routers are registered. Is a v2 registration planned, or should the frontend simply target `/api/v1/`?
- Q3: Does the backend WebSocket layer emit `stock:trade` events? If yes, the `handleTradeUpdate` missing method (F-12-012) is an active production crash. If not, the handler is dead but inert.
- Q4: Is `react-beautiful-dnd` planned for a future drag-and-drop feature (e.g. watchlist ordering)? If so, it can stay in devDependencies but should not be in production `dependencies`.
