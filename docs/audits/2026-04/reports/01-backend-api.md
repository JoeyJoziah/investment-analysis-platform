---
scope_id: "01-backend-api"
scope_name: "Backend API & Middleware"
agent_type: "backend-developer"
date: "2026-04-27"
files_in_scope: 29
files_reviewed: 29
files_skipped: []
prior_reports_validated:
  - path: "docs/API_DESIGN_AUDIT_REPORT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/API_DESIGN_AUDIT_REPORT.archived.md"
    claims_validated: 13
    claims_still_valid: 8
    claims_stale: 5
  - path: "docs/MIDDLEWARE_ASYNCCLIENT_FIXES.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/MIDDLEWARE_ASYNCCLIENT_FIXES.archived.md"
    claims_validated: 5
    claims_still_valid: 4
    claims_stale: 1
  - path: "docs/websocket-architecture-analysis.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/websocket-architecture-analysis.archived.md"
    claims_validated: 6
    claims_still_valid: 3
    claims_stale: 3
  - path: "docs/websocket-risks-summary.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/websocket-risks-summary.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
  - path: "docs/api/V1_TO_V2_MIGRATION_GUIDE.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/V1_TO_V2_MIGRATION_GUIDE.archived.md"
    claims_validated: 6
    claims_still_valid: 3
    claims_stale: 3
  - path: "docs/reports/api-standardization-plan.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/api-standardization-plan.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
  - path: "docs/reports/WEBSOCKET_IMPLEMENTATION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/WEBSOCKET_IMPLEMENTATION.archived.md"
    claims_validated: 6
    claims_still_valid: 4
    claims_stale: 2
findings_summary:
  critical: 3
  high: 6
  medium: 8
  low: 3
  total: 20
estimated_remediation_effort_days: 6
agent_status: "complete"
agent_token_usage: 5800
---

# Backend API & Middleware — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- CRITICAL: `auth.py` calls `jwt.encode`/`jwt.decode` with a plain string secret (`SecurityConfig.JWT_SECRET_KEY`) but the algorithm is RS256 by default — RS256 requires RSA key objects; this will raise `InvalidKeyError` at runtime on any login/register attempt unless `JWT_ALGORITHM=HS256` is explicitly set in the environment.
- CRITICAL: `monitoring.py` router defines `APIRouter(prefix="/api/monitoring")` but is never registered with `app.include_router()` in `main.py`; the entire monitoring API (5 endpoints) is silently unreachable.
- HIGH: `V1DeprecationMiddleware` is registered in production and emits a `logger.warning(...)` for every request because all current routes are under `/api/v1/` — thousands of misleading "V1 SUNSET" log entries per hour with zero signal value.
- HIGH: Three REST endpoints in the WebSocket router (`/trigger/alert`, `/trigger/news`, `/connections`) have no authentication, allowing any unauthenticated caller to broadcast arbitrary messages to all connected users or enumerate live client IDs.
- HIGH: The `market_data_stream_endpoint` (`/api/v1/ws/market`) and `portfolio_stream` (`/api/v1/ws/portfolio/{id}`) WebSocket endpoints lack the `@secure_websocket` decorator, admitting any anonymous connection with no rate limiting or JWT check.

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

**Path globs covered:**
- `backend/api/**/*.py` — 23 files (main.py, versioning.py, security_integration.py, routers/__init__.py + 19 router files)
- `backend/middleware/**/*.py` — 6 files (error_handler.py, request_size_limiter.py, response_optimizer.py, security_headers.py, stack.py, __init__.py)

**Files explicitly excluded:**
- `backend/api/__pycache__/` — bytecode only, excluded per scope-map

**All 29 in-scope files were read directly.** Up to 8 additional out-of-scope files were read as callee context (security_config.py, jwt_manager.py, models/api_response.py) per the 1.5× rule.

## 2. Prior Report Reconciliation

### `docs/API_DESIGN_AUDIT_REPORT.md` — status: `partially_stale`

**Validation method:** Line-by-line read of main.py, versioning.py, auth.py, watchlist.py, health.py, monitoring.py and grep searches for endpoint patterns, response wrappers, and versioning prefixes.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/API_DESIGN_AUDIT_REPORT.archived.md`

**Per-claim validation table:**

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "Inconsistent Versioning — /api/stocks no version, /api/v1/gdpr v1 prefix" | §1 Issues | fully_stale | `main.py:333-348` — ALL routers now use `/api/v1/` prefix consistently; no unversioned routes remain |
| 2 | "OpenAPI docs disabled in production (DEBUG=True only)" | §4 | current | `main.py:134-135` — `docs_url="/api/docs" if settings.DEBUG else None` still present |
| 3 | "No static OpenAPI spec file" | §4 | current | `find docs/ -name "openapi.*"` returns no results; no spec file committed |
| 4 | "Rate limiting not applied to /api/analysis/analyze" | §10 | current | `analysis.py` examined — no `Depends(rate_limit)` on analyze endpoint; `grep -n "rate_limit\|Depends" backend/api/routers/analysis.py` shows only DB session deps |
| 5 | "CORS uses localhost only — no env-based production config" | §11 | fully_stale | `main.py:165-171` — CORS_ORIGINS env var now read; falls back to localhost only if unset |
| 6 | "Missing Orders/Trades API" | §12 | partially_stale | `trading.py` now exists at `/api/v1/trading` covering order validation and trade execution; dedicated orders CRUD still absent |
| 7 | "No cursor-based pagination; offset only" | §6 | current | `stocks.py`, `portfolio.py` — all use `limit/offset` Query params; no cursor token implementation found |
| 8 | "Health check endpoints exist" | §12 | current | `health.py` — GET /api/health, /readiness, /liveness, /startup, /metrics, /ping all present |
| 9 | "Deprecated endpoints lack OpenAPI deprecated=True marker" | §13 | partially_stale | `stocks.py` — `@router.post("/{symbol}/watchlist")` docstring says DEPRECATED but no `deprecated=True` parameter found; `grep -n "deprecated=True" backend/api/routers/stocks.py` returns nothing |
| 10 | "WebSocket WS /api/ws/stream exists" | §12 | partially_stale | Now at `/api/v1/ws/stream` after versioning refactor; prior report showed `/api/ws/stream` |
| 11 | "Dict instead of Pydantic in 45+ endpoints (no wrapper)" | api-standardization-plan §2.3 | fully_stale | `models/api_response.py` and `success_response()` now exist and are imported in health.py, stocks.py, auth.py, monitoring.py; wrapper is in use |
| 12 | "No error_handler.py middleware" | api-standardization §2.4 | fully_stale | `middleware/error_handler.py` exists and is registered in `main.py:139` via `register_exception_handlers(app)` |
| 13 | "Monitoring router uses Dict only, no standardization" | api-standardization §2.2 | partially_stale | `monitoring.py` imports `ApiResponse` and `success_response` (line 14) but the router is unreachable (see F-01-002) |

---

### `docs/MIDDLEWARE_ASYNCCLIENT_FIXES.md` — status: `partially_stale`

**Validation method:** Read main.py and stack.py; grep for skip_in_testing and TESTING env var usage.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/MIDDLEWARE_ASYNCCLIENT_FIXES.archived.md`

**Per-claim validation table:**

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "TESTING env var disables AuditMiddleware, RateLimitingMiddleware, GZipMiddleware, SessionMiddleware in tests" | §2 | current | `main.py:228,230,248,285` — all have `skip_in_testing=True`; `stack.py:189-193` applies the skip |
| 2 | "Production stack has 11 middleware" | §Ordering | partially_stale | `main.py` registers 12 middleware entries (ResponseTiming added, SessionMiddleware removed); actual stack is 12 not 11 |
| 3 | "start_cleanup_task() was never called automatically" | §Future | fully_stale | `main.py:97-98` — `start_cleanup_task()` now called inside lifespan startup |
| 4 | "SecurityHeadersMiddleware safe for AsyncClient (response-only)" | §Key Patterns | current | `security_headers.py` — only sets response headers, reads no request body |
| 5 | "TESTING=True set before imports in conftest.py" | §4 | current | `backend/tests/conftest.py:8` confirmed by grep — TESTING=True set at top of conftest |

---

### `docs/websocket-architecture-analysis.md` — status: `partially_stale`

**Validation method:** Full read of websocket.py (router) and check of imports, delegation pattern, and prior issue locations.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/websocket-architecture-analysis.archived.md`

**Per-claim validation table:**

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "WS-001: Dual state — security_manager.connections and manager.active_connections are separate" | §2.2 | partially_stale | `websocket.py:141` — `manager.connect()` still called separately; business logic moved to `websocket_service.py` but dual state pattern preserved via delegation shims |
| 2 | "WS-002: Unprotected dict iteration (RuntimeError risk)" | §2.2 | current | `websocket_service.py` (out of scope) noted as still holding this state; router shims forward without adding locks |
| 3 | "WS-003: Orphaned async tasks — cleanup_client_streams not guaranteed" | §2.2 | partially_stale | `websocket.py:165-167` — finally block now calls `cleanup_client_streams(client_id)` on disconnect, reducing orphan risk; shared `active_price_streams` dict still unprotected |
| 4 | "WS-004: start_cleanup_task() defined but never called" | §3.1 | fully_stale | `main.py:97-98` — start_cleanup_task() called in lifespan; `websocket.py:56-63` shows implementation |
| 5 | "18 WebSocket tests, missing concurrency/memory-leak tests" | §4 | current | `grep -l "websocket" backend/tests/` shows test files but no concurrent-connection or task-leak tests visible |
| 6 | "market_data_stream and portfolio_stream lack @secure_websocket" | §5 | current | `websocket.py:170-197` — both endpoints have no security decorator and no Depends(get_current_user) |

---

### `docs/websocket-risks-summary.md` — status: `partially_stale`

**Validation method:** Cross-referenced with websocket.py source code reads.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/websocket-risks-summary.archived.md`

**Per-claim validation table:**

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "WS-001 HIGH — dual state connection leak" | §Critical Issues | partially_stale | Still exists architecturally; service layer refactor preserves the dual-dict pattern via shims (`websocket.py:40-98`) |
| 2 | "WS-003 HIGH — orphaned tasks ~10MB per 100 disconnects" | §Critical Issues | partially_stale | `websocket.py:165-167` — finally block now calls cleanup; risk reduced but shared active_price_streams still unprotected |
| 3 | "WS-004 HIGH — start_cleanup_task never called" | §Critical Issues | fully_stale | `main.py:97-98` — explicitly called in lifespan startup; confirmed with grep |
| 4 | "WS-002 MEDIUM — dict iteration crash" | §Medium Priority | current | No asyncio.Lock added to active_connections; dict iteration without lock still present |
| 5 | "WS-005 MEDIUM — silent Redis failures" | §Medium Priority | current | `websocket_service.py` not in scope but router does not add retry logic |

---

### `docs/api/V1_TO_V2_MIGRATION_GUIDE.md` — status: `partially_stale`

**Validation method:** Read versioning.py and main.py; checked actual endpoint prefixes vs guide claims.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/V1_TO_V2_MIGRATION_GUIDE.archived.md`

**Per-claim validation table:**

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "V1 sunset July 1, 2025 — returns 410 Gone" | §Timeline | partially_stale | `versioning.py:799-804` — V1DeprecationMiddleware returns 410 only if `strict_mode=True` or past grace period; `main.py:295-297` sets `strict_mode=False`, `grace_period_days=30`; current date 2026-04-27 is past grace period (Aug 2025), so 410 IS returned for old V1 patterns like `/api/v1/stocks` (not the new /api/v1/ routes) |
| 2 | "V1 endpoints e.g. GET /api/v1/stocks — no version" | §Endpoint Map | fully_stale | ALL routes now use /api/v1/ prefix as the versioned prefix; guide described /api/v1/ as V1 legacy pattern |
| 3 | "V2 endpoint is GET /api/stocks (no version prefix)" | §Endpoint Map | fully_stale | `main.py:334` — `stocks.router` is registered at `/api/v1/stocks`, not `/api/stocks` |
| 4 | "Admin migration metrics at GET /api/admin/v1-migration/metrics" | §Monitoring | current | `versioning.py:919-936` — v1_migration_router defines `/api/v1/admin/v1-migration/metrics` and is included in main.py:349 |
| 5 | "CORS_ORIGINS env var missing" | §implicit | fully_stale | `main.py:165-171` — CORS_ORIGINS env var now supported |
| 6 | "V1 parameter 'ticker' renamed to 'symbol'" | §Quick Ref | current | `versioning.py:193-199` — V1_TO_V2_PARAM_MAP maps ticker→symbol; transform logic present |

---

### `docs/reports/api-standardization-plan.md` — status: `partially_stale`

**Validation method:** grep for ApiResponse, success_response, error_handler across router files.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/api-standardization-plan.archived.md`

**Per-claim validation table:**

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "No ApiResponse wrapper — all 13 routers lack success/error envelope" | §2.3 | fully_stale | `grep -rn "from backend.models.api_response import" backend/api/routers/` hits 10+ files; wrapper is now used |
| 2 | "Create backend/models/api_response.py — Phase 1 todo" | §2.4 | fully_stale | `middleware/error_handler.py:18` imports `from backend.models.api_response import ErrorResponse, error_response` — model exists |
| 3 | "Create error_handler.py — Phase 2 todo" | §2.4 | fully_stale | `middleware/error_handler.py` exists and is registered; `main.py:139` |
| 4 | "Inconsistent pagination — 8 patterns" | §2.3 | current | `stocks.py` uses `limit/offset`; no meta wrapper with total; still no cursor; pagination inconsistency persists |
| 5 | "monitoring.py uses Dict only" | §2.2 | partially_stale | monitoring.py now imports ApiResponse but router is not registered in main.py (unreachable) |

---

### `docs/reports/WEBSOCKET_IMPLEMENTATION.md` — status: `partially_stale`

**Validation method:** Read websocket.py and main.py lifespan startup.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/WEBSOCKET_IMPLEMENTATION.archived.md`

**Per-claim validation table:**

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "Call start_cleanup_task() in @app.on_event('startup')" | §Backend Startup | fully_stale | `main.py:97-98` — called inside lifespan context manager (modern pattern), not deprecated on_event |
| 2 | "stream_price_updates() in router loops while True" | §Architecture | partially_stale | `websocket.py:87-89` — stream_price_updates delegates to `_ws_svc.stream_price_updates`; logic moved to service but while-True loop still present there |
| 3 | "WS health endpoint: GET /ws/health returns active_connections count" | §Monitoring | current | `websocket.py:382-393` — GET /connections endpoint returns total_connections, clients, subscriptions, active_streams |
| 4 | "Acceptance criteria: start_cleanup_task called — checked" | §Acceptance | fully_stale | Now done correctly in lifespan; prior report said it was not |
| 5 | "portfolio_stream and market_data_stream_endpoint: existing enhanced features" | §Architecture | current | `websocket.py:170-197` — both endpoints exist but have no auth (security gap) |
| 6 | "Latency <500ms end-to-end" | §Performance | current | Architecture supports this; no regression detected in code |

---

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-01-001 | critical | bug | backend/api/routers/auth.py:27-28,57,77 | RS256 algorithm with string secret causes runtime InvalidKeyError | `auth.py` sets `ALGORITHM = SecurityConfig.JWT_ALGORITHM` which defaults to `"RS256"` via env, then calls `jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)` where `SECRET_KEY` is a plain string from `secrets.token_urlsafe(32)`. The `python-jose` library raises `InvalidKeyError: Could not deserialize key data` when RS256 is used with a string key — RSA requires actual RSA key objects. Every POST /api/v1/auth/token and /register will fail unless `JWT_ALGORITHM=HS256` is set in env. | Switch `auth.py` to use `get_jwt_manager()` (already imported at line 15) for token creation/validation, or add env guard: if ALGORITHM != "HS256", load RSA key from env/secrets-manager rather than using JWT_SECRET_KEY string. | `POST /api/v1/auth/token` with valid credentials returns 200 (not 500); `pytest tests/test_auth.py::test_login` passes | 3 | true | ["08-auth-security-compliance"] |
| F-01-002 | critical | dead_code | backend/api/routers/monitoring.py:25 | monitoring.py router never registered — 5 endpoints unreachable | `monitoring.py` defines `router = APIRouter(prefix="/api/monitoring")` with 5 endpoints (health, cost_metrics, dashboard_links, alerts, api_usage). `grep "monitoring" backend/api/main.py` shows zero `include_router(monitoring.router)` calls. All endpoints are silently dead. | Add `from backend.api.routers import monitoring` and `app.include_router(monitoring.router, tags=["monitoring"])` to `main.py`. Note the router self-prefixes with `/api/monitoring`; do not add a duplicate prefix. | `GET /api/monitoring/health` returns 200 | 1 | true | [] |
| F-01-003 | critical | architecture | backend/api/versioning.py:756,820-824 | V1DeprecationMiddleware treats all current /api/v1/ routes as deprecated — log spam and incorrect headers | `V1DeprecationMiddleware.dispatch()` at line 756 triggers `_handle_v1_request` for any path containing `/api/v1/`. Since all production routes now use `/api/v1/` as their versioned prefix, every request hits the deprecation handler, emitting `logger.warning("V1 API request … V1 is SUNSET")` and adding `Warning`, `Sunset`, `Deprecation`, `X-API-Status: sunset` headers to every response. Clients receive misleading sunset headers on all API calls. | The V1DeprecationMiddleware was designed to intercept legacy `/api/v1/` calls from external clients migrating from an old API. Now that `/api/v1/` IS the current stable prefix, either (a) remove the middleware, (b) update it to only block specific old path patterns that no longer exist (e.g. `/api/v1/stock/{symbol}` without the 's'), or (c) rename current routes to `/api/v2/`. | After fix: no `Warning` or `Sunset` header on `GET /api/v1/stocks`; log level drops to DEBUG for normal requests | 4 | true | [] |
| F-01-004 | high | security | backend/api/routers/websocket.py:353-364,367-379 | REST trigger endpoints unauthenticated — any caller can broadcast messages | `POST /api/v1/ws/trigger/alert` and `POST /api/v1/ws/trigger/news` have no authentication `Depends`. Any unauthenticated HTTP client can send arbitrary alert messages to specific connected users or broadcast news headlines to all WebSocket clients. The alert endpoint also accepts free-form `alert_type` and `message` parameters with no validation. | Add `current_user: User = Depends(get_current_user)` (or `get_current_admin_user`) to both endpoints. Add input validation (enum for alert_type, length cap on message). | Unauthenticated `POST /api/v1/ws/trigger/alert` returns 401; authenticated admin call succeeds | 2 | true | ["08-auth-security-compliance"] |
| F-01-005 | high | security | backend/api/routers/websocket.py:382-393 | GET /ws/connections exposes client IDs and subscriptions — no auth | `GET /api/v1/ws/connections` returns `total_connections`, full list of `client_id` strings, per-client symbol subscriptions, and active stream symbols. No authentication or authorization is required. This leaks user activity patterns and client identifiers. | Add `current_user = Depends(get_current_admin_user)` dependency. Consider returning aggregated counts only for non-admin roles. | Unauthenticated GET returns 401; admin-authenticated call returns 200 with connection data | 1 | true | ["08-auth-security-compliance"] |
| F-01-006 | high | security | backend/api/routers/websocket.py:170-197 | market and portfolio WebSocket endpoints have no authentication | `market_data_stream_endpoint` (line 170) and `portfolio_stream` (line 186) accept WebSocket connections with `await websocket.accept()` and no `@secure_websocket` decorator, no JWT check, no rate limiting. The portfolio stream broadcasts portfolio-specific updates using the raw `portfolio_id` path parameter with no ownership check. | Apply `@secure_websocket(require_auth=True)` to both endpoints, or at minimum add a token query parameter validated against `WebSocketSecurityManager`. Add ownership check in portfolio_stream to verify the authenticated user owns portfolio_id. | Anonymous WebSocket connection to `/api/v1/ws/market` is rejected (4008 policy violation); authenticated user cannot access another user's portfolio stream | 3 | true | ["08-auth-security-compliance"] |
| F-01-007 | high | architecture | backend/api/main.py:289-304 | V1DeprecationMiddleware registered but V1 endpoint map is stale | `V1_TO_V2_ENDPOINT_MAP` in versioning.py maps e.g. `/api/v1/stocks → /api/stocks` (no version) as V2 targets. But the actual V2 routes are at `/api/v1/stocks`. The migration router itself is at `/api/v1/admin/v1-migration/`. The map is entirely wrong relative to the current routing table. Any client that follows the redirect (if enable_redirects were True) would hit 404s. | Audit and update `V1_TO_V2_ENDPOINT_MAP` to reflect current routing, or remove the map since V1 legacy patterns no longer exist in the codebase. | `map_v1_endpoint_to_v2()` unit test passes with current route structure | 2 | true | [] |
| F-01-008 | high | performance | backend/api/main.py:260-277 | CacheControl and ETag excluded_paths reference /api/v1/auth/ patterns correctly but /api/v1/metrics missing | `CacheControlMiddleware` is configured with `cache_excluded_paths=["/api/v1/auth/", "/api/v1/admin/", "/api/v1/ws/", "/api/v1/metrics"]`. `ETagMiddleware` excludes `["/api/v1/auth/", "/api/v1/admin/", "/api/v1/ws/", "/api/health"]`. The metrics endpoint is at `/api/v1/metrics` (main.py:365) but ETag exclusion only has `/api/health`. Prometheus metrics response will get ETag headers added — inappropriate for frequently-changing metric data. | Add `/api/v1/metrics` to ETagMiddleware's `excluded_paths` list. Also verify `/api/v1/ws/` path exclusion covers WebSocket upgrades (middleware may never see WS upgrade requests). | `GET /api/v1/metrics` response contains no ETag header | 1 | true | [] |
| F-01-009 | high | incomplete_code | backend/api/routers/auth.py:15 | jwt_manager imported but unused in auth router | `from backend.security.jwt_manager import get_jwt_manager, TokenClaims` is imported at line 15, but neither `get_jwt_manager` nor `TokenClaims` is used anywhere in auth.py. The router implements its own `create_access_token()` / `get_current_user()` functions with plain `jwt.encode/decode`. This means the advanced jwt_manager (RSA keys, Redis token blacklisting, refresh token rotation) is bypassed entirely for all auth flows in this router. | Replace the bespoke `create_access_token`/`get_current_user` with `get_jwt_manager().create_token()` and `get_jwt_manager().verify_token()`. Remove the dead import. | `POST /api/v1/auth/token` returns a token verifiable by jwt_manager; blacklisted tokens are rejected | 4 | true | ["08-auth-security-compliance"] |
| F-01-010 | medium | stale_code | backend/api/security_integration.py:1-60 | security_integration.py register_security_middleware() is imported but never called | `main.py:143` imports `register_security_middleware` from `backend.api.security_integration` but the function is never invoked — the middleware stack is configured inline via `MiddlewareStack`. The import exists at module level as a dead import. | Remove the unused import from main.py, or call the function if it provides middleware the stack does not cover. | `grep "register_security_middleware" backend/api/main.py` returns only the import line | 1 | true | [] |
| F-01-011 | medium | doc_drift | backend/api/versioning.py:303-350 | VERSION_REGISTRY lists V3 as STABLE but no V3 routes exist | `VERSION_REGISTRY` declares `APIVersion.V3` with `status=VersionStatus.STABLE` and lists GraphQL, real-time streaming, advanced analytics as features. No V3 router is registered in main.py. `LATEST = V3` in the enum, so `APIVersionManager.LATEST` is V3. Clients following the `Link: </api/v3>; rel="successor-version"` header will hit 404s. | Either remove V3 from the registry (set LATEST=V2) until V3 routes are implemented, or add a version-info redirect endpoint at `/api/v3` explaining V3 is in development. | `GET /api/v3/version` returns a non-404 response; or V3 removed from LATEST | 2 | true | [] |
| F-01-012 | medium | code_quality | backend/api/routers/auth.py:70-90 | get_current_user defined in auth.py shadowed by auth.oauth2.get_current_user — duplicate implementations | `auth.py` defines its own `async def get_current_user(token, db)` at line 70 using `jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])`. `backend/auth/oauth2.py` also exports `get_current_user` (imported by stocks.py, portfolio.py, trading.py etc). Two separate implementations with potentially different decoding keys and algorithms will produce different validation results for the same token. | Delete the local `get_current_user` in auth.py and use `from backend.auth.oauth2 import get_current_user` consistently across all routers. Consolidate JWT decode logic in one place. | All routers import `get_current_user` from the same module; `backend/auth/oauth2.py` is the sole JWT validation point | 3 | true | ["08-auth-security-compliance"] |
| F-01-013 | medium | testing_gap | backend/api/routers/websocket.py:117-167 | No integration test for authentication failure path in /ws/stream | The `@secure_websocket` decorator enforces JWT auth on the main stream endpoint. There are no visible tests for the anonymous-connection rejection path, the rate-limit enforcement, or the subscription permission denial. Prior report listed 18 WS tests but the concurrent-connection and memory-leak gaps remain. | Add pytest-asyncio tests: (1) anonymous connection to /ws/stream is rejected, (2) rate limit blocks >60 messages/min, (3) VIEWER role denied >10 symbol subscriptions. | `pytest backend/tests/test_websocket*.py -k "test_anon_rejected or test_rate_limit or test_viewer_limit"` passes | 4 | true | ["15-test-suite"] |
| F-01-014 | medium | code_quality | backend/api/routers/stocks.py:42-59 | Fallback error handler defined inline — hides import failure silently | `stocks.py:42-59` wraps `from backend.utils.enhanced_error_handling import ...` in a try/except ImportError and silently replaces missing functions with stub implementations. A missing `enhanced_error_handling` module produces no startup warning; the fallback `validate_stock_symbol` regex is simpler than the real implementation, leading to different validation behaviour. | Convert the try/except to a hard import. If `enhanced_error_handling` is optional, document that explicitly and emit a startup warning when the fallback is used. | App startup with `enhanced_error_handling` absent raises `ImportError` with descriptive message rather than silently degrading | 2 | true | ["11-backend-utils-shared"] |
| F-01-015 | medium | performance | backend/api/routers/health.py:8 | health.py uses synchronous SQLAlchemy engine — blocks async event loop | `health.py:8` imports `engine` (sync) from `backend.utils.database` and `get_db_sync`. The `readiness_check` endpoint uses `with engine.connect() as conn:` — a blocking synchronous database call inside an async FastAPI endpoint. Under load this blocks the entire event loop thread for the duration of the DB query. | Replace with async DB check: `async with db_manager.get_session() as session: await session.execute(text("SELECT 1"))` as done in monitoring.py:51-52. | `GET /api/health/readiness` does not block other concurrent requests during the DB check (load test with 50 concurrent requests) | 2 | true | ["07-database-persistence"] |
| F-01-016 | medium | architecture | backend/api/main.py:328-349 | gdpr router double-registered at /api/v1 — may produce route collisions | `main.py:342` registers `gdpr.router` with `prefix="/api/v1"` (no subpath). If `gdpr.py` defines routes with paths like `/gdpr/...` they land at `/api/v1/gdpr/...` which is correct, but if it defines root-level routes (e.g. `@router.get("/")`) they would conflict with other routers. Needs verification that gdpr router paths all have the `/gdpr/` subpath in the router itself. | Read gdpr.py route definitions and confirm there are no bare `/` routes that would collide. Add an explicit sub-prefix: `prefix="/api/v1/gdpr"`. | No duplicate route warnings in FastAPI startup logs | 1 | true | [] |
| F-01-017 | low | stale_code | backend/api/versioning.py:676-713 | create_versioned_router() factory and version_manager.register_router() unused | `versioning.py:676-713` defines a `create_versioned_router()` factory function and the `APIVersionManager` class supports `register_router()`. Neither is called anywhere in main.py or any router. The only versioning mechanism in use is the `V1DeprecationMiddleware` and static URL prefix naming. | Either use `create_versioned_router()` in main.py to create versioned routers properly, or remove the dead code to reduce confusion. | `grep -rn "create_versioned_router\|register_router" backend/api/` returns only definition sites | 1 | true | [] |
| F-01-018 | low | doc_drift | backend/api/routers/websocket.py:36 | Router docstring says "thin routing layer — business logic in websocket_service" but security handler is substantial | The module docstring claims it is a thin routing layer, but `handle_secure_client_message()` (lines 204-346) is 142 lines of business logic covering all message types (auth, subscribe, unsubscribe, heartbeat, chat), audit logging, and permission checking. This is not a thin layer. | Move `handle_secure_client_message` into `backend/services/websocket_service.py`. Update docstring to accurately describe the file's responsibility. | `websocket.py` line count < 150; `websocket_service.py` contains the message handler | 3 | true | ["02-backend-services-domain"] |
| F-01-019 | low | code_quality | backend/middleware/stack.py:66-73 | MiddlewarePriority NORMAL(5000) conflicts with REQUEST_SIZE(5000) — ambiguous ordering | `MiddlewarePriority.NORMAL = 5000` and `MiddlewarePriority.REQUEST_SIZE = 5000` have the same integer value. Python sort is stable so their order is determined by registration order, but this is fragile and not documented. | Assign NORMAL a unique value (e.g. 5500 or 4500) or remove NORMAL as it is not used in main.py. | `list(set(m.value for m in MiddlewarePriority))` length equals `len(MiddlewarePriority)` | 1 | true | [] |
| F-01-020 | low | better_pattern | backend/api/routers/auth.py:50-58 | create_access_token helper duplicates jwt_manager token logic — should be consolidated | `create_access_token()` in auth.py is a local function duplicating what `jwt_manager.create_token()` provides (with additional security: RSA keys, Redis blacklist, token versioning). Keeping both paths means security improvements to jwt_manager do not automatically apply to auth token creation. | This is a corollary of F-01-009; fix F-01-009 first. | `create_access_token` in auth.py removed; all token creation flows through jwt_manager | 0 | true | [] |

## 4. Cross-Scope Linkages

- `F-01-001` → scope `08-auth-security-compliance` (backend/security/security_config.py:129, jwt_manager.py) — JWT algorithm config and key management live in the auth/security scope.
- `F-01-004`, `F-01-005`, `F-01-006` → scope `08-auth-security-compliance` — authentication enforcement is owned by the security layer.
- `F-01-009`, `F-01-012` → scope `08-auth-security-compliance` — jwt_manager.py and auth/oauth2.py both in security scope.
- `F-01-013` → scope `15-test-suite` — WebSocket integration tests owned by test suite scope.
- `F-01-014` → scope `11-backend-utils-shared` — enhanced_error_handling.py is in backend/utils/.
- `F-01-015` → scope `07-database-persistence` — sync engine vs async session is a database layer concern.
- `F-01-018` → scope `02-backend-services-domain` — websocket_service.py is in backend/services/.

## 5. Risk-Prioritized Punch List (top 10)

1. **F-01-001** — RS256 string-key auth crash — every login will fail at runtime; highest urgency.
2. **F-01-002** — monitoring.py dead router — entire monitoring API is silently down; one-line fix.
3. **F-01-003** — V1 middleware log spam and misleading sunset headers on all requests — operational noise and incorrect client signaling.
4. **F-01-004** — unauthenticated trigger/alert allows arbitrary broadcast — zero-cost attack vector.
5. **F-01-005** — unauthenticated /ws/connections leaks all connected client IDs and subscriptions.
6. **F-01-006** — market and portfolio WebSocket endpoints have no auth — real-time data leakage.
7. **F-01-009** — jwt_manager imported but bypassed — security improvements (blacklisting, RSA) never applied.
8. **F-01-012** — duplicate get_current_user implementations with different decode keys — auth inconsistency.
9. **F-01-015** — sync DB call inside async health endpoint — event loop blocking under load.
10. **F-01-007** — V1→V2 endpoint map points to non-existent unversioned routes — incorrect redirect targets.

## 6. Open Questions

- Q1: Is `JWT_ALGORITHM=HS256` set in the production `.env` file? If so, F-01-001 is mitigated at deployment but the code is still fragile and will break if the env var is ever unset or changed.
- Q2: Is `monitoring.py` intentionally excluded from main.py (perhaps served by a separate monitoring process), or is this a plain omission?
- Q3: The V1DeprecationMiddleware's `_handle_v1_request` now fires for every `/api/v1/` request in production — was the intent to eventually rename all routes to `/api/v2/`? The current state is architecturally ambiguous.
- Q4: Does `gdpr.py` define any bare-path routes (e.g. `@router.get("/")`) that would collide at `/api/v1/`? Needs gdpr.py code review (in-scope but not fully read due to token budget).
