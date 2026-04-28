# Cluster F — Frontend ↔ Backend API Contract Drift

**Cluster lead:** Worker F
**Member findings (7):** F-12-001, F-12-002, F-01-003, F-01-007, F-01-008, F-01-016, F-01-017
**Source scopes:** `01-backend-api`, `12-frontend`
**Theme (per EXECUTIVE_SUMMARY §4):** API client misaligned with current backend versioning prefix; deprecation middleware misfires on every request because `/api/v1/` is the *current* (not legacy) prefix; one wrong response-field read silently breaks auth. Single coordinated refactor of the API client + versioning middleware + ETag/cache excluded-paths.

---

## 1. Cluster overview

The frontend `apiConfig.endpoints.*` table omits the `/v1` segment that all backend routers use, so 40+ endpoints return 404 in every environment (F-12-001). Compounding this, the login thunk reads `response.data.token` while the backend `Token` schema field is `access_token` wrapped in an `ApiResponse.data` envelope, so even when a route hits, auth silently stores `undefined` (F-12-002). On the backend, `V1DeprecationMiddleware` was authored under the assumption that `/api/v1/` was a legacy prefix to be sunset; the codebase has since standardized on `/api/v1/` as the *current* stable prefix. The middleware therefore fires on every production request, attaching `Sunset`, `Deprecation`, and `Warning` headers and emitting `WARNING V1 API request … V1 is SUNSET` log spam (F-01-003), while its companion `V1_TO_V2_ENDPOINT_MAP` points at non-existent `/api/` (no-version) targets (F-01-007). Adjacent middleware drift: `ETagMiddleware` excludes `/api/health` but not `/api/v1/metrics`, so Prometheus scrape responses get inappropriate ETag headers (F-01-008); the GDPR router is mounted at the bare `/api/v1` prefix risking root-route collisions (F-01-016); and the dead `create_versioned_router()` factory adds confusion (F-01-017).

Single coordinated refactor: pick `/api/v1/` as canonical (it already is), align the frontend client, fix the response-field read, retire the deprecation middleware, and clean up adjacent excluded-paths/router-prefix drift in the same PR so frontend and backend deploy together.

## 2. Member findings

| ID | Severity | File:Line | Title (short) |
|---|---|---|---|
| F-12-001 | critical | `frontend/web/src/config/api.config.ts:19-139` | API base paths omit `/v1` — 40+ endpoints 404 |
| F-12-002 | critical | `frontend/web/src/store/slices/appSlice.ts:68` | Login reads `response.data.token`; backend field is `access_token` |
| F-01-003 | critical | `backend/api/versioning.py:756,820-824` | `V1DeprecationMiddleware` treats current `/api/v1/` as deprecated |
| F-01-007 | high | `backend/api/main.py:289-304` (map in `versioning.py`) | `V1_TO_V2_ENDPOINT_MAP` stale; targets non-existent `/api/` routes |
| F-01-008 | high | `backend/api/main.py:260-277` | `/api/v1/metrics` missing from `ETagMiddleware.excluded_paths` |
| F-01-016 | medium | `backend/api/main.py:328-349` | `gdpr.router` registered at bare `/api/v1` prefix — potential collisions |
| F-01-017 | low | `backend/api/versioning.py:676-713` | `create_versioned_router()` factory + `register_router()` are dead code |

## 3. Sequenced fix steps

All steps fail-first where applicable. Path verification: every file referenced below was confirmed in the source slice. Frontend and backend changes ship in a **single PR** to avoid a window where the deploy ordering breaks production.

### Step 1 — Decide canonical prefix (DEFAULT: keep `/api/v1/`)

**Decision:** Keep `/api/v1/` as the canonical, current prefix. There is no `v2`. The deprecation middleware was authored against an earlier mental model where `v1` would be sunset in favor of unversioned `/api/`; that migration never happened and is no longer planned (per EXECUTIVE_SUMMARY §4).

**Action:** Document this in the PR description. Update `backend/api/versioning.py` module docstring to state: *"`/api/v1/` is the current stable prefix. Future major versions would mount at `/api/v2/`."*

**Fail-first test (Loki should run before any code change):**
```bash
# This currently fails — frontend cannot reach backend
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5173/api/auth/login   # → 404
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5173/api/v1/auth/login # → 405 (POST-only) or 200
```

### Step 2 — Update frontend `apiConfig` paths (F-12-001)

**File:** `frontend/web/src/config/api.config.ts:19-139`

**Action:** Add the `v1/` segment to every `endpoints.*` value. Prefer explicit paths over a Vite-proxy rewrite — explicit paths are greppable and testable. Roughly:

```ts
// before
login: '/api/auth/login',
stocks: '/api/stocks',
// after
login: '/api/v1/auth/login',
stocks: '/api/v1/stocks',
```

Touch every endpoint key (40+); do not silently leave any on `/api/`.

**Fail-first test:** Add a Vitest snapshot test that asserts every value in `apiConfig.endpoints` matches `/^\/api\/v1\//`. The current config fails this test; after the fix it passes.

### Step 3 — Fix login response-field read (F-12-002)

**File:** `frontend/web/src/store/slices/appSlice.ts:68`

**Action:**
```ts
// before
localStorage.setItem('access_token', response.data.token)
// after
const token = response.data?.data?.access_token
if (!token) {
  console.warn('login response missing access_token', response.data)
  throw new Error('Login response did not include access_token')
}
localStorage.setItem('access_token', token)
```

**Fail-first test:** Vitest unit test for the login thunk: mock the axios response as `{ data: { success: true, data: { access_token: 'tok' } } }` and assert `localStorage.getItem('access_token') === 'tok'`. Today's code stores `undefined` and the test fails.

### Step 4 — Remove (or neuter) `V1DeprecationMiddleware` (F-01-003)

**File:** `backend/api/versioning.py:756,820-824`; registration in `backend/api/main.py`.

**Recommended:** *Remove* the middleware registration entirely (delete the `app.add_middleware(V1DeprecationMiddleware, ...)` call in `main.py`). Keep the class definition behind a feature flag for future re-use only if a real `v2` migration is queued; otherwise delete it together with `V1_TO_V2_ENDPOINT_MAP` (Step 5).

**Fail-first test:** Integration test:
```python
def test_v1_routes_emit_no_sunset_headers(client):
    resp = client.get("/api/v1/stocks")
    assert "Sunset" not in resp.headers
    assert "Deprecation" not in resp.headers
    assert resp.headers.get("X-API-Status") != "sunset"
```
Currently fails (all three headers present); passes after removal.

### Step 5 — Delete or fix `V1_TO_V2_ENDPOINT_MAP` and dead factory (F-01-007, F-01-017)

**File:** `backend/api/versioning.py` (map and `create_versioned_router()`/`register_router()`).

**Action:** Since there is no `v2` migration, delete `V1_TO_V2_ENDPOINT_MAP`, `map_v1_endpoint_to_v2()`, the `create_versioned_router()` factory, and `APIVersionManager.register_router()`. Run `grep -rn 'create_versioned_router\|register_router\|V1_TO_V2_ENDPOINT_MAP\|map_v1_endpoint_to_v2' backend/` and confirm only definition sites match before deletion; the slice (F-01-017) confirms zero call sites today.

**Fail-first test:** Replace the existing `map_v1_endpoint_to_v2` unit test (if any) with a `grep`-based CI check that the symbols are absent from `backend/api/`.

### Step 6 — Add `/api/v1/metrics` to ETag exclusions and tighten GDPR prefix (F-01-008, F-01-016)

**Files:** `backend/api/main.py:260-277` (ETag config), `backend/api/main.py:328-349` (gdpr router registration).

**Actions:**
- `ETagMiddleware(excluded_paths=[..., '/api/v1/metrics'])` — append the metrics path; verify `/api/v1/ws/` is genuinely excluded (slice notes WS upgrade requests may bypass middleware entirely).
- Change `app.include_router(gdpr.router, prefix='/api/v1')` to `prefix='/api/v1/gdpr'` and remove any `/gdpr/` segment that becomes redundant inside the router itself. Read `backend/api/routers/gdpr.py` first to confirm route paths and avoid double-prefixing.

**Fail-first tests:**
- `GET /api/v1/metrics` → response has no `ETag` header.
- FastAPI startup emits no duplicate-route warnings.

---

## 4. Files touched

**Frontend (2):**
- `frontend/web/src/config/api.config.ts`
- `frontend/web/src/store/slices/appSlice.ts`

**Backend (2):**
- `backend/api/versioning.py` (delete `V1DeprecationMiddleware`, map, factory, manager method)
- `backend/api/main.py` (remove middleware registration; update ETag excluded_paths; tighten gdpr prefix)

**Tests (new/updated):**
- `frontend/web/src/store/slices/__tests__/appSlice.test.ts` — login thunk token-storage test
- `frontend/web/src/config/__tests__/api.config.test.ts` — `/api/v1/` prefix invariant
- `backend/tests/api/test_versioning.py` — assert no sunset headers on `/api/v1/*`
- `backend/tests/api/test_middleware.py` — assert no `ETag` on `/api/v1/metrics`

## 5. Acceptance tests

- **e2e:** `npm run e2e -- --grep 'stock fetch'` and `--grep 'login'` both pass; `localStorage.getItem('access_token')` is a non-empty string after login submit.
- **Backend logs:** `kubectl logs deploy/api | grep 'V1 SUNSET'` returns zero lines after deploy on `/api/v1/*` traffic (was thousands/min).
- **Headers:** `curl -I /api/v1/stocks` shows no `Sunset`, `Deprecation`, `Warning`, or `X-API-Status: sunset`.
- **Metrics:** `curl -I /api/v1/metrics` shows no `ETag`.
- **Routing:** FastAPI startup log contains no duplicate-route warnings.
- **Dead code:** `grep -rn 'V1_TO_V2_ENDPOINT_MAP\|create_versioned_router' backend/` returns nothing.

## 6. Rollback plan

Single revert of the merge commit restores both halves atomically. Because frontend and backend ship in the same PR, there is no half-rolled-back state. If the cluster is split into multiple commits within the PR, ensure the merge is squashed. Feature-flag fallback is **not** recommended for Step 2 (frontend paths) since both old and new clients cannot coexist against a single backend prefix.

If the deprecation middleware removal proves disruptive (e.g., an external monitor consumed the `Sunset` header as a health signal), the smaller revert is to re-add `app.add_middleware(V1DeprecationMiddleware, ...)` in `main.py` while keeping the frontend and response-field fixes; this restores headers without re-introducing the 404 storm.

## 7. Dependencies

- **AFTER B (auth stable):** Cluster B fixes auth/CORS issues; F-12-002 changes how the access_token is stored, which interacts with B's auth stack. Running F before B risks chasing auth bugs that B has already fixed.
- **AFTER E (tests un-excluded):** Cluster E re-enables the test signal. The fail-first tests in Steps 2, 3, 4, and 6 are only meaningful once the suite actually runs in CI.
- **No dependency** on clusters A, C, D.
- **Coordination requirement:** Frontend and backend artifacts must deploy together (same PR, same release). Do not merge a frontend-only or backend-only commit from this cluster.

## 8. Effort & cost

| Step | Effort (h) | Source |
|---|---|---|
| 1. Decide canonical prefix + doc | 0.5 | (synthesis) |
| 2. Frontend `apiConfig` paths | 4.0 | F-12-001 |
| 3. Login response-field fix | 1.0 | F-12-002 |
| 4. Remove `V1DeprecationMiddleware` | 4.0 | F-01-003 |
| 5. Delete stale map + dead factory | 3.0 | F-01-007 (2.0) + F-01-017 (1.0) |
| 6. ETag exclusions + gdpr prefix | 2.0 | F-01-008 (1.0) + F-01-016 (1.0) |
| **Total** | **14.5h** | |

Roughly 2 engineer-days. One engineer can execute the whole cluster; the work is sequential within the steps but does not benefit from parallelism beyond two reviewers.

## 9. Loki-actionable

**Largely yes.** Six of seven findings are mechanical (path edits, deletions, list additions). The one judgment call is Step 1 — the canonical-prefix decision could in principle go the other way (rename current routes to `/api/v2/` and keep deprecation middleware aimed at `/api/v1/`).

- **Step 1:** `partial` — default = keep `/api/v1/`. Loki should flag this for human confirmation before proceeding, but a sensible default exists.
- **Steps 2, 3, 6:** `full` — pure mechanical edits with clear acceptance tests.
- **Steps 4, 5:** `full` once Step 1 is confirmed; the deletions follow deterministically from "no v2 migration planned."

Overall cluster: `partial` (gated on Step 1 confirmation), then `full` for the remaining work.

## 10. Risks

- **Deploy ordering (highest):** A frontend-only deploy with new `/api/v1/` paths against a backend that still has aggressive deprecation headers will surface `Sunset` headers to end users. A backend-only deploy that removes the middleware while frontend still calls `/api/` will keep returning 404. **Mitigation:** ship as one PR, one release, with both artifacts.
- **External consumers of `Sunset` header:** If any downstream monitor or external client reads the `Sunset` header, removing the middleware may silently change their behavior. **Mitigation:** grep org-wide for `Sunset` header consumption before merge; communicate in release notes.
- **GDPR router prefix change:** Tightening the prefix to `/api/v1/gdpr` may shift route paths if `gdpr.py` already includes `/gdpr/` internally; could 404 GDPR endpoints. **Mitigation:** read `gdpr.py` first; adjust internal routes if needed; add an integration test for one GDPR endpoint.
- **WebSocket exclusion uncertainty (F-01-008 note):** Middleware may never see WS upgrade requests, in which case the `/api/v1/ws/` exclusion is decorative. Low risk — worst case is an inert config entry.
- **Hidden callers of dead factory (F-01-017):** Slice claims zero call sites; verify with a fresh grep at PR time before deletion.

---

**Assertion:** All 7 cluster IDs are referenced — F-12-001 (§2,3.2,8), F-12-002 (§2,3.3,8), F-01-003 (§2,3.4,8), F-01-007 (§2,3.5,8), F-01-008 (§2,3.6,8), F-01-016 (§2,3.6,8), F-01-017 (§2,3.5,8).
