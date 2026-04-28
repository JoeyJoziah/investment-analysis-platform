# Workstream B: jwt-auth

## 1. Cluster overview

**Problem.** Authentication is broken in production by a stack of compounding defects. `backend/api/routers/auth.py` creates tokens with `jwt.encode(payload, SECRET_KEY, algorithm="RS256")` where `SECRET_KEY` is a plain string from `secrets.token_urlsafe(32)` — `python-jose` raises `InvalidKeyError` because RS256 needs an RSA key object. The crash is currently *masked* by `security_config.py`'s ephemeral fallback (`os.getenv(..., secrets.token_urlsafe(32))`) which produces a per-process random key, so multi-worker deployments see N distinct secrets and HS256 fallback paths in `oauth2.py` "succeed" with tokens no other worker can verify. On top of that, `jwt_manager.revoke_token()` raises a `TypeError` (naive vs aware datetime) on every call and the error is swallowed by a bare `except`, so logout never blacklists. The full canonical solution — `backend/security/jwt_manager.py` with RSA keys, Redis blacklist, and refresh rotation — is *imported but unused* in `auth.py` (F-01-009).

**Root cause.** Two parallel JWT implementations (`auth.py` bespoke vs `jwt_manager.py` canonical) drifted; neither path is actually exercised end-to-end in tests because all JWT fixtures are `MagicMock` or instantiate `JWTManager(secret_key="test_...")` with HS256 strings (F-15-006, F-15-007).

**Scope.** All login/logout/refresh flows, every `Depends(get_current_user)` consumer (stocks, portfolio, trading, ws), Socket.IO + WebSocket auth, ML API auth, and the test suite that should have caught all of this.

**Sequencing constraint.** Cluster B blocks on Cluster A: secret rotation must complete and `JWT_PRIVATE_KEY` / `JWT_PUBLIC_KEY` must be present in env before we remove the ephemeral fallback. Removing the fallback first without keys present breaks login worse than it is now.

## 2. Member findings

All 25 assigned IDs:

| ID | Disposition |
|---|---|
| F-01-001 | **Primary.** RS256 + string secret crash. Resolved by Step 4 (consolidate on `jwt_manager`). |
| F-01-004 | **Primary.** Unauth REST trigger endpoints. Step 7. |
| F-01-005 | **Primary.** `/ws/connections` no auth. Step 7. |
| F-01-006 | **Primary.** market/portfolio WS no auth + no ownership check. Step 7. |
| F-01-009 | **Primary.** `jwt_manager` imported unused — *this is the canonical fix lever*. Step 4. |
| F-01-012 | **Primary.** Duplicate `get_current_user` in `auth.py` vs `oauth2.py`. Step 5. |
| F-01-013 | **Primary.** No WS auth-failure integration tests. Step 9. |
| F-01-020 | **Absorbed into F-01-009** — same root cause (local `create_access_token` duplicates jwt_manager); deletion happens in Step 4. |
| F-02-011 | **Primary.** Socket.IO `cors_allowed_origins="*"` + no handshake auth + no portfolio ownership check. Step 8. |
| F-03-001 | **Primary.** ML API mutating endpoints unauthenticated, `0.0.0.0` bind, `allow_origins=["*"]`. Step 7. |
| F-03-014 | **Primary.** No ML API integration tests. Step 9. |
| F-03-017 | **Primary (low).** MD5 → SHA256 in `feature_store._generate_cache_key`. Step 10 (cleanup). |
| F-08-002 | **Primary, root cause.** Ephemeral `JWT_SECRET_KEY` fallback. **Step 1** — fix first to surface F-01-001. |
| F-08-004 | **Primary.** Naive/aware datetime in `revoke_token`. Step 2. |
| F-08-005 | **Primary.** HS256 fallback in `oauth2.py`. Step 3. |
| F-08-008 | **Primary.** Redis blacklist fail-open. Step 6. |
| F-08-013 | **Primary.** CSRF exempts `/login` and `/register`. Step 7. |
| F-08-015 | **Primary.** MFA secrets keyed by username. Step 10. |
| F-08-017 | **Primary.** `jwt.decode` w/o verification before revoke; cap TTL. Step 2 (folded with F-08-004). |
| F-11-001 | **Primary.** Two competing exception hierarchies. Step 11 (sequenced last in cluster, touches auth-error mapping). |
| F-11-016 | **Primary.** `_user_to_dict` getattr → `UserPublic` Pydantic. Step 5 (folded). |
| F-12-011 | **Primary.** Frontend `response.data.token` vs `access_token` mismatch. Step 12. |
| F-12-015 | **Primary.** Missing unit test for token persistence. Step 12 (folded). |
| F-15-006 | **Primary.** JWT fixture is MagicMock. Step 9. |
| F-15-007 | **Primary.** `JWTManager(secret_key="test_...")` HS256 string fixture. Step 9 (folded). |

No duplicates discarded; F-01-020 absorbed into F-01-009 as the recommendation explicitly says "fix F-01-009 first" — same edit.

## 3. Sequenced fix steps

> **Order rationale.** F-08-002 first removes the mask hiding F-01-001's crash. F-08-004 next so revoke works before we wire it into login. F-08-005 removes the HS256 escape hatch so the *only* path is the canonical one. Then F-01-009/F-01-012 consolidate auth on `jwt_manager`. Then unauthenticated endpoints, Redis fail-secure, tests, frontend, exception unification, MD5/MFA cleanup.

**Step 1: Remove ephemeral SESSION/JWT_SECRET_KEY fallback (F-08-002)**
- Files: `backend/security/security_config.py:71`, `backend/security/security_config.py:152`
- Action: Replace `os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))` with a function that raises `RuntimeError("JWT_SECRET_KEY required in production")` when `ENVIRONMENT == "production"` and falls back to a *fixed sentinel only in tests*. Do the same for `SESSION_SECRET_KEY`. Move the default out of the class-level expression so it isn't evaluated at import time.
- Fail-first test (currently-broken): `tests/security/test_security_config.py::test_missing_jwt_secret_in_production_raises` — set `ENVIRONMENT=production`, unset `JWT_SECRET_KEY`, import `security_config` and assert it raises. Today this silently generates a random key and the test fails.
- Pass-after test: same test passes; second test asserts that two imports in the same process produce the *same* key (no per-process randomness).
- Path verified: yes (`backend/security/security_config.py` exists)

**Step 2: Fix revoke_token datetime bug + TTL cap (F-08-004, F-08-017)**
- Files: `backend/security/jwt_manager.py:347`, `backend/security/jwt_manager.py:352`
- Action: Change `datetime.fromtimestamp(exp)` → `datetime.fromtimestamp(exp, tz=timezone.utc)`. Replace bare `except:` with `except (jwt.JWTError, TypeError, ValueError) as e: logger.warning(...)`. Verify the JWT signature *before* extracting `exp` (use `jwt.decode(token, key, algorithms=...)` not `jwt.get_unverified_claims`). Cap derived blacklist TTL at `access_token_expire_minutes` regardless of claimed `exp`.
- Fail-first test (currently-broken): `tests/security/test_jwt_manager.py::test_revoke_blacklists_token` — call `jwt_mgr.revoke_token(valid_token)`, then assert `jwt_mgr._is_token_blacklisted(jti)` is True. Today this returns False because `revoke_token` silently TypeErrors.
- Pass-after test: same test passes; add `test_revoke_caps_ttl_on_forged_exp` (forge `exp=2099`, assert blacklist key TTL ≤ access_token_expire seconds).
- Path verified: yes

**Step 3: Remove HS256 fallback in oauth2 (F-08-005)**
- Files: `backend/auth/oauth2.py:89-96`, `backend/auth/oauth2.py:119-127`
- Action: Delete the `try: jwt_manager / except: HS256 fallback` blocks. If `get_jwt_manager()` raises, propagate as HTTP 500 (auth subsystem unavailable). Confirmed dependency: only safe to do *after* Step 1 ensures real keys are loaded.
- Fail-first test: `tests/auth/test_oauth2.py::test_no_hs256_fallback` — patch `get_jwt_manager` to raise; assert the dependency raises HTTPException(500), not a successful HS256 decode. Today this currently produces a token under HS256.
- Pass-after test: same passes; `grep -rn "HS256" backend/auth backend/api` returns 0 hits outside config + test files.
- Path verified: yes

**Step 4: Consolidate auth.py on jwt_manager (F-01-001, F-01-009, F-01-020)**
- Files: `backend/api/routers/auth.py:15`, `backend/api/routers/auth.py:27-28`, `backend/api/routers/auth.py:50-58`, `backend/api/routers/auth.py:57`, `backend/api/routers/auth.py:77`
- Action: Delete local `SECRET_KEY`, `ALGORITHM`, `create_access_token()`. In `/token` and `/register` handlers, call `get_jwt_manager().create_token(user_id=..., claims=TokenClaims(...))`. Return `{"access_token": token, "token_type": "bearer", "refresh_token": refresh}`.
- Fail-first test (currently-broken): `tests/api/test_auth_router.py::test_login_returns_200_not_500` — `client.post("/api/v1/auth/token", data={"username": ..., "password": ...})`, assert status_code == 200. Today this raises `InvalidKeyError` → 500 (and only after Step 1 unmasks it).
- Pass-after test: same passes; new test `test_login_token_verifiable_by_jwt_manager` decodes the returned token via `get_jwt_manager().verify_token()`.
- Path verified: yes

**Step 5: Unify get_current_user + UserPublic Pydantic (F-01-012, F-11-016)**
- Files: `backend/api/routers/auth.py:70-90` (delete local), `backend/utils/auth.py:39-60` (refactor `_user_to_dict`)
- Action: Delete `get_current_user` from `auth.py`. Add `UserPublic(BaseModel, model_config={"from_attributes": True})` in `backend/api/schemas/user.py` (or extend existing). Replace `_user_to_dict` with `UserPublic.model_validate(user).model_dump()`. Update all routers that used the auth.py copy (verify with `grep -rn "from backend.api.routers.auth import get_current_user"` — should be 0 after).
- Fail-first test: not currently-broken (silent drift); regression test `test_single_get_current_user_source` asserts `grep` finds only one definition.
- Pass-after test: full router test suite still passes; `UserPublic` schema test asserts only public fields are present.
- Path verified: yes

**Step 6: Redis blacklist fail-secure (F-08-008)**
- Files: `backend/security/jwt_manager.py:88-96`
- Action: When Redis client is `None` or raises, return `True` from `_is_token_blacklisted` (fail-closed) and emit a Prometheus counter `jwt_blacklist_redis_errors_total`. Add circuit breaker (e.g., 5 failures → open for 30s, during which all auth requests return 503).
- Fail-first test: `tests/security/test_jwt_manager.py::test_redis_outage_fails_secure` — patch redis to raise ConnectionError, assert revoked-then-checked token returns blacklisted=True. Today returns False.
- Pass-after: passes; chaos test asserts API returns 503 (not 200) during simulated Redis outage.
- Path verified: yes

**Step 7: Authenticate exposed endpoints (F-01-004, F-01-005, F-01-006, F-03-001, F-08-013)**
- Files:
  - `backend/api/routers/websocket.py:170-197` (market/portfolio WS)
  - `backend/api/routers/websocket.py:353-364` (trigger/alert)
  - `backend/api/routers/websocket.py:367-379` (trigger/news)
  - `backend/api/routers/websocket.py:382-393` (/ws/connections)
  - `backend/ml/ml_api_server.py:190-214` (ML mutating endpoints)
  - `backend/security/csrf_protection.py:77-80` (login/register exemption)
- Action: Add `current_user: User = Depends(get_current_user)` (or `get_current_admin_user`) to REST endpoints. Apply `@secure_websocket(require_auth=True)` decorator to market/portfolio WS; add ownership check `if portfolio_id not in user.portfolio_ids: close(4003)`. ML server: bind `127.0.0.1`, replace `allow_origins=["*"]` with `settings.ML_API_ALLOWED_ORIGINS`, add API key Depends. CSRF: keep exemption but add strict `Origin`/`Referer` allowlist check on `/login` and `/register`.
- Fail-first test: `tests/api/test_websocket_security.py::test_anon_market_ws_rejected` — anonymous WS connect → close code 4008. `tests/ml/test_ml_api_auth.py::test_unauth_delete_model` → 401. Currently both succeed unauthorized.
- Pass-after: tests pass; admin-authed call returns 200.
- Path verified: yes (all 6 files)

**Step 8: Socket.IO auth + CORS (F-02-011)**
- Files: `backend/services/socketio_service.py:48-53`, plus connect/subscribe handlers
- Action: Replace `cors_allowed_origins="*"` with `settings.CORS_ORIGINS`. Implement `@sio.event async def connect(sid, environ, auth)`: extract JWT from `auth["token"]`, call `get_jwt_manager().verify_token`, store user on `sio.save_session`. In `subscribe_portfolio`, validate ownership before `enter_room`.
- Fail-first test: `tests/services/test_socketio.py::test_foreign_origin_rejected` and `test_subscribe_others_portfolio_denied`. Currently both succeed.
- Pass-after: tests pass.
- Path verified: yes

**Step 9: Real JWT + WS + ML test coverage (F-01-013, F-03-014, F-15-006, F-15-007)**
- Files:
  - `backend/tests/security/test_security_modules.py:73-74` (replace MagicMock)
  - `backend/tests/test_security_compliance.py:74` (replace string-secret JWTManager)
  - new: `backend/tests/test_websocket_security.py`, `backend/tests/test_ml_api_server.py`
- Action: Replace `MagicMock()` and `JWTManager(secret_key="test_...")` fixtures with `get_jwt_manager()` (using a test RSA key pair from existing `rsa_keys` fixture at `test_security_modules.py:59-70`). Add `TestClient` integration tests for `ml_api_server.py` covering all CRUD + path traversal + unauth. Add pytest-asyncio tests for `/ws/stream`: anon rejection, rate limit (>60 msg/min), VIEWER role >10 symbols denied.
- Fail-first test: the new tests themselves serve as the regression suite — they fail today because the fixtures don't exercise real RS256 paths and ML/WS auth tests don't exist.
- Pass-after: `pytest backend/tests/security backend/tests/test_websocket_security.py backend/tests/test_ml_api_server.py` ≥80% endpoint coverage.
- Path verified: yes

**Step 10: Cleanup — MD5, MFA key (F-03-017, F-08-015)**
- Files: `backend/ml/feature_store.py:764-771`, `backend/security/jwt_manager.py:496-508`
- Action: `hashlib.md5` → `hashlib.sha256` in `_generate_cache_key`. Re-key MFA secrets store from `username` to `user.id` (UUID); add migration for existing rows.
- Fail-first test: not strictly currently-broken; regression tests `test_cache_key_uses_sha256` and `test_mfa_unique_after_username_reuse`.
- Pass-after: tests pass.
- Path verified: yes

**Step 11: Unify exception hierarchies (F-11-001)**
- Files: `backend/exceptions.py`, `backend/utils/exceptions.py`
- Action: Promote `backend/exceptions.py` as canonical. Move `RateLimitException`, `DataIngestionException`, `ExternalAPIException` etc. into it, renaming to the `…Error` suffix convention. Convert `backend/utils/exceptions.py` to a deprecation shim that re-exports + `warnings.warn(DeprecationWarning)`.
- Fail-first test: not currently-broken; regression test asserts no production code imports from `backend.utils.exceptions` (only the shim itself).
- Pass-after: `pytest backend/tests/ -k 'exception or validation'` passes; `grep -rn "from backend.utils.exceptions"` returns only the shim.
- Path verified: yes

**Step 12: Frontend token persistence + test (F-12-011, F-12-015)**
- Files: new `frontend/web/src/utils/tokenStorage.ts`; update `frontend/web/src/store/slices/appSlice.ts:68` and `frontend/web/src/services/api.service.ts:57-58`
- Action: Create `tokenStorage.ts` with `setAccessToken`, `getAccessToken`, `clearTokens`. Use `access_token` (matching backend response after Step 4) as the canonical key. Both login thunk and refresh interceptor call `setAccessToken(response.data.access_token)`.
- Fail-first test: `tokenStorage.test.ts` + extend `slices.test.ts` line 219 with `expect(localStorageMock.setItem).toHaveBeenCalledWith('access_token', 'expected-token')`. Currently no such assertion.
- Pass-after: tests pass; manual: existing user can log in and refresh works without re-login.
- Path verified: yes

## 4. Files touched

Backend (canonical auth):
- `backend/security/security_config.py` (Step 1)
- `backend/security/jwt_manager.py` (Steps 2, 6, 10)
- `backend/auth/oauth2.py` (Step 3)
- `backend/api/routers/auth.py` (Steps 4, 5)
- `backend/utils/auth.py` (Step 5)
- `backend/api/schemas/user.py` (Step 5 — add `UserPublic`)

Endpoints needing auth:
- `backend/api/routers/websocket.py` (Step 7)
- `backend/ml/ml_api_server.py` (Step 7)
- `backend/security/csrf_protection.py` (Step 7)
- `backend/services/socketio_service.py` (Step 8)

Tests:
- `backend/tests/security/test_security_modules.py` (Step 9)
- `backend/tests/test_security_compliance.py` (Step 9)
- `backend/tests/test_websocket_security.py` (new, Step 9)
- `backend/tests/test_ml_api_server.py` (new, Step 9)
- `backend/tests/security/test_security_config.py` (new fail-first, Step 1)
- `backend/tests/security/test_jwt_manager.py` (new/extend, Steps 2, 6)
- `backend/tests/api/test_auth_router.py` (Step 4)
- `backend/tests/auth/test_oauth2.py` (Step 3)
- `backend/tests/services/test_socketio.py` (Step 8)

Cleanup:
- `backend/ml/feature_store.py` (Step 10)
- `backend/exceptions.py` (Step 11)
- `backend/utils/exceptions.py` (Step 11 — deprecation shim)

Frontend:
- `frontend/web/src/utils/tokenStorage.ts` (new, Step 12)
- `frontend/web/src/store/slices/appSlice.ts` (Step 12)
- `frontend/web/src/services/api.service.ts` (Step 12)
- `frontend/web/src/utils/tokenStorage.test.ts` (new, Step 12)
- `frontend/web/src/store/slices/__tests__/slices.test.ts` (extend, Step 12)

## 5. Acceptance tests (consolidated)

Backend:
1. `pytest backend/tests/security/test_security_config.py::test_missing_jwt_secret_in_production_raises` — passes (F-08-002)
2. `pytest backend/tests/security/test_jwt_manager.py::test_revoke_blacklists_token` — passes (F-08-004)
3. `pytest backend/tests/security/test_jwt_manager.py::test_revoke_caps_ttl_on_forged_exp` — passes (F-08-017)
4. `pytest backend/tests/security/test_jwt_manager.py::test_redis_outage_fails_secure` — passes (F-08-008)
5. `pytest backend/tests/auth/test_oauth2.py::test_no_hs256_fallback` — passes (F-08-005)
6. `grep -rn "HS256" backend/auth backend/api | grep -v test | wc -l` returns 0 (F-08-005)
7. `pytest backend/tests/api/test_auth_router.py::test_login_returns_200_not_500` — passes (F-01-001)
8. `pytest backend/tests/api/test_auth_router.py::test_login_token_verifiable_by_jwt_manager` — passes (F-01-009)
9. `grep -rn "from backend.api.routers.auth import get_current_user"` returns 0 (F-01-012)
10. `pytest backend/tests/test_websocket_security.py -k "anon or rate_limit or viewer_limit or ownership"` — passes (F-01-006, F-01-013)
11. `curl -X POST http://localhost:8000/api/v1/ws/trigger/alert` → 401 (F-01-004)
12. `curl http://localhost:8000/api/v1/ws/connections` → 401 (F-01-005)
13. `curl -X DELETE http://localhost:8001/models/lstm_price_predictor` → 401 (F-03-001)
14. `pytest backend/tests/test_ml_api_server.py` ≥80% coverage on endpoints (F-03-014)
15. `pytest backend/tests/services/test_socketio.py::test_foreign_origin_rejected` — passes (F-02-011)
16. `pytest backend/tests/security/test_security_modules.py -k "jwt" -v` — passes against real `JWTManager` (F-15-006, F-15-007)
17. CSRF: `curl -X POST -H "Origin: https://evil.com" /api/v1/auth/token` → 403 (F-08-013)
18. `grep "hashlib.md5" backend/ml/feature_store.py` returns 0 (F-03-017)
19. `pytest -k "mfa_unique_after_username_reuse"` — passes (F-08-015)
20. `pytest backend/tests/ -k 'exception or validation'` — passes; `grep -rn "from backend.utils.exceptions" backend/ | grep -v utils/exceptions.py` returns 0 (F-11-001)
21. `UserPublic` schema test — passes; routers return identical shape (F-11-016)

Frontend:
22. `vitest tokenStorage.test.ts` — passes (F-12-011)
23. `vitest slices.test.ts` with `localStorageMock` assertion — passes after fix, fails before (F-12-015)

End-to-end:
24. Log in via real frontend → token stored under `access_token` → portfolio page loads → logout → token blacklisted → re-use of token returns 401.

## 6. Rollback plan

- **Commit boundaries.** One commit per Step (12 commits) so any single step can be reverted with `git revert`.
- **Critical rollback chain.** Steps 1–4 form an atomic group (must all roll back together — reverting Step 4 alone restores the broken bespoke `auth.py` that depends on the ephemeral fallback restored by reverting Step 1). Tag `pre-cluster-B` before Step 1 and `post-cluster-B-core` after Step 4.
- **Database/state.** No schema changes except Step 10 MFA re-keying, which has a forward migration; rollback requires a reverse migration script (write before deploy).
- **Cache invalidation.** All existing JWTs become invalid after Step 4 deploy → "user re-login storm" expected (see Risks). Communicate in release notes; do not roll back to "fix" this — rollback would re-introduce CVE.
- **Feature flag option.** Wrap Step 7's new auth requirements behind `ENABLE_STRICT_WS_AUTH` env flag for first 24h to allow fast disable without revert.
- **Redis blacklist (Step 6).** If circuit breaker false-positives flood 503s, env var `JWT_BLACKLIST_FAIL_OPEN=true` reverts to legacy behavior pending fix.

## 7. Dependencies

- depends_on: [{workstream: A, type: blocks, reason: 'JWT secret rotation must complete to avoid ephemeral fallback masking RS256 fix; JWT_PRIVATE_KEY/JWT_PUBLIC_KEY must be in env before Step 1 removes the fallback or login breaks worse than current state'}]
- depends_on: [{workstream: A, type: coordinates_with, reason: 'Cluster A secrets-manager hardcoded-salt fix (F-08-001) undermines all stored JWT keys; Step 9 RS256 fixture tests will fail until A lands'}]
- blocks: [{workstream: C-and-beyond, type: blocks, reason: 'RBAC roles/permissions work, audit logging of auth events, and any feature touching `Depends(get_current_user)` should land after Step 5 unifies the dependency'}]
- blocks: [{workstream: frontend-feature-work, type: blocks, reason: 'Step 12 tokenStorage refactor must land before any new frontend feature persists tokens'}]

## 8. Effort & cost

- Effort: **40–55 hours** total
  - Step 1: 2h · Step 2: 1.5h · Step 3: 3h · Step 4: 4h · Step 5: 5h · Step 6: 4h · Step 7: 7h · Step 8: 4h · Step 9: 8h · Step 10: 2.5h · Step 11: 6h · Step 12: 2.5h
  - Buffer for cross-step regression and frontend manual QA: ~5h
- Loki token cost: **~$1.80** (mostly Sonnet for Steps 4, 5, 7, 9; Haiku for mechanical edits Steps 1, 2, 3, 10, 12)

## 9. Loki-actionable status

- `requires_human_ack: false` for **Steps 1–4, 6, 8, 10, 12** — mechanical, well-bounded edits with deterministic test signals.
- `requires_human_ack: true` for:
  - **Step 5** — needs decision on `UserPublic` field set (does it expose `email`? `roles`? `mfa_enabled`?). Single-vs-multi-role decision pending → flag.
  - **Step 7** — admin-vs-user role distinction on `/ws/connections` and ML endpoints needs RBAC clarity (same single-vs-multi-role decision); also requires SecOps sign-off on ML server bind change to `127.0.0.1`.
  - **Step 9** — new test files in protected test directory; ack on coverage threshold.
  - **Step 11** — naming/migration breaks any external code importing from `backend.utils.exceptions`; needs maintainer ack.

Per-step:

| Step | Loki actionable | Human ack |
|---|---|---|
| 1 | yes | no |
| 2 | yes | no |
| 3 | yes | no |
| 4 | yes | no |
| 5 | yes | **yes** (UserPublic schema) |
| 6 | yes | no |
| 7 | yes | **yes** (RBAC + ML bind) |
| 8 | yes | no |
| 9 | yes | **yes** (coverage policy) |
| 10 | yes | no |
| 11 | yes | **yes** (deprecation policy) |
| 12 | yes | no |

## 10. Risks

1. **User re-login storm.** Step 4 invalidates all existing tokens (algorithm + key change). Expect every active user to be logged out simultaneously on deploy. Mitigations: deploy at low-traffic window; graceful 401 handling on frontend already routes to `/login`; communicate via in-app banner 24h ahead.
2. **Multi-worker key inconsistency window.** Between Cluster A landing keys and Cluster B Step 1 enforcing them, a misconfigured worker could still load the ephemeral fallback. Mitigate with a deploy gate: Step 1 deploy script asserts `JWT_PRIVATE_KEY` is set in target env before rollout.
3. **Redis fail-secure outage amplification.** Step 6 turns Redis outage from "blacklist degrades" into "auth returns 503". If Redis flaps, every API call 503s. Mitigation: circuit breaker with 30s open window, Prometheus alert + `JWT_BLACKLIST_FAIL_OPEN=true` rollback flag.
4. **CSRF tightening on `/login` (Step 7).** Strict Origin allowlist may reject legitimate logins from staging/preview environments. Pre-deploy: enumerate all valid origins (prod, staging, preview-*, mobile webview) and add to env.
5. **`UserPublic` schema breakage (Step 5).** Frontend currently consumes raw dict from `_user_to_dict`. If `UserPublic` excludes a field the UI uses, profile pages break. Run TypeScript codegen against the new schema before merging.
6. **WebSocket clients without token refresh logic.** Step 7's `@secure_websocket` rejects connections after token expiry; some clients may not re-handshake. Add WS protocol message `auth_expired` and document.
7. **Test fixture migration churn (Step 9).** Replacing `MagicMock` with real `JWTManager` may surface latent test bugs (tests that "passed" only because the mock was permissive). Budget extra debugging time.
8. **Exception hierarchy migration (Step 11) collides with parallel work.** Any in-flight PR importing from `backend.utils.exceptions` will need rebase. Coordinate timing with active feature branches.
