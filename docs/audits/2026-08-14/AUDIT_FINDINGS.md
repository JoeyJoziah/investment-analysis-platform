# Vulnerability hunt — six-project audit

Date: 2026-08-14  
Scope: `investment-analysis-platform` (Wave C surfaces), `portfolio-bridge`, `msos-options-monitor`, `wheel-analytics`, `market-intel`, `efinancialmodels-workshop`.  
Excluded: `tax-prep-2025`, `thesis-monitor` (not opened). Secrets/`.env` bodies not opened.  
Method: context from the prior pass, then re-read of the cited lines. No exploit payloads.

Severity is impact **if the process is reachable as deployed in this repo** (compose publishes `:8000`; `.env` in-tree is `ENVIRONMENT=development`; ML compose publishes `:8001`).

---

## Summary

| ID | Sev | Title | Project | Status |
|---|---|---|---|---|
| IAP-001 | **Critical** | Authenticated IDOR: any user can execute trades on any portfolio | IAP | **Fixed** — ownership required on validate/execute |
| IAP-002 | **High** | HS256 JWT fallback accepts tokens without iss/aud/type; dev secret is in source | IAP | **Fixed** — decode path is RS256 only |
| IAP-003 | **High** | Unauthenticated WebSocket trigger + connection dump | IAP | **Fixed** — admin-only |
| IAP-004 | **High** | ML inference API on `:8001` has no auth and is published | IAP | **Fixed** — `ML_API_TOKEN` + localhost bind |
| IAP-005 | **High** | Any logged-in user can write vendor API keys when `ENVIRONMENT` is unset or `development` | IAP | **Fixed** — admin + `settings.ENVIRONMENT` |
| IAP-006 | Medium | Any logged-in user can promote/rollback production models | IAP | **Fixed** — admin-only |
| IAP-007 | Medium | Logout does not revoke; `/refresh` extends a stolen access token | IAP | **Fixed** — revoke + refresh token required |
| IAP-008 | Medium | Unauthenticated analysis endpoints consume vendor quota / CPU | IAP | **Fixed** — `get_current_user` |
| IAP-009 | Medium | CSRF and HTTP rate-limit off whenever getenv `ENVIRONMENT` is `development` | IAP | **Fixed** — default is `settings.ENVIRONMENT` |
| IAP-010 | Medium | Consent withdrawal persists raw client IP | IAP | **Fixed** — anonymize before persist |
| IAP-011 | Low | Trade impact is the same IDOR without a write | IAP | **Fixed** — same ownership check |
| IAP-012 | Low | `GET /api/v1/metrics` is unauthenticated | IAP | **Fixed** — scrape token or admin JWT |
| MSOS-001 | Low | Polygon key line match is prefix, not exact | MSOS | **Fixed** — exact `KEY=` match |
| BRIDGE-001 | Low | Source `ok` does not mean data exists (integrity for downstream writers) | bridge | **Fixed** — empty parse is `partial`; IAP `--apply` refuses empty desired |

Not listed as vulns (checked, not exploitable as feared):

- Admin `POST /command` does **not** run OS commands; it returns a dict.
- Socket.IO room joins now require a verified access token on `connect`; `subscribe_alerts` is owner-only. Default compose still serves `main:app`, not `socket_app`.
- `market-intel` path construction was already hardened (`_SAFE_NAME`, containment under `raw/`).
- EFM has no network or auth surface.

---

## IAP-001 — Critical — Cross-user trade execution (IDOR)

**Where.** `backend/api/routers/trading.py` L106–128; `backend/services/trading_service.py` L137–193; `backend/repositories/portfolio_repository.py` L213–215.

**What.** `POST /api/v1/trading/orders/{portfolio_id}` requires a login (`get_current_user`) and then **never uses** `current_user`. `execute_trade` takes only an integer PK. `add_position` locks `Portfolio.id == portfolio_id` with no `user_id` predicate.

Any registered user who can guess or enumerate another user’s integer `portfolios.id` can buy or sell against that book: cash moves, positions upsert, a `transactions` row is written.

Register is public (`POST /api/v1/auth/register`). IDs are sequential integers.

**Related read.** `POST /orders/{portfolio_id}/impact` (L137–158) is the same missing ownership check and returns cash / allocation. Tracked as IAP-011.

**Fix direction.** Pass `current_user.id` into `add_position` / `get_by_id` and require `Portfolio.user_id == current_user.id` (or use the UUID `portfolio_id` plus `get_user_portfolio`). Fail 404 on miss.

---

## IAP-002 — High — HS256 verify fallback without claim checks

**Where.** `backend/auth/oauth2.py` L174–222; `backend/security/security_config.py` L187–215.

**What.** Live login mints RS256 via `jwt_manager`. Every HTTP `get_current_user` goes through `decode_access_token`, which:

1. Tries RS256 (`iss`, `aud`, `type=access`, optional Redis session).
2. On failure, `jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM, JWT_ALGORITHM_FALLBACK])` — i.e. RS256 **and** HS256 — with **no** issuer, audience, or token-type check.

Outside production, a missing `JWT_SECRET_KEY` becomes the **source-committed** string `dev-only-insecure-jwt-secret-do-not-use-in-production`. Anyone who can reach a non-prod API can mint an HS256 token whose `sub` is a victim email/username and pass `get_current_user`.

In production the HMAC secret is required, so the default string is not used. The fallback still accepts any HS256 token signed with that one secret, skipping the RS256 session/blacklist/`type` gates. A leaked `JWT_SECRET_KEY` is therefore equivalent to a master impersonation key, weaker than the RSA private key.

Comment at L177–182 claims auth.py still mints HS256. It does not (`create_tokens` is RS256). The fallback is leftover and broader than the comment.

**Fix direction.** Delete the HS256 fallback on the live path, or require `iss`/`aud`/`type=access` and a non-default secret in every environment. Do not list RS256 and HS256 against an HMAC key.

---

## IAP-003 — High — Unauthenticated realtime control plane

**Where.** `backend/api/routers/websocket.py` L441–481.

| Route | Auth | Effect |
|---|---|---|
| `POST /api/v1/ws/trigger/alert` | none | injects an alert to `client_id` |
| `POST /api/v1/ws/trigger/news` | none | broadcasts headline/summary to all native WS clients |
| `GET /api/v1/ws/connections` | none | dumps client ids and subscription maps |

`/api/v1/ws` is **not** in the CSRF exempt list. CSRF is also **not registered** in development (IAP-009). Even with CSRF on, these routes have no bearer check; a non-browser client is enough.

Mounted on the same `:8000` process compose publishes.

**Fix direction.** Admin-only or service-token; remove `GET /connections` from the public API or gate it.

---

## IAP-004 — High — Unauthenticated ML control plane on a published port

**Where.** `backend/ml/ml_api_server.py` L42–48, L195–276, L303–306; `docker-compose.ml-production.yml` L4–10.

**What.** Separate FastAPI app, CORS `allow_origins=["*"]`, `host="0.0.0.0"`, port 8001. Production compose publishes `8001:8001`.

No `Depends`, API key, or network policy in the Python. Routes include:

- `POST /predict`
- `POST /models/{model_name}/load`
- `DELETE /models/{model_name}`
- `POST /retrain` → `subprocess.run(['python3', 'backend/ml/minimal_training.py'], timeout=300)`

Path traversal on `model_name` was constrained (`_MODEL_NAME_RE` + `relative_to`). Unauthenticated load/unload/retrain remains.

**Fix direction.** Do not publish 8001. Bind localhost or internal network only. Add the same auth (or a dedicated service token) as `/api/v1/ml`.

---

## IAP-005 — High — Any user can persist vendor API keys in development

**Where.** `backend/api/routers/settings.py` L463–521; `backend/api/routers/auth.py` L119–151.

**What.** `PUT /api/v1/settings/api-keys` requires only `get_current_user`. The second gate is:

```python
environment = os.getenv("ENVIRONMENT", "development").lower()
```

If the process env var is unset, the default is **development** and the write proceeds. That disagrees with `Settings.ENVIRONMENT`, which defaults to `"production"`.

On allow: whitelist keys are written to repo-root `.env` (with `.env.bak`), then `os.environ[...] = value` and `setattr(app_settings, ...)`.

This repo’s compose loads `.env` with `ENVIRONMENT=development`. Combined with public register, any account can rotate Polygon/Finnhub/AV/FMP/News keys used by ingest and by MSOS (`config.yaml` points `polygon.env_file` at that same IAP `.env`).

**Fix direction.** Gate on `settings.ENVIRONMENT` (or require admin). Default deny. Never let a normal user write the process env.

---

## IAP-006 — Medium — Model promote / rollback is not admin-only

**Where.** `backend/api/routers/ml.py` L829–875.

**What.** `POST /api/v1/ml/versions/{model_name}/promote` and `/rollback` use `get_current_user` only. Any active account can move a version to `production`.

**Fix direction.** `get_current_admin_user` (and fix IAP-002 so admin is not forgeable).

---

## IAP-007 — Medium — Tokens are not revoked

**Where.** `backend/api/routers/auth.py` L240–257; `jwt_manager.revoke_token` exists and is unused here.

**What.** `/logout` logs and returns success. It does not blacklist the JWT or delete `user_session:*`. `/refresh` requires a still-valid **access** token and mints another access token. A stolen access token works until `exp` and can be refreshed into a new `exp` without the original refresh JWT.

**Fix direction.** Logout → `revoke_token` + session delete. Refresh should consume a refresh token, not an access token.

---

## IAP-008 — Medium — Unauthenticated analysis is a quota / CPU sink

**Where.** `backend/api/routers/analysis.py` L221–228, L612–632.

**What.** `POST /api/v1/analysis/analyze` and `/batch` have no `get_current_user`. They pull market data and run multi-layer analysis (cached 300s on analyze). `/batch` loops `analyze_stock` per symbol.

Anyone who can hit `:8000` can burn Finnhub/AV/CPU. CSRF does not apply to a non-browser client.

**Fix direction.** Require auth and a per-user rate limit that is actually registered (see IAP-009).

---

## IAP-009 — Medium — CSRF and HTTP rate-limit omitted in this repo’s default env

**Where.** `backend/api/main.py` L214–257.

**What.** CSRF and `RateLimitingMiddleware` register only when `os.getenv("ENVIRONMENT", "development") != "development"`. Compose interpolates `${ENVIRONMENT:-production}` **and** loads `.env`, which in this tree is `development`. A stock `docker compose up` here therefore has neither CSRF nor HTTP rate-limit.

Other middleware (CORS, headers, size, audit) still runs.

**Fix direction.** Use one ENVIRONMENT source. Default deny (enable CSRF/RL unless explicitly `development`). Do not ship `.env` with `development` if compose is treated as prod-like.

---

## IAP-010 — Medium — Withdraw-consent stores raw IP

**Where.** `backend/api/routers/gdpr.py` L456–470 vs L401–413.

**What.** `POST /users/me/consent` anonymizes IP before persist. `DELETE /users/me/consent/{type}` passes `raw_ip` into `withdraw_consent` → `record_consent` → `AuditLog.ip_address`. The HTTP response remasks; the row does not.

**Fix direction.** Anonymize before persist on both paths.

---

## IAP-011 — Low — Cross-user portfolio impact read

Same missing ownership as IAP-001 on `POST /api/v1/trading/orders/{portfolio_id}/impact`. Returns cash and allocation for any integer PK.

---

## IAP-012 — Low — Prometheus scrape is public

**Where.** `backend/api/main.py` L401–406.

`GET /api/v1/metrics` has no auth. Rate-limiter skip lists mention `/api/metrics` and `/api/health`, not this path.

---

## MSOS-001 — Low — Env-key prefix match

**Where.** `msos-options-monitor/src/msos_monitor/config.py` L67–73.

`stripped.startswith(key_name)` accepts `POLYGON_API_KEY_OLD=...` or `POLYGON_API_KEY_BACKUP=...` before the real line. First match wins. Can send the wrong Bearer token to Polygon (fail closed on 401, or use a stale key).

**Fix direction.** Match `KEY=` or `KEY =` as a full token.

---

## BRIDGE-001 — Low — Integrity: `ok` ≠ rows

**Where.** `portfolio-bridge/src/bridge/sync.py` L61–72.

A parser that returns an empty `NormalizedSource` is `ok`. Downstream that trust `source_status` (or that treat empty desired as “sell everything” — see IAP sync `compute_sync_actions`) can act on a silent zero. IAP apply currently does not persist (router DTO), so the live blast radius is other consumers (wheel, `latest.json` readers), not IAP books.

---

## Latent / not scored

**Socket.IO unauthenticated rooms** (`socketio_service.py` L159–231): `subscribe_portfolio` / `subscribe_alerts` join any id. Default compose serves `app`, not `socket_app`. Becomes High if someone switches the ASGI target.

**Admin `POST /command`:** whitelist + dict echo. Not RCE. Operators should not treat `status: executed` as a host change.

**`GET /agents/capabilities`:** now requires `get_current_user`.

Also closed alongside the findings:

- Portfolio `add_position` / `remove_position` persist through the repository with ownership (no more DTO-only write).
- `get_user_portfolio` / `get_portfolio_positions` / `get_recent_transactions` implemented so `compute_portfolio_detail` no longer calls missing methods.
- JWT revoke has an in-process blacklist when Redis is down.

---

## Remediation (2026-08-14)

All scored findings above were implemented in product code. Targeted tests: IAP unit remediations + JWT revoke + bridge sync + MSOS key parse (64 + 8 + pipeline suite green). Full IAP integration suite was not re-run.
