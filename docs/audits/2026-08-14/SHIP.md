# Ship record — 2026-08-14

Session: audit-context → vulnerability hunt → implement findings → verify → review/commit/push.

## Landed

### IAP `06ef192` on `origin/main`

- IAP-001/011: trade validate/execute/impact require ownership; miss is 404
- IAP-002: `decode_access_token` is RS256/`jwt_manager` only
- IAP-003: WS trigger/news/connections admin-only
- IAP-004: ML API `ML_API_TOKEN` except `/health`; compose `127.0.0.1:8001:8001`
- IAP-005: PUT api-keys admin + `settings.ENVIRONMENT`
- IAP-006: promote/rollback admin
- IAP-007: logout `revoke_token`; refresh consumes refresh token
- IAP-008: `/analyze` and `/batch` require login
- IAP-009: CSRF/RL getenv default is `settings.ENVIRONMENT`
- IAP-010: withdraw consent anonymizes IP before persist
- IAP-012: `/api/v1/metrics` scrape token or admin JWT
- Extra: portfolio add/remove persist + ownership; `get_user_portfolio`; Socket.IO connect auth; `/agents/capabilities` auth

### MSOS `358bec9` + `4e32038` on `origin/master`

- Exact `POLYGON_API_KEY=` match
- Daily runner no longer treats `claude` stderr as a failed chain

### portfolio-bridge `46403d4` on `origin/master`

- Empty parse is `partial`, not `ok`

## Verification (this session)

- IAP targeted: 91 passed, exit 0
- bridge `tests/test_sync.py`: 8 passed, exit 0
- MSOS `tests/test_pipeline.py`: exit 0
- Full IAP suite: not run

## Residuals (still true after ship)

1. `create_access_token` / `create_refresh_token` still mint HS256 if jwt_manager throws. Decode rejects those tokens.
2. IAP-009 is only the getenv default. Local `ENVIRONMENT=development` still skips CSRF/RL.
3. IAP `.claude/worktrees/` and `scripts/setup/*` agent dumps stay untracked on purpose.
4. `market-intel` has unrelated dirty files from another session (`src/mktintel/http.py`, `tests/test_http_review.py`). Not this work.
5. `efinancialmodels-workshop` is ahead of origin by 2 with untracked `specs/`. Not this work.

## Decisions

- Ownership miss → 404 (no existence leak).
- Refresh body is `{refresh_token}`.
- Empty broker parse is `partial`; IAP `--apply` refuses an empty desired set.
- Socket.IO connect requires a valid access token.
- MSOS had no GitHub remote; a **private** repo was created rather than leaving it local-only.

## Exclusions

Do not open `tax-prep-2025` or `thesis-monitor` unless asked.
