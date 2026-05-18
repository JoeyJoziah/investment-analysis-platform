# Local E2E Setup

This project's Playwright E2E suite (`frontend/web/tests/e2e/`) requires a running stack: PostgreSQL+TimescaleDB, Redis, backend (FastAPI), and frontend (Vite).

## One-command bootstrap

```powershell
pwsh scripts/e2e-local-bootstrap.ps1
```

Then from `frontend/web/`:
```powershell
npx playwright test --project=chromium auth.spec.ts        # 17 specs, ~5 min
npx playwright test --project=chromium portfolio.spec.ts   # 20 specs, ~5-10 min
npx playwright test --project=chromium                     # 37 specs, full chromium
npx playwright test                                        # 185 specs, 5 browsers
```

Stop with:
```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.e2e-portshift.yml stop postgres redis
```

## Why a port-shift overlay?

On Windows hosts with a native `postgres.exe` service bound to `0.0.0.0:5432`, Docker's `5432:5432` mapping is silently overridden by the OS. The native postgres wins the bind and Docker's published port becomes unreachable from the host.

`docker-compose.e2e-portshift.yml` shifts the Docker postgres to host `5433`:

```yaml
services:
  postgres:
    ports:
      - "5433:5432"
```

The bootstrap script always layers this overlay, so it's safe whether or not a native postgres is running. Backend's `DATABASE_URL` and `DB_PORT` are exported by the script to point at `5433`.

## Known caveats

1. **First login is slow (~22s).** First-call DB pool init + cache warming + middleware stack initialization. Subsequent logins are sub-second. Playwright `actionTimeout` is bumped to 30s in `playwright.config.ts` to accommodate.
2. **bcrypt must be `<4.1`.** passlib 1.7.4's `detect_wrap_bug` initialization crashes on bcrypt 4.1+. Pinned in `requirements.txt`.
3. **alembic upgrade head on a fresh DB will not produce a working schema.** Migration 001+ only do incremental ops (add indexes, partitioning, columns); base table DDL lives in `Base.metadata.create_all()`. The bootstrap script uses `create_all()` directly.
4. **Postgres data volume password.** The container `POSTGRES_PASSWORD` env applies only on first-init. If the volume already exists from a prior session with a different password, login will fail with `password authentication failed`. The bootstrap script issues `ALTER USER postgres WITH PASSWORD '...'` against the running container to align it with `.env` (idempotent, safe to run every time).
5. **E2E test users.** `existing@example.com` / `ExistingPass123!@#` and `portfolio-test@example.com` / `PortfolioTest123!` are seeded fresh on every bootstrap. Hardcoded in the spec files.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `docker: command not recognized` | Docker Desktop installed but not on PATH | Bootstrap script auto-adds `C:\Program Files\Docker\Docker\resources\bin`; or restart shell |
| `password authentication failed` | Stale data volume password | Re-run bootstrap (issues ALTER USER) or `pwsh scripts/e2e-local-bootstrap.ps1 -Reset` |
| Health check timeout | Backend cold-start (~60s first time) | Wait; subsequent requests are fast |
| All E2E tests fail at `beforeEach` | `localStorage.clear()` on `about:blank` (SecurityError) | Fixed: `auth.spec.ts` now navigates first |
| `button:has-text("Login")` not found | Frontend button is "Sign In" | Fixed in spec files |
| `expect(localStorage.getItem('token')).toBe(true)` always false | Frontend uses `access_token` / `refresh_token` keys | Fixed in spec files |