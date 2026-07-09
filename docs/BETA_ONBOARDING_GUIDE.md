# Beta Testing & Onboarding Guide

Wave 12 / #32 — quick path for beta testers to run the Investment Analysis Platform.

## 1. Prerequisites

- Docker + Docker Compose (or local Python 3.12 + Node 20)
- Git clone of this repository
- Optional API keys in `.env` (see `.env.example`): market data, OpenAI/Anthropic, Sentry, Slack

## 2. Local bring-up (fastest)

```bash
cp .env.example .env
# Edit .env — set SECRET_KEY, JWT_SECRET_KEY, DB_PASSWORD, REDIS_PASSWORD at minimum
docker compose up -d
```

- API: `http://localhost:8000/docs` (OpenAPI)
- Health: `http://localhost:8000/api/health` (or project health route)
- Web UI: `http://localhost:3000` (if frontend service is running)

More detail: [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md).

## 3. First-run checklist

| Step | Action |
|------|--------|
| 1 | Register / login via `/register` and `/login` |
| 2 | Open **Dashboard** — confirm shell loads |
| 3 | **Search / Market** — look up a ticker (e.g. AAPL) |
| 4 | **Portfolio** — add a paper position |
| 5 | **Watchlist** — pin symbols of interest |
| 6 | **Analysis** — open `/analysis/AAPL` |
| 7 | **Recommendations** — review model outputs (demo/ML keys affect quality) |
| 8 | **Settings** — persistence of preferences |
| 9 | **Thesis** — document rationale using the investment thesis template |

Thesis template: [templates/investment_thesis_template.md](./templates/investment_thesis_template.md).

## 4. Testing as a beta tester

- Unit/integration: `pytest backend/tests/unit -q` (subset)
- E2E (Playwright): `cd frontend/web && npm run test:e2e`
- Load/perf suites: see `backend/tests/test_performance_load.py`

Guides: [testing/TESTING_GUIDE.md](./testing/TESTING_GUIDE.md).

## 5. Staging & production notes

- Staging images: GitHub Actions `staging-deploy.yml` (builds/pushes GHCR on `main`)
- Canary: `canary-deploy.yml` (manual; health-gate + auto-rollback path)
- SSL: [SSL_DEPLOYMENT_GUIDE.md](./SSL_DEPLOYMENT_GUIDE.md) + `infrastructure/docker/nginx/nginx-ssl.conf`
- Full prod: [PRODUCTION_DEPLOYMENT_GUIDE.md](./PRODUCTION_DEPLOYMENT_GUIDE.md)

## 6. Feedback loop

1. File bugs as GitHub issues with repro steps + environment
2. Attach screenshots for UI issues
3. Note feature flags / `.env` keys that affect behavior (`DEMO_MODE`, ML keys, etc.)

## 7. Safety / compliance

- Not investment advice — platform outputs are research tooling
- Never commit real secrets; use `.env` (gitignored) or a secret manager
- GDPR endpoints exist under the API for data export/deletion when authenticated
