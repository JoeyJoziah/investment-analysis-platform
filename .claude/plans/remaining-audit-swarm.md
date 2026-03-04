# Remaining Audit Items — Parallel Swarm Execution Plan

## Baseline

- **Branch**: main
- **Tests**: 5028 collected, 5026 passed, 8 skipped, 2 xfailed
- **Frontend**: Vitest configured in vite.config.ts, 9 page tests + 4 component tests
- **Services**: 19 service files in backend/services/
- **Prior waves**: P0-P5 complete, Waves 1-14 remediation done, 3 audit waves done

---

## Remaining Work (5 Categories, 3 Waves)

### Wave 1: Backend Code Quality (4 parallel agents)

**Agent 1 — Random Data Removal: Services** (Sonnet)
Replace `random.*` calls with deterministic stubs in 7 service files + 1 router.

Files:
- `backend/services/recommendation_service.py`
- `backend/services/recommendation_crud.py`
- `backend/services/portfolio_rebalancing.py`
- `backend/services/socketio_service.py`
- `backend/services/websocket_service.py`
- `backend/services/agents_service.py`
- `backend/services/admin_service.py`
- `backend/api/routers/admin.py`

Pattern: Replace `random.uniform/randint/choice/random/sample` with `None`, `0.0`, empty list, or deterministic values. Remove `import random`. Add `"mock": True` flag where appropriate.

**Agent 2 — Random Data Removal: ML/ETL/Utils** (Sonnet)
Replace `random.*` calls in ML pipeline, ETL, and utility files.

Files:
- `backend/ml/feature_store.py`
- `backend/ml/load_balancer.py`
- `backend/ml/pipeline/deployment.py`
- `backend/ml/model_manager.py`
- `backend/ml/simple_training_pipeline.py`
- `backend/ml/training_pipeline.py`
- `backend/utils/data_anonymization.py`
- `backend/utils/redis_resilience.py`
- `backend/utils/deadlock_handler.py`
- `backend/utils/db_read_replicas.py`
- `backend/utils/cache_warming.py`
- `backend/utils/resilient_pipeline.py`
- `backend/utils/performance_tester.py`
- `backend/etl/unlimited_extractor_with_fallbacks.py`
- `backend/etl/concurrent_processor.py`
- `backend/etl/web_scrapers.py`
- `backend/etl/distributed_batch_processor.py`
- `backend/etl/multi_source_extractor.py`

NOTE: Some `random` usage in ML/ETL may be legitimate (e.g., random seeds for reproducibility, data augmentation, load balancing). For these cases, replace with seeded random (`random.Random(42)`) rather than removing entirely. Remove only fake/mock data generation.

**Agent 3 — Test Quality Fixes** (Sonnet)
Fix broken test fixtures and quality issues.

Tasks:
1. Fix `mock_current_user` fixture in `backend/tests/conftest.py` (line 363-368)
   - Current: `patch.object(app, 'dependency_overrides')` — doesn't work because it replaces the dict instead of modifying it
   - Fix: Use `app.dependency_overrides[get_current_user] = lambda: test_user` directly, with cleanup in `yield`
   - Check if any tests actually use this fixture; if not, delete it

2. Fix `test_integration_comprehensive.py` skip issue (line 24)
   - Currently skips ALL 14 tests when Docker unavailable
   - Move the `pytest.skip` inside individual tests or use `@pytest.mark.skipif` per test
   - OR: Mark entire module with `@pytest.mark.docker` so it's excluded from normal runs but individually runnable

3. Fix xfail tests if possible:
   - `test_middleware_unit.py:969` — weak ETag comparison bug in response_optimizer.py
   - `test_data_ingestion.py:925` — flaky import-chain mock

Files:
- `backend/tests/conftest.py`
- `backend/tests/test_integration_comprehensive.py`
- `backend/tests/unit/test_middleware_unit.py`
- `backend/tests/unit/test_data_ingestion.py`
- `backend/middleware/response_optimizer.py` (if fixing ETag bug)

**Agent 4 — Missing Service Tests: Batch 1** (Sonnet)
Write unit tests for 4 services with zero coverage.

Files to create:
- `backend/tests/unit/test_settings_service.py`
- `backend/tests/unit/test_market_data_service.py`
- `backend/tests/unit/test_news_service.py`
- `backend/tests/unit/test_portfolio_helpers.py`

Pattern:
- Read the service file first to understand methods
- Mock all external dependencies (DB, Redis, HTTP clients)
- Use `@pytest.mark.asyncio` for async methods
- Use `AsyncMock` for async dependencies
- Test happy path + error handling for each public method
- Target 80%+ coverage per service

### Wave 2: More Service Tests + Frontend (4 parallel agents)

**Agent 5 — Missing Service Tests: Batch 2** (Sonnet)
Write unit tests for remaining 4 services.

Files to create:
- `backend/tests/unit/test_portfolio_rebalancing.py`
- `backend/tests/unit/test_recommendation_analysis.py`
- `backend/tests/unit/test_recommendation_crud.py`
- `backend/tests/unit/test_socketio_service.py`

Same pattern as Agent 4.

**Agent 6 — Frontend Component Tests: Core** (Sonnet)
Write Vitest tests for high-value untested components.

Priority components (by user-facing importance):
- `frontend/web/src/components/Layout/index.tsx`
- `frontend/web/src/components/SearchModal/index.tsx`
- `frontend/web/src/components/NotificationPanel/index.tsx`
- `frontend/web/src/components/WebSocketIndicator/index.tsx`
- `frontend/web/src/components/common/ErrorBoundary.tsx`
- `frontend/web/src/components/common/LoadingSpinner.tsx`
- `frontend/web/src/components/common/PageSkeleton.tsx`

Pattern:
- Use `@testing-library/react` + `vitest`
- Mock Redux store with `@reduxjs/toolkit`
- Mock API calls
- Test rendering, user interactions, error states
- Place test files alongside components: `ComponentName.test.tsx`

**Agent 7 — Frontend Component Tests: Dashboard + Charts** (Sonnet)
Write tests for dashboard and chart components.

Components:
- `frontend/web/src/components/dashboard/DashboardLayout.tsx`
- `frontend/web/src/components/dashboard/MetricCard.tsx`
- `frontend/web/src/components/dashboard/PerformanceSection.tsx`
- `frontend/web/src/components/dashboard/HoldingsTable.tsx`
- `frontend/web/src/components/dashboard/HoldingsActions.tsx`
- `frontend/web/src/components/charts/MarketHeatmap.tsx`
- `frontend/web/src/components/charts/Sparkline.tsx`
- `frontend/web/src/components/charts/StockChart.tsx`

**Agent 8 — Frontend Component Tests: Features** (Sonnet)
Write tests for feature-specific components.

Components:
- `frontend/web/src/components/alerts/AlertForm.tsx`
- `frontend/web/src/components/alerts/AlertsList.tsx`
- `frontend/web/src/components/analysis/AnalysisCharts.tsx`
- `frontend/web/src/components/analysis/AnalysisFilters.tsx`
- `frontend/web/src/components/analysis/AnalysisTable.tsx`
- `frontend/web/src/components/market/MarketCharts.tsx`
- `frontend/web/src/components/market/MarketSummary.tsx`
- `frontend/web/src/components/market/MarketTickers.tsx`
- `frontend/web/src/components/panels/AllocationPanel.tsx`
- `frontend/web/src/components/panels/MarketOverviewPanel.tsx`
- `frontend/web/src/components/panels/NewsFeedPanel.tsx`
- `frontend/web/src/components/panels/RecommendationsPanel.tsx`

### Wave 3: Remaining Frontend + Verification (3 parallel agents)

**Agent 9 — Frontend Component Tests: Remaining** (Sonnet)
Write tests for remaining untested components.

Components:
- `frontend/web/src/components/portfolio/PortfolioActions.tsx`
- `frontend/web/src/components/portfolio/PortfolioChart.tsx`
- `frontend/web/src/components/portfolio/PortfolioTabs.tsx`
- `frontend/web/src/components/portfolio/CorrelationMatrix.tsx`
- `frontend/web/src/components/portfolio/EfficientFrontier.tsx`
- `frontend/web/src/components/portfolio/RiskDecomposition.tsx`
- `frontend/web/src/components/recommendations/RecommendationsFilter.tsx`
- `frontend/web/src/components/recommendations/RecommendationsList.tsx`
- `frontend/web/src/components/settings/SettingsForm.tsx`
- `frontend/web/src/components/settings/SettingsTabs.tsx`
- `frontend/web/src/components/watchlist/WatchlistActions.tsx`
- `frontend/web/src/components/watchlist/WatchlistTable.tsx`
- `frontend/web/src/components/cards/NewsCard.tsx`
- `frontend/web/src/components/cards/RecommendationCard.tsx`
- `frontend/web/src/components/cards/RecommendationCardCompact.tsx`
- `frontend/web/src/components/cards/recommendation/` (if any files)

**Agent 10 — Backend Test Verification** (Sonnet)
Run full backend test suite and fix any failures introduced by Waves 1-2.

Tasks:
1. `python3 -m pytest backend/tests/ -x -q` — verify all pass
2. Fix any failures from random data removal or new test files
3. Check for duplicate test function names across all files
4. Verify no `import random` remains in non-test backend code (except seeded `random.Random(42)`)
5. Run `python3 -m pytest backend/tests/ --co -q | tail -5` — report final test count

**Agent 11 — Frontend Test Verification** (Sonnet)
Run full frontend test suite and fix any failures.

Tasks:
1. `cd frontend/web && npx vitest run --reporter=verbose` — run all tests
2. Fix any failures from new test files
3. `npx tsc --noEmit` — verify TypeScript compiles
4. Report coverage summary

---

## File Disjointness Verification

| Wave | Agent | Files | Overlap |
|------|-------|-------|---------|
| 1 | 1 (Random: Services) | 7 service files + 1 router | None |
| 1 | 2 (Random: ML/ETL/Utils) | 18 ml/etl/utils files | None |
| 1 | 3 (Test Quality) | conftest.py, 3 test files, 1 middleware | None |
| 1 | 4 (Tests: Batch 1) | 4 NEW test files | None |
| 2 | 5 (Tests: Batch 2) | 4 NEW test files | None |
| 2 | 6 (FE Tests: Core) | 7 NEW test files | None |
| 2 | 7 (FE Tests: Dashboard) | 8 NEW test files | None |
| 2 | 8 (FE Tests: Features) | 12 NEW test files | None |
| 3 | 9 (FE Tests: Remaining) | 15+ NEW test files | None |
| 3 | 10 (BE Verification) | Read-only + fixes | After all BE agents |
| 3 | 11 (FE Verification) | Read-only + fixes | After all FE agents |

---

## Execution Instructions

### For the new context window:

```
Execute this plan NOW in 3 waves. Do not write a new plan. Do not enter plan mode.

Wave 1: Spawn 4 agents in background (Agents 1-4) — all Sonnet, all in parallel
Wave 2: After Wave 1 completes and tests pass, spawn 4 agents (Agents 5-8)
Wave 3: After Wave 2 completes, spawn 3 agents (Agents 9-11) for remaining + verification

After each wave:
1. Run `python3 -m pytest backend/tests/ -x -q` to verify backend
2. Fix any failures before proceeding to next wave
3. Commit changes with conventional commit messages

After all waves:
1. Run full backend test suite — target 5100+ tests
2. Run `cd frontend/web && npx vitest run` — target 60+ frontend tests
3. Run `npx tsc --noEmit` — verify TypeScript compiles
4. Stage everything, commit in logical groups, push to main
5. Update MEMORY.md with final counts
```

### Commit Plan

1. `fix: remove random data from services and routers (7 files)`
2. `fix: remove random data from ML/ETL/utils (18 files)`
3. `fix: repair test fixtures and quality issues`
4. `test: add unit tests for settings, market_data, news, portfolio_helpers services`
5. `test: add unit tests for portfolio_rebalancing, recommendation_analysis/crud, socketio services`
6. `test: add Vitest tests for core frontend components (7 components)`
7. `test: add Vitest tests for dashboard and chart components (8 components)`
8. `test: add Vitest tests for feature components (12 components)`
9. `test: add Vitest tests for remaining components (15 components)`

---

## Success Criteria

- [ ] Zero `random.uniform/randint/choice` in non-test backend code (except seeded RNG)
- [ ] Zero `import random` in non-test backend code (except seeded RNG)
- [ ] 8 new service unit test files (settings, market_data, news, portfolio_helpers, portfolio_rebalancing, recommendation_analysis, recommendation_crud, socketio)
- [ ] Backend test count >= 5100
- [ ] Frontend component test files >= 40 (currently 4)
- [ ] Frontend page test files = 9 (already done)
- [ ] `mock_current_user` fixture either fixed or removed
- [ ] `test_integration_comprehensive.py` properly marked (not silently skipping 14 tests)
- [ ] TypeScript compiles (`npx tsc --noEmit`)
- [ ] All changes committed and pushed to main
