# 2026-06 Consensus-Wave Sweep Findings

**Version: 1.0.0** · **Last Updated: 2026-06-17**

Durable in-repo record of the follow-up findings surfaced during the 2026-06
consensus-wave execution. Canonical tracker: **issue #242**.

## Open (needs disposition)

### A. Financial-data integrity / fabrication (#200 theme)
1. `backend/services/recommendation_crud.py` — `RecommendationCrudMixin` (~330 lines) is dead `random.uniform`/`random.choice` fabricator code, not imported by any live router. Delete.
2. `backend/api/routers/admin.py:590` — `execution_time_ms = random.randint(100, 5000)` served (admin-only, non-financial). Use real `time.perf_counter()`.
3. `backend/ml/backtesting.py` — bare `BacktestEngine()` (no provider) still synthesizes `np.random` prices in `_get_market_data`/`_get_benchmark_data`. Make the no-provider path fail-loud (the #231 provider path already is).
4. `backend/ml/backtesting.py` `_calculate_monthly_returns` — crashes on a single-month backtest window (column-length mismatch).

### B. Alembic migration drift — fresh-DB deploy blockers (#216 theme)
5. `001` `idx_stocks_active` → `is_tradeable` typo (model is `is_tradable`).
6. `001` `op.create_index` collides with model `__table_args__` indexes under `create_all` (no `IF NOT EXISTS`).
7. `008` `idx_recommendations_confidence_desc` — volatile `WHERE valid_until > CURRENT_TIMESTAMP` predicate (must be IMMUTABLE).
8. `008` references `stocks(sector)` but the model column is `sector_id`.

### C. Static-analysis hygiene
9. `backend/etl/data_loader.py` — semgrep `avoid-sqlalchemy-text` false positive (allowlist-only). The Semgrep platform app ignores inline `# nosemgrep`, so it blocks strict-green merge of #207/#235. Fix via SQLAlchemy-core refactor (plan item **B8**); greening it unblocks #207/#235.

## Already fixed this session (context)
- Migration `confidence_score`→`confidence` (001×2, 008×1) + volatile `CURRENT_DATE` predicate (008) → **#234**.
- Credential-rotation runbook had tabulated the live PG/Redis passwords in plaintext → redacted in **#232** before merge.
