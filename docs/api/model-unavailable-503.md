# API Contract — `503 model_unavailable`

**Status:** Stable contract (PRD audit 2026-04 Workstream D, Q4 default
recorded 2026-04-28)
**Owners:** Backend (this PR), Frontend G3-phase-4 (consumer)

## Why this exists

The platform is SEC-regulated. Several endpoints used to return values
sampled from `random.uniform()` / `np.random.*` / `Dummy*` model
fallbacks when the underlying ML model binaries were missing or the
feature data was insufficient. Those endpoints now refuse to serve a
fabricated response and emit a structured `503 Service Unavailable`
instead.

Per the legal assumption of record
(`docs/audits/2026-04/_synthesis/_meta/LEGAL_ASSUMPTION_OF_RECORD.md`,
Q5=B1 working assumption: the platform surfaces analytics and research
rather than personalized investment advice triggering fiduciary duty),
the 503-rather-than-fake-output choice is the SEC-conservative posture.

## Response shape

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "error": "model_unavailable",
  "model": "<name>",
  "reason": "<reason-code>",
  "request_id": "<uuid hex>"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `error` | string | Always the literal `"model_unavailable"` for this contract. |
| `model` | string | The logical model / metric whose unavailability triggered the 503. Examples: `recommendation_engine`, `recommendation_backtest`, `recommendation_performance`, `recommendation_alerts`, `risk_metrics`, `portfolio_performance`, `rsi_14d` (drift), generic `unknown`. |
| `reason` | string | Stable reason code, see below. |
| `request_id` | string | Hex UUID. Echoed in the Sentry breadcrumb tagged `model_unavailable` — useful for cross-referencing logs. |

### Reason codes

| Code | Meaning |
|------|---------|
| `binary_missing` | An ML model binary (`.pkl` / `.pth`) is not present in `ml_models/` and HuggingFace Hub fallback is disabled or empty. |
| `fallback_active` | `ModelManager` is using a `DummyLSTM` / `DummyXGBoost` / `DummyProphet` substitute. |
| `insufficient_data` | The downstream computation requires more observations than were available (e.g. risk metrics need ≥ 30 closes). |
| `not_implemented` | Production refusal where the legacy code path was synthetic-only and a real implementation is owned by another workstream (G2a). |
| `manager_unavailable` | `get_model_manager()` itself failed to instantiate. |
| `live_feed_not_configured` | Socket.IO `price_unavailable` event payload — no real-time market data wired up. |

## Endpoints that can return 503 model_unavailable

(Non-exhaustive — handler is registered globally so any service raising
`ModelUnavailableError` / `InsufficientDataError` produces this shape.)

- `POST /api/recommendations/backtest`
- `GET /api/recommendations/performance/track`
- `GET /api/recommendations/alerts/history`
- `POST /api/analysis/analyze` (when price history < 30 closes)
- `GET /api/portfolio/{id}/performance` (when models in fallback)

## Health gate

Inspect `GET /health` and `GET /readiness` for `fallback_models` (list)
and `fallback_models_count` (int). Empty list == healthy production.
Readiness probe fails when count > 0 so K8s removes the pod from the
load balancer.

## Sentry observability

Every 503 emits a Sentry breadcrumb:

```
category=model_unavailable
level=warning
message=503 model_unavailable: <model> (<reason>)
data={ model, reason, request_id, endpoint }
```

## Frontend implementation

Frontend G3-phase-4 ships the empty-state component that renders this
response as "No recommendation available — model retraining in
progress." This PR (Workstream D) does NOT ship the empty-state.

## References

- PRD audit 2026-04: `docs/audits/2026-04/PRD-for-loki.md` §3 D, §2 Q4
- Workpaper: `docs/audits/2026-04/_synthesis/workpaper/D.md`
- Legal: `docs/audits/2026-04/_synthesis/_meta/LEGAL_ASSUMPTION_OF_RECORD.md`
- Architecture: `docs/architecture/realtime-transport.md`
- Findings: F-02-003, F-02-018, F-03-003, F-03-005
