# Real-time Push Transport — Architecture Decision

**Status:** Documented (PRD audit 2026-04 Workstream D, F-02-015)
**Owner:** Backend
**Last reviewed:** 2026-04-28

## Context

The platform historically shipped two real-time push transports in
parallel:

1. `backend/services/websocket_service.py::EnhancedConnectionManager` —
   raw WebSocket, mounted via `backend/api/routers/websocket.py`.
2. `backend/services/socketio_service.py::sio` — `python-socketio`
   `AsyncServer`, mounted alongside FastAPI as an ASGI sub-app.

Both code paths broadcast price/portfolio/alert events with their own
subscription model. Clients must pick one URL; there is no integration
plan and no architectural reason to keep both. Audit 2026-04 (F-02-015)
flagged this as a maintenance burden + a silent-regression hazard
(fixing random-data emitters in one transport while the other still
ships fabricated values).

## Decision

**Standardize on Socket.IO** (`backend/services/socketio_service.py`).

### Why Socket.IO

- Friendlier behind corporate proxies / load balancers — falls back
  through long-polling when WebSocket is blocked.
- Built-in reconnection / heartbeat — fewer client-side workarounds.
- Room-based subscription model already in use for prices, portfolio,
  alerts.
- Frontend `socket.io-client` is a more common dependency than a custom
  WebSocket wrapper for the React/RN clients.

### Why not raw WebSocket

- No automatic reconnection or fallback.
- No standardized message envelope — every consumer rebuilds its own.
- LB sticky-session config required anyway (same as Socket.IO long-polling).
- Maintaining the two side-by-side multiplied the audit-2026-04
  random-data emitter footprint (F-02-003) for no benefit.

## What changes (workstream-D scope)

This workpaper documents the choice. Removing
`backend/services/websocket_service.py` + the
`backend/api/routers/websocket.py` mount + the frontend raw-WebSocket
client touches scope 12 (frontend) and is sequenced for
**G3-phase-4 / -phase-5**, not D. Until that lands:

- Both transports continue to mount, but only Socket.IO emits real /
  contract-stable payloads.
- `_stream_price_updates` in `socketio_service.py` is gated per F-02-003
  (emit `price_unavailable` in production, fabricate only behind
  `DEMO_MODE`).
- Any new real-time feature MUST use Socket.IO. New raw-WebSocket
  handlers are a code-review block.

## Frontend contract for the 503 / unavailable empty-state

When `DEMO_MODE=false` and live feeds aren't wired up:

- HTTP endpoints (recommendations / backtest / alerts / risk-metrics):
  return `503 Service Unavailable` with body
  ```json
  {
    "error": "model_unavailable",
    "model": "<name>",
    "reason": "binary_missing | insufficient_data | fallback_active | not_implemented | live_feed_not_configured | manager_unavailable",
    "request_id": "<uuid hex>"
  }
  ```
  Frontend should render the empty-state with the user-visible copy
  "No recommendation available — model retraining in progress" (G3
  phase 4).

- Socket.IO price stream: emits a single
  ```json
  {
    "symbol": "...",
    "error": "model_unavailable",
    "reason": "live_feed_not_configured",
    "timestamp": "..."
  }
  ```
  on the `price_unavailable` event, then closes the stream. Frontend
  should render the same empty-state and not auto-resubscribe in a
  tight loop.

## References

- PRD audit 2026-04 §3 D — `docs/audits/2026-04/PRD-for-loki.md`
- Workpaper — `docs/audits/2026-04/_synthesis/workpaper/D.md`
- Legal assumption (analytics, not investment advice) —
  `docs/audits/2026-04/_synthesis/_meta/LEGAL_ASSUMPTION_OF_RECORD.md`
- Findings: F-02-003 (random-data emitters), F-02-015 (transport choice).
