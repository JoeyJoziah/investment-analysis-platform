"""
Structured 503 error response helpers for model_unavailable / insufficient_data.

Per PRD audit 2026-04 Workstream D (Q4=default): when an ML model is in
``DummyLSTM``/``DummyXGBoost``/``DummyProphet`` fallback or feature data is
insufficient, the API MUST return HTTP 503 with a structured payload rather
than fabricating values. The frontend (G3 phase 4) consumes this contract.

Policy artifact:
    docs/audits/2026-04/_synthesis/_meta/LEGAL_ASSUMPTION_OF_RECORD.md
PRD reference:
    docs/audits/2026-04/PRD-for-loki.md §3 D
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status

from backend.exceptions import InsufficientDataError, ModelUnavailableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAPI snippet — attach via ``responses={**MODEL_UNAVAILABLE_503_RESPONSE}``
# on any FastAPI route that can refuse-to-serve. Frontend G3-phase-4 consumes
# this exact shape to render the empty-state component.
# ---------------------------------------------------------------------------

MODEL_UNAVAILABLE_503_EXAMPLE: Dict[str, Any] = {
    "error": "model_unavailable",
    "model": "recommendation_engine",
    "reason": "fallback_active",
    "request_id": "9b4f2e1c8a1d4e5b9d4f1c2a3e5d6f78",
}

MODEL_UNAVAILABLE_503_RESPONSE: Dict[int, Dict[str, Any]] = {
    503: {
        "description": (
            "Model unavailable — the platform refuses to fabricate "
            "investment outputs when the underlying ML model binaries are "
            "missing or the feature data is insufficient. See "
            "docs/api/model-unavailable-503.md."
        ),
        "content": {
            "application/json": {
                "example": MODEL_UNAVAILABLE_503_EXAMPLE,
                "schema": {
                    "type": "object",
                    "required": ["error", "model", "reason", "request_id"],
                    "properties": {
                        "error": {
                            "type": "string",
                            "enum": ["model_unavailable"],
                        },
                        "model": {"type": "string"},
                        "reason": {
                            "type": "string",
                            "enum": [
                                "binary_missing",
                                "fallback_active",
                                "insufficient_data",
                                "not_implemented",
                                "manager_unavailable",
                                "live_feed_not_configured",
                            ],
                        },
                        "request_id": {"type": "string"},
                    },
                },
            }
        },
    }
}

try:  # pragma: no cover - optional dependency surface
    import sentry_sdk
except Exception:  # pragma: no cover
    sentry_sdk = None  # type: ignore[assignment]


def _request_id(request: Optional[Request]) -> str:
    if request is not None:
        rid = request.headers.get("x-request-id") if hasattr(request, "headers") else None
        if rid:
            return rid
    return uuid.uuid4().hex


def _sentry_breadcrumb(category: str, message: str, data: Dict[str, Any]) -> None:
    if sentry_sdk is None:
        return
    try:
        sentry_sdk.add_breadcrumb(
            category=category,
            message=message,
            level="warning",
            data=data,
        )
    except Exception as exc:  # pragma: no cover - never let telemetry crash the request
        logger.debug("sentry breadcrumb failed: %s", exc)


def model_unavailable_payload(
    model: str,
    reason: str = "fallback_active",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the structured 503 body for a ``model_unavailable`` response.

    Stable shape (frontend contract):

    .. code-block:: json

        {
          "error": "model_unavailable",
          "model": "<name>",
          "reason": "binary_missing | insufficient_data | fallback_active",
          "request_id": "<uuid hex>"
        }
    """
    return {
        "error": "model_unavailable",
        "model": model,
        "reason": reason,
        "request_id": request_id or uuid.uuid4().hex,
    }


def raise_model_unavailable(
    model: str = "unknown",
    reason: str = "fallback_active",
    request: Optional[Request] = None,
) -> None:
    """Raise an HTTP 503 with the canonical ``model_unavailable`` payload.

    Logs a Sentry breadcrumb tagged ``model_unavailable`` so the model name
    and originating endpoint are visible to SRE.
    """
    rid = _request_id(request)
    _sentry_breadcrumb(
        "model_unavailable",
        f"503 model_unavailable: {model} ({reason})",
        {
            "model": model,
            "reason": reason,
            "request_id": rid,
            "endpoint": getattr(getattr(request, "url", None), "path", None),
        },
    )
    logger.warning(
        "model_unavailable model=%s reason=%s request_id=%s",
        model, reason, rid,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=model_unavailable_payload(model, reason, rid),
    )


def handle_unavailable(
    exc: Exception,
    request: Optional[Request] = None,
    default_model: str = "unknown",
) -> None:
    """Convert an ``InsufficientDataError``/``ModelUnavailableError`` to 503.

    Re-raises the original exception untouched if it is neither type, so this
    helper is safe to call from a broad ``except`` clause.
    """
    if isinstance(exc, ModelUnavailableError):
        raise_model_unavailable(
            model=exc.model,
            reason=exc.reason,
            request=request,
        )
    if isinstance(exc, InsufficientDataError):
        raise_model_unavailable(
            model=default_model,
            reason=exc.reason or "insufficient_data",
            request=request,
        )
    raise exc
