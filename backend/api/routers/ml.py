"""
ML Predictions Router - Issue #3: ML Predictions Endpoint (P0)

Provides POST /predictions endpoint for stock price forecasting using
LSTM, XGBoost, Prophet, or ensemble models. Integrates with the existing
ModelManager and ML pipeline infrastructure with Redis-backed caching.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone, date
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.oauth2 import get_current_user
from backend.config.database import get_async_db_session
from backend.config.settings import settings
from backend.ml.model_manager import get_model_manager, ModelManager
from backend.models.api_response import ApiResponse, success_response
from backend.models.unified_models import User
from backend.services.realtime_price_service import get_realtime_price_service
from backend.utils.cache import get_redis, CacheManager
from backend.utils.numpy_serializer import sanitize_numpy

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ml"])

# Redis cache manager for ML predictions (15 min TTL)
_ml_cache = CacheManager(prefix="ml:predictions")
ML_PREDICTION_CACHE_TTL = 900  # 15 minutes in seconds


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MLModelType(str, Enum):
    LSTM = "lstm"
    XGBOOST = "xgboost"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class MLPredictionRequest(BaseModel):
    """Request body for ML price predictions."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    model: MLModelType = Field(
        ...,
        description="ML model to use for prediction",
    )
    horizon_days: int = Field(
        ...,
        ge=1,
        le=365,
        description="Number of days to forecast (1-365)",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.isalpha():
            raise ValueError("Ticker must contain only alphabetic characters")
        return cleaned


class PredictionPoint(BaseModel):
    """A single predicted data-point with confidence interval."""

    date: str = Field(..., description="Prediction date (ISO 8601)")
    price: float = Field(..., description="Predicted closing price")
    confidence_lower: float = Field(
        ..., description="Lower bound of the 95% confidence interval"
    )
    confidence_upper: float = Field(
        ..., description="Upper bound of the 95% confidence interval"
    )


class AccuracyMetrics(BaseModel):
    """Accuracy metrics for the model used in prediction."""

    model_status: str = Field(
        ..., description="Whether a trained model was loaded or a fallback was used"
    )
    historical_rmse: Optional[float] = Field(
        None, description="Root Mean Squared Error on historical validation set"
    )
    historical_mae: Optional[float] = Field(
        None, description="Mean Absolute Error on historical validation set"
    )
    directional_accuracy: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Fraction of correct directional predictions on validation set",
    )
    training_date: Optional[str] = Field(
        None, description="ISO 8601 date when the model was last trained"
    )


class MLPredictionResponse(BaseModel):
    """Response body for ML price predictions."""

    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    ticker: str = Field(..., description="Stock ticker symbol")
    predictions: List[PredictionPoint] = Field(
        ..., description="List of predicted data-points"
    )
    model_used: str = Field(..., description="Model that generated the predictions")
    accuracy_metrics: AccuracyMetrics = Field(
        ..., description="Accuracy metrics for the model"
    )
    generated_at: str = Field(
        ..., description="Timestamp when the prediction was generated (ISO 8601)"
    )
    horizon_days: int = Field(..., description="Number of forecast days")
    cached: bool = Field(
        False, description="Whether this response was served from cache"
    )


# ---------------------------------------------------------------------------
# Internal model name mapping
# ---------------------------------------------------------------------------

_MODEL_NAME_MAP: Dict[MLModelType, str] = {
    MLModelType.LSTM: "lstm_price_predictor",
    MLModelType.XGBOOST: "xgboost_classifier",
    MLModelType.PROPHET: "prophet_forecaster",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_cache_key(ticker: str, model: str, horizon_days: int) -> str:
    """Build a deterministic cache key for the prediction request."""
    raw = f"{ticker}:{model}:{horizon_days}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{ticker}:{model}:h{horizon_days}:{digest}"


def _generate_prediction_points(
    base_price: float,
    raw_predictions: Any,
    horizon_days: int,
    model_type: MLModelType,
) -> List[PredictionPoint]:
    """
    Convert raw model output into a list of PredictionPoint objects.

    Confidence intervals are computed as +/- a percentage of the predicted
    price that grows with forecast distance (wider intervals further out).
    """
    points: List[PredictionPoint] = []
    now = datetime.now(timezone.utc)

    # Normalize raw predictions into a list of floats
    predicted_prices: List[float] = []

    # First sanitize numpy types in raw predictions
    raw_predictions = sanitize_numpy(raw_predictions)

    if isinstance(raw_predictions, dict):
        # Prophet returns list of dicts with yhat / yhat_lower / yhat_upper
        if isinstance(raw_predictions, list):
            for entry in raw_predictions:
                predicted_prices.append(float(entry.get("yhat", base_price)))
        else:
            # Single dict prediction
            value = raw_predictions.get("price", raw_predictions.get("forecast", base_price))
            predicted_prices.append(float(value))
    elif isinstance(raw_predictions, (list, tuple)):
        if raw_predictions and isinstance(raw_predictions[0], dict):
            # Prophet-style list of dicts
            for entry in raw_predictions:
                predicted_prices.append(float(entry.get("yhat", base_price)))
        else:
            # Numeric list/array (already converted from numpy by sanitize_numpy)
            for val in raw_predictions:
                try:
                    predicted_prices.append(float(val))
                except (TypeError, ValueError):
                    predicted_prices.append(base_price)
    else:
        try:
            predicted_prices.append(float(raw_predictions))
        except (TypeError, ValueError):
            predicted_prices.append(base_price)

    # Pad or truncate to match horizon_days
    if len(predicted_prices) < horizon_days:
        last = predicted_prices[-1] if predicted_prices else base_price
        predicted_prices.extend([last] * (horizon_days - len(predicted_prices)))
    predicted_prices = predicted_prices[:horizon_days]

    for day_offset, price in enumerate(predicted_prices, start=1):
        forecast_date = now + timedelta(days=day_offset)
        # Skip weekends for market dates
        while forecast_date.weekday() >= 5:
            forecast_date += timedelta(days=1)

        # Confidence interval widens with horizon (2% base + 0.5% per day)
        ci_pct = 0.02 + 0.005 * day_offset
        lower = round(price * (1 - ci_pct), 2)
        upper = round(price * (1 + ci_pct), 2)

        points.append(
            PredictionPoint(
                date=forecast_date.strftime("%Y-%m-%d"),
                price=round(price, 2),
                confidence_lower=lower,
                confidence_upper=upper,
            )
        )

    return points


def _build_accuracy_metrics(
    model_manager: ModelManager, model_key: str
) -> AccuracyMetrics:
    """Extract accuracy metrics from model metadata."""
    metadata = model_manager.model_metadata.get(model_key, {})
    model_status = metadata.get("status", "unknown")

    training_date = None
    loaded_at = metadata.get("loaded_at")
    if loaded_at and isinstance(loaded_at, datetime):
        training_date = loaded_at.isoformat()

    # If we have a real model loaded, provide baseline metrics from metadata
    historical_rmse = None
    historical_mae = None
    directional_accuracy = None

    if model_status == "loaded":
        # Extract from model bundle config if present
        model_bundle = model_manager.get_model(model_key)
        if isinstance(model_bundle, dict):
            config = model_bundle.get("config", {})
            historical_rmse = config.get("validation_rmse")
            historical_mae = config.get("validation_mae")
            directional_accuracy = config.get("directional_accuracy")

    return AccuracyMetrics(
        model_status=model_status,
        historical_rmse=historical_rmse,
        historical_mae=historical_mae,
        directional_accuracy=directional_accuracy,
        training_date=training_date,
    )


async def _run_single_model_prediction(
    model_manager: ModelManager,
    model_type: MLModelType,
    ticker: str,
    horizon_days: int,
    base_price: float,
    feature_data: Optional[Any] = None,
) -> tuple[Any, str]:
    """
    Run prediction for a single model type.

    Args:
        feature_data: Real market/feature data for the model. For LSTM, a
            pre-shaped numpy array of shape (1, sequence_length, n_features).
            For XGBoost, a 2D array of shape (horizon_days, n_features).
            If None (and BOOTSTRAP_MODELS env flag is NOT set), a 503 is raised.

    Returns:
        (raw_predictions, internal_model_key)

    Raises:
        ValueError: Model is unavailable.
        HTTPException 503: Required feature data was not supplied and
            BOOTSTRAP_MODELS is not enabled.
    """
    import os as _os

    model_key = _MODEL_NAME_MAP[model_type]
    model = model_manager.get_model(model_key)

    if model is None:
        raise ValueError(f"Model '{model_type.value}' is not available")

    # T1.3 (D1): refuse to serve a fallback (Dummy*) model — its predict()
    # returns np.random. Raises ModelUnavailableError -> HTTP 503 instead of
    # shipping fabricated LSTM/XGBoost/Prophet outputs to users.
    model_manager.assert_real_model(model_key)

    bootstrap_mode = _os.environ.get("BOOTSTRAP_MODELS", "").lower() in ("1", "true", "yes")

    # Prepare input data appropriate for each model type
    if model_type == MLModelType.LSTM:
        # LSTM expects a 3D tensor: (batch, sequence_length, features)
        if feature_data is not None:
            input_data = feature_data
        elif bootstrap_mode:
            # Bootstrap/test-only synthetic data — NEVER reached in production
            logger.warning(
                "BOOTSTRAP_MODELS is set: using synthetic LSTM input for %s — "
                "DO NOT run in production.",
                ticker,
            )
            input_data = np.zeros((1, 30, 5), dtype=np.float32)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"LSTM prediction for '{ticker}' requires real market feature data "
                    "which is not yet available. Refusing to substitute synthetic data."
                ),
            )
        raw = model_manager.predict(model_key, input_data)

    elif model_type == MLModelType.XGBOOST:
        if feature_data is not None:
            input_data = feature_data
        elif bootstrap_mode:
            logger.warning(
                "BOOTSTRAP_MODELS is set: using zero XGBoost input for %s — "
                "DO NOT run in production.",
                ticker,
            )
            input_data = np.zeros((horizon_days, 20), dtype=np.float32)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"XGBoost prediction for '{ticker}' requires real market feature data "
                    "which is not yet available. Refusing to substitute synthetic data."
                ),
            )
        raw = model_manager.predict(model_key, input_data)
        if isinstance(raw, dict) and "predictions" in raw:
            raw = raw["predictions"]

    elif model_type == MLModelType.PROPHET:
        import pandas as pd

        future_dates = [
            datetime.now(timezone.utc) + timedelta(days=i)
            for i in range(1, horizon_days + 1)
        ]
        raw = model_manager.predict(
            model_key,
            {"ticker": ticker, "df": pd.DataFrame({"ds": future_dates})},
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type.value}")

    return raw, model_key


async def _run_ensemble_prediction(
    model_manager: ModelManager,
    ticker: str,
    horizon_days: int,
    base_price: float,
    db: AsyncSession,
) -> tuple[List[PredictionPoint], AccuracyMetrics]:
    """
    Run predictions from all available models and average their outputs.

    Each model is fed real, windowed feature data engineered from the ticker's
    price history (#208 item 1).  Models with no usable feature data fail loud
    (503) inside ``_run_single_model_prediction`` and are skipped from the
    ensemble — they are never fed synthetic data.
    """
    all_points: Dict[int, List[float]] = {}  # day_offset -> list of prices
    metrics_sources: List[AccuracyMetrics] = []
    models_used: List[str] = []

    for mtype in [MLModelType.LSTM, MLModelType.XGBOOST, MLModelType.PROPHET]:
        try:
            feature_data = await _fetch_feature_data(
                model_manager, mtype, ticker, horizon_days, db
            )
            raw, model_key = await _run_single_model_prediction(
                model_manager, mtype, ticker, horizon_days, base_price,
                feature_data=feature_data,
            )
            points = _generate_prediction_points(base_price, raw, horizon_days, mtype)
            for i, pt in enumerate(points):
                all_points.setdefault(i, []).append(pt.price)
            metrics_sources.append(_build_accuracy_metrics(model_manager, model_key))
            models_used.append(mtype.value)
        except Exception as e:
            logger.warning(f"Ensemble: model {mtype.value} skipped: {e}")

    if not all_points:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No ML models available for ensemble prediction",
        )

    # Average predictions across models
    now = datetime.now(timezone.utc)
    ensemble_points: List[PredictionPoint] = []
    for day_offset in sorted(all_points.keys()):
        prices = all_points[day_offset]
        # Convert numpy types to native Python before calculations
        avg_price = round(float(sanitize_numpy(np.mean(prices))), 2)
        forecast_date = now + timedelta(days=day_offset + 1)
        while forecast_date.weekday() >= 5:
            forecast_date += timedelta(days=1)

        ci_pct = 0.02 + 0.005 * (day_offset + 1)
        # Widen CI for ensemble to account for model disagreement
        spread = float(sanitize_numpy(np.std(prices))) if len(prices) > 1 else 0.0
        ci_pct += spread / avg_price if avg_price > 0 else 0.0

        ensemble_points.append(
            PredictionPoint(
                date=forecast_date.strftime("%Y-%m-%d"),
                price=avg_price,
                confidence_lower=round(avg_price * (1 - ci_pct), 2),
                confidence_upper=round(avg_price * (1 + ci_pct), 2),
            )
        )

    # Build ensemble accuracy metrics (average of available)
    loaded_count = sum(1 for m in metrics_sources if m.model_status == "loaded")
    ensemble_metrics = AccuracyMetrics(
        model_status=f"ensemble ({len(models_used)} models: {', '.join(models_used)})",
        historical_rmse=None,
        historical_mae=None,
        directional_accuracy=None,
        training_date=None,
    )

    return ensemble_points, ensemble_metrics


# ---------------------------------------------------------------------------
# Real market data
# ---------------------------------------------------------------------------

async def _fetch_base_price(ticker: str, db: AsyncSession) -> Optional[float]:
    """Fetch the latest *real* market price for ``ticker``.

    Routes through the realtime price service, which resolves in priority order:
    in-memory cache -> Redis -> database fallback.  Returns ``None`` when no real
    price is available so callers keep failing loud (#200) instead of scaling
    confidence intervals off a fabricated constant.

    This resolves the ``base_price`` half of the #200 follow-up (#208 item 1).
    Real ``feature_data`` tensors are wired separately via ``_fetch_feature_data``.
    """
    try:
        service = await get_realtime_price_service()
        update = await service.get_latest_price(ticker, db)
    except Exception as exc:  # a quote lookup must never break prediction flow
        logger.warning(f"base_price lookup failed for {ticker} (non-fatal): {exc}")
        return None

    price = getattr(update, "price", None) if update is not None else None
    if not price:
        return None
    return float(price)


# How many days of price history to pull when engineering features.  Must comfortably
# exceed the LSTM sequence_length (60) plus the longest indicator lookback (sma_200).
_FEATURE_HISTORY_DAYS = 400


def _model_config(model_manager: ModelManager, model_key: str) -> Optional[Dict[str, Any]]:
    """Return the persisted ``config`` dict for a loaded model bundle, if any."""
    bundle = model_manager.get_model(model_key)
    if isinstance(bundle, dict):
        cfg = bundle.get("config")
        if isinstance(cfg, dict):
            return cfg
    return None


async def _fetch_feature_data(
    model_manager: ModelManager,
    model_type: MLModelType,
    ticker: str,
    horizon_days: int,
    db: AsyncSession,
) -> Optional[Any]:
    """Build a real, model-shaped feature tensor from windowed price history.

    Pulls the most recent ``_FEATURE_HISTORY_DAYS`` of OHLCV history for
    ``ticker`` via the price-history repository and engineers it into the exact
    input contract each model expects (LSTM 3D tensor / XGBoost 2D matrix),
    aligned to the model bundle's persisted ``feature_columns``.

    Returns ``None`` (so the caller fails loud with a 503 — #200) when:
      * the model type does not consume a feature tensor (Prophet), or
      * there is genuinely no price history for the ticker.

    Never fabricates prices: a missing history yields ``None``, not synthetic data.
    """
    # Prophet consumes future dates, not a feature tensor — nothing to fetch.
    if model_type not in (MLModelType.LSTM, MLModelType.XGBOOST):
        return None

    model_key = _MODEL_NAME_MAP[model_type]
    config = _model_config(model_manager, model_key)
    if not config or not config.get("feature_columns"):
        # No persisted contract (e.g. fallback/dummy model) → cannot build a real
        # tensor; fail loud upstream rather than guess a shape.
        logger.warning(
            "No feature_columns config for %s/%s; cannot build real feature data",
            ticker, model_type.value,
        )
        return None

    try:
        from backend.repositories.price_repository import price_repository

        records = await price_repository.get_price_history(
            ticker, limit=_FEATURE_HISTORY_DAYS, session=db
        )
    except Exception as exc:  # repository/DB failure must not fabricate data
        logger.warning(f"Feature history lookup failed for {ticker} (non-fatal): {exc}")
        return None

    if not records:
        # Genuinely no price history → fail loud (#200), do not synthesise.
        return None

    from backend.ml.feature_engineering import (
        InsufficientHistoryError,
        build_lstm_features,
        build_xgboost_features,
    )

    try:
        if model_type == MLModelType.LSTM:
            return build_lstm_features(records, config)
        # XGBoost: one feature row per forecast day to match the (horizon, n) shape.
        return build_xgboost_features(records, config, n_rows=max(1, horizon_days))
    except InsufficientHistoryError:
        return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/predictions",
    response_model=ApiResponse[MLPredictionResponse],
    status_code=status.HTTP_200_OK,
    summary="Generate ML price predictions",
    description=(
        "Generate price predictions for a given ticker using the specified ML "
        "model (LSTM, XGBoost, Prophet, or ensemble). Responses are cached in "
        "Redis for 15 minutes. Confidence intervals widen with forecast horizon."
    ),
)
async def create_prediction(
    request_body: MLPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[MLPredictionResponse]:
    """
    POST /predictions -- Generate ML price predictions.

    Accepts a ticker, model type, and forecast horizon. Returns a list of
    date/price prediction points with 95% confidence intervals and accuracy
    metrics for the model used.
    """
    ticker = request_body.ticker
    model_type = request_body.model
    horizon_days = request_body.horizon_days

    # ------------------------------------------------------------------
    # 1. Check Redis cache
    # ------------------------------------------------------------------
    cache_key = _build_cache_key(ticker, model_type.value, horizon_days)
    try:
        cached = await _ml_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"ML prediction cache HIT: {cache_key}")
            # Reconstruct response from cached dict, mark as cached
            if isinstance(cached, dict):
                cached["cached"] = True
                return success_response(data=MLPredictionResponse(**cached))
    except Exception as exc:
        logger.warning(f"Redis cache lookup failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # 2. Load model manager
    # ------------------------------------------------------------------
    try:
        model_manager = get_model_manager()
    except Exception as exc:
        logger.error(f"Failed to load ML model manager: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model service is temporarily unavailable",
        )

    # ------------------------------------------------------------------
    # 3. Run prediction
    # ------------------------------------------------------------------
    prediction_id = f"pred-{uuid.uuid4().hex[:12]}"
    generated_at = datetime.now(timezone.utc)

    # base_price must come from real market data.  It is used only to scale
    # confidence intervals, so a None is safe when the model returns absolute
    # prices.  If not available we refuse to substitute a hardcoded constant.
    # (#208 item 1) Wired to the realtime price service; stays None — and
    # therefore fail-loud downstream — when no real quote is available.
    base_price: Optional[float] = await _fetch_base_price(ticker, db)

    try:
        if model_type == MLModelType.ENSEMBLE:
            points, accuracy = await _run_ensemble_prediction(
                model_manager, ticker, horizon_days, base_price, db
            )
            model_used = "ensemble"
        else:
            # Engineer real feature data from windowed price history.  When no
            # history exists this stays None and _run_single_model_prediction
            # raises 503 (unless BOOTSTRAP_MODELS is set) — we never synthesise.
            feature_data = await _fetch_feature_data(
                model_manager, model_type, ticker, horizon_days, db
            )
            raw, model_key = await _run_single_model_prediction(
                model_manager, model_type, ticker, horizon_days, base_price,
                feature_data=feature_data,
            )
            points = _generate_prediction_points(base_price or 0.0, raw, horizon_days, model_type)
            accuracy = _build_accuracy_metrics(model_manager, model_key)
            model_used = model_type.value
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction failed for {ticker}/{model_type.value}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    # ------------------------------------------------------------------
    # 4. Build response
    # ------------------------------------------------------------------
    response_data = MLPredictionResponse(
        prediction_id=prediction_id,
        ticker=ticker,
        predictions=points,
        model_used=model_used,
        accuracy_metrics=accuracy,
        generated_at=generated_at.isoformat(),
        horizon_days=horizon_days,
        cached=False,
    )

    # ------------------------------------------------------------------
    # 5. Store in Redis cache (15 min TTL)
    # ------------------------------------------------------------------
    try:
        # Sanitize any remaining numpy types before caching
        cache_data = sanitize_numpy(response_data.model_dump(mode="json"))
        await _ml_cache.set(
            cache_key,
            cache_data,
            ttl=ML_PREDICTION_CACHE_TTL,
        )
        logger.debug(f"ML prediction cached: {cache_key} (TTL={ML_PREDICTION_CACHE_TTL}s)")
    except Exception as exc:
        logger.warning(f"Failed to cache ML prediction (non-fatal): {exc}")

    return success_response(data=response_data)


@router.get(
    "/models",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Get ML model status",
    description="Returns the health and status of all loaded ML models.",
)
async def get_model_status(
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """GET /models -- Return status of all ML models."""
    try:
        model_manager = get_model_manager()
        health = model_manager.health_check()
        return success_response(data=health)
    except Exception as exc:
        logger.error(f"Model health check failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model service is temporarily unavailable",
        )


# ---------------------------------------------------------------------------
# Drift Detection Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/drift/detect",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Run drift detection on a model",
)
async def detect_drift(
    model_name: str,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """
    Run data drift detection for a specific model against its reference
    distribution. Returns PSI scores and drift flags per feature.
    """
    try:
        from backend.ml.drift_detection import DriftDetector
        detector = DriftDetector()
        # If no reference distribution exists, return informational status
        if model_name not in detector.reference_distributions:
            return success_response(data={
                "model_name": model_name,
                "status": "no_reference",
                "message": "No reference distribution set. Call PUT /drift/reference first.",
            })
        result = detector.detect_data_drift(model_name, {})
        return success_response(data={
            "model_name": model_name,
            "drift_detected": getattr(result, "drift_detected", False),
            "details": getattr(result, "details", {}),
        })
    except Exception as exc:
        logger.error(f"Drift detection failed for {model_name}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {exc}",
        )


@router.get(
    "/drift/status",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Get drift detection status for all models",
)
async def get_drift_status(
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Return which models have reference distributions configured."""
    try:
        from backend.ml.drift_detection import DriftDetector
        detector = DriftDetector()
        models_with_ref = list(detector.reference_distributions.keys())
        return success_response(data={
            "models_with_reference": models_with_ref,
            "total": len(models_with_ref),
        })
    except Exception as exc:
        logger.error(f"Drift status check failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detection service unavailable",
        )


# ---------------------------------------------------------------------------
# Model Version Management Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/versions",
    response_model=ApiResponse[Dict[str, Any]],
    summary="List all registered model versions",
)
async def list_model_versions(
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Return all model versions from the model registry."""
    try:
        from backend.ml.model_versioning import get_model_version_manager
        manager = get_model_version_manager()
        registry = {}
        for model_name, versions in manager.model_registry.items():
            registry[model_name] = {
                v: {
                    "stage": ver.stage.value if hasattr(ver.stage, "value") else str(ver.stage),
                    "is_champion": ver.is_champion,
                    "created_at": ver.created_at.isoformat() if hasattr(ver, "created_at") and ver.created_at else None,
                }
                for v, ver in versions.items()
            }
        return success_response(data={
            "models": registry,
            "total_models": len(registry),
        })
    except Exception as exc:
        logger.error(f"Model version listing failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model version service unavailable",
        )


@router.post(
    "/versions/{model_name}/promote",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Promote a model version to a new stage",
)
async def promote_model_version(
    model_name: str,
    version: str,
    target_stage: str = "production",
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Promote a model version to staging or production."""
    try:
        from backend.ml.model_versioning import get_model_version_manager, ModelStage
        manager = get_model_version_manager()
        stage_map = {s.value: s for s in ModelStage}
        if target_stage not in stage_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage: {target_stage}. Valid: {list(stage_map.keys())}",
            )
        success = manager.promote_model(model_name, version, stage_map[target_stage])
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} version {version} not found",
            )
        return success_response(data={
            "model_name": model_name,
            "version": version,
            "new_stage": target_stage,
            "promoted": True,
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Model promotion failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model promotion failed: {exc}",
        )


@router.post(
    "/versions/{model_name}/rollback",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Rollback model to a previous version",
)
async def rollback_model_version(
    model_name: str,
    target_version: str,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Rollback a model to a specific previous version."""
    try:
        from backend.ml.model_versioning import get_model_version_manager
        manager = get_model_version_manager()
        success = manager.rollback_model(model_name, target_version)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} version {target_version} not found",
            )
        return success_response(data={
            "model_name": model_name,
            "rolled_back_to": target_version,
            "success": True,
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Model rollback failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model rollback failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Backtesting Endpoints
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    """Request body for running a backtest."""
    tickers: List[str] = Field(..., min_length=1, description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100000.0, gt=0, description="Starting capital")
    benchmark: str = Field("SPY", description="Benchmark ticker symbol")


@router.post(
    "/backtest",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Run a backtest",
)
async def run_backtest(
    request_body: BacktestRequest,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """
    Run a backtest with the specified parameters.
    Returns performance metrics including Sharpe ratio, max drawdown,
    total return, and trade history.
    """
    try:
        from backend.ml.backtesting import get_backtest_engine, BacktestConfig
        engine = get_backtest_engine()

        config = BacktestConfig(
            start_date=request_body.start_date,
            end_date=request_body.end_date,
            initial_capital=request_body.initial_capital,
            benchmark_symbol=request_body.benchmark,
        )

        # Use a simple buy-and-hold strategy as default
        def buy_and_hold(data, portfolio, date):
            return {ticker: 1.0 / len(data) for ticker in data}

        result = engine.backtest_strategy(
            strategy_func=buy_and_hold,
            universe=request_body.tickers,
            config=config,
        )

        return success_response(data={
            "tickers": request_body.tickers,
            "start_date": request_body.start_date,
            "end_date": request_body.end_date,
            "initial_capital": request_body.initial_capital,
            "total_return": getattr(result, "total_return", None),
            "sharpe_ratio": getattr(result, "sharpe_ratio", None),
            "max_drawdown": getattr(result, "max_drawdown", None),
            "total_trades": getattr(result, "total_trades", None),
        })
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest execution failed: {exc}",
        )
