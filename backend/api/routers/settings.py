"""
User Settings API Router
Provides user preferences and application settings management.
Settings are persisted to the database via the settings_service.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
import logging
import os
import re
from pathlib import Path

from backend.config.database import get_async_db_session
from backend.models.api_response import ApiResponse, success_response
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User
from backend.services import settings_service
from backend.config.settings import settings as app_settings
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# Pydantic Models
class UserPreferences(BaseModel):
    """User preference settings"""
    theme: str = Field(default="light", description="UI theme (light/dark)")
    language: str = Field(default="en", description="Language preference")
    timezone: str = Field(default="UTC", description="User timezone")
    currency: str = Field(default="USD", description="Preferred currency")
    notifications_enabled: bool = Field(default=True, description="Enable notifications")
    email_alerts: bool = Field(default=False, description="Enable email alerts")
    push_notifications: bool = Field(default=False, description="Enable push notifications")


class DisplaySettings(BaseModel):
    """Display and UI settings"""
    dashboard_layout: str = Field(default="default", description="Dashboard layout preference")
    default_chart_type: str = Field(default="candlestick", description="Default chart type")
    show_technical_indicators: bool = Field(default=True, description="Show technical indicators")
    show_news_feed: bool = Field(default=True, description="Show news feed")
    compact_mode: bool = Field(default=False, description="Use compact UI mode")


class TradingSettings(BaseModel):
    """Trading-related settings"""
    default_order_type: str = Field(default="market", description="Default order type")
    confirmation_required: bool = Field(default=True, description="Require order confirmation")
    default_quantity: int = Field(default=1, description="Default order quantity", ge=1)
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance (conservative/moderate/aggressive)")


class NotificationSettings(BaseModel):
    """Notification preferences"""
    price_alerts: bool = Field(default=True, description="Enable price alerts")
    news_alerts: bool = Field(default=True, description="Enable news alerts")
    portfolio_alerts: bool = Field(default=True, description="Enable portfolio alerts")
    recommendation_alerts: bool = Field(default=False, description="Enable recommendation alerts")
    alert_threshold: float = Field(default=5.0, description="Alert threshold percentage", ge=0.0, le=100.0)


@router.get("/preferences")
async def get_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[UserPreferences]:
    """
    Get user preferences from the database.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        User preference settings (persisted or defaults on first access)
    """
    try:
        data = await settings_service.get_preferences(current_user)
        preferences = UserPreferences(**data)

        logger.info(f"User {current_user.id} fetched preferences")
        return success_response(data=preferences)

    except Exception as e:
        logger.error(f"Error fetching preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user preferences"
        )


@router.put("/preferences")
async def update_preferences(
    preferences: UserPreferences,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[UserPreferences]:
    """
    Update and persist user preferences to the database.

    Args:
        preferences: Updated preference settings
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated user preferences
    """
    try:
        updated_data = await settings_service.update_preferences(
            db=db,
            user=current_user,
            data=preferences.model_dump(),
        )
        updated = UserPreferences(**updated_data)

        logger.info(f"User {current_user.id} updated preferences")
        return success_response(data=updated)

    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )


@router.get("/display")
async def get_display_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DisplaySettings]:
    """
    Get display and UI settings from the database.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Display settings
    """
    try:
        data = await settings_service.get_display_settings(current_user)
        settings_obj = DisplaySettings(**data)

        logger.info(f"User {current_user.id} fetched display settings")
        return success_response(data=settings_obj)

    except Exception as e:
        logger.error(f"Error fetching display settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch display settings"
        )


@router.put("/display")
async def update_display_settings(
    settings: DisplaySettings,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DisplaySettings]:
    """
    Update and persist display and UI settings to the database.

    Args:
        settings: Updated display settings
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated display settings
    """
    try:
        updated_data = await settings_service.update_display_settings(
            db=db,
            user=current_user,
            data=settings.model_dump(),
        )
        updated = DisplaySettings(**updated_data)

        logger.info(f"User {current_user.id} updated display settings")
        return success_response(data=updated)

    except Exception as e:
        logger.error(f"Error updating display settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update display settings"
        )


@router.get("/trading")
async def get_trading_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[TradingSettings]:
    """
    Get trading-related settings from the database.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Trading settings
    """
    try:
        data = await settings_service.get_trading_settings(current_user)
        settings_obj = TradingSettings(**data)

        logger.info(f"User {current_user.id} fetched trading settings")
        return success_response(data=settings_obj)

    except Exception as e:
        logger.error(f"Error fetching trading settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch trading settings"
        )


@router.put("/trading")
async def update_trading_settings(
    settings: TradingSettings,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[TradingSettings]:
    """
    Update and persist trading-related settings to the database.

    Args:
        settings: Updated trading settings
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated trading settings
    """
    try:
        updated_data = await settings_service.update_trading_settings(
            db=db,
            user=current_user,
            data=settings.model_dump(),
        )
        updated = TradingSettings(**updated_data)

        logger.info(f"User {current_user.id} updated trading settings")
        return success_response(data=updated)

    except Exception as e:
        logger.error(f"Error updating trading settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update trading settings"
        )


@router.get("/notifications")
async def get_notification_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[NotificationSettings]:
    """
    Get notification preferences from the database.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Notification settings
    """
    try:
        data = await settings_service.get_notification_settings(current_user)
        settings_obj = NotificationSettings(**data)

        logger.info(f"User {current_user.id} fetched notification settings")
        return success_response(data=settings_obj)

    except Exception as e:
        logger.error(f"Error fetching notification settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch notification settings"
        )


@router.put("/notifications")
async def update_notification_settings(
    settings: NotificationSettings,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[NotificationSettings]:
    """
    Update and persist notification preferences to the database.

    Args:
        settings: Updated notification settings
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated notification settings
    """
    try:
        updated_data = await settings_service.update_notification_settings(
            db=db,
            user=current_user,
            data=settings.model_dump(),
        )
        updated = NotificationSettings(**updated_data)

        logger.info(f"User {current_user.id} updated notification settings")
        return success_response(data=updated)

    except Exception as e:
        logger.error(f"Error updating notification settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update notification settings"
        )


@router.post("/reset")
async def reset_to_defaults(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[Dict[str, Any]]:
    """
    Reset all settings to default values and persist the reset to the database.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Confirmation message with reset settings
    """
    try:
        reset_data = await settings_service.reset_all_settings(db=db, user=current_user)

        logger.info(f"User {current_user.id} reset settings to defaults")
        return success_response(data={
            "message": "All settings reset to defaults",
            **reset_data,
        })

    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset settings"
        )


# ============================================================================
# Provider API keys
# ============================================================================
# UI field name -> .env variable name. Only these keys may ever be written.
_API_KEY_ENV_VARS = {
    "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
    "finnhub": "FINNHUB_API_KEY",
    "polygon": "POLYGON_API_KEY",
    "fmp": "FMP_API_KEY",
    "news_api": "NEWS_API_KEY",
    "marketaux": "MARKETAUX_API_KEY",
    "fred": "FRED_API_KEY",
    "openweather": "OPENWEATHER_API_KEY",
}
# Safe charset only -- prevents .env injection (no newlines, no '=').
_API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._\-]{8,128}$")
_PLACEHOLDER_RE = re.compile(r"your|placeholder|xxx|changeme|demo", re.I)


class ApiKeysUpdate(BaseModel):
    """Provider API keys. Any subset may be sent; blank/omitted values are ignored
    (so unchanged fields are never overwritten)."""
    alpha_vantage: Optional[str] = None
    finnhub: Optional[str] = None
    polygon: Optional[str] = None
    fmp: Optional[str] = None
    news_api: Optional[str] = None
    marketaux: Optional[str] = None
    fred: Optional[str] = None
    openweather: Optional[str] = None


def _mask_key(value: Optional[str]) -> Optional[str]:
    """Return a masked preview (e.g. 'abcd…wxyz') or None if unset/placeholder. Never
    returns the full secret."""
    if not value or _PLACEHOLDER_RE.search(value):
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def _current_key(env_var: str) -> Optional[str]:
    # Prefer runtime os.environ (reflects in-process updates) then startup settings.
    return os.environ.get(env_var) or getattr(app_settings, env_var, None)


def _masked_status() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ui_key, env_var in _API_KEY_ENV_VARS.items():
        masked = _mask_key(_current_key(env_var))
        out[ui_key] = {"configured": masked is not None, "masked": masked}
    return out


def _env_path() -> Path:
    # backend/api/routers/settings.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3] / ".env"


def _write_env_keys(updates: Dict[str, str]) -> None:
    """Persist whitelisted KEY=value lines to .env, preserving all other content.

    Backs up .env to .env.bak first, then replaces only matching ``KEY=`` lines
    (appending any missing). Values are pre-validated to a safe charset, so no .env
    injection is possible and unrelated secrets (DB password, JWT keys) are untouched.
    """
    env_path = _env_path()
    lines = (
        env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if env_path.exists() else []
    )
    if lines:
        (env_path.parent / ".env.bak").write_text("".join(lines), encoding="utf-8")

    remaining = dict(updates)
    out = []
    for line in lines:
        replaced = False
        for env_var, value in updates.items():
            if line.lstrip().startswith(f"{env_var}="):
                out.append(f"{env_var}={value}\n")
                remaining.pop(env_var, None)
                replaced = True
                break
        if not replaced:
            out.append(line)
    if out and not out[-1].endswith("\n"):
        out[-1] += "\n"
    for env_var, value in remaining.items():
        out.append(f"{env_var}={value}\n")
    env_path.write_text("".join(out), encoding="utf-8")


@router.get("/api-keys")
async def get_api_keys(
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Return masked configured-status for each provider key. Never returns full keys."""
    return success_response(data=_masked_status())


@router.put("/api-keys")
async def update_api_keys(
    payload: ApiKeysUpdate,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Persist provider API keys to .env (development only).

    Writing provider keys via the API is a local/single-user dev convenience and is
    refused outside development -- elsewhere keys come from the deployment environment
    or secret manager. Values are charset-validated; only whitelisted provider keys are
    written; .env is backed up first. A backend restart is recommended so module-level
    provider clients (built from ``settings.*`` at import) pick up the new values.
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    if environment != "development":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Saving API keys via the API is only permitted in development; "
                   "set them in the deployment environment otherwise.",
        )

    provided = payload.model_dump()
    updates: Dict[str, str] = {}
    rejected = []
    for ui_key, env_var in _API_KEY_ENV_VARS.items():
        value = provided.get(ui_key)
        if value is None:
            continue
        value = value.strip()
        if value == "" or "*" in value:  # blank or a masked echo from the UI -> ignore
            continue
        if not _API_KEY_PATTERN.match(value):
            rejected.append(ui_key)
            continue
        updates[env_var] = value

    if rejected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid key format for: {', '.join(rejected)} "
                   "(allowed: letters, digits, '.', '_', '-'; length 8-128).",
        )

    if updates:
        try:
            _write_env_keys(updates)
        except Exception as e:
            logger.error(f"Error persisting API keys to .env: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist API keys",
            )
        # Apply in-process so os.getenv consumers see them immediately.
        for env_var, value in updates.items():
            os.environ[env_var] = value
            try:
                setattr(app_settings, env_var, value)
            except Exception:  # pragma: no cover - settings is a plain attr holder
                pass

    logger.info(
        "User %s updated API keys: %s", current_user.id, sorted(updates.keys())
    )
    return success_response(data={
        "updated": [k for k, v in _API_KEY_ENV_VARS.items() if v in updates],
        "restart_recommended": bool(updates),
        "keys": _masked_status(),
    })
