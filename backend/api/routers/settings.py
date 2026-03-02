"""
User Settings API Router
Provides user preferences and application settings management.
Settings are persisted to the database via the settings_service.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
import logging

from backend.config.database import get_async_db_session
from backend.models.api_response import ApiResponse, success_response
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User
from backend.services import settings_service
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
