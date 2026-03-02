"""
Settings Service
Persist and retrieve user preferences/settings from the User.preferences
and User.notification_settings JSON columns in the database.

The User model already has:
  - preferences = Column(JSON, default={})
  - notification_settings = Column(JSON, default={...})

This service stores all settings categories inside the preferences JSON dict
under namespaced keys so that we never need a separate migration.
"""

import logging
from typing import Any, Dict, Optional

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.unified_models import User

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace keys stored inside User.preferences JSON column
# ---------------------------------------------------------------------------
_NS_PREFERENCES = "user_preferences"
_NS_DISPLAY = "display_settings"
_NS_TRADING = "trading_settings"
_NS_NOTIFICATIONS = "notification_settings"


# ---------------------------------------------------------------------------
# Default values (mirrors the Pydantic models in the router)
# ---------------------------------------------------------------------------

def _default_preferences() -> Dict[str, Any]:
    return {
        "theme": "light",
        "language": "en",
        "timezone": "UTC",
        "currency": "USD",
        "notifications_enabled": True,
        "email_alerts": False,
        "push_notifications": False,
    }


def _default_display() -> Dict[str, Any]:
    return {
        "dashboard_layout": "default",
        "default_chart_type": "candlestick",
        "show_technical_indicators": True,
        "show_news_feed": True,
        "compact_mode": False,
    }


def _default_trading() -> Dict[str, Any]:
    return {
        "default_order_type": "market",
        "confirmation_required": True,
        "default_quantity": 1,
        "risk_tolerance": "moderate",
    }


def _default_notifications() -> Dict[str, Any]:
    return {
        "price_alerts": True,
        "news_alerts": True,
        "portfolio_alerts": True,
        "recommendation_alerts": False,
        "alert_threshold": 5.0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_prefs(user: User) -> Dict[str, Any]:
    """Return a mutable copy of the user's preferences dict (never None)."""
    prefs = user.preferences
    if not isinstance(prefs, dict):
        return {}
    return dict(prefs)


async def _patch_preferences(
    db: AsyncSession,
    user_id: int,
    namespace: str,
    data: Dict[str, Any],
    current_prefs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge *data* into *current_prefs[namespace]* and persist to DB.

    Returns the merged namespace dict.
    """
    merged = {**current_prefs.get(namespace, {}), **data}
    new_prefs = {**current_prefs, namespace: merged}

    stmt = (
        update(User)
        .where(User.id == user_id)
        .values(preferences=new_prefs)
    )
    await db.execute(stmt)
    await db.commit()
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_preferences(user: User) -> Dict[str, Any]:
    """
    Return user preferences, falling back to defaults for missing keys.

    Args:
        user: ORM User instance (already loaded by the router).

    Returns:
        Dict matching the UserPreferences schema.
    """
    stored = _safe_prefs(user).get(_NS_PREFERENCES, {})
    return {**_default_preferences(), **stored}


async def update_preferences(
    db: AsyncSession,
    user: User,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Persist updated user preferences to the database.

    Args:
        db: Async database session.
        user: ORM User instance.
        data: Dict of preference fields to update.

    Returns:
        Updated preferences dict.
    """
    current_prefs = _safe_prefs(user)
    # Merge with defaults so we never lose unset fields
    merged = {**_default_preferences(), **current_prefs.get(_NS_PREFERENCES, {}), **data}
    await _patch_preferences(db, user.id, _NS_PREFERENCES, merged, current_prefs)
    logger.info(f"Updated preferences for user {user.id}")
    return merged


async def get_display_settings(user: User) -> Dict[str, Any]:
    """Return display settings, falling back to defaults."""
    stored = _safe_prefs(user).get(_NS_DISPLAY, {})
    return {**_default_display(), **stored}


async def update_display_settings(
    db: AsyncSession,
    user: User,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist updated display settings."""
    current_prefs = _safe_prefs(user)
    merged = {**_default_display(), **current_prefs.get(_NS_DISPLAY, {}), **data}
    await _patch_preferences(db, user.id, _NS_DISPLAY, merged, current_prefs)
    logger.info(f"Updated display settings for user {user.id}")
    return merged


async def get_trading_settings(user: User) -> Dict[str, Any]:
    """Return trading settings, falling back to defaults."""
    stored = _safe_prefs(user).get(_NS_TRADING, {})
    return {**_default_trading(), **stored}


async def update_trading_settings(
    db: AsyncSession,
    user: User,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist updated trading settings."""
    current_prefs = _safe_prefs(user)
    merged = {**_default_trading(), **current_prefs.get(_NS_TRADING, {}), **data}
    await _patch_preferences(db, user.id, _NS_TRADING, merged, current_prefs)
    logger.info(f"Updated trading settings for user {user.id}")
    return merged


async def get_notification_settings(user: User) -> Dict[str, Any]:
    """Return notification settings, falling back to defaults."""
    # Also check the dedicated notification_settings column first
    stored_col = user.notification_settings
    stored_prefs = _safe_prefs(user).get(_NS_NOTIFICATIONS, {})
    base = _default_notifications()
    if isinstance(stored_col, dict):
        base.update(stored_col)
    base.update(stored_prefs)
    return base


async def update_notification_settings(
    db: AsyncSession,
    user: User,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist updated notification settings."""
    current_prefs = _safe_prefs(user)
    merged = {**_default_notifications(), **current_prefs.get(_NS_NOTIFICATIONS, {}), **data}
    await _patch_preferences(db, user.id, _NS_NOTIFICATIONS, merged, current_prefs)
    logger.info(f"Updated notification settings for user {user.id}")
    return merged


async def reset_all_settings(
    db: AsyncSession,
    user: User,
) -> Dict[str, Any]:
    """
    Reset all settings to factory defaults for the given user.

    Args:
        db: Async database session.
        user: ORM User instance.

    Returns:
        Dict with all reset settings.
    """
    new_prefs: Dict[str, Any] = {
        _NS_PREFERENCES: _default_preferences(),
        _NS_DISPLAY: _default_display(),
        _NS_TRADING: _default_trading(),
        _NS_NOTIFICATIONS: _default_notifications(),
    }

    stmt = update(User).where(User.id == user.id).values(preferences=new_prefs)
    await db.execute(stmt)
    await db.commit()

    logger.info(f"Reset all settings to defaults for user {user.id}")
    return {
        "preferences": _default_preferences(),
        "display": _default_display(),
        "trading": _default_trading(),
        "notifications": _default_notifications(),
    }
