"""
Integration tests for the Settings API router.

Tests validate the API contract for user preferences, display settings,
trading settings, notification settings, and the reset-to-defaults endpoint.
Even though the router is currently a stub (returning defaults, not persisting
to the database), these tests lock down the expected request/response shapes
so that any future implementation must honour the contract.

Endpoints under test (all mounted at /api/settings):
    GET  /preferences      - Fetch user preferences
    PUT  /preferences      - Update user preferences
    GET  /display          - Fetch display settings
    PUT  /display          - Update display settings
    GET  /trading          - Fetch trading settings
    PUT  /trading          - Update trading settings
    GET  /notifications    - Fetch notification settings
    PUT  /notifications    - Update notification settings
    POST /reset            - Reset all settings to defaults
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient

from backend.api.main import app
from backend.auth.oauth2 import get_current_user
from backend.config.database import get_async_db_session


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Expected default values (mirrors the Pydantic models in settings.py)
# ---------------------------------------------------------------------------

DEFAULT_PREFERENCES = {
    "theme": "light",
    "language": "en",
    "timezone": "UTC",
    "currency": "USD",
    "notifications_enabled": True,
    "email_alerts": False,
    "push_notifications": False,
}

DEFAULT_DISPLAY = {
    "dashboard_layout": "default",
    "default_chart_type": "candlestick",
    "show_technical_indicators": True,
    "show_news_feed": True,
    "compact_mode": False,
}

DEFAULT_TRADING = {
    "default_order_type": "market",
    "confirmation_required": True,
    "default_quantity": 1,
    "risk_tolerance": "moderate",
}

DEFAULT_NOTIFICATIONS = {
    "price_alerts": True,
    "news_alerts": True,
    "portfolio_alerts": True,
    "recommendation_alerts": False,
    "alert_threshold": 5.0,
}


# ===========================================================================
# 1. Authentication requirement tests
# ===========================================================================


class TestSettingsAuthenticationRequired:
    """Every settings endpoint must reject unauthenticated requests."""

    @pytest.mark.asyncio
    async def test_get_preferences_requires_auth(self, async_client: AsyncClient):
        """GET /api/settings/preferences returns 401 without a token."""
        response = await async_client.get("/api/settings/preferences")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_put_preferences_requires_auth(self, async_client: AsyncClient):
        """PUT /api/settings/preferences returns 401 without a token."""
        response = await async_client.put(
            "/api/settings/preferences",
            json=DEFAULT_PREFERENCES,
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_display_requires_auth(self, async_client: AsyncClient):
        """GET /api/settings/display returns 401 without a token."""
        response = await async_client.get("/api/settings/display")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_put_display_requires_auth(self, async_client: AsyncClient):
        """PUT /api/settings/display returns 401 without a token."""
        response = await async_client.put(
            "/api/settings/display",
            json=DEFAULT_DISPLAY,
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_trading_requires_auth(self, async_client: AsyncClient):
        """GET /api/settings/trading returns 401 without a token."""
        response = await async_client.get("/api/settings/trading")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_put_trading_requires_auth(self, async_client: AsyncClient):
        """PUT /api/settings/trading returns 401 without a token."""
        response = await async_client.put(
            "/api/settings/trading",
            json=DEFAULT_TRADING,
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_notifications_requires_auth(self, async_client: AsyncClient):
        """GET /api/settings/notifications returns 401 without a token."""
        response = await async_client.get("/api/settings/notifications")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_put_notifications_requires_auth(self, async_client: AsyncClient):
        """PUT /api/settings/notifications returns 401 without a token."""
        response = await async_client.put(
            "/api/settings/notifications",
            json=DEFAULT_NOTIFICATIONS,
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_reset_requires_auth(self, async_client: AsyncClient):
        """POST /api/settings/reset returns 401 without a token."""
        response = await async_client.post("/api/settings/reset")
        assert response.status_code == 401


# ===========================================================================
# 2. GET endpoints - default values and response format
# ===========================================================================


class TestGetPreferences:
    """GET /api/settings/preferences returns the ApiResponse wrapper
    with default user preference values."""

    @pytest.mark.asyncio
    async def test_returns_200(self, async_client: AsyncClient, test_user, auth_headers):
        """Authenticated request succeeds with 200."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(
                "/api/settings/preferences",
                headers=auth_headers,
            )
            assert response.status_code == 200
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_response_wraps_in_api_response(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Response body follows the ApiResponse envelope."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(
                "/api/settings/preferences",
                headers=auth_headers,
            )
            body = response.json()
            assert body["success"] is True
            assert "data" in body
            assert body.get("error") is None
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_default_preference_values(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Default preferences match the Pydantic model defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(
                "/api/settings/preferences",
                headers=auth_headers,
            )
            data = response.json()["data"]
            for key, expected_value in DEFAULT_PREFERENCES.items():
                assert data[key] == expected_value, (
                    f"preferences.{key}: expected {expected_value!r}, got {data[key]!r}"
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)


class TestGetDisplaySettings:
    """GET /api/settings/display returns default display settings."""

    @pytest.mark.asyncio
    async def test_returns_default_display_settings(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Display settings match Pydantic model defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(
                "/api/settings/display",
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            for key, expected_value in DEFAULT_DISPLAY.items():
                assert data[key] == expected_value, (
                    f"display.{key}: expected {expected_value!r}, got {data[key]!r}"
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)


class TestGetTradingSettings:
    """GET /api/settings/trading returns default trading settings."""

    @pytest.mark.asyncio
    async def test_returns_default_trading_settings(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Trading settings match Pydantic model defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(
                "/api/settings/trading",
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            for key, expected_value in DEFAULT_TRADING.items():
                assert data[key] == expected_value, (
                    f"trading.{key}: expected {expected_value!r}, got {data[key]!r}"
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)


class TestGetNotificationSettings:
    """GET /api/settings/notifications returns default notification settings."""

    @pytest.mark.asyncio
    async def test_returns_default_notification_settings(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Notification settings match Pydantic model defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(
                "/api/settings/notifications",
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            for key, expected_value in DEFAULT_NOTIFICATIONS.items():
                assert data[key] == expected_value, (
                    f"notifications.{key}: expected {expected_value!r}, got {data[key]!r}"
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)


# ===========================================================================
# 3. PUT endpoints - update settings and validate echo-back
# ===========================================================================


class TestUpdatePreferences:
    """PUT /api/settings/preferences accepts and echoes updated values."""

    @pytest.mark.asyncio
    async def test_update_preferences_returns_submitted_values(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Submitted preference values are returned in the response."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "theme": "dark",
                "language": "es",
                "timezone": "America/New_York",
                "currency": "EUR",
                "notifications_enabled": False,
                "email_alerts": True,
                "push_notifications": True,
            }
            response = await async_client.put(
                "/api/settings/preferences",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            body = response.json()
            assert body["success"] is True
            data = body["data"]
            for key, expected_value in payload.items():
                assert data[key] == expected_value, (
                    f"Updated preferences.{key}: expected {expected_value!r}, "
                    f"got {data[key]!r}"
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_update_with_partial_defaults_fills_missing(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """When only some fields are sent, Pydantic fills defaults for omitted ones."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {"theme": "dark"}
            response = await async_client.put(
                "/api/settings/preferences",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["theme"] == "dark"
            # Omitted fields should fall back to their Pydantic defaults
            assert data["language"] == "en"
            assert data["timezone"] == "UTC"
        finally:
            app.dependency_overrides.pop(get_current_user, None)


class TestUpdateDisplaySettings:
    """PUT /api/settings/display accepts and echoes updated values."""

    @pytest.mark.asyncio
    async def test_update_display_returns_submitted_values(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Submitted display values are returned in the response."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "dashboard_layout": "custom",
                "default_chart_type": "line",
                "show_technical_indicators": False,
                "show_news_feed": False,
                "compact_mode": True,
            }
            response = await async_client.put(
                "/api/settings/display",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            for key, expected_value in payload.items():
                assert data[key] == expected_value
        finally:
            app.dependency_overrides.pop(get_current_user, None)


class TestUpdateTradingSettings:
    """PUT /api/settings/trading accepts and echoes updated values."""

    @pytest.mark.asyncio
    async def test_update_trading_returns_submitted_values(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Submitted trading values are returned in the response."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "default_order_type": "limit",
                "confirmation_required": False,
                "default_quantity": 100,
                "risk_tolerance": "aggressive",
            }
            response = await async_client.put(
                "/api/settings/trading",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            for key, expected_value in payload.items():
                assert data[key] == expected_value
        finally:
            app.dependency_overrides.pop(get_current_user, None)


class TestUpdateNotificationSettings:
    """PUT /api/settings/notifications accepts and echoes updated values."""

    @pytest.mark.asyncio
    async def test_update_notifications_returns_submitted_values(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Submitted notification values are returned in the response."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "price_alerts": False,
                "news_alerts": False,
                "portfolio_alerts": False,
                "recommendation_alerts": True,
                "alert_threshold": 10.0,
            }
            response = await async_client.put(
                "/api/settings/notifications",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            for key, expected_value in payload.items():
                assert data[key] == expected_value
        finally:
            app.dependency_overrides.pop(get_current_user, None)


# ===========================================================================
# 4. Input validation tests
# ===========================================================================


class TestSettingsInputValidation:
    """PUT endpoints must reject payloads that violate Pydantic constraints."""

    @pytest.mark.asyncio
    async def test_trading_quantity_must_be_positive(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """default_quantity has ge=1, so 0 is invalid (422)."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "default_order_type": "market",
                "confirmation_required": True,
                "default_quantity": 0,
                "risk_tolerance": "moderate",
            }
            response = await async_client.put(
                "/api/settings/trading",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_trading_quantity_negative_rejected(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Negative default_quantity is rejected (422)."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "default_order_type": "market",
                "confirmation_required": True,
                "default_quantity": -5,
                "risk_tolerance": "moderate",
            }
            response = await async_client.put(
                "/api/settings/trading",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_notification_threshold_over_100_rejected(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """alert_threshold has le=100.0, so 101 is invalid (422)."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "price_alerts": True,
                "news_alerts": True,
                "portfolio_alerts": True,
                "recommendation_alerts": False,
                "alert_threshold": 101.0,
            }
            response = await async_client.put(
                "/api/settings/notifications",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_notification_threshold_negative_rejected(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """alert_threshold has ge=0.0, so -1 is invalid (422)."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "price_alerts": True,
                "news_alerts": True,
                "portfolio_alerts": True,
                "recommendation_alerts": False,
                "alert_threshold": -1.0,
            }
            response = await async_client.put(
                "/api/settings/notifications",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_preferences_invalid_json_rejected(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Malformed JSON body returns 422."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.put(
                "/api/settings/preferences",
                content=b"not valid json",
                headers={**auth_headers, "Content-Type": "application/json"},
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.pop(get_current_user, None)


# ===========================================================================
# 5. POST /reset endpoint
# ===========================================================================


class TestResetSettings:
    """POST /api/settings/reset returns all categories at their defaults."""

    @pytest.mark.asyncio
    async def test_reset_returns_200(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Reset endpoint succeeds with 200."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.post(
                "/api/settings/reset",
                headers=auth_headers,
            )
            assert response.status_code == 200
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_reset_response_contains_all_categories(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Reset response body includes preferences, display, trading, and
        notifications -- all at their default values."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.post(
                "/api/settings/reset",
                headers=auth_headers,
            )
            body = response.json()
            assert body["success"] is True

            data = body["data"]
            assert "message" in data
            assert "preferences" in data
            assert "display" in data
            assert "trading" in data
            assert "notifications" in data
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_reset_preferences_match_defaults(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Preferences section of reset response matches expected defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.post(
                "/api/settings/reset",
                headers=auth_headers,
            )
            prefs = response.json()["data"]["preferences"]
            for key, expected_value in DEFAULT_PREFERENCES.items():
                assert prefs[key] == expected_value, (
                    f"reset preferences.{key}: expected {expected_value!r}, "
                    f"got {prefs[key]!r}"
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_reset_display_matches_defaults(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Display section of reset response matches expected defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.post(
                "/api/settings/reset",
                headers=auth_headers,
            )
            display = response.json()["data"]["display"]
            for key, expected_value in DEFAULT_DISPLAY.items():
                assert display[key] == expected_value
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_reset_trading_matches_defaults(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Trading section of reset response matches expected defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.post(
                "/api/settings/reset",
                headers=auth_headers,
            )
            trading = response.json()["data"]["trading"]
            for key, expected_value in DEFAULT_TRADING.items():
                assert trading[key] == expected_value
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_reset_notifications_matches_defaults(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Notifications section of reset response matches expected defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.post(
                "/api/settings/reset",
                headers=auth_headers,
            )
            notifs = response.json()["data"]["notifications"]
            for key, expected_value in DEFAULT_NOTIFICATIONS.items():
                assert notifs[key] == expected_value
        finally:
            app.dependency_overrides.pop(get_current_user, None)


# ===========================================================================
# 6. Response format consistency
# ===========================================================================


class TestResponseFormatConsistency:
    """All settings endpoints must return the standard ApiResponse envelope."""

    SETTINGS_GET_ENDPOINTS = [
        "/api/settings/preferences",
        "/api/settings/display",
        "/api/settings/trading",
        "/api/settings/notifications",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", SETTINGS_GET_ENDPOINTS)
    async def test_all_get_endpoints_return_api_response_envelope(
        self, async_client: AsyncClient, test_user, auth_headers, endpoint
    ):
        """Every GET endpoint wraps its result in {success, data, error, meta}."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(endpoint, headers=auth_headers)
            assert response.status_code == 200
            body = response.json()
            assert "success" in body
            assert "data" in body
            # error should be absent or None on success
            assert body.get("error") is None
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_content_type_is_json(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Response Content-Type header is application/json."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.get(
                "/api/settings/preferences",
                headers=auth_headers,
            )
            assert "application/json" in response.headers.get("content-type", "")
        finally:
            app.dependency_overrides.pop(get_current_user, None)


# ===========================================================================
# 7. Edge cases and boundary values
# ===========================================================================


class TestSettingsEdgeCases:
    """Boundary conditions and edge-case inputs."""

    @pytest.mark.asyncio
    async def test_notification_threshold_at_zero_accepted(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """alert_threshold=0.0 is the lower boundary and should be accepted."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "price_alerts": True,
                "news_alerts": True,
                "portfolio_alerts": True,
                "recommendation_alerts": False,
                "alert_threshold": 0.0,
            }
            response = await async_client.put(
                "/api/settings/notifications",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert response.json()["data"]["alert_threshold"] == 0.0
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_notification_threshold_at_100_accepted(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """alert_threshold=100.0 is the upper boundary and should be accepted."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "price_alerts": True,
                "news_alerts": True,
                "portfolio_alerts": True,
                "recommendation_alerts": False,
                "alert_threshold": 100.0,
            }
            response = await async_client.put(
                "/api/settings/notifications",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert response.json()["data"]["alert_threshold"] == 100.0
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_trading_quantity_at_one_accepted(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """default_quantity=1 is the lower boundary and should be accepted."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "default_order_type": "market",
                "confirmation_required": True,
                "default_quantity": 1,
                "risk_tolerance": "moderate",
            }
            response = await async_client.put(
                "/api/settings/trading",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert response.json()["data"]["default_quantity"] == 1
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_empty_body_on_put_preferences_uses_defaults(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """An empty JSON object on PUT fills all fields with Pydantic defaults."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.put(
                "/api/settings/preferences",
                json={},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            for key, expected_value in DEFAULT_PREFERENCES.items():
                assert data[key] == expected_value
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_extra_fields_ignored_on_put_preferences(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """Unknown fields in the request body are silently ignored."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            payload = {
                "theme": "dark",
                "nonexistent_field": "should_be_ignored",
            }
            response = await async_client.put(
                "/api/settings/preferences",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["theme"] == "dark"
            assert "nonexistent_field" not in data
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @pytest.mark.asyncio
    async def test_wrong_http_method_returns_405(
        self, async_client: AsyncClient, test_user, auth_headers
    ):
        """DELETE on a settings endpoint returns 405 Method Not Allowed."""
        app.dependency_overrides[get_current_user] = lambda: test_user
        try:
            response = await async_client.delete(
                "/api/settings/preferences",
                headers=auth_headers,
            )
            assert response.status_code == 405
        finally:
            app.dependency_overrides.pop(get_current_user, None)
