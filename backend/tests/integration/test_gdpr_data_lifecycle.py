"""
Integration tests for GDPR data lifecycle and compliance.

Tests cover complete user data lifecycle including registration, consent management,
data export, anonymization, deletion cascades, and audit trail compliance.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import json
from unittest.mock import AsyncMock, patch
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.tables import (
    User, Portfolio, Position, Transaction, Alert,
    Watchlist, WatchlistItem, Stock, AuditLog,
    UserSession, ApiLog, UserRoleEnum, AssetTypeEnum
)
from backend.api.main import app
from httpx import AsyncClient, ASGITransport


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def gdpr_test_user(db_session: AsyncSession):
    """Create a test user with GDPR-relevant data."""
    user = User(
        email="gdpr.test@example.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU2VXhI0Asei",
        full_name="GDPR Test User",
        role=UserRoleEnum.BASIC_USER,
        is_active=True,
        is_verified=True,
        phone_number="+1234567890",
        country="US",
        timezone="America/New_York",
        preferences={
            "notifications": True,
            "marketing_emails": False,
            "data_sharing": False
        },
        notification_settings={
            "email": True,
            "sms": False,
            "push": True
        }
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def user_complete_data(db_session: AsyncSession, gdpr_test_user: User):
    """Create complete user data ecosystem for GDPR testing."""
    # Create stocks
    stock1 = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        asset_type=AssetTypeEnum.STOCK,
        is_active=True,
        is_tradable=True
    )
    stock2 = Stock(
        symbol="MSFT",
        name="Microsoft Corporation",
        exchange="NASDAQ",
        asset_type=AssetTypeEnum.STOCK,
        is_active=True,
        is_tradable=True
    )
    db_session.add_all([stock1, stock2])
    await db_session.commit()

    # Create portfolio
    portfolio = Portfolio(
        user_id=gdpr_test_user.id,
        name="Main Portfolio",
        cash_balance=Decimal("10000.00"),
        is_public=False,
        is_default=True
    )
    db_session.add(portfolio)
    await db_session.commit()

    # Create positions
    position1 = Position(
        portfolio_id=portfolio.id,
        stock_id=stock1.id,
        quantity=Decimal("50"),
        average_cost=Decimal("150.00")
    )
    position2 = Position(
        portfolio_id=portfolio.id,
        stock_id=stock2.id,
        quantity=Decimal("30"),
        average_cost=Decimal("300.00")
    )
    db_session.add_all([position1, position2])
    await db_session.commit()

    # Create transactions
    transaction = Transaction(
        portfolio_id=portfolio.id,
        stock_id=stock1.id,
        transaction_type="buy",
        quantity=Decimal("50"),
        price=Decimal("150.00"),
        commission=Decimal("5.00"),
        executed_at=datetime.utcnow()
    )
    db_session.add(transaction)
    await db_session.commit()

    # Create watchlist
    watchlist = Watchlist(
        user_id=gdpr_test_user.id,
        name="Tech Watchlist",
        is_public=False
    )
    db_session.add(watchlist)
    await db_session.commit()

    watchlist_item = WatchlistItem(
        watchlist_id=watchlist.id,
        stock_id=stock2.id,
        target_price=Decimal("350.00"),
        alert_enabled=True
    )
    db_session.add(watchlist_item)
    await db_session.commit()

    # Create alerts
    alert = Alert(
        user_id=gdpr_test_user.id,
        stock_id=stock1.id,
        alert_type="price_threshold",
        condition={"type": "above", "threshold": 170.00},
        is_active=True
    )
    db_session.add(alert)
    await db_session.commit()

    # Create session
    session = UserSession(
        user_id=gdpr_test_user.id,
        session_token="test_session_token",
        refresh_token="test_refresh_token",
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0",
        is_active=True,
        expires_at=datetime.utcnow() + timedelta(hours=24)
    )
    db_session.add(session)
    await db_session.commit()

    # Create audit logs
    audit_log = AuditLog(
        user_id=gdpr_test_user.id,
        action="user_login",
        entity_type="user",
        entity_id=gdpr_test_user.id,
        details={"ip": "192.168.1.1", "timestamp": datetime.utcnow().isoformat()},
        ip_address="192.168.1.1"
    )
    db_session.add(audit_log)
    await db_session.commit()

    return {
        "stocks": [stock1, stock2],
        "portfolio": portfolio,
        "positions": [position1, position2],
        "transaction": transaction,
        "watchlist": watchlist,
        "watchlist_item": watchlist_item,
        "alert": alert,
        "session": session,
        "audit_log": audit_log
    }


@pytest.mark.asyncio
async def test_user_registration_to_data_export(
    async_client: AsyncClient,
    db_session: AsyncSession
):
    """
    Test complete data lifecycle: registration -> usage -> export.

    Validates that all user data can be exported in machine-readable format
    for GDPR Article 20 (Right to Data Portability) compliance.
    """
    # Step 1: User registration
    registration_data = {
        "email": "newuser@example.com",
        "password": "SecurePassword123!",
        "full_name": "New User",
        "consent": {
            "terms_of_service": True,
            "privacy_policy": True,
            "data_processing": True,
            "marketing": False
        }
    }

    response = await async_client.post(
        "/api/v1/auth/register",
        json=registration_data
    )
    assert response.status_code == 201
    user_data = response.json()
    assert user_data["data"]["email"] == "newuser@example.com"
    user_id = user_data["data"]["id"]

    # Step 2: User performs actions (login, create portfolio, etc.)
    login_response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "email": "newuser@example.com",
            "password": "SecurePassword123!"
        }
    )
    assert login_response.status_code == 200
    token = login_response.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create portfolio
    await async_client.post(
        "/api/v1/portfolios",
        headers=headers,
        json={
            "name": "My Portfolio",
            "cash_balance": 5000.00
        }
    )

    # Step 3: Request full data export
    response = await async_client.get(
        "/api/v1/gdpr/export",
        headers=headers
    )
    assert response.status_code == 200
    export_data = response.json()

    # Verify export completeness
    assert "user_profile" in export_data["data"]
    assert "portfolios" in export_data["data"]
    assert "transactions" in export_data["data"]
    assert "watchlists" in export_data["data"]
    assert "alerts" in export_data["data"]
    assert "sessions" in export_data["data"]
    assert "audit_logs" in export_data["data"]
    assert "consent_history" in export_data["data"]

    # Verify personal data included
    assert export_data["data"]["user_profile"]["email"] == "newuser@example.com"
    assert export_data["data"]["user_profile"]["full_name"] == "New User"

    # Verify export format
    assert export_data["data"]["export_metadata"]["format"] == "JSON"
    assert export_data["data"]["export_metadata"]["version"] == "1.0"
    assert "timestamp" in export_data["data"]["export_metadata"]


@pytest.mark.asyncio
async def test_consent_affects_data_collection(
    async_client: AsyncClient,
    db_session: AsyncSession,
    gdpr_test_user: User
):
    """
    Test consent-based data collection filtering.

    Validates that user consent preferences properly control what data
    is collected and processed (GDPR Article 6 - Lawful Basis).
    """
    from backend.auth.oauth2 import create_access_token

    token = create_access_token(data={"sub": str(gdpr_test_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: Update consent preferences
    response = await async_client.put(
        "/api/v1/gdpr/consent",
        headers=headers,
        json={
            "analytics": False,
            "marketing": False,
            "third_party_sharing": False,
            "essential_only": True
        }
    )
    assert response.status_code == 200

    # Step 2: Perform action that would normally log analytics
    with patch("backend.services.analytics_service.AnalyticsService.track_event") as mock_analytics:
        response = await async_client.get(
            "/api/v1/stocks/AAPL",
            headers=headers
        )
        assert response.status_code == 200

        # Analytics should NOT be called when consent is False
        assert not mock_analytics.called

    # Step 3: Enable analytics consent
    response = await async_client.put(
        "/api/v1/gdpr/consent",
        headers=headers,
        json={
            "analytics": True,
            "marketing": False,
            "third_party_sharing": False
        }
    )
    assert response.status_code == 200

    # Step 4: Now analytics should be collected
    with patch("backend.services.analytics_service.AnalyticsService.track_event") as mock_analytics:
        response = await async_client.get(
            "/api/v1/stocks/MSFT",
            headers=headers
        )
        assert response.status_code == 200

        # Analytics SHOULD be called when consent is True
        assert mock_analytics.called
        call_args = mock_analytics.call_args[1]
        assert call_args["event_type"] == "stock_view"
        assert call_args["user_id"] == gdpr_test_user.id


@pytest.mark.asyncio
async def test_data_deletion_cascades(
    async_client: AsyncClient,
    db_session: AsyncSession,
    gdpr_test_user: User,
    user_complete_data: dict
):
    """
    Test cascading deletion of all related user data.

    Validates that when user requests account deletion, ALL related data
    is properly deleted (GDPR Article 17 - Right to Erasure).
    """
    from backend.auth.oauth2 import create_access_token

    token = create_access_token(data={"sub": str(gdpr_test_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    user_id = gdpr_test_user.id

    # Verify data exists before deletion
    stmt = select(Portfolio).where(Portfolio.user_id == user_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is not None

    stmt = select(Alert).where(Alert.user_id == user_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is not None

    # Request account deletion
    response = await async_client.delete(
        "/api/v1/gdpr/delete-account",
        headers=headers,
        json={
            "confirmation": "DELETE MY ACCOUNT",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200
    deletion_data = response.json()
    assert deletion_data["data"]["status"] == "deleted"

    # Verify user deleted
    stmt = select(User).where(User.id == user_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None

    # Verify cascading deletions
    stmt = select(Portfolio).where(Portfolio.user_id == user_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None

    stmt = select(Alert).where(Alert.user_id == user_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None

    stmt = select(Watchlist).where(Watchlist.user_id == user_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None

    stmt = select(UserSession).where(UserSession.user_id == user_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None

    # Verify audit trail preserved (anonymized)
    stmt = select(AuditLog).where(AuditLog.user_id == user_id)
    result = await db_session.execute(stmt)
    audit_logs = result.scalars().all()

    # Audit logs should be anonymized but preserved
    for log in audit_logs:
        assert log.details.get("anonymized") == True
        assert log.details.get("deletion_date") is not None


@pytest.mark.asyncio
async def test_anonymization_in_analytics(
    async_client: AsyncClient,
    db_session: AsyncSession,
    gdpr_test_user: User
):
    """
    Test PII scrubbing in analytics and aggregated data.

    Validates that analytics data is properly anonymized with no
    personally identifiable information exposed.
    """
    from backend.auth.oauth2 import create_access_token

    token = create_access_token(data={"sub": str(gdpr_test_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Generate some analytics events
    await async_client.get("/api/v1/stocks/AAPL", headers=headers)
    await async_client.get("/api/v1/stocks/MSFT", headers=headers)
    await async_client.get("/api/v1/stocks/GOOGL", headers=headers)

    # Request anonymized analytics data
    response = await async_client.get(
        "/api/v1/analytics/aggregated",
        headers=headers,
        params={"period": "7d"}
    )
    assert response.status_code == 200
    analytics_data = response.json()

    # Verify PII is scrubbed
    data_str = json.dumps(analytics_data["data"])
    assert gdpr_test_user.email not in data_str
    assert gdpr_test_user.full_name not in data_str
    if gdpr_test_user.phone_number:
        assert gdpr_test_user.phone_number not in data_str

    # Verify anonymized identifiers used
    assert "user_events" in analytics_data["data"]
    for event in analytics_data["data"]["user_events"]:
        # Should have anonymized user ID, not actual user ID
        if "user_id" in event:
            assert event["user_id"] != str(gdpr_test_user.id)
            assert len(event["user_id"]) >= 32  # Hash-like identifier

        # Should not have email or name
        assert "email" not in event
        assert "full_name" not in event
        assert "phone" not in event


@pytest.mark.asyncio
async def test_gdpr_compliance_audit_trail(
    async_client: AsyncClient,
    db_session: AsyncSession,
    gdpr_test_user: User,
    user_complete_data: dict
):
    """
    Test comprehensive audit trail for all GDPR operations.

    Validates that all data access, modifications, exports, and deletions
    are properly logged for compliance auditing (GDPR Article 30).
    """
    from backend.auth.oauth2 import create_access_token

    token = create_access_token(data={"sub": str(gdpr_test_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    user_id = gdpr_test_user.id

    # Perform various GDPR-relevant operations

    # 1. Update consent
    await async_client.put(
        "/api/v1/gdpr/consent",
        headers=headers,
        json={"analytics": False, "marketing": False}
    )

    # 2. Request data export
    await async_client.get("/api/v1/gdpr/export", headers=headers)

    # 3. Update profile
    await async_client.put(
        "/api/v1/users/profile",
        headers=headers,
        json={"full_name": "Updated Name"}
    )

    # 4. View audit trail
    response = await async_client.get(
        "/api/v1/gdpr/audit-trail",
        headers=headers
    )
    assert response.status_code == 200
    audit_data = response.json()

    # Verify all operations logged
    audit_entries = audit_data["data"]["entries"]
    assert len(audit_entries) >= 3

    # Check for specific operation types
    operation_types = [entry["action"] for entry in audit_entries]
    assert "consent_update" in operation_types
    assert "data_export" in operation_types
    assert "profile_update" in operation_types

    # Verify audit entry structure
    for entry in audit_entries:
        assert "timestamp" in entry
        assert "action" in entry
        assert "ip_address" in entry
        assert "user_agent" in entry
        assert "details" in entry

        # Sensitive data should be hashed in audit logs
        if "password" in entry["details"]:
            assert entry["details"]["password"] == "[REDACTED]"

    # Verify audit trail immutability
    initial_entry = audit_entries[0]

    # Attempt to modify audit log (should fail)
    response = await async_client.put(
        f"/api/v1/gdpr/audit-trail/{initial_entry['id']}",
        headers=headers,
        json={"action": "modified_action"}
    )
    assert response.status_code == 403  # Forbidden - audit logs are immutable

    # Verify retention policy
    response = await async_client.get(
        "/api/v1/gdpr/retention-policy",
        headers=headers
    )
    assert response.status_code == 200
    retention_data = response.json()

    assert "audit_logs" in retention_data["data"]
    assert retention_data["data"]["audit_logs"]["retention_days"] >= 365
    assert retention_data["data"]["audit_logs"]["immutable"] == True
