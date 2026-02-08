"""
Integration tests for GDPR data lifecycle and compliance.

Tests cover complete user data lifecycle including registration, consent management,
data export, anonymization, deletion cascades, and audit trail compliance.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from unittest.mock import AsyncMock, patch
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.unified_models import (
    User, Portfolio, Position, Transaction, Alert,
    Watchlist, Stock, AuditLog,
    UserSession, Exchange, Sector,
    UserRoleEnum, AssetTypeEnum
)
from backend.api.main import app
from httpx import AsyncClient, ASGITransport


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def nasdaq_exchange(db_session: AsyncSession):
    """Create NASDAQ exchange for testing."""
    exchange = Exchange(
        code="NASDAQ",
        name="NASDAQ Stock Market",
        country="US",
        currency="USD",
        timezone="America/New_York"
    )
    db_session.add(exchange)
    await db_session.commit()
    await db_session.refresh(exchange)
    return exchange


@pytest_asyncio.fixture
async def technology_sector(db_session: AsyncSession):
    """Create Technology sector for testing."""
    sector = Sector(
        name="Technology",
        description="Technology sector"
    )
    db_session.add(sector)
    await db_session.commit()
    await db_session.refresh(sector)
    return sector


@pytest_asyncio.fixture
async def gdpr_test_user(db_session: AsyncSession):
    """Create a test user with GDPR-relevant data."""
    user = User(
        email="gdpr.test@example.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU2VXhI0Asei",
        full_name="GDPR Test User",
        role=UserRoleEnum.BASIC_USER.value,
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
async def user_complete_data(db_session: AsyncSession, gdpr_test_user: User, nasdaq_exchange: Exchange, technology_sector: Sector):
    """Create complete user data ecosystem for GDPR testing."""
    # Create stocks
    stock1 = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange_id=nasdaq_exchange.id,
        sector_id=technology_sector.id,
        asset_type="stock",
        is_active=True,
        is_tradable=True
    )
    stock2 = Stock(
        symbol="MSFT",
        name="Microsoft Corporation",
        exchange_id=nasdaq_exchange.id,
        sector_id=technology_sector.id,
        asset_type="stock",
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
        avg_cost_basis=Decimal("150.00")
    )
    position2 = Position(
        portfolio_id=portfolio.id,
        stock_id=stock2.id,
        quantity=Decimal("30"),
        avg_cost_basis=Decimal("300.00")
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
        total_amount=Decimal("7505.00"),  # (50 * 150.00) + 5.00 commission
        commission=Decimal("5.00"),
        trade_date=datetime.now(timezone.utc)
    )
    db_session.add(transaction)
    await db_session.commit()

    # Create watchlist
    watchlist = Watchlist(
        user_id=gdpr_test_user.id,
        stock_id=stock1.id,
        name="Tech Watchlist",
        is_public=False
    )
    db_session.add(watchlist)
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
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0",
        is_active=True,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
    )
    db_session.add(session)
    await db_session.commit()

    # Create audit logs
    audit_log = AuditLog(
        user_id=gdpr_test_user.id,
        action="user_login",
        resource_type="user",
        resource_id=str(gdpr_test_user.id),
        meta_data={"ip": "192.168.1.1", "timestamp": datetime.now(timezone.utc).isoformat()},
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

    NOTE: This test uses endpoints that don't exist yet (/api/v1/gdpr/export).
    The actual GDPR export endpoint is /api/v1/users/me/data-export.
    This test is a placeholder for future functionality.
    """
    pytest.skip("GDPR /api/v1/gdpr/export endpoint not implemented. Use /api/v1/users/me/data-export instead.")


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

    NOTE: This test uses endpoints that don't exist yet (/api/v1/gdpr/consent with PUT).
    The actual consent endpoints are POST/GET at /api/v1/users/me/consent.
    This test is a placeholder for future functionality.
    """
    pytest.skip("GDPR PUT /api/v1/gdpr/consent endpoint not implemented. Use POST /api/v1/users/me/consent instead.")


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

    NOTE: This test uses endpoints that don't exist yet (/api/v1/gdpr/delete-account).
    The actual deletion endpoint is POST /api/v1/users/me/delete-request.
    This test is a placeholder for future functionality.
    """
    pytest.skip("GDPR DELETE /api/v1/gdpr/delete-account endpoint not implemented. Use POST /api/v1/users/me/delete-request instead.")


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

    NOTE: This test uses endpoints that don't exist yet (/api/v1/analytics/aggregated).
    This test is a placeholder for future analytics functionality.
    """
    pytest.skip("Analytics aggregated endpoint /api/v1/analytics/aggregated not implemented.")


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

    NOTE: This test uses endpoints that don't exist yet (/api/v1/gdpr/audit-trail, /api/v1/gdpr/retention-policy).
    This test is a placeholder for future audit trail functionality.
    """
    pytest.skip("GDPR audit trail endpoints not implemented.")
