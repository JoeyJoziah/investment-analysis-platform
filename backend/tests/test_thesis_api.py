"""
Tests for Investment Thesis API endpoints
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal

from backend.models.thesis import InvestmentThesis
from backend.models.unified_models import User, Stock
from backend.tests.conftest import assert_success_response, assert_api_error_response


@pytest_asyncio.fixture
async def test_exchange(db_session: AsyncSession):
    """Create a test exchange"""
    from backend.models.unified_models import Exchange
    exchange = Exchange(
        code="NASDAQ",
        name="NASDAQ Stock Market",
        timezone="America/New_York",
        country="US",
        currency="USD",
        market_open="09:30",
        market_close="16:00"
    )
    db_session.add(exchange)
    await db_session.commit()
    await db_session.refresh(exchange)
    return exchange


@pytest_asyncio.fixture
async def test_stock(db_session: AsyncSession, test_exchange) -> Stock:
    """Create a test stock"""
    stock = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange_id=test_exchange.id,  # Fixed: Use exchange_id ForeignKey
        asset_type="stock",
        country="US",
        currency="USD"
    )
    db_session.add(stock)
    await db_session.commit()
    await db_session.refresh(stock)
    return stock


@pytest_asyncio.fixture
async def test_thesis(db_session: AsyncSession, test_user: User, test_stock: Stock) -> InvestmentThesis:
    """Create a test investment thesis"""
    thesis = InvestmentThesis(
        user_id=test_user.id,
        stock_id=test_stock.id,
        investment_objective="Long-term growth investment in technology leader",
        time_horizon="long-term",
        target_price=Decimal("200.00"),
        business_model="Hardware, software, and services ecosystem",
        competitive_advantages="Brand loyalty, ecosystem lock-in, vertical integration",
        financial_health="Strong balance sheet with significant cash reserves",
        growth_drivers="Services growth, AI integration, emerging markets",
        risks="Regulatory risks, supply chain dependencies, market saturation",
        valuation_rationale="Trading at 25x P/E, justified by 15% revenue growth",
        exit_strategy="Sell if growth slows below 10% or P/E exceeds 30x",
        content="# Investment Thesis: Apple Inc.\n\nStrong buy for long-term portfolio.",
        version=1
    )
    db_session.add(thesis)
    await db_session.commit()
    await db_session.refresh(thesis)
    return thesis


class TestInvestmentThesisAPI:
    """Test suite for Investment Thesis API endpoints"""

    @pytest.mark.asyncio
    async def test_create_thesis_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_stock: Stock
    ):
        """Test successful thesis creation"""
        thesis_data = {
            "stock_id": test_stock.id,
            "investment_objective": "Growth investment with 3-5 year horizon",
            "time_horizon": "medium-term",
            "target_price": 150.50,
            "business_model": "Tech company with strong moat",
            "competitive_advantages": "Network effects and brand",
            "financial_health": "Excellent",
            "growth_drivers": "New product launches",
            "risks": "Competition and regulation",
            "valuation_rationale": "DCF valuation shows upside",
            "exit_strategy": "Exit at target price or 2x return",
            "content": "# Full Thesis Document\n\nDetailed analysis..."
        }

        response = await client.post(
            "/api/v1/thesis/",
            json=thesis_data,
            headers=auth_headers
        )

        data = assert_success_response(response, expected_status=201)
        assert data["stock_id"] == test_stock.id
        assert data["investment_objective"] == thesis_data["investment_objective"]
        assert data["time_horizon"] == thesis_data["time_horizon"]
        assert float(data["target_price"]) == thesis_data["target_price"]
        assert data["version"] == 1
        assert data["stock_symbol"] == test_stock.symbol
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_create_thesis_missing_required_fields(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_stock: Stock
    ):
        """Test thesis creation with missing required fields"""
        thesis_data = {
            "stock_id": test_stock.id,
            # Missing investment_objective and time_horizon
        }

        response = await client.post(
            "/api/v1/thesis/",
            json=thesis_data,
            headers=auth_headers
        )

        assert_api_error_response(response, 422)

    @pytest.mark.asyncio
    async def test_create_thesis_invalid_stock_id(
        self,
        client: AsyncClient,
        auth_headers: dict
    ):
        """Test thesis creation with non-existent stock"""
        thesis_data = {
            "stock_id": 99999,  # Non-existent stock
            "investment_objective": "Growth investment",
            "time_horizon": "long-term"
        }

        response = await client.post(
            "/api/v1/thesis/",
            json=thesis_data,
            headers=auth_headers
        )

        assert_api_error_response(response, 404)

    @pytest.mark.asyncio
    async def test_create_duplicate_thesis(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_thesis: InvestmentThesis
    ):
        """Test that duplicate thesis for same user+stock is rejected"""
        thesis_data = {
            "stock_id": test_thesis.stock_id,
            "investment_objective": "Another thesis",
            "time_horizon": "short-term"
        }

        response = await client.post(
            "/api/v1/thesis/",
            json=thesis_data,
            headers=auth_headers
        )

        assert_api_error_response(response, 409)

    @pytest.mark.asyncio
    async def test_get_thesis_by_id(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_thesis: InvestmentThesis
    ):
        """Test retrieving thesis by ID"""
        response = await client.get(
            f"/api/v1/thesis/{test_thesis.id}",
            headers=auth_headers
        )

        data = assert_success_response(response)
        assert data["id"] == test_thesis.id
        assert data["investment_objective"] == test_thesis.investment_objective
        assert data["stock_symbol"] is not None

    @pytest.mark.asyncio
    async def test_get_thesis_by_stock_id(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_thesis: InvestmentThesis
    ):
        """Test retrieving thesis by stock ID"""
        response = await client.get(
            f"/api/v1/thesis/stock/{test_thesis.stock_id}",
            headers=auth_headers
        )

        data = assert_success_response(response)
        assert data["stock_id"] == test_thesis.stock_id
        assert data["investment_objective"] == test_thesis.investment_objective

    @pytest.mark.asyncio
    async def test_get_thesis_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict
    ):
        """Test retrieving non-existent thesis"""
        response = await client.get(
            "/api/v1/thesis/99999",
            headers=auth_headers
        )

        assert_api_error_response(response, 404)

    @pytest.mark.asyncio
    async def test_list_user_theses(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_thesis: InvestmentThesis
    ):
        """Test listing all theses for a user"""
        response = await client.get(
            "/api/v1/thesis/",
            headers=auth_headers
        )

        data = assert_success_response(response)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(t["id"] == test_thesis.id for t in data)

    @pytest.mark.asyncio
    async def test_list_theses_pagination(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_thesis: InvestmentThesis
    ):
        """Test pagination in thesis listing"""
        response = await client.get(
            "/api/v1/thesis/?limit=10&offset=0",
            headers=auth_headers
        )

        data = assert_success_response(response)
        assert isinstance(data, list)
        assert len(data) <= 10

    @pytest.mark.asyncio
    async def test_update_thesis(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_thesis: InvestmentThesis
    ):
        """Test updating an existing thesis"""
        update_data = {
            "investment_objective": "Updated investment objective",
            "target_price": 250.00,
            "content": "# Updated Content\n\nNew analysis..."
        }

        response = await client.put(
            f"/api/v1/thesis/{test_thesis.id}",
            json=update_data,
            headers=auth_headers
        )

        data = assert_success_response(response)
        assert data["investment_objective"] == update_data["investment_objective"]
        assert float(data["target_price"]) == update_data["target_price"]
        assert data["version"] == test_thesis.version + 1  # Version incremented

    @pytest.mark.asyncio
    async def test_update_thesis_not_owned(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        test_thesis: InvestmentThesis,
        test_stock: Stock
    ):
        """Test that users cannot update theses they don't own"""
        # Create another user
        other_user = User(
            email="other@example.com",
            username="otheruser",
            hashed_password="hashedpassword",
            full_name="Other User",
            role="free_user"
        )
        db_session.add(other_user)
        await db_session.commit()

        # Get token for other user (simplified - in real tests use proper auth)
        from backend.auth.oauth2 import create_access_token
        other_token = create_access_token({"sub": str(other_user.id)})
        other_headers = {"Authorization": f"Bearer {other_token}"}

        update_data = {
            "investment_objective": "Hacked update"
        }

        response = await client.put(
            f"/api/v1/thesis/{test_thesis.id}",
            json=update_data,
            headers=other_headers
        )

        assert response.status_code in [403, 404]
        assert_api_error_response(response, response.status_code)

    @pytest.mark.asyncio
    async def test_delete_thesis(
        self,
        client: AsyncClient,
        auth_headers: dict,
        test_thesis: InvestmentThesis
    ):
        """Test deleting a thesis"""
        response = await client.delete(
            f"/api/v1/thesis/{test_thesis.id}",
            headers=auth_headers
        )

        assert response.status_code == 204

        # Verify thesis is deleted
        get_response = await client.get(
            f"/api/v1/thesis/{test_thesis.id}",
            headers=auth_headers
        )
        assert_api_error_response(get_response, 404)

    @pytest.mark.asyncio
    async def test_delete_thesis_not_owned(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        test_thesis: InvestmentThesis
    ):
        """Test that users cannot delete theses they don't own"""
        # Create another user
        other_user = User(
            email="other2@example.com",
            username="otheruser2",
            hashed_password="hashedpassword",
            full_name="Other User 2",
            role="free_user"
        )
        db_session.add(other_user)
        await db_session.commit()

        from backend.auth.oauth2 import create_access_token
        other_token = create_access_token({"sub": str(other_user.id)})
        other_headers = {"Authorization": f"Bearer {other_token}"}

        response = await client.delete(
            f"/api/v1/thesis/{test_thesis.id}",
            headers=other_headers
        )

        assert response.status_code in [403, 404]
        assert_api_error_response(response, response.status_code)

    @pytest.mark.asyncio
    async def test_thesis_requires_authentication(
        self,
        client: AsyncClient,
        test_thesis: InvestmentThesis
    ):
        """Test that all endpoints require authentication"""
        # Test GET
        response = await client.get(f"/api/v1/thesis/{test_thesis.id}")
        assert_api_error_response(response, 401)

        # Test POST
        response = await client.post(
            "/api/v1/thesis/",
            json={
                "stock_id": 1,
                "investment_objective": "test",
                "time_horizon": "short-term"
            }
        )
        assert_api_error_response(response, 401)

        # Test PUT
        response = await client.put(
            f"/api/v1/thesis/{test_thesis.id}",
            json={"investment_objective": "test"}
        )
        assert_api_error_response(response, 401)

        # Test DELETE
        response = await client.delete(f"/api/v1/thesis/{test_thesis.id}")
        assert_api_error_response(response, 401)
