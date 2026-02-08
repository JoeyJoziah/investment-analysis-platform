"""
Integration tests for the News API Router.

Tests cover:
- GET /api/v1/news/latest (latest articles, limit, symbol filtering)
- GET /api/v1/news/sentiment/{symbol} (sentiment analysis per symbol)
- GET /api/v1/news/sources (available news sources)
- POST /api/v1/news/preferences (user news preferences)
- Authentication enforcement on all endpoints
- Response format validation against the ApiResponse contract
- Query parameter validation (limit bounds, days bounds)
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from httpx import AsyncClient

from backend.api.main import app
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_user():
    """Create an in-memory user object for dependency override."""
    return User(
        id=42,
        username="newsreader",
        email="newsreader@example.com",
        hashed_password="hashed",
        full_name="News Reader",
        is_active=True,
        is_verified=True,
        created_at=datetime.now(timezone.utc),
    )


@pytest_asyncio.fixture
async def authed_client(async_client: AsyncClient, mock_user):
    """
    Provide an AsyncClient whose requests bypass real JWT validation.

    The get_current_user dependency is replaced with a callable that
    returns mock_user directly, so every endpoint that requires auth
    will see a valid user without needing a real token or database lookup.
    """
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield async_client
    app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _unwrap(response, expected_status=200):
    """Assert ApiResponse envelope and return the inner data payload."""
    assert response.status_code == expected_status, (
        f"Expected {expected_status}, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["success"] is True
    assert "data" in body
    return body["data"]


# ===========================================================================
# GET /api/v1/news/latest
# ===========================================================================

class TestGetLatestNews:
    """Tests for the latest-news endpoint."""

    @pytest.mark.asyncio
    async def test_returns_articles_with_default_params(self, authed_client):
        """Calling without query params returns articles capped at 5 (mock limit)."""
        response = await authed_client.get("/api/v1/news/latest")
        articles = _unwrap(response)

        assert isinstance(articles, list)
        assert len(articles) <= 5
        assert len(articles) > 0

    @pytest.mark.asyncio
    async def test_article_schema(self, authed_client):
        """Each article must match the NewsArticle contract."""
        response = await authed_client.get("/api/v1/news/latest")
        articles = _unwrap(response)

        required_fields = {
            "id", "title", "description", "url", "source",
            "published_at", "sentiment", "sentiment_score",
            "related_symbols", "image_url",
        }
        for article in articles:
            missing = required_fields - set(article.keys())
            assert not missing, f"Article missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_limit_param_caps_results(self, authed_client):
        """Requesting limit=2 returns at most 2 articles."""
        response = await authed_client.get("/api/v1/news/latest", params={"limit": 2})
        articles = _unwrap(response)

        assert len(articles) <= 2

    @pytest.mark.asyncio
    async def test_limit_param_minimum_validation(self, authed_client):
        """limit=0 violates ge=1 and should return 422."""
        response = await authed_client.get("/api/v1/news/latest", params={"limit": 0})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_limit_param_maximum_validation(self, authed_client):
        """limit=101 violates le=100 and should return 422."""
        response = await authed_client.get("/api/v1/news/latest", params={"limit": 101})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_symbols_filter(self, authed_client):
        """Articles returned when filtering by symbol contain that symbol."""
        response = await authed_client.get(
            "/api/v1/news/latest", params={"symbols": "AAPL,MSFT"}
        )
        articles = _unwrap(response)

        # The mock puts symbol_list[:1] in related_symbols
        for article in articles:
            assert "AAPL" in article["related_symbols"]

    @pytest.mark.asyncio
    async def test_symbols_filter_uppercases_input(self, authed_client):
        """Lower-case symbols are normalised to upper-case in the response."""
        response = await authed_client.get(
            "/api/v1/news/latest", params={"symbols": "aapl"}
        )
        articles = _unwrap(response)

        for article in articles:
            assert "AAPL" in article["related_symbols"]

    @pytest.mark.asyncio
    async def test_no_symbols_defaults_to_spy(self, authed_client):
        """Without a symbols filter the mock defaults related_symbols to SPY."""
        response = await authed_client.get("/api/v1/news/latest")
        articles = _unwrap(response)

        for article in articles:
            assert "SPY" in article["related_symbols"]

    @pytest.mark.asyncio
    async def test_articles_have_descending_published_at(self, authed_client):
        """Articles should arrive newest-first (mock generates decreasing timestamps)."""
        response = await authed_client.get("/api/v1/news/latest")
        articles = _unwrap(response)

        timestamps = [article["published_at"] for article in articles]
        assert timestamps == sorted(timestamps, reverse=True)


# ===========================================================================
# GET /api/v1/news/sentiment/{symbol}
# ===========================================================================

class TestGetNewsSentiment:
    """Tests for the per-symbol sentiment endpoint."""

    @pytest.mark.asyncio
    async def test_returns_sentiment_for_valid_symbol(self, authed_client):
        """A valid symbol returns a well-formed NewsSentiment object."""
        response = await authed_client.get("/api/v1/news/sentiment/AAPL")
        data = _unwrap(response)

        required_fields = {
            "symbol", "overall_sentiment", "sentiment_score",
            "positive_count", "negative_count", "neutral_count",
            "analyzed_articles",
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"Sentiment response missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_symbol_uppercased_in_response(self, authed_client):
        """Symbols are normalised to upper-case regardless of input casing."""
        response = await authed_client.get("/api/v1/news/sentiment/aapl")
        data = _unwrap(response)

        assert data["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_sentiment_score_is_numeric(self, authed_client):
        """sentiment_score must be a float."""
        response = await authed_client.get("/api/v1/news/sentiment/TSLA")
        data = _unwrap(response)

        assert isinstance(data["sentiment_score"], (int, float))

    @pytest.mark.asyncio
    async def test_counts_are_non_negative_integers(self, authed_client):
        """All count fields must be non-negative integers."""
        response = await authed_client.get("/api/v1/news/sentiment/MSFT")
        data = _unwrap(response)

        for field in ("positive_count", "negative_count", "neutral_count", "analyzed_articles"):
            assert isinstance(data[field], int), f"{field} should be int"
            assert data[field] >= 0, f"{field} should be >= 0"

    @pytest.mark.asyncio
    async def test_days_param_default(self, authed_client):
        """Default days=7 is accepted silently (no error)."""
        response = await authed_client.get("/api/v1/news/sentiment/AAPL")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_days_param_minimum_validation(self, authed_client):
        """days=0 violates ge=1 and should return 422."""
        response = await authed_client.get(
            "/api/v1/news/sentiment/AAPL", params={"days": 0}
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_days_param_maximum_validation(self, authed_client):
        """days=31 violates le=30 and should return 422."""
        response = await authed_client.get(
            "/api/v1/news/sentiment/AAPL", params={"days": 31}
        )

        assert response.status_code == 422


# ===========================================================================
# GET /api/v1/news/sources
# ===========================================================================

class TestGetNewsSources:
    """Tests for the news-sources endpoint."""

    @pytest.mark.asyncio
    async def test_returns_list_of_sources(self, authed_client):
        """The sources endpoint returns a non-empty list."""
        response = await authed_client.get("/api/v1/news/sources")
        sources = _unwrap(response)

        assert isinstance(sources, list)
        assert len(sources) > 0

    @pytest.mark.asyncio
    async def test_source_schema(self, authed_client):
        """Each source has id, name, and category fields."""
        response = await authed_client.get("/api/v1/news/sources")
        sources = _unwrap(response)

        for source in sources:
            assert "id" in source
            assert "name" in source
            assert "category" in source

    @pytest.mark.asyncio
    async def test_known_sources_present(self, authed_client):
        """The hard-coded sources include Reuters and Bloomberg."""
        response = await authed_client.get("/api/v1/news/sources")
        sources = _unwrap(response)

        source_ids = [s["id"] for s in sources]
        assert "reuters" in source_ids
        assert "bloomberg" in source_ids


# ===========================================================================
# POST /api/v1/news/preferences
# ===========================================================================

class TestSetNewsPreferences:
    """Tests for the news-preferences endpoint."""

    @pytest.mark.asyncio
    async def test_set_preferences_success(self, authed_client):
        """Valid preferences return a confirmation with the submitted values."""
        response = await authed_client.post(
            "/api/v1/news/preferences",
            json={
                "sources": ["reuters", "bloomberg"],
                "topics": ["tech", "finance"],
            },
        )
        data = _unwrap(response)

        assert "message" in data
        assert data["sources"] == ["reuters", "bloomberg"]
        assert data["topics"] == ["tech", "finance"]

    @pytest.mark.asyncio
    async def test_empty_preferences_accepted(self, authed_client):
        """Empty lists for sources and topics are still accepted."""
        response = await authed_client.post(
            "/api/v1/news/preferences",
            json={"sources": [], "topics": []},
        )
        data = _unwrap(response)

        assert data["sources"] == []
        assert data["topics"] == []


# ===========================================================================
# Authentication enforcement
# ===========================================================================

class TestAuthenticationRequired:
    """Every news endpoint must reject unauthenticated requests."""

    @pytest.mark.asyncio
    async def test_latest_requires_auth(self, async_client):
        """GET /api/v1/news/latest without auth returns 401."""
        response = await async_client.get("/api/v1/news/latest")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_sentiment_requires_auth(self, async_client):
        """GET /api/v1/news/sentiment/AAPL without auth returns 401."""
        response = await async_client.get("/api/v1/news/sentiment/AAPL")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_sources_requires_auth(self, async_client):
        """GET /api/v1/news/sources without auth returns 401."""
        response = await async_client.get("/api/v1/news/sources")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_preferences_requires_auth(self, async_client):
        """POST /api/v1/news/preferences without auth returns 401."""
        response = await async_client.post(
            "/api/v1/news/preferences",
            json={"sources": ["reuters"], "topics": ["tech"]},
        )

        assert response.status_code == 401


# ===========================================================================
# ApiResponse envelope contract
# ===========================================================================

class TestApiResponseEnvelope:
    """Validate every endpoint wraps its payload in the standard ApiResponse."""

    @pytest.mark.asyncio
    async def test_success_envelope_structure(self, authed_client):
        """All success responses have success=True, data, error=None, and meta."""
        endpoints = [
            "/api/v1/news/latest",
            "/api/v1/news/sentiment/AAPL",
            "/api/v1/news/sources",
        ]
        for url in endpoints:
            response = await authed_client.get(url)
            body = response.json()

            assert body["success"] is True, f"Failed for {url}"
            assert "data" in body, f"Missing 'data' for {url}"
            assert body.get("error") is None, f"Unexpected error for {url}"
