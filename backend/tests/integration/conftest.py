"""
Integration-level test fixtures.

Provides patches for external service dependencies so that integration tests
can run without live API keys or a network connection. The patches are applied
automatically via autouse fixtures where required.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Canonical test articles returned by the mock news service
# ---------------------------------------------------------------------------

_TEST_ARTICLES = [
    {
        "id": f"test_news_{i}",
        "title": f"Market Update: Test News Article {i}",
        "description": "Integration test article",
        "url": f"https://example.com/news/{i}",
        "source": "Reuters",
        "published_at": datetime.now(timezone.utc) - timedelta(hours=i),
        "sentiment": "neutral",
        "sentiment_score": 0.0,
        "related_symbols": ["SPY"],
        "image_url": None,
    }
    for i in range(5)
]

_TEST_ARTICLES_AAPL = [
    {
        "id": f"test_aapl_{i}",
        "title": f"Apple Inc: Test News {i}",
        "description": "AAPL integration test article",
        "url": f"https://example.com/aapl/{i}",
        "source": "Bloomberg",
        "published_at": datetime.now(timezone.utc) - timedelta(hours=i),
        "sentiment": "positive",
        "sentiment_score": 0.5,
        "related_symbols": ["AAPL"],
        "image_url": None,
    }
    for i in range(3)
]

_TEST_SENTIMENT = {
    "symbol": "AAPL",
    "overall_sentiment": "neutral",
    "sentiment_score": 0.0,
    "positive_count": 5,
    "negative_count": 3,
    "neutral_count": 12,
    "analyzed_articles": 20,
}


# ---------------------------------------------------------------------------
# News-service mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_news_service(request):
    """
    Patch backend.services.news_service so integration tests do not make
    real HTTP requests to external news APIs.

    The fixture is autouse=True so it applies to every test in this package,
    but it checks for the 'no_news_mock' marker and skips patching when that
    marker is present (allows specific tests to opt out).
    """
    if request.node.get_closest_marker("no_news_mock"):
        yield
        return

    async def _fake_fetch_news(symbols=None, limit=20):
        if symbols:
            # Return symbol-tagged articles for any requested symbol list
            articles = []
            for i in range(min(5, limit)):
                articles.append({
                    "id": f"test_{symbols[0].lower()}_{i}",
                    "title": f"{symbols[0]}: Test News {i}",
                    "description": "Integration test article",
                    "url": f"https://example.com/{symbols[0].lower()}/{i}",
                    "source": "Reuters",
                    "published_at": datetime.now(timezone.utc) - timedelta(hours=i),
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "related_symbols": [symbols[0]],
                    "image_url": None,
                })
            return articles[:limit]
        return _TEST_ARTICLES[:limit]

    async def _fake_fetch_sentiment(symbol: str):
        return {**_TEST_SENTIMENT, "symbol": symbol.upper()}

    with patch(
        "backend.services.news_service.fetch_news",
        side_effect=_fake_fetch_news,
    ), patch(
        "backend.services.news_service.fetch_sentiment_for_symbol",
        side_effect=_fake_fetch_sentiment,
    ), patch(
        "backend.api.routers.news.fetch_news",
        side_effect=_fake_fetch_news,
    ), patch(
        "backend.api.routers.news.fetch_sentiment_for_symbol",
        side_effect=_fake_fetch_sentiment,
    ):
        yield
