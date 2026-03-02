"""
News Service
Fetches financial news from multiple providers with fallback chain.
Provider priority: Finnhub -> NewsAPI -> MarketAux
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import httpx

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache (TTL-based fallback when Redis is unavailable)
# ---------------------------------------------------------------------------

_mem_cache: Dict[str, Tuple[float, object]] = {}
_NEWS_CACHE_TTL = 900  # 15 minutes


def _mem_cache_get(key: str) -> Optional[object]:
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.monotonic() > expires_at:
        del _mem_cache[key]
        return None
    return value


def _mem_cache_set(key: str, value: object, ttl: int = _NEWS_CACHE_TTL) -> None:
    _mem_cache[key] = (time.monotonic() + ttl, value)


# ---------------------------------------------------------------------------
# Typed response models (plain dicts, no Pydantic dependency here)
# ---------------------------------------------------------------------------

def _make_article(
    article_id: str,
    title: str,
    url: str,
    source: str,
    published_at: datetime,
    description: Optional[str] = None,
    sentiment: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    related_symbols: Optional[List[str]] = None,
    image_url: Optional[str] = None,
) -> Dict:
    return {
        "id": article_id,
        "title": title,
        "description": description,
        "url": url,
        "source": source,
        "published_at": published_at,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "related_symbols": related_symbols or [],
        "image_url": image_url,
    }


def _stable_id(*parts: str) -> str:
    """Deterministic article ID from URL or other stable parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Provider fetchers
# ---------------------------------------------------------------------------

async def _fetch_from_finnhub(
    symbols: List[str],
    limit: int,
    client: httpx.AsyncClient,
) -> List[Dict]:
    """Fetch news from Finnhub API (supports both company and market news)."""
    api_key = settings.FINNHUB_API_KEY
    if not api_key:
        logger.warning("FINNHUB_API_KEY not configured - skipping Finnhub news")
        return []

    articles: List[Dict] = []
    base_url = "https://finnhub.io/api/v1"
    today = datetime.now(timezone.utc)
    from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    try:
        if symbols:
            # Fetch company-specific news for each requested symbol
            for symbol in symbols[:3]:  # Limit to 3 symbols to avoid rate limits
                resp = await client.get(
                    f"{base_url}/company-news",
                    params={"symbol": symbol, "from": from_date, "to": to_date, "token": api_key},
                    timeout=10,
                )
                if resp.status_code == 200:
                    items = resp.json()
                    for item in items[:limit]:
                        published_ts = item.get("datetime", 0)
                        published_at = (
                            datetime.fromtimestamp(published_ts, tz=timezone.utc)
                            if published_ts
                            else datetime.now(timezone.utc)
                        )
                        articles.append(
                            _make_article(
                                article_id=_stable_id(item.get("url", ""), str(item.get("id", ""))),
                                title=item.get("headline", ""),
                                url=item.get("url", ""),
                                source=item.get("source", "Finnhub"),
                                published_at=published_at,
                                description=item.get("summary"),
                                related_symbols=[symbol],
                                image_url=item.get("image"),
                            )
                        )
        else:
            # General market news
            resp = await client.get(
                f"{base_url}/news",
                params={"category": "general", "token": api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                items = resp.json()
                for item in items[:limit]:
                    published_ts = item.get("datetime", 0)
                    published_at = (
                        datetime.fromtimestamp(published_ts, tz=timezone.utc)
                        if published_ts
                        else datetime.now(timezone.utc)
                    )
                    related = []
                    if item.get("related"):
                        related = [s.strip() for s in item["related"].split(",") if s.strip()]
                    articles.append(
                        _make_article(
                            article_id=_stable_id(item.get("url", ""), str(item.get("id", ""))),
                            title=item.get("headline", ""),
                            url=item.get("url", ""),
                            source=item.get("source", "Finnhub"),
                            published_at=published_at,
                            description=item.get("summary"),
                            related_symbols=related,
                            image_url=item.get("image"),
                        )
                    )
    except Exception as exc:
        logger.warning(f"Finnhub news fetch failed: {exc}")

    return articles


async def _fetch_from_newsapi(
    symbols: List[str],
    limit: int,
    client: httpx.AsyncClient,
) -> List[Dict]:
    """Fetch news from NewsAPI.org."""
    api_key = settings.NEWS_API_KEY
    if not api_key:
        logger.warning("NEWS_API_KEY not configured - skipping NewsAPI")
        return []

    articles: List[Dict] = []
    base_url = "https://newsapi.org/v2"
    query = " OR ".join(symbols) if symbols else "stock market finance"

    try:
        resp = await client.get(
            f"{base_url}/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": min(limit, 100),
                "apiKey": api_key,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("articles", [])[:limit]:
                published_str = item.get("publishedAt", "")
                try:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    published_at = datetime.now(timezone.utc)

                source_name = (
                    item.get("source", {}).get("name", "NewsAPI")
                    if isinstance(item.get("source"), dict)
                    else "NewsAPI"
                )
                articles.append(
                    _make_article(
                        article_id=_stable_id(item.get("url", "")),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        source=source_name,
                        published_at=published_at,
                        description=item.get("description"),
                        related_symbols=symbols[:1] if symbols else [],
                        image_url=item.get("urlToImage"),
                    )
                )
        else:
            logger.warning(f"NewsAPI returned status {resp.status_code}")
    except Exception as exc:
        logger.warning(f"NewsAPI news fetch failed: {exc}")

    return articles


async def _fetch_from_marketaux(
    symbols: List[str],
    limit: int,
    client: httpx.AsyncClient,
) -> List[Dict]:
    """Fetch news from MarketAux API."""
    api_key = settings.MARKETAUX_API_KEY
    if not api_key:
        logger.warning("MARKETAUX_API_KEY not configured - skipping MarketAux")
        return []

    articles: List[Dict] = []
    base_url = "https://api.marketaux.com/v1"

    params: Dict = {
        "api_token": api_key,
        "language": "en",
        "limit": min(limit, 50),
    }
    if symbols:
        params["symbols"] = ",".join(symbols)

    try:
        resp = await client.get(
            f"{base_url}/news/all",
            params=params,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("data", [])[:limit]:
                published_str = item.get("published_at", "")
                try:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    published_at = datetime.now(timezone.utc)

                # Extract related symbols from entities
                related: List[str] = []
                for entity in item.get("entities", []):
                    sym = entity.get("symbol")
                    if sym:
                        related.append(sym)

                articles.append(
                    _make_article(
                        article_id=_stable_id(item.get("url", ""), item.get("uuid", "")),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        source=item.get("source", "MarketAux"),
                        published_at=published_at,
                        description=item.get("description"),
                        related_symbols=related or symbols[:1],
                        image_url=item.get("image_url"),
                    )
                )
        else:
            logger.warning(f"MarketAux returned status {resp.status_code}")
    except Exception as exc:
        logger.warning(f"MarketAux news fetch failed: {exc}")

    return articles


# ---------------------------------------------------------------------------
# Sentiment scoring helpers
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset({
    "surge", "soar", "rally", "gain", "profit", "beat", "record", "growth",
    "rise", "up", "bull", "boost", "strong", "exceed", "outperform",
})
_NEGATIVE_WORDS = frozenset({
    "fall", "drop", "decline", "loss", "miss", "crash", "plunge", "weak",
    "down", "bear", "cut", "layoff", "bankrupt", "warning", "disappoint",
})


def _score_sentiment(text: str) -> Tuple[str, float]:
    """Basic keyword-based sentiment scoring for an article title/description."""
    if not text:
        return "neutral", 0.0

    words = text.lower().split()
    positive = sum(1 for w in words if w in _POSITIVE_WORDS)
    negative = sum(1 for w in words if w in _NEGATIVE_WORDS)

    if positive == 0 and negative == 0:
        return "neutral", 0.0

    total = positive + negative
    score = (positive - negative) / total  # range: -1.0 to 1.0

    if score > 0.2:
        return "positive", round(score, 3)
    if score < -0.2:
        return "negative", round(score, 3)
    return "neutral", round(score, 3)


def _enrich_with_sentiment(articles: List[Dict]) -> List[Dict]:
    """Add sentiment scoring to articles that lack it."""
    enriched = []
    for article in articles:
        if article.get("sentiment") is None:
            text = f"{article.get('title', '')} {article.get('description', '') or ''}"
            sentiment, score = _score_sentiment(text)
            article = {**article, "sentiment": sentiment, "sentiment_score": score}
        enriched.append(article)
    return enriched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_news(
    symbols: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Fetch financial news articles using fallback chain:
    Finnhub -> NewsAPI -> MarketAux.

    Results are cached in-memory for 15 minutes per (symbols, limit) key.

    Args:
        symbols: Optional list of stock symbols to filter news.
        limit: Maximum number of articles to return.

    Returns:
        List of article dicts matching the NewsArticle response schema.
    """
    symbol_key = ",".join(sorted(symbols)) if symbols else "market"
    cache_key = f"news:{symbol_key}:{limit}"

    cached = _mem_cache_get(cache_key)
    if cached is not None:
        logger.debug(f"News cache hit for key={cache_key}")
        return cached  # type: ignore[return-value]

    articles: List[Dict] = []

    async with httpx.AsyncClient() as client:
        # Tier 1: Finnhub
        articles = await _fetch_from_finnhub(symbols or [], limit, client)
        if articles:
            logger.info(f"Fetched {len(articles)} articles from Finnhub")

        # Tier 2: NewsAPI (if Finnhub returned nothing)
        if not articles:
            articles = await _fetch_from_newsapi(symbols or [], limit, client)
            if articles:
                logger.info(f"Fetched {len(articles)} articles from NewsAPI")

        # Tier 3: MarketAux (if both above returned nothing)
        if not articles:
            articles = await _fetch_from_marketaux(symbols or [], limit, client)
            if articles:
                logger.info(f"Fetched {len(articles)} articles from MarketAux")

    if not articles:
        logger.warning("All news providers returned no results - returning empty list")

    # Enrich with sentiment and deduplicate by ID
    articles = _enrich_with_sentiment(articles)
    seen_ids: set = set()
    unique_articles: List[Dict] = []
    for article in articles:
        aid = article["id"]
        if aid not in seen_ids:
            seen_ids.add(aid)
            unique_articles.append(article)

    result = unique_articles[:limit]
    _mem_cache_set(cache_key, result, ttl=_NEWS_CACHE_TTL)
    return result


async def fetch_sentiment_for_symbol(symbol: str) -> Dict:
    """
    Fetch and aggregate news sentiment for a stock symbol.

    Tries Finnhub's dedicated news-sentiment endpoint first,
    then falls back to analysing fetched articles manually.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Dict with overall_sentiment, sentiment_score, and article counts.
    """
    cache_key = f"sentiment:{symbol}"
    cached = _mem_cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    api_key = settings.FINNHUB_API_KEY

    if api_key:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://finnhub.io/api/v1/news-sentiment",
                    params={"symbol": symbol, "token": api_key},
                    timeout=10,
                )
            if resp.status_code == 200:
                data = resp.json()
                buzz = data.get("buzz", {})
                sentiment_data = data.get("sentiment", {})
                bullish = sentiment_data.get("bullishPercent", 0.5)
                bearish = sentiment_data.get("bearishPercent", 0.5)
                score = round(bullish - bearish, 3)

                if score > 0.1:
                    overall = "positive"
                elif score < -0.1:
                    overall = "negative"
                else:
                    overall = "neutral"

                total = buzz.get("articlesInLastWeek", 0)
                positive_count = round(total * bullish)
                negative_count = round(total * bearish)
                neutral_count = max(0, total - positive_count - negative_count)

                result = {
                    "symbol": symbol,
                    "overall_sentiment": overall,
                    "sentiment_score": score,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "neutral_count": neutral_count,
                    "analyzed_articles": total,
                }
                _mem_cache_set(cache_key, result, ttl=3600)
                return result
        except Exception as exc:
            logger.warning(f"Finnhub sentiment endpoint failed for {symbol}: {exc}")

    # Fallback: score articles ourselves
    articles = await fetch_news(symbols=[symbol], limit=20)
    positive_count = sum(1 for a in articles if a.get("sentiment") == "positive")
    negative_count = sum(1 for a in articles if a.get("sentiment") == "negative")
    neutral_count = sum(1 for a in articles if a.get("sentiment") == "neutral")
    total = len(articles)

    if total == 0:
        score = 0.0
        overall = "neutral"
    else:
        score = round((positive_count - negative_count) / total, 3)
        if score > 0.1:
            overall = "positive"
        elif score < -0.1:
            overall = "negative"
        else:
            overall = "neutral"

    result = {
        "symbol": symbol,
        "overall_sentiment": overall,
        "sentiment_score": score,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "analyzed_articles": total,
    }
    _mem_cache_set(cache_key, result, ttl=3600)
    return result
