"""
News API Router
Provides financial news and sentiment analysis endpoints.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from backend.config.database import get_async_db_session
from backend.models.api_response import ApiResponse, success_response, error_response
from backend.auth.oauth2 import get_current_user
from backend.models.tables import User
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


# Pydantic Models
class NewsArticle(BaseModel):
    """News article response model"""
    id: str
    title: str
    description: Optional[str] = None
    url: str
    source: str
    published_at: datetime
    sentiment: Optional[str] = None  # positive, negative, neutral
    sentiment_score: Optional[float] = None
    related_symbols: List[str] = []
    image_url: Optional[str] = None


class NewsSentiment(BaseModel):
    """News sentiment analysis"""
    symbol: str
    overall_sentiment: str
    sentiment_score: float
    positive_count: int
    negative_count: int
    neutral_count: int
    analyzed_articles: int


@router.get("/latest")
async def get_latest_news(
    limit: int = Query(default=20, ge=1, le=100),
    symbols: Optional[str] = Query(default=None, description="Comma-separated stock symbols"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[List[NewsArticle]]:
    """
    Get latest financial news articles.

    Args:
        limit: Number of articles to return (1-100)
        symbols: Filter by stock symbols (comma-separated, e.g., "AAPL,MSFT,GOOGL")
        current_user: Authenticated user
        db: Database session

    Returns:
        List of news articles with sentiment analysis
    """
    try:
        # Parse symbols if provided
        symbol_list = [s.strip().upper() for s in symbols.split(",")] if symbols else []

        # TODO: Implement actual news fetching from NewsAPI
        # For now, return mock data to maintain API contract
        mock_articles = [
            NewsArticle(
                id=f"news_{i}",
                title=f"Market Update: {'Stock' if symbol_list else 'General Market'} News {i}",
                description="Latest market developments and analysis",
                url="https://example.com/news",
                source="Reuters",
                published_at=datetime.utcnow() - timedelta(hours=i),
                sentiment="neutral",
                sentiment_score=0.0,
                related_symbols=symbol_list[:1] if symbol_list else ["SPY"],
                image_url=None
            )
            for i in range(min(limit, 5))
        ]

        logger.info(f"User {current_user.id} fetched {len(mock_articles)} news articles")
        return success_response(data=mock_articles)

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch news articles"
        )


@router.get("/sentiment/{symbol}")
async def get_news_sentiment(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[NewsSentiment]:
    """
    Get aggregated news sentiment for a stock symbol.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        days: Number of days to analyze (1-30)
        current_user: Authenticated user
        db: Database session

    Returns:
        Aggregated sentiment analysis
    """
    try:
        symbol = symbol.upper()

        # TODO: Implement actual sentiment analysis
        # For now, return mock data
        mock_sentiment = NewsSentiment(
            symbol=symbol,
            overall_sentiment="neutral",
            sentiment_score=0.0,
            positive_count=5,
            negative_count=3,
            neutral_count=12,
            analyzed_articles=20
        )

        logger.info(f"User {current_user.id} fetched sentiment for {symbol}")
        return success_response(data=mock_sentiment)

    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch sentiment for {symbol}"
        )


@router.get("/sources")
async def get_news_sources(
    current_user: User = Depends(get_current_user)
) -> ApiResponse[List[dict]]:
    """
    Get available news sources.

    Args:
        current_user: Authenticated user

    Returns:
        List of news sources with metadata
    """
    sources = [
        {"id": "reuters", "name": "Reuters", "category": "general"},
        {"id": "bloomberg", "name": "Bloomberg", "category": "business"},
        {"id": "wsj", "name": "Wall Street Journal", "category": "business"},
        {"id": "cnbc", "name": "CNBC", "category": "business"},
        {"id": "ft", "name": "Financial Times", "category": "business"},
    ]

    return success_response(data=sources)


@router.post("/preferences")
async def set_news_preferences(
    sources: List[str],
    topics: List[str],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[dict]:
    """
    Set user news preferences.

    Args:
        sources: Preferred news sources
        topics: Preferred topics
        current_user: Authenticated user
        db: Database session

    Returns:
        Confirmation message
    """
    try:
        # TODO: Store preferences in database
        logger.info(f"User {current_user.id} updated news preferences")

        return success_response(data={
            "message": "News preferences updated successfully",
            "sources": sources,
            "topics": topics
        })

    except Exception as e:
        logger.error(f"Error updating news preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update news preferences"
        )
