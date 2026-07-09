"""
Market API Router - Public market data endpoints.

Serves real, build-up-over-time market data for the React frontend's
``/api/v1/market/*`` calls. These routes are PUBLIC (no auth), matching the
public stocks routes.

Data policy (PRD audit 2026-04, F-02-003 / F-03-003):
    No fabricated financial numbers. Indices come from real ETF-proxy quotes
    (cache-first, rate-limited via cost_monitor); movers / heatmap / breadth /
    sectors are computed from the stocks + price_history we have ingested so
    far; news reuses the real multi-provider news service. Where we have no
    real data yet, endpoints return graceful empty payloads (empty arrays /
    null breadth). Representative economic-calendar data is returned only when
    ``settings.DEMO_MODE`` is explicitly enabled.

Business logic lives in ``backend.services.market_service``; this module
contains only route definitions and thin handlers that delegate to it.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.database import get_async_db_session
from backend.config.settings import settings
from backend.models.api_response import ApiResponse, success_response
from backend.services.market_service import market_service
from backend.services.news_service import fetch_news

logger = logging.getLogger(__name__)

router = APIRouter(tags=["market"])


# ---------------------------------------------------------------------------
# News mapping helper
# ---------------------------------------------------------------------------

def _to_market_news(article: Dict[str, Any]) -> Dict[str, Any]:
    """Map a news_service article dict to the frontend MarketNews shape."""
    published_at = article.get("published_at")
    if hasattr(published_at, "isoformat"):
        published_at = published_at.isoformat()

    return {
        "id": article.get("id", ""),
        "title": article.get("title", ""),
        "summary": article.get("description") or "",
        "url": article.get("url", ""),
        "source": article.get("source", ""),
        "publishedAt": published_at,
        "sentiment": article.get("sentiment") or "neutral",
        "relatedTickers": article.get("related_symbols", []),
        "image": article.get("image_url"),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/overview")
async def get_market_overview(
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[Dict[str, Any]]:
    """
    Aggregate market snapshot: indices, top gainers/losers, most active, breadth.

    All fields derive from real data; absent data yields empty arrays / null
    breadth so the frontend can render graceful empty states.
    """
    try:
        indices = await market_service.get_indices(db=db)
        movers = await market_service.get_movers(db=db)
        breadth = await market_service.get_market_breadth(db=db)

        return success_response(data={
            "indices": indices,
            "topGainers": movers["gainers"],
            "topLosers": movers["losers"],
            "mostActive": movers["active"],
            "marketBreadth": breadth,
        })
    except Exception as e:
        logger.error(f"Error building market overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market overview: {str(e)}",
        )


@router.get("/indices")
async def get_market_indices(
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[Dict[str, Any]]]:
    """Return MarketIndex entries for the SPY/QQQ/DIA/IWM ETF proxies."""
    try:
        indices = await market_service.get_indices(db=db)
        return success_response(data=indices)
    except Exception as e:
        logger.error(f"Error retrieving market indices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market indices: {str(e)}",
        )


@router.get("/movers")
async def get_market_movers(
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[Dict[str, List[Dict[str, Any]]]]:
    """Return {gainers, losers, active} computed from ingested stocks."""
    try:
        movers = await market_service.get_movers(db=db)
        return success_response(data=movers)
    except Exception as e:
        logger.error(f"Error retrieving market movers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market movers: {str(e)}",
        )


@router.get("/sectors")
async def get_market_sectors(
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[Dict[str, Any]]]:
    """Return SectorPerformance entries (reuses the stocks sector aggregation)."""
    try:
        sectors = await market_service.get_sector_performance(db=db)
        return success_response(data=sectors)
    except Exception as e:
        logger.error(f"Error retrieving sector performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sector performance: {str(e)}",
        )


@router.get("/news")
async def get_market_news(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of articles"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[Dict[str, Any]]]:
    """
    Return market-wide MarketNews from the real multi-provider news service.

    ``category`` is accepted for API symmetry; the underlying providers serve
    general market news, so it is not used to filter today.
    """
    try:
        raw_articles = await fetch_news(symbols=None, limit=limit)
        articles = [_to_market_news(a) for a in raw_articles]
        return success_response(data=articles)
    except Exception as e:
        logger.error(f"Error retrieving market news: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market news: {str(e)}",
        )


@router.get("/heatmap")
async def get_market_heatmap(
    index: Optional[str] = Query(None, description="Optional index filter (symmetry only)"),
    sector: Optional[str] = Query(None, description="Optional sector filter"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[Dict[str, Any]]]:
    """Return HeatmapItem entries for ingested stocks (optionally sector-filtered)."""
    try:
        items = await market_service.get_heatmap(db=db, sector=sector)
        return success_response(data=items)
    except Exception as e:
        logger.error(f"Error retrieving market heatmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market heatmap: {str(e)}",
        )


@router.get("/calendar")
async def get_economic_calendar() -> ApiResponse[List[Dict[str, Any]]]:
    """
    Return EconomicEvent entries.

    We have no free economic-calendar provider, so production returns ``[]``.
    Representative calendar data is served ONLY when ``settings.DEMO_MODE`` is
    explicitly enabled (non-production demo environments), mirroring the DEMO
    gating used by the recommendations router.
    """
    if not settings.DEMO_MODE:
        return success_response(data=[])

    # DEMO-only: clearly representative placeholder events (never in production).
    demo_events: List[Dict[str, Any]] = [
        {
            "date": "2026-06-01",
            "time": "08:30",
            "event": "[DEMO] Nonfarm Payrolls",
            "importance": "high",
            "actual": None,
            "forecast": 180.0,
            "previous": 175.0,
        },
        {
            "date": "2026-06-12",
            "time": "14:00",
            "event": "[DEMO] FOMC Rate Decision",
            "importance": "high",
            "actual": None,
            "forecast": 5.25,
            "previous": 5.25,
        },
    ]
    return success_response(data=demo_events)
