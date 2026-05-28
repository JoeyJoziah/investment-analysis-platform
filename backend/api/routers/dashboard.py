"""
Dashboard API Router - Authenticated aggregate endpoint.

Serves the single ``GET /api/v1/dashboard`` aggregate consumed by the React
frontend's ``fetchDashboardData`` thunk (see
``frontend/web/src/store/slices/dashboardSlice.ts``). The slice's
``fetchDashboardData.fulfilled`` reducer reads these top-level keys off
``response.data``:

    marketOverview      -> { indices, heatmap, sectors }
    topRecommendations  -> Recommendation[]
    portfolioSummary    -> { ...portfolio aggregate... } | null
    recentNews          -> NewsItem[]
    marketSentiment     -> { overall, score, breakdown } | null
    costMetrics         -> CostMetricsState | null

This endpoint REQUIRES authentication (portfolio data is user-scoped) and
aggregates exclusively from REAL sources (cache-first), returning graceful
empty / null pieces where data is absent. No fabricated financial numbers.

Note on shapes: the dashboard ``marketOverview`` uses a *simpler* MarketIndex
({symbol, value, change, changePercent}) and adds ``heatmap`` + ``sectors``,
which is intentionally different from the richer ``/api/v1/market/overview``
payload. We map down to exactly what this slice consumes.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.oauth2 import get_current_user
from backend.config.database import get_async_db_session
from backend.models.api_response import ApiResponse, success_response
from backend.models.unified_models import User
from backend.repositories import recommendation_repository
from backend.services.market_service import market_service
from backend.services.news_service import fetch_news
from backend.services.portfolio_service import portfolio_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])


# ---------------------------------------------------------------------------
# Mapping helpers (each returns the exact shape the dashboard slice reducers
# read; all derive from real data with graceful empties).
# ---------------------------------------------------------------------------

async def _build_market_overview(db: AsyncSession) -> Dict[str, Any]:
    """
    Build the dashboard's marketOverview block.

    The dashboard slice expects the SIMPLER MarketIndex shape plus heatmap and
    sectors. We map the rich market_service outputs down to those keys.
    """
    indices = await market_service.get_indices(db=db)
    heatmap = await market_service.get_heatmap(db=db)
    sectors = await market_service.get_sector_performance(db=db)

    simple_indices = [
        {
            "symbol": idx["symbol"],
            "value": idx["value"],
            "change": idx["change"],
            "changePercent": idx["changePercent"],
        }
        for idx in indices
    ]

    simple_heatmap = [
        {
            "symbol": item["ticker"],
            "sector": item["sector"],
            "change": item["changePercent"],
            "changePercent": item["changePercent"],
            "marketCap": item["marketCap"],
        }
        for item in heatmap
    ]

    simple_sectors = [
        {
            "name": s["sector"],
            "change": s["changePercent"],
            "volume": s["volume"],
        }
        for s in sectors
    ]

    return {
        "indices": simple_indices,
        "heatmap": simple_heatmap,
        "sectors": simple_sectors,
    }


def _recommendation_to_dict(rec: Any) -> Dict[str, Any]:
    """Map a Recommendation ORM row to the dashboard Recommendation shape."""
    stock = getattr(rec, "stock", None)
    ticker = getattr(stock, "symbol", "") if stock else ""

    created_at = getattr(rec, "created_at", None)
    if hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()

    target_price = getattr(rec, "target_price", None)
    entry_price = getattr(rec, "entry_price", None)

    return {
        "id": getattr(rec, "recommendation_id", "") or str(getattr(rec, "id", "")),
        "ticker": ticker,
        "action": (getattr(rec, "action", "") or "").upper(),
        "confidence": float(getattr(rec, "confidence", 0.0) or 0.0),
        "targetPrice": float(target_price) if target_price is not None else 0.0,
        "currentPrice": float(entry_price) if entry_price is not None else 0.0,
        "rationale": getattr(rec, "reasoning", "") or "",
        "createdAt": created_at,
    }


async def _build_top_recommendations(db: AsyncSession) -> List[Dict[str, Any]]:
    """Fetch active stored recommendations (real DB rows) for the highlights."""
    try:
        recs = await recommendation_repository.get_top_recommendations(
            min_confidence=0.0, limit=5, session=db
        )
        return [_recommendation_to_dict(r) for r in recs]
    except Exception as exc:
        logger.warning(f"Could not load top recommendations for dashboard: {exc}")
        return []


async def _build_portfolio_summary(user_id: int, db: AsyncSession) -> Optional[Dict[str, Any]]:
    """
    Build the dashboard portfolioSummary block from the user's real portfolios.

    Aggregates across the user's portfolio summaries (reusing the portfolio
    service). Price-derived sub-blocks we do not compute here (performance
    history, per-position gainers/losers, allocation, risk metrics) report
    null-safe empty/zero values rather than fabricated numbers -- the frontend
    renders graceful empty states for them. Returns ``None`` when the user has
    no portfolio data.
    """
    try:
        summaries = await portfolio_service.compute_portfolio_summaries(
            user_id=user_id, db=db
        )
    except Exception as exc:
        logger.warning(f"Could not load portfolio summaries for dashboard: {exc}")
        return None

    if not summaries:
        return None

    total_value = sum(float(s.get("total_value", 0) or 0) for s in summaries)
    total_cost = sum(float(s.get("total_cost", 0) or 0) for s in summaries)
    total_return = total_value - total_cost
    total_return_percent = round((total_return / total_cost * 100), 4) if total_cost else 0.0
    day_change = sum(float(s.get("day_change", 0) or 0) for s in summaries)
    day_change_percent = (
        round((day_change / total_value * 100), 4) if total_value else 0.0
    )
    active_positions = sum(int(s.get("positions_count", 0) or 0) for s in summaries)
    cash_balance = sum(float(s.get("cash_balance", 0) or 0) for s in summaries)

    return {
        "totalValue": round(total_value, 4),
        "totalCost": round(total_cost, 4),
        "totalReturn": round(total_return, 4),
        "totalReturnPercent": total_return_percent,
        "dayChange": round(day_change, 4),
        "dayChangePercent": day_change_percent,
        # Multi-period changes require historical snapshots we do not aggregate
        # here; report 0 rather than fabricate.
        "weekChange": 0.0,
        "monthChange": 0.0,
        "yearChange": 0.0,
        "activePositions": active_positions,
        "performanceHistory": [],
        "topGainers": [],
        "topLosers": [],
        "allocation": [],
        "riskMetrics": {
            "sharpeRatio": 0.0,
            "beta": 0.0,
            "standardDeviation": 0.0,
            "maxDrawdown": 0.0,
        },
        "diversificationScore": 0.0,
        "cashBalance": round(cash_balance, 4),
        "marginUsed": 0.0,
    }


async def _build_recent_news() -> List[Dict[str, Any]]:
    """Fetch recent market news (real providers) in the dashboard NewsItem shape."""
    try:
        raw = await fetch_news(symbols=None, limit=10)
    except Exception as exc:
        logger.warning(f"Could not load news for dashboard: {exc}")
        return []

    items: List[Dict[str, Any]] = []
    for a in raw:
        published_at = a.get("published_at")
        if hasattr(published_at, "isoformat"):
            published_at = published_at.isoformat()
        items.append({
            "id": a.get("id", ""),
            "title": a.get("title", ""),
            "summary": a.get("description") or "",
            "source": a.get("source", ""),
            "url": a.get("url", ""),
            "publishedAt": published_at,
            "sentiment": a.get("sentiment") or "neutral",
            "tickers": a.get("related_symbols", []),
        })
    return items


def _build_market_sentiment(news_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Derive an aggregate market sentiment from the real fetched news items.

    Returns ``None`` when there is no news to derive sentiment from (frontend
    renders a graceful empty state).
    """
    if not news_items:
        return None

    positive = sum(1 for n in news_items if n.get("sentiment") == "positive")
    negative = sum(1 for n in news_items if n.get("sentiment") == "negative")
    neutral = sum(1 for n in news_items if n.get("sentiment") == "neutral")
    total = len(news_items)

    score = round((positive - negative) / total, 4) if total else 0.0
    if score > 0.1:
        overall = "positive"
    elif score < -0.1:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "score": score,
        "breakdown": {
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
        },
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("")
async def get_dashboard(
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """
    Aggregate dashboard payload for the authenticated user.

    Combines a real market snapshot, the user's portfolio aggregate, active
    recommendations, recent news, and a news-derived market sentiment. Cost
    metrics have no real source wired into this aggregate (the frontend fetches
    them separately from ``/api/v1/admin/metrics``), so ``costMetrics`` is
    returned as ``null`` here rather than fabricated.
    """
    try:
        market_overview = await _build_market_overview(db)
        top_recommendations = await _build_top_recommendations(db)
        portfolio_summary = await _build_portfolio_summary(current_user.id, db)
        recent_news = await _build_recent_news()
        market_sentiment = _build_market_sentiment(recent_news)

        return success_response(data={
            "marketOverview": market_overview,
            "topRecommendations": top_recommendations,
            "portfolioSummary": portfolio_summary,
            "recentNews": recent_news,
            "marketSentiment": market_sentiment,
            "costMetrics": None,
        })
    except Exception as e:
        logger.error(f"Error building dashboard for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving dashboard data: {str(e)}",
        )
