"""
Market Service
Business logic for the Market API and Dashboard aggregate endpoints.

Real-data-only policy (PRD audit 2026-04, F-02-003 / F-03-003):
    This service NEVER fabricates or synthesizes financial numbers. Every value
    returned is derived from one of:
      1. Our own database (stocks + price_history that have been ingested), or
      2. Live provider quotes routed through the same cache-first path the
         stocks quote endpoint uses (``get_real_time_quote``), gated by the
         shared ``cost_monitor`` rate-limit accounting.

    When we have no real data for a given view yet, the corresponding helper
    returns a graceful EMPTY payload (empty list / ``None`` breadth). This is
    the intended "build up over time" behavior -- the heatmap shows N stocks if
    only N have been ingested.

Payload keys are camelCase to match the frontend TypeScript interfaces; the
router wraps the dicts in ``success_response`` unchanged.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories import stock_repository, price_repository
from backend.services.stocks_service import stocks_service, get_real_time_quote
from backend.utils.cost_monitor import cost_monitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed set of ETF proxies for the major market indices.
# We use ETF proxies (not synthetic index values) so every number traces back
# to a real, quotable security. These are the only "hardcoded" inputs and they
# are identifiers, not financial values.
# ---------------------------------------------------------------------------
INDEX_PROXIES: List[Dict[str, str]] = [
    {"symbol": "SPY", "name": "S&P 500"},
    {"symbol": "QQQ", "name": "Nasdaq 100"},
    {"symbol": "DIA", "name": "Dow Jones Industrial Average"},
    {"symbol": "IWM", "name": "Russell 2000"},
]

# How many stocks to scan from the DB when computing movers / heatmap.
_MOVERS_SCAN_LIMIT = 500
_DEFAULT_MOVERS_COUNT = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Coerce a numeric-ish value to float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sector_name(stock: Any) -> Optional[str]:
    """Resolve the stock's sector relationship to a plain string."""
    sector = getattr(stock, "sector", None)
    if sector is None:
        return None
    return getattr(sector, "name", None)


async def _compute_change(
    symbol: str,
    latest_close: float,
    reference_date: Any,
    db: AsyncSession,
) -> Dict[str, float]:
    """
    Compute day-over-day change + percent for *symbol* from real DB prices.

    Uses the previous price record strictly before the latest record's date.
    If there is no previous record, change is 0.0 (we only have one data point).
    """
    previous = await price_repository.get_previous_price(symbol, reference_date, session=db)
    previous_close = _safe_float(previous.close) if previous else latest_close

    change = latest_close - previous_close if previous_close else 0.0
    change_percent = (change / previous_close * 100) if previous_close else 0.0
    return {
        "change": round(change, 4),
        "changePercent": round(change_percent, 4),
        "previousClose": round(previous_close, 4),
    }


# ---------------------------------------------------------------------------
# MarketService
# ---------------------------------------------------------------------------

class MarketService:
    """Aggregates real market data for the market + dashboard routers."""

    # ------------------------------------------------------------------
    # Indices (ETF proxies, cache-first live quotes)
    # ------------------------------------------------------------------

    async def get_indices(self, *, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Return MarketIndex dicts for the fixed ETF proxies.

        Each quote is fetched via ``get_real_time_quote`` (the same cache-first
        path the stocks quote route uses). Rate limits are respected through the
        shared ``cost_monitor``: if the provider limit is reached we fall back to
        the latest DB price for that proxy, and if neither is available we omit
        that proxy rather than invent a value.
        """
        indices: List[Dict[str, Any]] = []

        for proxy in INDEX_PROXIES:
            symbol = proxy["symbol"]
            index = await self._build_index_for_proxy(symbol, proxy["name"], db=db)
            if index is not None:
                indices.append(index)

        return indices

    async def _build_index_for_proxy(
        self,
        symbol: str,
        name: str,
        *,
        db: AsyncSession,
    ) -> Optional[Dict[str, Any]]:
        """Build a single MarketIndex dict from a live quote or DB fallback."""
        real_time_data: Optional[Dict[str, Any]] = None

        # Respect provider rate limits before reaching out. ``finnhub`` is the
        # default quote provider in cost_monitor; if it is rate-limited we skip
        # the network call and let the DB fallback handle it.
        try:
            allowed = await cost_monitor.check_api_limit("finnhub")
        except Exception as exc:  # pragma: no cover - accounting must not break the request
            logger.warning("cost_monitor check failed for %s: %s", symbol, exc)
            allowed = True

        if allowed:
            try:
                real_time_data = await get_real_time_quote(symbol)
            except Exception as exc:
                logger.warning("Live quote failed for index proxy %s: %s", symbol, exc)
                real_time_data = None

        try:
            quote = await stocks_service.get_stock_quote(
                symbol=symbol, real_time_data=real_time_data, db=db
            )
        except Exception as exc:
            # No live data AND no DB price for this proxy yet -> omit it.
            logger.info("No quote available for index proxy %s (%s); omitting", symbol, exc)
            return None

        timestamp = quote.get("timestamp") or datetime.now(timezone.utc)
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        price = _safe_float(quote.get("price"))
        prev_close = _safe_float(quote.get("previous_close"))

        # Without a positive current price we have no live value. Reporting
        # change against a stale previousClose yields a fabricated -100% move
        # (0 - prevClose), which is worse than no data. Fall back to the last
        # known close shown flat (change 0); omit the proxy only when we have
        # no real price at all.
        if not price or price <= 0:
            if prev_close and prev_close > 0:
                value, change, change_percent = prev_close, 0.0, 0.0
            else:
                logger.info("No real price for index proxy %s; omitting", symbol)
                return None
        else:
            value = price
            change = round(_safe_float(quote.get("change")) or 0.0, 4)
            change_percent = round(_safe_float(quote.get("change_percent")) or 0.0, 4)
            # Guard against a bogus percent when the previous close is missing.
            if not prev_close or prev_close <= 0:
                change, change_percent = 0.0, 0.0

        return {
            "symbol": symbol,
            "name": name,
            "value": round(value, 4),
            "change": change,
            "changePercent": change_percent,
            "volume": int(quote.get("volume") or 0),
            "high": _safe_float(quote.get("high")) or 0.0,
            "low": _safe_float(quote.get("low")) or 0.0,
            "previousClose": round(prev_close, 4) if prev_close else round(value, 4),
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    # Movers + heatmap (computed from ingested DB stocks/prices)
    # ------------------------------------------------------------------

    async def _build_priced_universe(self, *, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Build the working set of real stocks with computed day change.

        Returns a list of normalized dicts (ticker, companyName, price, change,
        changePercent, volume, marketCap, sector). Derived purely from stocks +
        price_history already ingested -- empty list when nothing is ingested.
        """
        rows = await stock_repository.get_stocks_with_latest_prices(
            limit=_MOVERS_SCAN_LIMIT, session=db
        )

        universe: List[Dict[str, Any]] = []
        for row in rows:
            stock = row.get("stock")
            latest_price = _safe_float(row.get("latest_price"))
            if stock is None or latest_price is None:
                continue

            change_data = await _compute_change(
                stock.symbol, latest_price, row.get("price_date"), db
            )

            universe.append({
                "ticker": stock.symbol,
                "companyName": stock.name,
                "price": round(latest_price, 4),
                "change": change_data["change"],
                "changePercent": change_data["changePercent"],
                "volume": int(row.get("volume") or 0),
                "marketCap": _safe_float(stock.market_cap) or 0.0,
                "sector": _sector_name(stock) or "Unknown",
            })

        return universe

    async def get_movers(
        self,
        *,
        db: AsyncSession,
        count: int = _DEFAULT_MOVERS_COUNT,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return {gainers, losers, active} computed from real ingested data.

        Empty arrays when no priced stocks exist yet.
        """
        universe = await self._build_priced_universe(db=db)
        if not universe:
            return {"gainers": [], "losers": [], "active": []}

        by_change = sorted(universe, key=lambda s: s["changePercent"], reverse=True)
        by_volume = sorted(universe, key=lambda s: s["volume"], reverse=True)

        gainers = [s for s in by_change if s["changePercent"] > 0][:count]
        losers = [s for s in reversed(by_change) if s["changePercent"] < 0][:count]
        active = by_volume[:count]

        return {"gainers": gainers, "losers": losers, "active": active}

    async def get_heatmap(
        self,
        *,
        db: AsyncSession,
        sector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return HeatmapItem dicts for ingested stocks (optionally sector-filtered).

        ``index`` is accepted by the route for API symmetry but, without an
        index-constituent mapping in our data, it does not filter; we return the
        real ingested universe. Empty list when nothing is ingested.
        """
        universe = await self._build_priced_universe(db=db)

        items: List[Dict[str, Any]] = []
        for s in universe:
            if sector and (s["sector"] or "").lower() != sector.lower():
                continue
            items.append({
                "ticker": s["ticker"],
                "name": s["companyName"],
                "sector": s["sector"],
                "changePercent": s["changePercent"],
                "marketCap": s["marketCap"],
                "volume": s["volume"],
            })

        return items

    # ------------------------------------------------------------------
    # Sectors (reuse the stocks sector-summary aggregation)
    # ------------------------------------------------------------------

    async def get_sector_performance(self, *, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Return SectorPerformance dicts.

        Reuses ``stocks_service.get_sector_summary`` (the same aggregation the
        stocks ``/sectors`` route family uses) for marketCap, then layers on
        per-sector gainer/loser counts and the top stock computed from the real
        priced universe. Sectors with no priced stocks report null-safe zeros
        for the price-derived fields rather than fabricated movement.
        """
        summary = await stocks_service.get_sector_summary(db=db)
        universe = await self._build_priced_universe(db=db)

        # Group the priced universe by sector for per-sector stats.
        by_sector: Dict[str, List[Dict[str, Any]]] = {}
        for s in universe:
            by_sector.setdefault(s["sector"], []).append(s)

        results: List[Dict[str, Any]] = []
        for item in summary:
            sector_name = item.get("sector")
            if not sector_name:
                continue

            members = by_sector.get(sector_name, [])
            gainers = sum(1 for m in members if m["changePercent"] > 0)
            losers = sum(1 for m in members if m["changePercent"] < 0)
            volume = sum(m["volume"] for m in members)

            # Average change across the sector's priced members (real data only).
            if members:
                avg_change = round(
                    sum(m["changePercent"] for m in members) / len(members), 4
                )
                top = max(members, key=lambda m: m["changePercent"])
                top_stock = {
                    "ticker": top["ticker"],
                    "changePercent": top["changePercent"],
                }
            else:
                avg_change = 0.0
                top_stock = {"ticker": "", "changePercent": 0.0}

            results.append({
                "sector": sector_name,
                "changePercent": avg_change,
                "marketCap": _safe_float(item.get("total_market_cap")) or 0.0,
                "volume": volume,
                "gainers": gainers,
                "losers": losers,
                "topStock": top_stock,
            })

        return results

    # ------------------------------------------------------------------
    # Market breadth (computed from the real priced universe)
    # ------------------------------------------------------------------

    async def get_market_breadth(
        self,
        *,
        db: AsyncSession,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute MarketBreadth from the real ingested universe.

        Returns ``None`` when there are no priced stocks yet (frontend renders a
        graceful empty state). newHighs / newLows require 52-week extremes which
        we do not track per-row here, so they report 0 rather than a fabricated
        count.
        """
        universe = await self._build_priced_universe(db=db)
        if not universe:
            return None

        advancers = sum(1 for s in universe if s["changePercent"] > 0)
        decliners = sum(1 for s in universe if s["changePercent"] < 0)
        unchanged = sum(1 for s in universe if s["changePercent"] == 0)

        up_volume = sum(s["volume"] for s in universe if s["changePercent"] > 0)
        down_volume = sum(s["volume"] for s in universe if s["changePercent"] < 0)
        total_volume = sum(s["volume"] for s in universe)

        advance_decline_ratio = round(advancers / decliners, 4) if decliners else float(advancers)

        return {
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "newHighs": 0,
            "newLows": 0,
            "advanceDeclineRatio": advance_decline_ratio,
            "upVolume": up_volume,
            "downVolume": down_volume,
            "totalVolume": total_volume,
        }


# Module-level singleton (matches the pattern used by other services)
market_service = MarketService()
