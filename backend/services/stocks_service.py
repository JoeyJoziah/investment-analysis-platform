"""
Stocks Service
Business logic for stock data retrieval, quoting, and analysis.
Extracted from backend/api/routers/stocks.py to keep the router thin.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories import (
    stock_repository,
    price_repository,
    FilterCriteria,
    PaginationParams,
    SortParams,
    SortDirection,
)
from backend.repositories.alert_repository import alert_repository
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.utils.api_cache_decorators import api_cache
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External data provider clients (initialized once at import time)
# ---------------------------------------------------------------------------

alpha_vantage_client = AlphaVantageClient() if settings.ALPHA_VANTAGE_API_KEY else None
finnhub_client = FinnhubClient() if settings.FINNHUB_API_KEY else None
try:
    polygon_client = PolygonClient() if settings.POLYGON_API_KEY else None
except Exception as e:
    logger.warning(f"Failed to initialize Polygon client: {e}")
    polygon_client = None


# ---------------------------------------------------------------------------
# Cached data-fetching helpers
# ---------------------------------------------------------------------------

@api_cache(
    data_type="real_time_quote",
    ttl_override={'l1': 60, 'l2': 300, 'l3': 1800},
    cost_tracking=True,
)
async def get_real_time_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time quote from available providers with intelligent caching."""
    try:
        if finnhub_client:
            return await finnhub_client.get_quote(symbol)
        if alpha_vantage_client:
            return await alpha_vantage_client.get_quote(symbol)
        if polygon_client:
            return await polygon_client.get_quote(symbol)
        return None
    except Exception as e:
        logger.error(f"Error fetching real-time quote for {symbol}: {e}")
        return None


@api_cache(
    data_type="company_overview",
    ttl_override={'l1': 7200, 'l2': 43200, 'l3': 604800},
    cost_tracking=True,
)
async def fetch_company_overview(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch company overview from available providers with intelligent caching."""
    try:
        if alpha_vantage_client:
            return await alpha_vantage_client.get_company_overview(symbol)
        if finnhub_client:
            return await finnhub_client.get_company_profile(symbol)
        return None
    except Exception as e:
        logger.error(f"Error fetching company overview for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# StocksService
# ---------------------------------------------------------------------------

class StocksService:
    """
    Business logic for all stock-related operations.

    Each public method corresponds to a route handler in the stocks router.
    The router delegates to these methods after parsing request parameters
    and returns the result wrapped in an ``ApiResponse``.
    """

    # ------------------------------------------------------------------
    # List / Search
    # ------------------------------------------------------------------

    async def get_stocks(
        self,
        *,
        sector: Optional[str],
        min_market_cap: Optional[float],
        max_market_cap: Optional[float],
        is_active: bool,
        limit: int,
        offset: int,
        sort_by: str,
        order: str,
        db: AsyncSession,
    ) -> List:
        """Return a filtered, sorted, paginated list of Stock ORM objects."""
        filters: list[FilterCriteria] = []

        if is_active:
            filters.append(FilterCriteria(field='is_active', operator='eq', value=True))
            filters.append(FilterCriteria(field='is_tradable', operator='eq', value=True))
        if sector:
            filters.append(FilterCriteria(field='sector', operator='eq', value=sector))
        if min_market_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='gte', value=int(min_market_cap)))
        if max_market_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='lte', value=int(max_market_cap)))

        sort_direction = SortDirection.DESC if order == "desc" else SortDirection.ASC
        sort_params = [SortParams(field=sort_by, direction=sort_direction)]
        pagination = PaginationParams(offset=offset, limit=limit)

        return await stock_repository.get_multi(
            filters=filters,
            sort_params=sort_params,
            pagination=pagination,
            session=db,
        )

    async def search_stocks(
        self,
        *,
        query: str,
        limit: int,
        db: AsyncSession,
    ) -> List:
        """Search stocks by symbol or company name."""
        return await stock_repository.search_stocks(
            query=query,
            limit=limit,
            session=db,
        )

    # ------------------------------------------------------------------
    # Detail
    # ------------------------------------------------------------------

    async def get_stock_detail(
        self,
        *,
        symbol: str,
        db: AsyncSession,
    ):
        """Return a single Stock ORM object or ``None``."""
        return await stock_repository.get_by_symbol(symbol, session=db)

    # ------------------------------------------------------------------
    # Quote  (real-time with database fallback)
    # ------------------------------------------------------------------

    async def get_stock_quote(
        self,
        *,
        symbol: str,
        real_time_data: Optional[Dict[str, Any]],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Build a quote dict for *symbol*.

        The caller is responsible for fetching ``real_time_data`` from the
        external provider (so that test patches on the module-level
        ``get_real_time_quote`` function continue to work).  When
        ``real_time_data`` is ``None``, this method falls back to the
        latest database price.

        Returns a flat dict that maps directly onto ``StockQuoteResponse``
        field names.
        """
        symbol = symbol.upper()
        logger.info(f"Fetching quote for {symbol}")

        if real_time_data:
            data_source = real_time_data.get('source', 'external_api')
            quote = self._build_quote_from_external(symbol, real_time_data, data_source)

            # Enrich with company-overview fundamentals (market cap, P/E,
            # 52-week range) that the provider /quote endpoint omits. Best
            # effort: a failure here must never change the returned quote.
            await self._enrich_quote_with_overview(symbol, quote)

            # Write-through: persist the REAL provider OHLCV we just fetched so
            # the database accumulates real coverage over time. Best effort:
            # persistence errors are logged and swallowed so they can never
            # break or slow the response. No extra provider calls are made.
            await self._persist_external_quote(symbol, real_time_data, quote, db)

            return quote

        # Fallback to database
        logger.info(f"Falling back to database for {symbol}")
        return await self._build_quote_from_db(symbol, db)

    def _build_quote_from_external(
        self,
        symbol: str,
        quote_data: Dict[str, Any],
        data_source: str,
    ) -> Dict[str, Any]:
        """Transform raw external provider data into a normalised quote dict."""
        # Provider clients normalize price under different keys: FinnhubClient
        # uses `current_price`, AlphaVantageClient uses `price`; raw payloads use
        # `c`. Read all so the current price isn't silently lost (which made the
        # whole quote fall back to a flat previous close).
        current_price = float(
            quote_data.get('current_price',
                quote_data.get('price',
                    quote_data.get('c', 0))) or 0
        )
        previous_close = float(quote_data.get('previous_close', quote_data.get('pc', 0)) or 0)
        open_price = float(quote_data.get('open', quote_data.get('o', 0)) or 0)

        # Providers (e.g. Finnhub when the market is closed / pre-open) sometimes
        # return a current price of 0 while previous close and OHLC are real.
        # Reporting $0.00 / -100% (0 - pc) is fabricated movement, so fall back
        # to the last known price (previous close, else open) shown flat.
        if current_price <= 0:
            current_price = previous_close or open_price
            change = 0.0
            change_percent = 0.0
        elif previous_close > 0:
            change = current_price - previous_close
            change_percent = change / previous_close * 100
        else:
            # Price but no previous close (e.g. AlphaVantage): use the provider's
            # own change fields rather than computing a delta against 0.
            change = float(quote_data.get('change', quote_data.get('d', 0)) or 0)
            change_percent = float(
                quote_data.get('percent_change',
                    quote_data.get('change_percent',
                        quote_data.get('dp', 0))) or 0
            )

        return {
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "change_percent": change_percent,
            "volume": int(quote_data.get('volume', quote_data.get('v', 0))),
            "timestamp": datetime.now(timezone.utc),
            "open": float(quote_data.get('open', quote_data.get('o'))) if quote_data.get('open') or quote_data.get('o') else None,
            "high": float(quote_data.get('high', quote_data.get('h'))) if quote_data.get('high') or quote_data.get('h') else None,
            "low": float(quote_data.get('low', quote_data.get('l'))) if quote_data.get('low') or quote_data.get('l') else None,
            "previous_close": previous_close if previous_close and previous_close != current_price else None,
            "bid": float(quote_data.get('bid')) if quote_data.get('bid') else None,
            "ask": float(quote_data.get('ask')) if quote_data.get('ask') else None,
            "fifty_two_week_high": float(quote_data.get('52_week_high')) if quote_data.get('52_week_high') else None,
            "fifty_two_week_low": float(quote_data.get('52_week_low')) if quote_data.get('52_week_low') else None,
            "pe_ratio": float(quote_data.get('pe')) if quote_data.get('pe') else None,
            "data_source": data_source,
            "last_updated": datetime.now(timezone.utc),
            "is_real_time": True,
        }

    async def _enrich_quote_with_overview(
        self,
        symbol: str,
        quote: Dict[str, Any],
    ) -> None:
        """
        Populate company-overview fundamentals on *quote* in place.

        The provider ``/quote`` endpoint returns only OHLCV + change, so
        ``market_cap``, ``pe_ratio`` and the 52-week range are missing (today
        they surface as 0/null in the API response). This pulls them from the
        already-cached ``fetch_company_overview`` helper and maps the provider
        keys onto the quote field names the API/frontend expect.

        Compliance: values are only set when the overview genuinely provides a
        non-zero figure. When the overview is unavailable (or a field is
        absent/0) the field is left as whatever the quote already had -- we
        never fabricate or coerce a real 0 into a fake number. Any value
        already present on the quote (e.g. a provider that did include it) is
        preserved and not overwritten.

        Best effort: any error is logged and swallowed so enrichment can never
        break the returned quote.
        """
        try:
            overview = await fetch_company_overview(symbol)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Overview enrichment failed for {symbol}: {e}")
            return

        if not overview:
            return

        # Map provider overview keys -> quote field names. AlphaVantage and the
        # Finnhub profile fallback both expose these under the keys below.
        field_map = {
            "market_cap": "market_cap",
            "pe_ratio": "pe_ratio",
            "52_week_high": "fifty_two_week_high",
            "52_week_low": "fifty_two_week_low",
        }

        for source_key, quote_key in field_map.items():
            # Do not clobber a real value the quote already carried.
            if quote.get(quote_key):
                continue

            raw = overview.get(source_key)
            value = self._coerce_positive_number(raw)
            if value is not None:
                quote[quote_key] = value

    @staticmethod
    def _coerce_positive_number(raw: Any) -> Optional[float]:
        """Return *raw* as a positive float, or ``None`` when absent/0/invalid.

        Overview providers default missing numerics to 0; treating those as
        real values would surface fabricated $0 market caps / P/E ratios, so a
        non-positive or unparseable figure is reported as ``None`` (unknown).
        """
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    async def _persist_external_quote(
        self,
        symbol: str,
        quote_data: Dict[str, Any],
        quote: Dict[str, Any],
        db: AsyncSession,
    ) -> None:
        """
        Write-through persistence of a freshly-fetched external quote.

        Ensures a ``stocks`` row exists for *symbol* (creating a minimal real
        row when missing) and inserts TODAY's ``price_history`` row using the
        REAL OHLCV from the provider quote -- but only when we don't already
        have a row for today (the daily backfill owns the authoritative
        end-of-day bar, so we never clobber or duplicate it).

        Compliance: the open/high/low/close written are the provider's own
        values. ``close`` is the current price; ``open``/``high``/``low`` come
        straight from the provider and are NOT synthesised to equal the close.
        Volume is the provider's reported volume (0 for Finnhub, which does not
        return intraday volume on /quote) -- never invented.

        Best effort: wrapped in try/except. Any failure is logged and swallowed
        so a write error can never change or slow the returned quote. Adds no
        extra provider API calls -- it only persists data already fetched.
        """
        try:
            close_price = quote.get("price")
            if not close_price or close_price <= 0:
                # No real, positive close to anchor a row -- skip rather than
                # persist a meaningless/zero price.
                return

            stock = await stock_repository.get_by_symbol(symbol, session=db)
            if stock is None:
                stock = await self._create_minimal_stock(symbol, db)
            if stock is None:
                return

            price_date = self._resolve_price_date(quote_data, quote)

            # bulk_upsert_prices conflict-resolves on the PK, not (stock_id,
            # date), so re-writing a date we already have (e.g. from the daily
            # backfill) raises a unique-constraint error. Skip when today's row
            # already exists -- the displayed quote is served live from the
            # provider regardless, and the daily backfill refreshes the EOD bar.
            from sqlalchemy import select as _select
            from backend.models.unified_models import PriceHistory

            existing = await db.execute(
                _select(PriceHistory.id)
                .where(
                    PriceHistory.stock_id == stock.id,
                    PriceHistory.date == price_date,
                )
                .limit(1)
            )
            if existing.first() is not None:
                return

            # Use the provider's REAL OHLC. Fall back to the close only when a
            # field is genuinely absent so the NOT NULL columns are satisfied;
            # this is the honest "last known price" rather than fabricated
            # intraday movement.
            open_price = self._extract_price(quote_data, ("open", "o"))
            high_price = self._extract_price(quote_data, ("high", "h"))
            low_price = self._extract_price(quote_data, ("low", "l"))
            volume = self._extract_volume(quote_data)

            row = {
                "stock_id": stock.id,
                "date": price_date,
                "open": Decimal(str(open_price if open_price is not None else close_price)),
                "high": Decimal(str(high_price if high_price is not None else close_price)),
                "low": Decimal(str(low_price if low_price is not None else close_price)),
                "close": Decimal(str(close_price)),
                "volume": int(volume),
            }

            affected = await price_repository.bulk_upsert_prices([row], session=db)
            logger.debug(
                f"Write-through persisted quote for {symbol} on {price_date} "
                f"({affected} row(s) affected)"
            )
        except Exception as e:
            # Persistence is strictly best-effort; never propagate.
            logger.warning(f"Write-through persistence failed for {symbol}: {e}")

    async def _create_minimal_stock(
        self,
        symbol: str,
        db: AsyncSession,
    ):
        """
        Create a minimal real ``stocks`` row for *symbol* when one is missing.

        Only writes data we actually know: the symbol and, when the cached
        company overview provides it, the real company name. Nullable fields
        are left null -- we do NOT fabricate market cap, sector, exchange, etc.
        Returns the created Stock (or ``None`` if creation was not possible).
        """
        try:
            name = symbol
            try:
                overview = await fetch_company_overview(symbol)
                if overview and overview.get("name"):
                    name = overview["name"]
            except Exception:  # pragma: no cover - defensive
                pass

            created = await stock_repository.create(
                {"symbol": symbol, "name": name},
                session=db,
            )
            return created
        except Exception as e:
            logger.warning(f"Could not create minimal stock row for {symbol}: {e}")
            return None

    @staticmethod
    def _extract_price(quote_data: Dict[str, Any], keys: tuple) -> Optional[float]:
        """Return the first present positive price among *keys*, else ``None``."""
        for key in keys:
            raw = quote_data.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    @staticmethod
    def _extract_volume(quote_data: Dict[str, Any]) -> int:
        """Return the provider's reported volume (0 when absent -- never invented)."""
        raw = quote_data.get("volume", quote_data.get("v", 0))
        try:
            value = int(raw or 0)
        except (TypeError, ValueError):
            return 0
        return value if value > 0 else 0

    @staticmethod
    def _resolve_price_date(quote_data: Dict[str, Any], quote: Dict[str, Any]) -> datetime:
        """
        Resolve the date to store the price row under.

        Uses the provider quote's timestamp date when parseable, otherwise the
        quote's own timestamp, otherwise today's UTC date. The value is
        normalised to midnight because ``PriceHistory.date`` is a daily column
        whose unique constraint is ``(stock_id, date)``.
        """
        ts = quote_data.get("timestamp")
        resolved: Optional[datetime] = None

        if isinstance(ts, str):
            try:
                resolved = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                resolved = None
        elif isinstance(ts, datetime):
            resolved = ts

        if resolved is None:
            quote_ts = quote.get("timestamp")
            if isinstance(quote_ts, datetime):
                resolved = quote_ts

        if resolved is None:
            resolved = datetime.now(timezone.utc)

        # Normalise to midnight so repeated intraday quotes upsert one row/day.
        return datetime(resolved.year, resolved.month, resolved.day)

    async def _build_quote_from_db(
        self,
        symbol: str,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Build a quote dict from database price history."""
        from fastapi import HTTPException, status

        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock '{symbol}' not found in database",
            )

        latest_price = await price_repository.get_latest_price(symbol, session=db)
        if not latest_price:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price data found for symbol '{symbol}'",
            )

        previous_price = await price_repository.get_previous_price(symbol, latest_price.date, session=db)
        previous_close = float(previous_price.close) if previous_price else float(latest_price.close)

        current_price = float(latest_price.close)
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0.0

        return {
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "change_percent": change_percent,
            "volume": latest_price.volume,
            "timestamp": datetime.combine(latest_price.date, datetime.min.time()),
            "open": float(latest_price.open),
            "high": float(latest_price.high),
            "low": float(latest_price.low),
            "previous_close": previous_close if previous_price else None,
            "market_cap": stock.market_cap,
            "data_source": "database",
            "last_updated": datetime.now(timezone.utc),
            "is_real_time": False,
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    async def get_price_history(
        self,
        *,
        symbol: str,
        start_date: Optional[date],
        end_date: Optional[date],
        limit: Optional[int],
        db: AsyncSession,
    ) -> List:
        """Return price history ORM rows for *symbol* within a date range."""
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=365)

        return await price_repository.get_price_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            session=db,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stock_statistics(
        self,
        *,
        symbol: str,
        days: int,
        db: AsyncSession,
    ) -> Optional[Dict[str, Any]]:
        """Return price statistics dict (or ``None`` when no data exists)."""
        statistics = await price_repository.get_price_statistics(
            symbol=symbol,
            days=days,
            session=db,
        )
        if not statistics:
            return None

        volatility = await price_repository.get_volatility(
            symbol=symbol,
            days=min(days, 30),
            session=db,
        )
        if volatility is not None:
            statistics['volatility_annualized'] = volatility

        return statistics

    # ------------------------------------------------------------------
    # Sectors
    # ------------------------------------------------------------------

    async def get_sectors(self, *, db: AsyncSession) -> List[str]:
        """Return a flat list of sector name strings."""
        sector_summary = await stock_repository.get_sector_summary(session=db)
        return [item['sector'] for item in sector_summary if item['sector']]

    async def get_sector_summary(self, *, db: AsyncSession) -> List[Dict[str, Any]]:
        """Return sector summary dicts with statistics."""
        return await stock_repository.get_sector_summary(session=db)

    # ------------------------------------------------------------------
    # Top performers
    # ------------------------------------------------------------------

    async def get_top_performers(
        self,
        *,
        timeframe: str,
        limit: int,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """Return a list of ``{stock, start_price, end_price, performance_pct}`` dicts."""
        return await stock_repository.get_top_performers(
            timeframe=timeframe,
            limit=limit,
            session=db,
        )

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    async def get_latest_fundamentals(
        self,
        *,
        symbol: str,
        db: AsyncSession,
    ):
        """
        Return the most recent ``Fundamentals`` ORM row for *symbol* (or ``None``).

        Reads only from our database — fundamentals are ingested separately
        (SEC filings / provider sync) and built up over time. We never
        fabricate ratios here.
        """
        from sqlalchemy import select, desc
        from backend.models.unified_models import Fundamentals, Stock

        query = (
            select(Fundamentals)
            .join(Stock, Fundamentals.stock_id == Stock.id)
            .where(Stock.symbol == symbol.upper())
            .order_by(desc(Fundamentals.period_date))
            .limit(1)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Similar / peer stocks (same sector, real data only)
    # ------------------------------------------------------------------

    async def get_similar_stocks(
        self,
        *,
        symbol: str,
        limit: int,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """
        Return up to *limit* real peer stocks in the same sector as *symbol*.

        Peers are sourced from our database (same ``sector_id``, active and
        tradable, excluding the target symbol). ``changePercent`` and
        ``correlation`` are computed from stored price history only. When a
        value cannot be computed from real data it is returned as ``0.0``
        rather than fabricated.
        """
        from sqlalchemy import select, and_
        from backend.models.unified_models import Stock

        symbol = symbol.upper()

        target = await stock_repository.get_by_symbol(symbol, session=db)
        if not target or target.sector_id is None:
            return []

        # Fetch same-sector peers (small over-fetch so we can rank by market cap)
        peers_query = (
            select(Stock)
            .where(
                and_(
                    Stock.sector_id == target.sector_id,
                    Stock.symbol != symbol,
                    Stock.is_active == True,  # noqa: E712 - SQLAlchemy boolean filter
                    Stock.is_tradable == True,  # noqa: E712
                )
            )
            .order_by(Stock.market_cap.desc().nullslast())
            .limit(limit)
        )
        result = await db.execute(peers_query)
        peers = result.scalars().all()
        if not peers:
            return []

        # Pre-load the target's recent closes once for correlation.
        target_history = await price_repository.get_price_history(
            symbol=symbol,
            start_date=date.today() - timedelta(days=120),
            end_date=date.today(),
            limit=90,
            session=db,
        )
        # Repository returns newest-first; reverse to chronological order.
        target_closes_by_date = {
            p.date: float(p.close) for p in target_history
        }

        peer_results: List[Dict[str, Any]] = []
        for peer in peers:
            peer_history = await price_repository.get_price_history(
                symbol=peer.symbol,
                start_date=date.today() - timedelta(days=120),
                end_date=date.today(),
                limit=90,
                session=db,
            )

            change_percent = self._latest_change_percent(peer_history)
            correlation = self._price_correlation(
                target_closes_by_date,
                {p.date: float(p.close) for p in peer_history},
            )

            peer_results.append({
                "ticker": peer.symbol,
                "name": peer.name,
                "correlation": correlation,
                "changePercent": change_percent,
            })

        return peer_results

    @staticmethod
    def _latest_change_percent(price_history: List) -> float:
        """Compute the latest day-over-day percent change from real prices.

        ``price_history`` is newest-first (per the repository contract).
        Returns ``0.0`` when there are fewer than two real price rows.
        """
        if not price_history or len(price_history) < 2:
            return 0.0
        latest = float(price_history[0].close)
        previous = float(price_history[1].close)
        if previous == 0:
            return 0.0
        return round((latest - previous) / previous * 100, 4)

    @staticmethod
    def _price_correlation(
        a_closes_by_date: Dict[date, float],
        b_closes_by_date: Dict[date, float],
    ) -> float:
        """Pearson correlation of daily returns over overlapping dates.

        Returns ``0.0`` when there is insufficient overlapping real history
        (fewer than ~20 shared trading days) instead of fabricating a value.
        """
        common_dates = sorted(set(a_closes_by_date) & set(b_closes_by_date))
        if len(common_dates) < 21:
            return 0.0

        a_series = [a_closes_by_date[d] for d in common_dates]
        b_series = [b_closes_by_date[d] for d in common_dates]

        a_returns = [
            (a_series[i] - a_series[i - 1]) / a_series[i - 1]
            for i in range(1, len(a_series))
            if a_series[i - 1] != 0
        ]
        b_returns = [
            (b_series[i] - b_series[i - 1]) / b_series[i - 1]
            for i in range(1, len(b_series))
            if b_series[i - 1] != 0
        ]
        n = min(len(a_returns), len(b_returns))
        if n < 20:
            return 0.0
        a_returns, b_returns = a_returns[:n], b_returns[:n]

        mean_a = sum(a_returns) / n
        mean_b = sum(b_returns) / n
        cov = sum((a_returns[i] - mean_a) * (b_returns[i] - mean_b) for i in range(n))
        var_a = sum((x - mean_a) ** 2 for x in a_returns)
        var_b = sum((x - mean_b) ** 2 for x in b_returns)
        denom = (var_a * var_b) ** 0.5
        if denom == 0:
            return 0.0
        return round(cov / denom, 4)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def create_price_alert(
        self,
        *,
        user_id: int,
        symbol: str,
        condition: str,
        threshold_price: float,
        is_recurring: bool,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Persist a new price alert and return a dict suitable for AlertResponse.

        Raises ``HTTPException`` if the stock does not exist.
        """
        from fastapi import HTTPException, status

        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found",
            )

        condition_payload = {
            "type": "price_threshold",
            "condition": condition,
            "threshold_price": threshold_price,
        }

        alert = await alert_repository.create(
            data={
                "user_id": user_id,
                "stock_id": stock.id,
                "alert_type": "price_threshold",
                "condition": condition_payload,
                "is_active": True,
                "is_recurring": is_recurring,
            },
            session=db,
        )

        return {
            "alert_id": alert.alert_id,
            "symbol": symbol,
            "condition": condition,
            "threshold_price": threshold_price,
            "is_active": alert.is_active,
            "is_recurring": alert.is_recurring,
            "status": "active",
            "created_at": alert.created_at or datetime.now(timezone.utc),
        }


# Module-level singleton (matches the pattern used by other services)
stocks_service = StocksService()
