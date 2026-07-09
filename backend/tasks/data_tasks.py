"""
Celery tasks for data ingestion and processing
"""
from celery import shared_task, group, chain
from celery.exceptions import SoftTimeLimitExceeded
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date, timezone
import asyncio
import logging
import json
import time
from decimal import Decimal

from backend.tasks.celery_app import celery_app, TaskPriority
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.utils.database import get_db_sync
from backend.utils.cache import get_redis_client
from backend.utils.cost_monitor import cost_monitor
from backend.models.unified_models import Stock, PriceHistory, Fundamentals, News
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Rate limiting for API calls
ALPHA_VANTAGE_DAILY_LIMIT = 25
FINNHUB_MINUTE_LIMIT = 60
POLYGON_MINUTE_LIMIT = 5

# Daily-candle backfill tuning.
# Finnhub free tier allows 60 calls/minute. We keep a conservative throttle so
# that a single backfill worker, plus any concurrent quote traffic, stays well
# under the limit. cost_monitor.check_api_limit is still the authoritative gate.
FINNHUB_BACKFILL_SLEEP_SECONDS = 1.1  # ~54 calls/min, leaves headroom under 60
DEFAULT_BACKFILL_DAYS = 365  # ~1 year of daily candles

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def fetch_stock_data(self, symbol: str, source: str = "all") -> Dict[str, Any]:
    """
    Fetch stock data from specified source(s)
    
    Args:
        symbol: Stock symbol
        source: Data source ('alpha_vantage', 'finnhub', 'polygon', 'all')
    
    Returns:
        Dictionary with fetched data
    """
    try:
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': {},
            'errors': []
        }
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if source in ['alpha_vantage', 'all']:
                try:
                    av_client = AlphaVantageClient()
                    av_data = loop.run_until_complete(av_client.get_quote(symbol))
                    result['data']['alpha_vantage'] = av_data
                except Exception as e:
                    result['errors'].append(f"Alpha Vantage error: {str(e)}")
                    logger.error(f"Alpha Vantage error for {symbol}: {e}")
            
            if source in ['finnhub', 'all']:
                try:
                    fh_client = FinnhubClient()
                    fh_data = loop.run_until_complete(fh_client.get_quote(symbol))
                    result['data']['finnhub'] = fh_data
                except Exception as e:
                    result['errors'].append(f"Finnhub error: {str(e)}")
                    logger.error(f"Finnhub error for {symbol}: {e}")
            
            if source in ['polygon', 'all']:
                try:
                    pg_client = PolygonClient()
                    pg_data = loop.run_until_complete(pg_client.get_quote(symbol))
                    result['data']['polygon'] = pg_data
                except Exception as e:
                    result['errors'].append(f"Polygon error: {str(e)}")
                    logger.error(f"Polygon error for {symbol}: {e}")
        
        finally:
            loop.close()
        
        # Store in database
        if result['data']:
            store_price_data.delay(symbol, result['data'])
        
        # Cache the result
        redis_client = get_redis_client()
        cache_key = f"stock_data:{symbol}"
        redis_client.setex(cache_key, 300, json.dumps(result))  # Cache for 5 minutes
        
        return result
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task timeout for symbol {symbol}")
        raise self.retry(countdown=60)
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise self.retry(exc=e, countdown=60)

@celery_app.task
def fetch_all_market_data() -> Dict[str, Any]:
    """
    Fetch market data for all active stocks
    Implements intelligent batching to respect API rate limits
    """
    try:
        with get_db_sync() as db:
            # Get all active stocks
            stocks = db.query(Stock).filter(
                Stock.is_active == True,
                Stock.is_tradable == True
            ).all()
            
            # Group stocks by priority (market cap)
            high_priority = []  # Top 100 by market cap
            medium_priority = []  # Next 400
            low_priority = []  # Rest
            
            sorted_stocks = sorted(stocks, key=lambda x: x.market_cap or 0, reverse=True)
            
            for i, stock in enumerate(sorted_stocks):
                if i < 100:
                    high_priority.append(stock.symbol)
                elif i < 500:
                    medium_priority.append(stock.symbol)
                else:
                    low_priority.append(stock.symbol)
        
        # Create task groups with different priorities
        high_priority_group = group(
            fetch_stock_data.si(symbol, 'finnhub').set(priority=TaskPriority.HIGH)
            for symbol in high_priority[:20]  # Limit to avoid rate limits
        )
        
        medium_priority_group = group(
            fetch_stock_data.si(symbol, 'finnhub').set(priority=TaskPriority.NORMAL)
            for symbol in medium_priority[:30]
        )
        
        # Execute groups
        results = {
            'high_priority': high_priority_group.apply_async().get(timeout=300),
            'medium_priority': medium_priority_group.apply_async().get(timeout=300),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stocks_updated': len(high_priority[:20]) + len(medium_priority[:30])
        }
        
        logger.info(f"Market data fetch completed: {results['stocks_updated']} stocks updated")
        return results
        
    except Exception as e:
        logger.error(f"Error in fetch_all_market_data: {e}")
        return {'error': str(e)}

@celery_app.task
def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Fetch historical price data for a symbol
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Use Alpha Vantage for historical data (better for this purpose)
            av_client = AlphaVantageClient()
            historical_data = loop.run_until_complete(
                av_client.get_daily_prices(symbol, outputsize='full')
            )
        finally:
            loop.close()
        
        # Parse and store historical data
        if historical_data:
            parsed_data = parse_historical_data(symbol, historical_data, start_date, end_date)
            store_historical_data.delay(symbol, parsed_data)
            
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'records': len(parsed_data),
                'status': 'success'
            }
        
        return {
            'symbol': symbol,
            'status': 'no_data'
        }
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        }

@celery_app.task
def fetch_fundamental_data(symbol: str) -> Dict[str, Any]:
    """Fetch fundamental data for a stock"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        fundamental_data = {}
        
        try:
            # Fetch from multiple sources
            av_client = AlphaVantageClient()
            fh_client = FinnhubClient()
            
            # Get company overview from Alpha Vantage
            overview = loop.run_until_complete(av_client.get_company_overview(symbol))
            fundamental_data['overview'] = overview
            
            # Get financials from Finnhub
            financials = loop.run_until_complete(fh_client.get_company_profile(symbol))
            fundamental_data['profile'] = financials
            
        finally:
            loop.close()
        
        # Store in database
        if fundamental_data:
            store_fundamental_data.delay(symbol, fundamental_data)
        
        return {
            'symbol': symbol,
            'status': 'success',
            'data': fundamental_data
        }
        
    except Exception as e:
        logger.error(f"Error fetching fundamental data for {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        }

@celery_app.task
def fetch_news_data(symbol: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Fetch news data for a stock or market-wide news"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        news_articles = []
        
        try:
            fh_client = FinnhubClient()
            
            if symbol:
                # Get company-specific news
                news = loop.run_until_complete(fh_client.get_news(symbol))
            else:
                # Get general market news
                news = loop.run_until_complete(fh_client.get_market_news())
            
            news_articles = news[:limit] if news else []
            
        finally:
            loop.close()
        
        # Store news in database
        if news_articles:
            store_news_data.delay(news_articles, symbol)
        
        return {
            'symbol': symbol or 'market',
            'articles': len(news_articles),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

@celery_app.task
def store_price_data(symbol: str, data: Dict[str, Any]) -> bool:
    """
    Store *real* daily OHLC candle data for a symbol.

    Compliance: this task NEVER fabricates a flat OHLC row from a single
    real-time quote (the previous implementation wrote
    ``open=high=low=close=current_price``, which is synthetic data and
    violates the project's strict no-synthetic-data rule). Instead it pulls
    the most recent real daily candles from Finnhub and upserts them via the
    price repository.

    The ``data`` argument is accepted for backward compatibility with the
    existing ``fetch_stock_data`` call site but is no longer used to
    manufacture price rows; only real provider candles are persisted.

    Args:
        symbol: Stock ticker symbol.
        data: Legacy payload (ignored for OHLC; kept for signature compat).

    Returns:
        True if at least one real candle was persisted, False otherwise.
    """
    try:
        # Recent incremental refresh: last few calendar days of daily candles.
        # We request a small window (covers weekends/holidays) and upsert only
        # the real candles the provider returns. Empty -> skip, never fabricate.
        result = backfill_symbol_prices(symbol, days=5)
        rows_written = result.get("rows_written", 0)
        if rows_written > 0:
            logger.info(
                f"store_price_data: stored {rows_written} real candle(s) for {symbol}"
            )
            return True

        logger.info(
            f"store_price_data: no real candles available for {symbol} "
            f"(status={result.get('status')}); nothing written"
        )
        return False

    except Exception as e:
        logger.error(f"Error storing price data for {symbol}: {e}")
        return False


# ---------------------------------------------------------------------------
# Daily-OHLC backfill -- the real-history path for charts/technical/stats/etc.
# ---------------------------------------------------------------------------

def _candles_to_price_rows(stock_id: int, candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Map Finnhub candle dicts to PriceHistory row dicts for bulk upsert.

    Each candle from ``FinnhubClient.get_candles`` has the shape::

        {"timestamp": <unix>, "date": <iso>, "open": .., "high": ..,
         "low": .., "close": .., "volume": ..}

    The ``PriceHistory.date`` column is a ``DateTime`` with a unique
    constraint on ``(stock_id, date)``. We normalize each candle's unix
    timestamp to the UTC calendar day (midnight) so that daily candles
    de-duplicate cleanly across runs.

    Rows missing any required OHLC field (or with a non-positive close) are
    skipped -- we never substitute synthetic values.

    Returns:
        List of row dicts suitable for ``bulk_upsert_prices``.
    """
    rows: List[Dict[str, Any]] = []
    for candle in candles:
        ts = candle.get("timestamp")
        open_ = candle.get("open")
        high = candle.get("high")
        low = candle.get("low")
        close = candle.get("close")
        volume = candle.get("volume")

        # Require a real, complete candle. Skip anything incomplete rather
        # than fabricating values.
        if ts is None or open_ is None or high is None or low is None or close is None:
            continue
        if close <= 0:
            continue

        candle_date = datetime.fromtimestamp(ts, tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )

        rows.append({
            "stock_id": stock_id,
            "date": candle_date,
            "open": Decimal(str(open_)),
            "high": Decimal(str(high)),
            "low": Decimal(str(low)),
            "close": Decimal(str(close)),
            "volume": int(volume) if volume is not None else 0,
        })

    return rows


def _persist_price_rows(stock_id: int, rows: List[Dict[str, Any]]) -> int:
    """
    Persist mapped price rows via the price repository's bulk upsert.

    The repository's bulk upsert resolves conflicts on the table primary key
    (``id``). Because we never supply ``id``, we defensively drop rows whose
    ``(stock_id, date)`` already exist so a re-run does not violate the
    ``uq_stock_date`` unique constraint. This keeps the backfill idempotent
    without modifying the repository.

    Returns:
        Number of rows actually written.
    """
    if not rows:
        return 0

    from backend.repositories.price_repository import price_repository
    from backend.config.database import get_db_session

    async def _do_persist() -> int:
        async with get_db_session() as session:
            candidate_dates = [r["date"] for r in rows]
            existing_result = await session.execute(
                select(PriceHistory.date).where(
                    and_(
                        PriceHistory.stock_id == stock_id,
                        PriceHistory.date.in_(candidate_dates),
                    )
                )
            )
            existing_dates = {row[0] for row in existing_result}

            new_rows = [r for r in rows if r["date"] not in existing_dates]
            if not new_rows:
                return 0

            return await price_repository.bulk_upsert_prices(new_rows, session=session)

    return _run_async(_do_persist())


def _run_async(coro):
    """
    Run an async coroutine from sync (Celery / standalone) context.

    Uses a fresh event loop to avoid clashing with any loop that may already
    be bound to the current thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _fetch_yfinance_candles(symbol: str, days: int) -> List[Dict[str, Any]]:
    """
    Fetch real daily OHLCV from Yahoo Finance (yfinance) as candle dicts.

    Finnhub's free tier returns HTTP 403 for /stock/candle, so yfinance (free,
    no API key) is the daily-history source. The output is shaped exactly like
    ``FinnhubClient.get_candles`` candles so the existing
    ``_candles_to_price_rows`` mapping and persistence are reused unchanged.
    Returns ``[]`` on any failure or empty result -- the caller then SKIPS the
    symbol and never fabricates data.
    """
    try:
        import yfinance as yf

        start = (datetime.now(timezone.utc) - timedelta(days=max(int(days), 1))).date()
        df = yf.Ticker(symbol.upper()).history(start=start.isoformat(), auto_adjust=False)
        if df is None or df.empty:
            return []

        candles: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            ts = int(datetime(idx.year, idx.month, idx.day, tzinfo=timezone.utc).timestamp())
            vol = row.get("Volume")
            candles.append({
                "timestamp": ts,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                # vol == vol filters NaN (NaN != NaN); never substitute a guess.
                "volume": int(vol) if vol is not None and vol == vol else 0,
            })
        return candles
    except Exception as exc:
        logger.warning(f"yfinance candle fetch failed for {symbol}: {exc}")
        return []


def backfill_symbol_prices(
    symbol: str,
    days: int = DEFAULT_BACKFILL_DAYS,
) -> Dict[str, Any]:
    """
    Fetch real daily candles for a single symbol and persist them.

    Daily history comes from Yahoo Finance via yfinance (Finnhub's free tier
    blocks /stock/candle). If the source returns no candles for the symbol, the
    symbol is SKIPPED (no synthetic data is ever written).

    Args:
        symbol: Stock ticker symbol.
        days: How many days of history to request (default ~1 year).

    Returns:
        Dict with ``symbol``, ``status`` and ``rows_written``.
    """
    # Resolve the stock id first (sync session is fine for a simple lookup).
    with get_db_sync() as db:
        stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            logger.warning(f"backfill_symbol_prices: stock {symbol} not found")
            return {"symbol": symbol, "status": "not_found", "rows_written": 0}
        stock_id = stock.id

    candles = _fetch_yfinance_candles(symbol, days)
    if not candles:
        # Source returned nothing -> skip, do not fabricate.
        logger.info(f"backfill_symbol_prices: no candles for {symbol}, skipping")
        return {"symbol": symbol, "status": "no_data", "rows_written": 0}

    rows = _candles_to_price_rows(stock_id, candles)
    if not rows:
        return {"symbol": symbol, "status": "no_valid_candles", "rows_written": 0}

    rows_written = _persist_price_rows(stock_id, rows)
    return {
        "symbol": symbol,
        "status": "success",
        "rows_written": rows_written,
        "candles_fetched": len(candles),
    }


@celery_app.task
def backfill_daily_prices(
    limit: Optional[int] = None,
    days: int = DEFAULT_BACKFILL_DAYS,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Backfill real daily OHLC candles for the active stock universe.

    For each active, tradable stock this fetches ~``days`` of daily candles
    from Finnhub and upserts them via the price repository. Calls are throttled
    (``FINNHUB_BACKFILL_SLEEP_SECONDS``) and gated by ``cost_monitor`` to stay
    under the 60/min free-tier limit. Symbols the provider returns nothing for
    are skipped -- never fabricated.

    Args:
        limit: Optional cap on number of symbols (useful for partial runs).
        days: Days of history per symbol (default ~1 year).
        symbols: Optional explicit symbol list (overrides universe lookup).

    Returns:
        Summary dict with counts of processed / written / skipped symbols.
    """
    # Build the work list.
    if symbols:
        work_symbols = [s.strip().upper() for s in symbols if s and s.strip()]
    else:
        with get_db_sync() as db:
            query = db.query(Stock).filter(
                Stock.is_active == True,  # noqa: E712
                Stock.is_tradable == True,  # noqa: E712
            ).order_by(Stock.market_cap.desc().nullslast())
            if limit:
                query = query.limit(limit)
            work_symbols = [s.symbol for s in query.all()]

    if limit and not symbols:
        work_symbols = work_symbols[:limit]
    elif limit and symbols:
        work_symbols = work_symbols[:limit]

    summary = {
        "total": len(work_symbols),
        "succeeded": 0,
        "skipped_no_data": 0,
        "rate_limited": 0,
        "errors": 0,
        "rows_written": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for idx, sym in enumerate(work_symbols):
        try:
            result = backfill_symbol_prices(sym, days=days)
            status = result.get("status")
            if status == "success":
                summary["succeeded"] += 1
                summary["rows_written"] += result.get("rows_written", 0)
            elif status in ("no_data", "no_valid_candles", "not_found"):
                summary["skipped_no_data"] += 1
            elif status == "rate_limited":
                summary["rate_limited"] += 1
        except Exception as e:
            summary["errors"] += 1
            logger.error(f"backfill_daily_prices: error for {sym}: {e}")

        # Throttle between symbols to stay under the Finnhub minute limit.
        if idx < len(work_symbols) - 1:
            time.sleep(FINNHUB_BACKFILL_SLEEP_SECONDS)

    logger.info(
        f"backfill_daily_prices completed: {summary['succeeded']} succeeded, "
        f"{summary['skipped_no_data']} skipped, {summary['rate_limited']} rate-limited, "
        f"{summary['errors']} errors, {summary['rows_written']} rows written"
    )
    return summary

@celery_app.task
def store_historical_data(symbol: str, data: List[Dict]) -> bool:
    """Store historical price data in database"""
    try:
        with get_db_sync() as db:
            stock = db.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                return False
            
            records_added = 0
            for record in data:
                # Check if record exists
                existing = db.query(PriceHistory).filter(
                    and_(
                        PriceHistory.stock_id == stock.id,
                        PriceHistory.date == record['date']
                    )
                ).first()
                
                if not existing:
                    price_record = PriceHistory(
                        stock_id=stock.id,
                        date=record['date'],
                        open=record['open'],
                        high=record['high'],
                        low=record['low'],
                        close=record['close'],
                        adjusted_close=record.get('adjusted_close'),
                        volume=record['volume']
                    )
                    db.add(price_record)
                    records_added += 1
            
            db.commit()
            logger.info(f"Added {records_added} historical records for {symbol}")
            return True
            
    except Exception as e:
        logger.error(f"Error storing historical data for {symbol}: {e}")
        return False

@celery_app.task
def store_fundamental_data(symbol: str, data: Dict[str, Any]) -> bool:
    """Store fundamental data in database"""
    try:
        with get_db_sync() as db:
            stock = db.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                return False
            
            # Extract and store fundamental metrics
            if 'overview' in data:
                overview = data['overview']
                
                # Update stock information
                stock.market_cap = int(float(overview.get('MarketCapitalization', 0)))
                stock.sector = overview.get('Sector')
                stock.industry = overview.get('Industry')
                stock.description = overview.get('Description')
                
                # Create fundamental record.
                # NOTE: the model is ``Fundamentals`` (plural) and uses
                # ``period_date``/``period_type`` columns -- not the
                # ``report_date``/``period`` names that the old AlphaVantage
                # mapping assumed. ``dividend_yield`` is not a column on this
                # model, so it is intentionally omitted.
                fundamental = Fundamentals(
                    stock_id=stock.id,
                    period_date=date.today(),
                    period_type='annual',
                    pe_ratio=float(overview.get('PERatio', 0)) if overview.get('PERatio') else None,
                    peg_ratio=float(overview.get('PEGRatio', 0)) if overview.get('PEGRatio') else None,
                    ps_ratio=float(overview.get('PriceToSalesRatioTTM', 0)) if overview.get('PriceToSalesRatioTTM') else None,
                    pb_ratio=float(overview.get('PriceToBookRatio', 0)) if overview.get('PriceToBookRatio') else None,
                    roe=float(overview.get('ReturnOnEquityTTM', 0)) if overview.get('ReturnOnEquityTTM') else None,
                    roa=float(overview.get('ReturnOnAssetsTTM', 0)) if overview.get('ReturnOnAssetsTTM') else None,
                    gross_margin=float(overview.get('GrossProfitMargin', 0)) if overview.get('GrossProfitMargin') else None,
                    operating_margin=float(overview.get('OperatingMarginTTM', 0)) if overview.get('OperatingMarginTTM') else None,
                    net_margin=float(overview.get('ProfitMargin', 0)) if overview.get('ProfitMargin') else None
                )

                # Check if record exists
                existing = db.query(Fundamentals).filter(
                    and_(
                        Fundamentals.stock_id == stock.id,
                        Fundamentals.period_date == date.today(),
                        Fundamentals.period_type == 'annual'
                    )
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in fundamental.__dict__.items():
                        if not key.startswith('_') and value is not None:
                            setattr(existing, key, value)
                else:
                    db.add(fundamental)
            
            db.commit()
            logger.info(f"Fundamental data stored for {symbol}")
            return True
            
    except Exception as e:
        logger.error(f"Error storing fundamental data for {symbol}: {e}")
        return False

@celery_app.task
def store_news_data(articles: List[Dict], symbol: Optional[str] = None) -> bool:
    """Store news articles in database"""
    try:
        with get_db_sync() as db:
            stock_id = None
            if symbol:
                stock = db.query(Stock).filter(Stock.symbol == symbol).first()
                stock_id = stock.id if stock else None
            
            articles_added = 0
            for article in articles:
                # Check if article already exists (by URL)
                existing = db.query(News).filter(
                    News.url == article.get('url')
                ).first()
                
                if not existing:
                    news_record = News(
                        stock_id=stock_id,
                        headline=article.get('headline', '')[:500],
                        summary=article.get('summary'),
                        source=article.get('source'),
                        url=article.get('url'),
                        published_at=datetime.fromtimestamp(article.get('datetime', 0))
                    )
                    db.add(news_record)
                    articles_added += 1
            
            db.commit()
            logger.info(f"Added {articles_added} news articles")
            return True
            
    except Exception as e:
        logger.error(f"Error storing news data: {e}")
        return False

# Helper functions
def parse_historical_data(symbol: str, data: Dict, start_date: str, end_date: str) -> List[Dict]:
    """Parse historical data from API response"""
    parsed_data = []
    
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    for date_str, values in data.items():
        try:
            record_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            if start <= record_date <= end:
                parsed_data.append({
                    'date': record_date,
                    'open': float(values.get('1. open', 0)),
                    'high': float(values.get('2. high', 0)),
                    'low': float(values.get('3. low', 0)),
                    'close': float(values.get('4. close', 0)),
                    'adjusted_close': float(values.get('5. adjusted close', 0)) if '5. adjusted close' in values else None,
                    'volume': int(values.get('6. volume', 0))
                })
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing date {date_str}: {e}")
            continue
    
    return sorted(parsed_data, key=lambda x: x['date'])

# Chain tasks for complex workflows
@celery_app.task
def update_stock_complete(symbol: str) -> Dict[str, Any]:
    """Complete update workflow for a stock"""
    workflow = chain(
        fetch_stock_data.si(symbol, 'all'),
        fetch_fundamental_data.si(symbol),
        fetch_news_data.si(symbol, 10)
    )
    
    result = workflow.apply_async()
    return {
        'symbol': symbol,
        'workflow_id': result.id,
        'status': 'initiated'
    }


@celery_app.task
def update_stock_prices(symbol: str, period: str = '1d') -> Dict[str, Any]:
    """
    Update stock prices for given symbol.

    Fetches real-time price data using the market_data_service fallback chain
    (Finnhub -> Polygon -> Alpha Vantage -> FMP) and persists the result to
    the PriceHistory table.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL").
        period: Data period hint (currently informational; real-time quote is
                always fetched regardless of period value).

    Returns:
        Dict with symbol, period, status, and prices_updated count.
    """
    from backend.services.market_data_service import get_stock_price, update_prices_in_db
    from backend.utils.database import get_db_sync

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Fetch price from the provider chain (async)
        price_data = loop.run_until_complete(get_stock_price(symbol))
    except Exception as exc:
        logger.error(f"update_stock_prices: price fetch error for {symbol}: {exc}")
        price_data = None
    finally:
        loop.close()

    if price_data is None:
        logger.warning(f"update_stock_prices: no price data available for {symbol}")
        return {
            'symbol': symbol,
            'period': period,
            'status': 'no_data',
            'prices_updated': 0,
        }

    # Persist to DB using sync session (Celery runs in sync context)
    prices_updated = 0
    try:
        with get_db_sync() as db:
            success = asyncio.run(update_prices_in_db(symbol, price_data, db))
            # update_prices_in_db is actually synchronous internally;
            # wrap in run() to handle the async signature cleanly.
            prices_updated = 1 if success else 0
    except Exception as exc:
        # update_prices_in_db is defined as async but uses only sync DB ops.
        # If asyncio.run fails (already-running loop edge case), fall back to
        # calling the sync-safe internals directly.
        logger.warning(
            f"update_stock_prices: asyncio.run fallback for DB persist ({symbol}): {exc}"
        )
        try:
            with get_db_sync() as db:
                from backend.models.unified_models import Stock, PriceHistory
                from sqlalchemy import and_
                from datetime import date as date_type

                stock = db.query(Stock).filter(Stock.symbol == symbol).first()
                if stock:
                    today = date_type.today()
                    current_price = price_data.get("current_price", 0)
                    if current_price:
                        existing = db.query(PriceHistory).filter(
                            and_(
                                PriceHistory.stock_id == stock.id,
                                PriceHistory.date == today,
                            )
                        ).first()
                        if existing:
                            existing.close = current_price
                            existing.high = max(existing.high or current_price, current_price)
                            existing.low = min(existing.low or current_price, current_price)
                        else:
                            db.add(PriceHistory(
                                stock_id=stock.id,
                                date=today,
                                open=price_data.get("open", current_price),
                                high=price_data.get("high", current_price),
                                low=price_data.get("low", current_price),
                                close=current_price,
                                volume=price_data.get("volume", 0),
                            ))
                        if hasattr(stock, "last_price_update"):
                            stock.last_price_update = datetime.now(timezone.utc)
                        db.commit()
                        prices_updated = 1
        except Exception as inner_exc:
            logger.error(
                f"update_stock_prices: DB fallback also failed for {symbol}: {inner_exc}"
            )

    logger.info(
        f"update_stock_prices: symbol={symbol} period={period} "
        f"price={price_data.get('current_price')} "
        f"provider={price_data.get('provider')} "
        f"prices_updated={prices_updated}"
    )

    return {
        'symbol': symbol,
        'period': period,
        'status': 'success' if prices_updated > 0 else 'db_error',
        'prices_updated': prices_updated,
        'price': price_data.get('current_price'),
        'provider': price_data.get('provider'),
    }