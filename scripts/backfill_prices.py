#!/usr/bin/env python3
"""
Daily-OHLC Price Backfill (initial load)

Pulls REAL daily candles from Finnhub for the active stock universe and upserts
them into the ``price_history`` table via the price repository. This is the
initial-load companion to the scheduled Celery task
``backend.tasks.data_tasks.backfill_daily_prices`` and shares the exact same
candle -> row mapping and persistence path.

Compliance: only real provider candles are written. Symbols the provider returns
nothing for are SKIPPED -- the script NEVER fabricates flat/synthetic OHLC rows.

The run is rate-limit-safe: every Finnhub call is gated by ``cost_monitor`` and
throttled with a short sleep so a single run stays well under the 60/min
free-tier limit.

Usage:
    cd <repo root>

    # Small first run -- top 20 stocks by market cap:
    python scripts/backfill_prices.py --limit 20

    # Specific symbols only:
    python scripts/backfill_prices.py --symbols AAPL,MSFT,NVDA

    # Custom history window (default ~1 year):
    python scripts/backfill_prices.py --limit 20 --days 180

    # Fetch + map only, do not write to the DB:
    python scripts/backfill_prices.py --limit 5 --dry-run
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap - must run before any backend imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import select, and_  # noqa: E402

from backend.config.database import get_db_session, cleanup_database  # noqa: E402
from backend.data_ingestion.finnhub_client import FinnhubClient  # noqa: E402
from backend.models.unified_models import Stock, PriceHistory  # noqa: E402
from backend.repositories.price_repository import price_repository  # noqa: E402
from backend.tasks.data_tasks import (  # noqa: E402
    DEFAULT_BACKFILL_DAYS,
    FINNHUB_BACKFILL_SLEEP_SECONDS,
    _candles_to_price_rows,
    _fetch_yfinance_candles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_prices")


async def _resolve_symbols(
    *,
    limit: Optional[int],
    symbols: Optional[List[str]],
) -> List[Dict[str, int]]:
    """
    Resolve the work list of {symbol, stock_id} dicts.

    If ``symbols`` is given, only those (that exist + are active) are returned.
    Otherwise the active/tradable universe ordered by market cap is used,
    optionally capped at ``limit``.
    """
    async with get_db_session(readonly=True) as session:
        query = select(Stock.symbol, Stock.id).where(
            and_(Stock.is_active.is_(True), Stock.is_tradable.is_(True))
        )

        if symbols:
            upper = [s.strip().upper() for s in symbols if s and s.strip()]
            query = query.where(Stock.symbol.in_(upper))
        else:
            query = query.order_by(Stock.market_cap.desc().nullslast())

        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        return [{"symbol": row[0], "stock_id": row[1]} for row in result]


async def _persist_new_rows(stock_id: int, rows: List[dict]) -> int:
    """
    Upsert only the candle rows whose (stock_id, date) does not already exist.

    The repository upsert conflict-resolves on the table primary key, so we
    pre-filter on the ``uq_stock_date`` unique constraint to keep the backfill
    idempotent across re-runs.
    """
    if not rows:
        return 0

    async with get_db_session() as session:
        candidate_dates = [r["date"] for r in rows]
        existing = await session.execute(
            select(PriceHistory.date).where(
                and_(
                    PriceHistory.stock_id == stock_id,
                    PriceHistory.date.in_(candidate_dates),
                )
            )
        )
        existing_dates = {row[0] for row in existing}

        new_rows = [r for r in rows if r["date"] not in existing_dates]
        if not new_rows:
            return 0

        return await price_repository.bulk_upsert_prices(new_rows, session=session)


async def backfill(
    *,
    limit: Optional[int],
    symbols: Optional[List[str]],
    days: int,
    dry_run: bool,
) -> Dict[str, int]:
    """Run the backfill and return a summary dict."""
    work = await _resolve_symbols(limit=limit, symbols=symbols)
    total = len(work)

    summary = {
        "total": total,
        "succeeded": 0,
        "skipped_no_data": 0,
        "rate_limited": 0,
        "errors": 0,
        "rows_written": 0,
    }

    if total == 0:
        logger.warning("No matching active symbols found; nothing to backfill.")
        return summary

    logger.info(
        "Backfilling %d symbol(s), %d day(s) of daily candles each%s",
        total,
        days,
        " (dry-run)" if dry_run else "",
    )

    for idx, item in enumerate(work, start=1):
        symbol = item["symbol"]
        stock_id = item["stock_id"]

        try:
            # Daily history comes from Yahoo Finance (Finnhub's free tier blocks
            # /stock/candle with a 403). yfinance is free + keyless; we still
            # throttle between symbols to be polite to the source.
            candles = _fetch_yfinance_candles(symbol, days)

            if not candles:
                # Source returned nothing -> skip, never fabricate.
                summary["skipped_no_data"] += 1
                logger.info("[%d/%d] %s: no candles, skipped", idx, total, symbol)
                await asyncio.sleep(FINNHUB_BACKFILL_SLEEP_SECONDS)
                continue

            rows = _candles_to_price_rows(stock_id, candles)
            if not rows:
                summary["skipped_no_data"] += 1
                logger.info("[%d/%d] %s: no valid candles, skipped", idx, total, symbol)
                await asyncio.sleep(FINNHUB_BACKFILL_SLEEP_SECONDS)
                continue

            if dry_run:
                summary["succeeded"] += 1
                logger.info(
                    "[%d/%d] %s: %d candle(s) mapped (dry-run, not written)",
                    idx, total, symbol, len(rows),
                )
            else:
                written = await _persist_new_rows(stock_id, rows)
                summary["succeeded"] += 1
                summary["rows_written"] += written
                logger.info(
                    "[%d/%d] %s: %d candle(s) fetched, %d new row(s) written",
                    idx, total, symbol, len(rows), written,
                )

        except Exception as exc:  # noqa: BLE001 - report and continue
            summary["errors"] += 1
            logger.error("[%d/%d] %s: error: %s", idx, total, symbol, exc)

        # Throttle between symbols to stay under the minute limit.
        if idx < total:
            await asyncio.sleep(FINNHUB_BACKFILL_SLEEP_SECONDS)

    return summary


def _print_summary(summary: Dict[str, int], *, dry_run: bool) -> None:
    print("\n" + "=" * 56)
    print("PRICE BACKFILL SUMMARY" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 56)
    print(f"Symbols processed : {summary['total']}")
    print(f"Succeeded         : {summary['succeeded']}")
    print(f"Skipped (no data) : {summary['skipped_no_data']}")
    print(f"Rate-limited      : {summary['rate_limited']}")
    print(f"Errors            : {summary['errors']}")
    print(f"Rows written      : {summary['rows_written']:,}")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill real daily OHLC candles for the active stock universe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of symbols to process (by market cap). Useful for a small first run.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated explicit symbols (e.g. AAPL,MSFT,NVDA). Overrides universe order.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_BACKFILL_DAYS,
        help=f"Days of daily history to fetch per symbol (default: {DEFAULT_BACKFILL_DAYS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + map candles but do not write to the database.",
    )

    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None

    async def _run() -> int:
        try:
            summary = await backfill(
                limit=args.limit,
                symbols=symbols,
                days=args.days,
                dry_run=args.dry_run,
            )
            _print_summary(summary, dry_run=args.dry_run)
            # Non-zero exit only on hard errors with zero successes.
            if summary["total"] > 0 and summary["succeeded"] == 0 and summary["errors"] > 0:
                return 1
            return 0
        finally:
            await cleanup_database()

    exit_code = asyncio.run(_run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
