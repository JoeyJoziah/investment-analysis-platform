#!/usr/bin/env python3
"""
Stock Universe Seed Script (S&P 500)

Populates the ``stocks`` table with the REAL S&P 500 constituent universe so
the sectors / heatmap / breadth / movers features and the daily price-backfill
have a genuine universe to work with.

The constituent list comes from the real Wikipedia "List of S&P 500 companies"
table (parsed with pandas). No tickers or sectors are ever fabricated: if the
source is unavailable the script FAILS LOUDLY (non-zero exit) rather than
seeding an invented universe.

Idempotent: stocks are upserted by ``symbol`` and sectors by ``name``, so it is
safe to run repeatedly. A second run inserts no duplicates.

Usage:
    cd <repo root>
    python scripts/seed_universe.py

    # Fetch + parse only, do not write to the DB:
    python scripts/seed_universe.py --dry-run

    # Use NASDAQ as the listing exchange instead of NYSE:
    python scripts/seed_universe.py --exchange NASDAQ
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap - must run before any backend imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so settings (DATABASE_URL etc.) are satisfied before backend imports
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from backend.config.database import get_db_session, cleanup_database  # noqa: E402
from backend.tasks.stock_universe_fetcher import (  # noqa: E402
    UniverseSourceError,
    fetch_sp500_universe,
    persist_universe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_universe")


async def seed_universe(*, exchange: str = "NYSE", dry_run: bool = False) -> dict:
    """Fetch the real S&P 500 universe and persist it idempotently.

    Returns the persist summary dict, or the fetch-only summary in dry-run mode.
    Raises UniverseSourceError (propagated) if the real source is unavailable.
    """
    logger.info("=" * 60)
    logger.info("Investment Analysis Platform - S&P 500 Universe Seeder")
    if dry_run:
        logger.info("MODE: DRY RUN (no changes will be written)")
    logger.info("=" * 60)

    stocks = fetch_sp500_universe()
    sectors = {s.sector for s in stocks if s.sector}
    logger.info(
        "Fetched %d real constituents across %d sectors",
        len(stocks),
        len(sectors),
    )

    if dry_run:
        return {
            "seeded": 0,
            "created": 0,
            "updated": 0,
            "sectors": len(sectors),
            "fetched": len(stocks),
        }

    # The caller owns the transaction. get_db_session commits on clean exit and
    # rolls back on exception, so persist runs atomically.
    async with get_db_session() as session:
        summary = await persist_universe(
            stocks, session=session, default_exchange=exchange
        )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the stocks table with the real S&P 500 universe."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse the real source only; do not write to the DB.",
    )
    parser.add_argument(
        "--exchange",
        default="NYSE",
        help="Listing exchange code to associate with seeded stocks (default: NYSE).",
    )
    args = parser.parse_args()

    try:
        summary = asyncio.run(
            seed_universe(exchange=args.exchange, dry_run=args.dry_run)
        )
    except UniverseSourceError as exc:
        # Real source unavailable -> fail loudly, do not invent data.
        logger.error("Universe seed FAILED: %s", exc)
        return 1
    except Exception as exc:  # surface any unexpected failure clearly
        logger.exception("Universe seed FAILED unexpectedly: %s", exc)
        return 1
    finally:
        # Dispose the async engine/pool so the process exits cleanly.
        try:
            asyncio.run(cleanup_database())
        except Exception:  # cleanup is best-effort
            pass

    if args.dry_run:
        print(
            f"\n[DRY-RUN] Would seed {summary['fetched']} stocks "
            f"across {summary['sectors']} sectors (no changes written)."
        )
    else:
        print(
            f"\nSeeded {summary['seeded']} stocks across "
            f"{summary['sectors']} sectors "
            f"({summary['created']} new, {summary['updated']} updated)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
