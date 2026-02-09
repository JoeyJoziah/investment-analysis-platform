"""
Cache Warming Utility for Investment Analysis Platform

Warms cache for top stocks on startup and refreshes periodically
to reduce latency and API costs.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone

from backend.utils.comprehensive_cache import get_cache_manager

logger = logging.getLogger(__name__)


class CacheWarmer:
    """
    Simple cache warmer for top stocks
    Warms most common queries to reduce latency
    """

    def __init__(self):
        self.top_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "BRK.B", "JPM", "V",
            "JNJ", "WMT", "PG", "MA", "UNH",
            "HD", "DIS", "BAC", "ADBE", "NFLX"
        ]
        self.etf_list = ["SPY", "QQQ", "IWM", "VTI", "VOO"]
        self.is_warming = False
        self._warming_task: Optional[asyncio.Task] = None

    async def warm_top_stocks(self, data_fetchers: Dict[str, callable]) -> Dict[str, int]:
        """
        Warm cache for top stocks using provided data fetchers

        Args:
            data_fetchers: Dictionary of {data_type: async_fetch_function}

        Returns:
            Dictionary with warming statistics
        """
        if self.is_warming:
            logger.info("Cache warming already in progress")
            return {"status": "already_running"}

        self.is_warming = True
        cache_manager = await get_cache_manager()

        stats = {
            "warmed": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": datetime.now(timezone.utc).isoformat()
        }

        try:
            symbols_to_warm = self.top_stocks + self.etf_list

            for symbol in symbols_to_warm:
                for data_type, fetch_func in data_fetchers.items():
                    try:
                        # Check if already cached
                        cache_key = f"{symbol}:{data_type}"
                        existing, _ = await cache_manager.get(
                            data_type=data_type,
                            identifier=cache_key
                        )

                        if existing:
                            stats["skipped"] += 1
                            logger.debug(f"Cache already exists for {symbol} {data_type}")
                            continue

                        # Fetch and cache data
                        data = await fetch_func(symbol)

                        if data:
                            await cache_manager.set(
                                data_type=data_type,
                                identifier=cache_key,
                                data=data
                            )
                            stats["warmed"] += 1
                            logger.debug(f"Warmed cache for {symbol} {data_type}")
                        else:
                            stats["failed"] += 1

                    except Exception as e:
                        stats["failed"] += 1
                        logger.warning(f"Failed to warm cache for {symbol} {data_type}: {e}")

                    # Rate limiting - avoid overwhelming APIs
                    await asyncio.sleep(0.2)

            stats["end_time"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"Cache warming completed: {stats['warmed']} warmed, {stats['failed']} failed, {stats['skipped']} skipped")

        except Exception as e:
            logger.error(f"Cache warming error: {e}")
            stats["error"] = str(e)
        finally:
            self.is_warming = False

        return stats

    async def start_periodic_warming(
        self,
        data_fetchers: Dict[str, callable],
        interval_hours: int = 4
    ):
        """
        Start periodic cache warming in background

        Args:
            data_fetchers: Dictionary of {data_type: async_fetch_function}
            interval_hours: How often to refresh cache (default 4 hours)
        """
        if self._warming_task and not self._warming_task.done():
            logger.warning("Periodic warming already running")
            return

        async def _warming_loop():
            while True:
                try:
                    logger.info(f"Starting periodic cache warming (interval: {interval_hours}h)")
                    await self.warm_top_stocks(data_fetchers)

                    # Sleep until next warming cycle
                    await asyncio.sleep(interval_hours * 3600)

                except asyncio.CancelledError:
                    logger.info("Periodic cache warming cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in periodic cache warming: {e}")
                    # Wait 30 minutes before retry on error
                    await asyncio.sleep(1800)

        self._warming_task = asyncio.create_task(_warming_loop())
        logger.info("Periodic cache warming started")

    def stop_periodic_warming(self):
        """Stop periodic cache warming"""
        if self._warming_task and not self._warming_task.done():
            self._warming_task.cancel()
            logger.info("Periodic cache warming stopped")

    def get_warming_status(self) -> Dict[str, any]:
        """Get current warming status"""
        return {
            "is_warming": self.is_warming,
            "periodic_warming_active": self._warming_task is not None and not self._warming_task.done(),
            "top_stocks_count": len(self.top_stocks),
            "etf_count": len(self.etf_list)
        }


# Global cache warmer instance
_cache_warmer: Optional[CacheWarmer] = None


def get_cache_warmer() -> CacheWarmer:
    """Get global cache warmer instance"""
    global _cache_warmer

    if _cache_warmer is None:
        _cache_warmer = CacheWarmer()

    return _cache_warmer
