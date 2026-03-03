"""
Tests for Cache Warming System
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from backend.utils.cache_warmer import CacheWarmer, get_cache_warmer
from backend.utils.comprehensive_cache import get_cache_manager


@pytest.fixture
def cache_warmer():
    """Create a cache warmer instance"""
    return CacheWarmer()


@pytest.fixture
def mock_data_fetchers():
    """Create mock data fetchers for testing"""
    async def mock_quote_fetcher(symbol: str):
        return {
            "symbol": symbol,
            "price": 100.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def mock_overview_fetcher(symbol: str):
        return {
            "symbol": symbol,
            "name": f"{symbol} Inc.",
            "sector": "Technology"
        }

    return {
        "real_time_quote": mock_quote_fetcher,
        "company_overview": mock_overview_fetcher
    }


@pytest.mark.asyncio
async def test_cache_warmer_initialization(cache_warmer):
    """Test cache warmer initializes correctly"""
    assert cache_warmer is not None
    assert len(cache_warmer.top_stocks) == 20
    assert len(cache_warmer.etf_list) == 5
    assert cache_warmer.is_warming is False
    assert "AAPL" in cache_warmer.top_stocks
    assert "SPY" in cache_warmer.etf_list


@pytest.mark.slow
@pytest.mark.asyncio
async def test_warm_top_stocks(cache_warmer, mock_data_fetchers):
    """Test warming cache for top stocks"""
    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get.return_value = (None, 'miss')  # Cache miss
        mock_manager.set = AsyncMock()
        mock_get_manager.return_value = mock_manager

        stats = await cache_warmer.warm_top_stocks(mock_data_fetchers)

        assert stats is not None
        assert "warmed" in stats
        assert "failed" in stats
        assert "skipped" in stats
        assert stats["warmed"] > 0


@pytest.mark.asyncio
async def test_warm_top_stocks_skip_existing(cache_warmer, mock_data_fetchers):
    """Test that existing cache entries are skipped"""
    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        # Cache hit - should skip
        mock_manager.get.return_value = ({"existing": "data"}, 'l1')
        mock_manager.set = AsyncMock()
        mock_get_manager.return_value = mock_manager

        stats = await cache_warmer.warm_top_stocks(mock_data_fetchers)

        assert stats["skipped"] > 0
        # Should not call set if cache already exists
        assert mock_manager.set.call_count == 0


@pytest.mark.asyncio
async def test_warm_top_stocks_handles_errors(cache_warmer):
    """Test that warming handles errors gracefully"""
    async def failing_fetcher(symbol: str):
        raise Exception("API error")

    data_fetchers = {"real_time_quote": failing_fetcher}

    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get.return_value = (None, 'miss')
        mock_manager.set = AsyncMock()
        mock_get_manager.return_value = mock_manager

        stats = await cache_warmer.warm_top_stocks(data_fetchers)

        assert "failed" in stats
        assert stats["failed"] > 0


@pytest.mark.asyncio
async def test_prevent_concurrent_warming(cache_warmer, mock_data_fetchers):
    """Test that concurrent warming is prevented"""
    cache_warmer.is_warming = True

    stats = await cache_warmer.warm_top_stocks(mock_data_fetchers)

    assert stats["status"] == "already_running"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_warming_status_is_reset(cache_warmer, mock_data_fetchers):
    """Test that is_warming flag is reset after completion"""
    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get.return_value = (None, 'miss')
        mock_manager.set = AsyncMock()
        mock_get_manager.return_value = mock_manager

        assert cache_warmer.is_warming is False

        await cache_warmer.warm_top_stocks(mock_data_fetchers)

        assert cache_warmer.is_warming is False


@pytest.mark.asyncio
async def test_warming_status_reset_on_error(cache_warmer):
    """Test that is_warming flag is reset even on error"""
    async def failing_fetcher(symbol: str):
        raise Exception("Critical error")

    data_fetchers = {"real_time_quote": failing_fetcher}

    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get.side_effect = Exception("Manager error")
        mock_get_manager.return_value = mock_manager

        try:
            await cache_warmer.warm_top_stocks(data_fetchers)
        except:
            pass

        # Should reset is_warming even on error
        assert cache_warmer.is_warming is False


@pytest.mark.asyncio
async def test_get_warming_status(cache_warmer):
    """Test getting warming status"""
    status = cache_warmer.get_warming_status()

    assert "is_warming" in status
    assert "periodic_warming_active" in status
    assert "top_stocks_count" in status
    assert "etf_count" in status
    assert status["top_stocks_count"] == 20
    assert status["etf_count"] == 5


@pytest.mark.asyncio
async def test_stop_periodic_warming(cache_warmer):
    """Test stopping periodic warming"""
    # Create a mock task
    mock_task = MagicMock()
    mock_task.done.return_value = False
    mock_task.cancel = MagicMock()
    cache_warmer._warming_task = mock_task

    cache_warmer.stop_periodic_warming()

    mock_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_get_cache_warmer_singleton():
    """Test that get_cache_warmer returns singleton"""
    warmer1 = get_cache_warmer()
    warmer2 = get_cache_warmer()

    assert warmer1 is warmer2


@pytest.mark.asyncio
async def test_rate_limiting_between_stocks(cache_warmer, mock_data_fetchers):
    """Test that rate limiting is applied between stocks"""
    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        with patch('asyncio.sleep', new=AsyncMock()) as mock_sleep:
            mock_manager = AsyncMock()
            mock_manager.get.return_value = (None, 'miss')
            mock_manager.set = AsyncMock()
            mock_get_manager.return_value = mock_manager

            await cache_warmer.warm_top_stocks(mock_data_fetchers)

            # Should have called sleep for rate limiting
            assert mock_sleep.call_count > 0


@pytest.mark.asyncio
async def test_warm_all_symbol_types(cache_warmer, mock_data_fetchers):
    """Test that both stocks and ETFs are warmed"""
    warmed_symbols = set()

    async def tracking_fetcher(symbol: str):
        warmed_symbols.add(symbol)
        return {"symbol": symbol}

    data_fetchers = {"real_time_quote": tracking_fetcher}

    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get.return_value = (None, 'miss')
        mock_manager.set = AsyncMock()
        mock_get_manager.return_value = mock_manager

        await cache_warmer.warm_top_stocks(data_fetchers)

        # Check that we warmed both stocks and ETFs
        assert any(stock in warmed_symbols for stock in cache_warmer.top_stocks)
        assert any(etf in warmed_symbols for etf in cache_warmer.etf_list)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_warming_statistics_complete(cache_warmer, mock_data_fetchers):
    """Test that warming statistics are complete"""
    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get.return_value = (None, 'miss')
        mock_manager.set = AsyncMock()
        mock_get_manager.return_value = mock_manager

        stats = await cache_warmer.warm_top_stocks(mock_data_fetchers)

        assert "start_time" in stats
        assert "end_time" in stats
        assert "warmed" in stats
        assert "failed" in stats
        assert "skipped" in stats


@pytest.mark.asyncio
async def test_partial_fetch_failure(cache_warmer):
    """Test handling of partial failures during warming"""
    call_count = {"count": 0}

    async def intermittent_fetcher(symbol: str):
        call_count["count"] += 1
        if call_count["count"] % 3 == 0:
            raise Exception("Intermittent failure")
        return {"symbol": symbol}

    data_fetchers = {"real_time_quote": intermittent_fetcher}

    with patch('backend.utils.cache_warmer.get_cache_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get.return_value = (None, 'miss')
        mock_manager.set = AsyncMock()
        mock_get_manager.return_value = mock_manager

        stats = await cache_warmer.warm_top_stocks(data_fetchers)

        # Should have both successes and failures
        assert stats["warmed"] > 0
        assert stats["failed"] > 0
