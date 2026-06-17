"""
Regression test for #208 item 1 (#200 follow-up): base_price wiring.

`create_prediction` used to hardcode ``base_price = None`` with a
``TODO(#200-follow-up)``.  It now resolves a *real* quote via
``_fetch_base_price`` -> realtime price service.  The critical contract is that
the helper stays None (fail-loud per #200) whenever a real price is unavailable,
and never substitutes a fabricated constant.

Run with::

    ENVIRONMENT=test JWT_SECRET_KEY=x SECRET_KEY=y MASTER_SECRET_KEY=z \
      DATABASE_URL=postgresql://u:p@localhost/db REDIS_URL=redis://localhost \
      pytest backend/tests/test_ml_base_price_wiring.py --noconftest
"""
import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock

ml = importlib.import_module("backend.api.routers.ml")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _service_returning(update):
    service = MagicMock()
    service.get_latest_price = AsyncMock(return_value=update)
    return AsyncMock(return_value=service)


def test_returns_real_price_when_available(monkeypatch):
    update = MagicMock()
    update.price = 123.45
    monkeypatch.setattr(ml, "get_realtime_price_service", _service_returning(update))
    assert _run(ml._fetch_base_price("AAPL", MagicMock())) == 123.45


def test_none_when_no_quote(monkeypatch):
    monkeypatch.setattr(ml, "get_realtime_price_service", _service_returning(None))
    assert _run(ml._fetch_base_price("AAPL", MagicMock())) is None


def test_none_when_price_is_zero(monkeypatch):
    update = MagicMock()
    update.price = 0
    monkeypatch.setattr(ml, "get_realtime_price_service", _service_returning(update))
    assert _run(ml._fetch_base_price("AAPL", MagicMock())) is None


def test_none_on_service_error(monkeypatch):
    monkeypatch.setattr(
        ml, "get_realtime_price_service", AsyncMock(side_effect=RuntimeError("boom"))
    )
    assert _run(ml._fetch_base_price("AAPL", MagicMock())) is None
