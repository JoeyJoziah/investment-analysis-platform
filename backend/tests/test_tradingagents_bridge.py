"""Unit tests for backend.tradingagents_bridge.persistence — no DB required."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.tradingagents_bridge.persistence import (
    map_action_to_iap,
    persist_tradingagents_decision,
)


def test_map_action_buy_low_confidence():
    assert map_action_to_iap("BUY", 0.5) == "buy"


def test_map_action_buy_high_confidence():
    assert map_action_to_iap("BUY", 0.85) == "strong_buy"


def test_map_action_sell_low_confidence():
    assert map_action_to_iap("SELL", 0.4) == "sell"


def test_map_action_hold_default():
    assert map_action_to_iap("HOLD") == "hold"
    assert map_action_to_iap("anything-else") == "hold"


def test_skipped_when_persist_disabled(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_PERSIST", "0")
    result = persist_tradingagents_decision(
        state={"final_trade_decision": "BUY"},
        ticker="NVDA",
        trade_date="2025-04-25",
        session=MagicMock(),
    )
    assert result.persisted is False
    assert "disabled" in result.reason


def test_skipped_when_no_decision(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_PERSIST", "1")
    result = persist_tradingagents_decision(
        state={},
        ticker="NVDA",
        trade_date="2025-04-25",
        session=MagicMock(),
    )
    assert result.persisted is False
    assert "no decision" in result.reason


def test_skipped_when_stock_missing(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_PERSIST", "1")
    session = MagicMock()
    session.query.return_value.filter_by.return_value.first.return_value = None
    result = persist_tradingagents_decision(
        state={"final_trade_decision": "BUY NVDA"},
        ticker="NVDA",
        trade_date="2025-04-25",
        session=session,
    )
    assert result.persisted is False
    assert "universe" in result.reason


def test_persist_happy_path(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_PERSIST", "1")
    session = MagicMock()
    fake_stock = MagicMock()
    fake_stock.id = 42
    session.query.return_value.filter_by.return_value.first.return_value = fake_stock
    fake_rec = MagicMock()
    fake_rec.id = 7
    # session.refresh is called after add/commit; mutate the row's id attribute.
    def _refresh(obj):
        obj.id = 7
    session.refresh.side_effect = _refresh

    result = persist_tradingagents_decision(
        state={
            "final_trade_decision": "Recommendation: BUY NVDA with disciplined entry",
            "market_report": "ok",
            "news_report": "ok",
            "sentiment_report": "ok",
            "fundamentals_report": "ok",
            "investment_debate_state": {"bull_history": "b1", "bear_history": "b2"},
            "risk_debate_state": {"history": "h"},
        },
        ticker="NVDA",
        trade_date="2025-04-25",
        session=session,
        confidence=0.65,
    )
    assert result.persisted is True
    assert result.recommendation_id == 7
    session.add.assert_called_once()
    session.commit.assert_called_once()
