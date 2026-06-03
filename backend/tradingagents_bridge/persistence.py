"""Persist a TradingAgents decision into IAP's Postgres schema."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from backend.models.unified_models import Recommendation, Stock

logger = logging.getLogger(__name__)


# Mapping from TradingAgents action -> IAP RecommendationTypeEnum string value.
# IAP uses: strong_buy, buy, hold, sell, strong_sell.
def map_action_to_iap(action: str, confidence: float | None = None) -> str:
    a = (action or "HOLD").upper()
    c = confidence if confidence is not None else 0.6
    if a == "BUY":
        return "strong_buy" if c >= 0.8 else "buy"
    if a == "SELL":
        return "strong_sell" if c >= 0.8 else "sell"
    return "hold"


@dataclass
class PersistenceResult:
    persisted: bool
    recommendation_id: int | None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.persisted


def _disabled() -> bool:
    return os.getenv("TRADINGAGENTS_PERSIST", "0") in ("0", "false", "False", "")


def persist_tradingagents_decision(
    *,
    state: dict[str, Any],
    ticker: str,
    trade_date: str,
    session: Session,
    confidence: float = 0.65,
    extra_factors: dict[str, Any] | None = None,
) -> PersistenceResult:
    """Insert a TradingAgents result as a row in IAP's `recommendations` table.

    Args:
        state: full agent state from the state log JSON (or in-memory propagate
            result). Must contain final_trade_decision and the four reports.
        ticker: stock symbol.
        trade_date: ISO date (YYYY-MM-DD) the recommendation was produced for.
        session: a SQLAlchemy sync session (use SessionLocal from
            backend.utils.database).
        confidence: caller-supplied confidence in [0,1]. Default 0.65 if the
            parser doesn't surface one.
        extra_factors: merged into key_factors JSON.

    Skips silently (returns persisted=False) when:
        - TRADINGAGENTS_PERSIST env var is not enabled
        - The stock is not present in IAP's `stocks` table
        - `final_trade_decision` is absent
    """
    if _disabled():
        return PersistenceResult(False, None, "persistence disabled (env)")

    decision_text = state.get("final_trade_decision") or state.get(
        "trader_investment_plan"
    )
    if not decision_text:
        return PersistenceResult(False, None, "no decision text in state")

    stock = session.query(Stock).filter_by(symbol=ticker.upper()).first()
    if stock is None:
        logger.warning(
            "TradingAgents decision skipped: ticker %s not in IAP `stocks` table",
            ticker,
        )
        return PersistenceResult(False, None, f"ticker {ticker} not in IAP universe")

    # Cheap action heuristic — parse only the first 200 chars to avoid LLM cost.
    head = decision_text[:200].upper()
    if "STRONG BUY" in head:
        action = "strong_buy"
    elif "BUY" in head and "DO NOT BUY" not in head:
        action = "buy"
    elif "STRONG SELL" in head:
        action = "strong_sell"
    elif "SELL" in head:
        action = "sell"
    else:
        action = "hold"

    factors: dict[str, Any] = {
        "source": "tradingagents",
        "trade_date": trade_date,
        "has_market_report": bool(state.get("market_report")),
        "has_news_report": bool(state.get("news_report")),
        "has_sentiment_report": bool(state.get("sentiment_report")),
        "has_fundamentals_report": bool(state.get("fundamentals_report")),
    }
    if extra_factors:
        factors.update(extra_factors)

    risk_state = state.get("risk_debate_state") or {}
    rec = Recommendation(
        stock_id=stock.id,
        action=action,
        confidence=float(confidence),
        priority=7 if action in {"strong_buy", "strong_sell"} else 5,
        reasoning=decision_text,
        key_factors=factors,
        risks=(state.get("investment_debate_state") or {}).get("bear_history"),
        opportunities=(state.get("investment_debate_state") or {}).get("bull_history"),
        catalysts=risk_state.get("history"),
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    logger.info(
        "Persisted TradingAgents recommendation id=%s action=%s ticker=%s",
        rec.id,
        action,
        ticker,
    )
    return PersistenceResult(True, rec.id, "ok")
