"""
Agents Service
Business logic for LLM agent analysis operations.
Handles engine singleton management, analysis orchestration, and metrics logging.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.utils.numpy_serializer import sanitize_numpy

logger = logging.getLogger(__name__)


# =======================
# Analysis Engine Singletons
# =======================

_fundamental_engine: Optional[FundamentalAnalysisEngine] = None
_technical_engine: Optional[TechnicalAnalysisEngine] = None
_sentiment_engine: Optional[SentimentAnalysisEngine] = None


def _get_fundamental_engine() -> FundamentalAnalysisEngine:
    global _fundamental_engine
    if _fundamental_engine is None:
        _fundamental_engine = FundamentalAnalysisEngine()
    return _fundamental_engine


def _get_technical_engine() -> TechnicalAnalysisEngine:
    global _technical_engine
    if _technical_engine is None:
        _technical_engine = TechnicalAnalysisEngine()
    return _technical_engine


def _get_sentiment_engine() -> SentimentAnalysisEngine:
    global _sentiment_engine
    if _sentiment_engine is None:
        _sentiment_engine = SentimentAnalysisEngine()
    return _sentiment_engine


# =======================
# Per-Engine Analysis Runners
# =======================

async def run_fundamental_analysis(ticker: str, depth: str) -> Dict[str, Any]:
    """
    Execute fundamental analysis and normalise the output into a standard dict.

    Returns a dict with keys: score, summary, details.
    """
    engine = _get_fundamental_engine()
    financials: Dict[str, Any] = {"ticker": ticker}
    market_data: Dict[str, Any] = {"ticker": ticker}

    analysis = await engine.analyze_company(
        ticker=ticker,
        financials=financials,
        market_data=market_data,
        peer_data=None,
    )

    analysis = sanitize_numpy(analysis)

    composite = analysis.get("composite_score", 0.0)
    risks = analysis.get("risks", [])
    opportunities = analysis.get("opportunities", [])

    risk_texts = [
        r.get("description", str(r)) if isinstance(r, dict) else str(r)
        for r in risks
    ]
    opp_texts = [
        o.get("description", str(o)) if isinstance(o, dict) else str(o)
        for o in opportunities
    ]

    parts = []
    if risk_texts:
        parts.append(f"Risks: {'; '.join(risk_texts[:3])}")
    if opp_texts:
        parts.append(f"Opportunities: {'; '.join(opp_texts[:3])}")
    summary = (
        f"Fundamental score {composite:.1f}/100. " + " ".join(parts)
        if parts
        else f"Fundamental score {composite:.1f}/100."
    )

    details = (
        analysis
        if depth == "deep"
        else {
            "composite_score": composite,
            "risks": risks[:3],
            "opportunities": opportunities[:3],
        }
    )

    return {"score": composite, "summary": summary, "details": details}


async def _load_ohlcv_frame(ticker: str, *, limit: int = 250):
    """Load real OHLCV for technical analysis (oldest→newest). None if missing."""
    import pandas as pd

    try:
        from backend.repositories.price_repository import price_repository

        records = await price_repository.get_price_history(ticker, limit=limit)
    except Exception as exc:
        logger.warning("price history load failed for %s: %s", ticker, exc)
        return None

    if not records or len(records) < 30:
        return None

    # Repository returns newest-first; technical engine expects chronological.
    rows = list(reversed(records))

    def _num(value: object) -> float:
        # PriceHistory columns are Decimal/int at runtime; cast for analysis/pyright.
        return float(value)  # type: ignore[arg-type]

    return pd.DataFrame(
        {
            "open": [_num(r.open) for r in rows],
            "high": [_num(r.high) for r in rows],
            "low": [_num(r.low) for r in rows],
            "close": [_num(r.close) for r in rows],
            "volume": [_num(r.volume) for r in rows],
        }
    )


def _synthetic_ohlcv_demo_only(ticker: str, n: int = 250):
    """Synthetic OHLCV — DEMO_MODE only. Never used in production paths."""
    import numpy as np
    import pandas as pd

    np.random.seed(hash(ticker) % (2**31))
    base = 150.0
    returns = np.random.normal(0.0005, 0.02, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 50_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


async def run_technical_analysis(ticker: str, depth: str) -> Dict[str, Any]:
    """
    Execute technical analysis and normalise the output into a standard dict.

    Uses real price history from the repository. Production refuses fabricated
    OHLCV (ModelUnavailableError). DEMO_MODE may fall back to tagged synthetic
    bars for demos only.

    Returns a dict with keys: score, summary, details.
    """
    from backend.config.settings import settings
    from backend.exceptions import ModelUnavailableError

    engine = _get_technical_engine()

    df = await _load_ohlcv_frame(ticker)
    data_source = "price_history"
    if df is None:
        if settings.DEMO_MODE:
            df = _synthetic_ohlcv_demo_only(ticker)
            data_source = "simulated"
            logger.warning(
                "run_technical_analysis: DEMO_MODE synthetic OHLCV for %s", ticker
            )
        else:
            raise ModelUnavailableError(
                model="technical_analysis",
                reason="insufficient_price_history",
            )

    analysis = engine.analyze_stock(df)
    analysis = sanitize_numpy(analysis)

    composite = analysis.get("composite_score", 0.0)
    signals = analysis.get("signals", [])
    trend = analysis.get("market_structure", {}).get("trend", "unknown")

    signal_strs = [s.get("name", "") for s in signals[:3]]
    summary = f"Technical score {composite:+.2f} (trend: {trend})."
    if signal_strs:
        summary += f" Signals: {', '.join(signal_strs)}."

    details = (
        analysis
        if depth == "deep"
        else {
            "composite_score": composite,
            "trend": trend,
            "signals": signals[:5],
            "data_source": data_source,
        }
    )
    if depth == "deep" and isinstance(details, dict):
        details = {**details, "data_source": data_source}

    return {"score": composite, "summary": summary, "details": details}


async def run_sentiment_analysis(ticker: str, depth: str) -> Dict[str, Any]:
    """
    Execute sentiment analysis and normalise the output into a standard dict.

    Returns a dict with keys: score, summary, details.
    """
    engine = _get_sentiment_engine()

    analysis = await engine.analyze_comprehensive_sentiment(ticker)
    analysis = sanitize_numpy(analysis)

    overall = analysis.get("overall_sentiment", {})
    score = overall.get("score", 0.0)
    label = overall.get("label", "neutral")
    confidence = overall.get("confidence", 0.0)

    summary = f"Sentiment is {label} (score {score:+.2f}, confidence {confidence:.0%})."

    details = (
        analysis
        if depth == "deep"
        else {
            "score": score,
            "label": label,
            "confidence": confidence,
            "sources_analyzed": analysis.get("sources_analyzed", 0),
        }
    )

    return {"score": score, "summary": summary, "details": details}


# Dispatch map keyed by analysis type name
ANALYSIS_RUNNERS: Dict[str, Any] = {
    "fundamental": run_fundamental_analysis,
    "technical": run_technical_analysis,
    "sentiment": run_sentiment_analysis,
}


# =======================
# Analysis Orchestration
# =======================

async def run_multi_engine_analysis(
    ticker: str,
    analysis_types: List[str],
    depth: str,
) -> Dict[str, Any]:
    """
    Orchestrate concurrent analysis across multiple engines.

    Args:
        ticker: Uppercase stock ticker symbol.
        analysis_types: List of engine names to run (fundamental/technical/sentiment).
        depth: Analysis depth - "standard" or "deep".

    Returns:
        Dict with keys:
          - results: {analysis_type: AnalysisTypeResult-compatible dict}
          - confidence_score: float in [0, 1]
          - analysis_id: str
          - timestamp: ISO-8601 UTC string
          - duration: float (seconds)

    Raises:
        ValueError: If an unsupported analysis type is requested.
        RuntimeError: If all analysis engines fail.
    """
    # Validate requested types up-front
    for atype in analysis_types:
        if atype not in ANALYSIS_RUNNERS:
            raise ValueError(f"Unsupported analysis type: {atype}")

    start_time = time.monotonic()

    tasks: Dict[str, asyncio.Task] = {
        atype: asyncio.create_task(ANALYSIS_RUNNERS[atype](ticker, depth))
        for atype in analysis_types
    }

    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    for atype, task in tasks.items():
        try:
            results[atype] = await task
        except Exception as exc:
            logger.error(
                "Analysis engine %s failed for %s: %s", atype, ticker, exc
            )
            errors[atype] = str(exc)

    if not results:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(f"All analysis engines failed: {detail}")

    # Aggregate confidence score (normalise each engine to [0, 1])
    confidence_values: List[float] = []
    for atype, result in results.items():
        score = result["score"]
        if atype == "fundamental":
            confidence_values.append(min(max(score / 100.0, 0.0), 1.0))
        else:  # technical and sentiment both produce [-1, 1]
            confidence_values.append(min(max((score + 1.0) / 2.0, 0.0), 1.0))

    confidence_score = (
        sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    )

    analysis_id = f"agt-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    duration = time.monotonic() - start_time

    return {
        "results": results,
        "confidence_score": round(confidence_score, 4),
        "analysis_id": analysis_id,
        "timestamp": timestamp,
        "duration": duration,
    }


# =======================
# Metrics Logging Helpers
# =======================

async def log_agent_analysis_metrics(
    analysis_id: str,
    ticker: str,
    analysis_types: List[str],
    depth: str,
    confidence_score: float,
    duration: float,
) -> None:
    """Background task: log multi-engine analysis metrics for monitoring."""
    try:
        logger.info(
            "Agent analysis metrics - ID: %s, Ticker: %s, Types: %s, "
            "Depth: %s, Confidence: %.4f, Duration: %.2fs",
            analysis_id,
            ticker,
            analysis_types,
            depth,
            confidence_score,
            duration,
        )
    except Exception as exc:
        logger.error("Failed to log agent analysis metrics: %s", exc)


async def log_analysis_metrics(
    ticker: str,
    complexity_level: str,
    cost: float,
    duration: float,
    agent_count: int,
) -> None:
    """Background task: log single-stock analysis metrics for monitoring."""
    try:
        logger.info(
            "Analysis metrics - Ticker: %s, Complexity: %s, "
            "Cost: $%.4f, Duration: %.1fs, Agents: %d",
            ticker,
            complexity_level,
            cost,
            duration,
            agent_count,
        )
    except Exception as exc:
        logger.error("Failed to log analysis metrics: %s", exc)


async def log_batch_analysis_metrics(
    requested: int,
    completed: int,
    total_cost: float,
    total_duration: float,
    agents_used_count: int,
) -> None:
    """Background task: log batch analysis metrics for monitoring."""
    try:
        logger.info(
            "Batch analysis metrics - Requested: %d, Completed: %d, "
            "Total cost: $%.4f, Total duration: %.1fs, Agent analyses: %d",
            requested,
            completed,
            total_cost,
            total_duration,
            agents_used_count,
        )
    except Exception as exc:
        logger.error("Failed to log batch analysis metrics: %s", exc)
