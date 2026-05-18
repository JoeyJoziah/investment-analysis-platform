"""
Regression tests for _run_sentiment_analysis.

F-09-001 (audit 2026-04, G2a sub-theme E step 32):
``recommendation_engine._run_sentiment_analysis`` called
``self.sentiment_engine.analyze_sentiment(ticker, text_data)`` but
``analyze_sentiment`` is the per-text entrypoint with signature
``(text: str, source: str = "unknown")`` — it rejects the
list-of-dicts shape this method assembles, raising AttributeError /
TypeError on every call.

The fix delegates to ``analyze_stock_sentiment(ticker, texts: List[str])``
and adapts the returned ``SentimentResult`` back to the dict shape
downstream consumers read via ``sentiment_analysis.get('overall_sentiment',
...)``.

Tests inspect source rather than instantiating the engine (which pulls
in heavy ML deps).
"""

from __future__ import annotations

import re
from pathlib import Path


_ENGINE = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "recommendation_engine.py"
)


def test_uses_analyze_stock_sentiment_not_analyze_sentiment() -> None:
    """F-09-001: must call analyze_stock_sentiment with extracted text list."""

    text = _ENGINE.read_text()
    # The legacy buggy call.
    assert "self.sentiment_engine.analyze_sentiment(ticker, text_data)" not in text, (
        "_run_sentiment_analysis still calls analyze_sentiment with "
        "the list-of-dicts shape; that method takes (text: str, source: str)"
    )
    # The corrected call.
    assert re.search(
        r"analyze_stock_sentiment\(\s*ticker\s*,\s*\[",
        text,
    ), "_run_sentiment_analysis must call analyze_stock_sentiment(ticker, [...])"


def test_returns_overall_sentiment_dict_shape() -> None:
    """F-09-001: downstream consumers expect ``overall_sentiment`` key."""

    text = _ENGINE.read_text()
    # The dict-shape adapter must still emit 'overall_sentiment' so the
    # downstream get('overall_sentiment', {}).get('score', 0) keeps
    # working.
    assert "'overall_sentiment'" in text and "'score'" in text
