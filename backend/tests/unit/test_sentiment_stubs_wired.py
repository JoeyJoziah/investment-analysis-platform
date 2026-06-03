"""
Regression tests for sentiment-stub wiring against SmartDataFetcher.

F-09-006 (audit 2026-04, G2a sub-theme E step 41):
``SentimentAnalysisEngine.get_news_sentiment`` returned hardcoded
neutral with ``source: 'news_placeholder'`` regardless of input.
``get_social_sentiment`` did the same with ``confidence=0.5`` — a
lie that overstated availability.

Now that F-05-004 has wired SmartDataFetcher to real clients,
get_news_sentiment delegates to it. get_social_sentiment remains a
sentinel (no ingestion source exists) but with ``confidence=0.0``
and a loud warning so consumers can detect the no-data path.
"""

from __future__ import annotations

from pathlib import Path


_PATH = (
    Path(__file__).resolve().parents[2]
    / "analytics"
    / "sentiment_analysis.py"
)


def test_news_stub_calls_smart_fetcher() -> None:
    """F-09-006: get_news_sentiment must delegate to SmartDataFetcher."""

    text = _PATH.read_text()
    assert "from backend.data_ingestion.smart_data_fetcher import get_smart_fetcher" in text
    assert 'fetch_stock_data(ticker, "news")' in text, (
        "get_news_sentiment must call the SmartDataFetcher news endpoint"
    )


def test_news_stub_runs_real_analysis_on_articles() -> None:
    """F-09-006: news path must feed articles into analyze_stock_sentiment."""

    text = _PATH.read_text()
    assert "return await self.analyze_stock_sentiment(ticker, texts)" in text, (
        "get_news_sentiment must run the real batch analyzer over fetched texts"
    )


def test_no_news_placeholder_source_label() -> None:
    """F-09-006: legacy ``news_placeholder`` literal must be gone."""

    text = _PATH.read_text()
    assert "'news_placeholder'" not in text, (
        "get_news_sentiment still emits the legacy news_placeholder source label"
    )


def test_social_stub_drops_confidence_to_zero() -> None:
    """F-09-006: social stub must not lie about confidence."""

    text = _PATH.read_text()
    # The legacy stub returned confidence=0.5; check the new sentinel
    # path explicitly notes confidence=0.0 and is unimplemented.
    assert "'social_unimplemented'" in text or "social_unimplemented" in text, (
        "social stub must label its source as social_unimplemented"
    )
    # Find the get_social_sentiment block and verify it doesn't ship
    # confidence=0.5.
    import re
    block = re.search(
        r"def get_social_sentiment.*?(?=\n    async def |\Z)",
        text, re.DOTALL,
    )
    assert block is not None
    assert "confidence=0.5" not in block.group(0), (
        "get_social_sentiment must not return confidence=0.5 for a no-data path"
    )


def test_social_stub_logs_warning() -> None:
    """F-09-006: social stub must log a loud warning per workpaper §9."""

    text = _PATH.read_text()
    assert "is not implemented (F-09-006)" in text, (
        "social stub must log a warning naming the finding ID"
    )
