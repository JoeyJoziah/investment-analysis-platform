"""
Regression tests for SEC EDGAR contact email enforcement.

F-05-006 (audit 2026-04, G2a sub-theme C step 21):
``backend/data_ingestion/sec_edgar_client.py`` shipped a
``contact@example.com`` placeholder in its User-Agent header. SEC
throttles or bans clients that don't supply real contact info, so this
silently degraded fundamentals fetching.
"""

from __future__ import annotations

import re
from pathlib import Path


_CLIENT = (
    Path(__file__).resolve().parents[2]
    / "data_ingestion"
    / "sec_edgar_client.py"
)


def test_no_placeholder_email_in_user_agent() -> None:
    """F-05-006: hardcoded ``contact@example.com`` must be gone."""

    text = _CLIENT.read_text()
    # The literal string survives in the validation message (used to
    # reject the env value), but not as a User-Agent default. The
    # disqualifying pattern is the User-Agent f-string with example.com.
    assert "(contact@example.com)" not in text, (
        "sec_edgar_client.py still embeds the placeholder contact email "
        "directly in the User-Agent header"
    )


def test_reads_contact_email_from_env() -> None:
    """F-05-006: must source contact email from ``SEC_EDGAR_CONTACT_EMAIL``."""

    text = _CLIENT.read_text()
    assert "SEC_EDGAR_CONTACT_EMAIL" in text, (
        "sec_edgar_client.py must read SEC_EDGAR_CONTACT_EMAIL env var"
    )
    assert re.search(
        r"os\.getenv\(\s*[\"']SEC_EDGAR_CONTACT_EMAIL", text
    ), "must use os.getenv to read SEC_EDGAR_CONTACT_EMAIL"


def test_fails_loudly_on_missing_email() -> None:
    """F-05-006: missing/placeholder env var must raise at construction."""

    text = _CLIENT.read_text()
    assert "raise RuntimeError" in text or "raise ValueError" in text, (
        "sec_edgar_client.py must raise when SEC_EDGAR_CONTACT_EMAIL is "
        "missing or set to a placeholder — silent degradation would let "
        "us run unidentified against sec.gov"
    )
