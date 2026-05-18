"""
Slack notifier — minimal, dependency-free webhook helper.

Usage:
    from backend.utils.slack_notifier import notify_slack
    notify_slack("ETL job complete", level="info")

Env vars:
    SLACK_WEBHOOK_URL — incoming webhook URL (required; no-op if unset)
    SLACK_DEFAULT_CHANNEL — optional channel override (e.g. "#ops")
    SLACK_NOTIFIER_TIMEOUT — request timeout seconds (default 5)

Levels are mapped to colors used by Slack attachments.
The function is fire-and-forget; failures are logged but never raise to callers.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_LEVEL_COLOR = {
    "info": "#2eb886",
    "warn": "#daa038",
    "warning": "#daa038",
    "error": "#e01e5a",
    "critical": "#9b0606",
    "debug": "#cccccc",
}


def notify_slack(
    text: str,
    *,
    level: str = "info",
    channel: Optional[str] = None,
    title: Optional[str] = None,
    fallback: Optional[str] = None,
) -> bool:
    """
    Post a message to Slack via incoming webhook.

    Returns True on HTTP 2xx, False on any failure (including no webhook configured).
    Never raises.
    """
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        logger.debug("SLACK_WEBHOOK_URL not set — slack notification skipped")
        return False

    level_lc = level.lower()
    color = _LEVEL_COLOR.get(level_lc, _LEVEL_COLOR["info"])
    target_channel = channel or os.getenv("SLACK_DEFAULT_CHANNEL")
    timeout = float(os.getenv("SLACK_NOTIFIER_TIMEOUT", "5"))

    payload: dict = {
        "attachments": [
            {
                "color": color,
                "title": title or f"[{level_lc.upper()}]",
                "text": text,
                "fallback": fallback or text,
                "mrkdwn_in": ["text"],
            }
        ]
    }
    if target_channel:
        payload["channel"] = target_channel

    try:
        r = httpx.post(webhook, json=payload, timeout=timeout)
        if 200 <= r.status_code < 300:
            return True
        logger.warning("Slack webhook returned %s: %s", r.status_code, r.text[:200])
        return False
    except Exception as e:
        logger.warning("Slack webhook failed: %s: %s", type(e).__name__, e)
        return False


__all__ = ["notify_slack"]
