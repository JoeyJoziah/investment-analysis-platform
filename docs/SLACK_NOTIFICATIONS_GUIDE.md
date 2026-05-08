# Slack Notifications Guide

Last updated: 2026-05-08
Status: scaffolding present in alertmanager + backup scripts, awaiting Slack webhook URL to enable.

## TL;DR

```bash
# 1. Create a Slack incoming webhook: https://api.slack.com/messaging/webhooks
# 2. Set the URL in your environment
echo "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../..." >> .env
echo "SLACK_API_URL=https://hooks.slack.com/services/T.../B.../..." >> .env
# 3. Restart the affected services
docker compose -f docker-compose.production.yml restart alertmanager backup backend
```

## Where Slack hooks live

| Surface | Path | Env var | Trigger |
|---|---|---|---|
| Alertmanager | `config/infrastructure/monitoring/alertmanager/alertmanager.yml` | `SLACK_API_URL` | Prometheus alert fires |
| Backup script | `scripts/backup.sh` | `SLACK_WEBHOOK_URL` | Backup or upload fails |
| App-level | `backend/utils/slack_notifier.py` | `SLACK_WEBHOOK_URL` | Manual `notify_slack(...)` call from app code |

## Alert receivers

`alertmanager.yml` defines five receivers, all Slack-aware:
- `default-receiver` — informational alerts
- `critical-alerts` — paging events (channel: #ops-critical)
- `budget-alerts` — cost/quota events (channel: #ops-budget)
- `security-alerts` — security events (channel: #ops-security)
- `performance-alerts` — latency/error-rate events (channel: #ops-perf)
- `data-quality-alerts` — data freshness/validation events (channel: #ops-data)

To customize channels, edit `slack_configs[].channel` per receiver.

## App-level usage

```python
from backend.utils.slack_notifier import notify_slack

# Fire-and-forget
notify_slack("Daily ETL completed: 6,012 stocks updated", level="info")
notify_slack("Cost monitor: Anthropic API hit 90% of monthly budget", level="warn")
notify_slack("CRITICAL: WebSocket bus stopped accepting connections", level="critical")
```

## Testing without firing real alerts

```bash
# Trigger a test webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"smoke test from `'$(hostname)'`"}' \
  "$SLACK_WEBHOOK_URL"
```

## Troubleshooting

- **No messages in Slack** — check `docker compose logs alertmanager` for `slack_api_url` errors. Empty string is the default and causes silent failure.
- **Rate limited** — Slack incoming webhooks allow ~1/sec per hook. Use channel-specific hooks if any receiver is high-volume.
- **Wrong channel** — webhook URLs are channel-bound at creation. To change, regenerate the hook in the target channel.

## See also
- `RUNBOOK.md` — escalation policy
- Alertmanager docs: https://prometheus.io/docs/alerting/latest/configuration/#slack_config