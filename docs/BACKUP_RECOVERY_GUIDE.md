# Backup & Recovery Guide

Version: 1.0.0
Last Updated: 2026-05-08
Status: scripts in place, S3 bucket + IAM credentials still required to enable cloud archival.

## TL;DR

```bash
# One-off backup (writes locally; uploads to S3 if BACKUP_S3_BUCKET is set)
BACKUP_DIR=/backups BACKUP_S3_BUCKET=my-bucket ./scripts/backup.sh

# Verify the latest backup
./scripts/verify-backup.sh /backups

# Restore (interactive)
./scripts/restore-backup.sh /backups/backup_investment_db_20260508_120000.sql.gz
```

## Existing scripts

| Script | Path | Purpose |
|---|---|---|
| Backup | `scripts/backup.sh` | pg_dump + gzip + sha256 + S3 upload + webhook on failure |
| Restore | `scripts/restore-backup.sh` | gunzip + psql restore with confirmation prompts |
| Verify | `scripts/verify-backup.sh` | Validate sha256, structure, restore-test in scratch DB |

## Required env vars

### Local backup (always)
- `BACKUP_DIR` — local dir for SQL dumps (default `/backups`)
- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` — postgres connection
- `BACKUP_RETENTION_DAYS` — local retention (default 90)

### S3 archival (optional)
- `BACKUP_S3_BUCKET` — bucket name (e.g. `my-investment-backups`)
- `AWS_REGION` — region (default `us-east-1`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — IAM credentials with `s3:PutObject` + `s3:PutObjectTagging`
- `SEC_ARCHIVE_RETENTION_YEARS` — for SEC 17a-4 compliance (default 7)

### Encryption (optional)
- `BACKUP_ENCRYPTION_KEY` — 32-char key enables AES-256 encryption of local backup

### Notifications (optional)
- `ALERT_WEBHOOK_URL` — generic webhook for backup failure alerts
- `SLACK_WEBHOOK_URL` — Slack-formatted failure alerts

## Setting up S3 archival (one-time)

```bash
# 1. Create bucket
aws s3 mb s3://my-investment-backups --region us-east-1

# 2. Enable versioning + lifecycle policy for SEC retention (7 years)
aws s3api put-bucket-versioning --bucket my-investment-backups \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-lifecycle-configuration --bucket my-investment-backups \
  --lifecycle-configuration file://infrastructure/aws/s3-lifecycle.json

# 3. Create IAM user with restricted policy (see infrastructure/aws/backup-policy.json)
aws iam create-user --user-name investment-backup
aws iam put-user-policy --user-name investment-backup \
  --policy-name S3BackupAccess --policy-document file://infrastructure/aws/backup-policy.json

# 4. Set credentials in .env or container env
echo "BACKUP_S3_BUCKET=my-investment-backups" >> .env
echo "AWS_ACCESS_KEY_ID=..." >> .env
echo "AWS_SECRET_ACCESS_KEY=..." >> .env
```

## Scheduling

Production compose includes a cron-driven backup container. Default: daily at 02:00 UTC.
Override via `BACKUP_CRON_SCHEDULE` in `.env` (e.g. `0 */6 * * *` for every 6 hours).

```bash
docker compose -f docker-compose.production.yml logs backup --tail 100
```

## Disaster recovery drill

Run quarterly:
```bash
# 1. Pick a backup at random from S3
aws s3 ls s3://my-investment-backups/ | shuf -n 1
# 2. Download and verify
aws s3 cp s3://my-investment-backups/backup_xxx.sql.gz /tmp/
./scripts/verify-backup.sh /tmp/
# 3. Restore to scratch DB and validate row counts vs prod
./scripts/restore-backup.sh /tmp/backup_xxx.sql.gz --target=scratch_db
```

## See also
- `RUNBOOK.md` — incident response
- `SECURITY.md` — encryption + key rotation
- `SEC_REGULATORY_COMPLIANCE_AUDIT.md` — 17a-4 retention requirements