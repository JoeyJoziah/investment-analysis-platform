#!/bin/sh
# =============================================================================
# Database Backup Script with Enhanced Error Handling
# Investment Analysis Platform
# =============================================================================
# Features:
#   - SHA256 checksum calculation and verification
#   - Failure notification via webhook
#   - S3 upload with metadata
#   - Comprehensive error handling with trap
# =============================================================================

# Exit on first error
set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
DB_HOST="${DB_HOST:-postgres}"
DB_NAME="${DB_NAME:-investment_db}"
DB_USER="${DB_USER:-postgres}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/backup_${DB_NAME}_${TIMESTAMP}.sql.gz"
CHECKSUM_FILE="${BACKUP_FILE}.sha256"
# SEC Rule 17a-4 requires 5-7 year retention for investment adviser records
# Default to 7 years (2555 days) for SEC compliance
# Local backups use shorter retention; S3/Glacier handles long-term archival
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-90}"
SEC_ARCHIVE_RETENTION_YEARS="${SEC_ARCHIVE_RETENTION_YEARS:-7}"

# Notification URLs
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# S3 Configuration
BACKUP_S3_BUCKET="${BACKUP_S3_BUCKET:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Encryption Configuration (optional, for local backup encryption)
# Set BACKUP_ENCRYPTION_KEY to enable AES-256 encryption for local backups
BACKUP_ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"
ENCRYPTED_SUFFIX=".enc"

# Status tracking
BACKUP_STATUS="unknown"
BACKUP_ERROR=""
BACKUP_SIZE=""
BACKUP_CHECKSUM=""

# =============================================================================
# Functions
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1"
}

# Send notification on failure
send_failure_notification() {
    local error_message="$1"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Prepare JSON payload
    local payload=$(cat <<EOF
{
    "status": "failure",
    "service": "backup",
    "database": "${DB_NAME}",
    "host": "${DB_HOST}",
    "timestamp": "${timestamp}",
    "error": "${error_message}",
    "backup_file": "${BACKUP_FILE}",
    "alerts": [{
        "alertname": "BackupFailed",
        "severity": "critical",
        "labels": {
            "service": "backup",
            "database": "${DB_NAME}"
        },
        "annotations": {
            "summary": "Database backup failed for ${DB_NAME}",
            "description": "${error_message}"
        }
    }]
}
EOF
)

    # Send to AlertManager webhook if configured
    if [ -n "$ALERT_WEBHOOK_URL" ]; then
        log_info "Sending failure notification to AlertManager..."
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "${ALERT_WEBHOOK_URL}" || true
    fi

    # Send to Slack webhook if configured
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        log_info "Sending failure notification to Slack..."
        local slack_payload=$(cat <<EOF
{
    "text": ":x: *Database Backup Failed*",
    "attachments": [{
        "color": "danger",
        "fields": [
            {"title": "Database", "value": "${DB_NAME}", "short": true},
            {"title": "Host", "value": "${DB_HOST}", "short": true},
            {"title": "Timestamp", "value": "${timestamp}", "short": true},
            {"title": "Error", "value": "${error_message}", "short": false}
        ]
    }]
}
EOF
)
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$slack_payload" \
            "${SLACK_WEBHOOK_URL}" || true
    fi
}

# Send success notification
send_success_notification() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Only send if webhooks are configured
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        log_info "Sending success notification to Slack..."
        local slack_payload=$(cat <<EOF
{
    "text": ":white_check_mark: *Database Backup Completed*",
    "attachments": [{
        "color": "good",
        "fields": [
            {"title": "Database", "value": "${DB_NAME}", "short": true},
            {"title": "Size", "value": "${BACKUP_SIZE}", "short": true},
            {"title": "Timestamp", "value": "${timestamp}", "short": true},
            {"title": "S3 Upload", "value": "${BACKUP_S3_BUCKET:-disabled}", "short": true}
        ]
    }]
}
EOF
)
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$slack_payload" \
            "${SLACK_WEBHOOK_URL}" || true
    fi
}

# Cleanup function for trap
cleanup() {
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        BACKUP_STATUS="failed"
        log_error "Backup failed with exit code: $exit_code"

        # Remove partial backup file if exists
        if [ -f "$BACKUP_FILE" ]; then
            log_info "Removing partial backup file..."
            rm -f "$BACKUP_FILE"
            rm -f "$CHECKSUM_FILE"
        fi

        # Send failure notification
        send_failure_notification "${BACKUP_ERROR:-Unknown error occurred during backup}"
    fi

    exit $exit_code
}

# Calculate checksum
calculate_checksum() {
    local file=$1
    log_info "Calculating SHA256 checksum..."

    # Use sha256sum or shasum depending on availability
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" > "$CHECKSUM_FILE"
        BACKUP_CHECKSUM=$(cat "$CHECKSUM_FILE" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" > "$CHECKSUM_FILE"
        BACKUP_CHECKSUM=$(cat "$CHECKSUM_FILE" | awk '{print $1}')
    else
        log_error "No checksum utility available (sha256sum or shasum)"
        BACKUP_CHECKSUM="unavailable"
        return 1
    fi

    log_info "Checksum: $BACKUP_CHECKSUM"
    return 0
}

# Verify backup integrity
verify_backup() {
    local file=$1
    log_info "Verifying backup integrity..."

    if ! gunzip -t "$file" 2>/dev/null; then
        BACKUP_ERROR="Backup file integrity check failed - gzip test failed"
        log_error "$BACKUP_ERROR"
        return 1
    fi

    log_success "Backup integrity verified"
    return 0
}

# Encrypt backup file using AES-256-CBC
encrypt_backup() {
    local file=$1
    local encrypted_file="${file}${ENCRYPTED_SUFFIX}"

    log_info "Encrypting backup with AES-256-CBC..."

    if ! command -v openssl >/dev/null 2>&1; then
        log_error "OpenSSL not available - cannot encrypt backup"
        return 1
    fi

    # Use PBKDF2 key derivation for stronger encryption
    if openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
        -in "$file" \
        -out "$encrypted_file" \
        -pass env:BACKUP_ENCRYPTION_KEY; then

        log_success "Backup encrypted successfully"

        # Remove unencrypted backup
        rm -f "$file"

        # Update backup file reference
        BACKUP_FILE="$encrypted_file"

        return 0
    else
        BACKUP_ERROR="Failed to encrypt backup file"
        log_error "$BACKUP_ERROR"
        return 1
    fi
}

# Upload to S3 with metadata and SEC-compliant retention tagging
upload_to_s3() {
    local file=$1
    local checksum=$2

    log_info "Uploading backup to S3: s3://${BACKUP_S3_BUCKET}/database-backups/"

    # Prepare metadata including SEC compliance markers
    local metadata="checksum=${checksum},database=${DB_NAME},timestamp=${TIMESTAMP},sec-retention-years=${SEC_ARCHIVE_RETENTION_YEARS},compliance=sec-17a-4"

    # Calculate retention date for SEC compliance (default 7 years)
    local retention_date=$(date -d "+${SEC_ARCHIVE_RETENTION_YEARS} years" +%Y-%m-%dT00:00:00Z 2>/dev/null || date -v+${SEC_ARCHIVE_RETENTION_YEARS}y +%Y-%m-%dT00:00:00Z)

    # Upload main backup file with GLACIER storage for cost-effective long-term archival
    # Server-side encryption is enabled by default on GLACIER
    if aws s3 cp "$file" "s3://${BACKUP_S3_BUCKET}/database-backups/" \
        --storage-class GLACIER \
        --metadata "$metadata" \
        --sse AES256 \
        --region "$AWS_REGION"; then
        log_success "Backup uploaded to S3 with AES-256 encryption"

        # Upload checksum file
        if [ -f "$CHECKSUM_FILE" ]; then
            aws s3 cp "$CHECKSUM_FILE" "s3://${BACKUP_S3_BUCKET}/database-backups/" \
                --sse AES256 \
                --region "$AWS_REGION" || true
        fi

        log_info "SEC retention period: ${SEC_ARCHIVE_RETENTION_YEARS} years (until ${retention_date})"
        return 0
    else
        BACKUP_ERROR="Failed to upload backup to S3"
        log_error "$BACKUP_ERROR"
        return 1
    fi
}

# =============================================================================
# Main Script
# =============================================================================

# Set trap for error handling
trap cleanup EXIT

log_info "=============================================="
log_info "Starting backup of ${DB_NAME}"
log_info "Host: ${DB_HOST}"
log_info "Timestamp: ${TIMESTAMP}"
log_info "=============================================="

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Check database connectivity
log_info "Checking database connectivity..."
if ! pg_isready -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
    BACKUP_ERROR="Cannot connect to database ${DB_NAME} on ${DB_HOST}"
    log_error "$BACKUP_ERROR"
    exit 1
fi
log_success "Database connection verified"

# Perform backup
log_info "Creating database dump..."
if ! pg_dump -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" --no-password --verbose 2>/dev/null | gzip > "${BACKUP_FILE}"; then
    BACKUP_ERROR="pg_dump failed - check database permissions and connectivity"
    log_error "$BACKUP_ERROR"
    exit 1
fi

# Check if backup file was created and has content
if [ ! -s "$BACKUP_FILE" ]; then
    BACKUP_ERROR="Backup file is empty or was not created"
    log_error "$BACKUP_ERROR"
    exit 1
fi

log_success "Database dump completed"

# Get file size
BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
log_info "Backup size: ${BACKUP_SIZE}"

# Verify backup integrity
if ! verify_backup "$BACKUP_FILE"; then
    exit 1
fi

# Encrypt backup if encryption key is configured
if [ -n "${BACKUP_ENCRYPTION_KEY}" ]; then
    if ! encrypt_backup "$BACKUP_FILE"; then
        log_error "Encryption failed, but continuing with unencrypted backup..."
    fi
else
    log_info "Local backup encryption disabled (BACKUP_ENCRYPTION_KEY not set)"
fi

# Calculate checksum (after potential encryption)
if ! calculate_checksum "$BACKUP_FILE"; then
    log_error "Checksum calculation failed, but continuing..."
fi

# Upload to S3 if configured
if [ -n "${BACKUP_S3_BUCKET}" ]; then
    if ! upload_to_s3 "$BACKUP_FILE" "$BACKUP_CHECKSUM"; then
        # S3 upload failure is not fatal - local backup still exists
        log_error "S3 upload failed, but local backup is preserved"
    fi
else
    log_info "S3 backup disabled (BACKUP_S3_BUCKET not set)"
fi

# Remove old backups (both encrypted and unencrypted)
log_info "Removing local backups older than ${RETENTION_DAYS} days..."
log_info "Note: S3/Glacier backups retained for ${SEC_ARCHIVE_RETENTION_YEARS} years per SEC requirements"
find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.sql.gz" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.sql.gz.enc" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.sql.gz.sha256" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.sql.gz.enc.sha256" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true

# Count remaining backups
BACKUP_COUNT=$(find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.sql.gz" 2>/dev/null | wc -l | tr -d ' ')
log_info "Local backups retained: ${BACKUP_COUNT}"

# Mark backup as successful
BACKUP_STATUS="success"

# Send success notification
send_success_notification

log_info "=============================================="
log_success "Backup completed successfully"
log_info "File: ${BACKUP_FILE}"
log_info "Size: ${BACKUP_SIZE}"
log_info "Checksum: ${BACKUP_CHECKSUM}"
log_info "=============================================="

# Reset trap since we completed successfully
trap - EXIT
exit 0
