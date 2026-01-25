#!/bin/bash
# =============================================================================
# Backup Verification Script
# Investment Analysis Platform
# =============================================================================
# Usage: ./verify-backup.sh <backup_file> [--list-tables]
# Example: ./verify-backup.sh /backups/backup_investment_db_20240115_020000.sql.gz
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

usage() {
    echo "Usage: $0 <backup_file> [--list-tables]"
    echo ""
    echo "Arguments:"
    echo "  backup_file    Path to the gzipped backup file (.sql.gz)"
    echo "  --list-tables  Optional: List table counts from the backup"
    echo ""
    echo "Examples:"
    echo "  $0 /backups/backup_investment_db_20240115_020000.sql.gz"
    echo "  $0 /backups/backup_investment_db_20240115_020000.sql.gz --list-tables"
    exit 1
}

# =============================================================================
# Main Script
# =============================================================================

# Validate arguments
if [ $# -lt 1 ]; then
    log_error "Missing required argument: backup_file"
    usage
fi

BACKUP_FILE=$1
LIST_TABLES=false

if [ "$2" = "--list-tables" ]; then
    LIST_TABLES=true
fi

# Verify file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

log_info "Verifying backup: $BACKUP_FILE"
echo "=============================================="

# Get file information
FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
FILE_DATE=$(stat -f "%Sm" "$BACKUP_FILE" 2>/dev/null || stat -c "%y" "$BACKUP_FILE" 2>/dev/null)

echo "File size: $FILE_SIZE"
echo "Last modified: $FILE_DATE"
echo ""

# Verify gzip integrity
log_info "Verifying gzip integrity..."
if gunzip -t "$BACKUP_FILE" 2>/dev/null; then
    log_success "Gzip integrity check passed"
else
    log_error "Gzip integrity check FAILED - backup file may be corrupted"
    exit 1
fi

# Verify SQL structure (check for expected content)
log_info "Verifying SQL structure..."
HEADER=$(gunzip -c "$BACKUP_FILE" | head -50)

if echo "$HEADER" | grep -q "PostgreSQL database dump"; then
    log_success "PostgreSQL dump header verified"
else
    log_warn "Could not verify PostgreSQL dump header"
fi

# Check for expected tables
log_info "Checking for expected tables..."
EXPECTED_TABLES=("stocks" "stock_prices" "recommendations" "users" "portfolios" "watchlists" "alerts" "audit_logs")
FOUND_TABLES=0

for table in "${EXPECTED_TABLES[@]}"; do
    if gunzip -c "$BACKUP_FILE" | grep -q "CREATE TABLE.*$table\|COPY.*$table"; then
        log_success "Found table: $table"
        ((FOUND_TABLES++))
    else
        log_warn "Table not found: $table (may be empty or not exist)"
    fi
done

echo ""
log_info "Found $FOUND_TABLES of ${#EXPECTED_TABLES[@]} expected tables"

# Verify checksum if exists
CHECKSUM_FILE="${BACKUP_FILE}.sha256"
if [ -f "$CHECKSUM_FILE" ]; then
    log_info "Verifying SHA256 checksum..."
    EXPECTED_CHECKSUM=$(cat "$CHECKSUM_FILE" | awk '{print $1}')
    ACTUAL_CHECKSUM=$(sha256sum "$BACKUP_FILE" 2>/dev/null || shasum -a 256 "$BACKUP_FILE" | awk '{print $1}')

    if [ "$EXPECTED_CHECKSUM" = "$ACTUAL_CHECKSUM" ]; then
        log_success "Checksum verification passed"
    else
        log_error "Checksum verification FAILED"
        echo "Expected: $EXPECTED_CHECKSUM"
        echo "Actual:   $ACTUAL_CHECKSUM"
        exit 1
    fi
else
    log_warn "No checksum file found (${CHECKSUM_FILE})"
fi

# List tables with row counts if requested
if [ "$LIST_TABLES" = true ]; then
    echo ""
    log_info "Analyzing table data..."
    echo "=============================================="

    # Extract COPY statements which indicate data
    gunzip -c "$BACKUP_FILE" | grep -E "^COPY .* FROM stdin" | while read -r line; do
        TABLE_NAME=$(echo "$line" | sed -E 's/^COPY (public\.)?([^ ]+).*/\2/')
        echo "  - Table: $TABLE_NAME"
    done
fi

echo ""
echo "=============================================="
log_success "Backup verification complete"
echo ""
echo "Summary:"
echo "  - File: $BACKUP_FILE"
echo "  - Size: $FILE_SIZE"
echo "  - Gzip integrity: PASSED"
echo "  - Tables found: $FOUND_TABLES/${#EXPECTED_TABLES[@]}"
echo ""
log_info "Backup appears to be valid and usable for restoration."
