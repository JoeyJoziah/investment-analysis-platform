#!/bin/bash
# =============================================================================
# Database Restore Script
# Investment Analysis Platform
# =============================================================================
# Usage: ./restore-backup.sh <backup_file> [--force]
# Example: ./restore-backup.sh /backups/backup_investment_db_20240115_020000.sql.gz
#
# WARNING: This script will DROP and recreate the database!
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DB_HOST="${DB_HOST:-postgres}"
DB_NAME="${DB_NAME:-investment_db}"
DB_USER="${DB_USER:-postgres}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo "Usage: $0 <backup_file> [--force]"
    echo ""
    echo "Arguments:"
    echo "  backup_file    Path to the gzipped backup file (.sql.gz)"
    echo "  --force        Skip confirmation prompt (USE WITH CAUTION)"
    echo ""
    echo "Environment Variables:"
    echo "  DB_HOST        Database host (default: postgres)"
    echo "  DB_NAME        Database name (default: investment_db)"
    echo "  DB_USER        Database user (default: postgres)"
    echo "  PGPASSWORD     Database password (required)"
    echo ""
    echo "Examples:"
    echo "  $0 /backups/backup_investment_db_20240115_020000.sql.gz"
    echo "  PGPASSWORD=secret $0 /backups/backup.sql.gz --force"
    exit 1
}

verify_backup() {
    local backup_file=$1
    log_info "Verifying backup file integrity..."

    if [ -f "${SCRIPT_DIR}/verify-backup.sh" ]; then
        "${SCRIPT_DIR}/verify-backup.sh" "$backup_file"
        return $?
    else
        # Basic gzip verification
        if gunzip -t "$backup_file" 2>/dev/null; then
            log_success "Gzip integrity check passed"
            return 0
        else
            log_error "Gzip integrity check failed"
            return 1
        fi
    fi
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
FORCE=false

if [ "$2" = "--force" ]; then
    FORCE=true
fi

# Verify file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Verify PGPASSWORD is set
if [ -z "$PGPASSWORD" ]; then
    log_error "PGPASSWORD environment variable is required"
    echo "Set it with: export PGPASSWORD=your_password"
    exit 1
fi

echo ""
echo -e "${CYAN}=============================================="
echo "   DATABASE RESTORE - INVESTMENT PLATFORM"
echo "==============================================${NC}"
echo ""
log_warn "This operation will:"
echo "  1. Terminate all existing connections to the database"
echo "  2. DROP the existing database: $DB_NAME"
echo "  3. CREATE a new empty database: $DB_NAME"
echo "  4. RESTORE data from: $BACKUP_FILE"
echo ""
log_error "ALL EXISTING DATA WILL BE PERMANENTLY LOST!"
echo ""

# Verify backup integrity before proceeding
verify_backup "$BACKUP_FILE"
if [ $? -ne 0 ]; then
    log_error "Backup verification failed. Aborting restore."
    exit 1
fi

# Safety confirmation
if [ "$FORCE" != true ]; then
    echo ""
    echo -e "${YELLOW}To proceed, type the database name to confirm: ${NC}"
    read -p "Database name: " CONFIRM_DB

    if [ "$CONFIRM_DB" != "$DB_NAME" ]; then
        log_error "Database name does not match. Aborting restore."
        exit 1
    fi

    echo ""
    read -p "Are you ABSOLUTELY sure you want to proceed? [yes/NO] " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        log_info "Restore aborted by user."
        exit 0
    fi
fi

echo ""
log_info "Starting database restore..."
echo ""

# Create timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Step 1: Terminate existing connections
log_info "Step 1/4: Terminating existing database connections..."
psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$DB_NAME'
  AND pid <> pg_backend_pid();
" 2>/dev/null || true

log_success "Existing connections terminated"

# Step 2: Drop existing database
log_info "Step 2/4: Dropping existing database..."
psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"

log_success "Database dropped"

# Step 3: Create new database
log_info "Step 3/4: Creating new database..."
psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# Enable TimescaleDB extension if available
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" 2>/dev/null || true

log_success "Database created"

# Step 4: Restore from backup
log_info "Step 4/4: Restoring data from backup..."
log_info "This may take several minutes depending on backup size..."

# Get file size for progress indication
FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "  Backup file size: $FILE_SIZE"

START_TIME=$(date +%s)

# Restore the backup
gunzip -c "$BACKUP_FILE" | psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" --quiet

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log_success "Data restored successfully"

echo ""
echo -e "${CYAN}=============================================="
echo "          RESTORE COMPLETE"
echo "==============================================${NC}"
echo ""
echo "Summary:"
echo "  - Database: $DB_NAME"
echo "  - Host: $DB_HOST"
echo "  - Backup file: $BACKUP_FILE"
echo "  - Duration: ${DURATION} seconds"
echo "  - Completed at: $(date)"
echo ""

# Verify restoration
log_info "Verifying restoration..."

# Get table count
TABLE_COUNT=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*)
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_type = 'BASE TABLE';
")

# Get approximate row counts for key tables
echo ""
log_info "Table statistics after restore:"
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT
    schemaname,
    relname as table_name,
    n_live_tup as row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC
LIMIT 10;
"

echo ""
log_success "Restore verification complete"
log_info "Total tables restored: $(echo $TABLE_COUNT | tr -d ' ')"
echo ""
log_info "Next steps:"
echo "  1. Verify application connectivity"
echo "  2. Test critical functionality"
echo "  3. Review application logs for any issues"
