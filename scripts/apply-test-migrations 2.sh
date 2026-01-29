#!/bin/bash
#
# Apply Migrations to Test Database (Without CONCURRENT)
# For test databases, we don't need CONCURRENT index creation
# This script temporarily patches migrations to remove CONCURRENT keyword
#

set -e

echo "======================================"
echo "Apply Test Database Migrations"
echo "======================================"
echo ""

# Configuration
TEST_DB_URL="postgresql://postgres:CEP4j9ZHgd352ONsrj8VgKRCwoOR8Yp@localhost:5432/investment_db_test"
MIGRATIONS_DIR="backend/migrations/versions"

# Create backup directory
BACKUP_DIR=".migration_backups_$(date +%s)"
mkdir -p "$BACKUP_DIR"

echo "[1/4] Creating backups of migration files..."
find "$MIGRATIONS_DIR" -name "*.py" -type f | while read file; do
    if grep -q "CONCURRENTLY" "$file"; then
        cp "$file" "$BACKUP_DIR/"
        echo "  Backed up: $(basename $file)"
    fi
done
echo "✓ Backups created in $BACKUP_DIR"
echo ""

echo "[2/4] Temporarily removing CONCURRENT from migrations..."
find "$MIGRATIONS_DIR" -name "*.py" -type f | while read file; do
    if grep -q "CONCURRENTLY" "$file"; then
        # Remove CONCURRENTLY keyword for test execution
        sed -i '' 's/CONCURRENTLY //g' "$file"
        echo "  Patched: $(basename $file)"
    fi
done
echo "✓ CONCURRENT keywords removed"
echo ""

echo "[3/4] Running migrations..."
DATABASE_URL="$TEST_DB_URL" alembic upgrade heads 2>&1 | tail -30
MIGRATION_STATUS=$?
echo ""

echo "[4/4] Restoring original migration files..."
find "$BACKUP_DIR" -name "*.py" -type f | while read backup_file; do
    filename=$(basename "$backup_file")
    cp "$backup_file" "$MIGRATIONS_DIR/$filename"
    echo "  Restored: $filename"
done
rm -rf "$BACKUP_DIR"
echo "✓ Original files restored"
echo ""

if [ $MIGRATION_STATUS -eq 0 ]; then
    echo "======================================"
    echo "✓ Migrations Applied Successfully"
    echo "======================================"
    echo ""
    echo "Test database ready at:"
    echo "  $TEST_DB_URL"
    echo ""
    exit 0
else
    echo "======================================"
    echo "✗ Migration Failed"
    echo "======================================"
    echo ""
    echo "Check the error output above"
    echo "Original migration files have been restored"
    echo ""
    exit 1
fi
