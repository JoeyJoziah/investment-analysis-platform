#!/bin/sh
# Database Backup Script

set -e

# Configuration
BACKUP_DIR="/backups"
DB_HOST="postgres"
DB_NAME="${DB_NAME:-investment_db}"
DB_USER="${DB_USER:-postgres}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/backup_${DB_NAME}_${TIMESTAMP}.sql.gz"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Create backup directory if it doesn't exist
mkdir -p ${BACKUP_DIR}

echo "Starting backup of ${DB_NAME} at $(date)"

# Perform backup
pg_dump -h ${DB_HOST} -U ${DB_USER} -d ${DB_NAME} --no-password --verbose | gzip > ${BACKUP_FILE}

if [ $? -eq 0 ]; then
    echo "Backup completed successfully: ${BACKUP_FILE}"
    
    # Get file size
    SIZE=$(du -h ${BACKUP_FILE} | cut -f1)
    echo "Backup size: ${SIZE}"
    
    # Remove old backups
    echo "Removing backups older than ${RETENTION_DAYS} days"
    find ${BACKUP_DIR} -name "backup_${DB_NAME}_*.sql.gz" -mtime +${RETENTION_DAYS} -delete
    
    # Upload to S3 if configured
    if [ ! -z "${BACKUP_S3_BUCKET}" ]; then
        echo "Uploading backup to S3..."
        aws s3 cp ${BACKUP_FILE} s3://${BACKUP_S3_BUCKET}/database-backups/ --storage-class GLACIER
        echo "Backup uploaded to S3"
    fi
else
    echo "Backup failed!"
    exit 1
fi

echo "Backup process completed at $(date)"