output "backup_bucket_name" {
  description = "S3 bucket for automated DB backups"
  value       = aws_s3_bucket.backups.bucket
}

output "backup_bucket_arn" {
  description = "ARN of the backup bucket"
  value       = aws_s3_bucket.backups.arn
}

output "backup_prefix" {
  description = "Object key prefix for backups"
  value       = var.backup_prefix
}
