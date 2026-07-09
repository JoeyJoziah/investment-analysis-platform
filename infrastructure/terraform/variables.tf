variable "aws_region" {
  description = "AWS region for platform resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used in tags"
  type        = string
  default     = "investment-analysis-platform"
}

variable "environment" {
  description = "Deployment environment (staging, production)"
  type        = string
  default     = "staging"
}

variable "backup_bucket_name" {
  description = "Globally unique S3 bucket name for database backups"
  type        = string
}

variable "backup_prefix" {
  description = "Key prefix for backup objects"
  type        = string
  default     = "db-backups/"
}

variable "backup_retention_days" {
  description = "Days to retain backups (matches BACKUP_RETENTION_DAYS)"
  type        = number
  default     = 30
}
