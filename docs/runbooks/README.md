# Investment Analysis App - Operational Runbooks

This directory contains operational runbooks for the Investment Analysis Application. These runbooks provide step-by-step procedures for common operational tasks, troubleshooting, and incident response.

## Index of Runbooks

### Daily Operations
- [Daily Startup Procedure](./daily-startup.md) - Starting the application stack
- [Daily Health Checks](./daily-health-checks.md) - Routine monitoring and verification
- [Data Pipeline Monitoring](./data-pipeline-monitoring.md) - Monitoring ETL processes

### Incident Response
- [Service Down Recovery](./service-down-recovery.md) - Steps when services are unavailable
- [Database Issues](./database-issues.md) - Database connection and performance problems
- [API Rate Limit Exceeded](./api-rate-limit-exceeded.md) - Handling API quota exhaustion
- [Memory/CPU Issues](./memory-cpu-issues.md) - Resource exhaustion scenarios

### Maintenance
- [Weekly Maintenance](./weekly-maintenance.md) - Routine weekly tasks
- [Database Maintenance](./database-maintenance.md) - Database optimization and cleanup
- [Cache Management](./cache-management.md) - Redis cache maintenance
- [Log Rotation](./log-rotation.md) - Managing application logs

### Deployment & Configuration
- [Production Deployment](./production-deployment.md) - Deploying to production
- [Configuration Changes](./configuration-changes.md) - Updating application configuration
- [Environment Setup](./environment-setup.md) - Setting up new environments

### Monitoring & Alerting
- [Alert Response](./alert-response.md) - Responding to monitoring alerts
- [Performance Tuning](./performance-tuning.md) - Optimizing system performance
- [Cost Monitoring](./cost-monitoring.md) - Managing operational costs

## Emergency Contacts

- **System Administrator**: [Your contact info]
- **Database Administrator**: [DBA contact info]
- **On-call Engineer**: [On-call contact info]

## Quick Reference

### Service URLs
- **Application**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3001
- **Airflow**: http://localhost:8080

### Key Commands
```bash
# Check service status
make status

# View logs
make logs

# Restart services
make restart

# Emergency stop
make down
```

### Critical Thresholds
- **API Calls**: 80% of daily limit
- **Memory Usage**: 85% of available
- **Disk Space**: 90% full
- **Response Time**: >5 seconds
- **Error Rate**: >5%

## Using These Runbooks

1. **Follow procedures step by step** - Don't skip steps even if they seem obvious
2. **Document your actions** - Keep notes of what you do and the results
3. **Escalate when needed** - Don't hesitate to contact the team if issues persist
4. **Update runbooks** - If procedures change, update the runbooks immediately

## Runbook Template

When creating new runbooks, use this template:

```markdown
# Runbook Title

## Overview
Brief description of when and why to use this runbook.

## Prerequisites
- List of requirements
- Permissions needed
- Tools required

## Procedure
1. Step one with expected outcome
2. Step two with verification
3. Continue with numbered steps

## Verification
How to confirm the procedure was successful.

## Rollback
Steps to undo changes if needed.

## Related Runbooks
Links to related procedures.
```