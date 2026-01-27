# ML Operations Guide

**Version**: 1.0.0  
**Last Updated**: August 19, 2025  
**Target Audience**: DevOps Engineers, ML Engineers, Site Reliability Engineers

---

## Overview

This guide provides comprehensive operational procedures for the Investment Analysis Platform's Machine Learning Pipeline. It covers daily operations, monitoring, troubleshooting, performance tuning, and maintenance procedures to ensure reliable ML model serving at scale.

### Operational Objectives

- **99.9% Uptime**: Maintain high availability for ML inference services
- **Cost Management**: Keep operational costs under $50/month
- **Performance**: Sub-100ms prediction latency
- **Reliability**: Automated failover and recovery
- **Compliance**: Audit trails and data governance

---

## Daily Operations Checklist

### Morning (08:00 UTC)

#### ✅ System Health Verification
```bash
# 1. Check ML API server status
curl -s http://localhost:8001/health | jq '.'

# 2. Verify all models are loaded
curl -s http://localhost:8001/models | jq '.[] | .name'

# 3. Check docker services
docker-compose ps | grep -E "(ml-api|backend|database|redis)"

# 4. Review overnight logs for errors
docker-compose logs --since=24h ml-api | grep -i error

# 5. Check disk space for ML models
df -h backend/ml_models/
```

#### ✅ Performance Metrics Review
```bash
# Check prediction latency (last 24h)
curl -s http://localhost:8000/api/metrics | grep ml_prediction_duration

# Memory usage monitoring
docker stats --no-stream | grep ml-api

# API response times
curl -w "Total time: %{time_total}s\n" -s -o /dev/null http://localhost:8001/health
```

#### ✅ Training Status Check
```bash
# Check if models were retrained overnight
ls -la backend/ml_models/ | head -10

# Review training logs
tail -50 backend/ml_logs/training_$(date +%Y%m%d)*.log

# Verify training pipeline status
python3 backend/ml/simple_training_pipeline.py --status
```

### Midday (12:00 UTC)

#### ✅ Cost Monitoring
```bash
# Check API usage (if applicable)
python3 scripts/monitoring/check_api_costs.py

# Resource utilization review
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Storage usage
du -sh backend/ml_models/ backend/ml_logs/ backend/ml_registry/
```

### Evening (20:00 UTC)

#### ✅ Preparation for Overnight Training
```bash
# Ensure sufficient disk space for training
df -h | grep -E "(ml_models|ml_logs)"

# Check training pipeline configuration
cat backend/ml/pipeline/config.json | jq '.training'

# Backup current models (if needed)
./scripts/backup_ml_models.sh

# Verify database connectivity for training
python3 -c "from backend.utils.database import test_connection; print(test_connection())"
```

---

## Monitoring and Alerting

### Key Performance Indicators (KPIs)

#### System Metrics
- **API Uptime**: >99.9%
- **Response Time**: <100ms (95th percentile)
- **Throughput**: >1000 predictions/minute
- **Error Rate**: <1%

#### Model Metrics
- **Prediction Accuracy**: >80%
- **Model Freshness**: <24 hours since training
- **Data Drift Score**: <0.3 threshold
- **Training Success Rate**: >95%

#### Business Metrics
- **Daily Cost**: <$10 USD
- **Models Deployed**: ≥3 (LSTM, XGBoost, Prophet)
- **Feature Availability**: >95%
- **Recommendation Generation**: Daily completion

### Monitoring Commands

#### Real-time Monitoring
```bash
# Watch ML API health status
watch -n 30 'curl -s http://localhost:8001/health | jq ".status"'

# Monitor prediction requests
docker-compose logs -f ml-api | grep "POST /predict"

# Watch resource usage
docker stats ml-api backend database redis

# Monitor training pipeline
tail -f backend/ml_logs/training_$(date +%Y%m%d)*.log
```

#### Automated Monitoring Scripts
```bash
# Set up monitoring cron jobs
# Add to crontab: crontab -e

# Check health every 5 minutes
*/5 * * * * /opt/monitoring/health_check.sh

# Resource monitoring every 15 minutes
*/15 * * * * /opt/monitoring/resource_check.sh

# Daily cost report at 23:55
55 23 * * * /opt/monitoring/cost_report.sh

# Model performance check every hour
0 * * * * /opt/monitoring/model_performance_check.sh
```

### Alert Configuration

#### Critical Alerts (Immediate Response)
- ML API server down
- Database connection lost
- Memory usage >90%
- Disk space <500MB
- Error rate >5% for >5 minutes

#### Warning Alerts (Within 1 Hour)
- Response time >200ms for >10 minutes
- Model accuracy <75%
- Training pipeline failure
- Cost trending >$8/day

#### Info Alerts (Daily Review)
- New model deployed
- Data drift detected
- Performance improvement
- Training completed successfully

### Monitoring Script Examples

#### Health Check Script (`/opt/monitoring/health_check.sh`)
```bash
#!/bin/bash
HEALTH_URL="http://localhost:8001/health"
WEBHOOK_URL="YOUR_SLACK_WEBHOOK_URL"

response=$(curl -s -w "%{http_code}" $HEALTH_URL)
http_code=${response: -3}

if [ "$http_code" != "200" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🚨 ML API Health Check Failed: HTTP $http_code\"}" \
        $WEBHOOK_URL
    exit 1
fi

echo "Health check passed at $(date)"
```

#### Resource Monitoring Script (`/opt/monitoring/resource_check.sh`)
```bash
#!/bin/bash
WEBHOOK_URL="YOUR_SLACK_WEBHOOK_URL"

# Check memory usage
memory_usage=$(docker stats --no-stream ml-api --format "{{.MemPerc}}" | sed 's/%//')
if (( $(echo "$memory_usage > 85" | bc -l) )); then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"⚠️ ML API Memory Usage: ${memory_usage}%\"}" \
        $WEBHOOK_URL
fi

# Check disk space
disk_usage=$(df -h backend/ml_models | awk 'NR==2{print $5}' | sed 's/%//')
if [ "$disk_usage" -gt 85 ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"⚠️ ML Models Disk Usage: ${disk_usage}%\"}" \
        $WEBHOOK_URL
fi
```

---

## Performance Tuning

### Model Optimization

#### Memory Optimization
```python
# Model compression techniques
import torch

# 1. Model quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 2. Model pruning (remove less important weights)
import torch.nn.utils.prune as prune
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2  # Remove 20% of weights
)

# 3. Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

#### Inference Optimization
```python
# 1. Batch processing configuration
BATCH_SIZE = 32  # Optimal for GPU inference
MAX_BATCH_WAIT_TIME = 100  # milliseconds

# 2. Model warming
for _ in range(10):
    dummy_input = torch.randn(1, sequence_length, feature_size)
    _ = model(dummy_input)

# 3. ONNX conversion for faster inference
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    do_constant_folding=True
)
```

### Infrastructure Optimization

#### Docker Configuration
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  ml-api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    environment:
      - OMP_NUM_THREADS=2
      - TORCH_NUM_THREADS=2
    volumes:
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 1G
```

#### Redis Caching Optimization
```python
# Redis configuration for ML predictions
REDIS_CONFIG = {
    'maxmemory': '512mb',
    'maxmemory-policy': 'lru',
    'timeout': 0,
    'tcp-keepalive': 60,
    'databases': 16
}

# Prediction caching strategy
def cache_prediction(key, prediction, ttl=300):  # 5 min TTL
    redis_client.setex(key, ttl, json.dumps(prediction))

def get_cached_prediction(key):
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None
```

### Database Query Optimization

#### Training Data Queries
```sql
-- Create indexes for efficient data retrieval
CREATE INDEX CONCURRENTLY idx_stock_data_symbol_timestamp 
ON stock_data (symbol, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_technical_indicators_symbol_date 
ON technical_indicators (symbol, date DESC);

-- Optimize training data query
WITH latest_data AS (
    SELECT symbol, 
           timestamp,
           price_data,
           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
    FROM stock_data 
    WHERE timestamp >= NOW() - INTERVAL '1 year'
)
SELECT * FROM latest_data WHERE rn <= 252; -- Last year of trading days
```

---

## Backup and Recovery

### Backup Procedures

#### Model Backup
```bash
#!/bin/bash
# backup_ml_models.sh

BACKUP_DIR="/backups/ml_models"
DATE=$(date +%Y%m%d_%H%M%S)
MODELS_DIR="backend/ml_models"

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup model files
cp -r "$MODELS_DIR"/* "$BACKUP_DIR/$DATE/"

# Backup model metadata
cp -r backend/ml_logs "$BACKUP_DIR/$DATE/"
cp -r backend/ml_registry "$BACKUP_DIR/$DATE/"

# Compress backup
tar -czf "$BACKUP_DIR/ml_backup_$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "ml_backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: ml_backup_$DATE.tar.gz"
```

#### Database Backup
```bash
#!/bin/bash
# backup_ml_database.sh

BACKUP_DIR="/backups/database"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup ML-related tables
pg_dump -h localhost -U postgres -d investment_platform \
    -t stock_data -t model_predictions -t training_logs \
    --data-only --inserts \
    > "$BACKUP_DIR/ml_data_$DATE.sql"

# Backup schema
pg_dump -h localhost -U postgres -d investment_platform \
    -t stock_data -t model_predictions -t training_logs \
    --schema-only \
    > "$BACKUP_DIR/ml_schema_$DATE.sql"

echo "Database backup completed: ml_data_$DATE.sql"
```

### Recovery Procedures

#### Model Recovery
```bash
#!/bin/bash
# restore_ml_models.sh

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

echo "Stopping ML API server..."
docker-compose stop ml-api

echo "Backing up current models..."
mv backend/ml_models backend/ml_models.backup.$(date +%s)

echo "Restoring models from $BACKUP_FILE..."
tar -xzf "$BACKUP_FILE" -C backend/

echo "Starting ML API server..."
docker-compose start ml-api

echo "Verifying restoration..."
sleep 30
curl -s http://localhost:8001/health | jq '.'
```

#### Database Recovery
```bash
#!/bin/bash
# restore_ml_database.sh

SCHEMA_FILE=$1
DATA_FILE=$2

if [ -z "$SCHEMA_FILE" ] || [ -z "$DATA_FILE" ]; then
    echo "Usage: $0 <schema_file.sql> <data_file.sql>"
    exit 1
fi

echo "Restoring ML database schema..."
psql -h localhost -U postgres -d investment_platform -f "$SCHEMA_FILE"

echo "Restoring ML data..."
psql -h localhost -U postgres -d investment_platform -f "$DATA_FILE"

echo "Database restoration completed"
```

### Disaster Recovery Plan

#### Complete System Recovery
1. **Infrastructure Setup** (0-30 minutes)
   - Restore Docker environment
   - Restore database from backup
   - Configure networking

2. **ML Components Recovery** (30-60 minutes)
   - Restore model files from backup
   - Verify model compatibility
   - Load models into ML API server

3. **Validation and Testing** (60-90 minutes)
   - Health check verification
   - Sample prediction testing
   - Performance validation

4. **Full Service Restoration** (90-120 minutes)
   - Enable automatic retraining
   - Resume monitoring
   - Notify stakeholders

---

## Scaling Guidelines

### Horizontal Scaling

#### Load Balancer Configuration (Nginx)
```nginx
upstream ml_api {
    server ml-api-1:8001 weight=1;
    server ml-api-2:8001 weight=1;
    server ml-api-3:8001 weight=1;
    
    # Health check
    keepalive 32;
}

server {
    listen 80;
    server_name ml.example.com;
    
    location / {
        proxy_pass http://ml_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 30;
        proxy_send_timeout 30;
        proxy_read_timeout 30;
    }
    
    location /health {
        proxy_pass http://ml_api/health;
        access_log off;
    }
}
```

#### Multi-Instance Docker Compose
```yaml
version: '3.8'
services:
  ml-api-1:
    build: .
    ports:
      - "8001:8001"
    environment:
      - INSTANCE_ID=1
    volumes:
      - ./backend/ml_models:/app/ml_models:ro
  
  ml-api-2:
    build: .
    ports:
      - "8002:8001"
    environment:
      - INSTANCE_ID=2
    volumes:
      - ./backend/ml_models:/app/ml_models:ro
  
  ml-api-3:
    build: .
    ports:
      - "8003:8001"
    environment:
      - INSTANCE_ID=3
    volumes:
      - ./backend/ml_models:/app/ml_models:ro
```

### Vertical Scaling

#### Resource Allocation Guidelines
```yaml
# Production resource allocation
services:
  ml-api:
    deploy:
      resources:
        limits:
          memory: 8G      # For large models
          cpus: '4.0'     # Multi-threaded inference
        reservations:
          memory: 4G
          cpus: '2.0'
```

### Auto-scaling Configuration

#### Kubernetes HPA (Horizontal Pod Autoscaler)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Cost Optimization

### Cost Monitoring Dashboard

```python
#!/usr/bin/env python3
# cost_monitor.py

import json
import requests
from datetime import datetime, timedelta

class CostMonitor:
    def __init__(self):
        self.daily_limit = 10.0  # USD
        self.cost_breakdown = {}
    
    def calculate_daily_costs(self):
        """Calculate daily operational costs"""
        costs = {
            'api_calls': self.calculate_api_costs(),
            'compute': self.calculate_compute_costs(),
            'storage': self.calculate_storage_costs(),
            'network': self.calculate_network_costs()
        }
        
        total = sum(costs.values())
        
        return {
            'total': total,
            'breakdown': costs,
            'remaining_budget': self.daily_limit - total,
            'utilization_percent': (total / self.daily_limit) * 100
        }
    
    def calculate_api_costs(self):
        """Calculate external API costs"""
        # Alpha Vantage, Finnhub, Polygon.io usage
        # Assuming free tiers with rate limiting
        return 0.0
    
    def calculate_compute_costs(self):
        """Calculate compute resource costs"""
        # Assuming local deployment or free tier cloud
        return 0.0
    
    def calculate_storage_costs(self):
        """Calculate storage costs for models and data"""
        # Local storage or free tier cloud storage
        return 0.0
    
    def send_cost_alert(self, cost_data):
        """Send alert if cost threshold exceeded"""
        if cost_data['utilization_percent'] > 80:
            # Send alert to monitoring system
            alert = {
                'message': f"Daily cost at {cost_data['utilization_percent']:.1f}%",
                'total_cost': cost_data['total'],
                'timestamp': datetime.now().isoformat()
            }
            print(f"COST ALERT: {alert}")

if __name__ == "__main__":
    monitor = CostMonitor()
    costs = monitor.calculate_daily_costs()
    monitor.send_cost_alert(costs)
    print(json.dumps(costs, indent=2))
```

### Resource Optimization Strategies

#### 1. Model Optimization
- Use quantized models (8-bit instead of 32-bit)
- Implement model pruning (remove 20-30% of parameters)
- Enable gradient checkpointing to reduce memory

#### 2. Infrastructure Optimization
- Use spot instances for training workloads
- Implement auto-scaling to scale down during low usage
- Optimize container resource allocation

#### 3. API Cost Management
- Implement intelligent caching for API responses
- Batch API calls during off-peak hours
- Use free tier quotas efficiently across multiple keys

#### 4. Data Management
- Compress stored model files
- Implement data retention policies
- Use efficient data formats (Parquet vs CSV)

---

## Security and Compliance

### Security Checklist

#### ✅ Model Security
- Model files encrypted at rest
- Secure model serving endpoints
- Input validation for all predictions
- Rate limiting to prevent abuse

#### ✅ Data Security
- Training data encrypted
- PII anonymization in logs
- Secure database connections
- Regular security audits

#### ✅ Infrastructure Security
- Container vulnerability scanning
- Network segmentation
- Access control and authentication
- Regular security updates

### Compliance Monitoring

#### GDPR Compliance
```python
# Data anonymization for logs
import hashlib
import re

def anonymize_logs(log_content):
    """Anonymize sensitive information in logs"""
    # Remove IP addresses
    log_content = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 
                        '[IP_ANONYMIZED]', log_content)
    
    # Hash user identifiers
    def hash_match(match):
        return f"user_{hashlib.md5(match.group().encode()).hexdigest()[:8]}"
    
    log_content = re.sub(r'user_id=\w+', hash_match, log_content)
    
    return log_content
```

#### SEC Compliance
- Audit trails for all model predictions
- Model version tracking
- Decision transparency for recommendations
- Risk disclosure for automated trading

---

## Troubleshooting Runbook

### Common Issues and Solutions

#### Issue: ML API Server Not Responding
**Symptoms**: HTTP timeouts, connection refused errors
**Diagnosis**: 
```bash
# Check if container is running
docker-compose ps ml-api

# Check logs for errors
docker-compose logs ml-api | tail -50

# Check port binding
netstat -tlnp | grep 8001
```
**Resolution**:
```bash
# Restart ML API service
docker-compose restart ml-api

# If persistent, rebuild and restart
docker-compose down ml-api
docker-compose up -d ml-api
```

#### Issue: High Memory Usage
**Symptoms**: Out of memory errors, slow performance
**Diagnosis**:
```bash
# Check memory usage
docker stats --no-stream ml-api

# Check loaded models
curl -s http://localhost:8001/models | jq '.[] | .name'
```
**Resolution**:
```bash
# Unload unused models
curl -X DELETE http://localhost:8001/models/unused_model

# Restart with memory limits
docker-compose down ml-api
docker-compose up -d ml-api
```

#### Issue: Model Prediction Errors
**Symptoms**: HTTP 500 errors on prediction requests
**Diagnosis**:
```bash
# Check model status
curl -s http://localhost:8001/models/problem_model/info

# Test with simple input
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 0.0, 0.0, 0.0, 0.0]}'
```
**Resolution**:
```bash
# Reload problematic model
curl -X POST http://localhost:8001/models/problem_model/load

# If persistent, retrain model
curl -X POST http://localhost:8001/retrain
```

---

## Change Management

### Model Update Procedure

#### 1. Pre-deployment Testing
```bash
# Test new model offline
python3 scripts/test_model.py new_model.pkl test_data.csv

# Performance benchmark
python3 scripts/benchmark_model.py new_model.pkl
```

#### 2. Staging Deployment
```bash
# Deploy to staging environment
cp new_model.pkl backend/ml_models/staging_model.pkl

# Load in staging API
curl -X POST http://staging:8001/models/staging_model/load

# Run integration tests
pytest tests/integration/test_staging_model.py
```

#### 3. Production Deployment
```bash
# Blue-green deployment
./scripts/deploy_model.sh new_model.pkl production

# Health check
./scripts/health_check.sh

# Rollback if needed
./scripts/rollback_model.sh
```

### Configuration Changes

#### Infrastructure Changes
1. Update docker-compose files
2. Test in staging environment
3. Plan maintenance window
4. Apply changes with rollback plan
5. Verify functionality post-deployment

#### Model Configuration Changes
1. Update configuration files
2. Validate configuration syntax
3. Test with existing models
4. Deploy during low-traffic period
5. Monitor for issues

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-08-19 | Initial ML operations guide |

---

## Related Documentation

- [ML Pipeline Documentation](ML_PIPELINE_DOCUMENTATION.md)
- [ML API Reference](ML_API_REFERENCE.md)
- [ML Quick Start Guide](ML_QUICKSTART.md)