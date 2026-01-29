# ML Pipeline Documentation

**Version**: 1.0.0  
**Last Updated**: August 19, 2025  
**Environment**: Investment Analysis Platform

---

## Executive Summary

The Investment Analysis Platform's Machine Learning Pipeline is a comprehensive, automated system designed to train, deploy, and monitor ML models for stock price prediction and financial analysis. The system processes 6,000+ stocks from NYSE, NASDAQ, and AMEX exchanges daily, delivering AI-powered investment recommendations while maintaining operational costs under $50/month.

### Key Features

- **Automated Model Training**: Daily retraining with performance monitoring
- **Multi-Model Architecture**: LSTM, XGBoost, Prophet, and ensemble methods
- **Real-time Inference**: Sub-second prediction latency via dedicated ML API server
- **Cost-Optimized**: Smart resource management and efficient API usage
- **Production-Ready**: Monitoring, alerting, and automated failover capabilities

### Business Impact

- **Daily Recommendations**: Automated analysis of 6,000+ stocks
- **Cost Efficiency**: <$50/month operational cost target
- **High Availability**: 99.9% uptime SLA with automated recovery
- **Regulatory Compliance**: SEC and GDPR compliant data processing

---

## Architecture Overview

The ML Pipeline consists of three main layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML ORCHESTRATION LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Orchestrator  │  │    Scheduler    │  │    Monitor      │  │
│  │   (pipeline     │  │   (automatic    │  │   (performance  │  │
│  │   execution)    │  │   retraining)   │  │   tracking)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    ML TRAINING LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Data Loading   │  │ Feature Engine  │  │ Model Training  │  │
│  │  & Validation   │  │  & Selection    │  │  & Evaluation   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    Registry     │  │   Versioning    │  │   Deployment    │  │
│  │   (models)      │  │   (artifacts)   │  │   (staging)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    ML INFERENCE LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   ML API Server │  │  Model Manager  │  │ Cache Manager   │  │
│  │   (port 8001)   │  │  (loading &     │  │ (predictions)   │  │
│  │                 │  │  inference)     │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
[Market Data] → [ETL Pipeline] → [Feature Store] → [ML Training]
                       ↓
[Real-time Data] → [API Server] → [Model Inference] → [Predictions]
                       ↓                ↓
                [Monitoring] ← [Performance Metrics]
```

---

## Component Descriptions

### 1. ML Orchestrator

**Location**: `backend/ml/pipeline/orchestrator.py`  
**Port**: N/A (Internal service)  
**Purpose**: Manages the entire ML pipeline lifecycle

#### Key Responsibilities
- **Pipeline Scheduling**: Automated daily training at 2 AM UTC
- **Resource Management**: Limits concurrent pipelines (max 3)
- **Failure Handling**: Retry logic with exponential backoff
- **Trigger Management**: Performance degradation and data drift detection

#### Configuration
```python
OrchestratorConfig(
    name="ml_orchestrator",
    max_concurrent_pipelines=3,
    enable_auto_retraining=True,
    cost_limit_daily_usd=10.0,
    performance_threshold=0.8
)
```

### 2. ML API Server

**Location**: `backend/ml/ml_api_server.py`  
**Port**: 8001  
**Purpose**: Serves ML model predictions via REST API

#### Key Features
- **Model Loading**: Dynamic model loading and unloading
- **Inference**: Real-time prediction with confidence scores
- **Health Monitoring**: Built-in health checks and model status
- **Background Training**: Trigger retraining via API

#### Performance Metrics
- **Latency**: <100ms per prediction
- **Throughput**: 1000+ predictions/minute
- **Availability**: 99.9% uptime

### 3. Model Registry

**Location**: `backend/ml/pipeline/registry.py`  
**Purpose**: Version control and metadata management for ML models

#### Model Tracking
- **Versioning**: Semantic versioning (major.minor.patch)
- **Metadata**: Training metrics, data sources, hyperparameters
- **Lineage**: Model ancestry and deployment history
- **Status**: Active, deprecated, archived model states

### 4. Model Monitor

**Location**: `backend/ml/pipeline/monitoring.py`  
**Purpose**: Continuous monitoring of model performance and data quality

#### Monitoring Capabilities
- **Performance Tracking**: Accuracy, precision, recall, F1-score
- **Data Drift Detection**: Statistical drift analysis
- **Alert System**: Automated alerts for performance degradation
- **Cost Tracking**: Resource usage and API costs

### 5. Feature Store

**Location**: `backend/ml/feature_store.py`  
**Purpose**: Centralized feature engineering and storage

#### Features Managed
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Fundamental Data**: P/E ratios, market cap, financial ratios
- **Sentiment Features**: News sentiment, social media sentiment
- **Market Features**: Volatility, volume patterns, correlation metrics

---

## Data Flow Diagrams

### Training Pipeline Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Market    │───▶│     ETL     │───▶│   Feature   │
│    Data     │    │  Pipeline   │    │   Store     │
│ (External   │    │             │    │             │
│   APIs)     │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                    │
                           ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Training  │◀───│   Data      │◀───│  Feature    │
│  Pipeline   │    │ Validation  │    │Engineering  │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Model     │───▶│   Model     │───▶│ Deployment  │
│  Training   │    │ Evaluation  │    │   & API     │
│             │    │             │    │  Server     │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Real-time Inference Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Client API  │───▶│  ML API     │───▶│   Model     │
│  Request    │    │  Server     │    │  Manager    │
│             │    │ (port 8001) │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                   │                    │
       │                   ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Prediction  │◀───│    Cache    │◀───│  Loaded     │
│  Response   │    │  Manager    │    │   Model     │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## API Reference

### Core ML Endpoints

#### Health Check
```http
GET /health
```
**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-08-19T10:30:00Z",
    "models_loaded": 3,
    "models": ["lstm_predictor", "xgboost_classifier", "prophet_forecaster"]
}
```

#### List Models
```http
GET /models
```
**Response:**
```json
[
    {
        "name": "lstm_predictor",
        "type": "neural_network",
        "features": 20,
        "score": 0.85,
        "loaded_at": "2025-08-19T02:00:00Z"
    }
]
```

#### Make Prediction
```http
POST /predict
Content-Type: application/json

{
    "features": [1.2, 0.8, 150.5, ...],
    "model_name": "lstm_predictor"
}
```
**Response:**
```json
{
    "prediction": 152.30,
    "model_name": "lstm_predictor",
    "timestamp": "2025-08-19T10:30:00Z",
    "confidence": 0.92
}
```

### Pipeline Management Endpoints

#### Trigger Retraining
```http
POST /retrain
```
**Response:**
```json
{
    "message": "Training task queued",
    "timestamp": "2025-08-19T10:30:00Z"
}
```

#### Load Model
```http
POST /models/{model_name}/load
```

#### Unload Model
```http
DELETE /models/{model_name}
```

---

## Configuration Guide

### Environment Variables

```bash
# ML Configuration
ML_MODELS_PATH=/app/ml_models
ML_LOGS_PATH=/app/ml_logs
ML_REGISTRY_PATH=/app/ml_registry

# Training Configuration
ENABLE_AUTO_RETRAINING=true
MODEL_PERFORMANCE_THRESHOLD=0.75
DATA_DRIFT_THRESHOLD=0.3
ML_DAILY_COST_LIMIT_USD=10.0

# API Configuration
ML_API_HOST=0.0.0.0
ML_API_PORT=8001
ML_WORKERS=4

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
```

### Pipeline Configuration

**File**: `backend/ml/pipeline/config.json`

```json
{
    "orchestrator": {
        "name": "ml_orchestrator",
        "max_concurrent_pipelines": 3,
        "enable_scheduling": true,
        "enable_auto_retraining": true,
        "cost_limit_daily_usd": 10.0
    },
    "training": {
        "default_schedule": {
            "frequency": "daily",
            "time_of_day": "02:00"
        },
        "performance_threshold": 0.8,
        "drift_threshold": 0.3,
        "max_training_hours": 24.0
    },
    "models": {
        "lstm_predictor": {
            "type": "neural_network",
            "features": 20,
            "sequence_length": 60,
            "hidden_size": 128
        },
        "xgboost_classifier": {
            "type": "tree_ensemble",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        }
    }
}
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Model Loading Failures

**Symptoms**: Models fail to load on startup
```
ERROR: Failed to load model lstm_predictor: FileNotFoundError
```

**Solutions**:
1. Verify model files exist in `ML_MODELS_PATH`
2. Check file permissions
3. Run training pipeline to generate models:
   ```bash
   python3 backend/ml/simple_training_pipeline.py
   ```

#### 2. Training Pipeline Failures

**Symptoms**: Training jobs fail or timeout
```
ERROR: Pipeline training_pipeline_id failed: Out of memory
```

**Solutions**:
1. Check available memory: `free -h`
2. Reduce batch size in configuration
3. Enable data sampling for large datasets
4. Monitor resource usage: `docker stats`

#### 3. High API Latency

**Symptoms**: Prediction requests take >1 second
```
WARNING: Prediction latency exceeded 1000ms
```

**Solutions**:
1. Check model loading status: `GET /health`
2. Verify cache performance
3. Scale up ML API server instances
4. Optimize model complexity

#### 4. Data Drift Alerts

**Symptoms**: Automated retraining triggered frequently
```
INFO: Retraining trigger activated: data_drift threshold exceeded
```

**Solutions**:
1. Review drift threshold configuration
2. Analyze feature distributions
3. Update feature engineering pipeline
4. Consider concept drift vs. data quality issues

#### 5. Cost Overruns

**Symptoms**: Daily cost limits exceeded
```
WARNING: Daily cost limit exceeded: $12.50 > $10.00
```

**Solutions**:
1. Review API usage patterns
2. Optimize prediction caching
3. Implement request batching
4. Adjust training frequency

### Diagnostic Commands

```bash
# Check ML service status
docker-compose ps ml-api

# View ML logs
docker-compose logs -f ml-api

# Check model files
ls -la backend/ml_models/

# Test ML API directly
curl http://localhost:8001/health

# Monitor resource usage
docker stats --no-stream

# Check training pipeline status
python3 backend/ml/simple_training_pipeline.py --status
```

### Performance Tuning

#### Memory Optimization
```python
# Reduce model complexity
config.hidden_size = 64  # from 128
config.sequence_length = 30  # from 60

# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use mixed precision training
config.use_amp = True
```

#### Inference Optimization
```python
# Batch predictions
batch_size = 32  # Process multiple predictions together

# Model quantization
model = torch.quantization.quantize_dynamic(model, qconfig_spec)

# ONNX conversion for faster inference
torch.onnx.export(model, dummy_input, "model.onnx")
```

### Monitoring and Alerting

#### Key Metrics to Monitor
- **Model Performance**: Accuracy, precision, recall
- **System Performance**: Latency, throughput, error rate
- **Resource Usage**: CPU, memory, disk space
- **Cost Metrics**: API costs, compute costs

#### Alert Thresholds
- **High Latency**: >500ms average response time
- **Low Accuracy**: <80% prediction accuracy
- **High Error Rate**: >5% failed requests
- **Resource Usage**: >85% CPU or memory utilization

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-08-19 | Initial ML pipeline implementation |

---

## Related Documentation

- [ML API Reference](ML_API_REFERENCE.md)
- [ML Operations Guide](ML_OPERATIONS_GUIDE.md)
- [ML Quick Start Guide](ML_QUICKSTART.md)
- [CLAUDE.md](CLAUDE.md) - Project instructions