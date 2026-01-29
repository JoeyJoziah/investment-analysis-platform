# ML API Reference

**Version**: 1.0.0  
**Last Updated**: August 19, 2025  
**Base URL**: `http://localhost:8001`

---

## Overview

The ML API provides access to machine learning models for stock price prediction and financial analysis. The API is designed for high-performance inference with sub-second response times and comprehensive error handling.

### Key Features

- **Real-time Predictions**: <100ms response time
- **Multiple Model Support**: LSTM, XGBoost, Prophet models
- **Dynamic Model Management**: Load/unload models on demand
- **Background Training**: Trigger retraining via API
- **Comprehensive Monitoring**: Health checks and performance metrics

### Authentication

Currently, the ML API does not require authentication for internal usage. In production deployments, implement appropriate authentication mechanisms.

### Rate Limiting

- **Default**: 1000 requests/minute per client
- **Burst**: Up to 50 requests in 10 seconds
- **Headers**: Rate limit information included in response headers

---

## Base Endpoints

### Root Endpoint

```http
GET /
```

Get basic API information and available models.

**Response:**
```json
{
    "message": "ML Inference API",
    "version": "1.0.0",
    "timestamp": "2025-08-19T10:30:00Z",
    "loaded_models": ["lstm_predictor", "xgboost_classifier", "prophet_forecaster"]
}
```

### Health Check

```http
GET /health
```

Check API health and model status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-08-19T10:30:00Z",
    "models_loaded": 3,
    "models": ["lstm_predictor", "xgboost_classifier", "prophet_forecaster"]
}
```

**Status Codes:**
- `200`: API is healthy
- `503`: API is unhealthy or degraded

---

## Model Management

### List Models

```http
GET /models
```

Get information about all available models.

**Response:**
```json
[
    {
        "name": "lstm_predictor",
        "type": "neural_network",
        "features": 20,
        "score": 0.85,
        "loaded_at": "2025-08-19T02:00:00Z"
    },
    {
        "name": "xgboost_classifier", 
        "type": "tree_ensemble",
        "features": 15,
        "score": 0.82,
        "loaded_at": "2025-08-19T02:01:00Z"
    },
    {
        "name": "prophet_forecaster",
        "type": "time_series",
        "features": 5,
        "score": 0.78,
        "loaded_at": "2025-08-19T02:02:00Z"
    }
]
```

### Get Model Information

```http
GET /models/{model_name}/info
```

Get detailed information about a specific model.

**Parameters:**
- `model_name` (string, required): Name of the model

**Example Request:**
```http
GET /models/lstm_predictor/info
```

**Response:**
```json
{
    "name": "lstm_predictor",
    "type": "neural_network",
    "version": "1.2.3",
    "features": 20,
    "score": 0.85,
    "training_date": "2025-08-19T02:00:00Z",
    "loaded_at": "2025-08-19T02:00:00Z",
    "hyperparameters": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 60
    },
    "metrics": {
        "mse": 0.0145,
        "mae": 0.0876,
        "r2_score": 0.8543
    },
    "input_shape": [60, 20],
    "output_shape": [1]
}
```

**Error Responses:**
- `404`: Model not found

### Load Model

```http
POST /models/{model_name}/load
```

Load a specific model into memory.

**Parameters:**
- `model_name` (string, required): Name of the model to load

**Example Request:**
```http
POST /models/lstm_predictor/load
```

**Response:**
```json
{
    "message": "Model lstm_predictor loaded successfully",
    "timestamp": "2025-08-19T10:30:00Z"
}
```

**Error Responses:**
- `404`: Model file not found
- `500`: Loading failed (insufficient memory, corrupted file, etc.)

### Unload Model

```http
DELETE /models/{model_name}
```

Remove a model from memory to free resources.

**Parameters:**
- `model_name` (string, required): Name of the model to unload

**Example Request:**
```http
DELETE /models/lstm_predictor
```

**Response:**
```json
{
    "message": "Model lstm_predictor unloaded successfully",
    "timestamp": "2025-08-19T10:30:00Z"
}
```

**Error Responses:**
- `404`: Model not loaded

---

## Prediction Endpoints

### Make Prediction

```http
POST /predict
Content-Type: application/json
```

Generate predictions using a loaded model.

**Request Body:**
```json
{
    "features": [1.2, 0.8, 150.5, 0.95, -0.02, 0.15, 2.4, 0.78, 145.2, 0.88, 152.1, 149.8, 151.0, 0.012, 0.034, 0.056, 0.023, 0.45, 0.67, 0.89],
    "model_name": "lstm_predictor"
}
```

**Parameters:**
- `features` (array of float, required): Input features for prediction
- `model_name` (string, optional): Model to use for prediction (default: "sample_model")

**Response:**
```json
{
    "prediction": 152.30,
    "model_name": "lstm_predictor", 
    "timestamp": "2025-08-19T10:30:00Z",
    "confidence": 0.92
}
```

**Error Responses:**
- `400`: Invalid input data
- `404`: Model not found or not loaded
- `500`: Prediction failed

### Batch Prediction

```http
POST /predict/batch
Content-Type: application/json
```

Generate predictions for multiple samples.

**Request Body:**
```json
{
    "features": [
        [1.2, 0.8, 150.5, ...],
        [1.1, 0.9, 151.2, ...],
        [1.3, 0.7, 149.8, ...]
    ],
    "model_name": "lstm_predictor"
}
```

**Response:**
```json
{
    "predictions": [152.30, 151.45, 150.92],
    "model_name": "lstm_predictor",
    "timestamp": "2025-08-19T10:30:00Z",
    "confidence": [0.92, 0.89, 0.94],
    "batch_size": 3
}
```

---

## Training and Management

### Trigger Retraining

```http
POST /retrain
```

Trigger background model retraining.

**Response:**
```json
{
    "message": "Training task queued",
    "timestamp": "2025-08-19T10:30:00Z",
    "estimated_completion": "2025-08-19T12:30:00Z"
}
```

**Error Responses:**
- `503`: Training already in progress
- `500`: Failed to queue training task

### Get Training Status

```http
GET /training/status
```

Check the status of ongoing or recent training jobs.

**Response:**
```json
{
    "status": "in_progress",
    "started_at": "2025-08-19T10:30:00Z",
    "estimated_completion": "2025-08-19T12:30:00Z",
    "progress": 0.45,
    "current_stage": "feature_engineering",
    "models_being_trained": ["lstm_predictor", "xgboost_classifier"]
}
```

**Status Values:**
- `idle`: No training in progress
- `queued`: Training job queued
- `in_progress`: Currently training
- `completed`: Recently completed
- `failed`: Training failed

---

## Financial-Specific Endpoints

### Stock Price Prediction

```http
POST /predict/stock_price
Content-Type: application/json
```

Specialized endpoint for stock price prediction with financial context.

**Request Body:**
```json
{
    "symbol": "AAPL",
    "features": {
        "technical_indicators": {
            "rsi_14": 65.5,
            "macd": 0.23,
            "bollinger_upper": 152.5,
            "bollinger_lower": 148.2,
            "sma_20": 150.1,
            "sma_50": 149.8
        },
        "fundamental_data": {
            "pe_ratio": 28.5,
            "market_cap": 2500000000000,
            "revenue_growth": 0.15,
            "profit_margin": 0.21
        },
        "market_data": {
            "volume": 50000000,
            "volatility": 0.25,
            "market_sentiment": 0.65
        }
    },
    "prediction_horizon": "1d",
    "model_name": "lstm_predictor"
}
```

**Parameters:**
- `symbol` (string, required): Stock ticker symbol
- `features` (object, required): Structured financial features
- `prediction_horizon` (string, optional): "1d", "1w", "1m" (default: "1d")
- `model_name` (string, optional): Model to use

**Response:**
```json
{
    "symbol": "AAPL",
    "current_price": 150.25,
    "predicted_price": 152.30,
    "price_change": 2.05,
    "price_change_percent": 1.36,
    "confidence": 0.92,
    "prediction_horizon": "1d",
    "model_name": "lstm_predictor",
    "timestamp": "2025-08-19T10:30:00Z",
    "factors": {
        "technical_sentiment": "bullish",
        "fundamental_strength": "strong",
        "market_conditions": "neutral"
    }
}
```

### Portfolio Optimization

```http
POST /optimize/portfolio
Content-Type: application/json
```

Portfolio optimization using ML models.

**Request Body:**
```json
{
    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
    "weights": [0.25, 0.25, 0.25, 0.25],
    "risk_tolerance": "moderate",
    "optimization_objective": "sharpe_ratio",
    "constraints": {
        "max_weight_per_asset": 0.40,
        "min_weight_per_asset": 0.05
    }
}
```

**Response:**
```json
{
    "optimized_weights": [0.30, 0.35, 0.20, 0.15],
    "expected_return": 0.12,
    "expected_risk": 0.18,
    "sharpe_ratio": 0.67,
    "recommendations": [
        {
            "symbol": "AAPL",
            "current_weight": 0.25,
            "optimal_weight": 0.30,
            "action": "increase",
            "reason": "Strong technical momentum and earnings outlook"
        }
    ],
    "timestamp": "2025-08-19T10:30:00Z"
}
```

---

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
    "error": "Model not found",
    "error_code": "MODEL_NOT_FOUND",
    "message": "The requested model 'invalid_model' is not available",
    "timestamp": "2025-08-19T10:30:00Z",
    "request_id": "req_abc123"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Invalid request data or parameters |
| `MODEL_NOT_FOUND` | 404 | Requested model does not exist |
| `MODEL_NOT_LOADED` | 404 | Model exists but is not loaded in memory |
| `PREDICTION_FAILED` | 500 | Error during model inference |
| `INSUFFICIENT_MEMORY` | 503 | Not enough memory to load model |
| `TRAINING_IN_PROGRESS` | 503 | Cannot perform action during training |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

### Rate Limiting Headers

Rate limiting information is included in response headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995  
X-RateLimit-Reset: 1692446400
X-RateLimit-Window: 60
```

---

## Integration Examples

### Python Client

```python
import requests
import json

class MLAPIClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
    
    def predict(self, features, model_name="lstm_predictor"):
        """Make a prediction"""
        url = f"{self.base_url}/predict"
        payload = {
            "features": features,
            "model_name": model_name
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_health(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self):
        """List available models"""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

# Usage example
client = MLAPIClient()

# Check health
health = client.get_health()
print(f"API Status: {health['status']}")

# Make prediction
features = [1.2, 0.8, 150.5, ...]  # 20 features for LSTM
prediction = client.predict(features, "lstm_predictor")
print(f"Prediction: {prediction['prediction']}")
```

### JavaScript/Node.js Client

```javascript
class MLAPIClient {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
    }
    
    async predict(features, modelName = 'lstm_predictor') {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: features,
                model_name: modelName
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async getHealth() {
        const response = await fetch(`${this.baseUrl}/health`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Usage example
const client = new MLAPIClient();

// Check health
client.getHealth()
    .then(health => console.log('API Status:', health.status))
    .catch(error => console.error('Error:', error));

// Make prediction
const features = [1.2, 0.8, 150.5, /* ... */]; // 20 features
client.predict(features, 'lstm_predictor')
    .then(prediction => console.log('Prediction:', prediction.prediction))
    .catch(error => console.error('Error:', error));
```

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:8001/health

# List models
curl -X GET http://localhost:8001/models

# Make prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.2, 0.8, 150.5, 0.95, -0.02, 0.15, 2.4, 0.78, 145.2, 0.88, 152.1, 149.8, 151.0, 0.012, 0.034, 0.056, 0.023, 0.45, 0.67, 0.89],
    "model_name": "lstm_predictor"
  }'

# Load model
curl -X POST http://localhost:8001/models/xgboost_classifier/load

# Trigger retraining
curl -X POST http://localhost:8001/retrain
```

---

## Performance Considerations

### Optimization Tips

1. **Model Loading**: Load models during startup or low-traffic periods
2. **Batch Predictions**: Use batch endpoints for multiple predictions
3. **Feature Caching**: Cache computed features for repeated predictions
4. **Model Quantization**: Use quantized models for faster inference
5. **Connection Pooling**: Reuse HTTP connections for multiple requests

### Performance Metrics

| Endpoint | Avg Response Time | Throughput |
|----------|------------------|------------|
| `/health` | <10ms | 5000 req/min |
| `/predict` | <100ms | 1000 req/min |
| `/predict/batch` | <200ms | 500 req/min |
| `/models` | <20ms | 2000 req/min |

### Scaling Recommendations

1. **Horizontal Scaling**: Run multiple API server instances
2. **Load Balancing**: Use nginx or cloud load balancers
3. **Caching**: Implement Redis for prediction caching
4. **Model Partitioning**: Distribute models across different servers
5. **Auto-scaling**: Use container orchestration for demand-based scaling

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-08-19 | Initial ML API implementation |

---

## Related Documentation

- [ML Pipeline Documentation](ML_PIPELINE_DOCUMENTATION.md)
- [ML Operations Guide](ML_OPERATIONS_GUIDE.md)
- [ML Quick Start Guide](ML_QUICKSTART.md)