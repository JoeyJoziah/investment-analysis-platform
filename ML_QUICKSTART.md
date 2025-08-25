# ML Quick Start Guide

**Version**: 1.0.0  
**Last Updated**: August 19, 2025  
**Estimated Time**: 5-15 minutes

---

## 🚀 5-Minute Setup

Get the ML Pipeline running in minutes with this streamlined guide.

### Prerequisites

- Docker and Docker Compose installed
- 8GB+ RAM available
- 2GB+ free disk space
- Internet connection for downloading models

### Quick Start Commands

```bash
# 1. Clone and setup (if not already done)
git clone <repository-url>
cd investment-analysis-platform
./setup.sh

# 2. Start ML services
docker-compose up -d ml-api backend database redis

# 3. Verify ML API is running
curl http://localhost:8001/health

# 4. Make your first prediction!
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 0.8, 150.5, 0.95, -0.02], "model_name": "sample_model"}'
```

### Expected Output

If everything is working correctly, you should see:

```json
{
  "status": "healthy",
  "timestamp": "2025-08-19T10:30:00Z",
  "models_loaded": 1,
  "models": ["sample_model"]
}
```

**🎉 Congratulations!** Your ML Pipeline is now running.

---

## First Model Training

Train your first ML model with sample data in under 10 minutes.

### Step 1: Generate Training Data

```bash
# Generate sample financial data
python3 backend/ml/simple_training_pipeline.py --generate-data

# Verify data was created
ls -la data/training/
```

### Step 2: Train Your First Model

```bash
# Run the simplified training pipeline
python3 backend/ml/simple_training_pipeline.py --train

# This will:
# - Load sample data
# - Train an LSTM model
# - Save the model to backend/ml_models/
# - Generate performance metrics
```

### Step 3: Verify Model Training

```bash
# Check if model was created
ls -la backend/ml_models/

# Check training logs
tail -20 backend/ml_logs/simple_training_*.log

# Load the new model in API
curl -X POST http://localhost:8001/models/trained_model/load
```

### Training Output Example

```
2025-08-19 10:30:00 - INFO - Starting Simple ML Training Pipeline
2025-08-19 10:30:05 - INFO - Generated 5000 samples of training data
2025-08-19 10:30:30 - INFO - Training LSTM model...
2025-08-19 10:31:45 - INFO - Model training completed
2025-08-19 10:31:46 - INFO - Model saved to: backend/ml_models/trained_model.pkl
2025-08-19 10:31:47 - INFO - Training metrics: {'mse': 0.0145, 'mae': 0.087, 'r2': 0.854}
```

---

## Making Predictions

Learn how to use your trained models for predictions.

### Basic Prediction

```bash
# Simple prediction with sample data
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.2, 0.8, 150.5, 0.95, -0.02, 0.15, 2.4, 0.78, 145.2, 0.88, 152.1, 149.8, 151.0, 0.012, 0.034, 0.056, 0.023, 0.45, 0.67, 0.89],
    "model_name": "trained_model"
  }'
```

### Stock Price Prediction Example

```bash
# Stock-specific prediction with technical indicators
curl -X POST http://localhost:8001/predict/stock_price \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "features": {
      "technical_indicators": {
        "rsi_14": 65.5,
        "macd": 0.23,
        "sma_20": 150.1,
        "sma_50": 149.8,
        "volume": 50000000
      }
    },
    "prediction_horizon": "1d"
  }'
```

### Expected Prediction Response

```json
{
  "prediction": 152.30,
  "confidence": 0.92,
  "model_name": "trained_model",
  "timestamp": "2025-08-19T10:30:00Z",
  "factors": {
    "technical_sentiment": "bullish",
    "confidence_level": "high"
  }
}
```

---

## Common Use Cases

### Use Case 1: Daily Stock Analysis

Analyze a stock's potential for the next trading day.

```python
import requests
import json

def analyze_stock(symbol, current_price, volume, rsi, macd):
    """Analyze stock for next day trading"""
    
    # Prepare features (simplified example)
    features = [
        current_price / 100,  # Normalized price
        volume / 1000000,     # Volume in millions
        rsi / 100,            # RSI (0-1)
        macd,                 # MACD
        0.5,                  # Market sentiment placeholder
    ]
    
    # Add more features to match model requirements (20 features total)
    features.extend([0.0] * 15)  # Pad with zeros for demo
    
    # Make prediction
    response = requests.post('http://localhost:8001/predict', json={
        'features': features,
        'model_name': 'trained_model'
    })
    
    if response.status_code == 200:
        prediction = response.json()
        predicted_price = prediction['prediction'] * 100  # Denormalize
        confidence = prediction['confidence']
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': predicted_price - current_price,
            'confidence': confidence,
            'recommendation': 'BUY' if predicted_price > current_price else 'SELL'
        }
    else:
        return {'error': f'Prediction failed: {response.text}'}

# Example usage
result = analyze_stock('AAPL', 150.25, 45000000, 65.5, 0.23)
print(json.dumps(result, indent=2))
```

### Use Case 2: Portfolio Optimization

Optimize a simple portfolio using ML predictions.

```python
def optimize_portfolio(symbols, current_weights):
    """Basic portfolio optimization using ML predictions"""
    
    predictions = {}
    
    # Get predictions for each stock
    for symbol in symbols:
        # Simplified feature generation
        features = [1.0, 0.5, 0.8] + [0.0] * 17  # 20 features total
        
        response = requests.post('http://localhost:8001/predict', json={
            'features': features,
            'model_name': 'trained_model'
        })
        
        if response.status_code == 200:
            pred = response.json()
            predictions[symbol] = {
                'return': pred['prediction'],
                'confidence': pred['confidence']
            }
    
    # Simple optimization: weight by predicted return * confidence
    total_score = sum(p['return'] * p['confidence'] for p in predictions.values())
    
    optimized_weights = {}
    for symbol, pred in predictions.items():
        score = pred['return'] * pred['confidence']
        optimized_weights[symbol] = score / total_score
    
    return optimized_weights

# Example usage
portfolio = optimize_portfolio(['AAPL', 'GOOGL', 'MSFT'], [0.33, 0.33, 0.34])
print("Optimized Portfolio Weights:")
for symbol, weight in portfolio.items():
    print(f"{symbol}: {weight:.2%}")
```

### Use Case 3: Automated Trading Signal

Generate buy/sell signals automatically.

```python
def generate_trading_signals(watchlist):
    """Generate trading signals for a watchlist"""
    
    signals = []
    
    for symbol in watchlist:
        # Mock current market data
        current_data = {
            'price': 150.0,
            'volume': 45000000,
            'rsi': 65.5,
            'macd': 0.23,
            'sma_ratio': 1.02  # Price vs SMA-20
        }
        
        # Convert to model features
        features = [
            current_data['price'] / 100,
            current_data['volume'] / 10000000,
            current_data['rsi'] / 100,
            current_data['macd'],
            current_data['sma_ratio']
        ]
        features.extend([0.0] * 15)  # Pad to 20 features
        
        # Get prediction
        response = requests.post('http://localhost:8001/predict', json={
            'features': features,
            'model_name': 'trained_model'
        })
        
        if response.status_code == 200:
            prediction = response.json()
            predicted_return = prediction['prediction']
            confidence = prediction['confidence']
            
            # Generate signal based on prediction and confidence
            if predicted_return > 0.02 and confidence > 0.8:
                signal = 'STRONG_BUY'
            elif predicted_return > 0.01 and confidence > 0.7:
                signal = 'BUY'
            elif predicted_return < -0.02 and confidence > 0.8:
                signal = 'STRONG_SELL'
            elif predicted_return < -0.01 and confidence > 0.7:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            signals.append({
                'symbol': symbol,
                'signal': signal,
                'predicted_return': predicted_return,
                'confidence': confidence,
                'timestamp': prediction['timestamp']
            })
    
    return signals

# Example usage
watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
signals = generate_trading_signals(watchlist)

print("Trading Signals:")
for signal in signals:
    print(f"{signal['symbol']}: {signal['signal']} "
          f"(Return: {signal['predicted_return']:.2%}, "
          f"Confidence: {signal['confidence']:.1%})")
```

---

## Frequently Asked Questions (FAQ)

### Q: How do I add more training data?
**A:** Place your CSV files in the `data/training/` directory. The pipeline will automatically detect and use them. Format: `timestamp,symbol,open,high,low,close,volume,returns`.

### Q: Can I use my own ML models?
**A:** Yes! Save your trained model as a `.pkl` file in `backend/ml_models/` and use the model management API to load it:
```bash
curl -X POST http://localhost:8001/models/my_model/load
```

### Q: How do I improve model accuracy?
**A:** Try these approaches:
1. Add more training data
2. Include additional features (technical indicators, fundamental data)
3. Tune hyperparameters
4. Use ensemble methods
5. Increase model complexity (more layers, neurons)

### Q: What if the ML API is slow?
**A:** Optimize performance by:
1. Reducing model complexity
2. Using model quantization
3. Enabling batch predictions
4. Adding more RAM
5. Using GPU acceleration

### Q: How do I deploy to production?
**A:** Use the production configuration:
```bash
# Start production environment
./start.sh prod

# Or use Docker Compose directly
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Q: Can I train models automatically?
**A:** Yes! Enable auto-retraining in your configuration:
```bash
# Set environment variable
export ENABLE_AUTO_RETRAINING=true

# Or trigger manually
curl -X POST http://localhost:8001/retrain
```

### Q: How do I monitor model performance?
**A:** Check the monitoring endpoints:
```bash
# Model health
curl http://localhost:8001/health

# Model metrics
curl http://localhost:8001/models/trained_model/info

# View training logs
tail -f backend/ml_logs/training_*.log
```

### Q: What's the minimum hardware requirement?
**A:** For development:
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum
- **Storage**: 10GB free space
- **GPU**: Optional but recommended for training

### Q: How do I backup my models?
**A:** Models are automatically saved to `backend/ml_models/`. For backup:
```bash
# Backup models
cp -r backend/ml_models/ /backup/location/

# Backup with date
tar -czf ml_models_$(date +%Y%m%d).tar.gz backend/ml_models/
```

---

## Next Steps

### Intermediate Level
1. **Custom Features**: Add your own technical indicators
2. **Model Tuning**: Optimize hyperparameters
3. **Data Integration**: Connect to real market data APIs
4. **Portfolio Management**: Implement portfolio optimization algorithms

### Advanced Level  
1. **Ensemble Models**: Combine multiple models for better accuracy
2. **Real-time Processing**: Implement streaming data processing
3. **Risk Management**: Add position sizing and risk controls
4. **Production Deployment**: Deploy on cloud platforms

### Learning Resources
- [ML Pipeline Documentation](ML_PIPELINE_DOCUMENTATION.md) - Complete technical reference
- [ML API Reference](ML_API_REFERENCE.md) - Full API documentation
- [ML Operations Guide](ML_OPERATIONS_GUIDE.md) - Production operations
- [CLAUDE.md](CLAUDE.md) - Project guidelines

---

## Troubleshooting Quick Fixes

### Issue: "Connection refused" to ML API
```bash
# Check if container is running
docker-compose ps ml-api

# Restart if needed
docker-compose restart ml-api
```

### Issue: "Model not found" error
```bash
# List available models
curl http://localhost:8001/models

# Train a model if none exist
python3 backend/ml/simple_training_pipeline.py --train
```

### Issue: Out of memory during training
```bash
# Reduce training data size
export TRAINING_SAMPLE_SIZE=1000

# Or increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory > 8GB+
```

### Issue: Predictions seem random
```bash
# Check model training metrics
tail -20 backend/ml_logs/training_*.log

# Retrain with more data
python3 backend/ml/simple_training_pipeline.py --train --samples=10000
```

### Still Having Issues?
1. Check the [ML Operations Guide](ML_OPERATIONS_GUIDE.md) troubleshooting section
2. Review logs: `docker-compose logs ml-api`
3. Verify your environment meets the prerequisites
4. Try the Docker reset: `docker-compose down && docker-compose up -d`

---

**🎯 You're now ready to build AI-powered investment strategies!**

For more advanced features and production deployment, continue with the comprehensive documentation.