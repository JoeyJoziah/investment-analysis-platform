#!/usr/bin/env python3
"""
ML Monitoring Server
Provides monitoring dashboard and metrics for ML models on port 8002
"""

import os
import sys
import json
import logging
import uvicorn
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import glob

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ML Monitoring API",
    description="Machine Learning Model Monitoring Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelStatus(BaseModel):
    name: str
    status: str
    last_prediction: Optional[str] = None
    predictions_count: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0

class SystemMetrics(BaseModel):
    timestamp: str
    models_loaded: int
    total_predictions: int
    error_rate: float
    avg_response_time: float
    uptime: str

class MonitoringDashboard:
    def __init__(self):
        self.ml_api_url = os.getenv('ML_API_URL', 'http://localhost:8001')
        self.logs_path = os.getenv('ML_LOGS_PATH', 'backend/ml_logs')
        self.models_path = os.getenv('ML_MODELS_PATH', 'backend/ml_models')
        
    def get_ml_api_status(self) -> Dict[str, Any]:
        """Get status from ML API"""
        try:
            response = requests.get(f"{self.ml_api_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    def get_model_files(self) -> List[str]:
        """Get list of model files"""
        try:
            models_dir = Path(self.models_path)
            if models_dir.exists():
                return [f.stem for f in models_dir.glob("*.pkl")]
            return []
        except Exception as e:
            logger.error(f"Error reading model files: {e}")
            return []
    
    def get_recent_logs(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        try:
            logs_dir = Path(self.logs_path)
            log_entries = []
            
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    try:
                        stat = log_file.stat()
                        modified_time = datetime.fromtimestamp(stat.st_mtime)
                        
                        if datetime.now() - modified_time < timedelta(hours=hours):
                            log_entries.append({
                                "file": log_file.name,
                                "modified": modified_time.isoformat(),
                                "size": stat.st_size
                            })
                    except Exception as e:
                        logger.warning(f"Error reading log file {log_file}: {e}")
            
            return sorted(log_entries, key=lambda x: x["modified"], reverse=True)
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return []
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history from metadata files"""
        try:
            logs_dir = Path(self.logs_path)
            training_history = []
            
            if logs_dir.exists():
                for metadata_file in logs_dir.glob("*_metadata.json"):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            training_history.append({
                                "model_name": metadata_file.stem.replace("_metadata", ""),
                                "timestamp": metadata.get("timestamp", "unknown"),
                                "model_type": metadata.get("model_type", "unknown"),
                                "score": metadata.get("score", 0.0),
                                "features": metadata.get("features", 0),
                                "samples": metadata.get("samples", 0)
                            })
                    except Exception as e:
                        logger.warning(f"Error reading metadata file {metadata_file}: {e}")
            
            return sorted(training_history, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.error(f"Error reading training history: {e}")
            return []

# Global dashboard instance
dashboard = MonitoringDashboard()

@app.get("/")
async def root():
    """Root endpoint with dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Monitoring Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .status-healthy { color: green; font-weight: bold; }
            .status-unhealthy { color: red; font-weight: bold; }
            .metric-box { border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>ML Monitoring Dashboard</h1>
        <div id="status"></div>
        <div id="metrics"></div>
        <div id="models"></div>
        <div id="logs"></div>
        
        <script>
            async function loadDashboard() {
                try {
                    const [status, metrics, models, logs] = await Promise.all([
                        fetch('/status').then(r => r.json()),
                        fetch('/metrics').then(r => r.json()),
                        fetch('/models/status').then(r => r.json()),
                        fetch('/logs/recent').then(r => r.json())
                    ]);
                    
                    document.getElementById('status').innerHTML = `
                        <div class="metric-box">
                            <h2>System Status</h2>
                            <p class="status-${status.api_status === 'healthy' ? 'healthy' : 'unhealthy'}">
                                API Status: ${status.api_status}
                            </p>
                            <p>Models Loaded: ${status.models_loaded}</p>
                            <p>Last Updated: ${new Date(status.timestamp).toLocaleString()}</p>
                        </div>
                    `;
                    
                    document.getElementById('models').innerHTML = `
                        <div class="metric-box">
                            <h2>Model Status</h2>
                            <table>
                                <tr><th>Model</th><th>Status</th><th>Type</th><th>Score</th></tr>
                                ${models.map(m => `
                                    <tr>
                                        <td>${m.name}</td>
                                        <td>${m.status}</td>
                                        <td>${m.type || 'N/A'}</td>
                                        <td>${m.score ? m.score.toFixed(3) : 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </table>
                        </div>
                    `;
                    
                    document.getElementById('logs').innerHTML = `
                        <div class="metric-box">
                            <h2>Recent Logs</h2>
                            <table>
                                <tr><th>File</th><th>Modified</th><th>Size</th></tr>
                                ${logs.map(l => `
                                    <tr>
                                        <td>${l.file}</td>
                                        <td>${new Date(l.modified).toLocaleString()}</td>
                                        <td>${(l.size / 1024).toFixed(1)} KB</td>
                                    </tr>
                                `).join('')}
                            </table>
                        </div>
                    `;
                    
                } catch (error) {
                    console.error('Error loading dashboard:', error);
                }
            }
            
            // Load dashboard on page load and refresh every 30 seconds
            loadDashboard();
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ml-monitoring"
    }

@app.get("/status")
async def get_status():
    """Get overall system status"""
    api_status = dashboard.get_ml_api_status()
    model_files = dashboard.get_model_files()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "api_status": api_status.get("status", "unknown"),
        "models_loaded": api_status.get("models_loaded", 0),
        "model_files": len(model_files),
        "ml_api_url": dashboard.ml_api_url
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    api_status = dashboard.get_ml_api_status()
    
    return SystemMetrics(
        timestamp=datetime.now().isoformat(),
        models_loaded=api_status.get("models_loaded", 0),
        total_predictions=0,  # Would need to track this
        error_rate=0.0,       # Would need to track this
        avg_response_time=0.0, # Would need to track this
        uptime="unknown"      # Would need to track this
    )

@app.get("/models/status")
async def get_models_status():
    """Get status of all models"""
    try:
        api_response = requests.get(f"{dashboard.ml_api_url}/models", timeout=5)
        if api_response.status_code == 200:
            models = api_response.json()
            return [
                {
                    "name": model["name"],
                    "status": "loaded",
                    "type": model["type"],
                    "score": model["score"],
                    "features": model["features"],
                    "loaded_at": model["loaded_at"]
                }
                for model in models
            ]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return []

@app.get("/logs/recent")
async def get_recent_logs():
    """Get recent log files"""
    return dashboard.get_recent_logs()

@app.get("/training/history")
async def get_training_history():
    """Get training history"""
    return dashboard.get_training_history()

if __name__ == "__main__":
    # Set environment
    os.environ["ML_LOGS_PATH"] = os.getenv("ML_LOGS_PATH", "backend/ml_logs")
    os.environ["ML_MODELS_PATH"] = os.getenv("ML_MODELS_PATH", "backend/ml_models")
    
    # Run server
    uvicorn.run(
        "ml_monitoring_server:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )