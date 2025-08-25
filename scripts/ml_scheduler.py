#!/usr/bin/env python3
"""
ML Training Scheduler
Automated scheduling for ML model training and retraining
"""

import os
import sys
import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
import threading
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend/ml_logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLScheduler:
    """ML Training Scheduler"""
    
    def __init__(self):
        self.config = self._load_config()
        self.is_running = False
        
    def _load_config(self):
        """Load scheduler configuration"""
        return {
            'daily_training_time': os.getenv('ML_DAILY_TRAINING_TIME', '02:00'),
            'enable_daily_training': os.getenv('ENABLE_DAILY_TRAINING', 'true').lower() == 'true',
            'enable_hourly_monitoring': os.getenv('ENABLE_HOURLY_MONITORING', 'true').lower() == 'true',
            'training_script': os.getenv('ML_TRAINING_SCRIPT', 'backend/ml/minimal_training.py'),
            'max_training_duration': int(os.getenv('MAX_TRAINING_DURATION_MINUTES', '30')),
        }
    
    def run_training(self):
        """Execute training pipeline"""
        logger.info("Starting scheduled training...")
        
        try:
            import subprocess
            result = subprocess.run(
                ['python3', self.config['training_script']], 
                capture_output=True, 
                text=True,
                timeout=self.config['max_training_duration'] * 60
            )
            
            if result.returncode == 0:
                logger.info("Training completed successfully")
                logger.info(f"Training output: {result.stdout}")
            else:
                logger.error(f"Training failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out after {self.config['max_training_duration']} minutes")
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    def run_monitoring(self):
        """Execute monitoring checks"""
        logger.info("Running model monitoring...")
        
        try:
            # Check model files
            models_dir = Path('backend/ml_models')
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pkl'))
                logger.info(f"Found {len(model_files)} model files")
                
                # Log model status
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'models_count': len(model_files),
                    'model_files': [str(f) for f in model_files],
                    'status': 'healthy'
                }
                
                status_file = Path('backend/ml_logs/monitoring_status.json')
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)
                    
            else:
                logger.warning("Models directory not found")
                
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def setup_schedules(self):
        """Setup training and monitoring schedules"""
        logger.info("Setting up ML schedules...")
        
        if self.config['enable_daily_training']:
            schedule.every().day.at(self.config['daily_training_time']).do(self.run_training)
            logger.info(f"Daily training scheduled at {self.config['daily_training_time']}")
        
        if self.config['enable_hourly_monitoring']:
            schedule.every().hour.do(self.run_monitoring)
            logger.info("Hourly monitoring scheduled")
        
        # Initial monitoring run
        self.run_monitoring()
    
    def start(self):
        """Start the scheduler"""
        logger.info("Starting ML Scheduler...")
        self.is_running = True
        self.setup_schedules()
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping ML Scheduler...")
        self.is_running = False

def run_scheduler_daemon():
    """Run scheduler as daemon"""
    scheduler = MLScheduler()
    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.stop()
        logger.info("Scheduler stopped by user")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        # Run as daemon
        run_scheduler_daemon()
    else:
        # Run single training
        scheduler = MLScheduler()
        scheduler.run_training()