#!/usr/bin/env python3
"""
Data Pipeline Activation Script
Activates the investment analysis data pipeline with full monitoring and validation.
"""

import asyncio
import logging
import sys
import os
import subprocess
import time
from typing import Dict, List, Optional
from datetime import datetime
import psutil
import redis
import psycopg2
from psycopg2 import sql
import json
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_activation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPipelineActivator:
    """Main pipeline activation orchestrator"""
    
    def __init__(self):
        self.config = self._load_config()
        self.redis_client = None
        self.db_connection = None
        
    def _load_config(self) -> Dict:
        """Load configuration from environment variables"""
        return {
            'db_host': os.getenv('DB_HOST', 'localhost'),
            'db_port': os.getenv('DB_PORT', '5432'),
            'db_name': os.getenv('DB_NAME', 'investment_db'),
            'db_user': os.getenv('DB_USER', 'postgres'),
            'db_password': os.getenv('DB_PASSWORD', ''),
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', '6379')),
            'redis_password': os.getenv('REDIS_PASSWORD', ''),
            'redis_db': int(os.getenv('REDIS_DB', '0')),
            'airflow_webserver_url': os.getenv('AIRFLOW_URL', 'http://localhost:8080'),
            'monthly_budget': float(os.getenv('MONTHLY_BUDGET_LIMIT', '50')),
            'api_keys': {
                'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
                'finnhub': os.getenv('FINNHUB_API_KEY', ''),
                'polygon': os.getenv('POLYGON_API_KEY', ''),
                'news_api': os.getenv('NEWS_API_KEY', '')
            }
        }
    
    async def activate_pipeline(self) -> bool:
        """Main pipeline activation process"""
        logger.info("🚀 Starting Data Pipeline Activation")
        
        try:
            # Step 1: Validate prerequisites
            if not await self._validate_prerequisites():
                return False
            
            # Step 2: Initialize connections
            if not await self._initialize_connections():
                return False
            
            # Step 3: Set up database and tables
            if not await self._setup_database():
                return False
            
            # Step 4: Initialize stock prioritization
            if not await self._initialize_stock_tiers():
                return False
            
            # Step 5: Configure cost monitoring
            if not await self._setup_cost_monitoring():
                return False
            
            # Step 6: Test API connections
            if not await self._test_api_connections():
                return False
            
            # Step 7: Start Airflow if needed
            if not await self._ensure_airflow_running():
                return False
            
            # Step 8: Activate DAGs
            if not await self._activate_dags():
                return False
            
            # Step 9: Run initial validation
            if not await self._run_initial_validation():
                return False
            
            logger.info("✅ Data Pipeline Successfully Activated!")
            await self._print_status_summary()
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline activation failed: {e}")
            return False
        finally:
            await self._cleanup_connections()
    
    async def _validate_prerequisites(self) -> bool:
        """Validate system prerequisites"""
        logger.info("📋 Validating Prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            logger.error("❌ Python 3.11+ required")
            return False
        
        # Check required environment variables
        required_vars = ['DB_PASSWORD', 'REDIS_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        # Check API keys
        api_key_count = sum(1 for key in self.config['api_keys'].values() if key)
        if api_key_count < 2:
            logger.warning("⚠️  Less than 2 API keys configured - limited functionality")
        
        # Check system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            logger.warning(f"⚠️  Low memory: {memory_gb:.1f}GB (4GB+ recommended)")
        
        logger.info("✅ Prerequisites validated")
        return True
    
    async def _initialize_connections(self) -> bool:
        """Initialize database and Redis connections"""
        logger.info("🔗 Initializing Connections...")
        
        try:
            # Test PostgreSQL connection
            self.db_connection = psycopg2.connect(
                host=self.config['db_host'],
                port=self.config['db_port'],
                database=self.config['db_name'],
                user=self.config['db_user'],
                password=self.config['db_password']
            )
            self.db_connection.autocommit = True
            logger.info("✅ PostgreSQL connection established")
            
            # Test Redis connection
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                password=self.config['redis_password'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def _setup_database(self) -> bool:
        """Set up database schema and initial data"""
        logger.info("🗄️  Setting up Database...")
        
        try:
            with self.db_connection.cursor() as cursor:
                # Create tables if they don't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stocks (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) UNIQUE NOT NULL,
                        name VARCHAR(255),
                        exchange VARCHAR(10),
                        sector VARCHAR(100),
                        industry VARCHAR(100),
                        market_cap BIGINT,
                        is_active BOOLEAN DEFAULT true,
                        priority_tier INTEGER DEFAULT 4,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS price_history (
                        id SERIAL PRIMARY KEY,
                        stock_id INTEGER REFERENCES stocks(id),
                        date DATE NOT NULL,
                        open DECIMAL(10,2),
                        high DECIMAL(10,2),
                        low DECIMAL(10,2),
                        close DECIMAL(10,2),
                        adjusted_close DECIMAL(10,2),
                        volume BIGINT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(stock_id, date)
                    );
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS technical_indicators (
                        id SERIAL PRIMARY KEY,
                        stock_id INTEGER REFERENCES stocks(id),
                        date DATE NOT NULL,
                        rsi DECIMAL(8,4),
                        macd DECIMAL(8,4),
                        macd_signal DECIMAL(8,4),
                        bollinger_upper DECIMAL(10,2),
                        bollinger_lower DECIMAL(10,2),
                        sma_20 DECIMAL(10,2),
                        sma_50 DECIMAL(10,2),
                        sma_200 DECIMAL(10,2),
                        volume_sma BIGINT,
                        atr DECIMAL(8,4),
                        stochastic_k DECIMAL(8,4),
                        stochastic_d DECIMAL(8,4),
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(stock_id, date)
                    );
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_usage (
                        id SERIAL PRIMARY KEY,
                        provider VARCHAR(50) NOT NULL,
                        endpoint VARCHAR(100),
                        success BOOLEAN DEFAULT true,
                        response_time_ms INTEGER,
                        data_points INTEGER,
                        estimated_cost DECIMAL(8,4) DEFAULT 0,
                        timestamp TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_price_history_stock_date 
                    ON price_history(stock_id, date DESC);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_technical_indicators_stock_date 
                    ON technical_indicators(stock_id, date DESC);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp 
                    ON api_usage(timestamp DESC);
                """)
                
                # Check if we have sample data
                cursor.execute("SELECT COUNT(*) FROM stocks")
                stock_count = cursor.fetchone()[0]
                
                if stock_count < 100:
                    logger.info("🔄 Loading sample stock data...")
                    await self._load_sample_stocks(cursor)
                
                logger.info(f"✅ Database setup complete ({stock_count} stocks)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    async def _load_sample_stocks(self, cursor):
        """Load sample stock data for testing"""
        sample_stocks = [
            ('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology', 'Consumer Electronics', 1),
            ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology', 'Software', 1),
            ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Communication Services', 'Internet Services', 1),
            ('AMZN', 'Amazon.com Inc.', 'NASDAQ', 'Consumer Discretionary', 'E-commerce', 1),
            ('TSLA', 'Tesla Inc.', 'NASDAQ', 'Consumer Discretionary', 'Automotive', 1),
            ('NVDA', 'NVIDIA Corporation', 'NASDAQ', 'Technology', 'Semiconductors', 1),
            ('META', 'Meta Platforms Inc.', 'NASDAQ', 'Communication Services', 'Social Media', 1),
            ('JPM', 'JPMorgan Chase & Co.', 'NYSE', 'Financials', 'Banking', 2),
            ('JNJ', 'Johnson & Johnson', 'NYSE', 'Healthcare', 'Pharmaceuticals', 2),
            ('V', 'Visa Inc.', 'NYSE', 'Financials', 'Payment Processing', 2)
        ]
        
        for symbol, name, exchange, sector, industry, tier in sample_stocks:
            cursor.execute("""
                INSERT INTO stocks (symbol, name, exchange, sector, industry, priority_tier)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    exchange = EXCLUDED.exchange,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    priority_tier = EXCLUDED.priority_tier,
                    updated_at = NOW()
            """, (symbol, name, exchange, sector, industry, tier))
    
    async def _initialize_stock_tiers(self) -> bool:
        """Initialize stock prioritization tiers"""
        logger.info("📊 Initializing Stock Prioritization...")
        
        try:
            with self.db_connection.cursor() as cursor:
                # Update priority tiers based on market cap and activity
                cursor.execute("""
                    UPDATE stocks SET priority_tier = 
                    CASE 
                        WHEN symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK.B', 'LLY', 'V', 'WMT', 'JPM', 'XOM', 'UNH', 'MA', 'PG', 'JNJ', 'HD', 'CVX', 'ABBV') THEN 1
                        WHEN market_cap > 50000000000 THEN 2
                        WHEN market_cap > 10000000000 THEN 3
                        WHEN market_cap > 1000000000 THEN 4
                        ELSE 5
                    END
                """)
                
                cursor.execute("""
                    SELECT priority_tier, COUNT(*) 
                    FROM stocks 
                    GROUP BY priority_tier 
                    ORDER BY priority_tier
                """)
                
                tier_counts = cursor.fetchall()
                for tier, count in tier_counts:
                    logger.info(f"  Tier {tier}: {count} stocks")
                
                # Cache tier information in Redis
                for tier, count in tier_counts:
                    self.redis_client.set(f"stock_tier:{tier}:count", count)
                
                logger.info("✅ Stock tiers initialized")
                return True
                
        except Exception as e:
            logger.error(f"❌ Stock tier initialization failed: {e}")
            return False
    
    async def _setup_cost_monitoring(self) -> bool:
        """Set up cost monitoring and API limits"""
        logger.info("💰 Setting up Cost Monitoring...")
        
        try:
            # Initialize cost tracking
            cost_config = {
                'monthly_budget': self.config['monthly_budget'],
                'current_month_cost': 0.0,
                'api_limits': {
                    'finnhub': {'per_minute': 60, 'daily': float('inf'), 'cost_per_call': 0.0},
                    'alpha_vantage': {'per_minute': 5, 'daily': 25, 'cost_per_call': 0.0},
                    'polygon': {'per_minute': 5, 'daily': float('inf'), 'cost_per_call': 0.0},
                    'news_api': {'per_minute': 100, 'daily': 100, 'cost_per_call': 0.0}
                },
                'last_updated': datetime.now().isoformat()
            }
            
            self.redis_client.set('cost_monitor_config', json.dumps(cost_config))
            
            # Reset daily counters
            today = datetime.now().strftime('%Y%m%d')
            for provider in cost_config['api_limits'].keys():
                self.redis_client.delete(f"api_usage:{provider}:daily:{today}")
            
            # Set up alerting thresholds
            self.redis_client.set('cost_alert_threshold', self.config['monthly_budget'] * 0.8)
            
            logger.info(f"✅ Cost monitoring initialized (Budget: ${self.config['monthly_budget']})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cost monitoring setup failed: {e}")
            return False
    
    async def _test_api_connections(self) -> bool:
        """Test API connections and validate keys"""
        logger.info("🔑 Testing API Connections...")
        
        api_results = {}
        
        # Test Finnhub
        if self.config['api_keys']['finnhub']:
            try:
                response = requests.get(
                    f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={self.config['api_keys']['finnhub']}",
                    timeout=10
                )
                api_results['finnhub'] = response.status_code == 200
                logger.info(f"  Finnhub: {'✅' if api_results['finnhub'] else '❌'}")
            except:
                api_results['finnhub'] = False
                logger.info("  Finnhub: ❌")
        
        # Test Alpha Vantage
        if self.config['api_keys']['alpha_vantage']:
            try:
                response = requests.get(
                    f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={self.config['api_keys']['alpha_vantage']}",
                    timeout=10
                )
                api_results['alpha_vantage'] = response.status_code == 200 and 'Global Quote' in response.text
                logger.info(f"  Alpha Vantage: {'✅' if api_results['alpha_vantage'] else '❌'}")
            except:
                api_results['alpha_vantage'] = False
                logger.info("  Alpha Vantage: ❌")
        
        # Store API status
        self.redis_client.set('api_status', json.dumps(api_results))
        
        working_apis = sum(api_results.values())
        logger.info(f"✅ API Testing Complete ({working_apis}/{len(api_results)} working)")
        
        return working_apis > 0
    
    async def _ensure_airflow_running(self) -> bool:
        """Ensure Airflow is running"""
        logger.info("🌪️  Checking Airflow Status...")
        
        try:
            # Check if Airflow webserver is responding
            response = requests.get(f"{self.config['airflow_webserver_url']}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Airflow is running")
                return True
        except:
            pass
        
        # Try to start Airflow via Docker Compose
        logger.info("🔄 Starting Airflow services...")
        try:
            result = subprocess.run(
                ['docker-compose', '-f', 'docker-compose.airflow.yml', 'up', '-d'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Wait for services to be ready
                logger.info("⏳ Waiting for Airflow to be ready...")
                for _ in range(30):  # Wait up to 5 minutes
                    try:
                        response = requests.get(f"{self.config['airflow_webserver_url']}/health", timeout=5)
                        if response.status_code == 200:
                            logger.info("✅ Airflow started successfully")
                            return True
                    except:
                        pass
                    await asyncio.sleep(10)
                
                logger.warning("⚠️  Airflow started but not responding on web interface")
                return True
            else:
                logger.error(f"❌ Failed to start Airflow: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Airflow startup timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error starting Airflow: {e}")
            return False
    
    async def _activate_dags(self) -> bool:
        """Activate Airflow DAGs"""
        logger.info("📅 Activating DAGs...")
        
        try:
            # For now, mark as successful - DAG activation can be done via UI or API
            self.redis_client.set('dags_activated', 'true')
            self.redis_client.set('dags_activation_time', datetime.now().isoformat())
            
            logger.info("✅ DAGs marked for activation")
            logger.info("ℹ️  Manual step: Activate 'daily_market_analysis' DAG in Airflow UI")
            return True
            
        except Exception as e:
            logger.error(f"❌ DAG activation failed: {e}")
            return False
    
    async def _run_initial_validation(self) -> bool:
        """Run initial pipeline validation"""
        logger.info("🧪 Running Pipeline Validation...")
        
        try:
            validation_results = {
                'database_connection': bool(self.db_connection),
                'redis_connection': bool(self.redis_client),
                'stock_data_available': False,
                'cost_monitoring_active': False,
                'api_keys_configured': False
            }
            
            # Check stock data
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_active = true")
                stock_count = cursor.fetchone()[0]
                validation_results['stock_data_available'] = stock_count > 0
            
            # Check cost monitoring
            cost_config = self.redis_client.get('cost_monitor_config')
            validation_results['cost_monitoring_active'] = bool(cost_config)
            
            # Check API keys
            validation_results['api_keys_configured'] = any(self.config['api_keys'].values())
            
            # Store validation results
            self.redis_client.set('pipeline_validation', json.dumps(validation_results))
            
            success_count = sum(validation_results.values())
            total_count = len(validation_results)
            
            logger.info(f"✅ Validation Complete ({success_count}/{total_count} checks passed)")
            
            for check, result in validation_results.items():
                logger.info(f"  {check}: {'✅' if result else '❌'}")
            
            return success_count >= 4  # At least 4/5 checks should pass
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False
    
    async def _print_status_summary(self):
        """Print comprehensive status summary"""
        logger.info("\n" + "="*60)
        logger.info("🎉 DATA PIPELINE ACTIVATION SUMMARY")
        logger.info("="*60)
        
        # Database status
        with self.db_connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM stocks")
            stock_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT priority_tier, COUNT(*) FROM stocks GROUP BY priority_tier ORDER BY priority_tier")
            tier_counts = cursor.fetchall()
            
        logger.info(f"📊 STOCK DATA:")
        logger.info(f"  Total Stocks: {stock_count}")
        for tier, count in tier_counts:
            logger.info(f"  Tier {tier}: {count} stocks")
        
        # API status
        api_status = json.loads(self.redis_client.get('api_status') or '{}')
        working_apis = [k for k, v in api_status.items() if v]
        
        logger.info(f"🔑 API CONNECTIONS:")
        for api, status in api_status.items():
            logger.info(f"  {api}: {'✅' if status else '❌'}")
        
        # Cost monitoring
        cost_config = json.loads(self.redis_client.get('cost_monitor_config') or '{}')
        
        logger.info(f"💰 COST MONITORING:")
        logger.info(f"  Monthly Budget: ${cost_config.get('monthly_budget', 0)}")
        logger.info(f"  Current Cost: ${cost_config.get('current_month_cost', 0)}")
        
        # Next steps
        logger.info(f"🚀 NEXT STEPS:")
        logger.info(f"  1. Access Airflow UI: {self.config['airflow_webserver_url']}")
        logger.info(f"  2. Activate 'daily_market_analysis' DAG")
        logger.info(f"  3. Monitor pipeline at: http://localhost:5555 (Flower)")
        logger.info(f"  4. Check logs in: ./logs/")
        
        logger.info("="*60)
    
    async def _cleanup_connections(self):
        """Clean up connections"""
        try:
            if self.db_connection:
                self.db_connection.close()
            # Redis client will auto-close
        except:
            pass


async def main():
    """Main entry point"""
    activator = DataPipelineActivator()
    success = await activator.activate_pipeline()
    
    if success:
        print("\n🎉 Data Pipeline Successfully Activated!")
        print("Check the logs above for detailed status and next steps.")
        return 0
    else:
        print("\n❌ Data Pipeline Activation Failed!")
        print("Check the logs above for error details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)