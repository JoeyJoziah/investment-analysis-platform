#!/usr/bin/env python3
"""
Data Pipeline Monitoring Script
Real-time monitoring of the investment analysis data pipeline.
"""

import asyncio
import logging
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil
import redis
import psycopg2
import requests
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Comprehensive pipeline monitoring system"""
    
    def __init__(self):
        self.config = self._load_config()
        self.redis_client = None
        self.db_connection = None
        self.monitoring = True
        
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
            'airflow_url': os.getenv('AIRFLOW_URL', 'http://localhost:8080'),
            'flower_url': 'http://localhost:5555',
            'monthly_budget': float(os.getenv('MONTHLY_BUDGET_LIMIT', '50'))
        }
    
    async def start_monitoring(self, refresh_interval: int = 30):
        """Start continuous pipeline monitoring"""
        logger.info("📊 Starting Pipeline Monitoring")
        logger.info(f"🔄 Refresh interval: {refresh_interval} seconds")
        logger.info("📋 Press Ctrl+C to stop monitoring")
        
        # Initialize connections
        await self._initialize_connections()
        
        try:
            while self.monitoring:
                # Clear screen for dashboard effect
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Print dashboard header
                print("="*80)
                print("🎯 INVESTMENT ANALYSIS PIPELINE - LIVE DASHBOARD")
                print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                
                # Get all monitoring data
                monitoring_data = await self._collect_monitoring_data()
                
                # Display dashboard sections
                await self._display_system_status(monitoring_data)
                await self._display_pipeline_status(monitoring_data)
                await self._display_api_status(monitoring_data)
                await self._display_cost_monitoring(monitoring_data)
                await self._display_data_status(monitoring_data)
                await self._display_performance_metrics(monitoring_data)
                await self._display_alerts(monitoring_data)
                
                print("="*80)
                print(f"🔄 Next update in {refresh_interval} seconds | Press Ctrl+C to exit")
                print("="*80)
                
                # Wait for next refresh
                await asyncio.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            await self._cleanup_connections()
    
    async def _initialize_connections(self):
        """Initialize database and Redis connections"""
        try:
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                password=self.config['redis_password'],
                db=self.config['redis_db'],
                decode_responses=True,
                socket_timeout=5
            )
            
            # Database connection
            self.db_connection = psycopg2.connect(
                host=self.config['db_host'],
                port=self.config['db_port'],
                database=self.config['db_name'],
                user=self.config['db_user'],
                password=self.config['db_password'],
                connect_timeout=10
            )
            
        except Exception as e:
            logger.error(f"Connection initialization failed: {e}")
            raise
    
    async def _collect_monitoring_data(self) -> Dict:
        """Collect all monitoring data"""
        data = {
            'timestamp': datetime.now(),
            'system': await self._get_system_metrics(),
            'services': await self._get_service_status(),
            'database': await self._get_database_metrics(),
            'api': await self._get_api_metrics(),
            'cost': await self._get_cost_metrics(),
            'pipeline': await self._get_pipeline_metrics(),
            'alerts': await self._get_active_alerts()
        }
        return data
    
    async def _get_system_metrics(self) -> Dict:
        """Get system resource metrics"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"System metrics error: {e}")
            return {}
    
    async def _get_service_status(self) -> Dict:
        """Check status of all services"""
        services = {}
        
        # PostgreSQL
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                services['postgresql'] = {'status': 'online', 'response_time_ms': 50}
        except:
            services['postgresql'] = {'status': 'offline', 'response_time_ms': None}
        
        # Redis
        try:
            start_time = time.time()
            self.redis_client.ping()
            response_time = int((time.time() - start_time) * 1000)
            services['redis'] = {'status': 'online', 'response_time_ms': response_time}
        except:
            services['redis'] = {'status': 'offline', 'response_time_ms': None}
        
        # Airflow
        try:
            start_time = time.time()
            response = requests.get(f"{self.config['airflow_url']}/health", timeout=5)
            response_time = int((time.time() - start_time) * 1000)
            status = 'online' if response.status_code == 200 else 'degraded'
            services['airflow'] = {'status': status, 'response_time_ms': response_time}
        except:
            services['airflow'] = {'status': 'offline', 'response_time_ms': None}
        
        # Flower (Celery monitoring)
        try:
            start_time = time.time()
            response = requests.get(f"{self.config['flower_url']}/api/workers", timeout=5)
            response_time = int((time.time() - start_time) * 1000)
            status = 'online' if response.status_code == 200 else 'degraded'
            services['flower'] = {'status': status, 'response_time_ms': response_time}
        except:
            services['flower'] = {'status': 'offline', 'response_time_ms': None}
        
        return services
    
    async def _get_database_metrics(self) -> Dict:
        """Get database performance metrics"""
        try:
            with self.db_connection.cursor() as cursor:
                # Connection count
                cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                active_connections = cursor.fetchone()[0]
                
                # Database size
                cursor.execute(f"SELECT pg_size_pretty(pg_database_size('{self.config['db_name']}'))")
                db_size = cursor.fetchone()[0]
                
                # Recent activity
                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as hourly_inserts,
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 day') as daily_inserts
                    FROM price_history
                """)
                result = cursor.fetchone()
                hourly_inserts, daily_inserts = result if result else (0, 0)
                
                # Table sizes
                cursor.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as size,
                        n_tup_ins as inserts,
                        n_tup_upd as updates
                    FROM pg_stat_user_tables 
                    ORDER BY pg_relation_size(schemaname||'.'||tablename) DESC 
                    LIMIT 5
                """)
                table_stats = cursor.fetchall()
                
                return {
                    'active_connections': active_connections,
                    'database_size': db_size,
                    'hourly_inserts': hourly_inserts,
                    'daily_inserts': daily_inserts,
                    'table_stats': table_stats
                }
                
        except Exception as e:
            logger.error(f"Database metrics error: {e}")
            return {}
    
    async def _get_api_metrics(self) -> Dict:
        """Get API usage metrics"""
        try:
            today = datetime.now().strftime('%Y%m%d')
            api_metrics = {}
            
            providers = ['finnhub', 'alpha_vantage', 'polygon', 'news_api']
            
            for provider in providers:
                # Daily usage
                daily_key = f"api_usage:{provider}:daily:{today}"
                daily_count = int(self.redis_client.get(daily_key) or 0)
                
                # Current minute usage
                current_minute = datetime.now().strftime('%Y%m%d%H%M')
                minute_key = f"api_usage:{provider}:minute:{current_minute}"
                minute_count = int(self.redis_client.get(minute_key) or 0)
                
                api_metrics[provider] = {
                    'daily_calls': daily_count,
                    'current_minute_calls': minute_count,
                    'status': 'active' if daily_count > 0 else 'inactive'
                }
            
            # Get API limits from cost monitor config
            cost_config = self.redis_client.get('cost_monitor_config')
            if cost_config:
                limits = json.loads(cost_config).get('api_limits', {})
                for provider in api_metrics:
                    if provider in limits:
                        api_metrics[provider]['daily_limit'] = limits[provider].get('daily', float('inf'))
                        api_metrics[provider]['minute_limit'] = limits[provider].get('per_minute', float('inf'))
            
            return api_metrics
            
        except Exception as e:
            logger.error(f"API metrics error: {e}")
            return {}
    
    async def _get_cost_metrics(self) -> Dict:
        """Get cost monitoring metrics"""
        try:
            # Current month cost tracking
            current_month = datetime.now().strftime('%Y%m')
            
            # Get cost config
            cost_config = self.redis_client.get('cost_monitor_config')
            if cost_config:
                config = json.loads(cost_config)
                monthly_budget = config.get('monthly_budget', self.config['monthly_budget'])
                current_cost = config.get('current_month_cost', 0)
            else:
                monthly_budget = self.config['monthly_budget']
                current_cost = 0
            
            # Calculate usage percentage
            usage_percent = (current_cost / monthly_budget) * 100 if monthly_budget > 0 else 0
            
            # Get cost saving mode status
            cost_saving_mode = self.redis_client.get('cost_saving_mode') == '1'
            emergency_mode = self.redis_client.get('emergency_mode') == '1'
            
            # Days left in month
            today = datetime.now()
            if today.month == 12:
                next_month = today.replace(year=today.year + 1, month=1, day=1)
            else:
                next_month = today.replace(month=today.month + 1, day=1)
            days_left = (next_month - today).days
            
            # Daily burn rate
            days_passed = today.day
            daily_burn_rate = current_cost / days_passed if days_passed > 0 else 0
            projected_monthly_cost = daily_burn_rate * 30
            
            return {
                'monthly_budget': monthly_budget,
                'current_cost': current_cost,
                'usage_percent': usage_percent,
                'days_left': days_left,
                'daily_burn_rate': daily_burn_rate,
                'projected_cost': projected_monthly_cost,
                'cost_saving_mode': cost_saving_mode,
                'emergency_mode': emergency_mode,
                'remaining_budget': monthly_budget - current_cost
            }
            
        except Exception as e:
            logger.error(f"Cost metrics error: {e}")
            return {}
    
    async def _get_pipeline_metrics(self) -> Dict:
        """Get pipeline processing metrics"""
        try:
            with self.db_connection.cursor() as cursor:
                # Stock counts by tier
                cursor.execute("""
                    SELECT priority_tier, COUNT(*) 
                    FROM stocks 
                    WHERE is_active = true 
                    GROUP BY priority_tier 
                    ORDER BY priority_tier
                """)
                tier_counts = dict(cursor.fetchall())
                
                # Recent processing activity
                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE DATE(created_at) = CURRENT_DATE) as today_prices,
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as hourly_prices,
                        MAX(created_at) as last_update
                    FROM price_history
                """)
                result = cursor.fetchone()
                today_prices, hourly_prices, last_update = result if result else (0, 0, None)
                
                # Technical indicators
                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE DATE(created_at) = CURRENT_DATE) as today_indicators,
                        MAX(created_at) as last_indicator_update
                    FROM technical_indicators
                """)
                result = cursor.fetchone()
                today_indicators, last_indicator_update = result if result else (0, None)
                
                # Pipeline status from Redis
                pipeline_status = self.redis_client.get('pipeline_status') or 'unknown'
                last_run_time = self.redis_client.get('last_pipeline_run')
                
                return {
                    'tier_counts': tier_counts,
                    'today_prices': today_prices,
                    'hourly_prices': hourly_prices,
                    'today_indicators': today_indicators,
                    'last_price_update': last_update.isoformat() if last_update else None,
                    'last_indicator_update': last_indicator_update.isoformat() if last_indicator_update else None,
                    'pipeline_status': pipeline_status,
                    'last_run_time': last_run_time
                }
                
        except Exception as e:
            logger.error(f"Pipeline metrics error: {e}")
            return {}
    
    async def _get_active_alerts(self) -> List[Dict]:
        """Get active alerts"""
        try:
            # Get cost alerts
            cost_alerts = self.redis_client.lrange('cost_alerts', 0, 9)  # Last 10 alerts
            alerts = []
            
            for alert_json in cost_alerts:
                try:
                    alert = json.loads(alert_json)
                    alerts.append(alert)
                except json.JSONDecodeError:
                    continue
            
            # Add system alerts
            system_metrics = await self._get_system_metrics()
            
            if system_metrics.get('cpu_percent', 0) > 80:
                alerts.append({
                    'level': 'warning',
                    'message': f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat()
                })
            
            if system_metrics.get('memory_percent', 0) > 85:
                alerts.append({
                    'level': 'warning',
                    'message': f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat()
                })
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return alerts[:10]  # Return only latest 10 alerts
            
        except Exception as e:
            logger.error(f"Alerts error: {e}")
            return []
    
    async def _display_system_status(self, data: Dict):
        """Display system status section"""
        print("\n🖥️  SYSTEM STATUS")
        print("-" * 40)
        
        system = data.get('system', {})
        services = data.get('services', {})
        
        if system:
            print(f"CPU Usage:    {system.get('cpu_percent', 0):.1f}%")
            print(f"Memory:       {system.get('memory_percent', 0):.1f}% ({system.get('memory_used_gb', 0):.1f}GB / {system.get('memory_total_gb', 0):.1f}GB)")
            print(f"Disk:         {system.get('disk_percent', 0):.1f}% ({system.get('disk_used_gb', 0):.1f}GB / {system.get('disk_total_gb', 0):.1f}GB)")
            
            if system.get('load_avg'):
                load_avg = system['load_avg']
                print(f"Load Avg:     {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
        
        print("\n🔧 SERVICE STATUS")
        for service_name, service_info in services.items():
            status = service_info.get('status', 'unknown')
            response_time = service_info.get('response_time_ms')
            
            status_icon = {
                'online': '🟢',
                'offline': '🔴',
                'degraded': '🟡',
                'unknown': '⚪'
            }.get(status, '⚪')
            
            response_str = f" ({response_time}ms)" if response_time else ""
            print(f"{status_icon} {service_name.capitalize()}: {status.upper()}{response_str}")
    
    async def _display_pipeline_status(self, data: Dict):
        """Display pipeline status section"""
        print("\n🔄 PIPELINE STATUS")
        print("-" * 40)
        
        pipeline = data.get('pipeline', {})
        
        # Stock tiers
        tier_counts = pipeline.get('tier_counts', {})
        total_stocks = sum(tier_counts.values())
        
        print(f"Active Stocks: {total_stocks}")
        for tier in sorted(tier_counts.keys()):
            count = tier_counts[tier]
            tier_name = {1: 'Critical', 2: 'High', 3: 'Medium', 4: 'Low', 5: 'Minimal'}.get(tier, f'Tier {tier}')
            print(f"  {tier_name}: {count}")
        
        # Processing activity
        print(f"\nToday's Activity:")
        print(f"  Price Updates: {pipeline.get('today_prices', 0)}")
        print(f"  Technical Indicators: {pipeline.get('today_indicators', 0)}")
        print(f"  Hourly Updates: {pipeline.get('hourly_prices', 0)}")
        
        # Last updates
        last_price = pipeline.get('last_price_update')
        last_indicator = pipeline.get('last_indicator_update')
        
        if last_price:
            last_price_time = datetime.fromisoformat(last_price.replace('Z', '+00:00')).strftime('%H:%M:%S')
            print(f"  Last Price Update: {last_price_time}")
        
        if last_indicator:
            last_indicator_time = datetime.fromisoformat(last_indicator.replace('Z', '+00:00')).strftime('%H:%M:%S')
            print(f"  Last Technical Update: {last_indicator_time}")
        
        # Pipeline status
        status = pipeline.get('pipeline_status', 'unknown')
        print(f"\nPipeline Status: {status.upper()}")
    
    async def _display_api_status(self, data: Dict):
        """Display API status section"""
        print("\n🔑 API STATUS")
        print("-" * 40)
        
        api_data = data.get('api', {})
        
        if not api_data:
            print("No API data available")
            return
        
        # Create table for API status
        table_data = []
        for provider, metrics in api_data.items():
            daily_calls = metrics.get('daily_calls', 0)
            daily_limit = metrics.get('daily_limit', float('inf'))
            minute_calls = metrics.get('current_minute_calls', 0)
            minute_limit = metrics.get('minute_limit', float('inf'))
            status = metrics.get('status', 'unknown')
            
            # Format limits
            daily_limit_str = str(daily_limit) if daily_limit != float('inf') else '∞'
            minute_limit_str = str(minute_limit) if minute_limit != float('inf') else '∞'
            
            # Calculate usage percentages
            daily_pct = (daily_calls / daily_limit * 100) if daily_limit != float('inf') else 0
            minute_pct = (minute_calls / minute_limit * 100) if minute_limit != float('inf') else 0
            
            table_data.append([
                provider.replace('_', ' ').title(),
                f"{daily_calls}/{daily_limit_str} ({daily_pct:.1f}%)",
                f"{minute_calls}/{minute_limit_str}",
                status.upper()
            ])
        
        print(tabulate(
            table_data,
            headers=['Provider', 'Daily Usage', 'Current Min', 'Status'],
            tablefmt='simple'
        ))
    
    async def _display_cost_monitoring(self, data: Dict):
        """Display cost monitoring section"""
        print("\n💰 COST MONITORING")
        print("-" * 40)
        
        cost_data = data.get('cost', {})
        
        if not cost_data:
            print("No cost data available")
            return
        
        budget = cost_data.get('monthly_budget', 50)
        current_cost = cost_data.get('current_cost', 0)
        usage_percent = cost_data.get('usage_percent', 0)
        projected_cost = cost_data.get('projected_cost', 0)
        remaining = cost_data.get('remaining_budget', budget)
        days_left = cost_data.get('days_left', 0)
        
        # Budget status
        print(f"Monthly Budget: ${budget:.2f}")
        print(f"Current Cost:   ${current_cost:.2f} ({usage_percent:.1f}%)")
        print(f"Remaining:      ${remaining:.2f}")
        print(f"Projected Cost: ${projected_cost:.2f}")
        print(f"Days Left:      {days_left}")
        
        # Status indicators
        cost_saving_mode = cost_data.get('cost_saving_mode', False)
        emergency_mode = cost_data.get('emergency_mode', False)
        
        if emergency_mode:
            print("🚨 EMERGENCY MODE ACTIVE")
        elif cost_saving_mode:
            print("⚠️  Cost Saving Mode Active")
        elif usage_percent > 80:
            print("🟡 Approaching Budget Limit")
        else:
            print("🟢 Budget Status OK")
        
        # Budget bar
        bar_length = 30
        filled_length = int(bar_length * usage_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"Budget Usage: [{bar}] {usage_percent:.1f}%")
    
    async def _display_data_status(self, data: Dict):
        """Display data status section"""
        print("\n📊 DATA STATUS")
        print("-" * 40)
        
        db_data = data.get('database', {})
        
        if not db_data:
            print("No database data available")
            return
        
        print(f"Database Size: {db_data.get('database_size', 'Unknown')}")
        print(f"Active Connections: {db_data.get('active_connections', 0)}")
        print(f"Hourly Inserts: {db_data.get('hourly_inserts', 0)}")
        print(f"Daily Inserts: {db_data.get('daily_inserts', 0)}")
        
        # Top tables
        table_stats = db_data.get('table_stats', [])
        if table_stats:
            print("\nTop Tables by Size:")
            for schema, table, size, inserts, updates in table_stats[:3]:
                print(f"  {table}: {size} ({inserts} inserts, {updates} updates)")
    
    async def _display_performance_metrics(self, data: Dict):
        """Display performance metrics section"""
        print("\n⚡ PERFORMANCE")
        print("-" * 40)
        
        services = data.get('services', {})
        
        # Response times
        response_times = []
        for service, info in services.items():
            response_time = info.get('response_time_ms')
            if response_time is not None:
                response_times.append((service, response_time))
        
        if response_times:
            print("Service Response Times:")
            for service, time_ms in sorted(response_times, key=lambda x: x[1]):
                print(f"  {service.capitalize()}: {time_ms}ms")
        
        # Cache hit rates (if available)
        try:
            cache_info = self.redis_client.info('stats')
            if cache_info:
                keyspace_hits = cache_info.get('keyspace_hits', 0)
                keyspace_misses = cache_info.get('keyspace_misses', 0)
                total = keyspace_hits + keyspace_misses
                
                if total > 0:
                    hit_rate = (keyspace_hits / total) * 100
                    print(f"\nRedis Cache Hit Rate: {hit_rate:.1f}%")
        except:
            pass
    
    async def _display_alerts(self, data: Dict):
        """Display active alerts section"""
        alerts = data.get('alerts', [])
        
        if not alerts:
            return
        
        print("\n🚨 ACTIVE ALERTS")
        print("-" * 40)
        
        for alert in alerts[:5]:  # Show only top 5 alerts
            level = alert.get('level', 'info')
            message = alert.get('message', 'Unknown alert')
            timestamp = alert.get('timestamp', '')
            
            # Parse timestamp
            if timestamp:
                try:
                    alert_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = alert_time.strftime('%H:%M:%S')
                except:
                    time_str = timestamp[:8]  # Take first 8 chars
            else:
                time_str = '??:??:??'
            
            level_icon = {
                'critical': '🔴',
                'warning': '🟡',
                'info': '🔵'
            }.get(level, '⚪')
            
            print(f"{level_icon} [{time_str}] {message}")
    
    async def _cleanup_connections(self):
        """Clean up connections"""
        try:
            if self.db_connection:
                self.db_connection.close()
        except:
            pass


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Pipeline Monitoring')
    parser.add_argument('--interval', type=int, default=30,
                       help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    
    args = parser.parse_args()
    
    monitor = PipelineMonitor()
    
    try:
        if args.once:
            # Single run mode
            await monitor._initialize_connections()
            data = await monitor._collect_monitoring_data()
            
            print("🎯 INVESTMENT ANALYSIS PIPELINE - STATUS SNAPSHOT")
            print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            await monitor._display_system_status(data)
            await monitor._display_pipeline_status(data)
            await monitor._display_api_status(data)
            await monitor._display_cost_monitoring(data)
            await monitor._display_data_status(data)
            await monitor._display_performance_metrics(data)
            await monitor._display_alerts(data)
            
            await monitor._cleanup_connections()
        else:
            # Continuous monitoring mode
            await monitor.start_monitoring(args.interval)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        return 0
    except Exception as e:
        print(f"\n\n❌ Monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        sys.exit(0)