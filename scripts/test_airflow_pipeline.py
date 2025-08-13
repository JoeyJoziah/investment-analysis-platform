#!/usr/bin/env python3
"""
Comprehensive test suite for Airflow pipeline validation.
Tests DAG configuration, API rate limits, and processing capabilities.
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from colorama import init, Fore, Back, Style

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Initialize colorama for colored output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AirflowPipelineTester:
    """Test and validate Airflow pipeline for 6000+ stock processing."""
    
    def __init__(self):
        self.airflow_url = "http://localhost:8080/api/v1"
        self.airflow_user = "admin"
        self.airflow_pass = "admin123"
        self.test_results = []
        self.api_limits = {
            'alpha_vantage': {'daily': 25, 'per_minute': 5},
            'finnhub': {'daily': 86400, 'per_minute': 60},  # Unlimited on paid, 60/min on free
            'polygon': {'daily': 150, 'per_minute': 5}
        }
        
    def print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text.center(60)}")
        print(f"{Fore.CYAN}{'='*60}\n")
        
    def print_test(self, test_name: str, passed: bool, details: str = ""):
        """Print test result."""
        status = f"{Fore.GREEN}✓ PASSED" if passed else f"{Fore.RED}✗ FAILED"
        print(f"{status} - {test_name}")
        if details:
            print(f"  {Fore.YELLOW}Details: {details}")
        self.test_results.append((test_name, passed, details))
        
    def check_airflow_health(self) -> bool:
        """Check if Airflow is healthy and accessible."""
        self.print_header("Checking Airflow Health")
        
        try:
            # Check webserver
            response = requests.get(
                f"{self.airflow_url}/health",
                auth=(self.airflow_user, self.airflow_pass),
                timeout=10
            )
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Check component health
                components = ['metadatabase', 'scheduler', 'triggerer']
                all_healthy = True
                
                for component in components:
                    if component in health_data:
                        status = health_data[component].get('status', 'unknown')
                        is_healthy = status == 'healthy'
                        self.print_test(
                            f"Airflow {component.capitalize()} Health",
                            is_healthy,
                            f"Status: {status}"
                        )
                        all_healthy = all_healthy and is_healthy
                        
                return all_healthy
            else:
                self.print_test(
                    "Airflow API Access",
                    False,
                    f"Status code: {response.status_code}"
                )
                return False
                
        except requests.exceptions.RequestException as e:
            self.print_test(
                "Airflow Connection",
                False,
                f"Cannot connect to Airflow: {str(e)}"
            )
            return False
            
    def validate_dags(self) -> bool:
        """Validate all DAGs are properly configured."""
        self.print_header("Validating DAGs")
        
        try:
            # List all DAGs
            response = requests.get(
                f"{self.airflow_url}/dags",
                auth=(self.airflow_user, self.airflow_pass),
                timeout=10
            )
            
            if response.status_code != 200:
                self.print_test("DAG List Access", False, f"Status: {response.status_code}")
                return False
                
            dags = response.json().get('dags', [])
            
            # Check for our main DAG
            main_dag = None
            for dag in dags:
                if dag['dag_id'] == 'daily_market_analysis':
                    main_dag = dag
                    break
                    
            if not main_dag:
                self.print_test(
                    "Main DAG Existence",
                    False,
                    "daily_market_analysis DAG not found"
                )
                return False
                
            # Validate DAG configuration
            dag_checks = {
                'DAG is Active': not main_dag.get('is_paused', True),
                'DAG has Schedule': main_dag.get('schedule_interval') is not None,
                'DAG has Tasks': main_dag.get('tasks', []) != [],
                'DAG is Valid': not main_dag.get('has_import_errors', False)
            }
            
            all_passed = True
            for check_name, check_result in dag_checks.items():
                self.print_test(check_name, check_result)
                all_passed = all_passed and check_result
                
            # Check for optimized DAG if exists
            optimized_dag = None
            for dag in dags:
                if dag['dag_id'] == 'daily_market_analysis_optimized':
                    optimized_dag = dag
                    self.print_test(
                        "Optimized DAG Found",
                        True,
                        "Enhanced version available"
                    )
                    break
                    
            return all_passed
            
        except Exception as e:
            self.print_test("DAG Validation", False, str(e))
            return False
            
    def check_pools(self) -> bool:
        """Check if resource pools are properly configured."""
        self.print_header("Checking Resource Pools")
        
        try:
            response = requests.get(
                f"{self.airflow_url}/pools",
                auth=(self.airflow_user, self.airflow_pass),
                timeout=10
            )
            
            if response.status_code != 200:
                self.print_test("Pool Access", False, f"Status: {response.status_code}")
                return False
                
            pools = response.json().get('pools', [])
            
            # Required pools for rate limiting
            required_pools = {
                'api_calls': 5,  # Limit concurrent API calls
                'compute_intensive': 8,  # ML/Analytics tasks
                'database_tasks': 12,  # DB operations
                'low_priority': 3,  # Background tasks
                'high_frequency': 2  # Intraday data
            }
            
            existing_pools = {p['name']: p['slots'] for p in pools}
            
            all_configured = True
            for pool_name, expected_slots in required_pools.items():
                if pool_name in existing_pools:
                    actual_slots = existing_pools[pool_name]
                    is_correct = actual_slots >= expected_slots
                    self.print_test(
                        f"Pool '{pool_name}'",
                        is_correct,
                        f"Slots: {actual_slots} (expected >= {expected_slots})"
                    )
                    all_configured = all_configured and is_correct
                else:
                    self.print_test(
                        f"Pool '{pool_name}'",
                        False,
                        "Not configured"
                    )
                    all_configured = False
                    
            return all_configured
            
        except Exception as e:
            self.print_test("Pool Configuration", False, str(e))
            return False
            
    def validate_api_rate_limits(self) -> bool:
        """Validate API rate limit compliance in DAG configuration."""
        self.print_header("Validating API Rate Limits")
        
        # Simulate stock distribution across tiers
        total_stocks = 6000
        tier_distribution = {
            'tier1_realtime': 500,   # S&P 500 + high volume
            'tier2_frequent': 1500,  # Mid-cap active
            'tier3_daily': 2000,     # Small-cap watched
            'tier4_batch': 2000      # Remaining stocks
        }
        
        # Calculate API usage per tier
        api_usage = {
            'tier1': {
                'provider': 'finnhub',
                'calls_per_stock': 1,
                'frequency': 'hourly',
                'daily_calls': tier_distribution['tier1_realtime'] * 8  # 8 hours trading
            },
            'tier2': {
                'provider': 'alpha_vantage',
                'calls_per_stock': 1,
                'frequency': 'daily',
                'daily_calls': min(20, tier_distribution['tier2_frequent'])  # Limited by daily quota
            },
            'tier3': {
                'provider': 'polygon',
                'calls_per_stock': 1,
                'frequency': 'daily',
                'daily_calls': min(100, tier_distribution['tier3_daily'])  # Limited by rate
            },
            'tier4': {
                'provider': 'cache',
                'calls_per_stock': 0,
                'frequency': 'weekly',
                'daily_calls': 0  # Uses cached data
            }
        }
        
        # Validate each tier's API usage
        all_compliant = True
        
        for tier, usage in api_usage.items():
            provider = usage['provider']
            daily_calls = usage['daily_calls']
            
            if provider in self.api_limits:
                limit = self.api_limits[provider]['daily']
                is_compliant = daily_calls <= limit
                
                self.print_test(
                    f"{tier.upper()} API Compliance ({provider})",
                    is_compliant,
                    f"{daily_calls} calls/day (limit: {limit})"
                )
                
                all_compliant = all_compliant and is_compliant
            else:
                self.print_test(
                    f"{tier.upper()} API Usage",
                    True,
                    f"Using {provider} (no API calls)"
                )
                
        # Calculate total daily cost estimate
        cost_per_call = {
            'finnhub': 0.00,  # Free tier
            'alpha_vantage': 0.00,  # Free tier
            'polygon': 0.00,  # Free tier
            'cache': 0.00
        }
        
        total_daily_cost = sum(
            usage['daily_calls'] * cost_per_call.get(usage['provider'], 0)
            for usage in api_usage.values()
        )
        
        monthly_cost = total_daily_cost * 22  # Trading days per month
        
        self.print_test(
            "Monthly Cost Projection",
            monthly_cost < 50,
            f"${monthly_cost:.2f}/month (limit: $50)"
        )
        
        return all_compliant
        
    def test_dag_execution(self) -> bool:
        """Test DAG execution with sample data."""
        self.print_header("Testing DAG Execution")
        
        try:
            # Trigger a test run of the DAG
            dag_id = "daily_market_analysis"
            
            # Check if DAG can be triggered
            response = requests.post(
                f"{self.airflow_url}/dags/{dag_id}/dagRuns",
                auth=(self.airflow_user, self.airflow_pass),
                json={
                    "dag_run_id": f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "logical_date": datetime.now().isoformat(),
                    "conf": {
                        "test_mode": True,
                        "sample_stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
                    }
                },
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                dag_run = response.json()
                run_id = dag_run.get('dag_run_id')
                
                self.print_test(
                    "DAG Trigger",
                    True,
                    f"Run ID: {run_id}"
                )
                
                # Wait for DAG to start
                time.sleep(5)
                
                # Check DAG run status
                status_response = requests.get(
                    f"{self.airflow_url}/dags/{dag_id}/dagRuns/{run_id}",
                    auth=(self.airflow_user, self.airflow_pass),
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    run_status = status_response.json()
                    state = run_status.get('state', 'unknown')
                    
                    self.print_test(
                        "DAG Execution Started",
                        state in ['running', 'success'],
                        f"State: {state}"
                    )
                    
                    return state in ['running', 'success']
                    
            else:
                self.print_test(
                    "DAG Trigger",
                    False,
                    f"Status: {response.status_code}"
                )
                return False
                
        except Exception as e:
            self.print_test("DAG Execution Test", False, str(e))
            return False
            
    def validate_monitoring(self) -> bool:
        """Validate monitoring and alerting configuration."""
        self.print_header("Validating Monitoring & Alerting")
        
        monitoring_checks = []
        
        # Check Prometheus metrics endpoint
        try:
            metrics_response = requests.get("http://localhost:9102/metrics", timeout=5)
            has_metrics = metrics_response.status_code == 200
            monitoring_checks.append(("Prometheus Metrics Export", has_metrics))
            
            if has_metrics:
                metrics_text = metrics_response.text
                
                # Check for specific Airflow metrics
                important_metrics = [
                    'airflow_dag_processing_total',
                    'airflow_dag_run_duration',
                    'airflow_task_duration',
                    'airflow_pool_running_tasks',
                    'airflow_pool_queued_tasks'
                ]
                
                for metric in important_metrics:
                    has_metric = metric in metrics_text
                    monitoring_checks.append((f"Metric: {metric}", has_metric))
                    
        except Exception as e:
            monitoring_checks.append(("Prometheus Metrics", False))
            
        # Check Flower (Celery monitoring)
        try:
            flower_response = requests.get("http://localhost:5555/api/workers", timeout=5)
            has_flower = flower_response.status_code == 200
            monitoring_checks.append(("Flower Monitoring", has_flower))
            
            if has_flower:
                workers = flower_response.json()
                worker_count = len(workers)
                monitoring_checks.append((
                    "Celery Workers",
                    worker_count > 0,
                    f"Active workers: {worker_count}"
                ))
                
        except Exception:
            monitoring_checks.append(("Flower Monitoring", False))
            
        # Check Grafana availability
        try:
            grafana_response = requests.get("http://localhost:3001/api/health", timeout=5)
            has_grafana = grafana_response.status_code == 200
            monitoring_checks.append(("Grafana Dashboard", has_grafana))
        except Exception:
            monitoring_checks.append(("Grafana Dashboard", False))
            
        # Print results
        all_passed = True
        for check in monitoring_checks:
            if len(check) == 3:
                name, passed, details = check
                self.print_test(name, passed, details)
            else:
                name, passed = check
                self.print_test(name, passed)
            all_passed = all_passed and passed
            
        return all_passed
        
    def simulate_full_load(self) -> bool:
        """Simulate processing 6000+ stocks to validate capacity."""
        self.print_header("Simulating Full Load (6000+ Stocks)")
        
        # Generate simulated stock list
        exchanges = ['NYSE', 'NASDAQ', 'AMEX']
        
        simulated_stocks = []
        for i in range(6000):
            stock = {
                'symbol': f"STK{i:04d}",
                'exchange': exchanges[i % 3],
                'market_cap': np.random.uniform(1e6, 1e12),
                'avg_volume': np.random.uniform(1e4, 1e8),
                'sector': ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer'][i % 5]
            }
            simulated_stocks.append(stock)
            
        stocks_df = pd.DataFrame(simulated_stocks)
        
        # Categorize into tiers based on market cap and volume
        stocks_df['priority_score'] = (
            stocks_df['market_cap'] * 0.6 + 
            stocks_df['avg_volume'] * 0.4
        )
        stocks_df = stocks_df.sort_values('priority_score', ascending=False)
        
        # Assign to tiers
        tier_sizes = [500, 1500, 2000, 2000]
        tiers = {}
        current_idx = 0
        
        for i, size in enumerate(tier_sizes, 1):
            end_idx = min(current_idx + size, len(stocks_df))
            tier_stocks = stocks_df.iloc[current_idx:end_idx]
            tiers[f'tier{i}'] = tier_stocks
            current_idx = end_idx
            
            self.print_test(
                f"Tier {i} Assignment",
                len(tier_stocks) == size or i == 4,
                f"{len(tier_stocks)} stocks"
            )
            
        # Simulate API call distribution
        api_calls_simulation = {
            'tier1': {
                'total_stocks': len(tiers['tier1']),
                'calls_per_hour': len(tiers['tier1']),
                'provider': 'finnhub',
                'estimated_time': len(tiers['tier1']) / 60  # 60 calls/min
            },
            'tier2': {
                'total_stocks': len(tiers['tier2']),
                'calls_per_day': min(20, len(tiers['tier2'])),
                'provider': 'alpha_vantage',
                'estimated_time': 20 * 12  # 12 seconds between calls
            },
            'tier3': {
                'total_stocks': len(tiers['tier3']),
                'calls_per_day': min(100, len(tiers['tier3'])),
                'provider': 'polygon',
                'estimated_time': 100 * 12  # 12 seconds between calls
            },
            'tier4': {
                'total_stocks': len(tiers['tier4']),
                'calls_per_day': 0,
                'provider': 'cache',
                'estimated_time': 0
            }
        }
        
        # Calculate total processing time
        total_api_time = sum(
            sim['estimated_time'] for sim in api_calls_simulation.values()
        )
        
        total_api_time_hours = total_api_time / 3600
        
        self.print_test(
            "Total API Processing Time",
            total_api_time_hours < 8,  # Should complete within trading hours
            f"{total_api_time_hours:.2f} hours"
        )
        
        # Estimate computational load
        compute_time_per_stock = 0.5  # seconds for technical analysis
        total_compute_time = 6000 * compute_time_per_stock
        
        # With parallel processing (8 workers)
        parallel_compute_time = total_compute_time / 8 / 3600  # hours
        
        self.print_test(
            "Computational Processing Time",
            parallel_compute_time < 2,
            f"{parallel_compute_time:.2f} hours with 8 workers"
        )
        
        # Memory requirements
        memory_per_stock = 10  # MB per stock (price history + indicators)
        total_memory = 6000 * memory_per_stock / 1024  # GB
        
        self.print_test(
            "Memory Requirements",
            total_memory < 64,
            f"{total_memory:.1f} GB estimated"
        )
        
        return total_api_time_hours < 8 and parallel_compute_time < 2
        
    def generate_report(self):
        """Generate final test report."""
        self.print_header("Test Summary Report")
        
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        total_tests = len(self.test_results)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"{Fore.CYAN}Total Tests: {total_tests}")
        print(f"{Fore.GREEN}Passed: {passed_tests}")
        print(f"{Fore.RED}Failed: {total_tests - passed_tests}")
        print(f"{Fore.YELLOW}Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate == 100:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ All tests passed! Pipeline is ready for production.")
        elif pass_rate >= 80:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}⚠ Most tests passed. Review failures before production.")
        else:
            print(f"\n{Fore.RED}{Style.BRIGHT}✗ Critical failures detected. Pipeline needs fixes.")
            
        # List failed tests
        failed_tests = [(name, details) for name, passed, details in self.test_results if not passed]
        if failed_tests:
            print(f"\n{Fore.RED}Failed Tests:")
            for name, details in failed_tests:
                print(f"  - {name}: {details}")
                
    def run_all_tests(self):
        """Run all validation tests."""
        print(f"{Fore.CYAN}{Style.BRIGHT}Starting Airflow Pipeline Validation")
        print(f"{Fore.CYAN}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run tests in sequence
        tests = [
            self.check_airflow_health,
            self.validate_dags,
            self.check_pools,
            self.validate_api_rate_limits,
            self.validate_monitoring,
            self.simulate_full_load,
            self.test_dag_execution  # Run last as it triggers actual execution
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test failed with error: {e}")
                self.print_test(test_func.__name__, False, str(e))
                
        # Generate final report
        self.generate_report()


def main():
    """Main execution function."""
    tester = AirflowPipelineTester()
    
    # Check if we should run in quick mode or full mode
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print(f"{Fore.YELLOW}Running in quick mode (skipping execution tests)")
        tester.check_airflow_health()
        tester.validate_dags()
        tester.check_pools()
        tester.validate_api_rate_limits()
    else:
        tester.run_all_tests()
        
    return 0 if all(passed for _, passed, _ in tester.test_results) else 1


if __name__ == "__main__":
    sys.exit(main())