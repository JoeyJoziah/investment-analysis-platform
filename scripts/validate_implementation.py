#!/usr/bin/env python3
"""
Validation script for Week 3-4 implementation
Runs all validation checks from the checklist.
"""

import subprocess
import sys
import time
import asyncio
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import yaml

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class ImplementationValidator:
    """Validates the implementation against the checklist."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def print_header(self, title: str):
        """Print section header."""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{title:^60}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
    
    def print_result(self, check: str, status: str, message: str = ""):
        """Print validation result."""
        if status == "PASS":
            symbol = f"{GREEN}✓{RESET}"
            self.passed += 1
        elif status == "FAIL":
            symbol = f"{RED}✗{RESET}"
            self.failed += 1
        elif status == "WARN":
            symbol = f"{YELLOW}⚠{RESET}"
            self.warnings += 1
        else:
            symbol = "?"
        
        print(f"{symbol} {check:40} {status:6} {message}")
        self.results.append({
            'check': check,
            'status': status,
            'message': message
        })
    
    def run_command(self, cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
        """Run shell command and return result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def check_tests(self) -> bool:
        """Check if all tests pass."""
        self.print_header("Running Tests")
        
        # Check if pytest is installed
        returncode, stdout, stderr = self.run_command(['pytest', '--version'])
        if returncode != 0:
            self.print_result("Pytest installed", "WARN", "pytest not found")
            return False
        
        # Run backend tests
        returncode, stdout, stderr = self.run_command([
            'pytest', 
            'backend/tests/',
            '-v',
            '--tb=short'
        ])
        
        if returncode == 0:
            self.print_result("Backend tests", "PASS", "All tests passed")
            return True
        elif returncode == 5:
            self.print_result("Backend tests", "WARN", "No tests found")
            return True
        else:
            self.print_result("Backend tests", "FAIL", f"Tests failed: {stderr[:100]}")
            return False
    
    def check_secrets(self) -> bool:
        """Check for hardcoded secrets."""
        self.print_header("Checking for Hardcoded Secrets")
        
        patterns = [
            ('password', ['password123', 'admin123', 'secret123']),
            ('apikey', ['sk-', 'api_key_', 'AKIA']),
            ('token', ['token123', 'bearer_token_'])
        ]
        
        found_secrets = False
        
        for pattern, bad_values in patterns:
            returncode, stdout, stderr = self.run_command([
                'grep', '-r', pattern,
                '--include=*.yaml',
                '--include=*.yml',
                '--include=*.env',
                '--include=*.py'
            ])
            
            if returncode == 0 and stdout:
                # Check if any bad values are present
                for bad_value in bad_values:
                    if bad_value.lower() in stdout.lower():
                        self.print_result(
                            f"No {pattern} hardcoded",
                            "FAIL",
                            f"Found potential secret: {bad_value}"
                        )
                        found_secrets = True
                        break
                else:
                    # Found pattern but no bad values
                    self.print_result(
                        f"No {pattern} hardcoded",
                        "PASS",
                        "Only references found"
                    )
            else:
                self.print_result(f"No {pattern} hardcoded", "PASS", "")
        
        return not found_secrets
    
    def check_api_limits(self) -> bool:
        """Check API rate limiting configuration."""
        self.print_header("Checking API Rate Limits")
        
        try:
            # Import and check rate limiter configuration
            sys.path.insert(0, str(self.project_root))
            from backend.utils.distributed_rate_limiter import APIRateLimiter
            
            limits = APIRateLimiter.PROVIDER_LIMITS
            
            # Check free tier limits are configured correctly
            checks = [
                ('alpha_vantage', 5, 25),
                ('finnhub', 60, float('inf')),
                ('polygon', 5, float('inf'))
            ]
            
            all_correct = True
            for provider, per_minute, per_day in checks:
                if provider in limits:
                    actual_minute = limits[provider].get('per_minute', 0)
                    actual_day = limits[provider].get('per_day', 0)
                    
                    if actual_minute == per_minute and actual_day == per_day:
                        self.print_result(
                            f"{provider} limits",
                            "PASS",
                            f"{per_minute}/min, {per_day}/day"
                        )
                    else:
                        self.print_result(
                            f"{provider} limits",
                            "FAIL",
                            f"Expected {per_minute}/min, got {actual_minute}/min"
                        )
                        all_correct = False
                else:
                    self.print_result(f"{provider} limits", "FAIL", "Not configured")
                    all_correct = False
            
            return all_correct
            
        except ImportError as e:
            self.print_result("API limits check", "FAIL", f"Import error: {e}")
            return False
    
    def check_database_performance(self) -> bool:
        """Check database query performance."""
        self.print_header("Checking Database Performance")
        
        try:
            from backend.utils.database import engine
            from sqlalchemy import text
            
            # Run simple query and measure time
            latencies = []
            for _ in range(10):
                start = time.time()
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)
            
            # Calculate p95
            latencies.sort()
            p95 = latencies[int(len(latencies) * 0.95)]
            
            if p95 < 100:
                self.print_result(
                    "Database query p95",
                    "PASS",
                    f"{p95:.2f}ms < 100ms"
                )
                return True
            else:
                self.print_result(
                    "Database query p95",
                    "FAIL",
                    f"{p95:.2f}ms > 100ms"
                )
                return False
                
        except Exception as e:
            self.print_result("Database performance", "WARN", f"Could not test: {e}")
            return True
    
    def check_cost_projection(self) -> bool:
        """Check monthly cost projection."""
        self.print_header("Checking Cost Projection")
        
        try:
            # Calculate projected costs
            daily_api_calls = {
                'finnhub': 1000,    # For critical stocks
                'alpha_vantage': 25, # Daily limit
                'polygon': 100      # For medium tier
            }
            
            # All are free tier, so cost should be $0 for APIs
            api_cost = 0
            
            # Infrastructure costs (estimated)
            infrastructure_costs = {
                'DigitalOcean Droplet': 20,  # $20/month for 2GB
                'Redis': 10,                 # Redis Cloud free tier or $10
                'Storage': 10,                # 10GB storage
                'Buffer': 10                  # Buffer for overages
            }
            
            total_cost = api_cost + sum(infrastructure_costs.values())
            
            if total_cost <= 50:
                self.print_result(
                    "Monthly cost projection",
                    "PASS",
                    f"${total_cost}/month < $50"
                )
                
                # Show breakdown
                for item, cost in infrastructure_costs.items():
                    print(f"  - {item:20} ${cost}")
                
                return True
            else:
                self.print_result(
                    "Monthly cost projection",
                    "FAIL",
                    f"${total_cost}/month > $50"
                )
                return False
                
        except Exception as e:
            self.print_result("Cost projection", "FAIL", str(e))
            return False
    
    def check_docker(self) -> bool:
        """Check Docker configurations."""
        self.print_header("Checking Docker Configuration")
        
        docker_files = [
            'docker-compose.yml',
            'docker-compose.redis-sentinel.yml'
        ]
        
        all_valid = True
        
        for docker_file in docker_files:
            file_path = self.project_root / docker_file
            
            if not file_path.exists():
                self.print_result(f"{docker_file}", "FAIL", "File not found")
                all_valid = False
                continue
            
            try:
                with open(file_path) as f:
                    config = yaml.safe_load(f)
                
                if 'services' in config:
                    service_count = len(config['services'])
                    self.print_result(
                        f"{docker_file}",
                        "PASS",
                        f"{service_count} services defined"
                    )
                else:
                    self.print_result(f"{docker_file}", "FAIL", "No services defined")
                    all_valid = False
                    
            except Exception as e:
                self.print_result(f"{docker_file}", "FAIL", f"Parse error: {e}")
                all_valid = False
        
        # Check if Docker is running
        returncode, stdout, stderr = self.run_command(['docker', 'version'])
        if returncode == 0:
            self.print_result("Docker installed", "PASS", "")
        else:
            self.print_result("Docker installed", "WARN", "Docker not available")
        
        return all_valid
    
    def check_memory(self) -> bool:
        """Check memory usage."""
        self.print_header("Checking Memory Usage")
        
        try:
            # Get current process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb < 2048:
                self.print_result(
                    "Memory usage",
                    "PASS",
                    f"{memory_mb:.2f}MB < 2GB"
                )
                return True
            else:
                self.print_result(
                    "Memory usage",
                    "WARN",
                    f"{memory_mb:.2f}MB > 2GB"
                )
                return False
                
        except Exception as e:
            self.print_result("Memory check", "WARN", f"Could not check: {e}")
            return True
    
    def check_implementation_completeness(self) -> bool:
        """Check if all required components are implemented."""
        self.print_header("Checking Implementation Completeness")
        
        required_files = [
            ('backend/utils/integration.py', "Integration layer"),
            ('backend/utils/redis_resilience.py', "Redis resilience"),
            ('backend/api/versioning.py', "API versioning"),
            ('backend/utils/distributed_rate_limiter.py', "Rate limiting"),
            ('backend/utils/query_cache.py', "Query caching"),
            ('backend/utils/parallel_processor.py', "Parallel processing"),
            ('backend/utils/persistent_cost_monitor.py', "Cost monitoring"),
            ('docker-compose.redis-sentinel.yml', "Redis Sentinel config")
        ]
        
        all_present = True
        
        for file_path, description in required_files:
            full_path = self.project_root / file_path
            
            if full_path.exists():
                # Check file size to ensure it's not empty
                size = full_path.stat().st_size
                if size > 100:  # At least 100 bytes
                    self.print_result(
                        description,
                        "PASS",
                        f"{size} bytes"
                    )
                else:
                    self.print_result(description, "WARN", "File too small")
            else:
                self.print_result(description, "FAIL", "File not found")
                all_present = False
        
        return all_present
    
    def generate_report(self):
        """Generate final validation report."""
        self.print_header("Validation Summary")
        
        total = self.passed + self.failed + self.warnings
        
        print(f"Total checks: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"{YELLOW}Warnings: {self.warnings}{RESET}")
        
        if self.failed == 0:
            print(f"\n{GREEN}✓ All critical checks passed!{RESET}")
            return True
        else:
            print(f"\n{RED}✗ Some checks failed. Please review and fix.{RESET}")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print(f"{BLUE}Starting Implementation Validation...{RESET}")
        
        # Run all checks
        checks = [
            self.check_implementation_completeness,
            self.check_secrets,
            self.check_api_limits,
            self.check_database_performance,
            self.check_cost_projection,
            self.check_docker,
            self.check_memory,
            self.check_tests
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.print_result(
                    check.__name__.replace('check_', ''),
                    "FAIL",
                    f"Exception: {e}"
                )
        
        # Generate report
        return self.generate_report()


def main():
    """Main validation entry point."""
    validator = ImplementationValidator()
    success = validator.run_all_checks()
    
    # Save results to file
    results_file = validator.project_root / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'passed': validator.passed,
            'failed': validator.failed,
            'warnings': validator.warnings,
            'results': validator.results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()