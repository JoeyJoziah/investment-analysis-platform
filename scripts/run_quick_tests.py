#!/usr/bin/env python3
"""
Quick Test Execution Script for Investment Analysis Platform
Python version for cross-platform compatibility
"""

import requests
import subprocess
import sys
import time
from typing import Dict, Tuple, List
from datetime import datetime
import json

# Configuration
BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

# Colors for terminal output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

# Test results tracking
test_results = {
    'passed': 0,
    'failed': 0,
    'total': 0,
    'failed_tests': []
}

# Store generated test data
test_data = {
    'token': None,
    'email': None,
    'portfolio_id': None
}

def print_header(message: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}{message}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")

def print_test(message: str):
    """Print test description"""
    print(f"{Colors.YELLOW}Testing:{Colors.NC} {message}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ PASS:{Colors.NC} {message}")
    test_results['passed'] += 1
    test_results['total'] += 1

def print_failure(message: str, details: str = ""):
    """Print failure message"""
    print(f"{Colors.RED}✗ FAIL:{Colors.NC} {message}")
    if details:
        print(f"{Colors.RED}       {details}{Colors.NC}")
    test_results['failed'] += 1
    test_results['total'] += 1
    test_results['failed_tests'].append(f"{message}: {details}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ WARNING:{Colors.NC} {message}")
    test_results['total'] += 1

def check_response(url: str, expected_code: int = 200, description: str = "") -> bool:
    """Check HTTP response status code"""
    print_test(description)
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == expected_code:
            print_success(f"{description} (HTTP {response.status_code})")
            return True
        else:
            print_failure(description, f"Expected HTTP {expected_code}, got HTTP {response.status_code}")
            return False
    except Exception as e:
        print_failure(description, f"Request failed: {str(e)}")
        return False

def check_json_field(url: str, field: str, description: str) -> bool:
    """Check if JSON response contains a specific field"""
    print_test(description)
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        # Handle nested fields (e.g., "checks.database")
        fields = field.split('.')
        value = data
        for f in fields:
            if isinstance(value, dict) and f in value:
                value = value[f]
            else:
                print_failure(description, f"Field '{field}' not found in response")
                return False

        print_success(description)
        return True
    except Exception as e:
        print_failure(description, f"Error: {str(e)}")
        return False

def check_service_running(service_name: str, description: str) -> bool:
    """Check if Docker service is running"""
    print_test(description)
    try:
        result = subprocess.run(
            ['docker-compose', 'ps'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if service_name in result.stdout and 'Up' in result.stdout:
            print_success(description)
            return True
        else:
            print_failure(description, f"Service {service_name} is not running")
            return False
    except Exception as e:
        print_failure(description, f"Error checking service: {str(e)}")
        return False

def test_pre_flight_checks():
    """Pre-flight checks"""
    print_header("PRE-FLIGHT CHECKS")

    # Check Docker
    print_test("Docker daemon")
    try:
        subprocess.run(['docker', 'info'], capture_output=True, check=True, timeout=5)
        print_success("Docker is running")
    except:
        print_failure("Docker is running", "Docker daemon is not running")
        sys.exit(1)

    # Check Docker Compose
    print_test("Docker Compose")
    try:
        subprocess.run(['docker-compose', '--version'], capture_output=True, check=True, timeout=5)
        print_success("Docker Compose is available")
    except:
        print_failure("Docker Compose is available", "docker-compose not found")
        sys.exit(1)

def test_service_health():
    """Check all services are running"""
    print_header("SERVICE HEALTH CHECKS")

    check_service_running("investment_db", "PostgreSQL Database")
    check_service_running("investment_cache", "Redis Cache")
    check_service_running("investment_api", "Backend API")
    check_service_running("investment_frontend", "Frontend Web")

def test_backend_health():
    """Test backend health endpoints"""
    print_header("BACKEND API - HEALTH ENDPOINTS")

    check_response(f"{BASE_URL}/api/health", 200, "Basic health check")
    check_json_field(f"{BASE_URL}/api/health", "status", "Health status field exists")
    check_json_field(f"{BASE_URL}/api/health", "version", "Version field exists")

    check_response(f"{BASE_URL}/api/health/readiness", 200, "Readiness check")
    check_json_field(f"{BASE_URL}/api/health/readiness", "checks.database", "Database readiness check")
    check_json_field(f"{BASE_URL}/api/health/readiness", "checks.cache", "Cache readiness check")

    check_response(f"{BASE_URL}/api/health/liveness", 200, "Liveness probe")
    check_response(f"{BASE_URL}/api/health/metrics", 200, "System metrics")

def test_authentication():
    """Test authentication endpoints"""
    print_header("AUTHENTICATION ENDPOINTS")

    # Register new user
    print_test("User registration")
    timestamp = int(time.time())
    test_email = f"test{timestamp}@example.com"
    test_data['email'] = test_email

    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": test_email,
                "password": "SecurePass123!",
                "full_name": "Test User"
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if 'access_token' in data:
                test_data['token'] = data['access_token']
                print_success("User registration (received JWT token)")
            else:
                print_failure("User registration", "No access_token in response")
        else:
            print_failure("User registration", f"HTTP {response.status_code}")
    except Exception as e:
        print_failure("User registration", str(e))

    # Login
    print_test("User login")
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "email": test_email,
                "password": "SecurePass123!"
            },
            timeout=10
        )

        if response.status_code == 200 and 'access_token' in response.json():
            print_success("User login (received JWT token)")
        else:
            print_failure("User login", "No access_token in response")
    except Exception as e:
        print_failure("User login", str(e))

    # Get current user
    if test_data['token']:
        print_test("Get current user (authenticated)")
        try:
            response = requests.get(
                f"{BASE_URL}/api/auth/me",
                headers={"Authorization": f"Bearer {test_data['token']}"},
                timeout=10
            )

            if response.status_code == 200 and 'email' in response.json():
                print_success("Get current user")
            else:
                print_failure("Get current user", "No email in response")
        except Exception as e:
            print_failure("Get current user", str(e))

    # Test unauthorized access
    print_test("Protected endpoint without token (should fail)")
    try:
        response = requests.get(f"{BASE_URL}/api/portfolio", timeout=10)
        if response.status_code == 401:
            print_success("Unauthorized access blocked (HTTP 401)")
        else:
            print_failure("Unauthorized access blocked", f"Expected HTTP 401, got HTTP {response.status_code}")
    except Exception as e:
        print_failure("Unauthorized access blocked", str(e))

def test_stock_data():
    """Test stock data endpoints"""
    print_header("STOCK DATA ENDPOINTS")

    check_response(f"{BASE_URL}/api/stocks?limit=10", 200, "Get stock list")
    check_response(f"{BASE_URL}/api/stocks/AAPL", 200, "Get stock by symbol (AAPL)")
    check_response(f"{BASE_URL}/api/stocks/INVALID", 404, "Invalid stock symbol returns 404")
    check_response(f"{BASE_URL}/api/stocks/AAPL/history?period=1M", 200, "Get stock price history")
    check_response(f"{BASE_URL}/api/stocks/search?query=apple", 200, "Search stocks")

def test_analysis():
    """Test analysis endpoints"""
    print_header("ANALYSIS ENDPOINTS")

    check_response(f"{BASE_URL}/api/analysis/AAPL", 200, "Get comprehensive analysis")
    check_response(f"{BASE_URL}/api/analysis/AAPL/technical", 200, "Get technical analysis")
    check_response(f"{BASE_URL}/api/analysis/AAPL/fundamental", 200, "Get fundamental analysis")
    check_response(f"{BASE_URL}/api/analysis/AAPL/sentiment", 200, "Get sentiment analysis")

def test_recommendations():
    """Test recommendations endpoints"""
    print_header("RECOMMENDATIONS ENDPOINTS")

    if test_data['token']:
        print_test("Get daily recommendations (authenticated)")
        try:
            response = requests.get(
                f"{BASE_URL}/api/recommendations",
                headers={"Authorization": f"Bearer {test_data['token']}"},
                timeout=10
            )

            if response.status_code == 200 and 'recommendations' in response.json():
                print_success("Get daily recommendations")
            else:
                print_failure("Get daily recommendations", "No recommendations field in response")
        except Exception as e:
            print_failure("Get daily recommendations", str(e))

    # Test without auth
    print_test("Recommendations without auth (should fail)")
    try:
        response = requests.get(f"{BASE_URL}/api/recommendations", timeout=10)
        if response.status_code == 401:
            print_success("Recommendations require authentication (HTTP 401)")
        else:
            print_failure("Recommendations require authentication", f"Expected HTTP 401, got HTTP {response.status_code}")
    except Exception as e:
        print_failure("Recommendations require authentication", str(e))

def test_portfolio():
    """Test portfolio endpoints"""
    print_header("PORTFOLIO ENDPOINTS")

    if test_data['token']:
        # Create portfolio
        print_test("Create portfolio")
        try:
            response = requests.post(
                f"{BASE_URL}/api/portfolio",
                headers={"Authorization": f"Bearer {test_data['token']}"},
                json={
                    "name": "Test Portfolio",
                    "description": "Automated test portfolio",
                    "initial_balance": 10000
                },
                timeout=10
            )

            if response.status_code in [200, 201]:
                data = response.json()
                if 'id' in data:
                    test_data['portfolio_id'] = data['id']
                    print_success(f"Create portfolio (ID: {test_data['portfolio_id']})")
                else:
                    print_failure("Create portfolio", "No portfolio ID in response")
            else:
                print_failure("Create portfolio", f"HTTP {response.status_code}")
        except Exception as e:
            print_failure("Create portfolio", str(e))

        # Get portfolios
        print_test("Get user portfolios")
        try:
            response = requests.get(
                f"{BASE_URL}/api/portfolio",
                headers={"Authorization": f"Bearer {test_data['token']}"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print_success("Get user portfolios")
                else:
                    print_failure("Get user portfolios", "No portfolios in response")
            else:
                print_failure("Get user portfolios", f"HTTP {response.status_code}")
        except Exception as e:
            print_failure("Get user portfolios", str(e))

def test_database():
    """Test database integration"""
    print_header("DATABASE INTEGRATION")

    # PostgreSQL connection
    print_test("PostgreSQL connection")
    try:
        result = subprocess.run(
            ['docker', 'exec', 'investment_db', 'psql', '-U', 'postgres', '-d', 'investment_db', '-c', 'SELECT 1;'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if '1 row' in result.stdout:
            print_success("PostgreSQL connection")
        else:
            print_failure("PostgreSQL connection", "Cannot connect to database")
    except Exception as e:
        print_failure("PostgreSQL connection", str(e))

    # TimescaleDB extension
    print_test("TimescaleDB extension")
    try:
        result = subprocess.run(
            ['docker', 'exec', 'investment_db', 'psql', '-U', 'postgres', '-d', 'investment_db',
             '-c', "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb';", '-t'],
            capture_output=True,
            text=True,
            timeout=10
        )

        count = result.stdout.strip()
        if count == '1':
            print_success("TimescaleDB extension enabled")
        else:
            print_failure("TimescaleDB extension enabled", "TimescaleDB not found")
    except Exception as e:
        print_failure("TimescaleDB extension enabled", str(e))

    # Check tables exist
    print_test("Required tables exist")
    try:
        result = subprocess.run(
            ['docker', 'exec', 'investment_db', 'psql', '-U', 'postgres', '-d', 'investment_db',
             '-c', "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';", '-t'],
            capture_output=True,
            text=True,
            timeout=10
        )

        count = int(result.stdout.strip())
        if count > 0:
            print_success(f"Database tables exist (count: {count})")
        else:
            print_failure("Database tables exist", "No tables found")
    except Exception as e:
        print_failure("Database tables exist", str(e))

def test_redis():
    """Test Redis integration"""
    print_header("REDIS CACHE INTEGRATION")

    # Redis connection
    print_test("Redis connection")
    try:
        result = subprocess.run(
            ['docker', 'exec', 'investment_cache', 'redis-cli', 'PING'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if 'PONG' in result.stdout:
            print_success("Redis connection")
        else:
            print_failure("Redis connection", "Redis did not respond with PONG")
    except Exception as e:
        print_failure("Redis connection", str(e))

def test_frontend():
    """Test frontend accessibility"""
    print_header("FRONTEND ACCESSIBILITY")

    check_response(f"{FRONTEND_URL}", 200, "Frontend homepage loads")

def test_performance():
    """Basic performance checks"""
    print_header("BASIC PERFORMANCE CHECKS")

    # Health endpoint response time
    print_test("Health endpoint response time")
    try:
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        elapsed = time.time() - start

        if elapsed < 1.0 and response.status_code == 200:
            print_success(f"Health endpoint response time: {elapsed:.3f}s (< 1s)")
        else:
            print_failure("Health endpoint response time", f"Took {elapsed:.3f}s (should be < 1s)")
    except Exception as e:
        print_failure("Health endpoint response time", str(e))

    # Stock data response time
    print_test("Stock data endpoint response time")
    try:
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/stocks/AAPL", timeout=10)
        elapsed = time.time() - start

        if elapsed < 2.0 and response.status_code == 200:
            print_success(f"Stock data response time: {elapsed:.3f}s (< 2s)")
        else:
            print_failure("Stock data response time", f"Took {elapsed:.3f}s (should be < 2s)")
    except Exception as e:
        print_failure("Stock data response time", str(e))

def test_security():
    """Basic security checks"""
    print_header("BASIC SECURITY CHECKS")

    # CORS headers
    print_test("CORS headers present")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        cors_headers = [h for h in response.headers if 'access-control' in h.lower()]

        if cors_headers:
            print_success("CORS headers present")
        else:
            print_failure("CORS headers present", "No CORS headers found")
    except Exception as e:
        print_failure("CORS headers present", str(e))

def print_summary():
    """Print test summary"""
    print_header("TEST SUMMARY")

    print(f"Total Tests:  {test_results['total']}")
    print(f"{Colors.GREEN}Passed:       {test_results['passed']}{Colors.NC}")
    print(f"{Colors.RED}Failed:       {test_results['failed']}{Colors.NC}")

    if test_results['failed'] > 0:
        print(f"\n{Colors.RED}Failed Tests:{Colors.NC}")
        for test in test_results['failed_tests']:
            print(f"  {Colors.RED}✗{Colors.NC} {test}")

    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    if test_results['failed'] == 0:
        print(f"{Colors.GREEN}✓ ALL TESTS PASSED!{Colors.NC}")
        print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")
        return 0
    else:
        print(f"{Colors.RED}✗ SOME TESTS FAILED{Colors.NC}")
        print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")
        return 1

def main():
    """Main test execution"""
    print_header(f"Investment Analysis Platform - Quick Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        test_pre_flight_checks()
        test_service_health()
        test_backend_health()
        test_authentication()
        test_stock_data()
        test_analysis()
        test_recommendations()
        test_portfolio()
        test_database()
        test_redis()
        test_frontend()
        test_performance()
        test_security()

        exit_code = print_summary()
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test execution interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.NC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
