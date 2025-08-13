#!/usr/bin/env python3
"""
CI/CD Setup Validation Script
Validates that all components of the CI/CD pipeline are properly configured
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import urllib.error

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print_success(f"Found: {filepath}")
        return True
    else:
        print_error(f"Missing: {filepath}")
        return False

def check_github_workflows() -> Tuple[int, int]:
    """Check if all GitHub workflow files exist"""
    print_header("GitHub Workflows Check")
    
    workflows = [
        ".github/workflows/ci.yml",
        ".github/workflows/staging-deploy.yml",
        ".github/workflows/production-deploy.yml",
        ".github/workflows/security-scan.yml",
        ".github/workflows/dependency-updates.yml",
        ".github/workflows/migration-check.yml",
        ".github/workflows/cleanup.yml",
        ".github/workflows/reusable-test.yml",
        ".github/workflows/reusable-build.yml"
    ]
    
    found = 0
    for workflow in workflows:
        if check_file_exists(workflow):
            found += 1
    
    return found, len(workflows)

def check_configuration_files() -> Tuple[int, int]:
    """Check if configuration files exist"""
    print_header("Configuration Files Check")
    
    configs = [
        ".github/dependabot.yml",
        ".github/codeql/codeql-config.yml",
        ".gitleaks.toml",
        ".github/pull_request_template.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml"
    ]
    
    found = 0
    for config in configs:
        if check_file_exists(config):
            found += 1
    
    return found, len(configs)

def check_docker_files() -> Tuple[int, int]:
    """Check Docker configuration files"""
    print_header("Docker Configuration Check")
    
    docker_files = [
        "Dockerfile.backend",
        "docker-compose.yml",
        "docker-compose.test.yml",
        "docker-compose.development.yml",
        "docker-compose.production.yml"
    ]
    
    found = 0
    for docker_file in docker_files:
        if check_file_exists(docker_file):
            found += 1
    
    return found, len(docker_files)

def check_environment_variables() -> Dict[str, bool]:
    """Check which environment variables are set"""
    print_header("Environment Variables Check")
    
    required_vars = [
        "DATABASE_URL",
        "REDIS_URL",
        "ALPHA_VANTAGE_API_KEY",
        "FINNHUB_API_KEY",
        "POLYGON_API_KEY",
        "NEWS_API_KEY"
    ]
    
    env_status = {}
    for var in required_vars:
        if os.getenv(var):
            print_success(f"{var} is set")
            env_status[var] = True
        else:
            print_warning(f"{var} is not set (will need to be added as GitHub Secret)")
            env_status[var] = False
    
    return env_status

def check_python_dependencies() -> bool:
    """Check if Python dependencies are installed"""
    print_header("Python Dependencies Check")
    
    try:
        # Check if requirements.txt exists
        if not Path("requirements.txt").exists():
            print_error("requirements.txt not found")
            return False
        
        print_success("requirements.txt found")
        
        # Check key packages
        packages = ["fastapi", "pytest", "black", "flake8", "mypy"]
        missing = []
        
        for package in packages:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print_success(f"{package} is installed")
            else:
                print_warning(f"{package} is not installed")
                missing.append(package)
        
        if missing:
            print_info(f"Install missing packages with: pip install {' '.join(missing)}")
            return False
        
        return True
    except Exception as e:
        print_error(f"Error checking Python dependencies: {e}")
        return False

def check_nodejs_setup() -> bool:
    """Check Node.js and frontend setup"""
    print_header("Node.js/Frontend Check")
    
    frontend_path = Path("frontend/web")
    
    if not frontend_path.exists():
        print_error("frontend/web directory not found")
        return False
    
    print_success("frontend/web directory found")
    
    package_json = frontend_path / "package.json"
    if package_json.exists():
        print_success("package.json found")
        
        # Check if node_modules exists
        node_modules = frontend_path / "node_modules"
        if node_modules.exists():
            print_success("node_modules found (dependencies installed)")
        else:
            print_warning("node_modules not found - run 'npm install' in frontend/web")
            return False
    else:
        print_error("package.json not found in frontend/web")
        return False
    
    return True

def check_database_connection() -> bool:
    """Check if database is accessible"""
    print_header("Database Connection Check")
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print_warning("DATABASE_URL not set - skipping database check")
        print_info("Set DATABASE_URL to test database connectivity")
        return False
    
    try:
        # Try to connect using psycopg2 if available
        import psycopg2
        conn = psycopg2.connect(db_url)
        conn.close()
        print_success("Successfully connected to PostgreSQL database")
        return True
    except ImportError:
        print_warning("psycopg2 not installed - cannot test database connection")
        print_info("Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        print_error(f"Failed to connect to database: {e}")
        return False

def check_redis_connection() -> bool:
    """Check if Redis is accessible"""
    print_header("Redis Connection Check")
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        print_success("Successfully connected to Redis")
        return True
    except ImportError:
        print_warning("redis package not installed - cannot test Redis connection")
        print_info("Install with: pip install redis")
        return False
    except Exception as e:
        print_error(f"Failed to connect to Redis: {e}")
        return False

def test_api_keys() -> Dict[str, bool]:
    """Test if API keys are valid by making test requests"""
    print_header("API Keys Validation")
    
    results = {}
    
    # Test Alpha Vantage
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if av_key:
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={av_key}"
            response = urllib.request.urlopen(url)
            data = json.loads(response.read())
            if "Global Quote" in data:
                print_success("Alpha Vantage API key is valid")
                results["alpha_vantage"] = True
            else:
                print_error("Alpha Vantage API key might be invalid")
                results["alpha_vantage"] = False
        except Exception as e:
            print_error(f"Alpha Vantage API test failed: {e}")
            results["alpha_vantage"] = False
    else:
        print_warning("ALPHA_VANTAGE_API_KEY not set")
        results["alpha_vantage"] = False
    
    # Test Finnhub
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_key:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key}"
            response = urllib.request.urlopen(url)
            data = json.loads(response.read())
            if "c" in data:  # Current price
                print_success("Finnhub API key is valid")
                results["finnhub"] = True
            else:
                print_error("Finnhub API key might be invalid")
                results["finnhub"] = False
        except Exception as e:
            print_error(f"Finnhub API test failed: {e}")
            results["finnhub"] = False
    else:
        print_warning("FINNHUB_API_KEY not set")
        results["finnhub"] = False
    
    return results

def generate_summary(results: Dict) -> None:
    """Generate a summary report"""
    print_header("CI/CD Setup Summary")
    
    total_score = 0
    max_score = 0
    
    # Calculate scores
    if "workflows" in results:
        found, total = results["workflows"]
        total_score += found
        max_score += total
        print(f"GitHub Workflows: {found}/{total}")
    
    if "configs" in results:
        found, total = results["configs"]
        total_score += found
        max_score += total
        print(f"Configuration Files: {found}/{total}")
    
    if "docker" in results:
        found, total = results["docker"]
        total_score += found
        max_score += total
        print(f"Docker Files: {found}/{total}")
    
    if "python" in results:
        if results["python"]:
            print_success("Python Dependencies: Ready")
            total_score += 5
        else:
            print_warning("Python Dependencies: Incomplete")
        max_score += 5
    
    if "nodejs" in results:
        if results["nodejs"]:
            print_success("Node.js/Frontend: Ready")
            total_score += 5
        else:
            print_warning("Node.js/Frontend: Not Ready")
        max_score += 5
    
    if "database" in results:
        if results["database"]:
            print_success("Database: Connected")
            total_score += 5
        else:
            print_warning("Database: Not Connected")
        max_score += 5
    
    if "redis" in results:
        if results["redis"]:
            print_success("Redis: Connected")
            total_score += 5
        else:
            print_warning("Redis: Not Connected")
        max_score += 5
    
    # Overall score
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    
    print(f"\n{Colors.BOLD}Overall Setup Score: {total_score}/{max_score} ({percentage:.1f}%){Colors.END}")
    
    if percentage >= 90:
        print_success("Excellent! Your CI/CD pipeline is nearly complete!")
    elif percentage >= 70:
        print_success("Good progress! Just a few more steps to complete.")
    elif percentage >= 50:
        print_warning("Halfway there! Continue with the setup guide.")
    else:
        print_warning("Just getting started. Follow the setup guide for next steps.")
    
    print("\n" + "="*60)
    print(f"{Colors.BOLD}Next Steps:{Colors.END}")
    
    if percentage < 100:
        print("1. Add missing GitHub Secrets in repository settings")
        print("2. Configure branch protection rules")
        print("3. Test the pipeline with a pull request")
        print("4. Set up Kubernetes cluster for deployment")
        print("5. Configure monitoring and notifications")
    else:
        print_success("Your CI/CD pipeline is fully configured!")
        print("1. Create a feature branch and test the pipeline")
        print("2. Deploy to staging environment")
        print("3. Set up production deployment")

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}{Colors.GREEN}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Investment Analysis App CI/CD Validator         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    results = {}
    
    # Check GitHub workflows
    results["workflows"] = check_github_workflows()
    
    # Check configuration files
    results["configs"] = check_configuration_files()
    
    # Check Docker files
    results["docker"] = check_docker_files()
    
    # Check environment variables
    results["env_vars"] = check_environment_variables()
    
    # Check Python setup
    results["python"] = check_python_dependencies()
    
    # Check Node.js setup
    results["nodejs"] = check_nodejs_setup()
    
    # Check database connection
    results["database"] = check_database_connection()
    
    # Check Redis connection
    results["redis"] = check_redis_connection()
    
    # Test API keys
    results["api_keys"] = test_api_keys()
    
    # Generate summary
    generate_summary(results)

if __name__ == "__main__":
    main()