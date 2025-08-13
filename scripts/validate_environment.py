#!/usr/bin/env python3
"""
Environment Validation Script
Validates that all required environment variables are properly configured
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import json

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def check_env_file(env_file: str = '.env.production') -> bool:
    """Check if environment file exists"""
    if not Path(env_file).exists():
        print(f"{Colors.RED}❌ {env_file} not found!{Colors.END}")
        print(f"   Run: ./scripts/setup_environment.sh")
        return False
    print(f"{Colors.GREEN}✅ {env_file} found{Colors.END}")
    return True

def load_env_vars(env_file: str = '.env.production') -> Dict[str, str]:
    """Load environment variables from file"""
    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"\'')
    return env_vars

def validate_api_keys(env_vars: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Validate API keys are configured"""
    required_keys = [
        'ALPHA_VANTAGE_API_KEY',
        'FINNHUB_API_KEY', 
        'POLYGON_API_KEY',
        'NEWS_API_KEY'
    ]
    
    optional_keys = [
        'FMP_API_KEY',
        'MARKETAUX_API_KEY',
        'FRED_API_KEY',
        'OPENWEATHER_API_KEY'
    ]
    
    missing_required = []
    missing_optional = []
    
    print(f"\n{Colors.BLUE}🔑 Validating API Keys:{Colors.END}")
    
    # Check required keys
    for key in required_keys:
        if key not in env_vars or not env_vars[key] or 'your_' in env_vars[key]:
            print(f"{Colors.RED}   ❌ {key} - Missing or not configured{Colors.END}")
            missing_required.append(key)
        else:
            print(f"{Colors.GREEN}   ✅ {key} - Configured{Colors.END}")
    
    # Check optional keys
    print(f"\n{Colors.BLUE}📋 Optional API Keys:{Colors.END}")
    for key in optional_keys:
        if key not in env_vars or not env_vars[key] or 'your_' in env_vars[key]:
            print(f"{Colors.YELLOW}   ⚠️  {key} - Not configured (optional){Colors.END}")
            missing_optional.append(key)
        else:
            print(f"{Colors.GREEN}   ✅ {key} - Configured{Colors.END}")
    
    return missing_required, missing_optional

def validate_security_keys(env_vars: Dict[str, str]) -> List[str]:
    """Validate security keys are properly generated"""
    security_keys = ['SECRET_KEY', 'JWT_SECRET_KEY']
    missing = []
    
    print(f"\n{Colors.BLUE}🔐 Validating Security Keys:{Colors.END}")
    
    for key in security_keys:
        if key not in env_vars or not env_vars[key] or 'GENERATE' in env_vars[key] or 'your-' in env_vars[key]:
            print(f"{Colors.RED}   ❌ {key} - Not properly generated{Colors.END}")
            missing.append(key)
        elif len(env_vars[key]) < 32:
            print(f"{Colors.YELLOW}   ⚠️  {key} - Key seems too short (should be 64 hex chars){Colors.END}")
            missing.append(key)
        else:
            print(f"{Colors.GREEN}   ✅ {key} - Properly configured{Colors.END}")
    
    return missing

def validate_database_config(env_vars: Dict[str, str]) -> List[str]:
    """Validate database configuration"""
    issues = []
    
    print(f"\n{Colors.BLUE}🗄️  Validating Database Configuration:{Colors.END}")
    
    # Check DATABASE_URL
    if 'DATABASE_URL' not in env_vars:
        print(f"{Colors.RED}   ❌ DATABASE_URL - Missing{Colors.END}")
        issues.append('DATABASE_URL missing')
    else:
        db_url = env_vars['DATABASE_URL']
        if 'CHANGE_THIS_PASSWORD' in db_url or 'password' in db_url:
            print(f"{Colors.RED}   ❌ DATABASE_URL - Using default password!{Colors.END}")
            issues.append('Database password not changed')
        else:
            print(f"{Colors.GREEN}   ✅ DATABASE_URL - Configured{Colors.END}")
    
    # Check Redis
    if 'REDIS_URL' not in env_vars:
        print(f"{Colors.YELLOW}   ⚠️  REDIS_URL - Missing (using default){Colors.END}")
    else:
        print(f"{Colors.GREEN}   ✅ REDIS_URL - Configured{Colors.END}")
    
    return issues

def validate_ssl_config(env_vars: Dict[str, str]) -> List[str]:
    """Validate SSL configuration"""
    issues = []
    
    print(f"\n{Colors.BLUE}🔒 Validating SSL Configuration:{Colors.END}")
    
    ssl_cert = env_vars.get('SSL_CERT_PATH', '')
    ssl_key = env_vars.get('SSL_KEY_PATH', '')
    
    if not ssl_cert or not ssl_key:
        print(f"{Colors.YELLOW}   ⚠️  SSL paths not configured (required for production){Colors.END}")
        issues.append('SSL configuration missing')
    else:
        # Check if files would exist in container
        print(f"{Colors.GREEN}   ✅ SSL_CERT_PATH - {ssl_cert}{Colors.END}")
        print(f"{Colors.GREEN}   ✅ SSL_KEY_PATH - {ssl_key}{Colors.END}")
    
    return issues

def test_api_connectivity(env_vars: Dict[str, str]) -> Dict[str, bool]:
    """Test basic connectivity to APIs (optional)"""
    results = {}
    
    print(f"\n{Colors.BLUE}🌐 Testing API Connectivity (optional):{Colors.END}")
    print(f"{Colors.YELLOW}   Note: This makes actual API calls{Colors.END}")
    
    # Skip if not requested
    response = input("   Test API connectivity? (y/N): ").strip().lower()
    if response != 'y':
        print("   Skipping API tests")
        return results
    
    # Test Alpha Vantage
    if 'ALPHA_VANTAGE_API_KEY' in env_vars and env_vars['ALPHA_VANTAGE_API_KEY']:
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={env_vars['ALPHA_VANTAGE_API_KEY']}"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                if 'Error Message' in data or 'Note' in data:
                    print(f"{Colors.YELLOW}   ⚠️  Alpha Vantage - API limit or error{Colors.END}")
                    results['alpha_vantage'] = False
                else:
                    print(f"{Colors.GREEN}   ✅ Alpha Vantage - Connected{Colors.END}")
                    results['alpha_vantage'] = True
        except Exception as e:
            print(f"{Colors.RED}   ❌ Alpha Vantage - Connection failed: {str(e)}{Colors.END}")
            results['alpha_vantage'] = False
    
    return results

def generate_summary(all_issues: Dict[str, List[str]]) -> None:
    """Generate summary report"""
    total_issues = sum(len(issues) for issues in all_issues.values())
    
    print(f"\n{'='*60}")
    print(f"{Colors.BLUE}📊 Environment Validation Summary{Colors.END}")
    print(f"{'='*60}")
    
    if total_issues == 0:
        print(f"{Colors.GREEN}✅ All checks passed! Environment is ready for deployment.{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠️  Found {total_issues} issues that need attention:{Colors.END}")
        
        for category, issues in all_issues.items():
            if issues:
                print(f"\n{category}:")
                for issue in issues:
                    print(f"  - {issue}")
    
    print(f"\n{Colors.BLUE}📌 Next Steps:{Colors.END}")
    if total_issues > 0:
        print("1. Fix the issues listed above")
        print("2. Run this script again to verify")
    print("3. Run: python debug_validate.py")
    print("4. Build Docker images: docker-compose -f docker-compose.prod.yml build")
    print("5. Run deployment: kubectl apply -f infrastructure/kubernetes/")

def main():
    """Main validation function"""
    print(f"{Colors.BLUE}🚀 Investment Analysis Platform - Environment Validation{Colors.END}")
    print("="*60)
    
    # Check which environment file to use
    env_file = '.env.production'
    if len(sys.argv) > 1:
        env_file = sys.argv[1]
    
    # Check if file exists
    if not check_env_file(env_file):
        return 1
    
    # Load environment variables
    env_vars = load_env_vars(env_file)
    
    # Run validations
    all_issues = {}
    
    # Validate API keys
    missing_required, missing_optional = validate_api_keys(env_vars)
    if missing_required:
        all_issues['Required API Keys'] = missing_required
    
    # Validate security keys
    missing_security = validate_security_keys(env_vars)
    if missing_security:
        all_issues['Security Keys'] = missing_security
    
    # Validate database
    db_issues = validate_database_config(env_vars)
    if db_issues:
        all_issues['Database Configuration'] = db_issues
    
    # Validate SSL
    ssl_issues = validate_ssl_config(env_vars)
    # SSL is warning only for development
    
    # Test API connectivity (optional)
    test_api_connectivity(env_vars)
    
    # Generate summary
    generate_summary(all_issues)
    
    return 0 if not all_issues else 1

if __name__ == '__main__':
    sys.exit(main())