#!/usr/bin/env python3
"""
Validate environment variables have been properly transferred
"""
import os
import re
from pathlib import Path

def validate_env_file(file_path):
    """Validate a single .env file"""
    print(f"\nValidating: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    issues = []
    valid_vars = []
    
    # Check for critical variables
    critical_vars = {
        'SECRET_KEY': r'^[a-f0-9]{64}$',
        'JWT_SECRET_KEY': r'^[a-f0-9]{64}$',
        'DB_PASSWORD': r'^[^\s]+$',  # Non-empty, no spaces
        'POSTGRES_PASSWORD': r'^[^\s]+$',
        'REDIS_PASSWORD': r'^[^\s]+$',
        'ALPHA_VANTAGE_API_KEY': r'^[A-Z0-9]+$',
        'FINNHUB_API_KEY': r'^[a-z0-9]+$',
        'POLYGON_API_KEY': r'^[a-zA-Z0-9_]+$',
        'NEWS_API_KEY': r'^[a-f0-9]{32}$',
        'FMP_API_KEY': r'^[a-zA-Z0-9]+$',
        'MARKETAUX_API_KEY': r'^[a-zA-Z0-9]+$',
        'FRED_API_KEY': r'^[a-f0-9]{32}$',
        'OPENWEATHER_API_KEY': r'^[a-f0-9]{32}$'
    }
    
    for line in content.split('\n'):
        if '=' in line and not line.strip().startswith('#'):
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Check if it's a critical variable
            if key in critical_vars:
                if 'CHANGE_THIS' in value or 'demo' in value.lower() or not value:
                    issues.append(f"❌ {key} still has placeholder value or is empty")
                elif key in ['DB_PASSWORD', 'POSTGRES_PASSWORD', 'REDIS_PASSWORD']:
                    # For passwords, just check they're not empty and look reasonable
                    if len(value) > 0 and value != 'postgres':
                        valid_vars.append(f"✅ {key} = [SECURED - {len(value)} chars]")
                    else:
                        issues.append(f"❌ {key} appears to be default or weak")
                elif not re.match(critical_vars[key], value):
                    issues.append(f"⚠️  {key} format might be incorrect: '{value[:20]}...'")
                else:
                    valid_vars.append(f"✅ {key} = {value[:20]}...")
    
    # Print results
    if valid_vars:
        print("Valid variables found:")
        for var in valid_vars:
            print(f"  {var}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ All critical variables appear to be properly set!")
        return True

def main():
    """Main validation function"""
    print("Environment Variables Validation Report")
    print("=" * 60)
    
    env_files = [
        '.env',
        '.env.secure',
        'scripts/.env',
    ]
    
    all_valid = True
    for env_file in env_files:
        if not validate_env_file(env_file):
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ SUCCESS: All environment files are properly configured!")
        print("\n⚠️  IMPORTANT: Remember to change these exposed passwords later as mentioned.")
    else:
        print("❌ ISSUES FOUND: Some environment variables need attention.")
    
    return all_valid

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)