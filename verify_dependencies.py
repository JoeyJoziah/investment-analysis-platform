#!/usr/bin/env python3
"""
Dependency Verification Script
Verifies all required packages for test_all_passwords.py are installed
"""

import sys
import importlib
from typing import Tuple, List

def check_module(module_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a module is installed and can be imported"""
    try:
        if import_name:
            # For packages like python-dotenv that import as dotenv
            module = importlib.import_module(import_name)
        else:
            module = importlib.import_module(module_name)
        
        version = "Unknown"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        elif module_name == 'psycopg2':
            # Special case for psycopg2
            import psycopg2
            version = psycopg2.__version__
        
        return True, f"v{version}"
    except ImportError as e:
        return False, str(e)

def main():
    """Check all required dependencies"""
    print("=" * 60)
    print("DEPENDENCY VERIFICATION FOR test_all_passwords.py")
    print("=" * 60)
    
    # List of required packages (pip name, import name if different)
    packages = [
        ("psycopg2", None),
        ("redis", None),
        ("requests", None),
        ("python-dotenv", "dotenv"),
    ]
    
    all_ok = True
    results = []
    
    for pip_name, import_name in packages:
        check_name = import_name if import_name else pip_name
        success, info = check_module(check_name)
        
        if success:
            status = "✅ Installed"
            print(f"{pip_name:20} {status:15} {info}")
        else:
            status = "❌ Missing"
            print(f"{pip_name:20} {status:15}")
            all_ok = False
        
        results.append((pip_name, success, info))
    
    print("=" * 60)
    
    if all_ok:
        print("\n✅ All dependencies are installed!")
        print("\nYou can run the test script with:")
        print("  python3 test_all_passwords.py")
        
        # Try to import all at once as final verification
        try:
            import psycopg2
            import redis
            import requests
            from dotenv import load_dotenv
            print("\n✅ All imports verified successfully!")
        except ImportError as e:
            print(f"\n⚠️ Import error during verification: {e}")
            return 1
    else:
        print("\n❌ Some dependencies are missing!")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        for pip_name, success, _ in results:
            if not success:
                print(f"  pip install {pip_name}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())