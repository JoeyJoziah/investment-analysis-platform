#!/usr/bin/env python3
"""
Diagnostic script for import issues
Helps troubleshoot module import problems
"""

import sys
import subprocess
import importlib.util

def diagnose_import(module_name):
    """Diagnose import issues for a specific module"""
    print(f"\n{'='*50}")
    print(f"Diagnosing: {module_name}")
    print('='*50)
    
    # Check if module is installed
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {module_name} is NOT installed")
        
        # Suggest installation command
        if module_name == "psycopg2":
            print(f"   To install: pip3 install psycopg2-binary")
        elif module_name == "dotenv":
            print(f"   To install: pip3 install python-dotenv")
        else:
            print(f"   To install: pip3 install {module_name}")
        
        # Check pip list
        try:
            result = subprocess.run(['pip3', 'list'], capture_output=True, text=True)
            if module_name in result.stdout:
                print(f"⚠️  Module appears in pip list but can't be imported")
                print("   Try: python3 -m pip install --upgrade --force-reinstall", module_name)
        except:
            pass
    else:
        print(f"✅ {module_name} is installed")
        print(f"   Location: {spec.origin}")
        
        # Try to import and get version
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, '__version__'):
                print(f"   Version: {module.__version__}")
            print(f"✅ Import successful!")
        except ImportError as e:
            print(f"❌ Import failed: {e}")

def main():
    print("="*60)
    print("PYTHON IMPORT DIAGNOSTIC TOOL")
    print("="*60)
    
    # Print Python information
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path:")
    for path in sys.path[:5]:  # Show first 5 paths
        print(f"  - {path}")
    
    # Check required modules
    modules_to_check = [
        "psycopg2",
        "redis", 
        "requests",
        "dotenv"
    ]
    
    for module in modules_to_check:
        diagnose_import(module)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Quick verification
    all_ok = True
    try:
        import psycopg2
        import redis
        import requests
        from dotenv import load_dotenv
        print("✅ All required imports are working!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        all_ok = False
    
    if not all_ok:
        print("\nTo fix import issues:")
        print("1. Run: pip3 install -r requirements.txt")
        print("2. Or run: ./install_test_dependencies.sh")
        print("3. If issues persist, try:")
        print("   python3 -m pip install --upgrade pip")
        print("   python3 -m pip install --upgrade --force-reinstall psycopg2-binary redis requests python-dotenv")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())