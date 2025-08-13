#!/usr/bin/env python3
"""
Simple Local Migration Script
"""
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["SECRETS_DIR"] = str(project_root / "secrets")
os.environ.setdefault("MASTER_SECRET_KEY", "dev-master-secret-32-chars-long!!")

def main():
    """Run basic setup"""
    print("Starting basic migration...")
    
    # Create secrets directory
    secrets_dir = Path(os.environ["SECRETS_DIR"])
    secrets_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created secrets directory: {secrets_dir}")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded environment variables")
    except ImportError:
        print("⚠️  python-dotenv not available, skipping .env loading")
    
    print("✓ Basic migration completed!")
    print("\nNext steps:")
    print("1. Install full dependencies: pip install -r requirements-clean.txt")
    print("2. Set up PostgreSQL and Redis locally")
    print("3. Update API keys in .env file") 
    print("4. Run tests: python -m pytest backend/tests/ -v")

if __name__ == "__main__":
    main()
