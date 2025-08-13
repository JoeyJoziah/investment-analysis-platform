#!/usr/bin/env python3
"""
Quick Setup Script for Investment Analysis App
Handles dependencies, paths, and basic configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def install_minimal_dependencies():
    """Install minimal dependencies needed for setup"""
    print("Installing minimal dependencies...")
    
    minimal_deps = [
        "python-dotenv",
        "pathlib",
        "cryptography>=41.0.0",
        "sqlalchemy>=2.0.0",
        "asyncpg",
        "psycopg2-binary",
        "redis",
        "pydantic",
        "fastapi",
        "uvicorn"
    ]
    
    for dep in minimal_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print(f"Warning: Could not install {dep}")

def create_env_file():
    """Create a basic .env file for development"""
    env_content = """# Development Environment
DEBUG=true
ENVIRONMENT=development

# Security
MASTER_SECRET_KEY=dev-master-secret-32-chars-long!!
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production

# Local paths
SECRETS_DIR=./secrets
LOGS_DIR=./logs
DATA_DIR=./data

# Database (adjust for your local setup)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/investment_db_dev
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_db_dev
DB_USER=postgres
DB_PASSWORD=postgres

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys (replace with actual keys)
ALPHA_VANTAGE_API_KEY=demo
FINNHUB_API_KEY=demo
POLYGON_API_KEY=demo
NEWS_API_KEY=demo
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✓ Created .env file")
    else:
        print("✓ .env file already exists")

def create_directories():
    """Create necessary directories"""
    directories = [
        "secrets",
        "logs", 
        "data",
        "data/cache",
        "reports",
        "backend/ml/models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_simple_migrate_script():
    """Create a simple migration script that works locally"""
    migrate_script = '''#!/usr/bin/env python3
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
    print("\\nNext steps:")
    print("1. Install full dependencies: pip install -r requirements-clean.txt")
    print("2. Set up PostgreSQL and Redis locally")
    print("3. Update API keys in .env file") 
    print("4. Run tests: python -m pytest backend/tests/ -v")

if __name__ == "__main__":
    main()
'''
    
    Path("scripts").mkdir(exist_ok=True)
    script_path = Path("scripts/simple_migrate.py")
    with open(script_path, 'w') as f:
        f.write(migrate_script)
    
    print(f"✓ Created {script_path}")

def main():
    """Main setup function"""
    print("🚀 Quick Setup for Investment Analysis App")
    print("=" * 50)
    
    # Install minimal dependencies first
    install_minimal_dependencies()
    
    # Create basic structure
    create_directories()
    create_env_file()
    create_simple_migrate_script()
    
    print("\\n✅ Quick setup completed!")
    print("\\nNext steps:")
    print("1. Run: python scripts/simple_migrate.py")
    print("2. Install full dependencies: pip install -r requirements-clean.txt")
    print("3. Set up databases (PostgreSQL, Redis)")
    print("4. Update .env with your API keys")
    print("5. Run: python -m uvicorn backend.api.main:app --reload")

if __name__ == "__main__":
    main()