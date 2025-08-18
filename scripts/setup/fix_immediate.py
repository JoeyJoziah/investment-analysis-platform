#!/usr/bin/env python3
"""
Immediate Fix Script
Addresses the specific errors you encountered
"""

import os
import sys
from pathlib import Path

def fix_secrets_path():
    """Fix the secrets manager path issue"""
    print("Fixing secrets manager path...")
    
    # Create the secrets directory
    secrets_dir = Path("secrets")
    secrets_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created secrets directory: {secrets_dir.absolute()}")
    
    # Set environment variable
    os.environ["SECRETS_DIR"] = str(secrets_dir.absolute())
    print(f"✓ Set SECRETS_DIR environment variable")

def create_minimal_migrate_script():
    """Create a working migration script"""
    script_content = '''#!/usr/bin/env python3
"""
Minimal Migration Script - Works without dependencies
"""
import os
import sys
from pathlib import Path

def main():
    print("Starting minimal migration...")
    
    # Create necessary directories
    directories = ["secrets", "logs", "data", "reports"]
    for dirname in directories:
        Path(dirname).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dirname}")
    
    # Set environment variables for this session
    os.environ["SECRETS_DIR"] = str(Path.cwd() / "secrets")
    os.environ["LOGS_DIR"] = str(Path.cwd() / "logs")
    os.environ["DATA_DIR"] = str(Path.cwd() / "data")
    os.environ.setdefault("MASTER_SECRET_KEY", "dev-master-secret-32-chars-long!!")
    
    print("✓ Set environment variables")
    print("✓ Minimal migration completed!")
    
    print("\\nTo run the application:")
    print("1. Install dependencies in virtual environment")
    print("2. Set your API keys in .env file")
    print("3. Set up PostgreSQL and Redis")

if __name__ == "__main__":
    main()
'''
    
    Path("scripts").mkdir(exist_ok=True)
    with open("scripts/migrate_minimal.py", "w") as f:
        f.write(script_content)
    print("✓ Created minimal migration script")

def create_env_file():
    """Create environment file with proper paths"""
    env_content = '''# Investment Analysis App - Development Environment
DEBUG=true
ENVIRONMENT=development

# Security Keys (change in production)
MASTER_SECRET_KEY=dev-master-secret-32-chars-long!!
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production

# Local Directory Paths (relative to project root)
SECRETS_DIR=./secrets
LOGS_DIR=./logs
DATA_DIR=./data

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/investment_db_dev
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_db_dev
DB_USER=postgres
DB_PASSWORD=postgres

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# External API Keys (replace with your actual keys)
ALPHA_VANTAGE_API_KEY=demo
FINNHUB_API_KEY=demo
POLYGON_API_KEY=demo
NEWS_API_KEY=demo

# Optional: Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
'''
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        print("✓ Created .env file with local paths")
    else:
        print("✓ .env file already exists")

def create_install_guide():
    """Create step-by-step installation guide"""
    guide = '''# Investment Analysis App - Installation Guide

## Quick Start (Ubuntu/WSL)

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 2. Install Core Dependencies
```bash
pip install --upgrade pip
pip install python-dotenv sqlalchemy[asyncio] asyncpg psycopg2-binary
pip install redis pydantic fastapi uvicorn cryptography
```

### 3. Install Additional Dependencies (Optional)
```bash
pip install -r requirements-clean.txt
```

### 4. Set Up Environment
```bash
python scripts/migrate_minimal.py
```

### 5. Update API Keys
Edit the `.env` file and replace demo values with your actual API keys:
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- Finnhub: https://finnhub.io/dashboard
- Polygon: https://polygon.io/dashboard
- News API: https://newsapi.org/

### 6. Set Up Databases (Local Development)

#### PostgreSQL:
```bash
# Ubuntu/WSL
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo -u postgres createuser --superuser $USER
sudo -u postgres createdb investment_db_dev
```

#### Redis:
```bash
# Ubuntu/WSL  
sudo apt install redis-server
sudo systemctl start redis-server
```

### 7. Test Installation
```bash
python -c "import fastapi, sqlalchemy, redis; print('Core dependencies OK')"
```

### 8. Run Application
```bash
python -m uvicorn backend.api.main:app --reload
```

## Troubleshooting

### Common Issues:

1. **Module not found errors**: Make sure virtual environment is activated
2. **Database connection errors**: Check PostgreSQL is running and credentials in .env
3. **Redis connection errors**: Check Redis is running on localhost:6379
4. **Permission errors**: Check directory permissions (especially secrets/)

### Testing Without Databases:
The application can run in demo mode without external databases for testing the setup.
'''
    
    with open("INSTALLATION.md", "w") as f:
        f.write(guide)
    print("✓ Created installation guide")

def main():
    """Run immediate fixes"""
    print("🔧 Applying immediate fixes for setup issues")
    print("=" * 50)
    
    fix_secrets_path()
    create_minimal_migrate_script()
    create_env_file()
    create_install_guide()
    
    print("\n✅ Immediate fixes applied!")
    print("\nNext steps:")
    print("1. Run: python scripts/migrate_minimal.py")
    print("2. Follow instructions in INSTALLATION.md")
    print("3. Set up virtual environment and install dependencies")

if __name__ == "__main__":
    main()