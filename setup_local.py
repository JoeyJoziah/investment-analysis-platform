#!/usr/bin/env python3
"""
Local Development Setup Script
Prepares the investment analysis application for local development
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True):
    """Run a command and return the result"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def create_directories():
    """Create necessary directories for local development"""
    base_dir = Path.cwd()
    
    directories = [
        "secrets",
        "logs",
        "data",
        "data/cache",
        "data/backups",
        "reports",
        "reports/performance",
        "reports/coverage",
        "reports/security",
        ".pytest_cache",
        "backend/monitoring/dashboards",
        "backend/ml/models",
        "backend/ml/artifacts"
    ]
    
    print("Creating necessary directories...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    # Set proper permissions for secrets directory
    secrets_dir = base_dir / "secrets"
    if platform.system() != "Windows":
        os.chmod(secrets_dir, 0o700)
    print("✓ Set permissions for secrets directory")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # First, upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install dependencies
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    print("✓ Python dependencies installed")

def setup_environment():
    """Set up environment variables"""
    print("Setting up environment...")
    
    env_file = Path(".env.local")
    if env_file.exists():
        print("✓ .env.local already exists")
        return
    
    # Create local environment file
    env_content = """
# Local Development Environment
DEBUG=true
ENVIRONMENT=development

# Database Configuration (local PostgreSQL)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/investment_db_dev
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_db_dev
DB_USER=postgres
DB_PASSWORD=postgres

# Redis Configuration (local Redis)
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Security Configuration
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production
MASTER_SECRET_KEY=dev-master-secret-32-chars-long!!

# API Keys (set your actual keys)
ALPHA_VANTAGE_API_KEY=demo
FINNHUB_API_KEY=demo
POLYGON_API_KEY=demo
NEWS_API_KEY=demo

# Local paths (not Docker paths)
SECRETS_DIR=./secrets
LOGS_DIR=./logs
DATA_DIR=./data

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001

# Testing
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/investment_db_test
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content.strip())
    
    print(f"✓ Created {env_file}")
    print("⚠️  Please update API keys in .env.local with your actual keys")

def check_system_dependencies():
    """Check if system dependencies are available"""
    print("Checking system dependencies...")
    
    required_commands = ["python3", "pip", "git"]
    optional_commands = ["docker", "docker-compose", "psql", "redis-cli"]
    
    for cmd in required_commands:
        result = run_command(f"which {cmd}", check=False)
        if result.returncode == 0:
            print(f"✓ {cmd} is available")
        else:
            print(f"❌ {cmd} is required but not found")
            return False
    
    for cmd in optional_commands:
        result = run_command(f"which {cmd}", check=False)
        if result.returncode == 0:
            print(f"✓ {cmd} is available")
        else:
            print(f"⚠️  {cmd} is recommended but not found")
    
    return True

def create_local_migration_script():
    """Create a local-friendly migration script"""
    script_content = '''#!/usr/bin/env python3
"""
Local Development Migration Script
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment for local development
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRETS_DIR", str(Path.cwd() / "secrets"))
os.environ.setdefault("LOGS_DIR", str(Path.cwd() / "logs"))

def migrate_secrets():
    """Migrate secrets for local development"""
    from backend.security.secrets_manager import SecretsManager, SecretType
    from dotenv import load_dotenv
    
    # Load local environment
    load_dotenv('.env.local')
    
    secrets_manager = SecretsManager()
    
    # Migrate API keys from environment
    api_keys = [
        ('ALPHA_VANTAGE_API_KEY', SecretType.API_KEY),
        ('FINNHUB_API_KEY', SecretType.API_KEY),
        ('POLYGON_API_KEY', SecretType.API_KEY),
        ('NEWS_API_KEY', SecretType.API_KEY),
    ]
    
    for env_key, secret_type in api_keys:
        value = os.getenv(env_key)
        if value and value != 'demo':
            secrets_manager.store_secret(env_key, value, secret_type)
            print(f"✓ Migrated {env_key}")
        else:
            print(f"⚠️  {env_key} not set or using demo value")
    
    print("✓ Secrets migration completed")

def setup_database():
    """Set up local database"""
    try:
        import psycopg2
        from sqlalchemy import create_engine, text
        
        # Create database if it doesn't exist
        try:
            engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
            with engine.connect() as conn:
                conn.execute(text("COMMIT"))  # End any existing transaction
                conn.execute(text("CREATE DATABASE investment_db_dev"))
            print("✓ Created development database")
        except Exception as e:
            if "already exists" in str(e):
                print("✓ Development database already exists")
            else:
                print(f"Database creation error: {e}")
        
        # Create test database
        try:
            with engine.connect() as conn:
                conn.execute(text("COMMIT"))
                conn.execute(text("CREATE DATABASE investment_db_test"))
            print("✓ Created test database")
        except Exception as e:
            if "already exists" in str(e):
                print("✓ Test database already exists")
            else:
                print(f"Test database creation error: {e}")
                
    except ImportError:
        print("⚠️  PostgreSQL not available locally, skipping database setup")

if __name__ == "__main__":
    print("Starting local migration...")
    migrate_secrets()
    setup_database()
    print("✓ Local migration completed")
'''
    
    script_path = Path("scripts/migrate_local.py")
    script_path.parent.mkdir(exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod(script_path, 0o755)
    
    print(f"✓ Created {script_path}")

def main():
    """Main setup function"""
    print("🚀 Setting up Investment Analysis App for Local Development")
    print("=" * 60)
    
    # Check system dependencies
    if not check_system_dependencies():
        print("❌ Missing required system dependencies")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Set up environment
    setup_environment()
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"Warning: Dependency installation failed: {e}")
        print("You may need to install dependencies manually")
    
    # Create local migration script
    create_local_migration_script()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Update API keys in .env.local")
    print("2. Start local PostgreSQL and Redis (or use Docker)")
    print("3. Run: python3 scripts/migrate_local.py")
    print("4. Run tests: python3 -m pytest backend/tests/")
    print("5. Start the app: python3 -m uvicorn backend.api.main:app --reload")

if __name__ == "__main__":
    main()