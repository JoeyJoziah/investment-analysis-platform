#!/usr/bin/env python3
"""
Simple database initialization script
Works around path issues
"""

import sys
import os

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("Working directory:", os.getcwd())
print("Python path:", sys.path[0])

try:
    # Now import the actual initialization
    from backend.utils.db_init import DatabaseInitializer
    
    def main():
        """Initialize database with schema and data"""
        print("Initializing Investment Analysis Database...")
        
        initializer = DatabaseInitializer()
        success = initializer.initialize()
        
        if success:
            print("✅ Database initialization completed successfully!")
            return 0
        else:
            print("❌ Database initialization failed!")
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTrying alternative initialization method...")
    
    # Alternative: Direct database setup
    try:
        import subprocess
        
        # Check if PostgreSQL is available
        result = subprocess.run(['which', 'psql'], capture_output=True, text=True)
        if result.returncode == 0:
            print("PostgreSQL client found. Attempting direct initialization...")
            
            # Read .env file for database credentials
            env_vars = {}
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            env_vars[key] = value
            
            db_url = env_vars.get('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/investment_db')
            
            print(f"Using database URL: {db_url.split('@')[1]}")  # Hide password
            
            # Create database if it doesn't exist
            create_db_sql = """
            -- Create database if not exists
            SELECT 'CREATE DATABASE investment_db'
            WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'investment_db')\\gexec
            """
            
            # You can add more SQL initialization here
            print("Database initialization would run here...")
            print("For now, please run: ")
            print("  sudo -u postgres createdb investment_db")
            print("  sudo -u postgres psql investment_db < scripts/init_db.sql")
            
        else:
            print("PostgreSQL not found. Please install PostgreSQL first:")
            print("  sudo apt-get update")
            print("  sudo apt-get install postgresql postgresql-client")
            
    except Exception as e2:
        print(f"Alternative method failed: {e2}")
        print("\nManual initialization required:")
        print("1. Install PostgreSQL if not installed")
        print("2. Create database: sudo -u postgres createdb investment_db")
        print("3. Run migrations: alembic upgrade head")