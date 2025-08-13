#!/usr/bin/env python3
"""
Database Credentials Setup Helper
Helps generate secure database credentials for production
"""

import os
import secrets
import string
import sys
from pathlib import Path

def generate_password(length=32):
    """Generate a secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    # Avoid problematic characters in passwords
    alphabet = alphabet.replace("'", "").replace('"', "").replace("\\", "")
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def update_env_file(env_file, updates):
    """Update environment file with new values"""
    lines = []
    updated_keys = set()
    
    # Read existing file
    if Path(env_file).exists():
        with open(env_file, 'r') as f:
            for line in f:
                key_found = False
                for key, value in updates.items():
                    if line.strip().startswith(f"{key}="):
                        lines.append(f"{key}={value}\n")
                        updated_keys.add(key)
                        key_found = True
                        break
                if not key_found:
                    lines.append(line)
    
    # Add any missing keys
    for key, value in updates.items():
        if key not in updated_keys:
            lines.append(f"{key}={value}\n")
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(lines)

def main():
    print("🔐 Database Credentials Setup")
    print("=" * 40)
    
    # Determine environment
    env_file = ".env.production"
    if len(sys.argv) > 1:
        env_file = sys.argv[1]
    
    print(f"Configuring: {env_file}")
    print()
    
    # Production database configuration
    print("Choose database setup:")
    print("1. Managed Database Service (DigitalOcean, AWS RDS, etc.)")
    print("2. Self-hosted PostgreSQL")
    print("3. Docker Compose (development)")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    updates = {}
    
    if choice == "1":
        print("\n📊 Managed Database Configuration")
        print("-" * 40)
        
        db_host = input("Database Host (e.g., db-cluster-xxx.ondigitalocean.com): ").strip()
        db_port = input("Database Port [25060]: ").strip() or "25060"
        db_name = input("Database Name [investment_db]: ").strip() or "investment_db"
        db_user = input("Database User [doadmin]: ").strip() or "doadmin"
        
        # For managed databases, password is usually provided
        db_password = input("Database Password (provided by service): ").strip()
        
        # Generate application user credentials
        app_user = "investment_app"
        app_password = generate_password()
        readonly_password = generate_password()
        
        # Build connection URLs
        admin_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
        app_url = f"postgresql://{app_user}:{app_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
        
        updates = {
            "DATABASE_URL": app_url,
            "DATABASE_ADMIN_URL": admin_url,
            "DB_HOST": db_host,
            "DB_PORT": db_port,
            "DB_NAME": db_name,
            "DB_USER": app_user,
            "DB_PASSWORD": app_password,
            "DB_ADMIN_USER": db_user,
            "DB_ADMIN_PASSWORD": db_password,
            "DB_SSL_MODE": "require",
            "DB_READONLY_PASSWORD": readonly_password
        }
        
        print("\n✅ Generated secure passwords for application users")
        print("\n⚠️  IMPORTANT: Save these credentials securely!")
        print(f"App User: {app_user}")
        print(f"App Password: {app_password}")
        print(f"Readonly Password: {readonly_password}")
        
        print("\n📝 Next Steps:")
        print("1. Connect to database as admin:")
        print(f"   psql '{admin_url}'")
        print("\n2. Create application user:")
        print(f"   CREATE USER {app_user} WITH ENCRYPTED PASSWORD '{app_password}';")
        print(f"   GRANT CONNECT ON DATABASE {db_name} TO {app_user};")
        print(f"   GRANT USAGE ON SCHEMA public TO {app_user};")
        print(f"   GRANT CREATE ON SCHEMA public TO {app_user};")
        print(f"   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {app_user};")
        print(f"   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {app_user};")
        
    elif choice == "2":
        print("\n🖥️  Self-Hosted PostgreSQL Configuration")
        print("-" * 40)
        
        db_host = input("Database Host [localhost]: ").strip() or "localhost"
        db_port = input("Database Port [5432]: ").strip() or "5432"
        db_name = input("Database Name [investment_db]: ").strip() or "investment_db"
        
        # Generate all passwords
        postgres_password = generate_password()
        app_password = generate_password()
        readonly_password = generate_password()
        
        app_user = "investment_app"
        
        # Build connection URL
        app_url = f"postgresql://{app_user}:{app_password}@{db_host}:{db_port}/{db_name}"
        
        updates = {
            "DATABASE_URL": app_url,
            "DB_HOST": db_host,
            "DB_PORT": db_port,
            "DB_NAME": db_name,
            "DB_USER": app_user,
            "DB_PASSWORD": app_password,
            "DB_POSTGRES_PASSWORD": postgres_password,
            "DB_READONLY_PASSWORD": readonly_password,
            "DB_SSL_MODE": "prefer"
        }
        
        print("\n✅ Generated secure passwords")
        print(f"Postgres Password: {postgres_password}")
        print(f"App Password: {app_password}")
        print(f"Readonly Password: {readonly_password}")
        
    else:
        print("\n🐳 Docker Compose Configuration")
        print("-" * 40)
        
        # For Docker, generate a simple password
        db_password = generate_password(24)
        
        updates = {
            "DATABASE_URL": f"postgresql://postgres:{db_password}@postgres:5432/investment_db",
            "DB_HOST": "postgres",
            "DB_PORT": "5432",
            "DB_NAME": "investment_db",
            "DB_USER": "postgres",
            "DB_PASSWORD": db_password,
            "POSTGRES_PASSWORD": db_password,
            "DB_SSL_MODE": "disable"
        }
        
        print(f"\n✅ Generated password: {db_password}")
    
    # Redis configuration
    print("\n🔴 Redis Configuration")
    print("-" * 40)
    redis_password = generate_password(24)
    
    redis_host = input("Redis Host [redis]: ").strip() or "redis"
    redis_port = input("Redis Port [6379]: ").strip() or "6379"
    
    updates["REDIS_URL"] = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
    updates["REDIS_PASSWORD"] = redis_password
    
    # Update environment file
    print(f"\n📝 Updating {env_file}...")
    update_env_file(env_file, updates)
    
    print("✅ Configuration updated successfully!")
    
    # Create backup
    backup_file = f"{env_file}.backup"
    if Path(env_file).exists():
        import shutil
        shutil.copy(env_file, backup_file)
        print(f"📦 Backup created: {backup_file}")
    
    print("\n🔒 Security Reminders:")
    print("- Never commit .env files to version control")
    print("- Store passwords in a secure password manager")
    print("- Enable audit logging in production")
    print("- Use SSL/TLS for all database connections")
    print("- Implement IP whitelisting for database access")
    
    print("\n✅ Database credentials configured!")
    print(f"Run './scripts/init_database.sh' to initialize the database")

if __name__ == "__main__":
    main()