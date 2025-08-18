#!/usr/bin/env python3
"""
Validate project structure and imports
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING")
        return False

def check_imports(filepath, imports, description):
    """Check if a file contains specific imports"""
    if not os.path.exists(filepath):
        print(f"✗ {description}: File not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    missing = []
    for imp in imports:
        if imp not in content:
            missing.append(imp)
    
    if missing:
        print(f"✗ {description}: Missing imports: {missing}")
        return False
    else:
        print(f"✓ {description}: All imports present")
        return True

def main():
    """Run validation checks"""
    print("=" * 60)
    print("INVESTMENT ANALYSIS PLATFORM - STRUCTURE VALIDATION")
    print("=" * 60)
    
    errors = 0
    
    # Check critical files
    print("\n1. CHECKING CRITICAL FILES:")
    print("-" * 40)
    
    critical_files = [
        ("backend/api/main.py", "Main FastAPI application"),
        ("backend/config/settings.py", "Settings configuration"),
        ("backend/utils/database.py", "Database utilities"),
        ("backend/utils/cache.py", "Cache utilities"),
        ("backend/tasks/celery_app.py", "Celery configuration"),
        ("backend/tasks/scheduler.py", "Async scheduler"),
        ("backend/models/tables.py", "Database models"),
        ("backend/models/schemas.py", "Pydantic schemas"),
        (".env", "Environment variables"),
        ("docker-compose.yml", "Docker compose config"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    for filepath, desc in critical_files:
        if not check_file_exists(filepath, desc):
            errors += 1
    
    # Check data ingestion clients
    print("\n2. CHECKING DATA INGESTION CLIENTS:")
    print("-" * 40)
    
    clients = [
        "backend/data_ingestion/alpha_vantage_client.py",
        "backend/data_ingestion/finnhub_client.py",
        "backend/data_ingestion/polygon_client.py",
    ]
    
    for client in clients:
        if not check_file_exists(client, f"Client: {os.path.basename(client)}"):
            errors += 1
    
    # Check API routers
    print("\n3. CHECKING API ROUTERS:")
    print("-" * 40)
    
    routers = [
        "health", "auth", "stocks", "analysis", 
        "recommendations", "portfolio", "websocket", "admin"
    ]
    
    for router in routers:
        filepath = f"backend/api/routers/{router}.py"
        if not check_file_exists(filepath, f"Router: {router}"):
            errors += 1
    
    # Check Celery tasks
    print("\n4. CHECKING CELERY TASKS:")
    print("-" * 40)
    
    tasks = [
        "data_tasks", "analysis_tasks", "portfolio_tasks",
        "notification_tasks", "maintenance_tasks"
    ]
    
    for task in tasks:
        filepath = f"backend/tasks/{task}.py"
        if not check_file_exists(filepath, f"Task: {task}"):
            errors += 1
    
    # Check imports in main.py
    print("\n5. CHECKING MAIN.PY IMPORTS:")
    print("-" * 40)
    
    main_imports = [
        "from backend.api.routers import",
        "from backend.utils.database import init_db",
        "from backend.utils.cache import init_cache",
        "from backend.tasks.scheduler import start_scheduler",
        "from backend.config.settings import settings"
    ]
    
    if not check_imports("backend/api/main.py", main_imports, "Main.py imports"):
        errors += 1
    
    # Check environment variables
    print("\n6. CHECKING ENVIRONMENT VARIABLES:")
    print("-" * 40)
    
    if os.path.exists(".env"):
        with open(".env", 'r') as f:
            env_content = f.read()
        
        required_keys = [
            "ALPHA_VANTAGE_API_KEY",
            "FINNHUB_API_KEY", 
            "POLYGON_API_KEY",
            "NEWS_API_KEY",
            "SECRET_KEY",
            "JWT_SECRET_KEY"
        ]
        
        for key in required_keys:
            if key in env_content:
                # Don't print the actual value for security
                print(f"✓ {key}: Present")
            else:
                print(f"✗ {key}: MISSING")
                errors += 1
    else:
        print("✗ .env file not found")
        errors += 1
    
    # Check Docker files
    print("\n7. CHECKING DOCKER CONFIGURATION:")
    print("-" * 40)
    
    docker_files = [
        ("docker-compose.yml", "Docker Compose"),
        ("infrastructure/docker/backend/Dockerfile", "Backend Dockerfile"),
        ("infrastructure/docker/frontend/Dockerfile", "Frontend Dockerfile"),
        ("infrastructure/docker/postgres/init.sql", "Postgres init script"),
    ]
    
    for filepath, desc in docker_files:
        if not check_file_exists(filepath, desc):
            errors += 1
    
    # Summary
    print("\n" + "=" * 60)
    if errors == 0:
        print("✓ VALIDATION SUCCESSFUL: All checks passed!")
    else:
        print(f"✗ VALIDATION FAILED: {errors} issues found")
    print("=" * 60)
    
    return 0 if errors == 0 else 1

if __name__ == "__main__":
    sys.exit(main())