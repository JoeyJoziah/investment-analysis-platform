#!/usr/bin/env python3
"""
Create admin user for the Investment Analysis Platform
"""

import argparse
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from backend.models.database import User
from backend.auth.oauth2 import get_password_hash
from backend.config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_admin_user(username: str, email: str, password: str):
    """Create an admin user in the database"""
    
    # Create engine
    engine = create_engine(settings.DATABASE_URL)
    
    with Session(engine) as session:
        # Check if user already exists
        existing_user = session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            logger.error(f"User with username '{username}' or email '{email}' already exists")
            return False
        
        # Create new admin user
        admin_user = User(
            username=username,
            email=email,
            hashed_password=get_password_hash(password),
            full_name="Admin User",
            is_active=True,
            is_admin=True,
            is_premium=True,
            risk_tolerance="moderate",
            investment_style="balanced"
        )
        
        session.add(admin_user)
        session.commit()
        
        logger.info(f"Admin user '{username}' created successfully!")
        logger.info(f"Email: {email}")
        logger.info("You can now login with these credentials.")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Create admin user")
    parser.add_argument("--username", required=True, help="Admin username")
    parser.add_argument("--email", required=True, help="Admin email")
    parser.add_argument("--password", required=True, help="Admin password")
    
    args = parser.parse_args()
    
    # Validate password length
    if len(args.password) < 8:
        logger.error("Password must be at least 8 characters long")
        sys.exit(1)
    
    # Create admin user
    success = create_admin_user(args.username, args.email, args.password)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()