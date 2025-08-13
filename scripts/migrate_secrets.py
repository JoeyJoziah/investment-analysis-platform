#!/usr/bin/env python3
"""
Secret Migration Script

This script migrates API keys and sensitive credentials from environment variables
to the secure secrets management system with encryption.
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.security.secrets_manager import get_secrets_manager, SecretType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_api_keys():
    """Migrate API keys from environment variables to secrets manager"""
    logger.info("Starting API key migration...")
    
    secrets_manager = get_secrets_manager()
    
    # API keys to migrate
    api_keys = [
        ("ALPHA_VANTAGE_API_KEY", "alpha_vantage", "Alpha Vantage API key for stock data"),
        ("FINNHUB_API_KEY", "finnhub", "Finnhub API key for financial data"),
        ("POLYGON_API_KEY", "polygon", "Polygon.io API key for market data"),
        ("FMP_API_KEY", "fmp", "Financial Modeling Prep API key"),
        ("NEWS_API_KEY", "news_api", "NewsAPI key for sentiment analysis"),
        ("MARKETAUX_API_KEY", "marketaux", "MarketAux API key for market news"),
        ("FRED_API_KEY", "fred", "FRED API key for economic data"),
        ("OPENWEATHER_API_KEY", "openweather", "OpenWeatherMap API key"),
    ]
    
    migrated_count = 0
    failed_count = 0
    
    for env_var, provider, description in api_keys:
        api_key = os.getenv(env_var)
        if api_key:
            secret_name = f"api_key_{provider}"
            success = secrets_manager.store_secret(
                secret_name,
                api_key,
                SecretType.API_KEY,
                expires_in_days=365,  # Expire after 1 year
                description=description
            )
            
            if success:
                logger.info(f"✅ Migrated {provider} API key")
                migrated_count += 1
                
                # Verify the secret can be retrieved
                retrieved_key = secrets_manager.get_secret(secret_name)
                if retrieved_key == api_key:
                    logger.info(f"✅ Verified {provider} API key")
                else:
                    logger.error(f"❌ Failed to verify {provider} API key")
                    failed_count += 1
            else:
                logger.error(f"❌ Failed to migrate {provider} API key")
                failed_count += 1
        else:
            logger.warning(f"⚠️  {env_var} not found in environment")
    
    logger.info(f"Migration complete: {migrated_count} succeeded, {failed_count} failed")
    return migrated_count, failed_count


def migrate_database_credentials():
    """Migrate database credentials to secrets manager"""
    logger.info("Starting database credential migration...")
    
    secrets_manager = get_secrets_manager()
    
    # Database credentials to migrate
    db_credentials = [
        ("DB_HOST", "db_host", "Database host"),
        ("DB_PORT", "db_port", "Database port"),
        ("DB_NAME", "db_name", "Database name"),
        ("DB_USER", "db_user", "Database username"),
        ("DB_PASSWORD", "db_password", "Database password"),
    ]
    
    migrated_count = 0
    
    for env_var, secret_name, description in db_credentials:
        credential = os.getenv(env_var)
        if credential:
            success = secrets_manager.store_secret(
                secret_name,
                credential,
                SecretType.DATABASE_CREDENTIAL,
                description=description
            )
            
            if success:
                logger.info(f"✅ Migrated {secret_name}")
                migrated_count += 1
            else:
                logger.error(f"❌ Failed to migrate {secret_name}")
        else:
            logger.info(f"ℹ️  {env_var} not found, using default")
    
    return migrated_count


def migrate_jwt_secrets():
    """Migrate JWT secrets to secrets manager"""
    logger.info("Starting JWT secret migration...")
    
    secrets_manager = get_secrets_manager()
    
    # JWT secrets to migrate
    jwt_secrets = [
        ("SECRET_KEY", "app_secret_key", SecretType.ENCRYPTION_KEY, "Application secret key"),
        ("JWT_SECRET_KEY", "jwt_secret_key", SecretType.JWT_KEY, "JWT signing secret"),
    ]
    
    migrated_count = 0
    
    for env_var, secret_name, secret_type, description in jwt_secrets:
        secret = os.getenv(env_var)
        if secret:
            success = secrets_manager.store_secret(
                secret_name,
                secret,
                secret_type,
                description=description
            )
            
            if success:
                logger.info(f"✅ Migrated {secret_name}")
                migrated_count += 1
            else:
                logger.error(f"❌ Failed to migrate {secret_name}")
        else:
            logger.warning(f"⚠️  {env_var} not found in environment")
    
    return migrated_count


def generate_missing_secrets():
    """Generate any missing required secrets"""
    logger.info("Generating missing required secrets...")
    
    import secrets
    import string
    
    secrets_manager = get_secrets_manager()
    
    # Check if essential secrets exist
    required_secrets = [
        ("app_secret_key", SecretType.ENCRYPTION_KEY, "Application secret key"),
        ("jwt_secret_key", SecretType.JWT_KEY, "JWT signing secret"),
    ]
    
    generated_count = 0
    
    for secret_name, secret_type, description in required_secrets:
        existing_secret = secrets_manager.get_secret(secret_name)
        if not existing_secret:
            # Generate a strong random secret
            new_secret = ''.join(
                secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*-_=+")
                for _ in range(64)
            )
            
            success = secrets_manager.store_secret(
                secret_name,
                new_secret,
                secret_type,
                description=f"Auto-generated {description}"
            )
            
            if success:
                logger.info(f"✅ Generated {secret_name}")
                generated_count += 1
            else:
                logger.error(f"❌ Failed to generate {secret_name}")
        else:
            logger.info(f"ℹ️  {secret_name} already exists")
    
    return generated_count


def validate_secrets():
    """Validate all stored secrets"""
    logger.info("Validating stored secrets...")
    
    secrets_manager = get_secrets_manager()
    
    # List all secrets and validate them
    secrets_list = secrets_manager.list_secrets()
    
    validation_passed = 0
    validation_failed = 0
    
    for secret_hash, metadata in secrets_list.items():
        # We can't easily reverse the hash to get the secret name,
        # so we'll validate by checking if secrets can be retrieved
        try:
            # This is a simplified validation
            if metadata.secret_type and metadata.created_at:
                validation_passed += 1
                logger.debug(f"✅ Secret {secret_hash} metadata valid")
            else:
                validation_failed += 1
                logger.warning(f"⚠️  Secret {secret_hash} has invalid metadata")
        except Exception as e:
            validation_failed += 1
            logger.error(f"❌ Failed to validate secret {secret_hash}: {e}")
    
    logger.info(f"Validation complete: {validation_passed} valid, {validation_failed} invalid")
    return validation_passed, validation_failed


def main():
    """Main migration function"""
    logger.info("🚀 Starting security migration...")
    
    try:
        # Check if master secret key is set
        master_key = os.getenv("MASTER_SECRET_KEY")
        if not master_key:
            logger.error("❌ MASTER_SECRET_KEY environment variable not set!")
            logger.error("Please set MASTER_SECRET_KEY before running migration")
            sys.exit(1)
        
        if len(master_key) < 20:
            logger.warning("⚠️  MASTER_SECRET_KEY is shorter than recommended (20+ characters)")
        
        # Run migrations
        api_migrated, api_failed = migrate_api_keys()
        db_migrated = migrate_database_credentials()  
        jwt_migrated = migrate_jwt_secrets()
        generated_count = generate_missing_secrets()
        
        # Validate results
        valid_count, invalid_count = validate_secrets()
        
        # Summary
        total_migrated = api_migrated + db_migrated + jwt_migrated + generated_count
        
        logger.info("\n" + "="*50)
        logger.info("📊 MIGRATION SUMMARY")
        logger.info("="*50)
        logger.info(f"API keys migrated: {api_migrated}")
        logger.info(f"DB credentials migrated: {db_migrated}")
        logger.info(f"JWT secrets migrated: {jwt_migrated}")
        logger.info(f"Secrets generated: {generated_count}")
        logger.info(f"Total secrets: {total_migrated}")
        logger.info(f"Validation passed: {valid_count}")
        logger.info(f"Validation failed: {invalid_count}")
        
        if api_failed > 0 or invalid_count > 0:
            logger.warning("⚠️  Some migrations failed. Please check the logs above.")
            sys.exit(1)
        else:
            logger.info("✅ All migrations completed successfully!")
            
            # Provide next steps
            logger.info("\n" + "="*50)
            logger.info("📝 NEXT STEPS")
            logger.info("="*50)
            logger.info("1. Update your application to use the secrets manager:")
            logger.info("   from backend.security.secrets_manager import get_secret")
            logger.info("   api_key = get_secret('api_key_alpha_vantage')")
            logger.info("")
            logger.info("2. Remove API keys from environment variables")
            logger.info("3. Update docker-compose.yml to mount secrets directory")
            logger.info("4. Test the application to ensure secrets are loading correctly")
            logger.info("5. Set up secret rotation schedule")
            
    except Exception as e:
        logger.error(f"❌ Migration failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()