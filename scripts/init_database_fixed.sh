#!/bin/bash

# Database Initialization Script
# This script sets up the database for the Investment Analysis Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️  Investment Analysis Platform - Database Setup${NC}"
echo "================================================"

# Determine which environment file to use
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    if [ -f ".env.production" ]; then
        ENV_FILE=".env.production"
    else
        echo -e "${RED}❌ No .env file found!${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}Using environment file: ${ENV_FILE}${NC}"

# Load environment variables
set -a
source "$ENV_FILE"
set +a

# Use direct environment variables if DATABASE_URL parsing fails
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-investment_db}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-password}"

# Try to parse DATABASE_URL if direct variables not set properly
if [ -n "$DATABASE_URL" ]; then
    # Use Python to parse the URL reliably (handles special characters)
    if command -v python3 &> /dev/null; then
        eval $(python3 -c "
from urllib.parse import urlparse
url = '$DATABASE_URL'
parsed = urlparse(url)
print(f'DB_HOST={parsed.hostname}')
print(f'DB_PORT={parsed.port or 5432}')
print(f'DB_USER={parsed.username}')
print(f'DB_PASSWORD={parsed.password}')
print(f'DB_NAME={parsed.path.lstrip(\"/\")}')")
    fi
fi

echo -e "${BLUE}Database Configuration:${NC}"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo ""

# Check if running in Docker or need to use docker exec
if [ -f /.dockerenv ]; then
    # We're inside a Docker container
    PSQL_CMD="psql"
    CREATEDB_CMD="createdb"
else
    # Check if postgres container is running
    if docker ps --format '{{.Names}}' | grep -q '^investment_db$'; then
        echo -e "${BLUE}Using Docker container 'investment_db'${NC}"
        PSQL_CMD="docker exec -i investment_db psql"
        CREATEDB_CMD="docker exec -i investment_db createdb"
        # For docker exec, we need to pass user differently
        DB_HOST="localhost"
    else
        # Use local psql if available
        if command -v psql &> /dev/null; then
            PSQL_CMD="psql"
            CREATEDB_CMD="createdb"
        else
            echo -e "${RED}❌ PostgreSQL client not found!${NC}"
            echo "Please either:"
            echo "1. Install PostgreSQL client: sudo apt-get install postgresql-client"
            echo "2. Or start Docker container: docker-compose up -d postgres"
            exit 1
        fi
    fi
fi

# Function to check if PostgreSQL is accessible
check_postgres() {
    echo -e "${BLUE}Checking PostgreSQL connection...${NC}"
    
    # Try to connect to postgres database (default)
    if PGPASSWORD="$DB_PASSWORD" $PSQL_CMD -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c '\q' 2>/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL is accessible${NC}"
        return 0
    else
        echo -e "${RED}❌ Cannot connect to PostgreSQL${NC}"
        echo "Error details:"
        PGPASSWORD="$DB_PASSWORD" $PSQL_CMD -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c '\q'
        return 1
    fi
}

# Function to create database if it doesn't exist
create_database() {
    echo -e "${BLUE}Creating database '${DB_NAME}' if not exists...${NC}"
    
    # Check if database exists
    if PGPASSWORD="$DB_PASSWORD" $PSQL_CMD -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1; then
        echo -e "${YELLOW}Database '${DB_NAME}' already exists${NC}"
    else
        if PGPASSWORD="$DB_PASSWORD" $CREATEDB_CMD -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"; then
            echo -e "${GREEN}✅ Database '${DB_NAME}' created${NC}"
        else
            echo -e "${RED}❌ Failed to create database${NC}"
            return 1
        fi
    fi
}

# Function to run initialization SQL
run_init_sql() {
    echo -e "${BLUE}Running database initialization...${NC}"
    
    # Create SQL initialization script
    SQL_FILE="/tmp/init_investment_db_$$.sql"
    
    cat > "$SQL_FILE" << 'EOF'
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml_models;

-- Create enum types
DO $$ BEGIN
    CREATE TYPE stock_exchange AS ENUM ('NYSE', 'NASDAQ', 'AMEX', 'OTHER');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create main stocks table
CREATE TABLE IF NOT EXISTS public.stocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    exchange stock_exchange,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for stocks
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);
CREATE INDEX IF NOT EXISTS idx_stocks_active ON stocks(is_active);

-- Create daily price data table (partitioned by year)
CREATE TABLE IF NOT EXISTS market_data.daily_prices (
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    adjusted_close DECIMAL(10, 2),
    volume BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date)
) PARTITION BY RANGE (date);

-- Create partitions for 2024-2026
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'daily_prices_2024' AND schemaname = 'market_data') THEN
        CREATE TABLE market_data.daily_prices_2024 
            PARTITION OF market_data.daily_prices 
            FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'daily_prices_2025' AND schemaname = 'market_data') THEN
        CREATE TABLE market_data.daily_prices_2025 
            PARTITION OF market_data.daily_prices 
            FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'daily_prices_2026' AND schemaname = 'market_data') THEN
        CREATE TABLE market_data.daily_prices_2026 
            PARTITION OF market_data.daily_prices 
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
    END IF;
END $$;

-- Create remaining tables (simplified for initial setup)
CREATE TABLE IF NOT EXISTS market_data.technical_indicators (
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    sma_20 DECIMAL(10, 2),
    sma_50 DECIMAL(10, 2),
    sma_200 DECIMAL(10, 2),
    rsi_14 DECIMAL(5, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date)
);

CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_value DECIMAL(10, 2),
    confidence_score DECIMAL(3, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics.recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    recommendation_date DATE NOT NULL,
    action VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(3, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS public.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_provider VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255),
    request_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions
GRANT USAGE ON SCHEMA market_data, analytics, ml_models TO "$DB_USER";
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public, market_data, analytics, ml_models TO "$DB_USER";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public, market_data, analytics, ml_models TO "$DB_USER";

-- Display summary
SELECT 'Database initialization complete!' as status;
EOF

    # Run the SQL script
    if PGPASSWORD="$DB_PASSWORD" $PSQL_CMD -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$SQL_FILE"; then
        echo -e "${GREEN}✅ Database initialization complete${NC}"
    else
        echo -e "${RED}❌ Database initialization failed${NC}"
        # Keep the SQL file for debugging
        echo "SQL file kept at: $SQL_FILE"
        return 1
    fi
    
    # Clean up
    rm -f "$SQL_FILE"
}

# Function to verify tables
verify_tables() {
    echo -e "${BLUE}Verifying database tables...${NC}"
    
    TABLES=$(PGPASSWORD="$DB_PASSWORD" $PSQL_CMD -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -Atc "
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema IN ('public', 'market_data', 'analytics', 'ml_models')
        AND table_type = 'BASE TABLE';
    ")
    
    echo -e "${GREEN}✅ Found ${TABLES} tables in database${NC}"
    
    # List tables
    echo -e "${BLUE}Tables created:${NC}"
    PGPASSWORD="$DB_PASSWORD" $PSQL_CMD -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT table_schema || '.' || table_name as \"Table\" 
        FROM information_schema.tables 
        WHERE table_schema IN ('public', 'market_data', 'analytics', 'ml_models')
        AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name;
    "
}

# Main execution
main() {
    echo -e "${BLUE}Starting database initialization...${NC}"
    
    # Check PostgreSQL connection
    if ! check_postgres; then
        echo -e "${RED}Please ensure PostgreSQL is running and accessible${NC}"
        echo "Troubleshooting tips:"
        echo "1. If using Docker: docker-compose up -d postgres"
        echo "2. Check if the password contains special characters that need escaping"
        echo "3. Verify the host and port are correct"
        exit 1
    fi
    
    # Create database
    if ! create_database; then
        exit 1
    fi
    
    # Run initialization
    if ! run_init_sql; then
        exit 1
    fi
    
    # Verify
    verify_tables
    
    echo -e "${GREEN}🎉 Database setup completed successfully!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Start the backend: docker-compose up backend"
    echo "2. The application will automatically run migrations"
    echo "3. Access the API at: http://localhost:8000"
}

# Run main function
main