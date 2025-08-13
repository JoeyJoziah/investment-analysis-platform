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

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
elif [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
else
    echo -e "${RED}❌ No .env file found!${NC}"
    exit 1
fi

# Parse DATABASE_URL
if [[ $DATABASE_URL =~ postgresql://([^:]+):([^@]+)@([^:]+):([^/]+)/(.+) ]]; then
    DB_USER="${BASH_REMATCH[1]}"
    DB_PASSWORD="${BASH_REMATCH[2]}"
    DB_HOST="${BASH_REMATCH[3]}"
    DB_PORT="${BASH_REMATCH[4]}"
    DB_NAME="${BASH_REMATCH[5]}"
else
    echo -e "${RED}❌ Invalid DATABASE_URL format${NC}"
    exit 1
fi

# Function to check if PostgreSQL is accessible
check_postgres() {
    echo -e "${BLUE}Checking PostgreSQL connection...${NC}"
    if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c '\q' 2>/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL is accessible${NC}"
        return 0
    else
        echo -e "${RED}❌ Cannot connect to PostgreSQL${NC}"
        return 1
    fi
}

# Function to create database if it doesn't exist
create_database() {
    echo -e "${BLUE}Creating database '${DB_NAME}' if not exists...${NC}"
    
    # Check if database exists
    if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
        echo -e "${YELLOW}Database '${DB_NAME}' already exists${NC}"
    else
        PGPASSWORD=$DB_PASSWORD createdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME
        echo -e "${GREEN}✅ Database '${DB_NAME}' created${NC}"
    fi
}

# Function to run initialization SQL
run_init_sql() {
    echo -e "${BLUE}Running database initialization...${NC}"
    
    # Create SQL initialization script
    cat > /tmp/init_investment_db.sql << 'EOF'
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

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
CREATE INDEX IF NOT EXISTS idx_stocks_name_trgm ON stocks USING gin(name gin_trgm_ops);

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
CREATE TABLE IF NOT EXISTS market_data.daily_prices_2024 
    PARTITION OF market_data.daily_prices 
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS market_data.daily_prices_2025 
    PARTITION OF market_data.daily_prices 
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS market_data.daily_prices_2026 
    PARTITION OF market_data.daily_prices 
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Create technical indicators table
CREATE TABLE IF NOT EXISTS market_data.technical_indicators (
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    sma_20 DECIMAL(10, 2),
    sma_50 DECIMAL(10, 2),
    sma_200 DECIMAL(10, 2),
    ema_12 DECIMAL(10, 2),
    ema_26 DECIMAL(10, 2),
    rsi_14 DECIMAL(5, 2),
    macd DECIMAL(10, 2),
    macd_signal DECIMAL(10, 2),
    bollinger_upper DECIMAL(10, 2),
    bollinger_lower DECIMAL(10, 2),
    volume_sma_20 BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date)
);

-- Create ML predictions table
CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    prediction_date DATE NOT NULL,
    prediction_type VARCHAR(50), -- 'price', 'direction', 'volatility'
    prediction_horizon INTEGER, -- days ahead
    predicted_value DECIMAL(10, 2),
    confidence_score DECIMAL(3, 2),
    features_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_stock_date ON ml_models.predictions(stock_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON ml_models.predictions(model_name, model_version);

-- Create recommendations table
CREATE TABLE IF NOT EXISTS analytics.recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    recommendation_date DATE NOT NULL,
    action VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence_score DECIMAL(3, 2),
    target_price DECIMAL(10, 2),
    stop_loss DECIMAL(10, 2),
    reasoning JSONB,
    model_used VARCHAR(100),
    expires_at DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_recommendations_stock_date ON analytics.recommendations(stock_id, recommendation_date);
CREATE INDEX IF NOT EXISTS idx_recommendations_action ON analytics.recommendations(action, recommendation_date);

-- Create API usage tracking table
CREATE TABLE IF NOT EXISTS public.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_provider VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255),
    request_count INTEGER DEFAULT 1,
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    cost_estimate DECIMAL(10, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_api_usage_provider_date ON api_usage(api_provider, created_at);

-- Create cost tracking table
CREATE TABLE IF NOT EXISTS public.cost_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    cost_type VARCHAR(50), -- 'api', 'compute', 'storage'
    amount DECIMAL(10, 4),
    currency VARCHAR(3) DEFAULT 'USD',
    billing_period DATE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create users table (for future use)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,
    hashed_password VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create audit log table
CREATE TABLE IF NOT EXISTS public.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action, created_at);

-- Create update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_stocks_updated_at') THEN
        CREATE TRIGGER update_stocks_updated_at 
        BEFORE UPDATE ON stocks 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_users_updated_at') THEN
        CREATE TRIGGER update_users_updated_at 
        BEFORE UPDATE ON users 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- Grant permissions
GRANT USAGE ON SCHEMA market_data TO postgres;
GRANT USAGE ON SCHEMA analytics TO postgres;
GRANT USAGE ON SCHEMA ml_models TO postgres;

-- Create read-only user for analytics (optional)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_user WHERE usename = 'analytics_readonly') THEN
        CREATE USER analytics_readonly WITH PASSWORD 'analytics_password_here';
        GRANT CONNECT ON DATABASE investment_db TO analytics_readonly;
        GRANT USAGE ON SCHEMA public, market_data, analytics TO analytics_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA public, market_data, analytics TO analytics_readonly;
    END IF;
END $$;

-- Display summary
SELECT 'Database initialization complete!' as status;
EOF

    # Run the SQL script
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /tmp/init_investment_db.sql
    
    # Clean up
    rm -f /tmp/init_investment_db.sql
    
    echo -e "${GREEN}✅ Database initialization complete${NC}"
}

# Function to verify tables
verify_tables() {
    echo -e "${BLUE}Verifying database tables...${NC}"
    
    TABLES=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema IN ('public', 'market_data', 'analytics', 'ml_models')
        AND table_type = 'BASE TABLE';
    ")
    
    echo -e "${GREEN}✅ Found ${TABLES} tables in database${NC}"
    
    # List tables
    echo -e "${BLUE}Tables created:${NC}"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
        SELECT table_schema, table_name 
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
        echo "If using Docker, run: docker-compose up -d postgres"
        exit 1
    fi
    
    # Create database
    create_database
    
    # Run initialization
    run_init_sql
    
    # Verify
    verify_tables
    
    echo -e "${GREEN}🎉 Database setup completed successfully!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Run migrations: alembic upgrade head"
    echo "2. Load initial data: python backend/utils/load_initial_data.py"
    echo "3. Start the application: docker-compose up"
}

# Run main function
main