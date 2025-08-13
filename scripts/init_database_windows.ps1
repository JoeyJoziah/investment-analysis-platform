# Database Initialization Script for Windows
# Run this from PowerShell in the project directory

Write-Host "🗄️  Investment Analysis Platform - Database Setup (Windows)" -ForegroundColor Blue
Write-Host "================================================" -ForegroundColor Blue

# Check if Docker Desktop is running
$dockerRunning = docker version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker Desktop is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Docker Desktop is running" -ForegroundColor Green

# Start PostgreSQL and Redis containers
Write-Host "`n📦 Starting database containers..." -ForegroundColor Blue
docker compose up -d postgres redis

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start containers" -ForegroundColor Red
    exit 1
}

# Wait for PostgreSQL to be ready
Write-Host "`n⏳ Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    $attempt++
    Write-Host "Checking PostgreSQL (attempt $attempt/$maxAttempts)..." -NoNewline
    
    $result = docker exec investment_db pg_isready -U postgres 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅ Ready!" -ForegroundColor Green
        break
    }
    
    Write-Host " Not ready yet..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

if ($attempt -eq $maxAttempts) {
    Write-Host "❌ PostgreSQL failed to start in time" -ForegroundColor Red
    exit 1
}

# Load environment variables
Write-Host "`n📋 Loading environment configuration..." -ForegroundColor Blue
$envFile = ".env"
if (-not (Test-Path $envFile)) {
    $envFile = ".env.production"
}

# Read environment variables
$envVars = @{}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        $envVars[$matches[1].Trim()] = $matches[2].Trim()
    }
}

$dbPassword = $envVars['DB_PASSWORD']
if (-not $dbPassword) {
    # Try to extract from DATABASE_URL
    if ($envVars['DATABASE_URL'] -match 'postgresql://[^:]+:([^@]+)@') {
        $dbPassword = $matches[1]
    }
}

if (-not $dbPassword) {
    $dbPassword = "9v1g^OV9XUwzUP6cEgCYgNOE"  # Use the generated password
}

Write-Host "Database password configured" -ForegroundColor Green

# Create database if it doesn't exist
Write-Host "`n🔨 Creating database..." -ForegroundColor Blue
$env:PGPASSWORD = $dbPassword

# Check if database exists
$dbExists = docker exec investment_db psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='investment_db'" 2>$null
if ($dbExists -eq "1") {
    Write-Host "Database 'investment_db' already exists" -ForegroundColor Yellow
} else {
    docker exec investment_db createdb -U postgres investment_db
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database 'investment_db' created" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create database" -ForegroundColor Red
        exit 1
    }
}

# Initialize database schema
Write-Host "`n🏗️  Initializing database schema..." -ForegroundColor Blue

# Create initialization SQL
$initSQL = @'
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

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);

-- Create price data table
CREATE TABLE IF NOT EXISTS market_data.daily_prices (
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date)
) PARTITION BY RANGE (date);

-- Create partitions
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'daily_prices_2024' AND schemaname = 'market_data') THEN
        CREATE TABLE market_data.daily_prices_2024 PARTITION OF market_data.daily_prices 
            FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'daily_prices_2025' AND schemaname = 'market_data') THEN
        CREATE TABLE market_data.daily_prices_2025 PARTITION OF market_data.daily_prices 
            FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
    END IF;
END $$;

-- Create other essential tables
CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id),
    model_name VARCHAR(100) NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_value DECIMAL(10, 2),
    confidence_score DECIMAL(3, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics.recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id),
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
GRANT ALL ON SCHEMA market_data, analytics, ml_models TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA public, market_data, analytics, ml_models TO postgres;

SELECT 'Database initialization complete!' as status;
'@

# Save SQL to temporary file
$sqlFile = "temp_init.sql"
$initSQL | Out-File -FilePath $sqlFile -Encoding UTF8

# Execute SQL
Get-Content $sqlFile | docker exec -i investment_db psql -U postgres -d investment_db

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Database schema initialized successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to initialize database schema" -ForegroundColor Red
    Remove-Item $sqlFile
    exit 1
}

# Clean up
Remove-Item $sqlFile

# Verify tables
Write-Host "`n📊 Verifying database tables..." -ForegroundColor Blue
$tableCount = docker exec investment_db psql -U postgres -d investment_db -tAc @"
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema IN ('public', 'market_data', 'analytics', 'ml_models')
AND table_type = 'BASE TABLE'
"@

Write-Host "✅ Created $tableCount tables" -ForegroundColor Green

# List tables
Write-Host "`nTables created:" -ForegroundColor Blue
docker exec investment_db psql -U postgres -d investment_db -c @"
SELECT table_schema || '.' || table_name as "Table" 
FROM information_schema.tables 
WHERE table_schema IN ('public', 'market_data', 'analytics', 'ml_models')
AND table_type = 'BASE TABLE'
ORDER BY table_schema, table_name
"@

Write-Host "`n🎉 Database setup completed successfully!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Blue
Write-Host "1. Start the backend: docker compose up backend" -ForegroundColor Cyan
Write-Host "2. Access the API at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "3. View logs: docker compose logs -f" -ForegroundColor Cyan

Write-Host "`n✅ Your database is ready for use!" -ForegroundColor Green