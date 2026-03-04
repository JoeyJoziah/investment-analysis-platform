-- Database initialization script for Investment Analysis Platform
-- This script creates the necessary database structure and initial configuration

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create Airflow database for metadata
CREATE DATABASE airflow;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE investment_db TO postgres;
GRANT ALL PRIVILEGES ON DATABASE airflow TO postgres;

-- =============================================================================
-- Application Role: Least-privilege user for backend connections
-- =============================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'investment_user') THEN
        CREATE ROLE investment_user WITH LOGIN PASSWORD 'investment_pass';
    END IF;
END
$$;

-- Grant usage on public schema
GRANT USAGE ON SCHEMA public TO investment_user;

-- Grant DML privileges on all existing tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO investment_user;

-- Grant usage on all existing sequences
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO investment_user;

-- Ensure future tables/sequences also get grants automatically
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO investment_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO investment_user;

-- =============================================================================
-- Seed: S&P 500 sector reference data (top tickers per sector)
-- =============================================================================
CREATE TABLE IF NOT EXISTS stock_sectors (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100) NOT NULL,
    sub_industry VARCHAR(255),
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO stock_sectors (ticker, company_name, sector, sub_industry) VALUES
    -- Information Technology
    ('AAPL', 'Apple Inc.', 'Information Technology', 'Technology Hardware'),
    ('MSFT', 'Microsoft Corporation', 'Information Technology', 'Systems Software'),
    ('NVDA', 'NVIDIA Corporation', 'Information Technology', 'Semiconductors'),
    ('AVGO', 'Broadcom Inc.', 'Information Technology', 'Semiconductors'),
    ('CRM', 'Salesforce Inc.', 'Information Technology', 'Application Software'),
    -- Health Care
    ('UNH', 'UnitedHealth Group', 'Health Care', 'Managed Health Care'),
    ('JNJ', 'Johnson & Johnson', 'Health Care', 'Pharmaceuticals'),
    ('LLY', 'Eli Lilly and Company', 'Health Care', 'Pharmaceuticals'),
    ('ABBV', 'AbbVie Inc.', 'Health Care', 'Biotechnology'),
    ('PFE', 'Pfizer Inc.', 'Health Care', 'Pharmaceuticals'),
    -- Financials
    ('BRK.B', 'Berkshire Hathaway', 'Financials', 'Multi-Sector Holdings'),
    ('JPM', 'JPMorgan Chase & Co.', 'Financials', 'Diversified Banks'),
    ('V', 'Visa Inc.', 'Financials', 'Transaction Processing'),
    ('MA', 'Mastercard Inc.', 'Financials', 'Transaction Processing'),
    ('BAC', 'Bank of America Corp.', 'Financials', 'Diversified Banks'),
    -- Consumer Discretionary
    ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'Broadline Retail'),
    ('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Automobile Manufacturers'),
    ('HD', 'The Home Depot', 'Consumer Discretionary', 'Home Improvement Retail'),
    ('MCD', 'McDonald''s Corporation', 'Consumer Discretionary', 'Restaurants'),
    ('NKE', 'NIKE Inc.', 'Consumer Discretionary', 'Footwear'),
    -- Communication Services
    ('META', 'Meta Platforms Inc.', 'Communication Services', 'Interactive Media'),
    ('GOOGL', 'Alphabet Inc.', 'Communication Services', 'Interactive Media'),
    ('NFLX', 'Netflix Inc.', 'Communication Services', 'Movies & Entertainment'),
    ('DIS', 'The Walt Disney Company', 'Communication Services', 'Movies & Entertainment'),
    ('CMCSA', 'Comcast Corporation', 'Communication Services', 'Cable & Satellite'),
    -- Industrials
    ('GE', 'GE Aerospace', 'Industrials', 'Aerospace & Defense'),
    ('CAT', 'Caterpillar Inc.', 'Industrials', 'Construction Machinery'),
    ('UNP', 'Union Pacific Corp.', 'Industrials', 'Railroads'),
    ('HON', 'Honeywell International', 'Industrials', 'Industrial Conglomerates'),
    ('BA', 'The Boeing Company', 'Industrials', 'Aerospace & Defense'),
    -- Consumer Staples
    ('PG', 'Procter & Gamble Co.', 'Consumer Staples', 'Household Products'),
    ('KO', 'The Coca-Cola Company', 'Consumer Staples', 'Soft Drinks'),
    ('PEP', 'PepsiCo Inc.', 'Consumer Staples', 'Soft Drinks'),
    ('COST', 'Costco Wholesale', 'Consumer Staples', 'Hypermarkets'),
    ('WMT', 'Walmart Inc.', 'Consumer Staples', 'Hypermarkets'),
    -- Energy
    ('XOM', 'Exxon Mobil Corp.', 'Energy', 'Integrated Oil & Gas'),
    ('CVX', 'Chevron Corporation', 'Energy', 'Integrated Oil & Gas'),
    ('COP', 'ConocoPhillips', 'Energy', 'Oil & Gas Exploration'),
    ('SLB', 'Schlumberger Limited', 'Energy', 'Oil & Gas Equipment'),
    ('EOG', 'EOG Resources Inc.', 'Energy', 'Oil & Gas Exploration'),
    -- Utilities
    ('NEE', 'NextEra Energy Inc.', 'Utilities', 'Electric Utilities'),
    ('SO', 'The Southern Company', 'Utilities', 'Electric Utilities'),
    ('DUK', 'Duke Energy Corp.', 'Utilities', 'Electric Utilities'),
    ('AEP', 'American Electric Power', 'Utilities', 'Electric Utilities'),
    ('D', 'Dominion Energy Inc.', 'Utilities', 'Electric Utilities'),
    -- Real Estate
    ('PLD', 'Prologis Inc.', 'Real Estate', 'Industrial REITs'),
    ('AMT', 'American Tower Corp.', 'Real Estate', 'Telecom Tower REITs'),
    ('EQIX', 'Equinix Inc.', 'Real Estate', 'Data Center REITs'),
    ('SPG', 'Simon Property Group', 'Real Estate', 'Retail REITs'),
    ('O', 'Realty Income Corp.', 'Real Estate', 'Retail REITs'),
    -- Materials
    ('LIN', 'Linde plc', 'Materials', 'Industrial Gases'),
    ('APD', 'Air Products & Chemicals', 'Materials', 'Industrial Gases'),
    ('SHW', 'Sherwin-Williams Co.', 'Materials', 'Specialty Chemicals'),
    ('ECL', 'Ecolab Inc.', 'Materials', 'Specialty Chemicals'),
    ('FCX', 'Freeport-McMoRan Inc.', 'Materials', 'Copper')
ON CONFLICT (ticker) DO NOTHING;

-- Grant access on seed table to application role
GRANT SELECT, INSERT, UPDATE, DELETE ON stock_sectors TO investment_user;