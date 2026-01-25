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
