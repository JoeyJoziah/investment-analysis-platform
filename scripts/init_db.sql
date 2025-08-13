-- Database initialization script
CREATE DATABASE IF NOT EXISTS investment_db;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE investment_db TO postgres;

-- Create extensions
\c investment_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create initial schema version table
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial version
INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
