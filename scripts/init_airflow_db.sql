-- ============================================================================
-- AIRFLOW DATABASE INITIALIZATION SCRIPT
-- ============================================================================
-- This script creates the Airflow database and user with proper permissions
-- Run this script to set up the Airflow metadata database
-- ============================================================================

-- Create Airflow database
CREATE DATABASE airflow_db;

-- Create Airflow user
CREATE USER airflow_user WITH PASSWORD 'secure_airflow_db_password_456';

-- Grant all privileges on Airflow database to airflow_user
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- Connect to airflow_db and grant schema permissions
\c airflow_db;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO airflow_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO airflow_user;

-- Grant default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO airflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO airflow_user;

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Print success message
SELECT 'Airflow database and user created successfully!' as status;