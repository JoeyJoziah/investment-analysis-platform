-- Migration: Add version columns for row locking
-- Description: Adds version columns to Portfolio and Position tables for optimistic locking
-- Date: 2024-01-28
-- Related: docs/security/ROW_LOCKING.md

-- ============================================================================
-- ADD VERSION COLUMNS
-- ============================================================================

-- Add version to portfolios table
ALTER TABLE portfolios
ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

COMMENT ON COLUMN portfolios.version IS 'Optimistic locking version, increments on each update';

-- Add version to positions table
ALTER TABLE positions
ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

COMMENT ON COLUMN positions.version IS 'Optimistic locking version, increments on each update';

-- Note: investment_thesis table already has version column
-- Verify it exists and has correct default
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'investment_thesis' AND column_name = 'version'
    ) THEN
        ALTER TABLE investment_thesis
        ADD COLUMN version INTEGER NOT NULL DEFAULT 1;

        COMMENT ON COLUMN investment_thesis.version IS 'Optimistic locking version, increments on each update';
    END IF;
END $$;

-- ============================================================================
-- CREATE INDEXES FOR VERSION QUERIES
-- ============================================================================

-- Index for version-based queries on portfolios
CREATE INDEX IF NOT EXISTS idx_portfolios_version
ON portfolios(version);

-- Index for version-based queries on positions
CREATE INDEX IF NOT EXISTS idx_positions_version
ON positions(version);

-- Index for version-based queries on investment_thesis
CREATE INDEX IF NOT EXISTS idx_investment_thesis_version
ON investment_thesis(version);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC VERSION INCREMENT (OPTIONAL)
-- ============================================================================

-- Trigger function to auto-increment version on update
CREATE OR REPLACE FUNCTION increment_version()
RETURNS TRIGGER AS $$
BEGIN
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to portfolios
DROP TRIGGER IF EXISTS portfolio_version_increment ON portfolios;
CREATE TRIGGER portfolio_version_increment
    BEFORE UPDATE ON portfolios
    FOR EACH ROW
    EXECUTE FUNCTION increment_version();

-- Apply trigger to positions
DROP TRIGGER IF EXISTS position_version_increment ON positions;
CREATE TRIGGER position_version_increment
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION increment_version();

-- Apply trigger to investment_thesis
DROP TRIGGER IF EXISTS thesis_version_increment ON investment_thesis;
CREATE TRIGGER thesis_version_increment
    BEFORE UPDATE ON investment_thesis
    FOR EACH ROW
    EXECUTE FUNCTION increment_version();

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify version columns exist
SELECT
    table_name,
    column_name,
    data_type,
    column_default,
    is_nullable
FROM information_schema.columns
WHERE table_name IN ('portfolios', 'positions', 'investment_thesis')
  AND column_name = 'version'
ORDER BY table_name;

-- Verify indexes exist
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexname LIKE '%version%'
  AND tablename IN ('portfolios', 'positions', 'investment_thesis')
ORDER BY tablename;

-- Verify triggers exist
SELECT
    trigger_name,
    event_object_table,
    action_statement
FROM information_schema.triggers
WHERE trigger_name LIKE '%version%'
  AND event_object_table IN ('portfolios', 'positions', 'investment_thesis')
ORDER BY event_object_table;

-- ============================================================================
-- ROLLBACK SCRIPT (if needed)
-- ============================================================================

/*
-- Remove triggers
DROP TRIGGER IF EXISTS portfolio_version_increment ON portfolios;
DROP TRIGGER IF EXISTS position_version_increment ON positions;
DROP TRIGGER IF EXISTS thesis_version_increment ON investment_thesis;

-- Remove trigger function
DROP FUNCTION IF EXISTS increment_version();

-- Remove indexes
DROP INDEX IF EXISTS idx_portfolios_version;
DROP INDEX IF EXISTS idx_positions_version;
DROP INDEX IF EXISTS idx_investment_thesis_version;

-- Remove version columns
ALTER TABLE portfolios DROP COLUMN IF EXISTS version;
ALTER TABLE positions DROP COLUMN IF EXISTS version;
-- Note: Keep investment_thesis.version as it was pre-existing
*/
