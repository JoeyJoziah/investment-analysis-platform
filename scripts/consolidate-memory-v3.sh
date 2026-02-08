#!/usr/bin/env bash
# Phase 1: Memory System Consolidation Script
# Unifies all memory systems into V3 primary database (.swarm/memory.db)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
BACKUP_DIR=".swarm/backups/consolidation-$(date +%s)"
V3_DB=".swarm/memory.db"
LEGACY_DB=".claude/memory.db"
BACKEND_DB="backend/.swarm/memory.db"
LEARNED_PATTERNS_DIR=".claude/learned-patterns"
WAVE6_MEMORY_DIR=".claude-flow/memory"

# Statistics counters
TOTAL_MIGRATED=0
TOTAL_ERRORS=0

# Create backup directory
mkdir -p "$BACKUP_DIR"
log_info "Created backup directory: $BACKUP_DIR"

# Backup V3 database before migration
if [ -f "$V3_DB" ]; then
    cp "$V3_DB" "$BACKUP_DIR/memory.db.backup"
    log_success "Backed up V3 database to $BACKUP_DIR"
fi

# Function to migrate a pattern to V3
migrate_pattern() {
    local namespace="$1"
    local key="$2"
    local value="$3"
    local tags="${4:-}"

    log_info "Migrating: [$namespace] $key"

    # Build command
    local cmd="npx @claude-flow/cli@latest memory store --namespace \"$namespace\" --key \"$key\" --value \"$value\""

    if [ -n "$tags" ]; then
        cmd="$cmd --tags \"$tags\""
    fi

    # Execute migration
    if eval "$cmd" &>/dev/null; then
        ((TOTAL_MIGRATED++))
        log_success "  ✓ Migrated $key"
        return 0
    else
        ((TOTAL_ERRORS++))
        log_error "  ✗ Failed to migrate $key"
        return 1
    fi
}

# Function to migrate JSON patterns
migrate_json_pattern() {
    local file="$1"
    local namespace="$2"

    log_info "Processing JSON pattern: $(basename "$file")"

    # Extract pattern data
    local pattern_name=$(jq -r '.pattern_name // .pattern_id // "unknown"' "$file" 2>/dev/null || echo "unknown")
    local category=$(jq -r '.category // .subcategory // "general"' "$file" 2>/dev/null || echo "general")
    local success_score=$(jq -r '.success_score // .confidence // 0.85' "$file" 2>/dev/null || echo "0.85")

    # Create key
    local key="${pattern_name}-$(basename "$file" .json)"

    # Create tags
    local tags=$(jq -r '.tags // [] | join(",")' "$file" 2>/dev/null || echo "$category")

    # Create value (compact JSON)
    local value=$(jq -c '.' "$file" 2>/dev/null || cat "$file")

    migrate_pattern "$namespace" "$key" "$value" "$tags"
}

# Function to migrate markdown pattern
migrate_markdown_pattern() {
    local file="$1"
    local namespace="$2"

    log_info "Processing markdown pattern: $(basename "$file")"

    # Extract pattern name from filename
    local pattern_name=$(basename "$file" .md)

    # Read content
    local content=$(cat "$file")

    # Extract tags from content
    local tags=$(echo "$content" | grep -oP '(?<=^## Tags\n`)[^`]+' || echo "markdown,documentation")

    migrate_pattern "$namespace" "$pattern_name" "$content" "$tags"
}

echo "=================================================================="
echo "  Phase 1: Memory System Consolidation"
echo "  Target: V3 Primary Database (.swarm/memory.db)"
echo "=================================================================="
echo ""

# Step 1: Migrate learned patterns (JSON)
log_info "Step 1: Migrating learned patterns from $LEARNED_PATTERNS_DIR"
if [ -d "$LEARNED_PATTERNS_DIR" ]; then
    find "$LEARNED_PATTERNS_DIR" -name "*.json" -type f | while read -r pattern_file; do
        migrate_json_pattern "$pattern_file" "learned-patterns"
    done
else
    log_warning "Learned patterns directory not found"
fi
echo ""

# Step 2: Migrate Wave 6 memory (JSON)
log_info "Step 2: Migrating Wave 6 memory from $WAVE6_MEMORY_DIR"
if [ -d "$WAVE6_MEMORY_DIR" ]; then
    # Migrate JSON patterns
    find "$WAVE6_MEMORY_DIR" -name "*.json" -type f | while read -r pattern_file; do
        # Determine namespace from filename
        local filename=$(basename "$pattern_file" .json)
        case "$filename" in
            agent-memory*)
                migrate_json_pattern "$pattern_file" "agent-memory"
                ;;
            agentdb-patterns*)
                migrate_json_pattern "$pattern_file" "agentdb-patterns"
                ;;
            hive-mind*)
                migrate_json_pattern "$pattern_file" "hive-mind"
                ;;
            session-state*)
                migrate_json_pattern "$pattern_file" "session-state"
                ;;
            project-memory*)
                migrate_json_pattern "$pattern_file" "project-memory"
                ;;
            *)
                migrate_json_pattern "$pattern_file" "wave6-patterns"
                ;;
        esac
    done
else
    log_warning "Wave 6 memory directory not found"
fi
echo ""

# Step 3: Migrate markdown patterns from .claude/memory
log_info "Step 3: Migrating markdown patterns from .claude/memory"
if [ -d ".claude/memory" ]; then
    find ".claude/memory" -name "*.md" -type f | while read -r pattern_file; do
        migrate_markdown_pattern "$pattern_file" "legacy-patterns"
    done
else
    log_warning ".claude/memory directory not found"
fi
echo ""

# Step 4: Archive duplicate database
log_info "Step 4: Archiving duplicate backend database"
if [ -f "$BACKEND_DB" ]; then
    cp "$BACKEND_DB" "$BACKUP_DIR/backend-memory.db.backup"
    log_success "Backed up backend database to $BACKUP_DIR"

    # Check if it's truly duplicate (same size/modification time)
    if [ -f "$V3_DB" ]; then
        V3_SIZE=$(stat -f%z "$V3_DB" 2>/dev/null || stat -c%s "$V3_DB")
        BACKEND_SIZE=$(stat -f%z "$BACKEND_DB" 2>/dev/null || stat -c%s "$BACKEND_DB")

        log_info "V3 DB size: $V3_SIZE bytes"
        log_info "Backend DB size: $BACKEND_SIZE bytes"
    fi
else
    log_warning "Backend database not found"
fi
echo ""

# Step 5: Verify migration
log_info "Step 5: Verifying migration"
echo ""

# Get V3 statistics
log_info "Fetching V3 memory statistics..."
npx @claude-flow/cli@latest memory stats

echo ""
log_info "Listing namespaces..."
npx @claude-flow/cli@latest memory list --limit 5

echo ""
echo "=================================================================="
echo "  Migration Summary"
echo "=================================================================="
log_success "Total patterns migrated: $TOTAL_MIGRATED"
[ $TOTAL_ERRORS -gt 0 ] && log_error "Total errors: $TOTAL_ERRORS" || log_success "Total errors: 0"
log_info "Backup location: $BACKUP_DIR"
echo ""

# Step 6: Generate migration report
REPORT_FILE="$BACKUP_DIR/migration-report.txt"
cat > "$REPORT_FILE" <<EOF
Phase 1: Memory System Consolidation Report
Generated: $(date)

Migration Statistics:
- Total patterns migrated: $TOTAL_MIGRATED
- Total errors: $TOTAL_ERRORS
- Backup directory: $BACKUP_DIR

Source Locations:
- Learned patterns: $LEARNED_PATTERNS_DIR
- Wave 6 memory: $WAVE6_MEMORY_DIR
- Legacy memory: .claude/memory
- Backend database: $BACKEND_DB

Target Location:
- V3 database: $V3_DB

Namespaces Created:
- learned-patterns (JSON patterns from .claude/learned-patterns)
- wave6-patterns (Wave 6 session memory)
- agent-memory (Agent-specific patterns)
- agentdb-patterns (AgentDB patterns)
- hive-mind (Hive mind coordination)
- session-state (Session state snapshots)
- project-memory (Project-level memory)
- legacy-patterns (Markdown patterns from .claude/memory)

Verification Commands:
- npx @claude-flow/cli@latest memory stats
- npx @claude-flow/cli@latest memory list --namespace learned-patterns
- npx @claude-flow/cli@latest memory search --query "database compatibility"

Rollback Instructions:
If migration fails, restore from backup:
  cp $BACKUP_DIR/memory.db.backup $V3_DB
EOF

log_success "Migration report saved to $REPORT_FILE"
echo ""

# Step 7: HNSW index verification
log_info "Step 7: Verifying HNSW indexing"
if [ -f ".swarm/hnsw.index" ]; then
    HNSW_SIZE=$(stat -f%z ".swarm/hnsw.index" 2>/dev/null || stat -c%s ".swarm/hnsw.index")
    log_success "HNSW index found: $HNSW_SIZE bytes"

    # Check metadata
    if [ -f ".swarm/hnsw.metadata.json" ]; then
        log_info "HNSW metadata:"
        cat ".swarm/hnsw.metadata.json" | jq '.indexStats' || cat ".swarm/hnsw.metadata.json"
    fi
else
    log_warning "HNSW index not found - may need to rebuild"
fi

echo ""
log_success "Phase 1: Memory System Consolidation COMPLETE"
echo ""
echo "Next Steps:"
echo "1. Verify patterns: npx @claude-flow/cli@latest memory search --query 'test'"
echo "2. Check namespaces: npx @claude-flow/cli@latest memory list"
echo "3. Review report: cat $REPORT_FILE"
echo ""
