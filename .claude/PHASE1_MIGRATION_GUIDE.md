# Phase 1: Memory System Consolidation Guide

**Objective**: Unify all memory systems into V3 primary database (.swarm/memory.db)

## Current State Analysis

### Memory Systems Identified
1. **V3 Primary Database**: `.swarm/memory.db` (56 entries, HNSW-indexed)
2. **Learned Patterns**: `.claude/learned-patterns/*.json` (3 files)
3. **Wave 6 Memory**: `.claude-flow/memory/*.json` (5 files)
4. **Legacy Memory**: `.claude/memory/*.md` (1 file)
5. **Backend Duplicate**: `backend/.swarm/memory.db` (to be archived)

### Target Namespaces
- `learned-patterns` - JSON patterns from .claude/learned-patterns
- `wave6-patterns` - Wave 6 session memory
- `agent-memory` - Agent-specific patterns
- `agentdb-patterns` - AgentDB patterns
- `hive-mind` - Hive mind coordination
- `session-state` - Session state snapshots
- `project-memory` - Project-level memory
- `legacy-patterns` - Markdown patterns from .claude/memory

## Manual Migration Steps

### Step 1: Backup Current V3 Database

```bash
mkdir -p .swarm/backups
cp .swarm/memory.db .swarm/backups/memory.db.pre-consolidation
echo "Backup created: $(date)" >> .swarm/backups/migration.log
```

### Step 2: Migrate Learned Patterns

```bash
# Database driver compatibility pattern
npx @claude-flow/cli@latest memory store \
  --namespace "learned-patterns" \
  --key "database-driver-compatibility" \
  --value "$(cat .claude/learned-patterns/database-driver-compatibility.json)" \
  --tags "database,sqlalchemy,postgresql,sqlite,testing,configuration"

# Middleware test compatibility pattern
npx @claude-flow/cli@latest memory store \
  --namespace "learned-patterns" \
  --key "middleware-test-compatibility" \
  --value "$(cat .claude/learned-patterns/middleware-test-compatibility.json)" \
  --tags "middleware,testing,csrf,fastapi,asyncclient,pytest"

# Wave 6 patterns index
npx @claude-flow/cli@latest memory store \
  --namespace "learned-patterns" \
  --key "wave6-patterns-index" \
  --value "$(cat .claude/learned-patterns/wave6-patterns-index.md)" \
  --tags "index,wave6,documentation"
```

### Step 3: Migrate Wave 6 Memory Files

```bash
# Agent memory
npx @claude-flow/cli@latest memory store \
  --namespace "agent-memory" \
  --key "wave6-specialist" \
  --value "$(cat .claude-flow/memory/agent-memory-wave6.json)" \
  --tags "agent,wave6,fastapi,sqlalchemy,pytest"

# AgentDB patterns
npx @claude-flow/cli@latest memory store \
  --namespace "agentdb-patterns" \
  --key "wave6-patterns" \
  --value "$(cat .claude-flow/memory/agentdb-patterns-wave6.json)" \
  --tags "agentdb,patterns,wave6,hnsw,vector-search"

# Hive mind coordination
npx @claude-flow/cli@latest memory store \
  --namespace "hive-mind" \
  --key "coordination-wave6" \
  --value "$(cat .claude-flow/memory/hive-mind-coordination-wave6.json)" \
  --tags "hive-mind,coordination,swarm,wave6"

# Session state
npx @claude-flow/cli@latest memory store \
  --namespace "session-state" \
  --key "wave6-phase1" \
  --value "$(cat .claude-flow/memory/session-state-wave6-phase1.json)" \
  --tags "session,wave6,phase1,state"

# Project memory
npx @claude-flow/cli@latest memory store \
  --namespace "project-memory" \
  --key "investment-platform" \
  --value "$(cat .claude-flow/memory/project-memory-investment-platform.json)" \
  --tags "project,investment-platform,architecture"
```

### Step 4: Migrate Legacy Markdown Pattern

```bash
npx @claude-flow/cli@latest memory store \
  --namespace "legacy-patterns" \
  --key "wave6-database-fix" \
  --value "$(cat .claude/memory/wave6-database-fix-pattern.md)" \
  --tags "database,wave6,postgresql,sqlite,fix"
```

### Step 5: Archive Backend Duplicate

```bash
# Backup backend database
cp backend/.swarm/memory.db .swarm/backups/backend-memory.db.backup

# Compare sizes
ls -lh .swarm/memory.db backend/.swarm/memory.db
```

### Step 6: Verify Migration

```bash
# Check statistics
npx @claude-flow/cli@latest memory stats

# List all namespaces
npx @claude-flow/cli@latest memory list --limit 5

# Test each namespace
npx @claude-flow/cli@latest memory list --namespace learned-patterns
npx @claude-flow/cli@latest memory list --namespace agent-memory
npx @claude-flow/cli@latest memory list --namespace agentdb-patterns
npx @claude-flow/cli@latest memory list --namespace hive-mind
npx @claude-flow/cli@latest memory list --namespace session-state
npx @claude-flow/cli@latest memory list --namespace project-memory
npx @claude-flow/cli@latest memory list --namespace legacy-patterns

# Test semantic search
npx @claude-flow/cli@latest memory search --query "database compatibility"
npx @claude-flow/cli@latest memory search --query "middleware testing"
npx @claude-flow/cli@latest memory search --query "hive mind coordination"
```

### Step 7: Verify HNSW Indexing

```bash
# Check HNSW index file
ls -lh .swarm/hnsw.index

# Check metadata
cat .swarm/hnsw.metadata.json | jq '.indexStats'

# Test vector search performance
time npx @claude-flow/cli@latest memory search --query "authentication patterns"
```

## Expected Results

### Pre-Migration State
- Total entries: 56
- Namespaces: 2-3 (patterns, neural-training)
- Learned patterns: 3 (in separate files)
- Wave 6 memory: 5 (in separate files)
- Legacy patterns: 1 (in markdown)

### Post-Migration State
- Total entries: 65+ (56 + 9 migrated)
- Namespaces: 8 (learned-patterns, agent-memory, agentdb-patterns, hive-mind, session-state, project-memory, legacy-patterns, patterns, neural-training)
- All patterns HNSW-indexed
- All patterns searchable via semantic search
- Backup files preserved

## Validation Checklist

- [ ] Backup created (.swarm/backups/memory.db.pre-consolidation)
- [ ] 3 learned patterns migrated
- [ ] 5 Wave 6 memory files migrated
- [ ] 1 legacy markdown pattern migrated
- [ ] Backend database archived
- [ ] Total entries >= 65
- [ ] 8+ namespaces present
- [ ] HNSW index updated
- [ ] Semantic search functional
- [ ] All patterns retrievable

## Success Criteria

1. **Quantitative**:
   - 100+ total patterns in V3 database
   - 8+ namespaces active
   - HNSW indexing confirmed (hnsw.index > 1MB)
   - Search latency < 100ms

2. **Qualitative**:
   - All legacy patterns accessible via semantic search
   - Cross-session knowledge preserved
   - Pattern confidence scores maintained
   - Timestamps preserved

## Rollback Procedure

If migration fails:

```bash
# Restore V3 database from backup
cp .swarm/backups/memory.db.pre-consolidation .swarm/memory.db

# Verify restoration
npx @claude-flow/cli@latest memory stats

# Rebuild HNSW index if needed
npx @claude-flow/cli@latest memory init --force
```

## Next Steps After Migration

1. Test retrieval performance: `npx @claude-flow/cli@latest memory search --query "test"`
2. Archive source files (DO NOT DELETE - keep as backup)
3. Update documentation with new namespace structure
4. Run Phase 2: HNSW Performance Validation
5. Document learned patterns for future sessions

## Notes

- **DO NOT DELETE** source files after migration - keep as backup
- HNSW index will automatically update when new patterns are added
- Use `--tags` for better semantic search accuracy
- Namespaces are case-sensitive
- Pattern keys should be unique within namespace
