# Phase 1: Memory Consolidation - Quick Start Guide

**5-Minute Execution Guide** | V3 Memory Specialist

---

## TL;DR - Execute in 3 Commands

```bash
# 1. Backup
mkdir -p .swarm/backups && cp .swarm/memory.db .swarm/backups/memory.db.pre-consolidation

# 2. Run consolidation (choose one)
python3 scripts/phase1-consolidation.py  # RECOMMENDED

# 3. Verify
npx @claude-flow/cli@latest memory stats
```

---

## What This Does

Migrates 9 patterns from 3 locations into unified V3 database:
- 3 patterns from `.claude/learned-patterns/` → `learned-patterns` namespace
- 5 patterns from `.claude-flow/memory/` → Multiple namespaces
- 1 pattern from `.claude/memory/` → `legacy-patterns` namespace

**Result**: 56 → 65+ entries, 3 → 8+ namespaces, all HNSW-indexed

---

## Quick Execution

### Option 1: Python Script (Recommended)
```bash
python3 scripts/phase1-consolidation.py
```

**Pros**: Best error handling, detailed output, cross-platform

### Option 2: Manual CLI Commands
```bash
# Migrate learned patterns
npx @claude-flow/cli@latest memory store \
  --namespace "learned-patterns" \
  --key "database-driver-compatibility" \
  --value "$(cat .claude/learned-patterns/database-driver-compatibility.json)" \
  --tags "database,sqlalchemy,postgresql,sqlite"

npx @claude-flow/cli@latest memory store \
  --namespace "learned-patterns" \
  --key "middleware-test-compatibility" \
  --value "$(cat .claude/learned-patterns/middleware-test-compatibility.json)" \
  --tags "middleware,testing,fastapi,pytest"

# Migrate Wave 6 memory
npx @claude-flow/cli@latest memory store \
  --namespace "agent-memory" \
  --key "wave6-specialist" \
  --value "$(cat .claude-flow/memory/agent-memory-wave6.json)" \
  --tags "agent,wave6"

npx @claude-flow/cli@latest memory store \
  --namespace "agentdb-patterns" \
  --key "wave6-patterns" \
  --value "$(cat .claude-flow/memory/agentdb-patterns-wave6.json)" \
  --tags "agentdb,hnsw"

npx @claude-flow/cli@latest memory store \
  --namespace "hive-mind" \
  --key "coordination-wave6" \
  --value "$(cat .claude-flow/memory/hive-mind-coordination-wave6.json)" \
  --tags "swarm,coordination"

npx @claude-flow/cli@latest memory store \
  --namespace "session-state" \
  --key "wave6-phase1" \
  --value "$(cat .claude-flow/memory/session-state-wave6-phase1.json)" \
  --tags "session,wave6"

npx @claude-flow/cli@latest memory store \
  --namespace "project-memory" \
  --key "investment-platform" \
  --value "$(cat .claude-flow/memory/project-memory-investment-platform.json)" \
  --tags "project,architecture"

# Migrate legacy markdown
npx @claude-flow/cli@latest memory store \
  --namespace "legacy-patterns" \
  --key "wave6-database-fix" \
  --value "$(cat .claude/memory/wave6-database-fix-pattern.md)" \
  --tags "database,fix"
```

---

## Quick Verification

```bash
# Check entry count (should be 65+)
npx @claude-flow/cli@latest memory stats | grep "Total Entries"

# List namespaces (should be 8+)
npx @claude-flow/cli@latest memory list --limit 5

# Test semantic search
npx @claude-flow/cli@latest memory search --query "database compatibility"
```

**Expected Output**:
```
Total Entries: 65+
Namespaces: learned-patterns, agent-memory, agentdb-patterns, hive-mind, session-state, project-memory, legacy-patterns, patterns, neural-training
Search: Returns database-driver-compatibility pattern
```

---

## Quick Rollback

If something goes wrong:

```bash
cp .swarm/backups/memory.db.pre-consolidation .swarm/memory.db
npx @claude-flow/cli@latest memory stats  # Verify restoration
```

---

## What Gets Migrated

| Source | File | Target Namespace | Key |
|--------|------|------------------|-----|
| Learned | `database-driver-compatibility.json` | learned-patterns | database-driver-compatibility |
| Learned | `middleware-test-compatibility.json` | learned-patterns | middleware-test-compatibility |
| Learned | `wave6-patterns-index.md` | learned-patterns | wave6-patterns-index |
| Wave 6 | `agent-memory-wave6.json` | agent-memory | wave6-specialist |
| Wave 6 | `agentdb-patterns-wave6.json` | agentdb-patterns | wave6-patterns |
| Wave 6 | `hive-mind-coordination-wave6.json` | hive-mind | coordination-wave6 |
| Wave 6 | `session-state-wave6-phase1.json` | session-state | wave6-phase1 |
| Wave 6 | `project-memory-investment-platform.json` | project-memory | investment-platform |
| Legacy | `wave6-database-fix-pattern.md` | legacy-patterns | wave6-database-fix |

---

## Success Indicators

✓ Total entries increases from 56 to 65+
✓ Namespaces increase from 3 to 8+
✓ HNSW index size > 1.5MB
✓ Search returns relevant patterns
✓ No errors in migration log

---

## Next Actions

After successful migration:

1. **Test retrieval**: `npx @claude-flow/cli@latest memory search --query "middleware"`
2. **Archive sources**: Keep original files as backup (DO NOT DELETE)
3. **Phase 2**: Run HNSW performance validation
4. **Document**: Update project docs with new namespace structure

---

## Troubleshooting

**"File not found" errors**:
- Check source files exist: `ls .claude/learned-patterns/ .claude-flow/memory/`

**"Permission denied"**:
- Make script executable: `chmod +x scripts/phase1-consolidation.py`

**Low entry count after migration**:
- Check migration log: `cat .swarm/backups/consolidation-*/migration-report.txt`

**HNSW index not updating**:
- Rebuild: `npx @claude-flow/cli@latest memory init --force`

---

## Documentation

- **Full Guide**: `.claude/PHASE1_MIGRATION_GUIDE.md`
- **Execution Summary**: `.claude/PHASE1_EXECUTION_SUMMARY.md`
- **Migration Report**: `.swarm/backups/consolidation-*/migration-report.txt`

---

**Estimated Time**: 5-10 minutes
**Risk Level**: LOW (backup available, rollback tested)
**Success Rate**: HIGH (scripts validated, automation tested)

**Ready to Execute**: YES ✓
