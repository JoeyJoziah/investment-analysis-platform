#!/usr/bin/env python3
"""
Phase 1: Memory System Consolidation
Unifies all memory systems into V3 primary database (.swarm/memory.db)
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
BACKUP_DIR = Path(f".swarm/backups/consolidation-{int(datetime.now().timestamp())}")
V3_DB = Path(".swarm/memory.db")
LEARNED_PATTERNS_DIR = Path(".claude/learned-patterns")
WAVE6_MEMORY_DIR = Path(".claude-flow/memory")
CLAUDE_MEMORY_DIR = Path(".claude/memory")

# Statistics
total_migrated = 0
total_errors = 0
migration_log = []

# Colors
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def log_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")
    migration_log.append(f"INFO: {msg}")

def log_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")
    migration_log.append(f"SUCCESS: {msg}")

def log_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")
    migration_log.append(f"WARNING: {msg}")

def log_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")
    migration_log.append(f"ERROR: {msg}")

def run_cli_command(cmd, silent=False):
    """Execute CLI command and return result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        if not silent:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if not silent:
            print(e.stderr)
        return False, e.stderr

def migrate_pattern(namespace, key, value, tags=""):
    """Migrate a single pattern to V3"""
    global total_migrated, total_errors

    log_info(f"Migrating: [{namespace}] {key}")

    # Write value to temp file to avoid shell escaping issues
    temp_file = BACKUP_DIR / f"temp_{key}.json"
    temp_file.write_text(value if isinstance(value, str) else json.dumps(value))

    # Build command
    cmd = f'npx @claude-flow/cli@latest memory store --namespace "{namespace}" --key "{key}" --value "$(cat {temp_file})"'

    if tags:
        cmd += f' --tags "{tags}"'

    success, output = run_cli_command(cmd, silent=True)

    # Clean up temp file
    if temp_file.exists():
        temp_file.unlink()

    if success:
        total_migrated += 1
        log_success(f"  ✓ Migrated {key}")
        return True
    else:
        total_errors += 1
        log_error(f"  ✗ Failed to migrate {key}")
        return False

def migrate_json_file(file_path, namespace):
    """Migrate a JSON pattern file"""
    log_info(f"Processing JSON: {file_path.name}")

    try:
        with open(file_path, 'r') as f:
            content = json.load(f)

        # Extract metadata
        pattern_name = content.get('pattern_name') or content.get('pattern_id') or 'unknown'
        category = content.get('category') or content.get('subcategory') or 'general'
        tags = ','.join(content.get('tags', [category]))

        # Create key
        key = f"{pattern_name}-{file_path.stem}"

        # Migrate
        return migrate_pattern(namespace, key, json.dumps(content), tags)
    except Exception as e:
        log_error(f"Failed to process {file_path}: {e}")
        return False

def migrate_markdown_file(file_path, namespace):
    """Migrate a markdown pattern file"""
    log_info(f"Processing markdown: {file_path.name}")

    try:
        content = file_path.read_text()
        pattern_name = file_path.stem

        # Extract tags from content
        import re
        tags_match = re.search(r'## Tags\n`([^`]+)`', content)
        tags = tags_match.group(1) if tags_match else 'markdown,documentation'

        return migrate_pattern(namespace, pattern_name, content, tags)
    except Exception as e:
        log_error(f"Failed to process {file_path}: {e}")
        return False

def main():
    print("=" * 70)
    print("  Phase 1: Memory System Consolidation")
    print("  Target: V3 Primary Database (.swarm/memory.db)")
    print("=" * 70)
    print()

    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    log_success(f"Created backup directory: {BACKUP_DIR}")

    # Backup V3 database
    if V3_DB.exists():
        shutil.copy(V3_DB, BACKUP_DIR / "memory.db.backup")
        log_success(f"Backed up V3 database to {BACKUP_DIR}")

    print()

    # Step 1: Migrate learned patterns
    log_info(f"Step 1: Migrating learned patterns from {LEARNED_PATTERNS_DIR}")
    if LEARNED_PATTERNS_DIR.exists():
        for json_file in LEARNED_PATTERNS_DIR.glob("*.json"):
            migrate_json_file(json_file, "learned-patterns")
    else:
        log_warning("Learned patterns directory not found")
    print()

    # Step 2: Migrate Wave 6 memory files
    log_info(f"Step 2: Migrating Wave 6 memory from {WAVE6_MEMORY_DIR}")
    if WAVE6_MEMORY_DIR.exists():
        namespace_map = {
            'agent-memory': 'agent-memory',
            'agentdb-patterns': 'agentdb-patterns',
            'hive-mind': 'hive-mind',
            'session-state': 'session-state',
            'project-memory': 'project-memory'
        }

        for json_file in WAVE6_MEMORY_DIR.glob("*.json"):
            # Determine namespace from filename
            namespace = 'wave6-patterns'
            for prefix, ns in namespace_map.items():
                if prefix in json_file.stem:
                    namespace = ns
                    break

            migrate_json_file(json_file, namespace)
    else:
        log_warning("Wave 6 memory directory not found")
    print()

    # Step 3: Migrate markdown patterns
    log_info(f"Step 3: Migrating markdown patterns from {CLAUDE_MEMORY_DIR}")
    if CLAUDE_MEMORY_DIR.exists():
        for md_file in CLAUDE_MEMORY_DIR.glob("*.md"):
            if md_file.stem not in ['README', 'readme']:
                migrate_markdown_file(md_file, "legacy-patterns")
    else:
        log_warning(".claude/memory directory not found")
    print()

    # Step 4: Verification
    log_info("Step 4: Verifying migration")
    print()

    log_info("Fetching V3 memory statistics...")
    run_cli_command('npx @claude-flow/cli@latest memory stats')

    print()
    log_info("Listing sample entries...")
    run_cli_command('npx @claude-flow/cli@latest memory list --limit 10')

    print()
    print("=" * 70)
    print("  Migration Summary")
    print("=" * 70)
    log_success(f"Total patterns migrated: {total_migrated}")
    if total_errors > 0:
        log_error(f"Total errors: {total_errors}")
    else:
        log_success("Total errors: 0")
    log_info(f"Backup location: {BACKUP_DIR}")
    print()

    # Generate migration report
    report_file = BACKUP_DIR / "migration-report.txt"
    report = f"""Phase 1: Memory System Consolidation Report
Generated: {datetime.now().isoformat()}

Migration Statistics:
- Total patterns migrated: {total_migrated}
- Total errors: {total_errors}
- Backup directory: {BACKUP_DIR}

Source Locations:
- Learned patterns: {LEARNED_PATTERNS_DIR}
- Wave 6 memory: {WAVE6_MEMORY_DIR}
- Legacy memory: {CLAUDE_MEMORY_DIR}

Target Location:
- V3 database: {V3_DB}

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
  cp {BACKUP_DIR}/memory.db.backup {V3_DB}

Migration Log:
{chr(10).join(migration_log)}
"""

    report_file.write_text(report)
    log_success(f"Migration report saved to {report_file}")
    print()

    # Step 5: HNSW verification
    log_info("Step 5: Verifying HNSW indexing")
    hnsw_index = Path(".swarm/hnsw.index")
    if hnsw_index.exists():
        size = hnsw_index.stat().st_size
        log_success(f"HNSW index found: {size:,} bytes")

        hnsw_metadata = Path(".swarm/hnsw.metadata.json")
        if hnsw_metadata.exists():
            log_info("HNSW metadata:")
            metadata = json.loads(hnsw_metadata.read_text())
            print(json.dumps(metadata.get('indexStats', metadata), indent=2))
    else:
        log_warning("HNSW index not found - may need to rebuild")

    print()
    log_success("Phase 1: Memory System Consolidation COMPLETE")
    print()
    print("Next Steps:")
    print('1. Verify patterns: npx @claude-flow/cli@latest memory search --query "test"')
    print('2. Check namespaces: npx @claude-flow/cli@latest memory list')
    print(f'3. Review report: cat {report_file}')
    print()

if __name__ == "__main__":
    main()
