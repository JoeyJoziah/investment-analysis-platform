#!/usr/bin/env node
/**
 * Phase 1: Memory System Consolidation
 * Unifies all memory systems into V3 primary database (.swarm/memory.db)
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const BACKUP_DIR = `.swarm/backups/consolidation-${Date.now()}`;
const V3_DB = '.swarm/memory.db';
const LEARNED_PATTERNS_DIR = '.claude/learned-patterns';
const WAVE6_MEMORY_DIR = '.claude-flow/memory';
const CLAUDE_MEMORY_DIR = '.claude/memory';

// Statistics
let totalMigrated = 0;
let totalErrors = 0;

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m'
};

// Logging functions
const log = {
  info: (msg) => console.log(`${colors.blue}[INFO]${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}[SUCCESS]${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}[WARNING]${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}[ERROR]${colors.reset} ${msg}`)
};

// Helper to execute CLI commands
function execCLI(command, silent = false) {
  try {
    const result = execSync(command, {
      encoding: 'utf-8',
      stdio: silent ? 'pipe' : 'inherit'
    });
    return { success: true, output: result };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Migrate a single pattern
async function migratePattern(namespace, key, value, tags = '') {
  log.info(`Migrating: [${namespace}] ${key}`);

  // Escape quotes in value
  const escapedValue = JSON.stringify(value).replace(/"/g, '\\"');

  // Build command
  let cmd = `npx @claude-flow/cli@latest memory store --namespace "${namespace}" --key "${key}" --value "${escapedValue}"`;

  if (tags) {
    cmd += ` --tags "${tags}"`;
  }

  const result = execCLI(cmd, true);

  if (result.success) {
    totalMigrated++;
    log.success(`  ✓ Migrated ${key}`);
    return true;
  } else {
    totalErrors++;
    log.error(`  ✗ Failed to migrate ${key}: ${result.error}`);
    return false;
  }
}

// Migrate JSON pattern file
async function migrateJsonPattern(filePath, namespace) {
  log.info(`Processing JSON pattern: ${path.basename(filePath)}`);

  try {
    const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

    // Extract metadata
    const patternName = content.pattern_name || content.pattern_id || 'unknown';
    const category = content.category || content.subcategory || 'general';
    const tags = Array.isArray(content.tags) ? content.tags.join(',') : category;

    // Create key
    const key = `${patternName}-${path.basename(filePath, '.json')}`;

    // Migrate
    await migratePattern(namespace, key, JSON.stringify(content), tags);
  } catch (error) {
    log.error(`Failed to process ${filePath}: ${error.message}`);
    totalErrors++;
  }
}

// Migrate markdown pattern file
async function migrateMarkdownPattern(filePath, namespace) {
  log.info(`Processing markdown pattern: ${path.basename(filePath)}`);

  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const patternName = path.basename(filePath, '.md');

    // Extract tags from content
    const tagsMatch = content.match(/## Tags\n`([^`]+)`/);
    const tags = tagsMatch ? tagsMatch[1] : 'markdown,documentation';

    await migratePattern(namespace, patternName, content, tags);
  } catch (error) {
    log.error(`Failed to process ${filePath}: ${error.message}`);
    totalErrors++;
  }
}

// Get all files in directory with extension
function getFilesWithExtension(dir, ext) {
  if (!fs.existsSync(dir)) return [];

  const files = [];
  const walk = (currentDir) => {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.name.endsWith(ext)) {
        files.push(fullPath);
      }
    }
  };
  walk(dir);
  return files;
}

// Main migration process
async function main() {
  console.log('================================================================');
  console.log('  Phase 1: Memory System Consolidation');
  console.log('  Target: V3 Primary Database (.swarm/memory.db)');
  console.log('================================================================');
  console.log('');

  // Create backup directory
  fs.mkdirSync(BACKUP_DIR, { recursive: true });
  log.success(`Created backup directory: ${BACKUP_DIR}`);

  // Backup V3 database
  if (fs.existsSync(V3_DB)) {
    fs.copyFileSync(V3_DB, path.join(BACKUP_DIR, 'memory.db.backup'));
    log.success(`Backed up V3 database to ${BACKUP_DIR}`);
  }

  console.log('');

  // Step 1: Migrate learned patterns (JSON)
  log.info(`Step 1: Migrating learned patterns from ${LEARNED_PATTERNS_DIR}`);
  if (fs.existsSync(LEARNED_PATTERNS_DIR)) {
    const jsonFiles = getFilesWithExtension(LEARNED_PATTERNS_DIR, '.json');
    for (const file of jsonFiles) {
      await migrateJsonPattern(file, 'learned-patterns');
    }
  } else {
    log.warning('Learned patterns directory not found');
  }
  console.log('');

  // Step 2: Migrate Wave 6 memory (JSON)
  log.info(`Step 2: Migrating Wave 6 memory from ${WAVE6_MEMORY_DIR}`);
  if (fs.existsSync(WAVE6_MEMORY_DIR)) {
    const jsonFiles = getFilesWithExtension(WAVE6_MEMORY_DIR, '.json');
    for (const file of jsonFiles) {
      const filename = path.basename(file, '.json');

      // Determine namespace from filename
      let namespace = 'wave6-patterns';
      if (filename.includes('agent-memory')) namespace = 'agent-memory';
      else if (filename.includes('agentdb-patterns')) namespace = 'agentdb-patterns';
      else if (filename.includes('hive-mind')) namespace = 'hive-mind';
      else if (filename.includes('session-state')) namespace = 'session-state';
      else if (filename.includes('project-memory')) namespace = 'project-memory';

      await migrateJsonPattern(file, namespace);
    }
  } else {
    log.warning('Wave 6 memory directory not found');
  }
  console.log('');

  // Step 3: Migrate markdown patterns
  log.info(`Step 3: Migrating markdown patterns from ${CLAUDE_MEMORY_DIR}`);
  if (fs.existsSync(CLAUDE_MEMORY_DIR)) {
    const mdFiles = getFilesWithExtension(CLAUDE_MEMORY_DIR, '.md');
    for (const file of mdFiles) {
      await migrateMarkdownPattern(file, 'legacy-patterns');
    }
  } else {
    log.warning('.claude/memory directory not found');
  }
  console.log('');

  // Step 4: Verify migration
  log.info('Step 4: Verifying migration');
  console.log('');

  log.info('Fetching V3 memory statistics...');
  execCLI('npx @claude-flow/cli@latest memory stats');

  console.log('');
  log.info('Listing sample entries...');
  execCLI('npx @claude-flow/cli@latest memory list --limit 10');

  console.log('');
  console.log('================================================================');
  console.log('  Migration Summary');
  console.log('================================================================');
  log.success(`Total patterns migrated: ${totalMigrated}`);
  if (totalErrors > 0) {
    log.error(`Total errors: ${totalErrors}`);
  } else {
    log.success('Total errors: 0');
  }
  log.info(`Backup location: ${BACKUP_DIR}`);
  console.log('');

  // Generate migration report
  const reportFile = path.join(BACKUP_DIR, 'migration-report.txt');
  const report = `
Phase 1: Memory System Consolidation Report
Generated: ${new Date().toISOString()}

Migration Statistics:
- Total patterns migrated: ${totalMigrated}
- Total errors: ${totalErrors}
- Backup directory: ${BACKUP_DIR}

Source Locations:
- Learned patterns: ${LEARNED_PATTERNS_DIR}
- Wave 6 memory: ${WAVE6_MEMORY_DIR}
- Legacy memory: ${CLAUDE_MEMORY_DIR}

Target Location:
- V3 database: ${V3_DB}

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
  cp ${BACKUP_DIR}/memory.db.backup ${V3_DB}
`;

  fs.writeFileSync(reportFile, report);
  log.success(`Migration report saved to ${reportFile}`);
  console.log('');

  // Step 5: HNSW verification
  log.info('Step 5: Verifying HNSW indexing');
  const hnswIndexPath = '.swarm/hnsw.index';
  if (fs.existsSync(hnswIndexPath)) {
    const stats = fs.statSync(hnswIndexPath);
    log.success(`HNSW index found: ${stats.size} bytes`);

    const metadataPath = '.swarm/hnsw.metadata.json';
    if (fs.existsSync(metadataPath)) {
      log.info('HNSW metadata:');
      const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
      console.log(JSON.stringify(metadata.indexStats || metadata, null, 2));
    }
  } else {
    log.warning('HNSW index not found - may need to rebuild');
  }

  console.log('');
  log.success('Phase 1: Memory System Consolidation COMPLETE');
  console.log('');
  console.log('Next Steps:');
  console.log('1. Verify patterns: npx @claude-flow/cli@latest memory search --query "test"');
  console.log('2. Check namespaces: npx @claude-flow/cli@latest memory list');
  console.log(`3. Review report: cat ${reportFile}`);
  console.log('');
}

// Run migration
main().catch(error => {
  log.error(`Migration failed: ${error.message}`);
  process.exit(1);
});
