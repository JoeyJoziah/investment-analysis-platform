# Documentation Automation Workflows

## Overview

Two production-ready GitHub Actions workflows that automate documentation management, validation, and synchronization.

## Workflows

### 1. Documentation Synchronization (`documentation-sync.yml`)

**Trigger:** Push to `main` or `develop` branches with documentation changes

**Purpose:** Automated documentation maintenance and synchronization

#### Features

- **Duplicate Detection**: Identifies exact and similar content across documentation
- **Link Validation**: Checks both external and internal markdown links
- **Version Tag Verification**: Ensures all docs have version and date metadata
- **Index Generation**: Auto-updates `DOCUMENTATION_INDEX.md`
- **Notion Sync**: Synchronizes documentation to Notion (optional)
- **Results Reporting**: Generates comprehensive workflow summaries

#### Jobs

| Job | Description | Outputs |
|-----|-------------|---------|
| `detect-duplicates` | Scans for duplicate content using SHA-256 hashing | `has_duplicates`, `duplicate_report` |
| `validate-links` | Checks external URLs and internal references | Link errors |
| `check-versions` | Validates version tags and last-updated dates | `version_report` |
| `update-index` | Generates categorized documentation index | Updates `DOCUMENTATION_INDEX.md` |
| `notion-sync` | Syncs docs to Notion database (optional) | Notion page IDs |
| `report-results` | Aggregates results and posts summary | GitHub Actions summary |

#### Setup

1. **No secrets required** for basic functionality
2. **Optional Notion integration** requires:
   ```bash
   # Add GitHub repository secrets
   NOTION_API_KEY=your-notion-integration-token
   NOTION_DATABASE_ID=your-database-id
   ```

3. **Manual trigger** available via workflow dispatch:
   ```bash
   # In GitHub Actions UI
   Run workflow -> Use workflow from: main -> Force sync: true
   ```

---

### 2. Documentation Validation (`documentation-validation.yml`)

**Trigger:** Pull requests modifying documentation files

**Purpose:** Pre-merge validation of documentation quality

#### Features

- **Change Analysis**: Identifies modified documentation files
- **Duplicate Prevention**: Prevents merging duplicate content
- **Broken Link Detection**: Validates all links in changed files
- **Version Tag Enforcement**: Requires version metadata
- **Index Consistency**: Checks if index needs updating
- **Quality Checks**: Linting and best practices validation
- **PR Comments**: Auto-posts validation results to PR

#### Jobs

| Job | Description | Failure Behavior |
|-----|-------------|------------------|
| `analyze-changes` | Lists changed documentation files | N/A |
| `duplicate-check` | Detects exact/similar duplicates | ❌ Fails PR if exact duplicates found |
| `link-validation` | Checks internal and external links | ❌ Fails PR if broken links found |
| `version-validation` | Ensures version tags present | ❌ Fails PR if tags missing |
| `index-check` | Warns if index not updated | ⚠️ Warning only |
| `quality-checks` | Lints documentation style | ⚠️ Suggestions only |
| `pr-comment` | Posts results as PR comment | N/A |
| `final-status` | Aggregates pass/fail status | ❌ Fails PR if critical checks fail |

#### Validation Criteria

**Critical (Must Pass):**
- ✅ No exact duplicate content
- ✅ All internal links resolve
- ✅ All external links return 2xx/3xx status
- ✅ Version tag present (e.g., `Version: 1.2.3`)
- ✅ Last updated date present (e.g., `Last Updated: 2026-01-29`)

**Advisory (Warnings Only):**
- ⚠️ Similar content (>80% similarity)
- ⚠️ Lines longer than 120 characters
- ⚠️ Code blocks without language specifier
- ⚠️ TODO/FIXME markers present
- ⚠️ Missing top-level heading

---

## Configuration Files

### `markdown-link-check.json`

Link validation configuration:

```json
{
  "ignorePatterns": [
    {"pattern": "^http://localhost"},
    {"pattern": "^https://localhost"}
  ],
  "timeout": "20s",
  "retryOn429": true,
  "retryCount": 3,
  "aliveStatusCodes": [200, 206, 301, 302, 307, 308]
}
```

**Customization:**
- Add URL patterns to ignore in `ignorePatterns`
- Adjust timeout for slow endpoints
- Add custom HTTP headers for authenticated APIs

---

## Usage Examples

### Running Validation Locally

```bash
# Install dependencies
npm install -g markdown-link-check@3.12.1

# Check specific file
markdown-link-check docs/README.md --config .github/markdown-link-check.json

# Check all docs
find docs -name "*.md" -exec markdown-link-check {} --config .github/markdown-link-check.json \;
```

### Triggering Manual Sync

```bash
# Via GitHub CLI
gh workflow run documentation-sync.yml -f force_sync=true

# Via GitHub UI
Actions -> Documentation Synchronization -> Run workflow -> Force sync: ✓
```

### Notion Integration Setup

1. **Create Notion Integration:**
   - Go to https://www.notion.so/my-integrations
   - Click "New integration"
   - Copy the "Internal Integration Token"

2. **Create Notion Database:**
   - Create a new database in Notion
   - Add properties: `Name` (title), `Path` (text), `Last Synced` (date)
   - Share database with your integration
   - Copy database ID from URL

3. **Add GitHub Secrets:**
   ```bash
   # Using GitHub CLI
   gh secret set NOTION_API_KEY --body "secret_xxx..."
   gh secret set NOTION_DATABASE_ID --body "abc123..."
   ```

4. **Verify Sync:**
   - Push to main branch
   - Check Actions tab for sync job
   - Verify pages appear in Notion database

---

## Troubleshooting

### Duplicate Detection False Positives

**Problem:** Valid similar files flagged as duplicates

**Solution:** Adjust similarity threshold in `documentation-sync.yml`:
```javascript
// Line ~180
if (similarity > 0.8) { // Change to 0.9 for stricter matching
```

### Link Validation Timeouts

**Problem:** External links timeout

**Solution:** Increase timeout in `markdown-link-check.json`:
```json
{
  "timeout": "30s",  // Increase from 20s
  "retryCount": 5    // More retries
}
```

### Notion Sync Failures

**Problem:** Notion API returns 400/401 errors

**Solutions:**
1. Verify integration token is valid
2. Ensure database is shared with integration
3. Check database properties match schema
4. Review Notion API rate limits

### Version Tag Enforcement

**Problem:** Legacy docs missing version tags

**Solution:** Add exemption pattern in `documentation-validation.yml`:
```javascript
// Skip version check for specific files
const exemptFiles = ['docs/archive/', 'docs/legacy/'];
if (exemptFiles.some(ex => file.startsWith(ex))) continue;
```

---

## Workflow Permissions

### documentation-sync.yml
- `contents: write` - Commits index updates
- `actions: read` - Reads workflow status

### documentation-validation.yml
- `contents: read` - Reads changed files
- `pull-requests: write` - Posts comments
- `issues: write` - Updates PR status

---

## Performance Optimization

### Caching

Both workflows cache npm dependencies:
```yaml
- uses: actions/setup-node@v4
  with:
    cache: 'npm'  # Speeds up subsequent runs
```

### Parallel Execution

Validation jobs run in parallel:
```
analyze-changes
    ├── duplicate-check
    ├── link-validation
    ├── version-validation
    ├── index-check
    └── quality-checks
         └── pr-comment
```

### Conditional Execution

```yaml
if: needs.analyze-changes.outputs.file_count > 0  # Skip if no docs changed
```

---

## Best Practices

### Documentation Standards

1. **Version Tags**: Always include in file headers
   ```markdown
   # Document Title
   **Version:** 1.2.3
   **Last Updated:** 2026-01-29
   ```

2. **Internal Links**: Use relative paths
   ```markdown
   ✅ [Guide](../guides/setup.md)
   ❌ [Guide](/docs/guides/setup.md)
   ```

3. **Code Blocks**: Specify language
   ```markdown
   ✅ ```javascript
   ❌ ```
   ```

4. **Line Length**: Keep under 120 characters for readability

### PR Workflow

1. Create feature branch
2. Add/modify documentation
3. Run local link check
4. Create PR
5. Review validation results
6. Address any failures
7. Merge after approval

### Index Management

- Index auto-updates on merge
- Manual updates optional but recommended
- Categories auto-detected from file paths
- Sorted alphabetically within categories

---

## Integration with Claude Flow

These workflows integrate with Claude Flow V3 continuous learning:

```bash
# After workflow runs, store learnings
npx @claude-flow/cli@latest memory store \
  --namespace patterns \
  --key "docs-validation-$(date +%s)" \
  --value "GitHub Actions documentation workflows validated successfully"

# Track metrics
npx @claude-flow/cli@latest hooks post-task \
  --task-id "workflow-validation" \
  --success true \
  --store-results true
```

---

## Maintenance

### Monthly Tasks
- Review duplicate detection reports
- Update link check timeout if needed
- Verify Notion sync working (if enabled)
- Check for deprecated external links

### Quarterly Tasks
- Audit version tags on all docs
- Update workflow dependencies (actions versions)
- Review and consolidate similar documents
- Performance optimization review

### Annual Tasks
- Major version bump for workflows
- Breaking change migration (if needed)
- Full documentation reorganization
- Archive outdated content

---

## Support

**Issues:** https://github.com/your-org/investment-analysis-platform/issues
**Documentation:** See `DOCUMENTATION_INDEX.md` for all docs
**Claude Flow:** See `CLAUDE.md` for V3 features

---

**Version:** 1.0.0
**Last Updated:** 2026-01-29
**Author:** GitHub Actions + Claude Flow V3
