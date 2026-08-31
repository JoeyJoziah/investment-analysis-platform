# Documentation Verification Scripts Guide

**Last Updated: 2026-01-29**

This guide covers the five verification scripts that ensure documentation integrity and repository cleanliness.

## Overview

The verification scripts are comprehensive tools for maintaining documentation quality:

| Script | Purpose | Key Checks |
|--------|---------|-----------|
| `verify-duplicate-files.sh` | Detect duplicate " 2" files | File duplication, wasted space |
| `verify-links.sh` | Validate markdown links | Internal refs, anchors, file existence |
| `verify-version-tags.sh` | Check version consistency | Semantic versioning, last updated dates |
| `verify-doc-count.sh` | Track documentation metrics | File count, size distribution, categories |
| `verify-root-cleanup.sh` | Ensure root organization | Clutter detection, organization |

## Installation

All scripts are located in `/scripts/` and are executable:

```bash
cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/scripts
chmod +x verify-*.sh
```

## Scripts in Detail

### 1. verify-duplicate-files.sh

Detects and reports duplicate files with " 2" suffix patterns.

#### Usage

```bash
# Scan for duplicates
./verify-duplicate-files.sh

# Generate detailed report
./verify-duplicate-files.sh --report

# Automatically remove duplicates
./verify-duplicate-files.sh --fix

# Both report and fix
./verify-duplicate-files.sh --report --fix

# Show help
./verify-duplicate-files.sh --help
```

#### Features

- Scans up to 3 levels deep in project root
- Excludes node_modules, dist, and build directories
- Calculates space wasted by duplicates
- Generates detailed reports with file sizes
- Safe removal mode (keeps original file)

#### Output Files

- **Report**: `.claude/verify/duplicate-files-report.txt`
- **Log**: `.claude/verify/duplicate-files.log`

#### Example Output

```
================================================================================
DUPLICATE FILES VERIFICATION
================================================================================

>>> Results: 7 duplicate file(s) found

  - CHANGES_MADE 2.md
  - TYPE_CONSISTENCY_IMPLEMENTATION 2.md
  - PHASE3_TYPE_FIX_GUIDE 2.md
  - LINE_BY_LINE_MAPPING 2.md
  - VALIDATION_DELIVERABLES 2.md
  - QUICK_START 2.md
  - TYPE_CONSISTENCY_ANALYSIS 2.md
```

### 2. verify-links.sh

Validates all internal links in markdown documentation.

#### Usage

```bash
# Validate all links
./verify-links.sh

# Generate report
./verify-links.sh --report

# Check external URLs (slower)
./verify-links.sh --external

# Verbose output
./verify-links.sh --verbose

# Full diagnostic
./verify-links.sh --report --external --verbose
```

#### Features

- Extracts markdown links: `[text](link)`
- Validates file existence
- Checks anchor references
- Resolves relative and absolute paths
- Detects broken links
- Reports external URL count

#### Validation Rules

1. **File References**: Must exist on filesystem
2. **Anchor Links**: Checked in target file if present
3. **Relative Paths**: Resolved from containing directory
4. **Absolute Paths**: Resolved from project root
5. **External URLs**: Skipped unless `--external` flag

#### Output Files

- **Report**: `.claude/verify/links-validation-report.txt`
- **Log**: `.claude/verify/links-validation.log`

#### Example Output

```
================================================================================
LINK VALIDATION
================================================================================

>>> Results Summary

Total Links: 127
Valid: 125
Broken: 2
External: 5

>>> Broken Links

  ✗ ../non-existent-file.md (from: docs | resolved: /path/to/non-existent-file.md)
  ✗ #missing-anchor in backend/README.md
```

### 3. verify-version-tags.sh

Ensures version tags and dates are consistent across documentation.

#### Usage

```bash
# Check version consistency
./verify-version-tags.sh

# Generate detailed report
./verify-version-tags.sh --report

# Update all versions to specific version
./verify-version-tags.sh --fix --update 3.0.0-alpha.180

# Strict mode (fail on mismatch)
./verify-version-tags.sh --strict

# Verbose output
./verify-version-tags.sh --verbose
```

#### Features

- Extracts version tags from markdown
- Validates semantic versioning format
- Checks last-updated dates
- Detects future dates (error condition)
- Updates versions consistently
- Updates last-modified dates automatically

#### Version Format

Supports standard semantic versioning:

```
3.0.0              # Release version
3.0.0-alpha.180    # Pre-release with build metadata
2.1.5-beta         # Beta release
```

#### Output Files

- **Report**: `.claude/verify/version-tags-report.txt`
- **Log**: `.claude/verify/version-tags.log`

#### Example Update

```bash
./verify-version-tags.sh --fix --update 3.0.1

# Updates all instances of:
# Version: 3.0.0 → Version: 3.0.1
# Last Updated: 2026-01-29 (automatically set to today)
```

### 4. verify-doc-count.sh

Tracks and validates documentation metrics.

#### Usage

```bash
# Count documentation files
./verify-doc-count.sh

# Generate detailed report
./verify-doc-count.sh --report

# Compare with baseline
./verify-doc-count.sh --compare

# Check with 15% threshold
./verify-doc-count.sh --compare --threshold 15

# Verbose analysis
./verify-doc-count.sh --report --verbose
```

#### Features

- Counts markdown files by category
- Calculates total documentation size
- Analyzes size distribution
- Compares with baseline
- Alerts on significant changes
- Provides coverage metrics

#### Categories Tracked

| Category | Location | Purpose |
|----------|----------|---------|
| `claude-internal` | `.claude/` | System documentation |
| `documentation` | `docs/` | Main documentation |
| `backend-docs` | `backend/` | Backend documentation |
| `frontend-docs` | `frontend/` | Frontend documentation |
| `root-docs` | Root directory | Root level docs |

#### Output Files

- **Report**: `.claude/verify/doc-count-report.txt`
- **Baseline**: `.claude/verify/doc-count-baseline.json`
- **Log**: `.claude/verify/doc-count.log`

#### Baseline Example

```json
{
  "timestamp": "2026-01-29 00:32:15",
  "total_docs": 47,
  "total_size": 1234567,
  "categories": {
    "claude-internal": 23,
    "documentation": 12,
    "backend-docs": 8,
    "frontend-docs": 3,
    "root-docs": 1
  }
}
```

### 5. verify-root-cleanup.sh

Ensures root directory is clean and well-organized.

#### Usage

```bash
# Scan root directory
./verify-root-cleanup.sh

# Generate report
./verify-root-cleanup.sh --report

# Clean automatically
./verify-root-cleanup.sh --clean

# Ignore specific files
./verify-root-cleanup.sh --clean --ignore "TEMP.md,DEBUG.txt"

# Strict mode
./verify-root-cleanup.sh --strict

# Verbose scan
./verify-root-cleanup.sh --verbose
```

#### Features

- Scans root directory for clutter
- Identifies temporary and backup files
- Detects duplicate " 2" files
- Verifies required directories exist
- Archives suspicious files (safe)
- Maintains allowed files list

#### Allowed Files

Essential files allowed in root:

```
README.md
LICENSE
package.json
package-lock.json
tsconfig.json
.gitignore
.env.example
.editorconfig
CLAUDE.md
[additional tracking files]
```

#### Temporary Patterns Detected

- `* 2.*` (duplicate files)
- `*.bak` (backups)
- `*.tmp` (temporaries)
- `*.swp` (vim swaps)
- `.DS_Store` (macOS metadata)
- `Thumbs.db` (Windows thumbnails)

#### Clean Mode Actions

```bash
./verify-root-cleanup.sh --clean
```

This action:
1. Removes all duplicate " 2" files
2. Removes temporary files (*.bak, *.tmp, etc.)
3. Archives suspicious files to `.claude/archived-root-files/`

#### Output Files

- **Report**: `.claude/verify/root-cleanup-report.txt`
- **Allowed List**: `.claude/verify/root-allowed-files.txt`
- **Log**: `.claude/verify/root-cleanup.log`
- **Archive**: `.claude/archived-root-files/` (if cleanup used)

## Integration Workflow

### Daily Verification

```bash
# Run all verification scripts
for script in verify-*.sh; do
  echo "Running $script..."
  ./$script --report
done
```

### Pre-Commit Checks

```bash
#!/bin/bash
# Add to .git/hooks/pre-commit

cd scripts

# Must pass all checks
./verify-duplicate-files.sh --strict || exit 1
./verify-links.sh --strict || exit 1
./verify-version-tags.sh --strict || exit 1
./verify-root-cleanup.sh --strict || exit 1
```

### Automated Cleanup

```bash
# Combine multiple cleanup operations
./verify-duplicate-files.sh --fix
./verify-root-cleanup.sh --clean
./verify-version-tags.sh --fix --update "3.0.1"
```

### Continuous Monitoring

```bash
#!/bin/bash
# Run verification suite with comparison

./verify-doc-count.sh --report --compare --threshold 10

if [[ $? -ne 0 ]]; then
  echo "Documentation metrics changed significantly"
  exit 1
fi
```

## Report Analysis

All scripts generate reports in `.claude/verify/`:

### Reading Reports

```bash
# View latest duplicate report
cat .claude/verify/duplicate-files-report.txt

# View all verification reports
ls -lt .claude/verify/*-report.txt | head -5
```

### Interpreting Metrics

**Documentation Count Report:**

```
Total Files: 47         # Total markdown files
Total Size: 1.2 MB     # Combined size of all docs

Documentation by Category
  claude-internal: 23   # System documentation
  documentation: 12     # Main docs
  backend-docs: 8       # Backend specific
  frontend-docs: 3      # Frontend specific
  root-docs: 1          # Root files
```

**Link Validation Report:**

```
Total Links: 127       # All extracted links
Valid: 125            # Working references
Broken: 2             # Missing targets
External: 5           # External URLs
```

**Root Cleanup Report:**

```
Allowed: 10           # Files meeting standards
Duplicates: 7         # " 2" suffixed files
Temporary: 2          # Backup/temp files
Suspicious: 1         # Unorganized files
```

## Troubleshooting

### Script Won't Execute

```bash
# Make scripts executable
chmod +x /path/to/scripts/verify-*.sh

# Verify permissions
ls -l scripts/verify-*.sh
```

### No Reports Generated

```bash
# Ensure .claude/verify directory exists
mkdir -p .claude/verify

# Check write permissions
touch .claude/verify/test.txt
rm .claude/verify/test.txt
```

### Broken Links Not Found

```bash
# Verify link extraction pattern
grep -oE '\[([^\]]+)\]\(([^)]+)\)' file.md

# Check for non-standard link formats
# Markdown links must follow: [text](url)
```

### Version Update Failed

```bash
# Verify semantic version format
# Valid: 3.0.0, 3.0.0-alpha.180, 2.1.5-beta
# Invalid: 3.0, v3.0.0, 3.0.0.1

./verify-version-tags.sh --update 3.0.0  # Correct
```

## Best Practices

### 1. Regular Verification

Run verification scripts weekly:

```bash
# Weekly verification
0 9 * * 1 cd /project && ./scripts/verify-*.sh --report
```

### 2. Pre-Release Checks

Before releasing:

```bash
# Comprehensive pre-release check
./verify-duplicate-files.sh --strict
./verify-links.sh --external --strict
./verify-version-tags.sh --strict
./verify-doc-count.sh --compare --threshold 5
./verify-root-cleanup.sh --strict
```

### 3. Documentation Standards

Maintain these standards:

- Keep documentation count >20 files
- Average file size <50KB
- All links validated before commit
- Consistent version tags
- Root directory <15 files

### 4. Cleanup Frequency

Regular cleanup prevents accumulation:

```bash
# Monthly cleanup
cd scripts
./verify-duplicate-files.sh --fix
./verify-root-cleanup.sh --clean
./verify-version-tags.sh --fix --update "3.0.1"
```

## Performance Notes

Script performance on typical projects:

| Script | Time | Files Scanned |
|--------|------|--------------|
| `verify-duplicate-files.sh` | <1s | All root files |
| `verify-links.sh` | 2-5s | All markdown files |
| `verify-version-tags.sh` | <1s | All markdown files |
| `verify-doc-count.sh` | <2s | All markdown files |
| `verify-root-cleanup.sh` | <1s | Root directory only |

With `--external` flag, link validation takes 30-60s depending on internet speed.

## Output Locations

All verification outputs are organized in `.claude/verify/`:

```
.claude/verify/
├── duplicate-files-report.txt
├── duplicate-files.log
├── links-validation-report.txt
├── links-validation.log
├── version-tags-report.txt
├── version-tags.log
├── doc-count-report.txt
├── doc-count-baseline.json
├── doc-count.log
├── root-cleanup-report.txt
├── root-allowed-files.txt
├── root-cleanup.log
└── archived-root-files/
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Documentation Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run verification scripts
        run: |
          cd scripts
          ./verify-duplicate-files.sh --strict
          ./verify-links.sh --strict
          ./verify-version-tags.sh --strict
          ./verify-root-cleanup.sh --strict

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: verification-reports
          path: .claude/verify/
```

## Support and Maintenance

Scripts are maintained in `/scripts/` and documented here. For issues or enhancements:

1. Check script logs: `.claude/verify/*.log`
2. Review report files: `.claude/verify/*-report.txt`
3. Run with `--verbose` flag for detailed output

## Version History

- **3.0.0** (2026-01-29): Initial verification scripts
  - Duplicate file detection
  - Link validation
  - Version tag consistency
  - Documentation count tracking
  - Root directory cleanup

---

**Next**: See individual script headers for command examples and advanced usage.
