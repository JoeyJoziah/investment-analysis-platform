# Verification Scripts Quick Start

**Last Updated: 2026-01-29**

Quick reference for the five documentation verification scripts.

## TL;DR

```bash
cd scripts

# 1. Check for duplicate " 2" files
./verify-duplicate-files.sh

# 2. Validate all documentation links
./verify-links.sh

# 3. Check version tag consistency
./verify-version-tags.sh

# 4. Track documentation metrics
./verify-doc-count.sh

# 5. Clean up root directory
./verify-root-cleanup.sh
```

## Quick Commands

### Duplicate Files

```bash
# Detect duplicates
./verify-duplicate-files.sh

# Remove duplicates (safe)
./verify-duplicate-files.sh --fix

# Generate report
./verify-duplicate-files.sh --report

# Report + fix
./verify-duplicate-files.sh --report --fix
```

**Detects**: Files with " 2" suffix pattern

### Link Validation

```bash
# Validate internal links
./verify-links.sh

# Generate detailed report
./verify-links.sh --report

# Check external URLs too (slower)
./verify-links.sh --external

# Full check with report
./verify-links.sh --report --external --verbose
```

**Checks**: File existence, anchors, relative paths

### Version Tags

```bash
# Check consistency
./verify-version-tags.sh

# Report issues
./verify-version-tags.sh --report

# Update to new version
./verify-version-tags.sh --fix --update 3.0.1

# Strict mode (fail on mismatch)
./verify-version-tags.sh --strict
```

**Format**: Semantic versioning (X.Y.Z or X.Y.Z-tag)

### Documentation Count

```bash
# Count documentation files
./verify-doc-count.sh

# Generate report
./verify-doc-count.sh --report

# Compare with baseline
./verify-doc-count.sh --compare

# Check with 15% threshold
./verify-doc-count.sh --compare --threshold 15
```

**Tracks**: File count, size, categories, changes

### Root Cleanup

```bash
# Scan root directory
./verify-root-cleanup.sh

# Generate report
./verify-root-cleanup.sh --report

# Clean automatically
./verify-root-cleanup.sh --clean

# Ignore certain files
./verify-root-cleanup.sh --clean --ignore "TEMP.md"

# Strict mode
./verify-root-cleanup.sh --strict
```

**Removes**: Duplicates, temp files, archives unorganized files

## Full Verification Suite

Run all verifications:

```bash
#!/bin/bash
cd scripts

echo "Running all verifications..."
./verify-duplicate-files.sh --report
./verify-links.sh --report
./verify-version-tags.sh --report
./verify-doc-count.sh --report --compare
./verify-root-cleanup.sh --report

echo "Reports saved to: .claude/verify/"
```

## Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash

cd scripts || exit 1

# Must pass all checks
./verify-duplicate-files.sh --strict || exit 1
./verify-links.sh --strict || exit 1
./verify-version-tags.sh --strict || exit 1
./verify-root-cleanup.sh --strict || exit 1

echo "All verification checks passed!"
```

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

## Outputs

All reports saved to `.claude/verify/`:

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
└── root-cleanup.log
```

View reports:

```bash
cat .claude/verify/duplicate-files-report.txt
cat .claude/verify/links-validation-report.txt
cat .claude/verify/version-tags-report.txt
cat .claude/verify/doc-count-report.txt
cat .claude/verify/root-cleanup-report.txt
```

## Common Issues

### Issue: No reports generated

**Solution**: Ensure directory exists
```bash
mkdir -p .claude/verify
chmod 755 .claude/verify
```

### Issue: Scripts not executable

**Solution**: Fix permissions
```bash
chmod +x scripts/verify-*.sh
```

### Issue: Duplicate files not detected

**Solution**: Files must be in root or 3 levels deep
```bash
# Detected
PROJECT_ROOT/FILE 2.md

# Not detected (too deep)
PROJECT_ROOT/very/deep/nested/FILE 2.md
```

### Issue: Links showing as broken

**Solution**: Verify file exists
```bash
# Check if file exists
ls -la path/to/file.md

# Use absolute path from project root
# Good: /docs/guide.md
# Bad: docs/guide.md (in links)
```

## Exit Codes

All scripts follow standard exit codes:

```
0 = Success (all checks passed)
1 = Failure (issues detected or strict mode failed)
```

Use in scripts:

```bash
./verify-duplicate-files.sh --strict
if [[ $? -ne 0 ]]; then
  echo "Duplicate files detected!"
  exit 1
fi
```

## Performance

Typical runtimes on this project:

| Script | Time | Speed |
|--------|------|-------|
| duplicate-files | <1s | Fast |
| links | 2-5s | Moderate |
| version-tags | <1s | Fast |
| doc-count | <2s | Fast |
| root-cleanup | <1s | Fast |

With `--external`: 30-60s for link validation

## Workflows

### Daily

```bash
# Once per day
./verify-duplicate-files.sh --report
./verify-root-cleanup.sh --report
```

### Weekly

```bash
# Full suite once per week
for script in verify-*.sh; do
  ./$script --report
done
```

### Pre-Commit

```bash
# Automatically (via git hook)
# See: .git/hooks/pre-commit
```

### Pre-Release

```bash
# Before each release
./verify-duplicate-files.sh --strict
./verify-links.sh --external --strict
./verify-version-tags.sh --strict
./verify-doc-count.sh --compare --threshold 5
./verify-root-cleanup.sh --strict
```

### After Merge

```bash
# After merging PR
./verify-duplicate-files.sh --fix
./verify-root-cleanup.sh --clean
./verify-doc-count.sh --compare
```

## Advanced Usage

### Custom Ignore Patterns

```bash
# Ignore specific files in root cleanup
./verify-root-cleanup.sh --ignore "TEMP.md,DEBUG.txt,NOTES.md"
```

### Version Update Across All Docs

```bash
# Update all version tags to 3.0.2
./verify-version-tags.sh --fix --update 3.0.2
```

### External Link Check

```bash
# Check all external URLs (requires internet)
./verify-links.sh --external

# External only in reports
./verify-links.sh --report --external
```

### Verbose Debugging

```bash
# Show all detected items
./verify-duplicate-files.sh --verbose
./verify-links.sh --verbose
./verify-doc-count.sh --verbose
```

## Integration

### GitHub Actions

```yaml
name: Verify Documentation

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Verify
        run: |
          cd scripts
          ./verify-duplicate-files.sh --strict
          ./verify-links.sh --strict
          ./verify-root-cleanup.sh --strict
```

### Local Git Hook

```bash
# Install to .git/hooks/pre-commit
cp scripts/verify-*.sh .git/hooks/

# Make executable
chmod +x .git/hooks/pre-commit
```

### CI/CD Pipeline

```bash
# Add to CI/CD configuration
verify-docs:
  script:
    - cd scripts
    - ./verify-duplicate-files.sh --strict
    - ./verify-links.sh --strict
    - ./verify-version-tags.sh --strict
    - ./verify-doc-count.sh --compare
    - ./verify-root-cleanup.sh --strict
```

## See Also

- **Full Guide**: `docs/VERIFICATION_SCRIPTS_GUIDE.md`
- **Individual Scripts**: `scripts/verify-*.sh`
- **Reports**: `.claude/verify/`

---

For detailed information, see the full guide in `docs/VERIFICATION_SCRIPTS_GUIDE.md`.
