# Documentation Reorganization Plan

**Created**: 2026-01-29
**Status**: Architecture Design Phase
**Scope**: 150+ markdown files reorganization

---

## Executive Summary

Current state: 150+ markdown files scattered across root, docs/, .claude/, backend/, and subdirectories
Target state: Organized, navigable documentation structure with breadcrumbs and cross-references
Method: Git-preserving file moves, automated link updates, comprehensive indexing

---

## Current State Analysis

### File Distribution

| Location | Count | Purpose | Issues |
|----------|-------|---------|--------|
| Root (/) | ~15 | Status reports, summaries | Cluttered, hard to find |
| docs/ | ~100+ | Main documentation | Needs categorization |
| docs/reports/ | ~30 | Implementation reports | Mixed with duplicates (" 2.md") |
| docs/deployment/ | ~10 | Deployment guides | Good structure, keep |
| docs/integration/ | ~5 | Integration status | Good structure, keep |
| docs/investigation/ | ~8 | Analysis reports | Archive or consolidate |
| docs/architecture/ | ~7 | Architecture docs | Expand this section |
| docs/ml/ | ~3 | ML documentation | Needs expansion |
| docs/api/ | ~1 | API migration | Needs expansion |
| docs/development/ | ~2 | Dev guidelines | Needs expansion |
| .claude/ | ~100+ | Agent configs, learning | Keep separate, but organize |
| backend/docs/ | ~3 | Backend-specific | Move to docs/backend/ |
| backend/tests/ | ~10 | Test documentation | Move to docs/testing/ |
| backend/security/ | ~1 | Security README | Move to docs/security/ |

### Key Issues Identified

1. **Root clutter**: Phase reports and summaries in root folder
2. **Duplicate files**: Files with " 2.md" suffix (38+ duplicates)
3. **Scattered backend docs**: Backend documentation not centralized
4. **Missing organization**: No clear categorization system
5. **Poor discoverability**: Hard to find relevant documentation
6. **No breadcrumbs**: Difficult navigation within docs
7. **Broken links risk**: Reorganization will break existing links

---

## Proposed Directory Structure

```
docs/
├── README.md                          # Main documentation entry point
├── DOCUMENTATION_INDEX.md             # Master index (exists, needs update)
│
├── 01-getting-started/                # NEW: Quick start guides
│   ├── README.md                      # Getting started overview
│   ├── QUICK_START.md                 # 5-minute quick start
│   ├── INSTALLATION_GUIDE.md          # Detailed installation
│   ├── DEVELOPMENT_SETUP.md           # Dev environment setup
│   └── TROUBLESHOOTING_QUICK.md       # Common issues quick ref
│
├── 02-architecture/                   # Architecture & design docs
│   ├── README.md                      # Architecture overview
│   ├── SYSTEM_ARCHITECTURE.md         # High-level architecture
│   ├── DATA_FLOW.md                   # Data flow diagrams
│   ├── CODEMAPS/                      # Detailed code maps (exists)
│   │   ├── README.md
│   │   ├── BACKEND.md
│   │   ├── FRONTEND.md
│   │   ├── INFRASTRUCTURE.md
│   │   └── DATA_FLOW.md
│   ├── DESIGN_DECISIONS.md            # Architecture Decision Records
│   ├── PERFORMANCE_ARCHITECTURE.md    # Performance design
│   ├── SECURITY_ARCHITECTURE.md       # Security design
│   ├── MULTI_SOURCE_ETL_SOLUTION.md   # ETL architecture (move from architecture/)
│   ├── UNLIMITED_DATA_EXTRACTION_SOLUTION.md
│   └── COMPREHENSIVE_CACHING_SYSTEM.md
│
├── 03-development/                    # Development guides
│   ├── README.md                      # Development overview
│   ├── CODING_STANDARDS.md            # Code style, conventions
│   ├── TYPE_GUIDELINES.md             # TypeScript type guidelines (move from development/)
│   ├── TYPE_MIGRATION_STATUS.md       # Type migration tracking
│   ├── API_DEVELOPMENT.md             # API development guide
│   ├── FRONTEND_DEVELOPMENT.md        # Frontend dev guide
│   ├── BACKEND_DEVELOPMENT.md         # Backend dev guide
│   ├── DATABASE_DEVELOPMENT.md        # Database dev guide
│   └── CONTRIBUTING.md                # Contribution guidelines
│
├── 04-api/                            # API documentation
│   ├── README.md                      # API overview
│   ├── API_REFERENCE.md               # Complete API reference
│   ├── API_AUTHENTICATION.md          # Auth & authorization
│   ├── API_RATE_LIMITING.md           # Rate limiting docs
│   ├── V1_TO_V2_MIGRATION_GUIDE.md    # Migration guide (exists)
│   ├── ENDPOINTS/                     # Endpoint-specific docs
│   │   ├── auth.md
│   │   ├── portfolio.md
│   │   ├── stocks.md
│   │   └── ml.md
│   └── EXAMPLES/                      # API usage examples
│       └── python_client.md
│
├── 05-security/                       # Security documentation
│   ├── README.md                      # Security overview (move from backend/security/)
│   ├── SECURITY.md                    # Security guide (move from root)
│   ├── AUTHENTICATION.md              # Auth implementation
│   ├── AUTHORIZATION.md               # RBAC implementation
│   ├── DATA_PROTECTION.md             # Encryption, PII
│   ├── API_SECURITY.md                # API security measures
│   ├── COMPLIANCE.md                  # SEC, GDPR compliance
│   ├── INCIDENT_RESPONSE.md           # Security incidents
│   └── SECURITY_CHECKLIST.md          # Security verification
│
├── 06-testing/                        # Testing documentation
│   ├── README.md                      # Testing overview (move from backend/tests/)
│   ├── TESTING_STRATEGY.md            # Overall test strategy
│   ├── UNIT_TESTING.md                # Unit test guide
│   ├── INTEGRATION_TESTING.md         # Integration test guide
│   ├── E2E_TESTING.md                 # E2E test guide
│   ├── PERFORMANCE_TESTING.md         # Load/performance testing
│   ├── COVERAGE_ANALYSIS.md           # Coverage reports (move from backend/tests/)
│   ├── TEST_INFRASTRUCTURE_GUIDE.md   # Test infrastructure (move from backend/tests/)
│   └── VALIDATION/
│       ├── API_STANDARDIZATION_VALIDATION.md
│       └── TEST_VERIFICATION_INDEX.md (move from root)
│
├── 07-ml/                             # Machine learning docs
│   ├── README.md                      # ML overview
│   ├── ML_ARCHITECTURE.md             # ML pipeline architecture
│   ├── ML_API_REFERENCE.md            # ML API docs (exists)
│   ├── ML_OPERATIONS_GUIDE.md         # MLOps guide (exists)
│   ├── MODEL_TRAINING.md              # Model training guide
│   ├── MODEL_DEPLOYMENT.md            # Model deployment
│   ├── GPU_SUPPORT.md                 # GPU configuration (exists)
│   └── MONITORING.md                  # ML model monitoring
│
├── 08-deployment/                     # Deployment & operations
│   ├── README.md                      # Deployment overview
│   ├── DEPLOYMENT.md                  # Main deployment guide (move from root)
│   ├── PRODUCTION_DEPLOYMENT_GUIDE.md # Production guide (move from root)
│   ├── README_PRODUCTION_GUIDE.md     # Production readme (move from root)
│   ├── ENVIRONMENT.md                 # Environment config (move from root)
│   ├── DOCKER.md                      # Docker setup
│   ├── KUBERNETES.md                  # K8s deployment
│   ├── DATABASE_SETUP.md              # Database initialization
│   ├── SSL_CERTIFICATES.md            # SSL/TLS setup
│   ├── MONITORING_SETUP.md            # Monitoring stack
│   ├── BACKUP_RECOVERY.md             # Backup procedures
│   └── PHASES/                        # Deployment phase reports (keep existing)
│       ├── PHASE1_EXECUTIVE_SUMMARY.md
│       ├── PHASE1_FINAL_SUMMARY.md
│       ├── PHASE2_EXECUTION_PLAN.md
│       ├── PHASE2B_ROOT_CAUSE_ANALYSIS.md
│       └── ...
│
├── 09-operations/                     # Day-to-day operations
│   ├── README.md                      # Operations overview
│   ├── RUNBOOK.md                     # Operations runbook
│   ├── TROUBLESHOOTING.md             # Troubleshooting guide (move from root)
│   ├── MONITORING.md                  # Monitoring guide
│   ├── PERFORMANCE_OPTIMIZATION.md    # Performance tuning (move from root)
│   ├── PERFORMANCE_BENCHMARKS.md      # Benchmark results (move from root)
│   ├── DATABASE_OPTIMIZATION_GUIDE.md # DB optimization (move from backend/docs/)
│   ├── INCIDENT_RESPONSE.md           # Incident procedures
│   ├── MAINTENANCE.md                 # Maintenance tasks
│   └── SCRIPTS_REFERENCE.md           # Script reference (move from root)
│
├── 10-integration/                    # Integration docs
│   ├── README.md                      # Integration overview (exists)
│   ├── INTEGRATION_SUMMARY.md         # Integration summary (exists)
│   ├── PHASE3_INTEGRATION_VALIDATION.md # Phase 3 validation (exists)
│   ├── CONFLICTS_RESOLVED.md          # Conflict resolution (exists)
│   ├── THIRD_PARTY_APIS.md            # External API integration
│   └── DATA_SOURCES.md                # Data source integration
│
├── 11-reports/                        # Implementation & status reports
│   ├── README.md                      # Reports overview
│   ├── CURRENT/                       # Current status reports
│   │   ├── IMPLEMENTATION_TRACKER.md  # Move from root
│   │   ├── IMPLEMENTATION_STATUS.md   # Move from root
│   │   ├── DELIVERY_SUMMARY.md        # Move from root
│   │   └── NOTION_TASK_MAPPING.md     # Move from root
│   ├── PHASE_REPORTS/                 # Phase completion reports
│   │   ├── PHASE1_IMPLEMENTATION_COMPLETE.md (move from .claude/)
│   │   ├── PHASE3_IMPLEMENTATION_COMPLETE.md (move from root)
│   │   ├── PHASE3_QUICK_REFERENCE.md (move from .claude/)
│   │   ├── PHASE4_REVIEW_SYNTHESIS.md
│   │   ├── PHASE5_SUMMARY.md (move from .claude/)
│   │   └── PHASE6_PHASE1_COMPLETE.md
│   ├── VALIDATION/                    # Validation reports
│   │   ├── PHASE3_VALIDATION_REPORT.md
│   │   ├── TEST_VERIFICATION_REPORT.md (move from root)
│   │   └── PRODUCTION_READINESS_CHECKLIST.md
│   └── ARCHIVE/                       # OLD: Archived reports
│       ├── phase2-checklist.md
│       ├── phase2-validation-report.md
│       └── wave4_completion_report.md
│
├── 12-workflows/                      # Workflow & process docs
│   ├── README.md                      # Workflow overview
│   ├── WORKFLOW_COORDINATION_SUMMARY.md # Move from root
│   ├── GITHUB_WORKFLOWS.md            # GitHub Actions guide
│   ├── CI_CD.md                       # CI/CD pipeline
│   ├── RELEASE_PROCESS.md             # Release management
│   └── CODE_REVIEW.md                 # Code review process
│
├── 13-reference/                      # Reference materials
│   ├── README.md                      # Reference overview
│   ├── GLOSSARY.md                    # Terms & definitions
│   ├── LINKS.md                       # Important links
│   ├── CONTACTS.md                    # Team contacts
│   └── RESOURCES.md                   # External resources
│
└── 99-archive/                        # Archived/deprecated docs
    ├── README.md                      # Archive index
    ├── DEPRECATED_AGENTS/             # Move from docs/archive/deprecated-agents/
    ├── INVESTIGATION/                 # Move investigation docs here
    │   ├── CONSOLIDATED_FINDINGS.md
    │   ├── DECISION_MATRIX.md
    │   └── INFRASTRUCTURE_ANALYSIS.md
    └── DUPLICATE_CLEANUP/             # Files with " 2.md" suffix
        └── README.md                  # Explains cleanup
```

---

## File Movement Plan

### Phase 1: Root Cleanup (Priority: HIGH)

Move from root to appropriate locations:

```bash
# Status & tracking docs → reports/CURRENT/
mv IMPLEMENTATION_TRACKER.md docs/11-reports/CURRENT/
mv IMPLEMENTATION_STATUS.md docs/11-reports/CURRENT/
mv DELIVERY_SUMMARY.md docs/11-reports/CURRENT/
mv NOTION_TASK_MAPPING.md docs/11-reports/CURRENT/

# Phase reports → reports/PHASE_REPORTS/
mv PHASE3_IMPLEMENTATION_COMPLETE.md docs/11-reports/PHASE_REPORTS/

# Test verification → testing/VALIDATION/
mv TEST_VERIFICATION_INDEX.md docs/06-testing/VALIDATION/
mv TEST_VERIFICATION_REPORT.md docs/11-reports/VALIDATION/
mv AUTH_FLOW_TEST_PROGRESS.md docs/11-reports/VALIDATION/

# Middleware fixes → reports/ARCHIVE/
mv MIDDLEWARE_FIXES_SUMMARY.md docs/11-reports/ARCHIVE/
mv NEXT_STEPS_DEBUG.md docs/11-reports/ARCHIVE/

# Core operational docs → appropriate sections
mv DEPLOYMENT.md docs/08-deployment/
mv PRODUCTION_DEPLOYMENT_GUIDE.md docs/08-deployment/
mv README_PRODUCTION_GUIDE.md docs/08-deployment/
mv ENVIRONMENT.md docs/08-deployment/
mv TROUBLESHOOTING.md docs/09-operations/
mv PERFORMANCE_OPTIMIZATION.md docs/09-operations/
mv PERFORMANCE_BENCHMARKS.md docs/09-operations/
mv SCRIPTS_REFERENCE.md docs/09-operations/
mv SECURITY.md docs/05-security/
mv WORKFLOW_COORDINATION_SUMMARY.md docs/12-workflows/
mv INSTALLATION_GUIDE.md docs/01-getting-started/
mv INTEGRATION_SUMMARY.md docs/10-integration/
mv INVESTMENT_THESIS_FEATURE.md docs/03-development/
mv QA_ACTION_PLAN.md docs/06-testing/
mv WSL_INSTALLATION_FIXES.md docs/01-getting-started/
```

### Phase 2: docs/ Reorganization (Priority: HIGH)

```bash
# Architecture docs → 02-architecture/
mv docs/architecture/* docs/02-architecture/
mv docs/CODEMAPS docs/02-architecture/

# Development docs → 03-development/
mv docs/development/* docs/03-development/

# API docs → 04-api/
mv docs/api/* docs/04-api/

# ML docs → 07-ml/
mv docs/ml/* docs/07-ml/

# Deployment docs → 08-deployment/PHASES/
mv docs/deployment/* docs/08-deployment/PHASES/

# Integration docs → 10-integration/
mv docs/integration/* docs/10-integration/

# Reports with " 2.md" → 11-reports/ARCHIVE/ or cleanup
mv docs/reports/*" 2.md" docs/11-reports/ARCHIVE/

# Investigation docs → 99-archive/INVESTIGATION/
mv docs/investigation/* docs/99-archive/INVESTIGATION/

# Main reports → 11-reports/
mv docs/reports/* docs/11-reports/PHASE_REPORTS/
```

### Phase 3: backend/ Documentation (Priority: MEDIUM)

```bash
# Backend docs → appropriate locations
mv backend/docs/DATABASE_OPTIMIZATION_GUIDE.md docs/09-operations/
mv backend/docs/deployment/* docs/08-deployment/PHASES/

# Security README → 05-security/
mv backend/security/README.md docs/05-security/BACKEND_SECURITY.md

# Test docs → 06-testing/
mv backend/tests/README.md docs/06-testing/BACKEND_TESTING.md
mv backend/tests/COVERAGE_ANALYSIS.md docs/06-testing/
mv backend/tests/TEST_INFRASTRUCTURE_GUIDE.md docs/06-testing/
mv backend/tests/API_STANDARDIZATION_VALIDATION.md docs/06-testing/VALIDATION/
mv backend/tests/BREAKING_CHANGES_SUMMARY.md docs/11-reports/ARCHIVE/
mv backend/tests/DOCUMENTATION_SUMMARY.md docs/11-reports/ARCHIVE/
mv backend/tests/integration/INTEGRATION_TEST_SUMMARY.md docs/06-testing/
```

### Phase 4: .claude/ Organization (Priority: LOW)

Keep .claude/ separate but organize internally:

```bash
# Move phase reports to docs/
mv .claude/PHASE1_IMPLEMENTATION_COMPLETE.md docs/11-reports/PHASE_REPORTS/
mv .claude/PHASE1_QUICKSTART.md docs/11-reports/PHASE_REPORTS/
mv .claude/PHASE1_EXECUTION_SUMMARY.md docs/11-reports/PHASE_REPORTS/
mv .claude/PHASE1_MIGRATION_GUIDE.md docs/11-reports/PHASE_REPORTS/
mv .claude/PHASE3_QUICK_REFERENCE.md docs/11-reports/PHASE_REPORTS/
mv .claude/PHASE5_SUMMARY.md docs/11-reports/PHASE_REPORTS/
mv .claude/PHASE5_WORKER_ENABLEMENT.md docs/11-reports/PHASE_REPORTS/

# Move general docs to appropriate docs/ locations
mv .claude/INTEGRATION_QUICKSTART.md docs/10-integration/
mv .claude/COMPREHENSIVE_INTEGRATION_COMPLETE.md docs/11-reports/VALIDATION/

# Keep .claude/ for:
# - Agent configurations (.claude/agents/)
# - Skills (.claude/skills/)
# - Hooks (.claude/hooks/)
# - Memory (.claude/memory/)
# - Learning patterns (.claude/learned-patterns/)
# - Scripts (.claude/scripts/)
```

### Phase 5: Duplicate Cleanup (Priority: HIGH)

Remove or consolidate files with " 2.md" suffix:

```bash
# List all duplicates (38+ files)
find docs -name "*\ 2.md" -type f

# Strategy:
# 1. Compare with original (diff)
# 2. If identical → delete duplicate
# 3. If different → merge changes into original, then delete
# 4. If original missing → rename " 2.md" to ".md"

# Example cleanup script:
for file in $(find docs -name "*\ 2.md"); do
  original="${file% 2.md}.md"
  if [ -f "$original" ]; then
    if diff -q "$file" "$original" > /dev/null; then
      echo "Removing duplicate: $file"
      git rm "$file"
    else
      echo "MANUAL REVIEW NEEDED: $file differs from $original"
    fi
  else
    echo "Renaming: $file → $original"
    git mv "$file" "$original"
  fi
done
```

---

## Git History Preservation

All file moves MUST use `git mv` to preserve history:

```bash
# GOOD: Preserves history
git mv old/path/file.md new/path/file.md

# BAD: Breaks history
mv old/path/file.md new/path/file.md
git add new/path/file.md
git rm old/path/file.md
```

### Verification

After each move, verify history is preserved:

```bash
git log --follow -- new/path/file.md
```

---

## Link Update Strategy

### Step 1: Inventory All Links

```bash
# Find all markdown links in docs
grep -r "\[.*\](.*\.md)" docs/ > links_inventory.txt

# Find all relative links
grep -r "\]\(./" docs/ >> links_inventory.txt

# Find all absolute links to docs/
grep -r "\](/docs/" docs/ >> links_inventory.txt
```

### Step 2: Create Link Mapping

```json
{
  "link_mappings": {
    "DEPLOYMENT.md": "08-deployment/DEPLOYMENT.md",
    "SECURITY.md": "05-security/SECURITY.md",
    "TROUBLESHOOTING.md": "09-operations/TROUBLESHOOTING.md",
    "IMPLEMENTATION_TRACKER.md": "11-reports/CURRENT/IMPLEMENTATION_TRACKER.md",
    "CODEMAPS/BACKEND.md": "02-architecture/CODEMAPS/BACKEND.md"
  }
}
```

### Step 3: Automated Link Updates

```bash
#!/bin/bash
# update_links.sh

# For each markdown file
find docs -name "*.md" -type f | while read file; do
  # Update links based on mapping
  sed -i 's|\](DEPLOYMENT.md)|\](../../08-deployment/DEPLOYMENT.md)|g' "$file"
  sed -i 's|\](SECURITY.md)|\](../../05-security/SECURITY.md)|g' "$file"
  # ... etc for all mappings
done
```

### Step 4: Validation

```bash
# Find broken links after update
find docs -name "*.md" -exec grep -H "\[.*\](.*\.md)" {} \; | \
  while IFS=: read file link; do
    target=$(echo "$link" | sed -n 's/.*](\(.*\.md\)).*/\1/p')
    if [ ! -f "$(dirname "$file")/$target" ]; then
      echo "BROKEN: $file → $target"
    fi
  done
```

---

## Breadcrumb Navigation Design

### Format

```markdown
<!-- At top of each file -->
**Navigation**: [Home](../README.md) > [Category](../README.md) > Current Page

---
```

### Examples

```markdown
# In docs/05-security/AUTHENTICATION.md
**Navigation**: [Docs Home](../README.md) > [Security](README.md) > Authentication

---

# In docs/08-deployment/PHASES/PHASE1_EXECUTIVE_SUMMARY.md
**Navigation**: [Docs Home](../../README.md) > [Deployment](../README.md) > [Phases](README.md) > Phase 1 Executive Summary

---
```

### Automated Breadcrumb Generation

```bash
#!/bin/bash
# add_breadcrumbs.sh

find docs -name "*.md" -type f | while read file; do
  # Calculate depth
  depth=$(echo "$file" | tr -cd '/' | wc -c)

  # Generate relative path to root
  root_path=$(printf '../%.0s' $(seq 1 $depth))

  # Extract category from directory
  category=$(dirname "$file" | xargs basename)

  # Generate breadcrumb
  breadcrumb="**Navigation**: [Docs Home](${root_path}README.md) > [$category](../README.md) > $(basename "$file" .md)"

  # Insert at top of file (after title if present)
  # ... implementation
done
```

---

## Cross-Reference Strategy

### Reference Types

1. **Related Documentation**: Links to related docs
2. **Prerequisites**: Required reading before this doc
3. **Next Steps**: What to read after this doc
4. **See Also**: Additional relevant docs

### Format

```markdown
## Related Documentation

- **Prerequisites**: [Installation Guide](../01-getting-started/INSTALLATION_GUIDE.md)
- **Related**: [Security Architecture](../02-architecture/SECURITY_ARCHITECTURE.md)
- **Next Steps**: [Deployment](../08-deployment/DEPLOYMENT.md)
- **See Also**:
  - [API Security](API_SECURITY.md)
  - [Compliance](COMPLIANCE.md)

---
```

### Cross-Reference Index

Create `docs/CROSS_REFERENCE_INDEX.md`:

```markdown
# Cross-Reference Index

## By Topic

### Security
- [Security Overview](05-security/README.md)
- [Security Architecture](02-architecture/SECURITY_ARCHITECTURE.md)
- [API Security](05-security/API_SECURITY.md)
- [Security Checklist](05-security/SECURITY_CHECKLIST.md)

### Deployment
- [Deployment Overview](08-deployment/README.md)
- [Production Guide](08-deployment/README_PRODUCTION_GUIDE.md)
- [Environment Setup](08-deployment/ENVIRONMENT.md)
...
```

---

## Master Index Updates

Update `docs/DOCUMENTATION_INDEX.md` with new structure:

```markdown
# Documentation Index

**Last Updated**: 2026-01-29
**Total Documents**: 150+
**Organization**: 13 categories + archive

---

## Quick Navigation

1. [Getting Started](01-getting-started/README.md) - Installation & setup
2. [Architecture](02-architecture/README.md) - System design & architecture
3. [Development](03-development/README.md) - Developer guides
4. [API](04-api/README.md) - API documentation
5. [Security](05-security/README.md) - Security & compliance
6. [Testing](06-testing/README.md) - Testing guides
7. [ML](07-ml/README.md) - Machine learning
8. [Deployment](08-deployment/README.md) - Deployment & DevOps
9. [Operations](09-operations/README.md) - Day-to-day operations
10. [Integration](10-integration/README.md) - Integration docs
11. [Reports](11-reports/README.md) - Status & reports
12. [Workflows](12-workflows/README.md) - Process & workflows
13. [Reference](13-reference/README.md) - Reference materials

---

## By Role

### Developers
- Start: [Development Guide](03-development/README.md)
- API: [API Reference](04-api/README.md)
- Architecture: [Code Maps](02-architecture/CODEMAPS/README.md)

### DevOps
- Start: [Deployment Guide](08-deployment/README.md)
- Operations: [Runbook](09-operations/RUNBOOK.md)
- Troubleshooting: [Troubleshooting](09-operations/TROUBLESHOOTING.md)

...
```

---

## Implementation Checklist

### Pre-Move Preparation

- [ ] Backup entire repository
- [ ] Create new branch: `docs/reorganization`
- [ ] Document current link inventory
- [ ] Test git mv on sample files
- [ ] Create link mapping file
- [ ] Write link update script
- [ ] Write breadcrumb script
- [ ] Review duplicate files manually

### Phase 1: Create New Structure

- [ ] Create all new category directories
- [ ] Create README.md in each category
- [ ] Create placeholder index files
- [ ] Commit structure: `git commit -m "docs: Create new documentation structure"`

### Phase 2: Move Files

- [ ] Execute root cleanup moves (git mv)
- [ ] Execute docs/ reorganization (git mv)
- [ ] Execute backend docs moves (git mv)
- [ ] Execute .claude/ moves (git mv)
- [ ] Verify history preservation for each file
- [ ] Commit moves: `git commit -m "docs: Reorganize documentation files"`

### Phase 3: Cleanup Duplicates

- [ ] Identify all " 2.md" files
- [ ] Compare with originals
- [ ] Merge differences
- [ ] Remove duplicates (git rm)
- [ ] Commit cleanup: `git commit -m "docs: Remove duplicate documentation files"`

### Phase 4: Update Links

- [ ] Run link update script
- [ ] Validate all links
- [ ] Fix broken links manually
- [ ] Commit link updates: `git commit -m "docs: Update internal documentation links"`

### Phase 5: Add Navigation

- [ ] Add breadcrumbs to all files
- [ ] Add cross-references to related docs
- [ ] Update DOCUMENTATION_INDEX.md
- [ ] Create CROSS_REFERENCE_INDEX.md
- [ ] Commit navigation: `git commit -m "docs: Add breadcrumb navigation and cross-references"`

### Phase 6: Validation

- [ ] Build documentation site (if applicable)
- [ ] Check all links manually
- [ ] Review navigation flow
- [ ] Test search functionality
- [ ] Get team review
- [ ] Commit final fixes: `git commit -m "docs: Final validation fixes"`

### Phase 7: Deployment

- [ ] Merge to main: `git merge docs/reorganization`
- [ ] Tag release: `git tag docs-v2.0.0`
- [ ] Update documentation site
- [ ] Announce reorganization to team
- [ ] Create migration guide for external links

---

## Risk Mitigation

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Broken external links | HIGH | Maintain redirects, publish changelog |
| Lost git history | CRITICAL | ALWAYS use `git mv`, verify after each move |
| Missing files | MEDIUM | Complete inventory before moves, verify after |
| Team confusion | MEDIUM | Clear communication, migration guide |
| Build failures | HIGH | Test builds before merging |
| Search index broken | LOW | Rebuild search index after reorganization |

### Rollback Plan

If issues arise:

```bash
# Rollback to pre-reorganization state
git checkout main
git branch -D docs/reorganization

# Or revert specific commits
git revert <commit-hash>
```

---

## Success Criteria

- [ ] All 150+ files organized into logical categories
- [ ] Zero broken internal links
- [ ] Git history preserved for all moved files
- [ ] Breadcrumb navigation on all docs
- [ ] Cross-references added to related docs
- [ ] Master index fully updated
- [ ] No duplicate files (all " 2.md" resolved)
- [ ] README.md in every category
- [ ] Team approval received
- [ ] Documentation builds successfully

---

## Timeline Estimate

| Phase | Effort | Duration |
|-------|--------|----------|
| Preparation | 4 hours | Day 1 |
| Structure creation | 2 hours | Day 1 |
| File moves | 6 hours | Day 2 |
| Duplicate cleanup | 4 hours | Day 2-3 |
| Link updates | 8 hours | Day 3-4 |
| Navigation | 6 hours | Day 4-5 |
| Validation | 4 hours | Day 5 |
| **Total** | **34 hours** | **5 days** |

---

## Post-Reorganization Maintenance

### Documentation Standards

1. **New files**: MUST go in appropriate category
2. **Breadcrumbs**: MUST be added to all new docs
3. **Cross-references**: SHOULD be added when relevant
4. **Index**: MUST be updated monthly
5. **Archive**: OLD docs move to 99-archive/

### Review Cadence

- **Weekly**: Check for misplaced files
- **Monthly**: Update DOCUMENTATION_INDEX.md
- **Quarterly**: Review and prune archive
- **Annually**: Major reorganization if needed

---

## Next Steps

1. Review this plan with team
2. Get approval from stakeholders
3. Create `docs/reorganization` branch
4. Execute Phase 1 (preparation)
5. Begin file moves systematically
6. Validate and merge

---

**Status**: Ready for implementation
**Estimated Completion**: 5 business days
**Risk Level**: Low (with proper git mv usage)
**Team Impact**: Medium (communication required)

---

*Document maintained by: Architecture Team*
*Last reviewed: 2026-01-29*
