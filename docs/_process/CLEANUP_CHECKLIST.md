# Dead Code Cleanup Checklist

**Quick Reference Guide | 2026-02-08**

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Copy-Paste This Command

```bash
cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform

# Delete all safe files (43 files + cache)
rm -f AUTH_FLOW_TEST_PROGRESS.md \
      DOCUMENTATION_UPDATE_SUMMARY.md \
      DUPLICATE_FILES_ANALYSIS.md \
      MIDDLEWARE_FIXES_SUMMARY.md \
      NEXT_STEPS_DEBUG.md \
      PHASE3_IMPLEMENTATION_COMPLETE.md \
      SECURITY_PATTERNS_SCAN.md \
      START_HERE_VERIFICATION.md \
      TEST_VERIFICATION_INDEX.md \
      TEST_VERIFICATION_REPORT.md \
      VERIFICATION_COMPLETE.md \
      VERIFICATION_DELIVERY_SUMMARY.md \
      VERIFICATION_FILE_INDEX.md \
      *" 2".* \
      backend/etl/data_extractor_original_backup.py

rm -f .claude/*" 2".*
rm -f docs/*" 2".*
rm -f backend/middleware/*" 2".py

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Update .gitignore
cat >> .gitignore << 'EOF'

# Python cache (auto-generated)
__pycache__/
*.pyc
*.pyo
*.pyd

# Backup files (duplicates)
* 2.*
*~
*.bak
EOF

echo "✅ Phase 1 cleanup complete! Run tests next."
```

### Step 2: Run Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# If all pass:
cd ..
echo "✅ Tests passed! Ready to commit."
```

### Step 3: Commit

```bash
git add .
git commit -m "refactor: Clean up dead code, duplicates, and cache

- Remove 13 completed progress reports from root
- Delete 29 duplicate ' 2' files across repo
- Remove backup file (preserved in git history)
- Clean 2,676 __pycache__ directories (~150 MB)
- Update .gitignore for cache and backups

See docs/DELETION_LOG.md for full audit report.
Estimated space savings: ~150 MB

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

**Done!** ✅ 150 MB freed, 0 risk, 5 minutes.

---

## 📋 Detailed Checklist

### Phase 1: Zero-Risk Cleanup (Execute Now)

#### Pre-Flight Checks
- [ ] Current directory is project root
- [ ] Git status is clean (or changes are committed)
- [ ] Backup exists (optional - git preserves everything)

#### Execution
- [ ] Delete 13 root progress reports
- [ ] Delete 29 " 2" duplicate files
- [ ] Delete 1 backup file
- [ ] Clean 2,676 `__pycache__` directories
- [ ] Update .gitignore

#### Verification
- [ ] Run `pytest backend/tests/ -v` → All pass
- [ ] Run `python -m backend.api.main` → No errors
- [ ] Run `git status` → See deletions only
- [ ] Check root directory → Only essential files remain

#### Commit
- [ ] Stage changes: `git add .`
- [ ] Commit with message (see Step 3 above)
- [ ] Push to remote: `git push`

**Status:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

### Phase 2: Documentation Organization (Optional - 30 min)

#### Pre-Flight Checks
- [ ] Phase 1 complete and tested
- [ ] Create backup branch: `git checkout -b cleanup/phase2`

#### Execution
- [ ] Create doc subdirectories
```bash
mkdir -p docs/{analysis,implementation,architecture,guides,testing,verification}
```

- [ ] Move files to subdirectories
```bash
mv ANALYSIS_RESULTS.md docs/analysis/
mv IMPLEMENTATION_COMPLETE.md docs/implementation/
mv LINE_BY_LINE_MAPPING.md docs/architecture/
mv PHASE3_TYPE_FIX_GUIDE.md docs/guides/
mv PYTEST_ASYNCIO_FIX_SUMMARY.md docs/testing/
mv REFACTOR_VERIFICATION.md docs/verification/
mv TEST_REFACTOR_SUMMARY.md docs/testing/
mv TYPE_CONSISTENCY_ANALYSIS.md docs/architecture/
mv TYPE_CONSISTENCY_IMPLEMENTATION.md docs/implementation/
mv VALIDATION_DELIVERABLES.md docs/verification/
```

- [ ] Update any broken links in documentation

#### Verification
- [ ] All moved files accessible in new locations
- [ ] Run `find . -maxdepth 1 -name "*.md"` → Only essentials
- [ ] Documentation builds successfully
- [ ] Links work correctly

#### Commit
- [ ] Stage: `git add .`
- [ ] Commit: `git commit -m "docs: Organize root documentation into subdirectories"`
- [ ] Merge to main: `git checkout main && git merge cleanup/phase2`

**Status:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

### Phase 3: .claude Consolidation (Optional - 1 hour)

#### Pre-Flight Checks
- [ ] Phase 1 & 2 complete
- [ ] Create backup branch: `git checkout -b cleanup/phase3`

#### Memory System Consolidation (8 → 2 files)
- [ ] Review all memory system docs
- [ ] Create consolidated MEMORY_SYSTEM_GUIDE.md
- [ ] Update MEMORY_SYSTEM_STATUS.md with latest
- [ ] Delete redundant files:
```bash
cd .claude
rm -f COMPLETE_MEMORY_SYSTEM_SUMMARY.md \
      HIVE_MIND_MEMORY_STATUS.md \
      V3_MEMORY_CONSOLIDATION_INDEX.md
```

#### Integration Docs Consolidation (5 → 1 file)
- [ ] Keep latest: INTEGRATION_STATUS.md
- [ ] Merge information from others
- [ ] Delete redundant:
```bash
rm -f COMPREHENSIVE_INTEGRATION_COMPLETE.md \
      COMPREHENSIVE_SYNC_COMPLETE.md \
      DOCUMENTATION_SYNC_COMPLETE.md \
      INTEGRATION_SUCCESS_SUMMARY.txt
```

#### Phase Reports Consolidation (7 → 1 file)
- [ ] Create IMPLEMENTATION_PHASES.md
- [ ] Combine all phase completion reports
- [ ] Delete originals:
```bash
rm -f PHASE1_EXECUTION_SUMMARY.md \
      PHASE1_IMPLEMENTATION_COMPLETE.md \
      PHASE5_COMPLETION_REPORT.txt \
      PHASE5_SUMMARY.md \
      PRIORITY_1_COMPLETE.md \
      PRIORITY_2_COMPLETE.md \
      SUCCESS_REPORT.md
```

#### Verification
- [ ] All information preserved in consolidated files
- [ ] .claude directory reduced by ~10 MB
- [ ] Documentation is clearer

#### Commit
- [ ] Stage: `git add .`
- [ ] Commit: `git commit -m "docs: Consolidate .claude documentation"`
- [ ] Merge to main

**Status:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

### Phase 4: Dependency Audit (Optional - 2 hours)

⚠️ **REQUIRES THOROUGH TESTING** ⚠️

#### Pre-Flight Checks
- [ ] All previous phases complete
- [ ] Create backup branch: `git checkout -b cleanup/dependencies`
- [ ] Full test suite passing
- [ ] Staging environment available for testing

#### For Each Suspected Unused Package
1. [ ] Search for imports:
```bash
grep -r "import <package>" backend/
grep -r "from <package>" backend/
```

2. [ ] If no results, remove from requirements.txt
3. [ ] Reinstall dependencies:
```bash
pip install -r requirements.txt
```

4. [ ] Run full test suite:
```bash
pytest backend/tests/ -v --cov
```

5. [ ] Test application manually
6. [ ] Deploy to staging and verify

#### Packages to Verify (One at a Time)
- [ ] lime==0.2.0.1
- [ ] pybreaker>=1.2.0
- [ ] geoip2>=4.8.0
- [ ] user-agents>=2.2.0
- [ ] html5lib>=1.1
- [ ] schedule==1.2.2

#### Verification
- [ ] All tests pass for each removal
- [ ] Application starts successfully
- [ ] No import errors in logs
- [ ] Staging environment works correctly
- [ ] Production deployment plan ready

#### Commit (After Each Successful Removal)
- [ ] Stage: `git add requirements.txt`
- [ ] Commit: `git commit -m "deps: Remove unused package <name>"`

**Status:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

## 🚨 Rollback Procedures

### If Phase 1 Breaks Something

```bash
# Restore all deletions
git restore .

# Or restore specific file
git restore <filename>

# Rebuild Python cache (automatic on next run)
python -m compileall backend/
```

### If Phase 2-3 Break Something

```bash
# Return to main branch
git checkout main

# Delete cleanup branch
git branch -D cleanup/phase2  # or phase3
```

### If Phase 4 Breaks Something

```bash
# Revert last commit
git revert HEAD

# Or restore requirements.txt
git restore requirements.txt

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Progress Tracking

| Phase | Files | Time | Risk | Status |
|-------|-------|------|------|--------|
| Phase 1 | 43 files + 2,676 dirs | 5 min | ZERO | ⬜ |
| Phase 2 | 10 files (move) | 30 min | LOW | ⬜ |
| Phase 3 | ~20 files (consolidate) | 1 hour | LOW | ⬜ |
| Phase 4 | 5-10 packages | 2 hours | MEDIUM | ⬜ |

**Legend:** ⬜ Not Started | 🟡 In Progress | ✅ Complete | ❌ Failed

---

## 🎯 Success Criteria

### Phase 1 Success
- ✅ ~150 MB freed
- ✅ All tests passing
- ✅ No import errors
- ✅ Application starts successfully
- ✅ Git history preserved

### Phase 2 Success
- ✅ Root directory has <10 .md files
- ✅ Documentation organized in subdirectories
- ✅ All links work correctly
- ✅ Tests still passing

### Phase 3 Success
- ✅ .claude reduced by ~10 MB
- ✅ Duplicate information consolidated
- ✅ Documentation easier to navigate
- ✅ All information preserved

### Phase 4 Success
- ✅ 50-100 MB additional savings
- ✅ Only necessary dependencies remain
- ✅ All tests passing
- ✅ Staging deployment successful
- ✅ No production issues

---

## 📝 Notes & Observations

**Add notes during execution:**

### Phase 1 Notes
- Date started: __________
- Issues encountered: __________
- Resolution: __________
- Completion time: __________

### Phase 2 Notes
- Date started: __________
- Issues encountered: __________
- Resolution: __________
- Completion time: __________

### Phase 3 Notes
- Date started: __________
- Issues encountered: __________
- Resolution: __________
- Completion time: __________

### Phase 4 Notes
- Date started: __________
- Packages removed: __________
- Issues encountered: __________
- Resolution: __________
- Completion time: __________

---

## 🔗 Related Documentation

- **Full Audit Report:** `docs/DELETION_LOG.md`
- **Executive Summary:** `docs/REFACTOR_AUDIT_SUMMARY.md`
- **This Checklist:** `docs/CLEANUP_CHECKLIST.md`

---

**Last Updated:** 2026-02-08
**Next Review:** 2026-03-08 (1 month)
