# Verification System Architecture

> **SUPERSEDED (2026-06): historical snapshot, not current truth. See [docs/STATUS.md](../../STATUS.md).**


**Visual Reference for the Documentation Verification System**

---

## System Overview

```
VERIFICATION SYSTEM (Production-Ready)
│
├─ INPUT: Project Repository
│  └─ Markdown Files (47 total)
│  └─ Documentation Files
│  └─ Root Directory Files
│
├─ PROCESSING: 5 Verification Scripts
│  ├─ verify-duplicate-files.sh     → Scan + Remove
│  ├─ verify-links.sh               → Validate + Report
│  ├─ verify-version-tags.sh        → Check + Update
│  ├─ verify-doc-count.sh           → Count + Compare
│  └─ verify-root-cleanup.sh        → Organize + Archive
│
├─ OUTPUT: Reports & Logs
│  └─ .claude/verify/
│     ├─ *-report.txt               (Summaries)
│     ├─ *-baseline.json            (Metrics)
│     ├─ *-allowed-files.txt        (Whitelists)
│     ├─ *.log                      (Detailed logs)
│     └─ archived-root-files/       (Moved items)
│
└─ INTEGRATION: Multiple Paths
   ├─ Pre-commit Hook (automatic on commit)
   ├─ CI/CD Pipeline (GitHub Actions, etc.)
   ├─ Scheduled Jobs (cron, etc.)
   └─ Manual Execution (on-demand)
```

---

## Script Execution Flow

### verify-duplicate-files.sh

```
START
  ↓
Setup Directories
  ↓
Parse Arguments (--fix, --report, --verbose)
  ↓
Find " 2" Suffixed Files
  ├─ Exclude: node_modules, dist, build, .git
  ├─ Scope: 3 levels deep from root
  └─ Collect: File list, sizes, duplicates
  ↓
Calculate Metrics
  ├─ Total duplicates
  ├─ Total wasted space
  └─ Individual file sizes
  ↓
[IF --REPORT] Generate Report
  ├─ Write: duplicate-files-report.txt
  └─ Log: duplicate-files.log
  ↓
[IF --FIX] Remove Duplicates
  ├─ Remove: Each " 2" file
  ├─ Verify: Originals still exist
  └─ Confirm: All removed
  ↓
Verify No Duplicates Remain
  ↓
Output Results
  ↓
EXIT (0=success, 1=failure)
```

### verify-links.sh

```
START
  ↓
Setup Directories
  ↓
Parse Arguments (--external, --report, --verbose)
  ↓
Find All Markdown Files
  ├─ Exclude: node_modules, .git, dist
  └─ Collect: File list
  ↓
FOR EACH Markdown File
  ├─ Extract Links: [text](url)
  ├─ Extract URLs: http(s)://...
  └─ Process Each Link
      ↓
      Validate Link
      ├─ Skip: External URLs (unless --external)
      ├─ Resolve: Relative/absolute paths
      ├─ Check: File exists
      └─ Validate: Anchors if present
      ↓
      Track: Valid/Broken/External
  ↓
[IF --REPORT] Generate Report
  ├─ Write: links-validation-report.txt
  └─ Log: links-validation.log
  ↓
Output Results
  ├─ Total links scanned
  ├─ Valid links count
  ├─ Broken links count
  └─ External URLs count
  ↓
EXIT (0=all valid, 1=broken found)
```

### verify-version-tags.sh

```
START
  ↓
Setup Directories
  ↓
Parse Arguments (--fix, --update VERSION, --strict)
  ↓
Scan Documentation Files
  ├─ Find: All .md files in /docs and /.claude
  ├─ Extract: Version tags (X.Y.Z format)
  ├─ Extract: Last Updated dates
  └─ Validate: Semantic versioning
  ↓
Check Consistency
  ├─ Compare: All versions found
  ├─ Detect: Mismatches
  └─ Validate: Semver format
  ↓
Check Dates
  ├─ Detect: Future dates (error)
  └─ Track: Last updated values
  ↓
[IF --FIX] Update Versions
  ├─ Replace: Old version with --update VALUE
  ├─ Update: Last Modified date to today
  └─ Preserve: Backup files (.bak)
  ↓
[IF --REPORT] Generate Report
  ├─ Write: version-tags-report.txt
  └─ Log: version-tags.log
  ↓
Output Results
  ├─ Files checked
  ├─ Consistent tags
  ├─ Inconsistent tags
  └─ Missing tags
  ↓
EXIT (0=consistent, 1=mismatch or --strict violated)
```

### verify-doc-count.sh

```
START
  ↓
Setup Directories
  ↓
Parse Arguments (--compare, --threshold N, --report)
  ↓
Count Documentation Files
  ├─ Find: All .md files (excluding node_modules, .git, dist)
  ├─ Categorize: By location
  │  ├─ .claude/ → claude-internal
  │  ├─ /docs/ → documentation
  │  ├─ /backend/ → backend-docs
  │  ├─ /frontend/ → frontend-docs
  │  └─ root → root-docs
  ├─ Count: Per category
  ├─ Calculate: Total size
  └─ Analyze: Size distribution
  ↓
[IF --COMPARE] Compare Baseline
  ├─ Load: doc-count-baseline.json
  ├─ Compare: Current vs baseline
  ├─ Calculate: Changes (delta %)
  ├─ Check: Threshold (default 10%)
  └─ Alert: If exceeded
  ↓
Save/Update Baseline
  ├─ JSON format with timestamp
  └─ Write: doc-count-baseline.json
  ↓
[IF --REPORT] Generate Report
  ├─ Write: doc-count-report.txt
  └─ Log: doc-count.log
  ↓
Output Results
  ├─ Total files
  ├─ Total size
  ├─ Category breakdown
  └─ Changes from baseline
  ↓
EXIT (0=success or threshold ok, 1=threshold exceeded in --strict)
```

### verify-root-cleanup.sh

```
START
  ↓
Setup Directories
  ↓
Parse Arguments (--clean, --ignore PATTERNS, --strict)
  ↓
Load Allowed Files List
  ├─ Essential files (README, package.json, etc.)
  ├─ Custom ignore patterns (from --ignore flag)
  └─ Build allowed/suspicious/temporary categories
  ↓
Scan Root Directory
  ├─ Find: All files in root (maxdepth 1)
  ├─ For Each File, Classify:
  │  ├─ Check: Is it temporary? (*.bak, * 2.*, etc.)
  │  ├─ Check: Is it duplicate? (* 2.*)
  │  ├─ Check: Is it allowed? (in whitelist)
  │  └─ Otherwise: Mark as suspicious
  │
  ├─ Track: Duplicates, Temp, Suspicious
  └─ Count: Each category
  ↓
Verify Required Directories
  ├─ Check: backend/ exists
  ├─ Check: frontend/ exists
  ├─ Check: scripts/ exists
  ├─ Check: docs/ exists
  ├─ Check: .claude/ exists
  └─ Check: data/ exists
  ↓
Check Organization
  ├─ Count: .md files in root (warn if >5)
  ├─ Count: Test files in root (warn if >0)
  └─ Suggest: Move to appropriate directories
  ↓
[IF --CLEAN] Perform Cleanup
  ├─ Remove: Duplicate " 2" files
  ├─ Remove: Temporary files (*.bak, *.tmp, etc.)
  ├─ Archive: Suspicious files to .claude/archived-root-files/
  └─ Verify: Changes successful
  ↓
[IF --REPORT] Generate Report
  ├─ Write: root-cleanup-report.txt
  ├─ Write: root-allowed-files.txt
  └─ Log: root-cleanup.log
  ↓
Output Results
  ├─ Total files in root
  ├─ Allowed files count
  ├─ Duplicate files count
  ├─ Temporary files count
  └─ Suspicious files count
  ↓
EXIT (0=success, 1=issues in --strict mode)
```

---

## Data Flow Diagram

```
Source Files
     │
     ├─ Markdown Files (.md)
     │  ├─ /docs/*.md
     │  ├─ /.claude/*.md
     │  ├─ /backend/**/*.md
     │  ├─ /frontend/**/*.md
     │  └─ root/*.md
     │
     └─ Directory Structure
        ├─ Root files
        ├─ Directories
        └─ Size/date metadata

        │
        ├─────────────────────────────────┬────────────────────────────────┬──────────────────────────────┐
        │                                 │                                │                              │
        ↓                                 ↓                                ↓                              ↓

Script 1: Duplicates          Script 2: Links              Script 3: Versions           Script 4: Count
├─ Find " 2" files           ├─ Extract markdown links    ├─ Extract version tags      ├─ Count files
├─ Calculate sizes           ├─ Validate file refs        ├─ Check dates              ├─ Calculate totals
└─ Generate report           ├─ Check anchors             ├─ Validate semver          ├─ Compare baseline
                             └─ Generate report          └─ Generate report          └─ Generate report


                                                          ↓                              ↓

                                                   Script 5: Root Cleanup
                                                   ├─ Scan root directory
                                                   ├─ Classify files
                                                   ├─ Verify directories
                                                   └─ Generate report

                        │
                        └──────────────────────┬──────────────────────┐
                                               │                      │
                                               ↓                      ↓
                                        Reports (.txt)        Baselines (.json)
                                        ├─ Summaries         ├─ Metrics reference
                                        ├─ Details           └─ For comparison
                                        └─ Recommendations
                                               │
                                               ↓
                                        .claude/verify/
                                        ├─ All reports
                                        ├─ All logs
                                        └─ Archived files
```

---

## Integration Points

### Pre-Commit Hook
```
Git Commit
    ↓
Hook Triggered
    ↓
Run: verify-duplicate-files.sh --strict
Run: verify-links.sh --strict
Run: verify-root-cleanup.sh --strict
    ↓
[PASS] → Commit allowed
[FAIL] → Commit rejected
```

### CI/CD Pipeline
```
GitHub Push
    ↓
GitHub Actions Triggered
    ↓
Checkout Code
    ↓
Run Verification Scripts
├─ verify-duplicate-files.sh --strict
├─ verify-links.sh --strict
├─ verify-version-tags.sh --strict
└─ verify-root-cleanup.sh --strict
    ↓
[PASS] → Tests continue
[FAIL] → Build fails
```

### Scheduled Jobs
```
Cron/Scheduler
    ↓
Daily/Weekly Trigger
    ↓
Run All Scripts with --report
    ↓
Generate Reports
    ↓
Email/Log Results
    ↓
Archive Old Reports (optional)
```

---

## Error Handling Flow

```
Script Execution
    ↓
[Error Detected]
    ├─ Log: Timestamp + level + message
    ├─ Console: Display error
    └─ Set: Exit code = 1
    ↓
[Strict Mode Active]
    ├─ EXIT with 1 (failure)
    └─ Stop processing
    ↓
[Report Mode Active]
    ├─ Generate report with issues
    └─ Highlight problems
    ↓
[Fix Mode Active]
    ├─ Attempt remediation
    ├─ Log each action
    └─ Verify results
```

---

## File Organization

```
Project Root
│
├── scripts/                 (5 executable scripts)
│   ├── verify-duplicate-files.sh
│   ├── verify-links.sh
│   ├── verify-version-tags.sh
│   ├── verify-doc-count.sh
│   └── verify-root-cleanup.sh
│
├── docs/                    (4 documentation guides)
│   ├── README_VERIFICATION.md
│   ├── VERIFICATION_QUICK_START.md
│   ├── VERIFICATION_SCRIPTS_GUIDE.md
│   ├── VERIFICATION_IMPLEMENTATION_SUMMARY.md
│   └── VERIFICATION_SYSTEM_ARCHITECTURE.md (this file)
│
├── .claude/verify/          (Auto-generated reports)
│   ├── duplicate-files-report.txt
│   ├── duplicate-files.log
│   ├── links-validation-report.txt
│   ├── links-validation.log
│   ├── version-tags-report.txt
│   ├── version-tags.log
│   ├── doc-count-report.txt
│   ├── doc-count-baseline.json
│   ├── doc-count.log
│   ├── root-cleanup-report.txt
│   ├── root-allowed-files.txt
│   ├── root-cleanup.log
│   └── archived-root-files/
│
└── Summary Docs
    ├── VERIFICATION_DELIVERY_SUMMARY.md
    ├── VERIFICATION_FILE_INDEX.md
    └── (this file)
```

---

## Script Dependencies

```
All Scripts
    ├─ bash (version 4+)
    ├─ Standard utilities: find, grep, sed, awk
    ├─ File operations: stat, mkdir, touch
    └─ None on each other (independent)

verify-duplicate-files.sh
    └─ No dependencies on other scripts

verify-links.sh
    └─ No dependencies on other scripts

verify-version-tags.sh
    └─ No dependencies on other scripts

verify-doc-count.sh
    └─ Creates: doc-count-baseline.json
    └─ Reads: doc-count-baseline.json (if exists)

verify-root-cleanup.sh
    └─ No dependencies on other scripts
```

---

## Performance Characteristics

```
Sequential Execution
├─ Script 1 (duplicates): <1s
├─ Script 2 (links): 2-5s
├─ Script 3 (versions): <1s
├─ Script 4 (count): <2s
└─ Script 5 (root): <1s
└─ Total: 6-10s

Parallel Execution (if run as background jobs)
├─ All scripts: 5s (bottleneck: links validation)

With External Link Checking
└─ Add: 30-60s (depends on internet speed)
```

---

## Quality Gates

```
BEFORE COMMIT (pre-commit hook)
├─ verify-duplicate-files.sh --strict
│  └─ MUST pass (0 duplicates)
├─ verify-links.sh --strict
│  └─ MUST pass (0 broken links)
└─ verify-root-cleanup.sh --strict
   └─ MUST pass (clean root)

BEFORE RELEASE (pre-release check)
├─ All above MUST pass
├─ verify-version-tags.sh --strict
│  └─ MUST pass (consistent versions)
├─ verify-doc-count.sh --compare --threshold 5
│  └─ MUST pass (metrics within 5%)
└─ Full --external link check
   └─ All external links must be valid

DAILY (background monitoring)
├─ All scripts with --report
└─ Reports saved for review
```

---

## Documentation Hierarchy

```
START
  ↓
README_VERIFICATION.md
(Entry point - 2 KB, ~250 lines)
  │
  ├─→ VERIFICATION_QUICK_START.md
  │   (Common commands - 12 KB, ~400 lines)
  │
  ├─→ VERIFICATION_SCRIPTS_GUIDE.md
  │   (Full reference - 50 KB, 3000+ lines)
  │
  ├─→ VERIFICATION_IMPLEMENTATION_SUMMARY.md
  │   (Technical details - 30 KB, 1500+ lines)
  │
  └─→ VERIFICATION_SYSTEM_ARCHITECTURE.md
      (This file - Visual reference)

Supporting Docs:
  ├─ VERIFICATION_DELIVERY_SUMMARY.md (Checklist)
  └─ VERIFICATION_FILE_INDEX.md (File manifest)
```

---

## Time Estimates

| Task | Duration | Complexity |
|------|----------|-----------|
| Single script run | 1-5s | Low |
| All scripts run | 6-10s | Low |
| With reports | +5s | Low |
| Full suite with cleanup | 15-20s | Medium |
| Pre-commit hook integration | 5 min | Low |
| CI/CD integration | 10-15 min | Low |
| Learning system | 30 min | Medium |
| Full understanding | 1-2 hours | High |

---

## Success Metrics

```
Documentation Quality
├─ Duplicate files: 0
├─ Broken links: 0
├─ Version consistency: 100%
├─ Documentation count: >20
└─ Root directory: <15 files

System Health
├─ Script execution: <10s (all)
├─ Report generation: Automatic
├─ Error handling: Comprehensive
├─ Logging: Complete
└─ Integration: Multiple paths
```

---

**Architecture Version**: 3.0.0
**Last Updated**: 2026-01-29
**Visual Reference**: Complete
