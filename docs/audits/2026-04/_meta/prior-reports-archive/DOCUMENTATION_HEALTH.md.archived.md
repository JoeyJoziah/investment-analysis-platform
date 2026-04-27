> **ARCHIVED 2026-04-27 by 18-docs-health**
> **Original**: docs/DOCUMENTATION_HEALTH.md
> **Validation summary**: 7/8 claims validated; status = partially_stale
> **Key findings**: CI/CD health check workflow (.github/workflows/doc-health.yml) does not exist in repo despite being specified in §2.3. Scripts exist but untested. Metrics in §1.2 are example/template values, not real measurements. See `docs/audits/2026-04/reports/18-docs-health.md` §2 for per-claim validation table.
>
> **Action required (post-archival):**
> - (F-18-004) Create `.github/workflows/doc-health.yml` based on spec in this document §2.3
> - (F-18-005) Implement real metrics collection pipeline; mark §1.2 metrics as "example" if not replaced
> - (F-18-006) Test doc validation scripts (check-doc-health.sh, validate_doc_health.py) for bitrot
> - (F-18-007) Update linting rules to reference markdownlint instead of outdated tool references

---

# Documentation Health Dashboard

**Version**: 1.0.0
**Last Updated**: 2026-01-29
**Status**: Active Monitoring
**Health Score**: Configurable

---

## Executive Summary

This document provides a comprehensive framework for tracking, measuring, and maintaining documentation quality across the Investment Analysis Platform. It includes automated health checks, metrics tracking, review schedules, and monitoring tools.

**Key Metrics**:
- Documentation Coverage: Track % of codebase with documentation
- Update Recency: Monitor documentation age and staleness
- Completeness: Measure sections with required content
- Accuracy: Validate documentation against code
- Accessibility: Track usability and clarity metrics

---

## 1. Metrics Tracking System

### 1.1 Core Health Metrics

#### Coverage Metrics
```yaml
Metric: Documentation_Coverage_Percent
Type: Percentage (0-100%)
Target: >= 95%
Measurement Unit: File Count
Collection Frequency: Daily
Priority: Critical

Definition: |
  Total documented modules / Total modules in codebase
  Includes: Code files with JSDoc, READMEs, guides
  Excludes: Test files, temporary files, dependencies

Current Status:
  - Backend Modules: 92%
  - Frontend Components: 87%
  - API Endpoints: 100%
  - Configuration: 88%
  - Overall: 91%
```

#### Recency Metrics
```yaml
Metric: Documentation_Age_Days
Type: Integer (days)
Target: <= 30 days
Measurement Unit: Days since last update
Collection Frequency: Daily
Priority: High

Definition: |
  Average age of documentation since last update
  Measured per document and across categories

Current Status:
  - Code Documentation: 5 days
  - API Documentation: 12 days
  - Deployment Guides: 28 days
  - Architecture Docs: 45 days (⚠️ Review needed)
  - Change Logs: 3 days
  - Average: 18.6 days
```

#### Completeness Metrics
```yaml
Metric: Documentation_Completeness_Score
Type: Percentage (0-100%)
Target: >= 90%
Measurement Unit: Sections with required content
Collection Frequency: Weekly
Priority: High

Definition: |
  Ratio of complete sections to required sections
  Measures: Purpose, Setup, Usage, Examples, Troubleshooting

Scoring:
  100% - All required sections present and populated
   90% - 1-2 minor sections missing
   75% - Multiple sections missing or incomplete
   50% - Major gaps in documentation
    0% - No documentation

Current Status:
  - Installation Guide: 95%
  - API Reference: 92%
  - Troubleshooting: 78% (⚠️ Review needed)
  - Architecture: 88%
  - Deployment: 93%
  - Average: 89.2%
```

#### Accuracy Metrics
```yaml
Metric: Documentation_Accuracy_Score
Type: Percentage (0-100%)
Target: >= 98%
Measurement Unit: Statements verified against code
Collection Frequency: Monthly
Priority: Critical

Definition: |
  Ratio of accurate documentation to total verifiable claims
  Includes: API signatures, commands, configurations, file paths

Scoring:
  100% - All claims verified and accurate
   98% - Minor inaccuracies (1-2 items)
   95% - Several outdated statements
   80% - Significant discrepancies
    0% - Severely outdated

Current Status:
  - API Endpoints: 100%
  - Installation Steps: 97%
  - Configuration Options: 95%
  - File Paths: 99%
  - Command Examples: 96%
  - Average: 97.4%
```

#### Accessibility Metrics
```yaml
Metric: Documentation_Accessibility_Score
Type: Percentage (0-100%)
Target: >= 90%
Measurement Unit: User satisfaction and readability
Collection Frequency: Quarterly
Priority: Medium

Definition: |
  Composite score measuring usability and clarity
  Factors: Organization, Search ability, Cross-linking, Format, Examples

Scoring Factors:
  - Clear Structure: 20%
  - Cross-linking: 15%
  - Code Examples: 20%
  - Search capability: 15%
  - Readability (Flesch): 15%
  - Visual Aids: 15%

Current Status:
  - Readability (Flesch Score): 65/100
  - Cross-linking Density: 3.2 links per doc
  - Code Example Coverage: 78%
  - Search Functionality: Available
  - Average: 85.3%
```

[Rest of document truncated for archive — see original docs/DOCUMENTATION_HEALTH.md for full content]

---

**Archive Note**: This document was archived because its specified CI/CD integrations (§2.3) do not currently exist in the codebase. However, the framework and specifications remain valuable reference material. The document should be reviewed and updated in conjunction with implementation of F-18-004 (Missing CI/CD health check workflow).
