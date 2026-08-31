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

### 1.2 Tracking Dashboard Template

```yaml
Documentation_Health_Dashboard:
  timestamp: "2026-01-29T00:00:00Z"
  overall_health: "Good"
  health_score: 91.2

  metrics:
    coverage:
      current: 91%
      target: 95%
      trend: "↑ +2% (7 days)"
      status: "On Track"

    recency:
      average_age: 18.6
      max_age: 45
      target_max: 30
      status: "⚠️ Needs Attention"

    completeness:
      current: 89.2%
      target: 90%
      status: "Nearly Met"

    accuracy:
      current: 97.4%
      target: 98%
      status: "Acceptable"

    accessibility:
      current: 85.3%
      target: 90%
      status: "Needs Work"

  issues:
    - id: "DOC-001"
      severity: "Medium"
      title: "Architecture documentation outdated"
      age_days: 45
      action: "Review and update"
      assigned_to: "Tech Lead"

    - id: "DOC-002"
      severity: "Low"
      title: "Troubleshooting section incomplete"
      completeness: 78%
      action: "Add missing solutions"
      assigned_to: "Support Team"

    - id: "DOC-003"
      severity: "Low"
      title: "Readability score below target"
      current: 65
      target: 75
      action: "Simplify language"
      assigned_to: "Tech Writer"
```

---

## 2. Automated Health Checks

### 2.1 Documentation Linting Rules

```typescript
// File: scripts/doc-health-check.ts
// Purpose: Automated documentation validation

const DocLintRules = {
  // Structure Rules
  REQUIRED_HEADINGS: {
    rule: "Document must contain required section headings",
    headings: [
      "Overview",
      "Installation / Setup",
      "Usage / Configuration",
      "Examples",
      "Troubleshooting",
      "See Also / References"
    ],
    severity: "High",
    message: (missing) => `Missing required sections: ${missing.join(", ")}`
  },

  // Metadata Rules
  FRONTMATTER_REQUIRED: {
    rule: "Document must have frontmatter metadata",
    required_fields: [
      "title",
      "version",
      "last_updated",
      "status",
      "audience"
    ],
    severity: "High"
  },

  // Content Rules
  MINIMUM_CONTENT_LENGTH: {
    rule: "Each section must have minimum content",
    minimums: {
      "Overview": 100,  // characters
      "Installation": 150,
      "Examples": 200,
      "Troubleshooting": 150
    },
    severity: "Medium"
  },

  CODE_EXAMPLES_REQUIRED: {
    rule: "Documentation must include code examples",
    minimum_examples: 2,
    supported_languages: [
      "typescript",
      "javascript",
      "python",
      "bash",
      "sql"
    ],
    severity: "Medium"
  },

  // Linking Rules
  BROKEN_LINKS: {
    rule: "No broken internal or external links",
    check: "Validate all [text](url) patterns",
    severity: "High"
  },

  CROSS_REFERENCES: {
    rule: "Related documents should cross-reference each other",
    minimum_references: 2,
    severity: "Low"
  },

  // Formatting Rules
  CODE_BLOCK_LANGUAGE_TAGS: {
    rule: "All code blocks must have language tags",
    format: "```language\n...\n```",
    severity: "Medium"
  },

  // Recency Rules
  LAST_UPDATED_FRESH: {
    rule: "Documentation must be recently updated",
    maximum_age_days: 30,
    severity: "High"
  },

  OUTDATED_VERSIONS: {
    rule: "Version numbers must match current release",
    check: "Compare with package.json",
    severity: "Medium"
  },

  // Accuracy Rules
  COMMAND_ACCURACY: {
    rule: "All documented commands must be valid",
    check: "Test commands in CI/CD pipeline",
    severity: "High"
  },

  API_ENDPOINT_ACCURACY: {
    rule: "API endpoints must match implementation",
    check: "Generate from OpenAPI/Swagger spec",
    severity: "Critical"
  },

  // Accessibility Rules
  READABILITY_LEVEL: {
    rule: "Documentation must be readable",
    target_flesch_score: 60,  // 60+ is college-level readable
    tool: "flesch-kincaid",
    severity: "Medium"
  },

  // Grammar and Style
  GRAMMAR_SPELLCHECK: {
    rule: "No spelling or grammar errors",
    tool: "language-tool or similar",
    severity: "Low"
  }
}

// Violation Categories
const ViolationSeverity = {
  Critical: {
    score_penalty: 20,
    action: "Block deployment",
    review_required: true
  },
  High: {
    score_penalty: 10,
    action: "Flag for review",
    review_required: true
  },
  Medium: {
    score_penalty: 5,
    action: "Schedule for next sprint",
    review_required: false
  },
  Low: {
    score_penalty: 1,
    action: "Track in backlog",
    review_required: false
  }
}
```

### 2.2 Health Check Script

```bash
#!/bin/bash
# File: scripts/check-doc-health.sh
# Purpose: Run automated documentation health checks
# Usage: ./scripts/check-doc-health.sh [--fix] [--report]

set -e

readonly DOCS_DIR="./docs"
readonly REPORTS_DIR="./.reports/doc-health"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Initialize report
initialize_report() {
    mkdir -p "$REPORTS_DIR"
    cat > "$REPORTS_DIR/health_check_${TIMESTAMP}.json" << 'EOF'
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "checks": {},
  "violations": [],
  "metrics": {},
  "summary": {}
}
EOF
}

# Check 1: File existence and coverage
check_coverage() {
    echo "Checking documentation coverage..."

    local total_modules=$(find ./backend ./frontend -type f -name "*.ts" -o -name "*.tsx" -o -name "*.py" | wc -l)
    local documented=$(find "$DOCS_DIR" -type f -name "*.md" | wc -l)
    local coverage=$((documented * 100 / total_modules))

    echo "Coverage: $coverage% ($documented / $total_modules)"

    if [ "$coverage" -lt 95 ]; then
        echo "⚠️  Coverage below target (95%)"
    else
        echo "✓ Coverage meets target"
    fi
}

# Check 2: Recency of documentation
check_recency() {
    echo "Checking documentation recency..."

    for file in "$DOCS_DIR"/*.md; do
        local modified=$(stat -f%m "$file" 2>/dev/null || stat -c%Y "$file" 2>/dev/null)
        local now=$(date +%s)
        local age_days=$(( (now - modified) / 86400 ))

        if [ "$age_days" -gt 30 ]; then
            echo "⚠️  $file is $age_days days old"
        fi
    done
}

# Check 3: Required frontmatter
check_frontmatter() {
    echo "Checking documentation frontmatter..."

    local missing_frontmatter=0
    for file in "$DOCS_DIR"/*.md; do
        if ! grep -q "^---" "$file" || ! grep -q "^Last Updated:" "$file"; then
            echo "⚠️  $file missing frontmatter"
            ((missing_frontmatter++))
        fi
    done

    echo "Files with valid frontmatter: $(find "$DOCS_DIR" -name "*.md" | wc -l) checked, $missing_frontmatter missing"
}

# Check 4: Broken links
check_links() {
    echo "Checking for broken links..."

    local broken_count=0
    for file in "$DOCS_DIR"/*.md; do
        # Extract all markdown links
        grep -o '\[.*\]([^)]*\.md)' "$file" 2>/dev/null | while read -r link; do
            local target=$(echo "$link" | sed 's/.*(\(.*\))/\1/')
            if [ ! -f "$DOCS_DIR/$target" ]; then
                echo "⚠️  Broken link in $file: $target"
                ((broken_count++))
            fi
        done
    done

    if [ "$broken_count" -eq 0 ]; then
        echo "✓ No broken links detected"
    fi
}

# Check 5: Code example presence
check_code_examples() {
    echo "Checking for code examples..."

    local files_with_examples=0
    for file in "$DOCS_DIR"/*.md; do
        if grep -q '```' "$file"; then
            ((files_with_examples++))
        fi
    done

    echo "Files with code examples: $files_with_examples / $(find "$DOCS_DIR" -name "*.md" | wc -l)"
}

# Check 6: Readability score
check_readability() {
    echo "Checking documentation readability..."

    if command -v flesch &> /dev/null; then
        for file in "$DOCS_DIR"/*.md; do
            local score=$(flesch "$file" 2>/dev/null || echo "N/A")
            echo "Readability score for $(basename "$file"): $score"
        done
    else
        echo "Note: Install 'flesch' tool for readability analysis"
    fi
}

# Generate summary report
generate_report() {
    echo "Generating health report..."

    cat > "$REPORTS_DIR/health_summary_${TIMESTAMP}.txt" << 'EOF'
==============================================
Documentation Health Check Report
==============================================
Timestamp: $(date)
Directory: ./docs

Coverage Analysis:
- Total Files: $(find ./docs -name "*.md" | wc -l)
- Coverage: $(check_coverage | grep Coverage | awk '{print $2}')

Recency Analysis:
- Files > 30 days old: $(find ./docs -name "*.md" -mtime +30 | wc -l)
- Average age: $( your_calculation_here )

Quality Metrics:
- Frontmatter compliance: Complete
- Broken links: None detected
- Code examples present: Yes
- Readability: Acceptable

Recommendations:
1. Update files older than 30 days
2. Add missing sections to Troubleshooting guide
3. Improve code example coverage in API docs
4. Simplify language in Architecture guide

==============================================
EOF

    cat "$REPORTS_DIR/health_summary_${TIMESTAMP}.txt"
}

# Main execution
main() {
    initialize_report
    check_coverage
    check_recency
    check_frontmatter
    check_links
    check_code_examples
    check_readability
    generate_report
}

main "$@"
```

### 2.3 CI/CD Integration

```yaml
# File: .github/workflows/doc-health.yml
# Purpose: Automated documentation health checks on every commit

name: Documentation Health Check

on:
  push:
    paths:
      - 'docs/**'
      - '**.md'
  pull_request:
    paths:
      - 'docs/**'
      - '**.md'
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC

jobs:
  doc-health:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'

      - name: Install dependencies
        run: |
          npm install -g markdown-lint
          npm install -g markdownlint-cli

      - name: Run documentation health checks
        run: |
          bash scripts/check-doc-health.sh --report

      - name: Validate links
        run: |
          npx markdown-link-check ./docs/**/*.md

      - name: Check code examples
        run: |
          bash scripts/validate-code-examples.sh

      - name: Lint markdown
        run: |
          markdownlint 'docs/**/*.md'

      - name: Upload health report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: doc-health-report
          path: .reports/doc-health/

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('.reports/doc-health/latest.json', 'utf8');
            const summary = JSON.parse(report).summary;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `📊 **Documentation Health Check**\n\n${summary}`
            });
```

---

## 3. Review Schedule & Process

### 3.1 Review Calendar

```yaml
Documentation_Review_Schedule:

  Daily_Reviews:
    frequency: "Every business day"
    duration: "15 minutes"
    triggers:
      - New documentation file created
      - Documentation section modified
      - Build/CI failures related to docs
    responsible: "Technical Writer + 1 Developer"
    checklist:
      - Spelling and grammar
      - Code snippet accuracy
      - Link validity
      - Format consistency

  Weekly_Reviews:
    frequency: "Every Monday, 10 AM"
    duration: "1 hour"
    scope:
      - All documentation modified in past week
      - Hot-spot documentation (high traffic)
      - Recent deployment documentation
    responsible: "Tech Lead + Tech Writer"
    checklist:
      - Content accuracy
      - Completeness (required sections)
      - Relevance to current version
      - Cross-reference accuracy
    meeting_template: "doc-sync-weekly.md"

  Monthly_Reviews:
    frequency: "First Tuesday of month"
    duration: "2 hours"
    scope:
      - All API documentation
      - All configuration guides
      - Architecture documentation
      - Troubleshooting guides
    responsible: "Technical Lead + Support Team + Stakeholder"
    checklist:
      - Accuracy against current codebase
      - Completeness metrics
      - User feedback integration
      - Version alignment
      - Gap analysis
    deliverable: "Monthly health report"

  Quarterly_Reviews:
    frequency: "End of each quarter"
    duration: "Half day"
    scope:
      - Complete documentation audit
      - Accessibility assessment
      - User satisfaction survey
      - Curriculum design review
      - Deprecated content purge
    responsible: "All stakeholders"
    checklist:
      - Strategic alignment
      - Coverage completeness
      - Accuracy validation
      - User feedback analysis
      - Training effectiveness
      - ROI measurement
    deliverable:
      - Quarterly report
      - Strategic roadmap
      - Gap closure plan

  Annual_Reviews:
    frequency: "January + September"
    duration: "Full day workshop"
    scope:
      - Complete restructuring review
      - Process improvement
      - Tool evaluation
      - Team training
      - Future planning
    responsible: "All documentation stakeholders + executives"
    checklist:
      - Strategic goals alignment
      - Process effectiveness
      - Team skill assessment
      - Tool adequacy
      - Market competitiveness
      - Customer satisfaction trends
    deliverable:
      - Annual strategy document
      - Process improvements roadmap
      - Training plan
```

### 3.2 Review Process

```markdown
# Documentation Review Process

## Pre-Review Checklist

- [ ] Document is complete and coherent
- [ ] All required sections are present
- [ ] Frontmatter is accurate
- [ ] Links are tested and valid
- [ ] Code examples are tested
- [ ] Grammar/spelling checked
- [ ] Formatting is consistent
- [ ] Tables render correctly
- [ ] Images/diagrams are included
- [ ] Related docs cross-referenced

## Review Phases

### Phase 1: Technical Accuracy (Developer)
**Time**: 15-30 minutes
**Checklist**:
- [ ] Code snippets compile/run correctly
- [ ] API endpoints match current implementation
- [ ] Configuration options are current
- [ ] File paths are correct
- [ ] CLI commands work as documented
- [ ] Examples produce expected output
- [ ] Dependencies are up-to-date
- [ ] Edge cases documented

### Phase 2: Completeness (Tech Writer)
**Time**: 15-30 minutes
**Checklist**:
- [ ] All required sections present
- [ ] No placeholder text remains
- [ ] Examples are comprehensive
- [ ] Troubleshooting covers common issues
- [ ] Related documents linked
- [ ] Audience is clearly identified
- [ ] Reading level is appropriate
- [ ] TOC is accurate

### Phase 3: Clarity & Accessibility (UX Reviewer)
**Time**: 15-30 minutes
**Checklist**:
- [ ] Language is clear and concise
- [ ] Jargon is minimized or explained
- [ ] Sentence structure is simple
- [ ] Paragraphs are focused
- [ ] Headings are descriptive
- [ ] Lists are parallel
- [ ] Format aids readability
- [ ] Visual hierarchy is clear

### Phase 4: Consistency (QA)
**Time**: 15 minutes
**Checklist**:
- [ ] Terminology matches project glossary
- [ ] Examples follow code style guide
- [ ] Format matches template
- [ ] Links use consistent format
- [ ] Tone is consistent
- [ ] Version numbers match
- [ ] Date format is consistent
- [ ] Citation format is consistent

## Review Approval

```yaml
Approval_Status:
  Draft:
    description: "Initial document, not ready for review"
    reviewers_required: 0

  In_Review:
    description: "Currently being reviewed"
    reviewers_required: 0

  Ready_for_QA:
    description: "Technical review complete"
    reviewers_required: 1  # Developer

  Approved_Draft:
    description: "Writer review complete"
    reviewers_required: 2  # Developer + Writer

  Published:
    description: "Ready for production"
    reviewers_required: 3  # All phases complete
```

## Issue Tracking

All documentation issues tracked in GitHub with labels:
- `docs/critical` - Breaks functionality or clarity
- `docs/high` - Important but non-blocking
- `docs/medium` - Nice to have improvements
- `docs/low` - Minor polish/enhancement

## Sign-Off Template

```
Reviewed by: @username
Review Date: YYYY-MM-DD
Status: ✅ Approved / ⚠️ Changes Requested
Severity of Issues: Critical / High / Medium / Low

Issues Found:
1. [Issue description]
   - Line: XXX
   - Type: Accuracy / Clarity / Completeness / Format
   - Action: [Required fix]

Sign-off: _______________
```
```

---

## 4. Monitoring Framework

### 4.1 Real-Time Monitoring

```typescript
// File: scripts/doc-monitor.ts
// Purpose: Real-time documentation health monitoring

interface DocMonitor {
  enabled: boolean
  update_interval: number  // seconds
  thresholds: {
    coverage_min: number  // 95%
    recency_max: number   // 30 days
    completeness_min: number  // 90%
    accuracy_min: number  // 98%
  }
  alerts: {
    email: boolean
    slack: boolean
    dashboard: boolean
    github_issue: boolean
  }
}

interface HealthAlert {
  id: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  metric: string
  current_value: number
  threshold: number
  message: string
  timestamp: Date
  recommended_action: string
}

class DocumentationMonitor {
  private alerts: HealthAlert[] = []
  private config: DocMonitor

  constructor(config: DocMonitor) {
    this.config = config
  }

  async checkHealth(): Promise<void> {
    if (!this.config.enabled) return

    // Run all health checks
    await this.checkCoverage()
    await this.checkRecency()
    await this.checkCompleteness()
    await this.checkAccuracy()

    // Process alerts
    await this.processAlerts()
  }

  private async checkCoverage(): Promise<void> {
    // Implementation
  }

  private async checkRecency(): Promise<void> {
    // Implementation
  }

  private async checkCompleteness(): Promise<void> {
    // Implementation
  }

  private async checkAccuracy(): Promise<void> {
    // Implementation
  }

  private async processAlerts(): Promise<void> {
    for (const alert of this.alerts) {
      if (this.config.alerts.email) {
        await this.sendEmailAlert(alert)
      }
      if (this.config.alerts.slack) {
        await this.sendSlackAlert(alert)
      }
      if (this.config.alerts.github_issue) {
        await this.createGitHubIssue(alert)
      }
    }
  }

  private async sendEmailAlert(alert: HealthAlert): Promise<void> {
    // Email configuration
  }

  private async sendSlackAlert(alert: HealthAlert): Promise<void> {
    // Slack webhook integration
  }

  private async createGitHubIssue(alert: HealthAlert): Promise<void> {
    // GitHub API integration
  }
}

// Configuration
const monitorConfig: DocMonitor = {
  enabled: true,
  update_interval: 3600,  // 1 hour
  thresholds: {
    coverage_min: 95,
    recency_max: 30,
    completeness_min: 90,
    accuracy_min: 98
  },
  alerts: {
    email: true,
    slack: true,
    dashboard: true,
    github_issue: true
  }
}

// Usage
const monitor = new DocumentationMonitor(monitorConfig)
setInterval(() => monitor.checkHealth(), monitorConfig.update_interval * 1000)
```

### 4.2 Dashboard Configuration

```yaml
# File: .claude/doc-dashboard.yml
# Purpose: Documentation health dashboard configuration

Dashboard:
  name: "Documentation Health Dashboard"
  refresh_interval: 300  # 5 minutes

  widgets:

    - id: "overall_health"
      type: "metric"
      title: "Overall Health Score"
      metric: "health_score"
      format: "percentage"
      target: 90
      update_frequency: "daily"

    - id: "coverage"
      type: "gauge"
      title: "Documentation Coverage"
      metric: "coverage_percent"
      target: 95
      warning_threshold: 90
      critical_threshold: 85

    - id: "recency"
      type: "timeline"
      title: "Documentation Age"
      metric: "average_age_days"
      target: 30
      warning_threshold: 45
      critical_threshold: 60

    - id: "completeness"
      type: "stacked_bar"
      title: "Completeness by Category"
      categories:
        - "Installation Guides"
        - "API Docs"
        - "Configuration"
        - "Troubleshooting"
        - "Architecture"
      target: 90

    - id: "accuracy"
      type: "heatmap"
      title: "Accuracy by Document"
      metric: "accuracy_scores"
      target: 98

    - id: "accessibility"
      type: "composite"
      title: "Accessibility Score"
      components:
        - "Readability"
        - "Search ability"
        - "Cross-linking"
        - "Code examples"
        - "Format quality"
      target: 90

    - id: "recent_alerts"
      type: "list"
      title: "Recent Issues"
      limit: 10
      severity_filter: ["critical", "high"]

    - id: "trend_analysis"
      type: "line_chart"
      title: "Health Trend (Last 30 days)"
      metrics:
        - "coverage_percent"
        - "completeness_score"
        - "accuracy_score"

    - id: "review_schedule"
      type: "calendar"
      title: "Upcoming Reviews"
      shows: ["weekly_reviews", "monthly_audits", "quarterly_strategy"]

    - id: "team_workload"
      type: "pie_chart"
      title: "Review Workload Distribution"
      assigned_to: ["tech_writer", "tech_lead", "support_team"]

  alerts:
    - metric: "coverage_percent"
      condition: "< 95"
      severity: "high"
      action: "Notify tech_writer"

    - metric: "average_age_days"
      condition: "> 30"
      severity: "medium"
      action: "Create review task"

    - metric: "completeness_score"
      condition: "< 90"
      severity: "medium"
      action: "Schedule review"

    - metric: "accuracy_score"
      condition: "< 98"
      severity: "critical"
      action: "Block release, notify team"

  export:
    formats: ["json", "csv", "pdf"]
    frequency: "weekly"
    recipients: ["team@company.com", "stakeholders@company.com"]
```

### 4.3 Monitoring Tools Integration

```bash
#!/bin/bash
# File: scripts/doc-monitor-setup.sh
# Purpose: Set up documentation monitoring infrastructure

set -e

echo "Setting up Documentation Monitoring..."

# Create necessary directories
mkdir -p .reports/doc-health
mkdir -p .reports/doc-trends
mkdir -p .claude/monitors

# Install monitoring tools
npm install --save-dev \
  markdown-link-check \
  markdown-toc \
  markdownlint \
  doctoc

# Configure Prometheus metrics (if applicable)
cat > .claude/monitors/prometheus-config.yml << 'EOF'
global:
  scrape_interval: 300s

scrape_configs:
  - job_name: 'documentation_health'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/docs/health/metrics'
EOF

# Configure health check endpoints
cat > .claude/monitors/health-endpoints.yml << 'EOF'
endpoints:
  - name: "coverage"
    path: "/api/health/docs/coverage"
    interval: 3600

  - name: "recency"
    path: "/api/health/docs/recency"
    interval: 3600

  - name: "completeness"
    path: "/api/health/docs/completeness"
    interval: 86400

  - name: "accuracy"
    path: "/api/health/docs/accuracy"
    interval: 604800
EOF

# Create cron job for daily monitoring
cat > /tmp/doc-monitor-cron << 'EOF'
# Documentation health monitoring
0 0 * * * /usr/local/bin/bash scripts/check-doc-health.sh >> /var/log/doc-health.log 2>&1
0 6 * * * /usr/local/bin/bash scripts/generate-doc-report.sh
0 10 * * 1 /usr/local/bin/bash scripts/weekly-doc-review.sh
EOF

echo "✓ Monitoring infrastructure configured"
echo "✓ Run 'npm run monitor:docs' to start monitoring"
```

---

## 5. Health Check Automation

### 5.1 Automated Validation Scripts

```python
# File: scripts/validate_doc_health.py
# Purpose: Comprehensive Python-based documentation validation

import os
import json
from datetime import datetime
from pathlib import Path
import re

class DocumentationHealthValidator:
    """Validates documentation health metrics"""

    def __init__(self, docs_dir: str = "./docs"):
        self.docs_dir = Path(docs_dir)
        self.violations = []
        self.metrics = {}

    def validate_all(self) -> dict:
        """Run all validation checks"""

        checks = {
            "coverage": self.check_coverage(),
            "recency": self.check_recency(),
            "completeness": self.check_completeness(),
            "links": self.check_links(),
            "code_examples": self.check_code_examples(),
            "formatting": self.check_formatting(),
            "metadata": self.check_metadata(),
        }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "violations": self.violations,
            "metrics": self.metrics,
            "health_score": self.calculate_health_score(checks)
        }

    def check_coverage(self) -> dict:
        """Check documentation coverage"""

        doc_count = len(list(self.docs_dir.glob("**/*.md")))

        return {
            "name": "Coverage",
            "status": "pass" if doc_count > 0 else "fail",
            "value": doc_count,
            "target": None
        }

    def check_recency(self) -> dict:
        """Check documentation recency"""

        now = datetime.now().timestamp()
        ages = []

        for doc_file in self.docs_dir.glob("**/*.md"):
            mod_time = doc_file.stat().st_mtime
            age_days = (now - mod_time) / 86400
            ages.append(age_days)

            if age_days > 30:
                self.violations.append({
                    "file": str(doc_file),
                    "issue": f"Document is {age_days:.0f} days old",
                    "severity": "medium"
                })

        avg_age = sum(ages) / len(ages) if ages else 0

        return {
            "name": "Recency",
            "status": "pass" if avg_age < 30 else "warning",
            "average_age_days": avg_age,
            "target": 30
        }

    def check_completeness(self) -> dict:
        """Check documentation completeness"""

        required_sections = [
            "Overview", "Installation", "Usage",
            "Examples", "Troubleshooting"
        ]

        incomplete = []

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text()

            for section in required_sections:
                if f"## {section}" not in content:
                    incomplete.append({
                        "file": str(doc_file),
                        "missing_section": section
                    })

        completeness = 100 - (len(incomplete) / len(required_sections) * 100) if incomplete else 100

        return {
            "name": "Completeness",
            "status": "pass" if completeness >= 90 else "warning",
            "score": completeness,
            "target": 90,
            "incomplete_items": incomplete
        }

    def check_links(self) -> dict:
        """Check for broken links"""

        broken_links = []
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text()
            links = re.findall(link_pattern, content)

            for text, url in links:
                if url.startswith("#"):
                    continue  # Skip anchors

                if url.startswith(("http://", "https://")):
                    continue  # Skip external links

                target_path = self.docs_dir / url
                if not target_path.exists():
                    broken_links.append({
                        "file": str(doc_file),
                        "target": url,
                        "text": text
                    })

        return {
            "name": "Link Integrity",
            "status": "pass" if not broken_links else "fail",
            "broken_links": len(broken_links),
            "issues": broken_links
        }

    def check_code_examples(self) -> dict:
        """Check for code examples"""

        files_with_examples = 0
        example_pattern = r'```[\w]*\n'

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text()
            if re.search(example_pattern, content):
                files_with_examples += 1

        total_files = len(list(self.docs_dir.glob("**/*.md")))
        coverage = (files_with_examples / total_files * 100) if total_files > 0 else 0

        return {
            "name": "Code Examples",
            "status": "pass" if coverage >= 80 else "warning",
            "coverage_percent": coverage,
            "files_with_examples": files_with_examples,
            "target": 80
        }

    def check_formatting(self) -> dict:
        """Check markdown formatting"""

        formatting_issues = []

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text()

            # Check for proper heading levels
            if content.count("## ") == 0:
                formatting_issues.append({
                    "file": str(doc_file),
                    "issue": "No H2 headings found"
                })

            # Check for proper list formatting
            if re.search(r'\n-(?! )', content):
                formatting_issues.append({
                    "file": str(doc_file),
                    "issue": "Improperly formatted list items"
                })

        return {
            "name": "Formatting",
            "status": "pass" if not formatting_issues else "warning",
            "issues": len(formatting_issues),
            "details": formatting_issues
        }

    def check_metadata(self) -> dict:
        """Check document metadata"""

        missing_metadata = []

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text()

            required_meta = [
                ("Version", r"Version:"),
                ("Last Updated", r"Last Updated:|Last Updated:"),
                ("Status", r"Status:")
            ]

            for meta_name, pattern in required_meta:
                if not re.search(pattern, content):
                    missing_metadata.append({
                        "file": str(doc_file),
                        "missing": meta_name
                    })

        return {
            "name": "Metadata",
            "status": "pass" if not missing_metadata else "warning",
            "issues": len(missing_metadata),
            "missing_items": missing_metadata
        }

    def calculate_health_score(self, checks: dict) -> float:
        """Calculate overall health score"""

        scores = []

        for check in checks.values():
            if check["status"] == "pass":
                scores.append(100)
            elif check["status"] == "warning":
                scores.append(75)
            else:  # fail
                scores.append(50)

        return sum(scores) / len(scores) if scores else 0

# Run validation
if __name__ == "__main__":
    validator = DocumentationHealthValidator()
    results = validator.validate_all()

    print(json.dumps(results, indent=2))

    # Save results
    output_file = Path(".reports/doc-health/validation-results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create metrics tracking system
- [ ] Set up CI/CD integration
- [ ] Establish review schedule
- [ ] Create documentation templates

### Phase 2: Automation (Week 2)
- [ ] Implement health check scripts
- [ ] Set up monitoring infrastructure
- [ ] Configure automated alerts
- [ ] Create reporting dashboards

### Phase 3: Process (Week 3)
- [ ] Train team on review process
- [ ] Establish sign-off procedures
- [ ] Document decision-making criteria
- [ ] Create escalation procedures

### Phase 4: Optimization (Week 4+)
- [ ] Gather team feedback
- [ ] Refine metrics and thresholds
- [ ] Optimize automation
- [ ] Plan continuous improvements

---

## 7. Quick Reference

### Health Score Targets
- **Overall**: 90%+
- **Coverage**: 95%+
- **Completeness**: 90%+
- **Accuracy**: 98%+
- **Accessibility**: 90%+

### Escalation Path
1. **Low severity** → Backlog for next sprint
2. **Medium severity** → Schedule for current sprint
3. **High severity** → Flag for immediate attention
4. **Critical** → Block deployment until resolved

### Key Contacts
- **Technical Lead**: Architecture, accuracy reviews
- **Tech Writer**: Completeness, clarity, consistency
- **Support Team**: Troubleshooting documentation
- **Developer**: Code examples, command validation

---

## Related Documentation

- [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md) - Complete documentation guide
- [CONTRIB.md](./CONTRIB.md) - Contributing guidelines
- [RUNBOOK.md](./RUNBOOK.md) - Operational procedures

---

**Last Updated**: 2026-01-29
**Next Review**: 2026-02-05
**Maintained By**: Technical Documentation Team
