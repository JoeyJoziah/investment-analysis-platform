#!/bin/bash
# File: scripts/check-doc-health.sh
# Purpose: Automated documentation health checks
# Usage: ./scripts/check-doc-health.sh [--fix] [--report] [--verbose]

set -e

readonly DOCS_DIR="./docs"
readonly REPORTS_DIR=".reports/doc-health"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'  # No Color

# Options
VERBOSE=false
GENERATE_REPORT=false
FIX_ISSUES=false

# Metrics
COVERAGE_SCORE=0
RECENCY_SCORE=0
COMPLETENESS_SCORE=0
LINKS_SCORE=0
EXAMPLES_SCORE=0
OVERALL_SCORE=0
TOTAL_FILES=0
TOTAL_VIOLATIONS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --fix)
            FIX_ISSUES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Initialize
initialize() {
    mkdir -p "$REPORTS_DIR"
    echo -e "${BLUE}Documentation Health Check${NC}"
    echo "Time: $(date)"
    echo "Directory: $DOCS_DIR"
    echo ""
}

# Check 1: Coverage
check_coverage() {
    echo -e "${BLUE}[1/6]${NC} Checking documentation coverage..."

    local backend_files=$(find ./backend -type f \( -name "*.ts" -o -name "*.py" \) 2>/dev/null | wc -l)
    local frontend_files=$(find ./frontend -type f \( -name "*.tsx" -o -name "*.ts" \) 2>/dev/null | wc -l)
    local total_modules=$((backend_files + frontend_files))

    local doc_files=$(find "$DOCS_DIR" -type f -name "*.md" 2>/dev/null | wc -l)

    if [ "$total_modules" -eq 0 ]; then
        COVERAGE_SCORE=100
    else
        COVERAGE_SCORE=$((doc_files * 100 / total_modules))
    fi

    echo "  Files: $doc_files documentation files"
    echo "  Modules: $total_modules source modules"
    echo "  Coverage: $COVERAGE_SCORE%"

    if [ "$COVERAGE_SCORE" -ge 95 ]; then
        echo -e "  ${GREEN}✓ Coverage meets target (95%)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Coverage below target${NC}"
        ((TOTAL_VIOLATIONS++))
    fi
    echo ""
}

# Check 2: Recency
check_recency() {
    echo -e "${BLUE}[2/6]${NC} Checking documentation recency..."

    local now=$(date +%s)
    local total_age=0
    local max_age=0
    local stale_count=0

    TOTAL_FILES=$(find "$DOCS_DIR" -type f -name "*.md" | wc -l)

    if [ "$TOTAL_FILES" -eq 0 ]; then
        echo "  No documentation files found"
        RECENCY_SCORE=0
        return
    fi

    while IFS= read -r file; do
        local modified=$(stat -f%m "$file" 2>/dev/null || stat -c%Y "$file" 2>/dev/null)
        local age_days=$(( (now - modified) / 86400 ))
        total_age=$((total_age + age_days))

        if [ "$age_days" -gt "$max_age" ]; then
            max_age=$age_days
        fi

        if [ "$age_days" -gt 30 ]; then
            stale_count=$((stale_count + 1))
            if [ "$VERBOSE" = true ]; then
                echo -e "  ${YELLOW}⚠ Stale:${NC} $(basename "$file") ($age_days days old)"
            fi
        fi
    done < <(find "$DOCS_DIR" -type f -name "*.md")

    local avg_age=$((total_age / TOTAL_FILES))

    echo "  Average age: $avg_age days"
    echo "  Maximum age: $max_age days"
    echo "  Stale files (>30 days): $stale_count"

    if [ "$avg_age" -le 30 ]; then
        RECENCY_SCORE=100
        echo -e "  ${GREEN}✓ Documentation is recent${NC}"
    elif [ "$avg_age" -le 45 ]; then
        RECENCY_SCORE=75
        echo -e "  ${YELLOW}⚠ Some documentation needs updating${NC}"
        ((TOTAL_VIOLATIONS++))
    else
        RECENCY_SCORE=50
        echo -e "  ${RED}✗ Documentation is outdated${NC}"
        ((TOTAL_VIOLATIONS++))
    fi
    echo ""
}

# Check 3: Completeness
check_completeness() {
    echo -e "${BLUE}[3/6]${NC} Checking documentation completeness..."

    local required_sections=("Overview" "Installation" "Usage" "Examples" "Troubleshooting")
    local complete_count=0
    local incomplete_docs=0

    for file in "$DOCS_DIR"/*.md; do
        if [ -f "$file" ]; then
            local file_sections=0
            for section in "${required_sections[@]}"; do
                if grep -q "^## $section" "$file"; then
                    ((file_sections++))
                fi
            done

            if [ "$file_sections" -lt "${#required_sections[@]}" ]; then
                ((incomplete_docs++))
                if [ "$VERBOSE" = true ]; then
                    echo -e "  ${YELLOW}⚠ Incomplete:${NC} $(basename "$file") ($file_sections/${#required_sections[@]} sections)"
                fi
            else
                ((complete_count++))
            fi
        fi
    done

    if [ "$TOTAL_FILES" -gt 0 ]; then
        COMPLETENESS_SCORE=$((complete_count * 100 / TOTAL_FILES))
    else
        COMPLETENESS_SCORE=0
    fi

    echo "  Complete documents: $complete_count / $TOTAL_FILES"
    echo "  Completeness score: $COMPLETENESS_SCORE%"

    if [ "$COMPLETENESS_SCORE" -ge 90 ]; then
        echo -e "  ${GREEN}✓ Documentation is complete${NC}"
    else
        echo -e "  ${YELLOW}⚠ Some documents need sections${NC}"
        ((TOTAL_VIOLATIONS++))
    fi
    echo ""
}

# Check 4: Links
check_links() {
    echo -e "${BLUE}[4/6]${NC} Checking documentation links..."

    local broken_count=0
    local link_pattern='\[([^\]]+)\]\(([^)]+)\)'

    for file in "$DOCS_DIR"/*.md; do
        if [ -f "$file" ]; then
            while IFS= read -r line; do
                if [[ $line =~ $link_pattern ]]; then
                    local url="${BASH_REMATCH[2]}"

                    # Skip external links and anchors
                    if [[ $url =~ ^http ]]; then
                        continue
                    fi
                    if [[ $url =~ ^# ]]; then
                        continue
                    fi

                    # Check if file exists
                    local target_path="$DOCS_DIR/$url"
                    if [ ! -f "$target_path" ]; then
                        broken_count=$((broken_count + 1))
                        if [ "$VERBOSE" = true ]; then
                            echo -e "  ${RED}✗ Broken link:${NC} $(basename "$file") -> $url"
                        fi
                    fi
                fi
            done < "$file"
        fi
    done

    if [ "$broken_count" -eq 0 ]; then
        LINKS_SCORE=100
        echo "  Broken links: 0"
        echo -e "  ${GREEN}✓ All links are valid${NC}"
    else
        LINKS_SCORE=50
        echo "  Broken links: $broken_count"
        echo -e "  ${RED}✗ Found broken links${NC}"
        ((TOTAL_VIOLATIONS++))
    fi
    echo ""
}

# Check 5: Code Examples
check_code_examples() {
    echo -e "${BLUE}[5/6]${NC} Checking for code examples..."

    local files_with_examples=0

    for file in "$DOCS_DIR"/*.md; do
        if [ -f "$file" ] && grep -q '```' "$file"; then
            ((files_with_examples++))
        fi
    done

    if [ "$TOTAL_FILES" -gt 0 ]; then
        EXAMPLES_SCORE=$((files_with_examples * 100 / TOTAL_FILES))
    else
        EXAMPLES_SCORE=0
    fi

    echo "  Files with code examples: $files_with_examples / $TOTAL_FILES"
    echo "  Code example coverage: $EXAMPLES_SCORE%"

    if [ "$EXAMPLES_SCORE" -ge 80 ]; then
        echo -e "  ${GREEN}✓ Good code example coverage${NC}"
    else
        echo -e "  ${YELLOW}⚠ Consider adding more examples${NC}"
    fi
    echo ""
}

# Check 6: Frontmatter
check_frontmatter() {
    echo -e "${BLUE}[6/6]${NC} Checking documentation metadata..."

    local missing_frontmatter=0

    for file in "$DOCS_DIR"/*.md; do
        if [ -f "$file" ]; then
            if ! grep -q "Last Updated\|Version\|Status" "$file"; then
                missing_frontmatter=$((missing_frontmatter + 1))
                if [ "$VERBOSE" = true ]; then
                    echo -e "  ${YELLOW}⚠ Missing metadata:${NC} $(basename "$file")"
                fi
            fi
        fi
    done

    if [ "$missing_frontmatter" -eq 0 ]; then
        echo -e "  ${GREEN}✓ All files have required metadata${NC}"
    else
        echo -e "  ${YELLOW}⚠ $missing_frontmatter files missing metadata${NC}"
        ((TOTAL_VIOLATIONS++))
    fi
    echo ""
}

# Calculate overall score
calculate_overall_score() {
    OVERALL_SCORE=$(( (COVERAGE_SCORE + RECENCY_SCORE + COMPLETENESS_SCORE + LINKS_SCORE + EXAMPLES_SCORE) / 5 ))
}

# Generate health report
generate_report() {
    if [ "$GENERATE_REPORT" = false ]; then
        return
    fi

    echo -e "${BLUE}Generating health report...${NC}"

    cat > "$REPORTS_DIR/health_report_${TIMESTAMP}.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "overall_health": {
    "score": $OVERALL_SCORE,
    "status": "$([ "$OVERALL_SCORE" -ge 90 ] && echo "Healthy" || echo "Needs Review")"
  },
  "metrics": {
    "coverage": $COVERAGE_SCORE,
    "recency": $RECENCY_SCORE,
    "completeness": $COMPLETENESS_SCORE,
    "links": $LINKS_SCORE,
    "code_examples": $EXAMPLES_SCORE
  },
  "summary": {
    "total_files": $TOTAL_FILES,
    "total_violations": $TOTAL_VIOLATIONS,
    "timestamp": "$(date)"
  }
}
EOF

    echo -e "  ${GREEN}✓ Report saved${NC}: $REPORTS_DIR/health_report_${TIMESTAMP}.json"
}

# Summary
print_summary() {
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}Documentation Health Summary${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo "Coverage:        $COVERAGE_SCORE%"
    echo "Recency:         $RECENCY_SCORE%"
    echo "Completeness:    $COMPLETENESS_SCORE%"
    echo "Links:           $LINKS_SCORE%"
    echo "Code Examples:   $EXAMPLES_SCORE%"
    echo ""
    echo -n "Overall Score:   "
    if [ "$OVERALL_SCORE" -ge 90 ]; then
        echo -e "${GREEN}$OVERALL_SCORE%${NC}"
    elif [ "$OVERALL_SCORE" -ge 75 ]; then
        echo -e "${YELLOW}$OVERALL_SCORE%${NC}"
    else
        echo -e "${RED}$OVERALL_SCORE%${NC}"
    fi
    echo -e "Violations:      ${YELLOW}$TOTAL_VIOLATIONS${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo ""
}

# Main execution
main() {
    initialize
    check_coverage
    check_recency
    check_completeness
    check_links
    check_code_examples
    check_frontmatter
    calculate_overall_score
    generate_report
    print_summary

    if [ "$OVERALL_SCORE" -lt 75 ]; then
        exit 1
    fi
}

main "$@"
