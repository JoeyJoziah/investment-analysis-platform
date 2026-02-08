#!/bin/bash

################################################################################
# Documentation Count Validator
#
# Purpose: Validates documentation count and organization
# Usage: ./verify-doc-count.sh [--report] [--compare] [--threshold N]
#
# Flags:
#   --report    Generate detailed report
#   --compare   Compare with baseline
#   --threshold N  Alert if changes exceed N%
#   --verbose   Show detailed output
#   --help      Show this help message
#
# Tracks:
#   - Total markdown files
#   - Files per category
#   - Documentation coverage
#   - File size distribution
################################################################################

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY_DIR="${PROJECT_ROOT}/.claude/verify"
BASELINE_FILE="${VERIFY_DIR}/doc-count-baseline.json"
REPORT_FILE="${VERIFY_DIR}/doc-count-report.txt"
LOG_FILE="${VERIFY_DIR}/doc-count.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Flags
REPORT_MODE=false
COMPARE_MODE=false
VERBOSE=false
THRESHOLD=10

# Counters
declare -A DOC_CATEGORIES=()
TOTAL_DOCS=0
TOTAL_SIZE=0
declare -a SIZE_DISTRIBUTION=()

################################################################################
# Helper Functions
################################################################################

setup_directories() {
  mkdir -p "${VERIFY_DIR}"
  touch "${LOG_FILE}"
}

log() {
  local level="$1"
  shift
  local message="$@"
  echo "[${TIMESTAMP}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_quiet() {
  local message="$@"
  echo "[${TIMESTAMP}] ${message}" >> "${LOG_FILE}"
}

show_help() {
  grep "^# " "${BASH_SOURCE[0]}" | head -25
  exit 0
}

print_header() {
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

print_section() {
  echo ""
  echo ">>> $1"
  echo ""
}

format_size() {
  local bytes=$1
  if [[ $bytes -lt 1024 ]]; then
    echo "${bytes}B"
  elif [[ $bytes -lt 1048576 ]]; then
    echo "$((bytes / 1024))KB"
  else
    echo "$((bytes / 1048576))MB"
  fi
}

get_category() {
  local file="$1"
  local dir=$(dirname "$file" | sed "s|${PROJECT_ROOT}||")

  if [[ "$dir" == *".claude"* ]]; then
    echo "claude-internal"
  elif [[ "$dir" == *"docs"* ]]; then
    echo "documentation"
  elif [[ "$dir" == *"backend"* ]]; then
    echo "backend-docs"
  elif [[ "$dir" == *"frontend"* ]]; then
    echo "frontend-docs"
  else
    echo "root-docs"
  fi
}

################################################################################
# Documentation Analysis
################################################################################

count_documentation() {
  print_section "Counting documentation files..."

  local md_files
  md_files=$(find "${PROJECT_ROOT}" \
    -name "*.md" \
    -not -path "*/node_modules/*" \
    -not -path "*/.git/*" \
    -not -path "*/dist/*" \
    -type f 2>/dev/null)

  local count=0
  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    local category=$(get_category "$file")
    local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)

    DOC_CATEGORIES["$category"]=$((${DOC_CATEGORIES["$category"]:-0} + 1))
    TOTAL_DOCS=$((TOTAL_DOCS + 1))
    TOTAL_SIZE=$((TOTAL_SIZE + file_size))

    SIZE_DISTRIBUTION+=("$file_size:$file")

    ((count++))

    if [[ "$VERBOSE" == true ]]; then
      log "INFO" "[$category] $(basename "$file") ($(format_size "$file_size"))"
    fi

  done <<< "$md_files"

  log "INFO" "Found ${TOTAL_DOCS} documentation files"
}

analyze_distribution() {
  print_section "Analyzing documentation distribution..."

  # Sort by size
  IFS=$'\n' sorted=($(sort -rn <<<"${SIZE_DISTRIBUTION[*]}"))

  local top_count=0
  local top_size=0

  for entry in "${sorted[@]}"; do
    [[ -z "$entry" ]] && continue
    ((top_count++))

    local size="${entry%%:*}"
    top_size=$((top_size + size))

    if [[ $top_count -le 5 ]]; then
      local file="${entry##*:}"
      log_quiet "Top file: $(basename "$file") - $(format_size "$size")"
    fi

    if [[ $top_count -ge 10 ]]; then
      break
    fi
  done

  local percentage=$((top_size * 100 / TOTAL_SIZE))
  log_quiet "Top 10 files account for ${percentage}% of total documentation"
}

################################################################################
# Baseline and Comparison
################################################################################

save_baseline() {
  {
    echo "{"
    echo "  \"timestamp\": \"${TIMESTAMP}\","
    echo "  \"total_docs\": ${TOTAL_DOCS},"
    echo "  \"total_size\": ${TOTAL_SIZE},"
    echo "  \"categories\": {"

    local first=true
    for category in "${!DOC_CATEGORIES[@]}"; do
      if [[ "$first" == false ]]; then
        echo ","
      fi
      echo -n "    \"$category\": ${DOC_CATEGORIES[$category]}"
      first=false
    done

    echo ""
    echo "  }"
    echo "}"
  } > "${BASELINE_FILE}"

  log "INFO" "Baseline saved to: ${BASELINE_FILE}"
}

compare_with_baseline() {
  if [[ ! -f "$BASELINE_FILE" ]]; then
    log "WARNING" "No baseline found. Creating baseline..."
    save_baseline
    return
  fi

  print_section "Comparing with baseline..."

  local baseline_docs=$(grep '"total_docs"' "$BASELINE_FILE" | grep -oE '[0-9]+')
  local baseline_size=$(grep '"total_size"' "$BASELINE_FILE" | grep -oE '[0-9]+')

  if [[ -z "$baseline_docs" ]] || [[ -z "$baseline_size" ]]; then
    log "ERROR" "Invalid baseline file format"
    return
  fi

  local doc_change=$((TOTAL_DOCS - baseline_docs))
  local size_change=$((TOTAL_SIZE - baseline_size))
  local doc_change_pct=$(( doc_change * 100 / baseline_docs ))
  local size_change_pct=$(( size_change * 100 / baseline_size ))

  log_quiet "Documentation count change: ${doc_change} files (${doc_change_pct}%)"
  log_quiet "Documentation size change: $(format_size "$size_change") (${size_change_pct}%)"

  # Alert if threshold exceeded
  if [[ ${doc_change_pct#-} -gt $THRESHOLD ]]; then
    log "WARNING" "Documentation count changed by ${doc_change_pct}% (threshold: ${THRESHOLD}%)"
  fi

  if [[ ${size_change_pct#-} -gt $THRESHOLD ]]; then
    log "WARNING" "Documentation size changed by ${size_change_pct}% (threshold: ${THRESHOLD}%)"
  fi

  # Update baseline
  save_baseline
}

################################################################################
# Report Generation
################################################################################

generate_report() {
  {
    echo "================================================================================"
    echo "DOCUMENTATION COUNT VALIDATION REPORT"
    echo "================================================================================"
    echo "Generated: ${TIMESTAMP}"
    echo "Project: ${PROJECT_ROOT}"
    echo ""
    echo "SUMMARY"
    echo "--------"
    echo "Total Markdown Files: ${TOTAL_DOCS}"
    echo "Total Documentation Size: $(format_size "$TOTAL_SIZE")"
    echo ""
    echo "BREAKDOWN BY CATEGORY"
    echo "--------"

    for category in "${!DOC_CATEGORIES[@]}"; do
      echo "  $category: ${DOC_CATEGORIES[$category]} files"
    done

    echo ""
    echo "SIZE DISTRIBUTION"
    echo "--------"

    # Calculate percentages
    local count=0
    for entry in "${SIZE_DISTRIBUTION[@]}"; do
      [[ -z "$entry" ]] && continue
      ((count++))
    done

    local avg_size=$((TOTAL_SIZE / TOTAL_DOCS))
    echo "  Average file size: $(format_size "$avg_size")"
    echo "  Total files: ${count}"

    echo ""
    echo "ANALYSIS"
    echo "--------"

    if [[ ${DOC_CATEGORIES["root-docs"]:-0} -gt 5 ]]; then
      echo "  ⚠ Warning: ${DOC_CATEGORIES["root-docs"]} markdown files in root directory"
    fi

    if [[ $TOTAL_DOCS -lt 20 ]]; then
      echo "  ⚠ Warning: Low documentation coverage (${TOTAL_DOCS} files)"
    else
      echo "  ✓ Good documentation coverage (${TOTAL_DOCS} files)"
    fi

    echo ""
    echo "RECOMMENDATIONS"
    echo "================================================================================"
    echo "1. Maintain documentation count above 20 files"
    echo "2. Organize docs in /docs or /.claude directories"
    echo "3. Keep average file size under 50KB"
    echo "4. Review and consolidate large files (>100KB)"

  } > "${REPORT_FILE}"

  log "INFO" "Report generated: ${REPORT_FILE}"
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --report)
        REPORT_MODE=true
        shift
        ;;
      --compare)
        COMPARE_MODE=true
        shift
        ;;
      --threshold)
        THRESHOLD="$2"
        shift 2
        ;;
      --verbose)
        VERBOSE=true
        shift
        ;;
      --help)
        show_help
        ;;
      *)
        echo "Unknown option: $1"
        show_help
        ;;
    esac
  done
}

################################################################################
# Main Execution
################################################################################

main() {
  parse_arguments "$@"

  setup_directories

  print_header "DOCUMENTATION COUNT VALIDATION"
  log "INFO" "Starting documentation analysis..."

  count_documentation
  analyze_distribution

  if [[ "$COMPARE_MODE" == true ]]; then
    compare_with_baseline
  else
    save_baseline
  fi

  if [[ "$REPORT_MODE" == true ]]; then
    generate_report
  fi

  print_section "Results Summary"
  echo "Total Documentation Files: ${TOTAL_DOCS}"
  echo "Total Size: $(format_size "$TOTAL_SIZE")"
  echo ""

  print_section "Documentation by Category"
  for category in "${!DOC_CATEGORIES[@]}"; do
    echo "  $category: ${DOC_CATEGORIES[$category]} files"
  done

  echo ""
  echo "Baseline saved to: ${BASELINE_FILE}"

  if [[ "$REPORT_MODE" == true ]]; then
    echo "Report saved to: ${REPORT_FILE}"
  fi

  echo "Log saved to: ${LOG_FILE}"
}

main "$@"
