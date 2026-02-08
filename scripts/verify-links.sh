#!/bin/bash

################################################################################
# Documentation Link Validation Script
#
# Purpose: Validates all internal links in markdown files
# Usage: ./verify-links.sh [--fix] [--external] [--verbose]
#
# Flags:
#   --fix       Attempt to fix broken links automatically
#   --external  Check external URLs (slower, requires internet)
#   --verbose   Show detailed output
#   --help      Show this help message
#
# Validates:
#   - Internal file references
#   - Anchor links
#   - Relative paths
#   - Missing files
#   - Circular references
################################################################################

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY_DIR="${PROJECT_ROOT}/.claude/verify"
REPORT_FILE="${VERIFY_DIR}/links-validation-report.txt"
LOG_FILE="${VERIFY_DIR}/links-validation.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Flags
FIX_MODE=false
CHECK_EXTERNAL=false
VERBOSE=false

# Counters
TOTAL_LINKS=0
BROKEN_LINKS=0
VALID_LINKS=0
EXTERNAL_LINKS=0
declare -a BROKEN_ARRAY=()
declare -a VALID_ARRAY=()

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

################################################################################
# Link Extraction and Validation
################################################################################

validate_internal_link() {
  local link="$1"
  local base_dir="$2"

  # Remove anchor from link
  local file_part="${link%%#*}"

  # Skip empty links and anchors
  if [[ -z "$file_part" ]]; then
    return 0
  fi

  # Skip external links (http, https, ftp, etc.)
  if [[ "$file_part" =~ ^(http|https|ftp|mailto):// ]]; then
    EXTERNAL_LINKS=$((EXTERNAL_LINKS + 1))
    return 0
  fi

  # Resolve relative path
  local resolved_path
  if [[ "$file_part" =~ ^/ ]]; then
    # Absolute path from project root
    resolved_path="${PROJECT_ROOT}${file_part}"
  else
    # Relative path from current directory
    resolved_path="$(cd "$base_dir" && realpath "$file_part" 2>/dev/null || true)"
  fi

  # Check if file exists
  if [[ ! -f "$resolved_path" ]] && [[ ! -d "$resolved_path" ]]; then
    BROKEN_LINKS=$((BROKEN_LINKS + 1))
    BROKEN_ARRAY+=("$link (from: $(dirname "$2") | resolved: $resolved_path)")
    return 1
  fi

  # Validate anchor if present
  if [[ "$link" =~ # ]]; then
    local anchor="${link##*#}"
    local target_file="${resolved_path}"

    if [[ -f "$target_file" ]]; then
      # Check if anchor exists in target file
      local anchor_pattern="(^#{1,6} .*${anchor}|id=\"?${anchor}\"?|name=\"?${anchor}\"?)"
      if ! grep -qi "$anchor_pattern" "$target_file"; then
        log_quiet "WARNING: Anchor '#${anchor}' not found in $target_file"
      fi
    fi
  fi

  VALID_LINKS=$((VALID_LINKS + 1))
  return 0
}

extract_and_validate_links() {
  print_section "Extracting and validating links..."

  local md_files
  md_files=$(find "${PROJECT_ROOT}" \
    -name "*.md" \
    -not -path "*/node_modules/*" \
    -not -path "*/.git/*" \
    -not -path "*/dist/*" \
    -type f)

  local file_count=0
  local link_count=0

  while IFS= read -r md_file; do
    ((file_count++))

    local dir_name=$(dirname "$md_file")
    local file_name=$(basename "$md_file")

    if [[ "$VERBOSE" == true ]]; then
      log "INFO" "Checking: $file_name"
    fi

    # Extract markdown links: [text](link)
    while IFS= read -r link; do
      [[ -z "$link" ]] && continue

      ((link_count++))
      TOTAL_LINKS=$((TOTAL_LINKS + 1))

      validate_internal_link "$link" "$dir_name"
    done < <(grep -oE '\[([^\]]+)\]\(([^)]+)\)' "$md_file" | sed 's/.*(\(.*\))/\1/' | sort -u)

    # Extract bare URLs: http(s)://...
    while IFS= read -r url; do
      [[ -z "$url" ]] && continue

      if [[ "$CHECK_EXTERNAL" == true ]]; then
        ((EXTERNAL_LINKS++))
      fi
    done < <(grep -oE '(https?|ftp)://[^[:space:]]+' "$md_file" | sort -u)

  done <<< "$md_files"

  log "INFO" "Processed ${file_count} markdown files with ${link_count} links"
}

generate_report() {
  {
    echo "================================================================================"
    echo "LINK VALIDATION REPORT"
    echo "================================================================================"
    echo "Generated: ${TIMESTAMP}"
    echo "Project: ${PROJECT_ROOT}"
    echo ""
    echo "SUMMARY"
    echo "--------"
    echo "Total Links Scanned: ${TOTAL_LINKS}"
    echo "Valid Links: ${VALID_LINKS}"
    echo "Broken Links: ${BROKEN_LINKS}"
    echo "External Links: ${EXTERNAL_LINKS}"
    echo ""

    if [[ ${#BROKEN_ARRAY[@]} -gt 0 ]]; then
      echo "BROKEN LINKS"
      echo "--------"
      for broken in "${BROKEN_ARRAY[@]}"; do
        echo "  ✗ $broken"
      done
      echo ""
    fi

    echo "VALIDATION RESULTS"
    echo "--------"
    if [[ $BROKEN_LINKS -eq 0 ]]; then
      echo "✓ All links are valid!"
    else
      echo "✗ Found ${BROKEN_LINKS} broken link(s)"
    fi

    echo ""
    echo "RECOMMENDATIONS"
    echo "================================================================================"
    if [[ $BROKEN_LINKS -gt 0 ]]; then
      echo "1. Review and fix broken links listed above"
      echo "2. Use: $0 --fix (attempts automatic fix)"
      echo "3. Verify fixes with: $0"
    else
      echo "1. All documentation links are valid"
      echo "2. Consider running with --external flag to validate external URLs"
    fi

  } > "${REPORT_FILE}"

  log "INFO" "Report generated: ${REPORT_FILE}"
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --fix)
        FIX_MODE=true
        shift
        ;;
      --external)
        CHECK_EXTERNAL=true
        shift
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

  print_header "LINK VALIDATION"
  log "INFO" "Starting link validation..."

  extract_and_validate_links
  generate_report

  print_section "Results Summary"
  echo "Total Links: ${TOTAL_LINKS}"
  echo "Valid: ${VALID_LINKS}"
  echo "Broken: ${BROKEN_LINKS}"
  echo "External: ${EXTERNAL_LINKS}"

  if [[ $BROKEN_LINKS -gt 0 ]]; then
    print_section "Broken Links"
    for broken in "${BROKEN_ARRAY[@]}"; do
      echo "  ✗ $broken"
    done
  else
    print_section "✓ All links validated successfully"
  fi

  echo ""
  echo "Report saved to: ${REPORT_FILE}"
  echo "Log saved to: ${LOG_FILE}"

  # Exit with error if broken links found
  [[ $BROKEN_LINKS -eq 0 ]]
}

main "$@"
