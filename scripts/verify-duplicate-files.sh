#!/bin/bash

################################################################################
# Duplicate Files Detection Script
#
# Purpose: Detects and reports duplicate " 2" suffixed files in the repository
# Usage: ./verify-duplicate-files.sh [--fix] [--report]
#
# Flags:
#   --fix      Remove duplicate " 2" files (keeps original)
#   --report   Generate detailed report file
#   --help     Show this help message
################################################################################

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
VERIFY_DIR="${PROJECT_ROOT}/.claude/verify"
REPORT_FILE="${VERIFY_DIR}/duplicate-files-report.txt"
LOG_FILE="${VERIFY_DIR}/duplicate-files.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Flags
FIX_MODE=false
REPORT_MODE=false
VERBOSE=false

# Counters
TOTAL_DUPLICATES=0
TOTAL_SIZE_SAVED=0
declare -a DUPLICATES=()

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
  grep "^# " "${BASH_SOURCE[0]}" | head -20
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
# Main Verification Functions
################################################################################

find_duplicate_files() {
  print_section "Scanning for duplicate ' 2' suffixed files..."

  local duplicate_count=0

  # Find all files with " 2" pattern (excluding node_modules and dist)
  while IFS= read -r -d '' file; do
    # Skip node_modules, dist, and build directories
    if [[ "$file" == *"/node_modules/"* ]] || \
       [[ "$file" == *"/dist/"* ]] || \
       [[ "$file" == *"/build/"* ]] || \
       [[ "$file" == *"/.git/"* ]]; then
      continue
    fi

    # Extract the base name without " 2" suffix
    local base_name="${file% 2*}"
    local extension="${file##*.}"
    local base_with_ext="${base_name}.${extension}"

    # Check if the original file exists
    if [[ -f "$base_with_ext" ]]; then
      DUPLICATES+=("$file")
      ((duplicate_count++))

      local dup_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
      local orig_size=$(stat -f%z "$base_with_ext" 2>/dev/null || stat -c%s "$base_with_ext" 2>/dev/null)

      log_quiet "DUPLICATE: $file (${dup_size} bytes)"
      log_quiet "  Original: $base_with_ext (${orig_size} bytes)"

      TOTAL_SIZE_SAVED=$((TOTAL_SIZE_SAVED + dup_size))
    fi
  done < <(find "${PROJECT_ROOT}" -maxdepth 3 -type f -name "* 2.*" -print0 2>/dev/null)

  TOTAL_DUPLICATES=$duplicate_count
  log "INFO" "Found ${duplicate_count} duplicate files"
}

generate_report() {
  if [[ ${#DUPLICATES[@]} -eq 0 ]]; then
    echo "No duplicate files found." > "${REPORT_FILE}"
    log "INFO" "No duplicates to report"
    return
  fi

  {
    echo "================================================================================"
    echo "DUPLICATE FILES VERIFICATION REPORT"
    echo "================================================================================"
    echo "Generated: ${TIMESTAMP}"
    echo "Project: ${PROJECT_ROOT}"
    echo ""
    echo "SUMMARY"
    echo "--------"
    echo "Total Duplicates Found: ${TOTAL_DUPLICATES}"
    echo "Total Space Wasted: $(numfmt --to=iec-i --suffix=B ${TOTAL_SIZE_SAVED} 2>/dev/null || echo ${TOTAL_SIZE_SAVED} bytes)"
    echo ""
    echo "DUPLICATE FILES"
    echo "--------"

    for dup in "${DUPLICATES[@]}"; do
      local size=$(stat -f%z "$dup" 2>/dev/null || stat -c%s "$dup" 2>/dev/null)
      local base_name="${dup% 2*}"
      local extension="${dup##*.}"
      local base_with_ext="${base_name}.${extension}"

      echo ""
      echo "Duplicate: $dup"
      echo "  Size: $(numfmt --to=iec-i --suffix=B ${size} 2>/dev/null || echo ${size} bytes)"
      echo "  Original: $base_with_ext"

      if [[ -f "$base_with_ext" ]]; then
        local orig_size=$(stat -f%z "$base_with_ext" 2>/dev/null || stat -c%s "$base_with_ext" 2>/dev/null)
        echo "  Original Size: $(numfmt --to=iec-i --suffix=B ${orig_size} 2>/dev/null || echo ${orig_size} bytes)"
      fi
    done

    echo ""
    echo "================================================================================"
    echo "RECOMMENDATIONS"
    echo "================================================================================"
    echo "1. Review each duplicate to ensure the original contains all needed content"
    echo "2. Run: $0 --fix to automatically remove duplicates"
    echo "3. After removal, verify repository integrity with: $0"

  } > "${REPORT_FILE}"

  log "INFO" "Report generated: ${REPORT_FILE}"
}

remove_duplicates() {
  if [[ ${#DUPLICATES[@]} -eq 0 ]]; then
    log "INFO" "No duplicates to remove"
    return
  fi

  print_section "Removing duplicate files..."

  for dup in "${DUPLICATES[@]}"; do
    if [[ -f "$dup" ]]; then
      log "INFO" "Removing: $dup"
      rm -f "$dup"
    fi
  done

  log "INFO" "Successfully removed ${TOTAL_DUPLICATES} duplicate files"
}

verify_no_duplicates() {
  print_section "Verifying no duplicates remain..."

  local remaining=$(find "${PROJECT_ROOT}" -maxdepth 3 -type f -name "* 2.*" -not -path "*/node_modules/*" -not -path "*/.git/*" 2>/dev/null | wc -l)

  if [[ $remaining -eq 0 ]]; then
    log "INFO" "✓ No duplicate files detected"
    return 0
  else
    log "ERROR" "✗ Found ${remaining} remaining duplicates"
    return 1
  fi
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
      --report)
        REPORT_MODE=true
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

  print_header "DUPLICATE FILES VERIFICATION"
  log "INFO" "Starting verification..."

  find_duplicate_files

  if [[ $TOTAL_DUPLICATES -gt 0 ]]; then
    print_section "Results: ${TOTAL_DUPLICATES} duplicate file(s) found"

    for dup in "${DUPLICATES[@]}"; do
      echo "  - $dup"
    done

    if [[ "$REPORT_MODE" == true ]]; then
      generate_report
    fi

    if [[ "$FIX_MODE" == true ]]; then
      remove_duplicates
      verify_no_duplicates
    fi
  else
    print_section "✓ No duplicate files found"
  fi

  log "INFO" "Verification complete. Log: ${LOG_FILE}"

  if [[ "$REPORT_MODE" == true ]]; then
    echo ""
    echo "Report saved to: ${REPORT_FILE}"
  fi
}

main "$@"
