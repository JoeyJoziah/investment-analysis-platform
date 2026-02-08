#!/bin/bash

################################################################################
# Version Tag Consistency Checker
#
# Purpose: Validates version tag consistency across documentation
# Usage: ./verify-version-tags.sh [--fix] [--update VERSION] [--strict]
#
# Flags:
#   --fix         Attempt to fix version inconsistencies
#   --update V    Update all versions to specified version
#   --strict      Fail on any version mismatch
#   --verbose     Show detailed output
#   --help        Show this help message
#
# Checks:
#   - Version tags in documentation
#   - Last Updated dates
#   - Semantic versioning format
#   - Consistency across files
################################################################################

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY_DIR="${PROJECT_ROOT}/.claude/verify"
REPORT_FILE="${VERIFY_DIR}/version-tags-report.txt"
LOG_FILE="${VERIFY_DIR}/version-tags.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Flags
FIX_MODE=false
STRICT_MODE=false
VERBOSE=false
TARGET_VERSION=""

# Counters
TOTAL_CHECKED=0
CONSISTENT_TAGS=0
INCONSISTENT_TAGS=0
MISSING_TAGS=0
declare -a INCONSISTENCIES=()
declare -a FILES_TO_UPDATE=()
declare -A VERSION_MAP=()

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

is_valid_semver() {
  local version="$1"
  if [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    return 0
  fi
  return 1
}

extract_version() {
  local file="$1"
  local version_pattern="Version[: ]*([0-9]+\.[0-9]+\.[0-9]+[^ \n]*)|version[: ]*([0-9]+\.[0-9]+\.[0-9]+[^ \n]*)|v([0-9]+\.[0-9]+\.[0-9]+[^ \n]*)"

  grep -oiE "$version_pattern" "$file" | head -1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" || echo ""
}

extract_last_updated() {
  local file="$1"
  local date_pattern="Last Updated[: ]*([0-9]{4}-[0-9]{2}-[0-9]{2}|[A-Za-z]+ [0-9]{1,2}, [0-9]{4})"

  grep -oE "$date_pattern" "$file" | head -1 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2}|[A-Za-z]+ [0-9]{1,2}, [0-9]{4}" || echo ""
}

################################################################################
# Version Checking Functions
################################################################################

scan_documentation_versions() {
  print_section "Scanning documentation for version tags..."

  local md_files
  md_files=$(find "${PROJECT_ROOT}/docs" "${PROJECT_ROOT}/.claude" \
    -name "*.md" \
    -not -path "*/node_modules/*" \
    -type f 2>/dev/null | sort)

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    ((TOTAL_CHECKED++))
    local file_name=$(basename "$file")
    local version=$(extract_version "$file")
    local last_updated=$(extract_last_updated "$file")

    if [[ -n "$version" ]]; then
      VERSION_MAP["$file"]="$version"
      CONSISTENT_TAGS=$((CONSISTENT_TAGS + 1))

      log_quiet "Found version '$version' in $file_name (Updated: $last_updated)"

      if [[ "$VERBOSE" == true ]]; then
        log "INFO" "✓ $file_name: v$version (Updated: $last_updated)"
      fi
    else
      MISSING_TAGS=$((MISSING_TAGS + 1))
      INCONSISTENCIES+=("Missing version tag in: $file_name")

      log_quiet "Missing version tag in $file_name"

      if [[ "$VERBOSE" == true ]]; then
        log "WARNING" "✗ $file_name: No version tag found"
      fi
    fi

  done <<< "$md_files"
}

check_version_consistency() {
  print_section "Checking version consistency..."

  local versions_found=()
  local version_count=0

  for file in "${!VERSION_MAP[@]}"; do
    local version="${VERSION_MAP[$file]}"
    versions_found+=("$version")
  done

  # Check if all versions are the same
  if [[ ${#versions_found[@]} -gt 0 ]]; then
    local primary_version="${versions_found[0]}"

    for version in "${versions_found[@]}"; do
      if [[ "$version" != "$primary_version" ]]; then
        INCONSISTENT_TAGS=$((INCONSISTENT_TAGS + 1))
        INCONSISTENCIES+=("Version mismatch: Expected '$primary_version' but found '$version'")
      fi
    done
  fi

  # Validate semantic versioning
  for file in "${!VERSION_MAP[@]}"; do
    local version="${VERSION_MAP[$file]}"
    if ! is_valid_semver "$version"; then
      INCONSISTENCIES+=("Invalid semver format: $version in $(basename "$file")")
      log "WARNING" "Invalid semver format: $version"
    fi
  done

  if [[ $INCONSISTENT_TAGS -eq 0 ]] && [[ $MISSING_TAGS -eq 0 ]]; then
    log "INFO" "✓ All version tags are consistent"
  else
    log "WARNING" "Found ${INCONSISTENT_TAGS} inconsistencies and ${MISSING_TAGS} missing tags"
  fi
}

check_date_consistency() {
  print_section "Checking date consistency..."

  local md_files
  md_files=$(find "${PROJECT_ROOT}/docs" "${PROJECT_ROOT}/.claude" \
    -name "*.md" \
    -not -path "*/node_modules/*" \
    -type f 2>/dev/null)

  local dates_found=()
  local today=$(date +"%Y-%m-%d")

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    local last_updated=$(extract_last_updated "$file")
    if [[ -n "$last_updated" ]]; then
      dates_found+=("$last_updated")

      # Check if date is in the future
      if [[ "$last_updated" > "$today" ]]; then
        INCONSISTENCIES+=("Future date found: $last_updated in $(basename "$file")")
        log "WARNING" "Future date: $last_updated in $(basename "$file")"
      fi
    fi
  done <<< "$md_files"

  log "INFO" "Checked ${#dates_found[@]} last-updated dates"
}

################################################################################
# Fix Functions
################################################################################

fix_version_tags() {
  if [[ -z "$TARGET_VERSION" ]]; then
    log "ERROR" "No target version specified. Use: --update VERSION"
    return 1
  fi

  print_section "Updating version tags to ${TARGET_VERSION}..."

  local md_files
  md_files=$(find "${PROJECT_ROOT}/docs" "${PROJECT_ROOT}/.claude" \
    -name "*.md" \
    -not -path "*/node_modules/*" \
    -type f 2>/dev/null)

  local update_count=0

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    local current_version=$(extract_version "$file")

    if [[ -n "$current_version" ]] && [[ "$current_version" != "$TARGET_VERSION" ]]; then
      # Update version tag
      sed -i.bak "s/Version: ${current_version}/Version: ${TARGET_VERSION}/g" "$file"
      sed -i.bak "s/version: ${current_version}/version: ${TARGET_VERSION}/g" "$file"
      sed -i.bak "s/v${current_version}/v${TARGET_VERSION}/g" "$file"

      # Update last modified date
      local today=$(date +"%Y-%m-%d")
      sed -i.bak "s/Last Updated: .*/Last Updated: ${today}/g" "$file"

      ((update_count++))
      log "INFO" "Updated $(basename "$file"): v${current_version} -> v${TARGET_VERSION}"

      # Clean up backup
      rm -f "${file}.bak"
    fi
  done <<< "$md_files"

  log "INFO" "Updated ${update_count} files to version ${TARGET_VERSION}"
}

update_last_modified_dates() {
  print_section "Updating last-modified dates..."

  local today=$(date +"%Y-%m-%d")
  local md_files
  md_files=$(find "${PROJECT_ROOT}/docs" "${PROJECT_ROOT}/.claude" \
    -name "*.md" \
    -not -path "*/node_modules/*" \
    -type f 2>/dev/null)

  local update_count=0

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    # Check if file has a "Last Updated" field
    if grep -q "Last Updated" "$file"; then
      sed -i.bak "s/Last Updated: .*/Last Updated: ${today}/g" "$file"
      ((update_count++))
      rm -f "${file}.bak"
    fi
  done <<< "$md_files"

  log "INFO" "Updated ${update_count} files with today's date"
}

################################################################################
# Report Generation
################################################################################

generate_report() {
  {
    echo "================================================================================"
    echo "VERSION TAG CONSISTENCY REPORT"
    echo "================================================================================"
    echo "Generated: ${TIMESTAMP}"
    echo "Project: ${PROJECT_ROOT}"
    echo ""
    echo "SUMMARY"
    echo "--------"
    echo "Files Checked: ${TOTAL_CHECKED}"
    echo "Consistent Tags: ${CONSISTENT_TAGS}"
    echo "Inconsistent Tags: ${INCONSISTENT_TAGS}"
    echo "Missing Tags: ${MISSING_TAGS}"
    echo ""

    if [[ ${#VERSION_MAP[@]} -gt 0 ]]; then
      echo "FOUND VERSIONS"
      echo "--------"
      for file in "${!VERSION_MAP[@]}"; do
        local version="${VERSION_MAP[$file]}"
        echo "  $(basename "$file"): v$version"
      done
      echo ""
    fi

    if [[ ${#INCONSISTENCIES[@]} -gt 0 ]]; then
      echo "INCONSISTENCIES"
      echo "--------"
      for issue in "${INCONSISTENCIES[@]}"; do
        echo "  ✗ $issue"
      done
      echo ""
    fi

    echo "VALIDATION RESULTS"
    echo "--------"
    if [[ $INCONSISTENT_TAGS -eq 0 ]] && [[ $MISSING_TAGS -eq 0 ]]; then
      echo "✓ All version tags are consistent!"
    else
      echo "✗ Found issues with version tags"
    fi

    echo ""
    echo "RECOMMENDATIONS"
    echo "================================================================================"
    if [[ $MISSING_TAGS -gt 0 ]]; then
      echo "1. Add version tags to ${MISSING_TAGS} files"
    fi
    if [[ $INCONSISTENT_TAGS -gt 0 ]]; then
      echo "2. Standardize version numbers across documentation"
      echo "3. Use: $0 --update <VERSION> to update all versions"
    fi
    echo "4. Keep Last Updated dates current"

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
      --update)
        FIX_MODE=true
        TARGET_VERSION="$2"
        if ! is_valid_semver "$TARGET_VERSION"; then
          log "ERROR" "Invalid version format: $TARGET_VERSION (expected: X.Y.Z)"
          exit 1
        fi
        shift 2
        ;;
      --strict)
        STRICT_MODE=true
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

  print_header "VERSION TAG CONSISTENCY CHECK"
  log "INFO" "Starting version tag validation..."

  scan_documentation_versions
  check_version_consistency
  check_date_consistency

  if [[ "$FIX_MODE" == true ]] && [[ -n "$TARGET_VERSION" ]]; then
    fix_version_tags
    update_last_modified_dates
  fi

  generate_report

  print_section "Results Summary"
  echo "Total Files Checked: ${TOTAL_CHECKED}"
  echo "Consistent Tags: ${CONSISTENT_TAGS}"
  echo "Inconsistent Tags: ${INCONSISTENT_TAGS}"
  echo "Missing Tags: ${MISSING_TAGS}"

  if [[ ${#INCONSISTENCIES[@]} -gt 0 ]]; then
    print_section "Issues Found"
    for issue in "${INCONSISTENCIES[@]}"; do
      echo "  ✗ $issue"
    done
  fi

  echo ""
  echo "Report saved to: ${REPORT_FILE}"
  echo "Log saved to: ${LOG_FILE}"

  if [[ "$STRICT_MODE" == true ]] && [[ $INCONSISTENT_TAGS -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
