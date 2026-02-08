#!/bin/bash

################################################################################
# Root Directory Cleanup Verifier
#
# Purpose: Verifies root directory is clean and organized
# Usage: ./verify-root-cleanup.sh [--clean] [--ignore FILE] [--strict]
#
# Flags:
#   --clean     Remove non-essential files from root
#   --ignore    Whitelist files/patterns (comma-separated)
#   --strict    Fail on any violations
#   --verbose   Show detailed output
#   --help      Show this help message
#
# Checks:
#   - Prevents duplicate " 2" files in root
#   - Ensures only essential files in root
#   - Verifies organization of documentation
#   - Detects temporary or backup files
################################################################################

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY_DIR="${PROJECT_ROOT}/.claude/verify"
REPORT_FILE="${VERIFY_DIR}/root-cleanup-report.txt"
LOG_FILE="${VERIFY_DIR}/root-cleanup.log"
ALLOWED_FILE="${VERIFY_DIR}/root-allowed-files.txt"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Flags
CLEAN_MODE=false
STRICT_MODE=false
VERBOSE=false
IGNORE_PATTERNS=""

# Counters
TOTAL_FILES=0
ALLOWED_FILES=0
SUSPICIOUS_FILES=0
DUPLICATE_FILES=0
TEMP_FILES=0
declare -a SUSPICIOUS_ARRAY=()
declare -a DUPLICATES_ARRAY=()
declare -a TEMP_ARRAY=()

# Allowed files in root (organized by type)
ESSENTIAL_FILES=(
  "README.md"
  "LICENSE"
  "package.json"
  "package-lock.json"
  "tsconfig.json"
  ".gitignore"
  ".env.example"
  ".editorconfig"
  "CLAUDE.md"
  "AUTH_FLOW_TEST_PROGRESS.md"
  "MIDDLEWARE_FIXES_SUMMARY.md"
  "NEXT_STEPS_DEBUG.md"
  "PHASE3_IMPLEMENTATION_COMPLETE.md"
  "TEST_VERIFICATION_INDEX.md"
  "TEST_VERIFICATION_REPORT.md"
)

# Temporary/Backup patterns to clean
TEMP_PATTERNS=(
  "* 2.*"
  "*.bak"
  "*.tmp"
  "*.swp"
  ".DS_Store"
  "Thumbs.db"
)

# Directories that should exist
REQUIRED_DIRECTORIES=(
  "backend"
  "frontend"
  "scripts"
  "docs"
  ".claude"
  "data"
)

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

is_allowed() {
  local file="$1"
  local filename=$(basename "$file")

  # Check essential files
  for allowed in "${ESSENTIAL_FILES[@]}"; do
    if [[ "$filename" == "$allowed" ]]; then
      return 0
    fi
  done

  # Check ignore patterns
  if [[ -n "$IGNORE_PATTERNS" ]]; then
    IFS=',' read -ra PATTERNS <<< "$IGNORE_PATTERNS"
    for pattern in "${PATTERNS[@]}"; do
      pattern=$(echo "$pattern" | xargs)  # Trim whitespace
      if [[ "$filename" == "$pattern" ]] || [[ "$filename" =~ $pattern ]]; then
        return 0
      fi
    done
  fi

  return 1
}

is_temporary() {
  local file="$1"
  local filename=$(basename "$file")

  for pattern in "${TEMP_PATTERNS[@]}"; do
    if [[ "$filename" == "$pattern" ]] || [[ "$filename" =~ $(echo "$pattern" | sed 's/\*/.*/' | sed 's/?/./g') ]]; then
      return 0
    fi
  done

  return 1
}

is_duplicate() {
  local file="$1"
  if [[ "$file" == * 2.* ]]; then
    return 0
  fi
  return 1
}

################################################################################
# Root Directory Analysis
################################################################################

scan_root_directory() {
  print_section "Scanning root directory..."

  local items
  items=$(find "${PROJECT_ROOT}" -maxdepth 1 -type f ! -name ".*" 2>/dev/null)

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    local filename=$(basename "$file")
    ((TOTAL_FILES++))

    log_quiet "Found file in root: $filename"

    if is_temporary "$file"; then
      TEMP_FILES=$((TEMP_FILES + 1))
      TEMP_ARRAY+=("$filename")
      log_quiet "  → Temporary file"

    elif is_duplicate "$file"; then
      DUPLICATE_FILES=$((DUPLICATE_FILES + 1))
      DUPLICATES_ARRAY+=("$filename")
      log_quiet "  → Duplicate file"

    elif is_allowed "$file"; then
      ALLOWED_FILES=$((ALLOWED_FILES + 1))
      log_quiet "  → Allowed file"

    else
      SUSPICIOUS_FILES=$((SUSPICIOUS_FILES + 1))
      SUSPICIOUS_ARRAY+=("$filename")
      log_quiet "  → Suspicious/Unorganized file"
    fi

    if [[ "$VERBOSE" == true ]]; then
      log "INFO" "  $filename"
    fi

  done <<< "$items"

  log "INFO" "Scanned ${TOTAL_FILES} files in root directory"
}

verify_required_directories() {
  print_section "Verifying required directories..."

  local missing_count=0

  for dir in "${REQUIRED_DIRECTORIES[@]}"; do
    if [[ ! -d "${PROJECT_ROOT}/${dir}" ]]; then
      log "WARNING" "Missing directory: $dir"
      ((missing_count++))
    else
      log_quiet "✓ Found: $dir"
    fi
  done

  if [[ $missing_count -eq 0 ]]; then
    log "INFO" "All required directories present"
  fi
}

check_organization() {
  print_section "Checking documentation organization..."

  # Check if docs are organized
  local root_md_count=$(find "${PROJECT_ROOT}" -maxdepth 1 -name "*.md" -type f 2>/dev/null | wc -l)

  if [[ $root_md_count -gt 5 ]]; then
    log "WARNING" "Too many markdown files in root (${root_md_count})"
    SUSPICIOUS_FILES=$((SUSPICIOUS_FILES + (root_md_count - 5)))
  fi

  # Check for test files in root
  local root_test_count=$(find "${PROJECT_ROOT}" -maxdepth 1 -name "*.test.*" -o -name "*.spec.*" 2>/dev/null | wc -l)

  if [[ $root_test_count -gt 0 ]]; then
    log "WARNING" "Test files found in root (should be in backend/tests)"
  fi

  log "INFO" "Organization check complete"
}

################################################################################
# Cleanup Functions
################################################################################

remove_temporary_files() {
  if [[ ${#TEMP_ARRAY[@]} -eq 0 ]]; then
    log "INFO" "No temporary files to remove"
    return
  fi

  print_section "Removing temporary files..."

  for file in "${TEMP_ARRAY[@]}"; do
    local full_path="${PROJECT_ROOT}/${file}"
    if [[ -f "$full_path" ]]; then
      log "INFO" "Removing: $file"
      rm -f "$full_path"
    fi
  done

  log "INFO" "Removed ${#TEMP_ARRAY[@]} temporary files"
}

move_suspicious_files() {
  if [[ ${#SUSPICIOUS_ARRAY[@]} -eq 0 ]]; then
    log "INFO" "No suspicious files to move"
    return
  fi

  print_section "Moving suspicious files..."

  local archive_dir="${PROJECT_ROOT}/.claude/archived-root-files"
  mkdir -p "$archive_dir"

  for file in "${SUSPICIOUS_ARRAY[@]}"; do
    local full_path="${PROJECT_ROOT}/${file}"
    if [[ -f "$full_path" ]]; then
      log "INFO" "Moving: $file -> .claude/archived-root-files/"
      mv "$full_path" "$archive_dir/"
    fi
  done

  log "INFO" "Archived ${#SUSPICIOUS_ARRAY[@]} suspicious files"
}

remove_duplicate_files() {
  if [[ ${#DUPLICATES_ARRAY[@]} -eq 0 ]]; then
    log "INFO" "No duplicate files to remove"
    return
  fi

  print_section "Removing duplicate ' 2' files..."

  for file in "${DUPLICATES_ARRAY[@]}"; do
    local full_path="${PROJECT_ROOT}/${file}"
    if [[ -f "$full_path" ]]; then
      log "INFO" "Removing duplicate: $file"
      rm -f "$full_path"
    fi
  done

  log "INFO" "Removed ${#DUPLICATES_ARRAY[@]} duplicate files"
}

################################################################################
# Report Generation
################################################################################

save_allowed_files_list() {
  {
    echo "# Allowed Files in Root Directory"
    echo "# Last Updated: ${TIMESTAMP}"
    echo ""
    echo "## Essential Files"

    for file in "${ESSENTIAL_FILES[@]}"; do
      echo "  - $file"
    done

  } > "${ALLOWED_FILE}"

  log "INFO" "Allowed files list saved to: ${ALLOWED_FILE}"
}

generate_report() {
  {
    echo "================================================================================"
    echo "ROOT DIRECTORY CLEANUP VERIFICATION REPORT"
    echo "================================================================================"
    echo "Generated: ${TIMESTAMP}"
    echo "Project: ${PROJECT_ROOT}"
    echo ""
    echo "SUMMARY"
    echo "--------"
    echo "Total Files in Root: ${TOTAL_FILES}"
    echo "Allowed Files: ${ALLOWED_FILES}"
    echo "Duplicate Files: ${DUPLICATE_FILES}"
    echo "Temporary Files: ${TEMP_FILES}"
    echo "Suspicious/Unorganized: ${SUSPICIOUS_FILES}"
    echo ""

    if [[ ${#DUPLICATES_ARRAY[@]} -gt 0 ]]; then
      echo "DUPLICATE FILES"
      echo "--------"
      for file in "${DUPLICATES_ARRAY[@]}"; do
        echo "  ✗ $file"
      done
      echo ""
    fi

    if [[ ${#TEMP_ARRAY[@]} -gt 0 ]]; then
      echo "TEMPORARY FILES"
      echo "--------"
      for file in "${TEMP_ARRAY[@]}"; do
        echo "  ⚠ $file"
      done
      echo ""
    fi

    if [[ ${#SUSPICIOUS_ARRAY[@]} -gt 0 ]]; then
      echo "SUSPICIOUS/UNORGANIZED FILES"
      echo "--------"
      for file in "${SUSPICIOUS_ARRAY[@]}"; do
        echo "  ? $file"
      done
      echo ""
    fi

    echo "ALLOWED FILES"
    echo "--------"
    for file in "${ESSENTIAL_FILES[@]}"; do
      if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
        echo "  ✓ $file"
      fi
    done

    echo ""
    echo "VALIDATION RESULTS"
    echo "--------"
    if [[ $DUPLICATE_FILES -eq 0 ]] && [[ $TEMP_FILES -eq 0 ]] && [[ $SUSPICIOUS_FILES -eq 0 ]]; then
      echo "✓ Root directory is clean and well-organized!"
    else
      echo "✗ Root directory needs cleanup"
    fi

    echo ""
    echo "RECOMMENDATIONS"
    echo "================================================================================"
    if [[ $DUPLICATE_FILES -gt 0 ]]; then
      echo "1. Remove ${DUPLICATE_FILES} duplicate ' 2' files"
    fi
    if [[ $TEMP_FILES -gt 0 ]]; then
      echo "2. Remove ${TEMP_FILES} temporary files"
    fi
    if [[ $SUSPICIOUS_FILES -gt 0 ]]; then
      echo "3. Move ${SUSPICIOUS_FILES} files to appropriate directories"
    fi
    echo ""
    echo "Use: $0 --clean to automatically clean up"

  } > "${REPORT_FILE}"

  log "INFO" "Report generated: ${REPORT_FILE}"
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --clean)
        CLEAN_MODE=true
        shift
        ;;
      --ignore)
        IGNORE_PATTERNS="$2"
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

  print_header "ROOT DIRECTORY CLEANUP VERIFICATION"
  log "INFO" "Starting root directory analysis..."

  scan_root_directory
  verify_required_directories
  check_organization

  if [[ "$CLEAN_MODE" == true ]]; then
    remove_duplicate_files
    remove_temporary_files
    move_suspicious_files
  fi

  save_allowed_files_list
  generate_report

  print_section "Results Summary"
  echo "Total Files in Root: ${TOTAL_FILES}"
  echo "Allowed: ${ALLOWED_FILES} | Duplicates: ${DUPLICATE_FILES} | Temp: ${TEMP_FILES} | Suspicious: ${SUSPICIOUS_FILES}"

  if [[ $DUPLICATE_FILES -gt 0 ]] || [[ $TEMP_FILES -gt 0 ]] || [[ $SUSPICIOUS_FILES -gt 0 ]]; then
    print_section "Issues Found"
    if [[ $DUPLICATE_FILES -gt 0 ]]; then
      echo "Duplicates:"
      for file in "${DUPLICATES_ARRAY[@]}"; do
        echo "  ✗ $file"
      done
    fi
    if [[ $TEMP_FILES -gt 0 ]]; then
      echo "Temporary files:"
      for file in "${TEMP_ARRAY[@]}"; do
        echo "  ⚠ $file"
      done
    fi
    if [[ $SUSPICIOUS_FILES -gt 0 ]]; then
      echo "Unorganized files:"
      for file in "${SUSPICIOUS_ARRAY[@]}"; do
        echo "  ? $file"
      done
    fi
  else
    print_section "✓ Root directory is clean"
  fi

  echo ""
  echo "Report saved to: ${REPORT_FILE}"
  echo "Allowed files list: ${ALLOWED_FILE}"
  echo "Log saved to: ${LOG_FILE}"

  if [[ "$STRICT_MODE" == true ]] && \
     ([[ $DUPLICATE_FILES -gt 0 ]] || [[ $TEMP_FILES -gt 0 ]] || [[ $SUSPICIOUS_FILES -gt 0 ]]); then
    exit 1
  fi
}

main "$@"
