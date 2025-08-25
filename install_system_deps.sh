#!/bin/bash

# Investment Analysis Platform - System Dependencies Installer
# ===========================================================
# 
# Production-grade system dependency installer for the investment analysis platform.
# Handles cross-platform installation of native libraries, development tools,
# and system packages required for Python dependencies.
#
# Features:
# - Cross-platform support (Ubuntu/Debian, CentOS/RHEL/Fedora, macOS, Arch)
# - Parallel installation where safe
# - Comprehensive error handling and logging
# - Dependency validation and version checking
# - Recovery and retry mechanisms
# - Performance optimizations
#
# Usage:
#   ./install_system_deps.sh [OPTIONS]
#
# Author: Claude Code (Platform Engineer)
# Version: 2.0.0

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# CONFIGURATION
# ============================================================================

# Script metadata
SCRIPT_NAME="install_system_deps.sh"
SCRIPT_VERSION="2.0.0"
LOG_FILE="system_deps_install.log"
LOCK_FILE="/tmp/system_deps_install.lock"

# Installation settings
MAX_RETRIES=3
RETRY_DELAY=5
INSTALL_TIMEOUT=300
PARALLEL_JOBS=4
VERBOSE=false
DRY_RUN=false
FORCE_INSTALL=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# LOGGING AND UTILITIES
# ============================================================================

# Initialize logging
setup_logging() {
    # Create log file with proper permissions
    touch "$LOG_FILE"
    chmod 644 "$LOG_FILE"
    
    # Log script start
    log_info "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    log_info "Log file: $LOG_FILE"
    log_info "Timestamp: $(date)"
    log_info "User: $(whoami)"
    log_info "Platform: $(uname -a)"
}

# Logging functions
log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1" | tee -a "$LOG_FILE"
    else
        echo "[DEBUG] $1" >> "$LOG_FILE"
    fi
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

# Progress indicator
show_progress() {
    local pid=$1
    local message="$2"
    local spin='/-\|'
    local i=0
    
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\r${CYAN}[${spin:$i:1}]${NC} %s" "$message"
        sleep 0.1
    done
    printf "\r"
}

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

# Detect operating system and distribution
detect_os() {
    local os_type="unknown"
    local os_version="unknown"
    local is_wsl=false
    
    # Check for WSL environment
    if [[ -n "${WSL_DISTRO_NAME:-}" ]] || [[ -n "${WSL_INTEROP:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
        is_wsl=true
    fi
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ -f /etc/os-release ]]; then
            . /etc/os-release
            case "$ID" in
                ubuntu)
                    os_type="ubuntu"
                    os_version="$VERSION_ID"
                    ;;
                debian)
                    os_type="debian"
                    os_version="$VERSION_ID"
                    ;;
                centos|rhel)
                    os_type="centos"
                    os_version="$VERSION_ID"
                    ;;
                fedora)
                    os_type="fedora"
                    os_version="$VERSION_ID"
                    ;;
                arch|manjaro)
                    os_type="arch"
                    os_version="rolling"
                    ;;
                *)
                    os_type="linux"
                    ;;
            esac
        else
            os_type="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        os_type="macos"
        os_version="$(sw_vers -productVersion)"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        os_type="windows"
        os_version="$(cmd.exe /c ver 2>/dev/null | tr -d '\r')"
    fi
    
    # Export WSL status for use by other functions
    export IS_WSL="$is_wsl"
    
    # Log WSL detection only once
    if [[ "$is_wsl" == "true" ]]; then
        log_info "WSL environment detected"
    fi
    
    echo "$os_type:$os_version"
}

# Check if running as root (required for system package installation)
check_privileges() {
    if [[ $EUID -ne 0 ]] && [[ "$os_type" != "macos" ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            log_error "This script requires root privileges or sudo access"
            return 1
        fi
        log_info "Using sudo for system package installation"
    fi
    return 0
}

# ============================================================================
# PACKAGE MANAGEMENT
# ============================================================================

# Update package manager cache
update_package_cache() {
    local os_type="$1"
    
    log_step "Updating package manager cache"
    
    case "$os_type" in
        ubuntu|debian)
            if [[ "$DRY_RUN" == "false" ]]; then
                # WSL-specific handling for package cache update
                if [[ "${IS_WSL:-false}" == "true" ]]; then
                    export DEBIAN_FRONTEND=noninteractive
                fi
                run_with_retry "sudo apt-get update" "Failed to update apt cache"
            else
                log_info "[DRY RUN] Would run: sudo apt-get update"
            fi
            ;;
        centos)
            if [[ "$DRY_RUN" == "false" ]]; then
                run_with_retry "sudo yum makecache" "Failed to update yum cache"
            else
                log_info "[DRY RUN] Would run: sudo yum makecache"
            fi
            ;;
        fedora)
            if [[ "$DRY_RUN" == "false" ]]; then
                run_with_retry "sudo dnf makecache" "Failed to update dnf cache"
            else
                log_info "[DRY RUN] Would run: sudo dnf makecache"
            fi
            ;;
        arch)
            if [[ "$DRY_RUN" == "false" ]]; then
                run_with_retry "sudo pacman -Sy" "Failed to update pacman cache"
            else
                log_info "[DRY RUN] Would run: sudo pacman -Sy"
            fi
            ;;
        macos)
            if command -v brew >/dev/null 2>&1; then
                if [[ "$DRY_RUN" == "false" ]]; then
                    run_with_retry "brew update" "Failed to update Homebrew"
                else
                    log_info "[DRY RUN] Would run: brew update"
                fi
            else
                log_warn "Homebrew not found on macOS"
            fi
            ;;
        *)
            log_warn "Package cache update not implemented for OS: $os_type"
            ;;
    esac
}

# Install packages with retry logic
install_packages() {
    local os_type="$1"
    shift
    local packages=("$@")
    
    if [[ ${#packages[@]} -eq 0 ]]; then
        log_debug "No packages to install"
        return 0
    fi
    
    log_step "Installing packages: ${packages[*]}"
    
    case "$os_type" in
        ubuntu|debian)
            if [[ "$DRY_RUN" == "false" ]]; then
                # WSL-specific handling to avoid interactive package configuration issues
                if [[ "${IS_WSL:-false}" == "true" ]]; then
                    # Set DEBIAN_FRONTEND to avoid interactive prompts in WSL
                    export DEBIAN_FRONTEND=noninteractive
                    # Skip problematic packages in WSL (like mail services)
                    local filtered_packages=()
                    for pkg in "${packages[@]}"; do
                        case "$pkg" in
                            postfix|exim4*|sendmail*)
                                log_warn "Skipping mail service package in WSL: $pkg"
                                ;;
                            *)
                                filtered_packages+=("$pkg")
                                ;;
                        esac
                    done
                    run_with_retry "sudo apt-get install -y --no-install-recommends ${filtered_packages[*]}" "Failed to install packages via apt"
                else
                    run_with_retry "sudo apt-get install -y ${packages[*]}" "Failed to install packages via apt"
                fi
            else
                log_info "[DRY RUN] Would run: sudo apt-get install -y ${packages[*]}"
            fi
            ;;
        centos)
            if [[ "$DRY_RUN" == "false" ]]; then
                run_with_retry "sudo yum install -y ${packages[*]}" "Failed to install packages via yum"
            else
                log_info "[DRY RUN] Would run: sudo yum install -y ${packages[*]}"
            fi
            ;;
        fedora)
            if [[ "$DRY_RUN" == "false" ]]; then
                run_with_retry "sudo dnf install -y ${packages[*]}" "Failed to install packages via dnf"
            else
                log_info "[DRY RUN] Would run: sudo dnf install -y ${packages[*]}"
            fi
            ;;
        arch)
            if [[ "$DRY_RUN" == "false" ]]; then
                run_with_retry "sudo pacman -S --noconfirm ${packages[*]}" "Failed to install packages via pacman"
            else
                log_info "[DRY RUN] Would run: sudo pacman -S --noconfirm ${packages[*]}"
            fi
            ;;
        macos)
            if command -v brew >/dev/null 2>&1; then
                if [[ "$DRY_RUN" == "false" ]]; then
                    run_with_retry "brew install ${packages[*]}" "Failed to install packages via brew"
                else
                    log_info "[DRY RUN] Would run: brew install ${packages[*]}"
                fi
            else
                log_error "Homebrew not found on macOS. Please install Homebrew first."
                return 1
            fi
            ;;
        *)
            log_error "Package installation not implemented for OS: $os_type"
            return 1
            ;;
    esac
}

# Run command with retry logic
run_with_retry() {
    local command="$1"
    local error_msg="$2"
    local attempt=1
    
    while [[ $attempt -le $MAX_RETRIES ]]; do
        log_debug "Attempt $attempt/$MAX_RETRIES: $command"
        
        if timeout $INSTALL_TIMEOUT bash -c "$command" &>>"$LOG_FILE"; then
            log_debug "Command succeeded on attempt $attempt"
            return 0
        else
            log_warn "Command failed on attempt $attempt"
            if [[ $attempt -lt $MAX_RETRIES ]]; then
                log_info "Retrying in ${RETRY_DELAY}s..."
                sleep $RETRY_DELAY
            fi
        fi
        
        ((attempt++))
    done
    
    log_error "$error_msg after $MAX_RETRIES attempts"
    return 1
}

# ============================================================================
# DEPENDENCY DEFINITIONS
# ============================================================================

# Define system dependencies for different categories
declare -A PYTHON_DEV_DEPS
# Ubuntu 24.04+ integrated distutils into python3-full, python3-distutils no longer exists
PYTHON_DEV_DEPS[ubuntu]="python3-dev python3-pip python3-venv python3-full"
PYTHON_DEV_DEPS[debian]="python3-dev python3-pip python3-venv"
PYTHON_DEV_DEPS[centos]="python3-devel python3-pip python3-setuptools"
PYTHON_DEV_DEPS[fedora]="python3-devel python3-pip python3-setuptools"
PYTHON_DEV_DEPS[arch]="python python-pip"
PYTHON_DEV_DEPS[macos]="python@3.12"

declare -A BUILD_DEPS
BUILD_DEPS[ubuntu]="build-essential gcc g++ make cmake pkg-config"
BUILD_DEPS[debian]="build-essential gcc g++ make cmake pkg-config"
BUILD_DEPS[centos]="gcc gcc-c++ make cmake pkgconfig kernel-devel"
BUILD_DEPS[fedora]="gcc gcc-c++ make cmake pkgconf-devel kernel-devel"
BUILD_DEPS[arch]="base-devel cmake pkgconf"
BUILD_DEPS[macos]="cmake pkg-config"

declare -A CRYPTO_DEPS
CRYPTO_DEPS[ubuntu]="libffi-dev libssl-dev"
CRYPTO_DEPS[debian]="libffi-dev libssl-dev"
CRYPTO_DEPS[centos]="libffi-devel openssl-devel"
CRYPTO_DEPS[fedora]="libffi-devel openssl-devel"
CRYPTO_DEPS[arch]="libffi openssl"
CRYPTO_DEPS[macos]="libffi openssl"

declare -A XML_DEPS
XML_DEPS[ubuntu]="libxml2-dev libxslt1-dev zlib1g-dev"
XML_DEPS[debian]="libxml2-dev libxslt1-dev zlib1g-dev"
XML_DEPS[centos]="libxml2-devel libxslt-devel zlib-devel"
XML_DEPS[fedora]="libxml2-devel libxslt-devel zlib-devel"
XML_DEPS[arch]="libxml2 libxslt zlib"
XML_DEPS[macos]="libxml2 libxslt"

declare -A DATABASE_DEPS
DATABASE_DEPS[ubuntu]="libpq-dev postgresql-client redis-tools"
DATABASE_DEPS[debian]="libpq-dev postgresql-client redis-tools"
DATABASE_DEPS[centos]="postgresql-devel postgresql redis"
DATABASE_DEPS[fedora]="postgresql-devel postgresql redis"
DATABASE_DEPS[arch]="postgresql-libs postgresql redis"
DATABASE_DEPS[macos]="postgresql redis"

declare -A KAFKA_DEPS
KAFKA_DEPS[ubuntu]="librdkafka-dev"
KAFKA_DEPS[debian]="librdkafka-dev"
KAFKA_DEPS[centos]="librdkafka-devel"
KAFKA_DEPS[fedora]="librdkafka-devel"
KAFKA_DEPS[arch]="librdkafka"
KAFKA_DEPS[macos]="librdkafka"

declare -A TALIB_DEPS
TALIB_DEPS[ubuntu]=""
TALIB_DEPS[debian]=""
TALIB_DEPS[centos]=""
TALIB_DEPS[fedora]=""
TALIB_DEPS[arch]="ta-lib"
TALIB_DEPS[macos]="ta-lib"

# ============================================================================
# WSL-SPECIFIC FIXES
# ============================================================================

# Fix broken postfix installation in WSL environment
fix_postfix_wsl() {
    if [[ "${IS_WSL:-false}" == "true" ]] && dpkg -l | grep -q "^iF.*postfix"; then
        log_info "Fixing broken postfix installation in WSL"
        
        # Configure postfix with a valid domain to prevent configuration errors
        if [[ "$DRY_RUN" == "false" ]]; then
            # Set a default domain to fix the configuration issue
            sudo debconf-set-selections <<< "postfix postfix/main_mailer_type string 'Local only'"
            sudo debconf-set-selections <<< "postfix postfix/mailname string localhost.localdomain"
            sudo debconf-set-selections <<< "postfix postfix/destinations string localhost.localdomain, localhost"
            
            # Try to reconfigure postfix
            if sudo DEBIAN_FRONTEND=noninteractive dpkg --configure postfix; then
                log_info "Fixed postfix configuration"
            else
                log_warn "Could not fix postfix, removing it"
                sudo apt-get remove --purge -y postfix 2>/dev/null || true
                sudo apt-get autoremove -y 2>/dev/null || true
            fi
        else
            log_info "[DRY RUN] Would fix postfix configuration or remove it"
        fi
    fi
}

# ============================================================================
# DEPENDENCY INSTALLATION FUNCTIONS
# ============================================================================

install_python_development() {
    local os_type="$1"
    
    log_step "Installing Python development dependencies"
    
    if [[ -n "${PYTHON_DEV_DEPS[$os_type]:-}" ]]; then
        install_packages "$os_type" ${PYTHON_DEV_DEPS[$os_type]}
    else
        log_warn "Python development dependencies not defined for $os_type"
    fi
}

install_build_tools() {
    local os_type="$1"
    
    log_step "Installing build tools and compilers"
    
    if [[ -n "${BUILD_DEPS[$os_type]:-}" ]]; then
        install_packages "$os_type" ${BUILD_DEPS[$os_type]}
    else
        log_warn "Build dependencies not defined for $os_type"
    fi
    
    # Install Xcode Command Line Tools on macOS if needed
    if [[ "$os_type" == "macos" ]]; then
        if ! xcode-select -p &>/dev/null; then
            log_step "Installing Xcode Command Line Tools"
            if [[ "$DRY_RUN" == "false" ]]; then
                xcode-select --install
                log_info "Xcode Command Line Tools installation initiated"
                log_warn "Please complete the installation in the GUI and re-run this script"
            else
                log_info "[DRY RUN] Would run: xcode-select --install"
            fi
        else
            log_info "Xcode Command Line Tools already installed"
        fi
    fi
}

install_crypto_libraries() {
    local os_type="$1"
    
    log_step "Installing cryptography libraries"
    
    if [[ -n "${CRYPTO_DEPS[$os_type]:-}" ]]; then
        install_packages "$os_type" ${CRYPTO_DEPS[$os_type]}
    else
        log_warn "Cryptography dependencies not defined for $os_type"
    fi
}

install_xml_libraries() {
    local os_type="$1"
    
    log_step "Installing XML processing libraries"
    
    if [[ -n "${XML_DEPS[$os_type]:-}" ]]; then
        install_packages "$os_type" ${XML_DEPS[$os_type]}
    else
        log_warn "XML dependencies not defined for $os_type"
    fi
}

install_database_libraries() {
    local os_type="$1"
    
    log_step "Installing database client libraries"
    
    if [[ -n "${DATABASE_DEPS[$os_type]:-}" ]]; then
        install_packages "$os_type" ${DATABASE_DEPS[$os_type]}
    else
        log_warn "Database dependencies not defined for $os_type"
    fi
}

install_kafka_libraries() {
    local os_type="$1"
    
    log_step "Installing Kafka client libraries"
    
    if [[ -n "${KAFKA_DEPS[$os_type]:-}" ]]; then
        install_packages "$os_type" ${KAFKA_DEPS[$os_type]}
    else
        log_warn "Kafka dependencies not defined for $os_type"
    fi
}

install_ta_lib() {
    local os_type="$1"
    
    log_step "Installing TA-Lib (Technical Analysis Library)"
    
    if [[ -n "${TALIB_DEPS[$os_type]:-}" ]] && [[ "${TALIB_DEPS[$os_type]}" != "" ]]; then
        # TA-Lib available in package manager
        install_packages "$os_type" ${TALIB_DEPS[$os_type]}
    else
        # Manual TA-Lib installation required
        log_info "Installing TA-Lib from source (package not available in $os_type)"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            install_ta_lib_from_source
        else
            log_info "[DRY RUN] Would install TA-Lib from source"
        fi
    fi
}

install_ta_lib_from_source() {
    local ta_lib_version="0.4.0"
    local ta_lib_url="https://prdownloads.sourceforge.net/ta-lib/ta-lib-${ta_lib_version}-src.tar.gz"
    local temp_dir="/tmp/ta-lib-build"
    
    log_info "Building TA-Lib from source (version $ta_lib_version)"
    
    # Create temporary build directory
    mkdir -p "$temp_dir"
    cd "$temp_dir"
    
    # Download and extract TA-Lib
    if ! wget -O "ta-lib-${ta_lib_version}-src.tar.gz" "$ta_lib_url"; then
        log_error "Failed to download TA-Lib source"
        return 1
    fi
    
    tar -xzf "ta-lib-${ta_lib_version}-src.tar.gz"
    cd "ta-lib"
    
    # Configure, build, and install
    if ./configure --prefix=/usr/local && make && sudo make install; then
        log_success "TA-Lib installed successfully from source"
        
        # Update library cache
        if command -v ldconfig >/dev/null 2>&1; then
            sudo ldconfig
        fi
    else
        log_error "Failed to build TA-Lib from source"
        return 1
    fi
    
    # Cleanup
    cd /
    rm -rf "$temp_dir"
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

validate_python_installation() {
    log_step "Validating Python installation"
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
        log_info "Python version: $python_version"
        
        # Check if Python 3.12+
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
            log_success "Python version is compatible (>= 3.12)"
        else
            log_warn "Python version may not be fully compatible (< 3.12)"
        fi
    else
        log_error "Python 3 not found"
        return 1
    fi
    
    # Check pip
    if command -v pip3 >/dev/null 2>&1; then
        local pip_version=$(pip3 --version | cut -d' ' -f2)
        log_info "pip version: $pip_version"
    else
        log_error "pip3 not found"
        return 1
    fi
}

validate_build_tools() {
    log_step "Validating build tools"
    
    local tools=("gcc" "g++" "make")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            local version=$("$tool" --version | head -n1)
            log_info "$tool: $version"
        else
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warn "Missing build tools: ${missing_tools[*]}"
        return 1
    else
        log_success "All build tools are available"
        return 0
    fi
}

validate_libraries() {
    log_step "Validating system libraries"
    
    local libraries=("libffi" "libssl" "libxml2" "libxslt")
    local missing_libs=()
    
    for lib in "${libraries[@]}"; do
        if pkg-config --exists "$lib" 2>/dev/null; then
            local version=$(pkg-config --modversion "$lib")
            log_info "$lib: $version"
        else
            # Try alternative ways to check
            case "$lib" in
                libffi)
                    if [[ -f "/usr/include/ffi.h" ]] || [[ -f "/usr/local/include/ffi.h" ]]; then
                        log_info "$lib: found (header)"
                    else
                        missing_libs+=("$lib")
                    fi
                    ;;
                libssl)
                    if command -v openssl >/dev/null 2>&1; then
                        local ssl_version=$(openssl version)
                        log_info "OpenSSL: $ssl_version"
                    else
                        missing_libs+=("$lib")
                    fi
                    ;;
                *)
                    missing_libs+=("$lib")
                    ;;
            esac
        fi
    done
    
    if [[ ${#missing_libs[@]} -gt 0 ]]; then
        log_warn "Missing or undetectable libraries: ${missing_libs[*]}"
        return 1
    else
        log_success "All required libraries are available"
        return 0
    fi
}

# ============================================================================
# LOCK FILE MANAGEMENT
# ============================================================================

acquire_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid=$(cat "$LOCK_FILE")
        if kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Another instance is running (PID: $lock_pid)"
            return 1
        else
            log_warn "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    log_debug "Acquired lock (PID: $$)"
}

release_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        rm -f "$LOCK_FILE"
        log_debug "Released lock"
    fi
}

# ============================================================================
# MAIN INSTALLATION ORCHESTRATOR
# ============================================================================

install_all_dependencies() {
    local os_info
    local os_type
    local os_version
    
    # Detect OS
    os_info=$(detect_os)
    os_type=$(echo "$os_info" | cut -d':' -f1)
    os_version=$(echo "$os_info" | cut -d':' -f2)
    
    log_info "Detected OS: $os_type (version: $os_version)"
    
    # Check privileges
    if ! check_privileges; then
        return 1
    fi
    
    # Fix any existing WSL-specific issues before proceeding
    if [[ "${IS_WSL:-false}" == "true" ]]; then
        fix_postfix_wsl
    fi
    
    # Update package cache
    update_package_cache "$os_type"
    
    # Install dependencies in order of importance
    local install_functions=(
        "install_python_development"
        "install_build_tools"
        "install_crypto_libraries"
        "install_xml_libraries"
        "install_database_libraries"
        "install_kafka_libraries"
        "install_ta_lib"
    )
    
    local failed_functions=()
    
    for func in "${install_functions[@]}"; do
        log_info "Executing: $func"
        if ! "$func" "$os_type"; then
            log_error "Failed: $func"
            failed_functions+=("$func")
            
            if [[ "$FORCE_INSTALL" == "false" ]]; then
                log_error "Stopping installation due to failure. Use --force to continue."
                return 1
            else
                log_warn "Continuing despite failure (--force enabled)"
            fi
        else
            log_success "Completed: $func"
        fi
    done
    
    # Validation
    log_step "Validating installation"
    validate_python_installation
    validate_build_tools
    validate_libraries
    
    # Report results
    if [[ ${#failed_functions[@]} -eq 0 ]]; then
        log_success "All system dependencies installed successfully!"
        log_info "You can now proceed with Python package installation:"
        log_info "  python3 install_dependencies.py"
    else
        log_warn "Installation completed with ${#failed_functions[@]} failures:"
        for func in "${failed_functions[@]}"; do
            log_warn "  - $func"
        done
        log_info "Check the log file for details: $LOG_FILE"
    fi
}

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

show_help() {
    cat << EOF
$SCRIPT_NAME v$SCRIPT_VERSION

Install system dependencies for the Investment Analysis Platform.

USAGE:
    $SCRIPT_NAME [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose          Enable verbose output
    -n, --dry-run          Show what would be installed without installing
    -f, --force            Continue installation despite failures
    -r, --retries NUM      Maximum number of retries (default: $MAX_RETRIES)
    -t, --timeout SEC      Installation timeout per command (default: $INSTALL_TIMEOUT)
    -j, --jobs NUM         Number of parallel jobs (default: $PARALLEL_JOBS)
    --log-file FILE        Log file path (default: $LOG_FILE)

EXAMPLES:
    $SCRIPT_NAME                    # Install all dependencies
    $SCRIPT_NAME --dry-run          # Show what would be installed
    $SCRIPT_NAME --verbose --force  # Verbose output, continue on failures
    $SCRIPT_NAME --retries 5 --timeout 600  # More retries and longer timeout

SUPPORTED PLATFORMS:
    - Ubuntu 18.04+
    - Debian 9+
    - CentOS 7+
    - Fedora 30+
    - Arch Linux
    - macOS 10.15+

REQUIREMENTS:
    - Root privileges or sudo access (except macOS)
    - Internet connection
    - Package manager (apt/yum/dnf/pacman/brew)

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE_INSTALL=true
                shift
                ;;
            -r|--retries)
                if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                    MAX_RETRIES="$2"
                    shift 2
                else
                    log_error "Invalid retries value: $2"
                    exit 1
                fi
                ;;
            -t|--timeout)
                if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                    INSTALL_TIMEOUT="$2"
                    shift 2
                else
                    log_error "Invalid timeout value: $2"
                    exit 1
                fi
                ;;
            -j|--jobs)
                if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                    PARALLEL_JOBS="$2"
                    shift 2
                else
                    log_error "Invalid jobs value: $2"
                    exit 1
                fi
                ;;
            --log-file)
                if [[ -n "$2" ]]; then
                    LOG_FILE="$2"
                    shift 2
                else
                    log_error "Log file path required"
                    exit 1
                fi
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# CLEANUP AND EXIT HANDLERS
# ============================================================================

cleanup() {
    local exit_code=$?
    log_debug "Cleanup function called with exit code: $exit_code"
    
    # Release lock
    release_lock
    
    # Log completion
    if [[ $exit_code -eq 0 ]]; then
        log_success "Script completed successfully"
    else
        log_error "Script completed with errors (exit code: $exit_code)"
    fi
    
    log_info "Log file: $LOG_FILE"
    log_info "End time: $(date)"
    
    exit $exit_code
}

# Set up signal handlers
trap cleanup EXIT
trap 'log_warn "Received SIGINT, cleaning up..."; exit 130' INT
trap 'log_warn "Received SIGTERM, cleaning up..."; exit 143' TERM

# ============================================================================
# MAIN FUNCTION
# ============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Setup logging
    setup_logging
    
    # Acquire lock
    if ! acquire_lock; then
        exit 1
    fi
    
    # Show configuration
    log_info "Configuration:"
    log_info "  Verbose: $VERBOSE"
    log_info "  Dry run: $DRY_RUN"
    log_info "  Force install: $FORCE_INSTALL"
    log_info "  Max retries: $MAX_RETRIES"
    log_info "  Timeout: ${INSTALL_TIMEOUT}s"
    log_info "  Parallel jobs: $PARALLEL_JOBS"
    log_info "  Log file: $LOG_FILE"
    
    # Check if dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE: No packages will be installed"
    fi
    
    # Run main installation
    if install_all_dependencies; then
        log_success "System dependencies installation completed successfully!"
        exit 0
    else
        log_error "System dependencies installation failed"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
