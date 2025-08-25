#!/bin/bash
# Investment Platform Setup Script - WSL/Windows Optimized
# Enhanced setup with comprehensive WSL compatibility and error handling

# Script configuration
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Default configuration
VERBOSE=0
QUIET=0
USE_VENV=1
USE_SYSTEM=0
CLEANUP_ON_FAILURE=1
RETRY_COUNT=3
RETRY_DELAY=2
PYTHON_MIN_VERSION="3.10"
PYTHON_MAX_WARNING_VERSION="3.12"
LOG_FILE="setup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking
TOTAL_STEPS=12
CURRENT_STEP=0

# WSL detection
IS_WSL=false
if [[ -n "${WSL_DISTRO_NAME:-}" ]] || [[ -n "${WSL_INTEROP:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
fi

# Cleanup tracking for failure recovery
CLEANUP_TASKS=()

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -v, --verbose           Enable verbose output
    -q, --quiet            Enable quiet mode (minimal output)
    -s, --system           Use system-wide Python installation (no venv)
    --venv                 Use virtual environment (default)
    --no-cleanup           Don't cleanup on failure
    --retry-count N        Number of retries for failed operations (default: 3)
    --retry-delay N        Delay between retries in seconds (default: 2)
    -h, --help             Show this help message

EXAMPLES:
    $0                     # Standard setup with virtual environment
    $0 --verbose           # Verbose output
    $0 --system --quiet    # System-wide installation, minimal output
    $0 --retry-count 5     # Retry failed operations 5 times

WSL FEATURES:
    - Automatic WSL environment detection
    - Fixed postfix configuration issues
    - Optimized package installation for WSL
    - Docker integration validation

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            QUIET=0
            shift
            ;;
        -q|--quiet)
            QUIET=1
            VERBOSE=0
            shift
            ;;
        -s|--system)
            USE_SYSTEM=1
            USE_VENV=0
            shift
            ;;
        --venv)
            USE_VENV=1
            USE_SYSTEM=0
            shift
            ;;
        --no-cleanup)
            CLEANUP_ON_FAILURE=0
            shift
            ;;
        --retry-count)
            RETRY_COUNT="$2"
            shift 2
            ;;
        --retry-delay)
            RETRY_DELAY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Log to console based on verbosity
    case $level in
        "ERROR")
            [[ $QUIET -eq 0 ]] && echo -e "${RED}❌ ERROR: $message${NC}" >&2
            ;;
        "WARN")
            [[ $QUIET -eq 0 ]] && echo -e "${YELLOW}⚠️ WARNING: $message${NC}" >&2
            ;;
        "INFO")
            [[ $QUIET -eq 0 ]] && echo -e "${GREEN}✅ $message${NC}"
            ;;
        "DEBUG")
            [[ $VERBOSE -eq 1 ]] && echo -e "${BLUE}🔍 DEBUG: $message${NC}"
            ;;
        "PROGRESS")
            [[ $QUIET -eq 0 ]] && echo -e "${CYAN}🔄 $message${NC}"
            ;;
        "WSL")
            [[ $QUIET -eq 0 ]] && echo -e "${PURPLE}🪟 WSL: $message${NC}"
            ;;
    esac
}

# Progress indicator
show_progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local percentage=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    [[ $QUIET -eq 0 ]] && echo -e "${PURPLE}[$CURRENT_STEP/$TOTAL_STEPS] ($percentage%) $1${NC}"
    log "PROGRESS" "Step $CURRENT_STEP/$TOTAL_STEPS: $1"
}

# Add cleanup task
add_cleanup_task() {
    CLEANUP_TASKS+=("$1")
    log "DEBUG" "Added cleanup task: $1"
}

# Execute cleanup tasks
cleanup() {
    if [[ $CLEANUP_ON_FAILURE -eq 1 && ${#CLEANUP_TASKS[@]} -gt 0 ]]; then
        log "INFO" "Executing cleanup tasks..."
        for task in "${CLEANUP_TASKS[@]}"; do
            log "DEBUG" "Executing cleanup: $task"
            eval "$task" 2>/dev/null || true
        done
    fi
}

# Error handler
error_handler() {
    local line_number=$1
    local error_code=$2
    log "ERROR" "Script failed at line $line_number with exit code $error_code"
    cleanup
    exit $error_code
}

# Set error trap
trap 'error_handler ${LINENO} $?' ERR

# Retry function
retry() {
    local cmd="$1"
    local description="$2"
    local count=0
    
    while [[ $count -lt $RETRY_COUNT ]]; do
        if eval "$cmd"; then
            log "DEBUG" "Command succeeded: $description"
            return 0
        else
            count=$((count + 1))
            if [[ $count -lt $RETRY_COUNT ]]; then
                log "WARN" "Attempt $count failed for: $description. Retrying in ${RETRY_DELAY}s..."
                sleep $RETRY_DELAY
            else
                log "ERROR" "All $RETRY_COUNT attempts failed for: $description"
                return 1
            fi
        fi
    done
}

# Version comparison function
version_compare() {
    local version1="$1"
    local version2="$2"
    local operator="$3"
    
    # Convert versions to comparable format
    local v1=$(echo "$version1" | sed 's/[^0-9.]//g' | awk -F. '{printf "%d%03d%03d", $1,$2,$3}')
    local v2=$(echo "$version2" | sed 's/[^0-9.]//g' | awk -F. '{printf "%d%03d%03d", $1,$2,$3}')
    
    case $operator in
        ">=") [[ $v1 -ge $v2 ]] ;;
        ">") [[ $v1 -gt $v2 ]] ;;
        "<=") [[ $v1 -le $v2 ]] ;;
        "<") [[ $v1 -lt $v2 ]] ;;
        "==") [[ $v1 -eq $v2 ]] ;;
        *) return 1 ;;
    esac
}

# WSL-specific fixes
fix_wsl_environment() {
    if [[ "$IS_WSL" == "true" ]]; then
        log "WSL" "Applying WSL-specific environment fixes..."
        
        # Fix line endings issues
        if command -v dos2unix >/dev/null 2>&1; then
            log "DEBUG" "Converting line endings for shell scripts"
            find . -name "*.sh" -type f -exec dos2unix {} \; 2>/dev/null || true
        fi
        
        # Set proper permissions for scripts
        find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
        
        # Fix any broken package states
        if dpkg -l | grep -q "^iF"; then
            log "WSL" "Fixing broken package installations..."
            sudo dpkg --configure -a 2>/dev/null || true
        fi
        
        # Set environment variables for non-interactive mode
        export DEBIAN_FRONTEND=noninteractive
        export NEEDRESTART_MODE=a
        
        log "WSL" "WSL environment fixes applied"
    fi
}

echo -e "${GREEN}🚀 Investment Platform Setup (WSL Optimized)${NC}"
echo -e "${GREEN}================================================${NC}"
if [[ "$IS_WSL" == "true" ]]; then
    echo -e "${PURPLE}🪟 WSL Environment Detected: ${WSL_DISTRO_NAME:-Unknown}${NC}"
fi
echo ""
log "INFO" "Setup started with options: verbose=$VERBOSE, quiet=$QUIET, use_venv=$USE_VENV, retry_count=$RETRY_COUNT, wsl=$IS_WSL"

# WSL environment fixes
show_progress "Checking and fixing WSL environment"
fix_wsl_environment

# System compatibility checks
show_progress "Checking system compatibility"

# Check operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    log "DEBUG" "Detected Linux operating system"
    if [[ "$IS_WSL" == "true" ]]; then
        log "WSL" "Running under Windows Subsystem for Linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    log "DEBUG" "Detected macOS operating system"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    log "DEBUG" "Detected Windows operating system"
else
    log "WARN" "Unknown operating system: $OSTYPE. Proceeding with caution."
    OS="unknown"
fi

# Check for required system tools
check_system_dependency() {
    local tool="$1"
    local package="$2"
    local description="$3"
    
    if ! command -v "$tool" &> /dev/null; then
        log "ERROR" "Required tool '$tool' not found. $description"
        if [[ -n "$package" ]]; then
            case $OS in
                "linux")
                    if [[ "$IS_WSL" == "true" ]]; then
                        log "ERROR" "Install with: sudo apt-get install $package"
                    else
                        log "ERROR" "Install with: sudo apt-get install $package (Ubuntu/Debian) or sudo yum install $package (RHEL/CentOS)"
                    fi
                    ;;
                "macos")
                    log "ERROR" "Install with: brew install $package"
                    ;;
                "windows")
                    log "ERROR" "Please install $tool manually or use Windows Subsystem for Linux (WSL)"
                    ;;
            esac
        fi
        return 1
    fi
    log "DEBUG" "Found required tool: $tool"
    return 0
}

# Check essential system dependencies
log "DEBUG" "Checking essential system dependencies"
check_system_dependency "curl" "curl" "Required for downloading packages"
check_system_dependency "git" "git" "Required for version control"

# Build tools check with WSL considerations
if [[ "$OS" == "linux" ]]; then
    if ! command -v gcc &> /dev/null || ! command -v make &> /dev/null; then
        log "WARN" "Build tools not found. Some Python packages may fail to install."
        if [[ "$IS_WSL" == "true" ]]; then
            log "WARN" "Install with: sudo apt-get install build-essential python3-dev"
        else
            log "WARN" "Install with: sudo apt-get install build-essential python3-dev (Ubuntu/Debian)"
            log "WARN" "Or: sudo yum groupinstall 'Development Tools' python3-devel (RHEL/CentOS)"
        fi
    fi
fi

# Check Python installation and version
show_progress "Checking Python installation"

PYTHON_CMD=""
if [[ $USE_SYSTEM -eq 1 ]]; then
    # Try different Python commands for system installation
    for cmd in python3 python python3.12 python3.11 python3.10; do
        if command -v "$cmd" &> /dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    done
else
    # For virtual environment, prefer python3
    for cmd in python3 python python3.12 python3.11 python3.10; do
        if command -v "$cmd" &> /dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    done
fi

if [[ -z "$PYTHON_CMD" ]]; then
    log "ERROR" "Python not found. Please install Python $PYTHON_MIN_VERSION or later."
    case $OS in
        "linux")
            if [[ "$IS_WSL" == "true" ]]; then
                log "ERROR" "Install with: sudo apt-get install python3 python3-pip python3-venv python3-full"
            else
                log "ERROR" "Install with: sudo apt-get install python3 python3-pip python3-venv (Ubuntu/Debian)"
                log "ERROR" "Or: sudo yum install python3 python3-pip (RHEL/CentOS)"
            fi
            ;;
        "macos")
            log "ERROR" "Install with: brew install python3 or download from https://python.org"
            ;;
        "windows")
            log "ERROR" "Download from https://python.org or use Windows Store"
            ;;
    esac
    exit 1
fi

# Get Python version
PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
log "INFO" "Found Python $PYTHON_VERSION at $(command -v "$PYTHON_CMD")"

# Check Python version compatibility
if [[ "$PYTHON_VERSION" != "unknown" ]]; then
    if ! version_compare "$PYTHON_VERSION" "$PYTHON_MIN_VERSION" ">="; then
        log "ERROR" "Python $PYTHON_VERSION is too old. Minimum required: $PYTHON_MIN_VERSION"
        log "ERROR" "Please upgrade Python or install a newer version."
        exit 1
    fi
    
    # Warn about newer versions
    if version_compare "$PYTHON_VERSION" "$PYTHON_MAX_WARNING_VERSION" ">"; then
        log "WARN" "Python $PYTHON_VERSION is newer than tested version $PYTHON_MAX_WARNING_VERSION"
        log "WARN" "Some packages may not be compatible. Consider using Python $PYTHON_MAX_WARNING_VERSION or earlier."
    fi
else
    log "WARN" "Could not determine Python version. Proceeding with caution."
fi

# Check pip availability
if [[ $USE_SYSTEM -eq 1 ]]; then
    PIP_CMD="$PYTHON_CMD -m pip"
else
    PIP_CMD="pip"
fi

if ! "$PYTHON_CMD" -m pip --version &> /dev/null; then
    log "ERROR" "pip not found. Please install pip for Python $PYTHON_VERSION"
    case $OS in
        "linux")
            if [[ "$IS_WSL" == "true" ]]; then
                log "ERROR" "Install with: sudo apt-get install python3-pip"
            else
                log "ERROR" "Install with: sudo apt-get install python3-pip (Ubuntu/Debian)"
            fi
            ;;
        "macos")
            log "ERROR" "pip should be included with Python. Try: python3 -m ensurepip"
            ;;
        "windows")
            log "ERROR" "pip should be included with Python. Try: python -m ensurepip"
            ;;
    esac
    exit 1
fi

log "INFO" "Python environment validation complete"

# Docker compatibility check for WSL
show_progress "Checking Docker compatibility"
if [[ "$IS_WSL" == "true" ]]; then
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            log "WSL" "Docker is running and accessible from WSL"
        else
            log "WARN" "Docker found but not running. Make sure Docker Desktop is started."
            log "WSL" "In Docker Desktop, enable 'Use the WSL 2 based engine' in Settings"
        fi
    else
        log "WARN" "Docker not found. Install Docker Desktop for Windows."
        log "WSL" "Enable WSL integration in Docker Desktop settings."
    fi
fi

# Check for .env file
show_progress "Setting up environment configuration"

if [ ! -f .env ]; then
    log "INFO" "Creating .env file from template"
    if [ -f .env.template ]; then
        if cp .env.template .env; then
            log "INFO" ".env file created successfully"
            add_cleanup_task "[[ -f .env && ! -f .env.backup ]] && rm -f .env"
        else
            log "ERROR" "Failed to create .env file"
            exit 1
        fi
    else
        log "ERROR" ".env.template not found"
        log "ERROR" "Please ensure .env.template exists in the current directory"
        exit 1
    fi
else
    log "INFO" ".env file already exists"
fi

# Generate secure passwords if not set
generate_secure_passwords() {
    log "INFO" "Generating secure passwords"
    
    # Create backup of .env file
    if cp .env .env.backup; then
        log "DEBUG" "Created backup of .env file"
        add_cleanup_task "[[ -f .env.backup ]] && mv .env.backup .env"
    else
        log "WARN" "Could not create backup of .env file"
    fi
    
    # Generate passwords using openssl or Python fallback
    local db_password redis_password secret_key jwt_secret fernet_key
    
    if command -v openssl &> /dev/null; then
        db_password=$(openssl rand -base64 32 | tr -d '\n')
        redis_password=$(openssl rand -base64 32 | tr -d '\n')
        secret_key=$(openssl rand -hex 32)
        jwt_secret=$(openssl rand -hex 32)
        fernet_key=$(openssl rand -base64 32 | tr -d '\n')
    else
        log "WARN" "openssl not available, using Python for password generation"
        db_password=$("$PYTHON_CMD" -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32)))")
        redis_password=$("$PYTHON_CMD" -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32)))")
        secret_key=$("$PYTHON_CMD" -c "import secrets; print(secrets.token_hex(32))")
        jwt_secret=$("$PYTHON_CMD" -c "import secrets; print(secrets.token_hex(32))")
        fernet_key=$("$PYTHON_CMD" -c "import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())")
    fi
    
    # Replace passwords in .env file using sed (WSL/Linux compatible)
    local replacements=(
        "s/DB_PASSWORD=CHANGE_ME/DB_PASSWORD=$db_password/g"
        "s/REDIS_PASSWORD=CHANGE_ME/REDIS_PASSWORD=$redis_password/g"
        "s/SECRET_KEY=CHANGE_ME/SECRET_KEY=$secret_key/g"
        "s/JWT_SECRET_KEY=CHANGE_ME/JWT_SECRET_KEY=$jwt_secret/g"
        "s/FERNET_KEY=CHANGE_ME/FERNET_KEY=$fernet_key/g"
        "s/secure_database_password/$db_password/g"
        "s/secure_redis_password/$redis_password/g"
        "s/your_secret_key_here/$secret_key/g"
        "s/your_jwt_secret_here/$jwt_secret/g"
        "s/your_fernet_key_here/$fernet_key/g"
    )
    
    for replacement in "${replacements[@]}"; do
        if sed -i "$replacement" .env; then
            log "DEBUG" "Applied password replacement"
        else
            log "WARN" "Failed to apply password replacement: $replacement"
        fi
    done
    
    log "INFO" "Secure passwords generated successfully"
}

# Check if passwords need to be generated
if grep -q "CHANGE_ME\|your_.*_key_here\|secure_.*_password" .env; then
    generate_secure_passwords
else
    log "INFO" "Secure passwords already configured"
fi

# Install system dependencies
show_progress "Installing system dependencies"

if [ -f "./install_system_deps.sh" ]; then
    log "INFO" "Running system dependencies installation script"
    chmod +x ./install_system_deps.sh
    
    # Build command with WSL-friendly options
    local sys_cmd="./install_system_deps.sh"
    [[ $VERBOSE -eq 1 ]] && sys_cmd="$sys_cmd --verbose"
    [[ $QUIET -eq 1 ]] && sys_cmd="$sys_cmd --quiet" # Note: script doesn't have quiet, but verbose handles it
    
    if retry "$sys_cmd" "System dependencies installation"; then
        log "INFO" "System dependencies installed successfully"
    else
        log "WARN" "System dependencies installation had issues, but continuing"
        log "WARN" "You may need to install some packages manually"
    fi
else
    log "WARN" "System dependencies script not found, skipping"
fi

# Setup Python environment
show_progress "Setting up Python environment"

setup_python_environment() {
    if [[ $USE_VENV -eq 1 ]]; then
        log "INFO" "Setting up Python virtual environment"
        
        if [ ! -d "venv" ]; then
            log "DEBUG" "Creating new virtual environment"
            if retry "'$PYTHON_CMD' -m venv venv" "Create virtual environment"; then
                log "INFO" "Virtual environment created successfully"
                add_cleanup_task "[[ -d venv ]] && rm -rf venv"
            else
                log "ERROR" "Failed to create virtual environment"
                if [[ "$IS_WSL" == "true" ]]; then
                    log "ERROR" "Try installing: sudo apt-get install python3-venv python3-full"
                else
                    log "ERROR" "Try installing python3-venv: sudo apt-get install python3-venv (Ubuntu/Debian)"
                fi
                exit 1
            fi
        else
            log "INFO" "Virtual environment already exists"
        fi
        
        # Activate virtual environment with proper path handling for WSL
        log "DEBUG" "Activating virtual environment"
        if source venv/bin/activate; then
            log "INFO" "Virtual environment activated"
            PYTHON_CMD="python"
            PIP_CMD="pip"
        else
            log "ERROR" "Failed to activate virtual environment"
            exit 1
        fi
    else
        log "INFO" "Using system-wide Python installation"
        log "WARN" "System-wide installation may require administrator privileges"
    fi
}

setup_python_environment

# Install Python dependencies
show_progress "Installing Python dependencies"

install_python_dependencies() {
    if [ ! -f requirements.txt ]; then
        log "WARN" "requirements.txt not found, skipping Python dependency installation"
        return 0
    fi
    
    log "INFO" "Installing Python dependencies from requirements.txt"
    
    # Upgrade pip first
    local pip_args=""
    [[ $QUIET -eq 1 ]] && pip_args="--quiet"
    [[ $VERBOSE -eq 1 ]] && pip_args="--verbose"
    
    log "DEBUG" "Upgrading pip"
    if retry "$PIP_CMD install --upgrade pip $pip_args" "Upgrade pip"; then
        log "INFO" "pip upgraded successfully"
    else
        log "WARN" "Failed to upgrade pip, continuing with current version"
    fi
    
    # Install wheel for better package compilation
    log "DEBUG" "Installing wheel"
    if retry "$PIP_CMD install wheel setuptools $pip_args" "Install wheel and setuptools"; then
        log "DEBUG" "wheel and setuptools installed successfully"
    else
        log "WARN" "Failed to install wheel/setuptools, some packages may take longer to compile"
    fi
    
    # Count total packages for progress tracking
    local total_packages=$(grep -c '^[^#]' requirements.txt || echo "unknown")
    log "INFO" "Installing $total_packages packages from requirements.txt"
    
    # Install dependencies with retry logic and WSL-optimized settings
    local install_args="$pip_args --no-cache-dir"
    [[ $USE_SYSTEM -eq 1 ]] && install_args="$install_args --user"
    
    # For WSL, add extra options to help with compilation
    if [[ "$IS_WSL" == "true" ]]; then
        install_args="$install_args --prefer-binary --no-warn-script-location"
    fi
    
    # Try to install all packages at once first
    log "DEBUG" "Attempting bulk installation with WSL optimizations"
    if $PIP_CMD install -r requirements.txt $install_args; then
        log "INFO" "All Python dependencies installed successfully"
        return 0
    fi
    
    log "WARN" "Bulk installation failed, trying individual package installation"
    
    # Install packages individually for better error reporting
    local failed_packages=()
    local line_num=0
    
    while IFS= read -r line; do
        line_num=$((line_num + 1))
        
        # Skip comments and empty lines
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        
        # Extract package name
        local package=$(echo "$line" | sed 's/[<>=!].*//' | tr -d ' ')
        
        log "DEBUG" "Installing package: $package"
        if retry "$PIP_CMD install '$line' $install_args" "Install $package"; then
            log "DEBUG" "Successfully installed: $package"
        else
            log "ERROR" "Failed to install: $package"
            failed_packages+=("$package")
        fi
    done < requirements.txt
    
    # Report results
    if [[ ${#failed_packages[@]} -eq 0 ]]; then
        log "INFO" "All Python dependencies installed successfully"
    else
        log "WARN" "${#failed_packages[@]} packages failed to install: ${failed_packages[*]}"
        log "WARN" "The application may not function correctly with missing dependencies"
        
        # Provide WSL-specific suggestions for common failure cases
        for package in "${failed_packages[@]}"; do
            case $package in
                "ta-lib")
                    log "WARN" "ta-lib requires TA-Lib C library. For WSL:"
                    log "WARN" "  sudo apt-get install libta-lib-dev"
                    log "WARN" "  Or compile from source if package not available"
                    ;;
                "psycopg2-binary")
                    log "WARN" "psycopg2 requires PostgreSQL development headers:"
                    log "WARN" "  sudo apt-get install libpq-dev"
                    ;;
                "confluent-kafka")
                    log "WARN" "confluent-kafka requires librdkafka:"
                    log "WARN" "  sudo apt-get install librdkafka-dev"
                    ;;
            esac
        done
        
        return 1
    fi
}

install_python_dependencies

# Install frontend dependencies
show_progress "Installing frontend dependencies"

install_frontend_dependencies() {
    if [ ! -d frontend/web ]; then
        log "INFO" "Frontend directory not found, skipping frontend dependency installation"
        return 0
    fi
    
    if ! command -v npm &> /dev/null; then
        log "WARN" "npm not found. Skipping frontend dependency installation."
        log "WARN" "Frontend dependencies will be installed in the Docker container."
        if [[ "$IS_WSL" == "true" ]]; then
            log "WSL" "Install Node.js for WSL: curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -"
            log "WSL" "Then: sudo apt-get install -y nodejs"
        fi
        return 0
    fi
    
    # Check Node.js version
    local node_version=$(node --version 2>/dev/null | sed 's/v//' || echo "unknown")
    local npm_version=$(npm --version 2>/dev/null || echo "unknown")
    log "INFO" "Found Node.js $node_version and npm $npm_version"
    
    # Check if package.json exists
    if [ ! -f frontend/web/package.json ]; then
        log "WARN" "package.json not found in frontend/web directory"
        return 0
    fi
    
    log "INFO" "Installing frontend dependencies"
    
    # Navigate to frontend directory
    local original_dir=$(pwd)
    cd frontend/web || {
        log "ERROR" "Failed to change to frontend/web directory"
        return 1
    }
    
    # Set npm install arguments
    local npm_args=""
    [[ $QUIET -eq 1 ]] && npm_args="--silent"
    [[ $VERBOSE -eq 1 ]] && npm_args="--verbose"
    
    # Clean npm cache if previous installation failed
    if [ -d node_modules ] && [ ! -f node_modules/.successfully_installed ]; then
        log "DEBUG" "Cleaning previous incomplete installation"
        rm -rf node_modules package-lock.json
    fi
    
    # Install dependencies with retry
    if retry "npm install $npm_args" "Install frontend dependencies"; then
        log "INFO" "Frontend dependencies installed successfully"
        touch node_modules/.successfully_installed
    else
        log "WARN" "Failed to install frontend dependencies"
        log "WARN" "Try running 'npm install' manually in the frontend/web directory"
        cd "$original_dir"
        return 1
    fi
    
    # Return to original directory
    cd "$original_dir" || {
        log "ERROR" "Failed to return to original directory"
        return 1
    }
    
    log "INFO" "Frontend setup completed"
}

install_frontend_dependencies

# Create necessary directories
show_progress "Creating necessary directories"

create_directories() {
    local directories=("logs" "data" "models/trained" "archive" "backup" "tmp" "cache")
    
    log "INFO" "Creating necessary directories"
    
    for dir in "${directories[@]}"; do
        if mkdir -p "$dir"; then
            log "DEBUG" "Created directory: $dir"
        else
            log "ERROR" "Failed to create directory: $dir"
            return 1
        fi
    done
    
    # Set appropriate permissions (WSL-friendly)
    chmod 755 logs data models archive backup tmp cache 2>/dev/null || true
    chmod 755 models/trained 2>/dev/null || true
    
    log "INFO" "All directories created successfully"
}

create_directories

# Initialize database services
show_progress "Initializing database services"

initialize_database() {
    # Check for Docker and Docker Compose
    local docker_available=0
    local docker_compose_cmd=""
    
    if command -v docker &> /dev/null; then
        log "DEBUG" "Docker found"
        
        # Check Docker daemon status
        if docker info &> /dev/null; then
            docker_available=1
            log "INFO" "Docker is running"
        else
            log "WARN" "Docker daemon is not running."
            if [[ "$IS_WSL" == "true" ]]; then
                log "WSL" "Make sure Docker Desktop is running on Windows"
                log "WSL" "Enable WSL integration in Docker Desktop settings"
            else
                log "WARN" "Start Docker with: sudo systemctl start docker"
            fi
            docker_available=0
        fi
    else
        log "WARN" "Docker not found"
        if [[ "$IS_WSL" == "true" ]]; then
            log "WSL" "Install Docker Desktop for Windows and enable WSL integration"
        fi
        docker_available=0
    fi
    
    # Check for Docker Compose (v2 or v1)
    if [[ $docker_available -eq 1 ]]; then
        if docker compose version &> /dev/null; then
            docker_compose_cmd="docker compose"
            log "DEBUG" "Docker Compose v2 found"
        elif command -v docker-compose &> /dev/null; then
            docker_compose_cmd="docker-compose"
            log "DEBUG" "Docker Compose v1 found"
        else
            log "WARN" "Docker Compose not found"
            docker_available=0
        fi
    fi
    
    if [[ $docker_available -eq 0 ]]; then
        log "WARN" "Docker/Docker Compose not available. Skipping database initialization."
        log "WARN" "Database services will need to be started manually."
        return 0
    fi
    
    # Check for docker-compose.yml
    if [ ! -f docker-compose.yml ]; then
        log "WARN" "docker-compose.yml not found. Cannot start database services."
        return 0
    fi
    
    log "INFO" "Starting database services with Docker"
    
    # Start database services
    local services=("postgres" "redis")
    for service in "${services[@]}"; do
        log "DEBUG" "Starting $service service"
        if retry "$docker_compose_cmd up -d $service" "Start $service service"; then
            log "INFO" "$service service started successfully"
        else
            log "WARN" "Failed to start $service service"
        fi
    done
    
    # Wait for services to be ready
    log "DEBUG" "Waiting for database services to be ready"
    sleep 5
    
    # Verify PostgreSQL is ready
    local postgres_ready=0
    for i in {1..30}; do
        if $docker_compose_cmd exec -T postgres pg_isready -U postgres &> /dev/null; then
            postgres_ready=1
            break
        fi
        log "DEBUG" "Waiting for PostgreSQL to be ready... ($i/30)"
        sleep 2
    done
    
    if [[ $postgres_ready -eq 1 ]]; then
        log "INFO" "PostgreSQL is ready"
        
        # Create database if it doesn't exist
        log "DEBUG" "Creating investment_db database"
        if $docker_compose_cmd exec -T postgres psql -U postgres -c "CREATE DATABASE investment_db;" &> /dev/null; then
            log "INFO" "Database 'investment_db' created successfully"
        else
            log "INFO" "Database 'investment_db' already exists or creation failed"
        fi
    else
        log "WARN" "PostgreSQL failed to start within timeout period"
    fi
    
    # Verify Redis is ready
    if $docker_compose_cmd exec -T redis redis-cli ping &> /dev/null; then
        log "INFO" "Redis is ready"
    else
        log "WARN" "Redis may not be ready"
    fi
    
    log "INFO" "Database services initialization completed"
    log "INFO" "Note: Run database migrations after starting the backend service"
}

initialize_database

# Validate setup
show_progress "Validating setup"

validate_setup() {
    log "INFO" "Performing setup validation"
    
    local validation_errors=0
    
    # Check .env file
    if [ -f .env ]; then
        log "DEBUG" "✓ .env file exists"
        
        # Check for placeholder values
        if grep -q "your_.*_key_here\|CHANGE_ME" .env; then
            log "WARN" "⚠ Some API keys still contain placeholder values"
            validation_errors=$((validation_errors + 1))
        fi
    else
        log "ERROR" "✗ .env file missing"
        validation_errors=$((validation_errors + 1))
    fi
    
    # Check Python environment
    if [[ $USE_VENV -eq 1 ]]; then
        if [ -d venv ]; then
            log "DEBUG" "✓ Virtual environment exists"
        else
            log "ERROR" "✗ Virtual environment missing"
            validation_errors=$((validation_errors + 1))
        fi
    fi
    
    # Check key directories
    local required_dirs=("logs" "data" "models/trained")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            log "DEBUG" "✓ Directory exists: $dir"
        else
            log "ERROR" "✗ Directory missing: $dir"
            validation_errors=$((validation_errors + 1))
        fi
    done
    
    # Test Python imports for critical packages
    if [[ $USE_VENV -eq 1 ]] && [ -d venv ]; then
        source venv/bin/activate
    fi
    
    local critical_packages=("fastapi" "pandas" "numpy" "sqlalchemy")
    for package in "${critical_packages[@]}"; do
        if "$PYTHON_CMD" -c "import $package" &> /dev/null; then
            log "DEBUG" "✓ Package importable: $package"
        else
            log "WARN" "⚠ Package import failed: $package"
            validation_errors=$((validation_errors + 1))
        fi
    done
    
    if [[ $validation_errors -eq 0 ]]; then
        log "INFO" "Setup validation passed"
        return 0
    else
        log "WARN" "Setup validation completed with $validation_errors warnings/errors"
        return 1
    fi
}

validate_setup
validation_result=$?

# Final cleanup - remove cleanup tasks since we succeeded
CLEANUP_TASKS=()

# Final WSL-specific recommendations
show_progress "Final WSL recommendations"
if [[ "$IS_WSL" == "true" ]]; then
    log "WSL" "WSL-specific setup completed successfully!"
    log "WSL" "Recommendations:"
    log "WSL" "• Keep Docker Desktop running for container services"
    log "WSL" "• Use Windows Terminal for better experience"
    log "WSL" "• File permissions are handled differently in WSL"
    log "WSL" "• Access files from Windows at: \\\\wsl$\\${WSL_DISTRO_NAME:-Ubuntu}"
fi

echo ""
if [[ $validation_result -eq 0 ]]; then
    log "INFO" "Setup completed successfully!"
else
    log "WARN" "Setup completed with warnings. Please review the log file: $LOG_FILE"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Update API keys in .env file (if not already set)"
if [[ "$IS_WSL" == "true" ]]; then
    echo "2. Ensure Docker Desktop is running on Windows"
    echo "3. Run: ./start.sh dev"
    echo "4. Access the application at http://localhost:3000"
else
    echo "2. Run: ./start.sh dev"
    echo "3. Access the application at http://localhost:3000"
fi
echo ""

if [[ $USE_VENV -eq 1 ]]; then
    echo -e "${BLUE}For Python development, activate the virtual environment:${NC}"
    echo "  source venv/bin/activate"
fi

echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "  ./start.sh dev     # Start development environment"
echo "  ./start.sh prod    # Start production environment"
echo "  ./logs.sh          # View application logs"
echo "  ./stop.sh          # Stop all services"
echo ""

if [[ "$IS_WSL" == "true" ]]; then
    echo -e "${PURPLE}WSL-specific notes:${NC}"
    echo "• This setup has been optimized for WSL environment"
    echo "• Docker services require Docker Desktop on Windows"
    echo "• VS Code with WSL extension is recommended for development"
    echo ""
fi

echo -e "${BLUE}Log file location: $LOG_FILE${NC}"

log "INFO" "Setup script completed"