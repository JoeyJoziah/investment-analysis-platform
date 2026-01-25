#!/bin/bash
# =============================================================================
# Initial SSL Provisioning with Let's Encrypt
# Investment Analysis Platform
# =============================================================================
# Usage: ./init-ssl.sh <domain> <email>
# Example: ./init-ssl.sh investment-platform.com admin@investment-platform.com
#
# This script provisions initial SSL certificates using Let's Encrypt.
# Subsequent renewals are handled automatically by the certbot container.
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
SSL_DIR="${PROJECT_ROOT}/ssl"
CERTBOT_WWW_DIR="${PROJECT_ROOT}/certbot-www"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 <domain> <email>"
    echo ""
    echo "Arguments:"
    echo "  domain    The domain name for the SSL certificate (e.g., investment-platform.com)"
    echo "  email     Email address for Let's Encrypt notifications"
    echo ""
    echo "Examples:"
    echo "  $0 investment-platform.com admin@investment-platform.com"
    echo "  $0 api.mycompany.com devops@mycompany.com"
    exit 1
}

validate_domain() {
    local domain=$1
    # Basic domain validation regex
    if [[ ! "$domain" =~ ^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$ ]]; then
        log_error "Invalid domain format: $domain"
        exit 1
    fi
}

validate_email() {
    local email=$1
    # Basic email validation regex
    if [[ ! "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        log_error "Invalid email format: $email"
        exit 1
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    log_info "All dependencies satisfied."
}

create_directories() {
    log_info "Creating required directories..."

    mkdir -p "$SSL_DIR"
    mkdir -p "$CERTBOT_WWW_DIR"

    # Set appropriate permissions
    chmod 755 "$SSL_DIR"
    chmod 755 "$CERTBOT_WWW_DIR"

    log_info "Directories created: $SSL_DIR, $CERTBOT_WWW_DIR"
}

generate_dhparam() {
    log_info "Generating Diffie-Hellman parameters (4096-bit)..."
    log_warn "This may take several minutes on slower hardware..."

    if [ -f "$SSL_DIR/dhparam.pem" ]; then
        log_info "dhparam.pem already exists, skipping generation."
        return 0
    fi

    # Generate 4096-bit DH parameters for Perfect Forward Secrecy
    docker run --rm -v "$SSL_DIR:/ssl" alpine/openssl dhparam \
        -out /ssl/dhparam.pem 4096

    log_info "Diffie-Hellman parameters generated."
}

generate_self_signed_cert() {
    local domain=$1
    log_info "Generating temporary self-signed certificate for initial nginx startup..."

    # Generate self-signed certificate for initial startup
    docker run --rm -v "$SSL_DIR:/ssl" alpine/openssl req \
        -x509 \
        -nodes \
        -newkey rsa:2048 \
        -days 1 \
        -keyout /ssl/privkey.pem \
        -out /ssl/fullchain.pem \
        -subj "/CN=$domain/O=Investment Analysis Platform/C=US"

    # Create a placeholder chain.pem for self-signed (same as fullchain for self-signed)
    cp "$SSL_DIR/fullchain.pem" "$SSL_DIR/chain.pem"

    log_info "Temporary self-signed certificate generated."
}

obtain_letsencrypt_cert() {
    local domain=$1
    local email=$2

    log_info "Obtaining Let's Encrypt certificate for $domain..."
    log_warn "Ensure port 80 is accessible from the internet and DNS points to this server."

    # Run certbot in standalone mode
    docker run --rm \
        -v "$SSL_DIR:/etc/letsencrypt" \
        -v "$CERTBOT_WWW_DIR:/var/www/certbot" \
        -p 80:80 \
        certbot/certbot certonly \
        --standalone \
        --preferred-challenges http \
        --email "$email" \
        --agree-tos \
        --no-eff-email \
        --force-renewal \
        -d "$domain"

    if [ $? -eq 0 ]; then
        log_info "Certificate obtained successfully!"

        # Create symlinks for nginx
        log_info "Creating symlinks for nginx..."
        ln -sf "$SSL_DIR/live/$domain/fullchain.pem" "$SSL_DIR/fullchain.pem"
        ln -sf "$SSL_DIR/live/$domain/privkey.pem" "$SSL_DIR/privkey.pem"
        ln -sf "$SSL_DIR/live/$domain/chain.pem" "$SSL_DIR/chain.pem"

        # Generate DH parameters if not already present
        generate_dhparam

        log_info "SSL setup complete!"
        echo ""
        log_info "Certificate files:"
        echo "  - Full chain: $SSL_DIR/fullchain.pem"
        echo "  - Private key: $SSL_DIR/privkey.pem"
        echo "  - CA chain: $SSL_DIR/chain.pem"
        echo "  - DH params: $SSL_DIR/dhparam.pem"
        echo ""
        log_info "Next steps:"
        echo "  1. Update docker-compose.prod.yml nginx config to use nginx-ssl.conf"
        echo "  2. Start the production stack: ./start.sh prod"
        echo "  3. Verify HTTPS is working: curl -I https://$domain"
    else
        log_error "Failed to obtain certificate. Please check:"
        echo "  1. Port 80 is open and accessible"
        echo "  2. DNS A record points to this server's IP"
        echo "  3. No other service is using port 80"
        exit 1
    fi
}

# =============================================================================
# Main Script
# =============================================================================

# Validate arguments
if [ $# -ne 2 ]; then
    log_error "Missing required arguments."
    usage
fi

DOMAIN=$1
EMAIL=$2

log_info "Starting SSL provisioning for Investment Analysis Platform"
echo "=============================================="
echo "Domain: $DOMAIN"
echo "Email:  $EMAIL"
echo "=============================================="
echo ""

# Validate inputs
validate_domain "$DOMAIN"
validate_email "$EMAIL"

# Check dependencies
check_dependencies

# Create directories
create_directories

# Ask user to confirm
echo ""
read -p "Continue with SSL provisioning? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Aborted by user."
    exit 0
fi

# Check if we should generate self-signed first or go straight to Let's Encrypt
echo ""
echo "Options:"
echo "  1) Obtain Let's Encrypt certificate (requires port 80 accessible from internet)"
echo "  2) Generate self-signed certificate (for testing/development)"
echo ""
read -p "Select option [1/2]: " -n 1 -r OPTION
echo ""

case $OPTION in
    1)
        obtain_letsencrypt_cert "$DOMAIN" "$EMAIL"
        ;;
    2)
        generate_self_signed_cert "$DOMAIN"
        generate_dhparam
        log_info "Self-signed certificate generated for testing."
        log_warn "Browsers will show security warnings with self-signed certificates."
        ;;
    *)
        log_error "Invalid option selected."
        exit 1
        ;;
esac

log_info "SSL provisioning complete!"
