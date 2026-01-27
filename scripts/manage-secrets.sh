#!/bin/bash

# Script to manage Kubernetes secrets securely
# This script helps create and manage secrets without hardcoding them

set -euo pipefail

NAMESPACE="investment-analysis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to generate secure random secrets
generate_secret() {
    openssl rand -base64 32
}

# Function to check if required tools are installed
check_requirements() {
    local tools=("kubectl" "openssl")
    for tool in "${tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            print_message "$RED" "Error: $tool is not installed"
            exit 1
        fi
    done
    
    # Check if kubeseal is installed (optional but recommended)
    if command -v kubeseal &> /dev/null; then
        print_message "$GREEN" "kubeseal found - will create sealed secrets"
        SEALED_SECRETS_AVAILABLE=true
    else
        print_message "$YELLOW" "Warning: kubeseal not found - creating regular secrets"
        print_message "$YELLOW" "Install sealed-secrets for production use:"
        print_message "$YELLOW" "  kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml"
        SEALED_SECRETS_AVAILABLE=false
    fi
}

# Function to create secrets from environment variables or prompt
create_secrets() {
    print_message "$GREEN" "Creating Kubernetes secrets..."
    
    # Database password
    if [ -z "${DB_PASSWORD:-}" ]; then
        print_message "$YELLOW" "DB_PASSWORD not set. Generating secure password..."
        DB_PASSWORD=$(generate_secret)
        print_message "$GREEN" "Generated DB_PASSWORD (save this): $DB_PASSWORD"
    fi
    
    # App secret key
    if [ -z "${APP_SECRET_KEY:-}" ]; then
        print_message "$YELLOW" "APP_SECRET_KEY not set. Generating secure key..."
        APP_SECRET_KEY=$(generate_secret)
        print_message "$GREEN" "Generated APP_SECRET_KEY (save this): $APP_SECRET_KEY"
    fi
    
    # JWT secret key
    if [ -z "${JWT_SECRET_KEY:-}" ]; then
        print_message "$YELLOW" "JWT_SECRET_KEY not set. Generating secure key..."
        JWT_SECRET_KEY=$(generate_secret)
        print_message "$GREEN" "Generated JWT_SECRET_KEY (save this): $JWT_SECRET_KEY"
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    if [ "$SEALED_SECRETS_AVAILABLE" = true ]; then
        # Create sealed secrets
        print_message "$GREEN" "Creating sealed secrets..."
        
        # Database secret
        echo -n "$DB_PASSWORD" | kubectl create secret generic db-secret \
            --namespace=$NAMESPACE \
            --from-file=password=/dev/stdin \
            --dry-run=client -o yaml | \
            kubeseal --namespace=$NAMESPACE -o yaml > "$PROJECT_ROOT/infrastructure/kubernetes/sealed-db-secret.yaml"
        
        # App secrets
        kubectl create secret generic app-secret \
            --namespace=$NAMESPACE \
            --from-literal=secret-key="$APP_SECRET_KEY" \
            --from-literal=jwt-secret="$JWT_SECRET_KEY" \
            --dry-run=client -o yaml | \
            kubeseal --namespace=$NAMESPACE -o yaml > "$PROJECT_ROOT/infrastructure/kubernetes/sealed-app-secret.yaml"
        
        # API keys (if provided)
        if [ -n "${ALPHA_VANTAGE_API_KEY:-}" ] && [ -n "${FINNHUB_API_KEY:-}" ]; then
            kubectl create secret generic api-keys \
                --namespace=$NAMESPACE \
                --from-literal=alpha-vantage-key="${ALPHA_VANTAGE_API_KEY}" \
                --from-literal=finnhub-key="${FINNHUB_API_KEY}" \
                --from-literal=polygon-key="${POLYGON_API_KEY:-}" \
                --from-literal=news-api-key="${NEWS_API_KEY:-}" \
                --from-literal=fmp-key="${FMP_API_KEY:-}" \
                --dry-run=client -o yaml | \
                kubeseal --namespace=$NAMESPACE -o yaml > "$PROJECT_ROOT/infrastructure/kubernetes/sealed-api-keys.yaml"
        fi
        
        print_message "$GREEN" "Sealed secrets created successfully!"
        print_message "$YELLOW" "Apply them with: kubectl apply -f infrastructure/kubernetes/sealed-*.yaml"
    else
        # Create regular secrets (for development)
        print_message "$YELLOW" "Creating regular secrets (development only)..."
        
        kubectl create secret generic db-secret \
            --namespace=$NAMESPACE \
            --from-literal=password="$DB_PASSWORD" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        kubectl create secret generic app-secret \
            --namespace=$NAMESPACE \
            --from-literal=secret-key="$APP_SECRET_KEY" \
            --from-literal=jwt-secret="$JWT_SECRET_KEY" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        if [ -n "${ALPHA_VANTAGE_API_KEY:-}" ] && [ -n "${FINNHUB_API_KEY:-}" ]; then
            kubectl create secret generic api-keys \
                --namespace=$NAMESPACE \
                --from-literal=alpha-vantage-key="${ALPHA_VANTAGE_API_KEY}" \
                --from-literal=finnhub-key="${FINNHUB_API_KEY}" \
                --from-literal=polygon-key="${POLYGON_API_KEY:-}" \
                --from-literal=news-api-key="${NEWS_API_KEY:-}" \
                --from-literal=fmp-key="${FMP_API_KEY:-}" \
                --dry-run=client -o yaml | kubectl apply -f -
        fi
        
        print_message "$GREEN" "Regular secrets created successfully!"
    fi
}

# Function to verify secrets
verify_secrets() {
    print_message "$GREEN" "Verifying secrets..."
    
    local secrets=("db-secret" "app-secret")
    for secret in "${secrets[@]}"; do
        if kubectl get secret $secret -n $NAMESPACE &> /dev/null; then
            print_message "$GREEN" "✓ Secret $secret exists"
        else
            print_message "$RED" "✗ Secret $secret not found"
        fi
    done
}

# Function to save secrets to .env file (for local development)
save_to_env() {
    local env_file="$PROJECT_ROOT/.env.production"
    
    print_message "$YELLOW" "Saving secrets to $env_file for backup..."
    
    cat > "$env_file" << EOF
# Production Secrets - KEEP THIS FILE SECURE!
# Generated on $(date)
DB_PASSWORD=$DB_PASSWORD
APP_SECRET_KEY=$APP_SECRET_KEY
JWT_SECRET_KEY=$JWT_SECRET_KEY

# Add your API keys here
ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY:-}
FINNHUB_API_KEY=${FINNHUB_API_KEY:-}
POLYGON_API_KEY=${POLYGON_API_KEY:-}
NEWS_API_KEY=${NEWS_API_KEY:-}
FMP_API_KEY=${FMP_API_KEY:-}
EOF
    
    chmod 600 "$env_file"
    print_message "$GREEN" "Secrets saved to $env_file (file permissions set to 600)"
    print_message "$RED" "IMPORTANT: Add $env_file to .gitignore and never commit it!"
}

# Main execution
main() {
    print_message "$GREEN" "=== Kubernetes Secrets Management ==="
    
    check_requirements
    
    case "${1:-create}" in
        create)
            create_secrets
            verify_secrets
            save_to_env
            ;;
        verify)
            verify_secrets
            ;;
        rotate)
            print_message "$YELLOW" "Rotating secrets..."
            DB_PASSWORD=$(generate_secret)
            APP_SECRET_KEY=$(generate_secret)
            JWT_SECRET_KEY=$(generate_secret)
            create_secrets
            verify_secrets
            save_to_env
            print_message "$GREEN" "Secrets rotated successfully!"
            print_message "$YELLOW" "Remember to update your application deployments!"
            ;;
        *)
            print_message "$RED" "Usage: $0 [create|verify|rotate]"
            exit 1
            ;;
    esac
}

main "$@"