#!/bin/bash
###############################################################################
# GIT SECRET CLEANUP SCRIPT
# Purpose: Remove secrets from git history
# WARNING: This rewrites git history - coordinate with all team members!
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== GIT SECRET CLEANUP SCRIPT ===${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: This script will rewrite git history!${NC}"
echo -e "${RED}⚠️  All collaborators must re-clone the repository after this!${NC}"
echo ""
echo "This script will:"
echo "  1. Create a backup branch"
echo "  2. Remove .env files from git history"
echo "  3. Remove sensitive configuration files"
echo "  4. Require force push to remote"
echo ""
echo -e "${YELLOW}Before proceeding:${NC}"
echo "  ✓ Notify all team members"
echo "  ✓ Ensure you have a backup"
echo "  ✓ Rotate all exposed secrets"
echo ""
read -p "Have you completed the above steps? (type 'YES' to continue): " confirm

if [ "$confirm" != "YES" ]; then
    echo -e "${RED}Aborted.${NC}"
    exit 1
fi

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo -e "${RED}ERROR: git-filter-repo not found${NC}"
    echo ""
    echo "Install with:"
    echo "  pip3 install git-filter-repo"
    echo "  OR"
    echo "  brew install git-filter-repo"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo -e "${GREEN}Step 1: Creating backup branch${NC}"
git branch "backup-before-secret-cleanup-${TIMESTAMP}" || {
    echo -e "${RED}Failed to create backup branch${NC}"
    exit 1
}
echo "✓ Backup created: backup-before-secret-cleanup-${TIMESTAMP}"

echo ""
echo -e "${GREEN}Step 2: Removing .env files from history${NC}"

# Files to remove
FILES_TO_REMOVE=(
    ".env"
    ".env.secure"
    ".env.production"
    ".env.airflow"
    ".env_backup_DONOTUSE/"
)

for file in "${FILES_TO_REMOVE[@]}"; do
    echo "  Removing: $file"
    git filter-repo --path "$file" --invert-paths --force || {
        echo -e "${YELLOW}  Warning: Could not remove $file (may not exist)${NC}"
    }
done

echo ""
echo -e "${GREEN}Step 3: Removing sensitive config files from history${NC}"

SENSITIVE_CONFIGS=(
    "config/secrets.yaml"
    "config/.secrets"
    "*.pem"
    "*.key"
    "*.crt"
    "id_rsa"
    "id_rsa.pub"
)

for pattern in "${SENSITIVE_CONFIGS[@]}"; do
    echo "  Removing pattern: $pattern"
    git filter-repo --path-glob "$pattern" --invert-paths --force || {
        echo -e "${YELLOW}  Warning: Could not remove $pattern (may not exist)${NC}"
    }
done

echo ""
echo -e "${GREEN}Step 4: Cleaning up refs${NC}"
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo -e "${GREEN}✓ Git history cleaned successfully!${NC}"
echo ""
echo -e "${YELLOW}=== NEXT STEPS ===${NC}"
echo ""
echo "1. Verify the cleanup:"
echo "   git log --all --oneline --decorate --graph"
echo ""
echo "2. Force push to remote (THIS WILL REWRITE REMOTE HISTORY):"
echo -e "   ${RED}git push origin --force --all${NC}"
echo -e "   ${RED}git push origin --force --tags${NC}"
echo ""
echo "3. Notify all collaborators to:"
echo "   - Save any uncommitted work"
echo "   - Delete their local repository"
echo "   - Re-clone from remote"
echo ""
echo "4. Update .gitignore to prevent re-committing:"
echo "   echo '.env*' >> .gitignore"
echo "   echo '!.env.example' >> .gitignore"
echo "   echo '!.env.template' >> .gitignore"
echo "   git add .gitignore"
echo "   git commit -m 'chore: update .gitignore to exclude .env files'"
echo ""
echo -e "${YELLOW}Backup branch available: backup-before-secret-cleanup-${TIMESTAMP}${NC}"
echo "If something goes wrong, restore with:"
echo "  git reset --hard backup-before-secret-cleanup-${TIMESTAMP}"
echo ""
echo -e "${GREEN}=== CLEANUP SCRIPT COMPLETE ===${NC}"
