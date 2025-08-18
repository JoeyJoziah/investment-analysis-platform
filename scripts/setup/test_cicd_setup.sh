#!/bin/bash

# CI/CD Setup Testing Script
# Tests various components of the CI/CD pipeline

set -e  # Exit on any error

echo "🚀 Investment Analysis App - CI/CD Setup Test"
echo "=============================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test functions
test_docker() {
    echo -e "${BLUE}🐳 Testing Docker setup...${NC}"
    
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}✅ Docker is installed${NC}"
        docker --version
    else
        echo -e "${RED}❌ Docker is not installed${NC}"
        return 1
    fi
    
    if command -v docker-compose &> /dev/null; then
        echo -e "${GREEN}✅ Docker Compose is installed${NC}"
        docker-compose --version
    else
        echo -e "${RED}❌ Docker Compose is not installed${NC}"
        return 1
    fi
    
    if [ -f "docker-compose.yml" ]; then
        echo -e "${GREEN}✅ docker-compose.yml found${NC}"
        echo "Validating docker-compose.yml syntax..."
        docker-compose config > /dev/null && echo -e "${GREEN}✅ docker-compose.yml is valid${NC}"
    else
        echo -e "${RED}❌ docker-compose.yml not found${NC}"
        return 1
    fi
}

test_github_workflows() {
    echo -e "${BLUE}⚙️ Testing GitHub Actions workflows...${NC}"
    
    if [ -d ".github/workflows" ]; then
        echo -e "${GREEN}✅ GitHub workflows directory found${NC}"
        
        # List workflow files
        workflow_count=$(find .github/workflows -name "*.yml" -o -name "*.yaml" | wc -l)
        echo -e "${GREEN}📋 Found ${workflow_count} workflow files:${NC}"
        find .github/workflows -name "*.yml" -o -name "*.yaml" | while read file; do
            echo "   - $(basename "$file")"
        done
        
        # Validate YAML syntax
        echo "Validating YAML syntax..."
        find .github/workflows -name "*.yml" -o -name "*.yaml" | while read file; do
            if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
                echo -e "${GREEN}✅ $file is valid YAML${NC}"
            else
                echo -e "${RED}❌ $file has YAML syntax errors${NC}"
            fi
        done
    else
        echo -e "${RED}❌ GitHub workflows directory not found${NC}"
        return 1
    fi
}

test_environment_vars() {
    echo -e "${BLUE}🔐 Testing environment variables...${NC}"
    
    # Check for .env file
    if [ -f ".env" ]; then
        echo -e "${YELLOW}⚠️ .env file found - ensure it's not committed to git${NC}"
        if git ls-files .env 2>/dev/null | grep -q .env; then
            echo -e "${RED}❌ .env file is tracked by git - this is a security risk!${NC}"
        else
            echo -e "${GREEN}✅ .env file is not tracked by git${NC}"
        fi
    fi
    
    # Check for .env.example
    if [ -f ".env.example" ] || [ -f ".env.template" ]; then
        echo -e "${GREEN}✅ Environment template file found${NC}"
    else
        echo -e "${YELLOW}⚠️ No .env.example or .env.template found${NC}"
    fi
    
    # List required environment variables
    echo "Required environment variables:"
    cat << 'EOF'
    - SECRET_KEY
    - JWT_SECRET_KEY
    - DB_PASSWORD
    - REDIS_PASSWORD
    - ALPHA_VANTAGE_API_KEY
    - FINNHUB_API_KEY
    - POLYGON_API_KEY
    - NEWS_API_KEY
EOF
}

test_python_setup() {
    echo -e "${BLUE}🐍 Testing Python setup...${NC}"
    
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version)
        echo -e "${GREEN}✅ Python 3 is installed: ${python_version}${NC}"
    else
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        return 1
    fi
    
    # Check for requirements files
    if [ -f "requirements.txt" ]; then
        echo -e "${GREEN}✅ requirements.txt found${NC}"
        echo "Python dependencies:"
        head -5 requirements.txt
        if [ $(wc -l < requirements.txt) -gt 5 ]; then
            echo "   ... and $(( $(wc -l < requirements.txt) - 5 )) more"
        fi
    elif [ -f "backend/requirements.txt" ]; then
        echo -e "${GREEN}✅ backend/requirements.txt found${NC}"
    else
        echo -e "${RED}❌ No requirements.txt found${NC}"
    fi
    
    # Check if pip is available
    if command -v pip3 &> /dev/null; then
        echo -e "${GREEN}✅ pip3 is available${NC}"
    else
        echo -e "${RED}❌ pip3 is not available${NC}"
    fi
}

test_nodejs_setup() {
    echo -e "${BLUE}📦 Testing Node.js setup...${NC}"
    
    if command -v node &> /dev/null; then
        node_version=$(node --version)
        echo -e "${GREEN}✅ Node.js is installed: ${node_version}${NC}"
    else
        echo -e "${YELLOW}⚠️ Node.js is not installed (needed for frontend)${NC}"
    fi
    
    if command -v npm &> /dev/null; then
        npm_version=$(npm --version)
        echo -e "${GREEN}✅ npm is installed: v${npm_version}${NC}"
    else
        echo -e "${YELLOW}⚠️ npm is not installed${NC}"
    fi
    
    # Check for package.json files
    if [ -f "frontend/web/package.json" ]; then
        echo -e "${GREEN}✅ Frontend package.json found${NC}"
    elif [ -f "package.json" ]; then
        echo -e "${GREEN}✅ package.json found${NC}"
    else
        echo -e "${YELLOW}⚠️ No package.json found${NC}"
    fi
}

test_git_setup() {
    echo -e "${BLUE}🔧 Testing Git setup...${NC}"
    
    if command -v git &> /dev/null; then
        echo -e "${GREEN}✅ Git is installed${NC}"
        git --version
    else
        echo -e "${RED}❌ Git is not installed${NC}"
        return 1
    fi
    
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${GREEN}✅ This is a Git repository${NC}"
        
        # Check if remote origin exists
        if git remote | grep -q origin; then
            echo -e "${GREEN}✅ Git remote 'origin' configured${NC}"
            git remote get-url origin
        else
            echo -e "${YELLOW}⚠️ No Git remote 'origin' configured${NC}"
        fi
        
        # Check current branch
        current_branch=$(git branch --show-current)
        echo -e "${GREEN}📍 Current branch: ${current_branch}${NC}"
        
    else
        echo -e "${RED}❌ This is not a Git repository${NC}"
        return 1
    fi
}

test_project_structure() {
    echo -e "${BLUE}📁 Testing project structure...${NC}"
    
    # Required directories
    required_dirs=(
        "backend"
        "frontend"
        ".github"
        ".github/workflows"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo -e "${GREEN}✅ Directory $dir exists${NC}"
        else
            echo -e "${RED}❌ Directory $dir missing${NC}"
        fi
    done
    
    # Required files
    required_files=(
        "docker-compose.yml"
        "README.md"
        "CLAUDE.md"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}✅ File $file exists${NC}"
        else
            echo -e "${YELLOW}⚠️ File $file missing${NC}"
        fi
    done
}

test_makefile() {
    echo -e "${BLUE}🔨 Testing Makefile...${NC}"
    
    if [ -f "Makefile" ]; then
        echo -e "${GREEN}✅ Makefile found${NC}"
        
        if command -v make &> /dev/null; then
            echo -e "${GREEN}✅ make command available${NC}"
            echo "Available make targets:"
            make help 2>/dev/null || echo "   (help target not available)"
        else
            echo -e "${YELLOW}⚠️ make command not available${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ Makefile not found${NC}"
    fi
}

run_validation_script() {
    echo -e "${BLUE}🔍 Running comprehensive validation...${NC}"
    
    if [ -f "scripts/validate_cicd.py" ]; then
        echo -e "${GREEN}✅ Validation script found${NC}"
        echo "Running validation script..."
        python3 scripts/validate_cicd.py
    else
        echo -e "${YELLOW}⚠️ Validation script not found${NC}"
    fi
}

print_github_setup_reminder() {
    echo ""
    echo -e "${YELLOW}📋 GitHub Setup Checklist:${NC}"
    echo ""
    echo "1. 🔐 Configure GitHub Secrets (25+ secrets required)"
    echo "   - Repository → Settings → Secrets and variables → Actions"
    echo "   - See SECRETS_CHECKLIST.md for complete list"
    echo ""
    echo "2. 🛡️ Set up Branch Protection Rules"
    echo "   - Repository → Settings → Branches → Add rule"
    echo "   - Protect 'main' and 'develop' branches"
    echo ""
    echo "3. ⚙️ Enable GitHub Actions"
    echo "   - Repository → Settings → Actions → General"
    echo "   - Allow all actions and reusable workflows"
    echo ""
    echo "4. 📊 Monitor First Workflow Run"
    echo "   - Repository → Actions tab"
    echo "   - Create test PR to trigger CI pipeline"
    echo ""
    echo -e "${GREEN}📖 Detailed setup guide: CICD_SETUP_GUIDE.md${NC}"
    echo -e "${GREEN}🎯 UI navigation help: GITHUB_UI_REFERENCE.md${NC}"
}

# Main execution
main() {
    echo "Starting CI/CD setup tests..."
    echo ""
    
    # Run all tests
    test_docker
    echo ""
    
    test_github_workflows  
    echo ""
    
    test_environment_vars
    echo ""
    
    test_python_setup
    echo ""
    
    test_nodejs_setup
    echo ""
    
    test_git_setup
    echo ""
    
    test_project_structure
    echo ""
    
    test_makefile
    echo ""
    
    run_validation_script
    echo ""
    
    print_github_setup_reminder
    
    echo ""
    echo -e "${GREEN}🎉 CI/CD setup test completed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review any ❌ failed or ⚠️ warning items above"
    echo "2. Follow the GitHub setup guide (CICD_SETUP_GUIDE.md)"  
    echo "3. Configure all required secrets (SECRETS_CHECKLIST.md)"
    echo "4. Test the pipeline with a small commit"
    echo ""
}

# Run main function
main "$@"