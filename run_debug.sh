#!/bin/bash
# Comprehensive Debug and Validation Runner

echo "======================================================"
echo "Investment Analysis Platform - Debug & Validation"
echo "======================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    exit 1
fi

if ! command_exists docker-compose; then
    echo -e "${RED}✗ Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"

# Run quick fixes
echo -e "\n${YELLOW}Running quick fixes...${NC}"
python3 quick_fix.py

# Make scripts executable
chmod +x debug_validate.py
chmod +x run_all_tests.py
chmod +x quick_fix.py

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "\n${YELLOW}Creating .env file from example...${NC}"
    cp .env.example .env 2>/dev/null || echo -e "${RED}Warning: .env.example not found${NC}"
fi

# Run main debug validation
echo -e "\n${YELLOW}Running comprehensive debug validation...${NC}"
python3 debug_validate.py

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Debug validation completed successfully!${NC}"
    
    # Ask if user wants to run full test suite
    echo -e "\n${YELLOW}Would you like to run the full test suite? (y/n)${NC}"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "\n${YELLOW}Running comprehensive test suite...${NC}"
        python3 run_all_tests.py
    fi
else
    echo -e "\n${RED}✗ Debug validation failed. Please check the logs.${NC}"
    exit 1
fi

# Show summary
echo -e "\n${GREEN}======================================================"
echo "Debug Process Complete!"
echo "======================================================"
echo -e "${NC}"

echo "Next steps:"
echo "1. Review debug_report.json for detailed results"
echo "2. Update .env with your actual API keys"
echo "3. Run 'make build' to build Docker images"
echo "4. Run 'make up' to start all services"
echo "5. Access the application at http://localhost:3000"

echo -e "\n${YELLOW}Documentation:${NC}"
echo "- README.md: General overview and setup"
echo "- OPTIMIZATION_GUIDE.md: Performance optimization tips"
echo "- DEPLOYMENT_CHECKLIST.md: Production deployment guide"

echo -e "\n${GREEN}Happy investing! 🚀${NC}"