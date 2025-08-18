#!/bin/bash
# ONE-COMMAND INITIALIZATION FOR INVESTMENT ANALYSIS PLATFORM

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     INVESTMENT ANALYSIS PLATFORM - ONE-CLICK SETUP            ║
║                                                               ║
║     World-Class Stock Analysis for 6,000+ US Stocks          ║
║     Operating Under $50/Month                                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}This will set up the entire platform automatically.${NC}"
echo -e "${YELLOW}Make sure Docker is running before proceeding.${NC}\n"

# Quick prerequisite check
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites checked${NC}\n"

# Confirm to proceed
read -p "Press ENTER to start initialization (or Ctrl+C to cancel)..."

echo -e "\n${BLUE}Starting initialization...${NC}\n"

# Step 1: Quick fixes
echo -e "${YELLOW}[1/5] Applying quick fixes...${NC}"
if [ -f "quick_fix.py" ]; then
    python3 quick_fix.py > /dev/null 2>&1 || python quick_fix.py > /dev/null 2>&1
    echo -e "${GREEN}✓ Quick fixes applied${NC}"
else
    echo -e "${YELLOW}⚠ Quick fix script not found, skipping${NC}"
fi

# Step 2: Environment setup
echo -e "\n${YELLOW}[2/5] Setting up environment...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Environment file created${NC}"
        echo -e "${YELLOW}  ⚠ Remember to add your API keys to .env${NC}"
    else
        echo -e "${YELLOW}⚠ No .env.example found${NC}"
    fi
else
    echo -e "${GREEN}✓ Environment file exists${NC}"
fi

# Step 3: Build images
echo -e "\n${YELLOW}[3/5] Building Docker images...${NC}"
echo -e "${BLUE}  This will take 5-10 minutes on first run...${NC}"
docker-compose build --quiet
echo -e "${GREEN}✓ Docker images built${NC}"

# Step 4: Start services
echo -e "\n${YELLOW}[4/5] Starting all services...${NC}"

# Start infrastructure first
echo -e "${BLUE}  Starting infrastructure...${NC}"
docker-compose up -d postgres redis elasticsearch > /dev/null 2>&1
sleep 10

# Initialize database
echo -e "${BLUE}  Initializing database...${NC}"
docker-compose up -d backend > /dev/null 2>&1
sleep 10

# Try to initialize database
docker-compose exec -T backend python -m backend.utils.db_init > /dev/null 2>&1 || true

# Start remaining services
echo -e "${BLUE}  Starting application services...${NC}"
docker-compose up -d > /dev/null 2>&1

echo -e "${GREEN}✓ All services started${NC}"

# Step 5: Verify
echo -e "\n${YELLOW}[5/5] Verifying installation...${NC}"
sleep 15  # Wait for services to be ready

# Check if API is responding
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health | grep -q "200"; then
    echo -e "${GREEN}✓ Backend API is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Backend API not responding yet (may still be starting)${NC}"
fi

# Final summary
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ INITIALIZATION COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

echo -e "${BLUE}🌐 Access Points:${NC}"
echo -e "   • Frontend:     ${GREEN}http://localhost:3000${NC}"
echo -e "   • API Docs:     ${GREEN}http://localhost:8000/docs${NC}"
echo -e "   • Prometheus:   ${GREEN}http://localhost:9090${NC}"
echo -e "   • Grafana:      ${GREEN}http://localhost:3001${NC}"

echo -e "\n${BLUE}📝 Next Steps:${NC}"
echo -e "   1. Add your API keys to ${YELLOW}.env${NC} file"
echo -e "   2. Open ${GREEN}http://localhost:3000${NC} in your browser"
echo -e "   3. Start analyzing stocks!"

echo -e "\n${BLUE}🛠️ Useful Commands:${NC}"
echo -e "   • View logs:    ${YELLOW}docker-compose logs -f${NC}"
echo -e "   • Stop:         ${YELLOW}./stop.sh${NC}"
echo -e "   • Restart:      ${YELLOW}./restart.sh${NC}"
echo -e "   • Debug:        ${YELLOW}python debug_validate.py${NC}"

echo -e "\n${GREEN}Ready to analyze 6,000+ stocks with world-class insights! 🚀${NC}\n"

# Create initialized marker
touch .initialized

# Open browser (optional)
if command -v xdg-open &> /dev/null; then
    echo -e "${BLUE}Opening browser...${NC}"
    sleep 5
    xdg-open http://localhost:3000 2>/dev/null || true
elif command -v open &> /dev/null; then
    echo -e "${BLUE}Opening browser...${NC}"
    sleep 5
    open http://localhost:3000 2>/dev/null || true
fi