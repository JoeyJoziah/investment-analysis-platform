#!/bin/bash
# Investment Analysis Platform - Restart Script

echo "======================================"
echo "Restarting Investment Analysis Platform"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Stop services
echo -e "${YELLOW}Stopping services...${NC}"
docker-compose down

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Services restarted successfully!${NC}"
    
    # Wait for services
    sleep 10
    
    # Show status
    echo -e "\n${GREEN}Service Status:${NC}"
    docker-compose ps
else
    echo -e "${RED}Failed to restart services.${NC}"
    exit 1
fi