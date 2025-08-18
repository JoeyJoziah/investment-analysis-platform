#!/bin/bash
# Investment Analysis Platform - Stop Script

echo "======================================"
echo "Stopping Investment Analysis Platform"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Stop all services
echo -e "${YELLOW}Stopping all services...${NC}"
docker-compose down

if [ $? -eq 0 ]; then
    echo -e "${GREEN}All services stopped successfully!${NC}"
else
    echo -e "${RED}Error stopping services.${NC}"
    exit 1
fi

# Ask if user wants to remove volumes
echo -e "\n${YELLOW}Do you want to remove data volumes? (y/n)${NC}"
echo "Warning: This will delete all data including the database!"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Removing volumes...${NC}"
    docker-compose down -v
    rm -f .initialized
    echo -e "${GREEN}Volumes removed.${NC}"
else
    echo -e "${GREEN}Volumes preserved.${NC}"
fi