#!/bin/bash
# Investment Analysis Platform - Quick Start Script

echo "======================================"
echo "Investment Analysis Platform"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if initialization has been run
if [ ! -f ".initialized" ]; then
    echo -e "${YELLOW}First time setup detected.${NC}"
    echo "Running initialization..."
    python3 init_app.py
    
    if [ $? -eq 0 ]; then
        touch .initialized
        echo -e "${GREEN}Initialization complete!${NC}"
    else
        echo -e "${RED}Initialization failed. Please check the logs.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Starting services...${NC}"
    
    # Start all services
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Services started successfully!${NC}"
        
        # Wait for services to be ready
        echo "Waiting for services to be ready..."
        sleep 10
        
        # Show status
        echo -e "\n${GREEN}Service Status:${NC}"
        docker-compose ps
        
        echo -e "\n${GREEN}Access Points:${NC}"
        echo "  • Frontend:     http://localhost:3000"
        echo "  • API Docs:     http://localhost:8000/docs"
        echo "  • API Health:   http://localhost:8000/api/health"
        echo "  • Prometheus:   http://localhost:9090"
        echo "  • Grafana:      http://localhost:3001"
        
        echo -e "\n${YELLOW}Commands:${NC}"
        echo "  • View logs:    docker-compose logs -f"
        echo "  • Stop:         docker-compose down"
        echo "  • Restart:      docker-compose restart"
        
    else
        echo -e "${RED}Failed to start services.${NC}"
        exit 1
    fi
fi