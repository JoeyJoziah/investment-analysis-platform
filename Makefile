# Investment Platform - Simplified Makefile
# Essential commands for development and deployment

.PHONY: help build up down test clean logs init

help:
	@echo "Investment Platform - Available Commands"
	@echo "======================================="
	@echo "  make setup    - Initial project setup"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start development environment"
	@echo "  make down     - Stop all services"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean up containers and volumes"
	@echo "  make logs     - View logs"
	@echo "  make shell    - Open backend shell"

setup:
	@./setup.sh

build:
	@echo "Building Docker images..."
	@docker-compose build

up:
	@./start.sh dev

down:
	@./stop.sh

test:
	@./start.sh test

clean:
	@./stop.sh --clean

logs:
	@./logs.sh

shell:
	@docker-compose exec backend bash

# Python specific commands
format:
	@echo "Formatting Python code..."
	@black backend/ --line-length 88
	@isort backend/ --profile black

lint:
	@echo "Linting Python code..."
	@flake8 backend/ --max-line-length 88
	@mypy backend/ --python-version 3.11

# Frontend specific commands
frontend-dev:
	@cd frontend/web && npm start

frontend-build:
	@cd frontend/web && npm run build

frontend-test:
	@cd frontend/web && npm test

# Database commands
db-migrate:
	@docker-compose exec backend alembic upgrade head

db-rollback:
	@docker-compose exec backend alembic downgrade -1

db-shell:
	@docker-compose exec postgres psql -U postgres -d investment_db

# Monitoring commands
monitor:
	@echo "Opening monitoring dashboard..."
	@open http://localhost:3001 || xdg-open http://localhost:3001