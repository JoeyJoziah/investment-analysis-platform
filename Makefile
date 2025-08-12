# Makefile for Investment Analysis Platform

.PHONY: help build up down test clean debug

help:
	@echo "Available commands:"
	@echo "  make build    - Build all Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make test     - Run all tests"
	@echo "  make clean    - Clean up containers and volumes"
	@echo "  make debug    - Run debug validation"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Services started! Access at:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API Docs: http://localhost:8000/docs"

down:
	docker-compose down

test:
	docker-compose run --rm backend pytest

clean:
	docker-compose down -v
	rm -rf logs/* data/cache/*

debug:
	python debug_validate.py

init-db:
	docker-compose exec postgres psql -U postgres -f /scripts/init_db.sql

logs:
	docker-compose logs -f

install-deps:
	pip install -r requirements.txt
	cd frontend/web && npm install
