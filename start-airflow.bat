@echo off
REM ============================================================================
REM AIRFLOW DOCKER COMPOSE STARTUP SCRIPT - WINDOWS COMPATIBLE
REM ============================================================================
REM This script sets up environment variables and starts Airflow services
REM Compatible with Windows PowerShell and Command Prompt
REM ============================================================================

echo Starting Investment Analysis App - Airflow Services
echo ===================================================

REM Set script directory as current directory
cd /d "%~dp0"

REM ============================================================================
REM CHECK PREREQUISITES
REM ============================================================================
echo Checking prerequisites...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running or not installed
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: docker-compose is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required files exist
if not exist ".env" (
    echo ERROR: .env file not found
    echo Please ensure .env file exists in the project root
    pause
    exit /b 1
)

if not exist ".env.airflow" (
    echo ERROR: .env.airflow file not found
    echo Please ensure .env.airflow file exists in the project root
    pause
    exit /b 1
)

if not exist "docker-compose.airflow.yml" (
    echo ERROR: docker-compose.airflow.yml file not found
    pause
    exit /b 1
)

echo Prerequisites check completed successfully
echo.

REM ============================================================================
REM SET AIRFLOW ENVIRONMENT VARIABLES
REM ============================================================================
echo Setting up Airflow environment variables...

REM Set Airflow UID for proper file permissions
set AIRFLOW_UID=50000
set AIRFLOW_GID=0

REM Set Docker Compose project name
set COMPOSE_PROJECT_NAME=investment-airflow

REM Set project directory for volume mounts
set AIRFLOW_PROJ_DIR=%cd%

REM Export environment variables for this session
echo AIRFLOW_UID=%AIRFLOW_UID%
echo AIRFLOW_GID=%AIRFLOW_GID%
echo COMPOSE_PROJECT_NAME=%COMPOSE_PROJECT_NAME%
echo AIRFLOW_PROJ_DIR=%AIRFLOW_PROJ_DIR%

echo Environment variables set successfully
echo.

REM ============================================================================
REM CREATE REQUIRED DIRECTORIES
REM ============================================================================
echo Creating required directories...

if not exist "data_pipelines\airflow\dags" mkdir "data_pipelines\airflow\dags"
if not exist "data_pipelines\airflow\logs" mkdir "data_pipelines\airflow\logs"
if not exist "data_pipelines\airflow\plugins" mkdir "data_pipelines\airflow\plugins"
if not exist "data_pipelines\airflow\config" mkdir "data_pipelines\airflow\config"

echo Directories created successfully
echo.

REM ============================================================================
REM CHECK FOR EXISTING CONTAINERS
REM ============================================================================
echo Checking for existing containers...

docker-compose -f docker-compose.airflow.yml ps -q >nul 2>&1
if not errorlevel 1 (
    echo Found existing containers. Stopping them first...
    docker-compose -f docker-compose.airflow.yml down
    echo.
)

REM ============================================================================
REM INITIALIZE AIRFLOW (First-time setup)
REM ============================================================================
echo Initializing Airflow database and user accounts...
echo This may take a few minutes on first run...
echo.

docker-compose -f docker-compose.airflow.yml up airflow-init
if errorlevel 1 (
    echo ERROR: Airflow initialization failed
    echo Check the logs above for details
    pause
    exit /b 1
)

echo.
echo Airflow initialization completed successfully
echo.

REM ============================================================================
REM START AIRFLOW SERVICES
REM ============================================================================
echo Starting Airflow services...
echo.

REM Start services in detached mode
docker-compose -f docker-compose.airflow.yml up -d

if errorlevel 1 (
    echo ERROR: Failed to start Airflow services
    echo Check the logs with: docker-compose -f docker-compose.airflow.yml logs
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo AIRFLOW SERVICES STARTED SUCCESSFULLY
echo ============================================================================
echo.
echo Available services:
echo - Airflow Webserver: http://localhost:8080
echo - Flower (Celery Monitor): http://localhost:5555
echo - StatsD Exporter: http://localhost:9102
echo.
echo Default login credentials:
echo - Username: admin
echo - Password: secure_admin_password_789
echo.
echo Useful commands:
echo - View logs: docker-compose -f docker-compose.airflow.yml logs -f [service-name]
echo - Stop services: docker-compose -f docker-compose.airflow.yml down
echo - Restart service: docker-compose -f docker-compose.airflow.yml restart [service-name]
echo.
echo ============================================================================

REM Wait for services to be ready
echo Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo Checking service health...
docker-compose -f docker-compose.airflow.yml ps

echo.
echo Airflow deployment completed!
echo Press any key to exit...
pause >nul