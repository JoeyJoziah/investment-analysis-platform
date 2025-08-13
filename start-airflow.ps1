# ============================================================================
# AIRFLOW DOCKER COMPOSE STARTUP SCRIPT - POWERSHELL
# ============================================================================
# This script sets up environment variables and starts Airflow services
# Compatible with Windows PowerShell 5.1+ and PowerShell Core
# ============================================================================

param(
    [switch]$Init,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Logs,
    [string]$Service = ""
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Red = @{ ForegroundColor = "Red" }
$Green = @{ ForegroundColor = "Green" }
$Yellow = @{ ForegroundColor = "Yellow" }
$Cyan = @{ ForegroundColor = "Cyan" }

Write-Host "Starting Investment Analysis App - Airflow Services" @Cyan
Write-Host "====================================================" @Cyan
Write-Host ""

# Set current directory to script location
Set-Location $PSScriptRoot

# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

function Test-Prerequisites {
    Write-Host "Checking prerequisites..." @Yellow
    
    # Check if Docker is running
    try {
        $null = docker info 2>$null
        Write-Host "✓ Docker is running" @Green
    } catch {
        Write-Host "✗ ERROR: Docker is not running or not installed" @Red
        Write-Host "Please start Docker Desktop and try again" @Red
        exit 1
    }
    
    # Check if docker-compose is available
    try {
        $null = docker-compose --version 2>$null
        Write-Host "✓ docker-compose is available" @Green
    } catch {
        Write-Host "✗ ERROR: docker-compose is not installed or not in PATH" @Red
        exit 1
    }
    
    # Check if required files exist
    $requiredFiles = @(".env", ".env.airflow", "docker-compose.airflow.yml")
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✓ Found $file" @Green
        } else {
            Write-Host "✗ ERROR: $file file not found" @Red
            Write-Host "Please ensure $file file exists in the project root" @Red
            exit 1
        }
    }
    
    Write-Host "Prerequisites check completed successfully" @Green
    Write-Host ""
}

function Set-AirflowEnvironment {
    Write-Host "Setting up Airflow environment variables..." @Yellow
    
    # Set Airflow UID for proper file permissions
    $env:AIRFLOW_UID = "50000"
    $env:AIRFLOW_GID = "0"
    
    # Set Docker Compose project name
    $env:COMPOSE_PROJECT_NAME = "investment-airflow"
    
    # Set project directory for volume mounts
    $env:AIRFLOW_PROJ_DIR = (Get-Location).Path
    
    # Display environment variables
    Write-Host "AIRFLOW_UID=$env:AIRFLOW_UID" @Cyan
    Write-Host "AIRFLOW_GID=$env:AIRFLOW_GID" @Cyan
    Write-Host "COMPOSE_PROJECT_NAME=$env:COMPOSE_PROJECT_NAME" @Cyan
    Write-Host "AIRFLOW_PROJ_DIR=$env:AIRFLOW_PROJ_DIR" @Cyan
    
    Write-Host "Environment variables set successfully" @Green
    Write-Host ""
}

function New-RequiredDirectories {
    Write-Host "Creating required directories..." @Yellow
    
    $directories = @(
        "data_pipelines\airflow\dags",
        "data_pipelines\airflow\logs", 
        "data_pipelines\airflow\plugins",
        "data_pipelines\airflow\config"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "✓ Created $dir" @Green
        } else {
            Write-Host "✓ Directory $dir already exists" @Green
        }
    }
    
    Write-Host "Directories created successfully" @Green
    Write-Host ""
}

function Stop-ExistingContainers {
    Write-Host "Checking for existing containers..." @Yellow
    
    $containers = docker-compose -f docker-compose.airflow.yml ps -q 2>$null
    if ($containers) {
        Write-Host "Found existing containers. Stopping them first..." @Yellow
        docker-compose -f docker-compose.airflow.yml down
        Write-Host "Existing containers stopped" @Green
    } else {
        Write-Host "No existing containers found" @Green
    }
    Write-Host ""
}

function Initialize-Airflow {
    Write-Host "Initializing Airflow database and user accounts..." @Yellow
    Write-Host "This may take a few minutes on first run..." @Yellow
    Write-Host ""
    
    $initResult = docker-compose -f docker-compose.airflow.yml up airflow-init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ ERROR: Airflow initialization failed" @Red
        Write-Host "Check the logs above for details" @Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Airflow initialization completed successfully" @Green
    Write-Host ""
}

function Start-AirflowServices {
    Write-Host "Starting Airflow services..." @Yellow
    Write-Host ""
    
    # Start services in detached mode
    docker-compose -f docker-compose.airflow.yml up -d
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ ERROR: Failed to start Airflow services" @Red
        Write-Host "Check the logs with: docker-compose -f docker-compose.airflow.yml logs" @Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "============================================================================" @Cyan
    Write-Host "AIRFLOW SERVICES STARTED SUCCESSFULLY" @Green
    Write-Host "============================================================================" @Cyan
    Write-Host ""
    Write-Host "Available services:" @Yellow
    Write-Host "- Airflow Webserver: http://localhost:8080" @Cyan
    Write-Host "- Flower (Celery Monitor): http://localhost:5555" @Cyan
    Write-Host "- StatsD Exporter: http://localhost:9102" @Cyan
    Write-Host ""
    Write-Host "Default login credentials:" @Yellow
    Write-Host "- Username: admin" @Cyan
    Write-Host "- Password: secure_admin_password_789" @Cyan
    Write-Host ""
    Write-Host "Useful commands:" @Yellow
    Write-Host "- View logs: docker-compose -f docker-compose.airflow.yml logs -f [service-name]" @Cyan
    Write-Host "- Stop services: docker-compose -f docker-compose.airflow.yml down" @Cyan
    Write-Host "- Restart service: docker-compose -f docker-compose.airflow.yml restart [service-name]" @Cyan
    Write-Host ""
    Write-Host "============================================================================" @Cyan
    
    # Wait for services to be ready
    Write-Host "Waiting for services to be ready..." @Yellow
    Start-Sleep -Seconds 30
    
    # Check service health
    Write-Host "Checking service health..." @Yellow
    docker-compose -f docker-compose.airflow.yml ps
    
    Write-Host ""
    Write-Host "✓ Airflow deployment completed!" @Green
}

function Show-Logs {
    param([string]$ServiceName)
    
    if ($ServiceName) {
        Write-Host "Showing logs for service: $ServiceName" @Yellow
        docker-compose -f docker-compose.airflow.yml logs -f $ServiceName
    } else {
        Write-Host "Showing logs for all services:" @Yellow
        docker-compose -f docker-compose.airflow.yml logs -f
    }
}

function Stop-AirflowServices {
    Write-Host "Stopping Airflow services..." @Yellow
    docker-compose -f docker-compose.airflow.yml down
    Write-Host "✓ Airflow services stopped" @Green
}

function Restart-AirflowServices {
    param([string]$ServiceName)
    
    if ($ServiceName) {
        Write-Host "Restarting service: $ServiceName" @Yellow
        docker-compose -f docker-compose.airflow.yml restart $ServiceName
        Write-Host "✓ Service $ServiceName restarted" @Green
    } else {
        Write-Host "Restarting all Airflow services..." @Yellow
        docker-compose -f docker-compose.airflow.yml restart
        Write-Host "✓ All Airflow services restarted" @Green
    }
}

# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

try {
    if ($Stop) {
        Test-Prerequisites
        Stop-AirflowServices
        return
    }
    
    if ($Logs) {
        Show-Logs -ServiceName $Service
        return
    }
    
    if ($Restart) {
        Test-Prerequisites
        Restart-AirflowServices -ServiceName $Service
        return
    }
    
    # Default: Start Airflow services
    Test-Prerequisites
    Set-AirflowEnvironment
    New-RequiredDirectories
    Stop-ExistingContainers
    
    if ($Init) {
        Initialize-Airflow
    }
    
    Start-AirflowServices
    
} catch {
    Write-Host ""
    Write-Host "✗ ERROR: $($_.Exception.Message)" @Red
    Write-Host "Stack trace:" @Red
    Write-Host $_.ScriptStackTrace @Red
    exit 1
}

Write-Host ""
Write-Host "Script completed successfully!" @Green