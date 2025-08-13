# ============================================================================
# AIRFLOW ENVIRONMENT DEBUGGING SCRIPT
# ============================================================================
# This script helps debug environment variable issues with Airflow Docker setup
# Run this script to check if all required variables are properly set
# ============================================================================

Write-Host "Airflow Environment Variables Debug Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Set current directory to script location
Set-Location $PSScriptRoot

# Check if .env files exist
Write-Host "Checking .env files:" -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env file exists" -ForegroundColor Green
} else {
    Write-Host "✗ .env file missing" -ForegroundColor Red
}

if (Test-Path ".env.airflow") {
    Write-Host "✓ .env.airflow file exists" -ForegroundColor Green
} else {
    Write-Host "✗ .env.airflow file missing" -ForegroundColor Red
}

Write-Host ""

# Load and check .env variables
Write-Host "Reading variables from .env files:" -ForegroundColor Yellow
Write-Host ""

# Function to read .env file and display variables
function Read-EnvFile {
    param([string]$FilePath, [string]$FileName)
    
    if (Test-Path $FilePath) {
        Write-Host "Variables from $FileName:" -ForegroundColor Cyan
        Get-Content $FilePath | Where-Object { 
            $_ -match '^[^#].*=.*' 
        } | ForEach-Object {
            $parts = $_ -split '=', 2
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            
            # Mask sensitive values
            if ($key -match "(PASSWORD|KEY|SECRET|TOKEN)") {
                $maskedValue = if ($value.Length -gt 8) { 
                    $value.Substring(0, 4) + "****" + $value.Substring($value.Length - 4)
                } else {
                    "****"
                }
                Write-Host "  $key = $maskedValue" -ForegroundColor Gray
            } else {
                Write-Host "  $key = $value" -ForegroundColor White
            }
        }
        Write-Host ""
    } else {
        Write-Host "File $FileName not found" -ForegroundColor Red
        Write-Host ""
    }
}

Read-EnvFile ".env" ".env"
Read-EnvFile ".env.airflow" ".env.airflow"

# Check specific Airflow variables
Write-Host "Checking critical Airflow variables:" -ForegroundColor Yellow

$criticalVars = @(
    "AIRFLOW_FERNET_KEY",
    "AIRFLOW_SECRET_KEY", 
    "SMTP_USER",
    "SMTP_PASSWORD",
    "FLOWER_PASSWORD",
    "_AIRFLOW_WWW_USER_USERNAME",
    "_AIRFLOW_WWW_USER_PASSWORD",
    "AIRFLOW_DB_PASSWORD",
    "REDIS_PASSWORD"
)

# Set environment for checking (simulate docker-compose environment loading)
$env:AIRFLOW_UID = "50000"

foreach ($var in $criticalVars) {
    $value = ""
    
    # Check in .env files
    if (Test-Path ".env") {
        $envMatch = Get-Content ".env" | Where-Object { $_ -match "^$var\s*=" }
        if ($envMatch) {
            $value = ($envMatch -split '=', 2)[1].Trim()
        }
    }
    
    if (Test-Path ".env.airflow") {
        $envAirflowMatch = Get-Content ".env.airflow" | Where-Object { $_ -match "^$var\s*=" }
        if ($envAirflowMatch) {
            $value = ($envAirflowMatch -split '=', 2)[1].Trim()
        }
    }
    
    if ($value) {
        if ($var -match "(PASSWORD|KEY|SECRET|TOKEN)") {
            $maskedValue = if ($value.Length -gt 8) { 
                $value.Substring(0, 4) + "****" + $value.Substring($value.Length - 4)
            } else {
                "****"
            }
            Write-Host "✓ $var = $maskedValue" -ForegroundColor Green
        } else {
            Write-Host "✓ $var = $value" -ForegroundColor Green
        }
    } else {
        Write-Host "✗ $var = (not set or empty)" -ForegroundColor Red
    }
}

Write-Host ""

# Test Docker Compose variable expansion
Write-Host "Testing Docker Compose variable expansion:" -ForegroundColor Yellow

try {
    $testResult = docker-compose -f docker-compose.airflow.yml config 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker Compose configuration is valid" -ForegroundColor Green
        
        # Look for warnings in the output
        $warnings = $testResult | Where-Object { $_ -match "WARNING" -or $_ -match "variable.*not set" }
        if ($warnings) {
            Write-Host ""
            Write-Host "Found warnings:" -ForegroundColor Yellow
            $warnings | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        } else {
            Write-Host "✓ No environment variable warnings found" -ForegroundColor Green
        }
    } else {
        Write-Host "✗ Docker Compose configuration has errors:" -ForegroundColor Red
        $testResult | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }
} catch {
    Write-Host "✗ Failed to test Docker Compose configuration: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Recommendations
Write-Host "Recommendations:" -ForegroundColor Yellow
Write-Host "1. Ensure both .env and .env.airflow files exist" -ForegroundColor White
Write-Host "2. Run 'docker-compose -f docker-compose.airflow.yml config' to test configuration" -ForegroundColor White
Write-Host "3. Use the start-airflow.ps1 or start-airflow.bat scripts for proper environment setup" -ForegroundColor White
Write-Host "4. Check Docker Desktop is running before starting Airflow" -ForegroundColor White

Write-Host ""
Write-Host "Debug script completed!" -ForegroundColor Green