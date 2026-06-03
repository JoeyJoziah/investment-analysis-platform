#requires -Version 7
<#
.SYNOPSIS
    Bring up the minimum stack needed for local Playwright E2E and seed the
    two test users that auth.spec.ts and portfolio.spec.ts hard-code.

.DESCRIPTION
    Pre-flight for `npx playwright test` from frontend/web/. Starts only the
    services the E2E suite needs (postgres + redis), creates the schema via
    Base.metadata.create_all() (alembic migrations assume a pre-existing
    schema), and seeds the two hardcoded users.

    Compose layering:
      - docker-compose.yml                 base
      - docker-compose.dev.yml             dev overrides (env, ports)
      - docker-compose.e2e-portshift.yml   shifts postgres host port to 5433
                                           to avoid conflict with native
                                           Windows postgres on 5432.

    Once this script reports READY, run:
        cd frontend/web
        npx playwright test --project=chromium auth.spec.ts

.NOTES
    Prerequisites:
      - Docker Desktop running (Add 'C:\Program Files\Docker\Docker\resources\bin' to PATH if `docker --version` fails.)
      - Python venv with project deps installed
      - npm install completed in frontend/web/
      - .env in repo root with DB_PASSWORD set (created by setup.sh)

    If the postgres data volume was created by a prior run with a different
    password, the script will ALTER USER to match .env (POSTGRES_PASSWORD
    env var only applies on first init of the data volume).
#>

[CmdletBinding()]
param(
    [switch]$Reset,
    [switch]$SkipSeed,
    [switch]$StartAppServers,
    [int]$WaitSeconds = 90
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "    FAIL: $msg" -ForegroundColor Red; exit 1 }

# Auto-add Docker Desktop to PATH if missing
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $candidate = 'C:\Program Files\Docker\Docker\resources\bin'
    if (Test-Path "$candidate\docker.exe") {
        $env:PATH = "$candidate;" + $env:PATH
        Ok "added Docker Desktop bin to PATH"
    }
}

Step 'Preflight'
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail 'docker still not on PATH. Install Docker Desktop and start it.'
}
try { docker info --format '{{.ServerVersion}}' 2>$null | Out-Null }
catch { Fail 'dockerd not reachable. Start Docker Desktop and retry.' }
Ok "docker $(docker --version)"

if (-not (Test-Path docker-compose.yml))             { Fail 'docker-compose.yml missing' }
if (-not (Test-Path docker-compose.dev.yml))         { Fail 'docker-compose.dev.yml missing' }
if (-not (Test-Path docker-compose.e2e-portshift.yml)) { Fail 'docker-compose.e2e-portshift.yml missing (port 5432 collision dodge)' }
if (-not (Test-Path .env))                           { Fail '.env missing — run setup.sh first' }

$composeArgs = @('-f','docker-compose.yml','-f','docker-compose.dev.yml','-f','docker-compose.e2e-portshift.yml')

if ($Reset) {
    Step 'Reset: tearing down volumes'
    docker compose @composeArgs down -v 2>&1 | Out-Null
}

Step 'Starting postgres + redis'
docker compose @composeArgs up -d postgres redis 2>&1 | Out-Null

Step "Waiting up to ${WaitSeconds}s for postgres healthy"
$pgReady = $false
1..$WaitSeconds | ForEach-Object {
    if ($pgReady) { return }
    $cid = docker compose @composeArgs ps -q postgres
    if ($cid) {
        $h = docker inspect --format '{{.State.Health.Status}}' $cid 2>$null
        if ($h -eq 'healthy') { $pgReady = $true; Ok 'postgres healthy'; return }
    }
    Start-Sleep -Seconds 1
}
if (-not $pgReady) { Fail 'postgres did not become healthy in time' }

Step 'Verifying postgres password matches .env (handles stale data volumes)'
$dbpw = (Select-String -Path .env -Pattern '^DB_PASSWORD=(.+)$').Matches[0].Groups[1].Value
# ALTER USER is idempotent and safe to run on every bootstrap
$null = docker exec investment_db psql -U postgres -d investment_db -c "ALTER USER postgres WITH PASSWORD '$dbpw';" 2>&1
Ok 'postgres password aligned'

if (-not $SkipSeed) {
    Step 'Creating schema (Base.metadata.create_all) + seeding E2E users'
    $env:DATABASE_URL = "postgresql://postgres:$dbpw@127.0.0.1:5433/investment_db"
    $env:DB_PORT = '5433'
    $seed = @"
import bcrypt, uuid
from datetime import datetime, timezone
from sqlalchemy import text
from backend.models.unified_models import Base
from backend.utils.database import engine

Base.metadata.create_all(bind=engine)
print('  schema created')

USERS = [
    ('existing@example.com',       'existinguser',   'Existing User',    'ExistingPass123!@#'),
    ('portfolio-test@example.com', 'portfolio-test', 'Portfolio Tester', 'PortfolioTest123!'),
]
now = datetime.now(timezone.utc).replace(tzinfo=None)
with engine.begin() as conn:
    for email, username, full_name, pw in USERS:
        if conn.execute(text('SELECT 1 FROM users WHERE email = :e'), {'e': email}).first():
            print(f'  exists: {email}'); continue
        h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        conn.execute(text(
            'INSERT INTO users (user_id, email, username, hashed_password, full_name, role, is_active, is_verified, created_at, updated_at) '
            'VALUES (:uid, :em, :un, :hp, :fn, :role, true, true, :now, :now)'
        ), {'uid': str(uuid.uuid4()), 'em': email, 'un': username, 'hp': h, 'fn': full_name, 'role': 'basic_user', 'now': now})
        print(f'  created: {email}')
    total = conn.execute(text('SELECT count(*) FROM users')).scalar()
    print(f'  total_users={total}')
"@
    $f = Join-Path $env:TEMP 'e2e_seed.py'
    Set-Content -Path $f -Value $seed -NoNewline
    python $f
    if ($LASTEXITCODE -ne 0) { Fail 'seed failed' }
    Remove-Item $f -ErrorAction SilentlyContinue
    Ok 'schema + users ready'
}

if ($StartAppServers) {
    Step 'Starting backend + frontend dev servers'
    Write-Host '    Playwright webServer will start these; only use this flag for manual debugging.'
    $env:DATABASE_URL = "postgresql://postgres:$dbpw@127.0.0.1:5433/investment_db"
    $env:DB_PORT = '5433'
    Start-Process -FilePath 'python' -ArgumentList '-m','uvicorn','backend.api.main:app','--port','8000','--host','127.0.0.1' -RedirectStandardOutput '_backend.log' -RedirectStandardError '_backend.err.log' -WindowStyle Hidden
    Push-Location frontend/web
    Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','dev' -RedirectStandardOutput '../../_frontend.log' -RedirectStandardError '../../_frontend.err.log' -WindowStyle Hidden
    Pop-Location
    Start-Sleep -Seconds 5
}

Write-Host ''
Write-Host 'READY for Playwright. Next steps:' -ForegroundColor Green
Write-Host '    cd frontend/web'
Write-Host "    `$env:DATABASE_URL = 'postgresql://postgres:" + $dbpw.Substring(0,4) + "...@127.0.0.1:5433/investment_db'"
Write-Host '    npx playwright test --project=chromium auth.spec.ts'
Write-Host '    npx playwright test --project=chromium portfolio.spec.ts'
Write-Host '    npx playwright test --project=chromium                     # full chromium pass'
Write-Host ''
Write-Host 'Teardown:'
Write-Host '    docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.e2e-portshift.yml stop postgres redis'