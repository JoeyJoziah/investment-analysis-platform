#!/bin/bash

# Investment Analysis Platform - Data Loading Startup Script
# This script starts the data loading process for the investment platform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}🚀 Investment Analysis Platform - Data Loading${NC}"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if we can use system Python with yfinance
echo -e "${BLUE}🐍 Checking Python environment...${NC}"
if python3 -c "import yfinance, pandas, sqlalchemy" &> /dev/null; then
    echo -e "${GREEN}✅ Required packages available in system Python${NC}"
    USE_VENV=false
elif [ -d "venv" ]; then
    echo -e "${GREEN}✅ Activating virtual environment${NC}"
    source venv/bin/activate
    USE_VENV=true
    # Test if packages are available in venv
    if ! python -c "import yfinance, pandas, sqlalchemy" &> /dev/null; then
        echo -e "${YELLOW}⚠️  Installing missing packages in virtual environment...${NC}"
        pip install yfinance pandas sqlalchemy psycopg2-binary tqdm
    fi
else
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    USE_VENV=true
    pip install yfinance pandas sqlalchemy psycopg2-binary tqdm
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p logs
mkdir -p scripts/data/cache

# Check if PostgreSQL is running
echo -e "${BLUE}🗄️  Checking database connection...${NC}"
if command -v psql &> /dev/null; then
    if psql "postgresql://postgres:9v1g^OV9XUwzUP6cEgCYgNOE@localhost:5432/investment_db" -c "SELECT 1;" &> /dev/null; then
        echo -e "${GREEN}✅ Database connection successful${NC}"
    else
        echo -e "${YELLOW}⚠️  Database connection failed. Make sure PostgreSQL is running:${NC}"
        echo "   docker-compose up postgres -d"
        echo "   OR ensure connection string is correct in .env"
    fi
else
    echo -e "${YELLOW}⚠️  psql not found. Assuming database is accessible via connection string${NC}"
fi

# Parse command line arguments
STOCKS=10
BACKGROUND=false
RESUME=false
VALIDATE_ONLY=false
MONITOR_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --stocks)
            STOCKS="$2"
            shift 2
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --monitor-only)
            MONITOR_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stocks N        Number of stocks to load (default: 10)"
            echo "  --background      Run in background with monitoring"
            echo "  --resume          Resume from previous session"
            echo "  --validate-only   Only validate existing data"
            echo "  --monitor-only    Only monitor existing pipeline"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Load 10 stocks"
            echo "  $0 --stocks 100 --background # Load 100 stocks in background"
            echo "  $0 --resume --background     # Resume previous session"
            echo "  $0 --validate-only          # Validate existing data"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show configuration
echo -e "${BLUE}⚙️  Configuration:${NC}"
echo "   Stocks to load: $STOCKS"
echo "   Background mode: $BACKGROUND"
echo "   Resume mode: $RESUME"
echo "   Validate only: $VALIDATE_ONLY"
echo "   Monitor only: $MONITOR_ONLY"
echo ""

# Validate only mode
if [ "$VALIDATE_ONLY" = true ]; then
    echo -e "${BLUE}🔍 Running data validation...${NC}"
    if [ "$USE_VENV" = true ]; then
        python scripts/validate_data.py --detailed
    else
        python3 scripts/validate_data.py --detailed
    fi
    exit $?
fi

# Check for existing progress
if [ -f "scripts/data/cache/loading_progress.json" ] && [ "$RESUME" = false ]; then
    echo -e "${YELLOW}⚠️  Previous loading progress found.${NC}"
    echo "   Use --resume to continue from where you left off"
    echo "   Or delete scripts/data/cache/loading_progress.json to start fresh"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}💡 Tip: Use --resume to continue previous session${NC}"
        exit 0
    fi
fi

# Build command arguments
PYTHON_ARGS=""
if [ "$RESUME" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --resume"
fi

if [ "$BACKGROUND" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --background"
fi

if [ "$MONITOR_ONLY" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --monitor-only"
else
    PYTHON_ARGS="$PYTHON_ARGS --stocks $STOCKS"
fi

# Start the data loading process
echo -e "${GREEN}🚀 Starting data loading process...${NC}"
echo ""

if [ "$BACKGROUND" = true ]; then
    echo -e "${BLUE}📊 Running in background mode with real-time monitoring${NC}"
    echo -e "${BLUE}📝 Logs will be written to: logs/data_pipeline.log${NC}"
    echo -e "${BLUE}📊 Progress will be saved to: scripts/data/cache/pipeline_status.json${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring (data loading will continue)${NC}"
    echo ""
fi

# Execute the Python script
if [ "$USE_VENV" = true ]; then
    python scripts/start_data_pipeline.py $PYTHON_ARGS
else
    python3 scripts/start_data_pipeline.py $PYTHON_ARGS
fi

EXIT_CODE=$?

# Handle exit codes
case $EXIT_CODE in
    0)
        echo ""
        echo -e "${GREEN}✅ Data loading completed successfully!${NC}"
        echo ""
        echo -e "${BLUE}📊 Next steps:${NC}"
        echo "   1. Validate the loaded data: ./start_data_loading.sh --validate-only"
        echo "   2. Start the API server: uvicorn backend.api.main:app --reload"
        echo "   3. Access API docs: http://localhost:8000/docs"
        ;;
    1)
        echo ""
        echo -e "${RED}❌ Data loading failed${NC}"
        echo "   Check logs/data_pipeline.log for details"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}⚠️  Data loading completed with warnings${NC}"
        echo "   Some stocks may have failed to load"
        echo "   Check logs/data_pipeline.log for details"
        ;;
    130)
        echo ""
        echo -e "${YELLOW}⏹️  Process interrupted by user${NC}"
        echo "   Use --resume to continue from where you left off"
        ;;
    *)
        echo ""
        echo -e "${RED}❌ Unexpected exit code: $EXIT_CODE${NC}"
        ;;
esac

# Show final status
if [ -f "scripts/data/cache/pipeline_status.json" ]; then
    echo ""
    echo -e "${BLUE}📊 Final Status:${NC}"
    python3 -c "
import json
try:
    with open('scripts/data/cache/pipeline_status.json', 'r') as f:
        status = json.load(f)
    
    progress = status.get('status', {}).get('loading_progress', {})
    print(f'   Completed: {progress.get(\"completed\", 0)}')
    print(f'   Failed: {progress.get(\"failed\", 0)}')
    print(f'   Skipped: {progress.get(\"skipped\", 0)}')
    print(f'   Records loaded: {progress.get(\"total_records\", 0):,}')
    
    api_usage = status.get('status', {}).get('api_usage', {})
    print(f'   API calls today: {api_usage.get(\"daily_calls\", 0)}')
    print(f'   Budget used: {api_usage.get(\"budget_used_pct\", 0):.1f}%')
    
except Exception as e:
    print(f'   Could not read status: {e}')
"
fi

exit $EXIT_CODE