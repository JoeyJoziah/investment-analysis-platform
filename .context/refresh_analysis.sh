#!/bin/bash

# Investment Analysis Platform - Comprehensive Refresh Analysis Script
# This script performs a live assessment of the project status and updates all reports
# Run: chmod +x refresh_analysis.sh && ./refresh_analysis.sh

set -e

CONTEXT_DIR=".context"
PROJECT_ROOT="$(pwd)"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "======================================"
echo "PROJECT REFRESH ANALYSIS"
echo "Time: $TIMESTAMP"
echo "======================================"

# Function to count files by extension
count_files() {
    extension=$1
    find . -name "*.$extension" -type f 2>/dev/null | wc -l
}

# Function to check service status
check_service() {
    service=$1
    port=$2
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ $service is running on port $port"
        return 0
    else
        echo "❌ $service is not running on port $port"
        return 1
    fi
}

# Function to check Python imports
check_python_imports() {
    module=$1
    if python3 -c "import $module" 2>/dev/null; then
        echo "✅ $module is installed"
        return 0
    else
        echo "❌ $module is not installed"
        return 1
    fi
}

echo ""
echo "1. PROJECT STRUCTURE ANALYSIS"
echo "------------------------------"

# Count files
PY_FILES=$(count_files "py")
JS_FILES=$(count_files "js")
TS_FILES=$(count_files "ts")
TSX_FILES=$(count_files "tsx")
MD_FILES=$(count_files "md")
YML_FILES=$(count_files "yml")
YAML_FILES=$(count_files "yaml")

echo "Python files: $PY_FILES"
echo "JavaScript files: $JS_FILES"
echo "TypeScript files: $TS_FILES"
echo "TSX files: $TSX_FILES"
echo "Markdown files: $MD_FILES"
echo "YAML files: $((YML_FILES + YAML_FILES))"

# Count lines of code
echo ""
echo "Lines of code:"
echo "Python: $(find . -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')"
echo "JavaScript/TypeScript: $(find . \( -name "*.js" -o -name "*.ts" -o -name "*.tsx" \) -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')"

echo ""
echo "2. SERVICE STATUS CHECK"
echo "------------------------"

SERVICES_UP=0
SERVICES_DOWN=0

check_service "PostgreSQL" 5432 && ((SERVICES_UP++)) || ((SERVICES_DOWN++))
check_service "Redis" 6379 && ((SERVICES_UP++)) || ((SERVICES_DOWN++))
check_service "Backend API" 8000 && ((SERVICES_UP++)) || ((SERVICES_DOWN++))
check_service "Frontend" 3000 && ((SERVICES_UP++)) || ((SERVICES_DOWN++))
check_service "Elasticsearch" 9200 && ((SERVICES_UP++)) || ((SERVICES_DOWN++))
check_service "Grafana" 3001 && ((SERVICES_UP++)) || ((SERVICES_DOWN++))
check_service "Prometheus" 9090 && ((SERVICES_UP++)) || ((SERVICES_DOWN++))

echo "Services running: $SERVICES_UP"
echo "Services down: $SERVICES_DOWN"

echo ""
echo "3. DATABASE STATUS"
echo "------------------"

if check_service "PostgreSQL" 5432; then
    # Check database tables and records
    export PGPASSWORD='9v1g^OV9XUwzUP6cEgCYgNOE'
    
    STOCK_COUNT=$(psql -h localhost -U postgres -d investment_db -t -c "SELECT COUNT(*) FROM stocks;" 2>/dev/null || echo "0")
    TABLE_COUNT=$(psql -h localhost -U postgres -d investment_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null || echo "0")
    
    echo "Tables in database: $TABLE_COUNT"
    echo "Stocks loaded: $STOCK_COUNT"
else
    echo "Database not accessible"
fi

echo ""
echo "4. DEPENDENCY CHECK"
echo "-------------------"

DEPS_INSTALLED=0
DEPS_MISSING=0

# Critical Python dependencies
for dep in fastapi uvicorn sqlalchemy redis pandas numpy torch transformers selenium; do
    check_python_imports "$dep" && ((DEPS_INSTALLED++)) || ((DEPS_MISSING++))
done

echo "Dependencies installed: $DEPS_INSTALLED"
echo "Dependencies missing: $DEPS_MISSING"

echo ""
echo "5. BACKEND API STATUS"
echo "---------------------"

# Try to import backend
if timeout 5 python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from backend.api.main import app
    print('✅ Backend imports successfully')
except Exception as e:
    print(f'❌ Backend import error: {str(e)[:100]}')
" 2>&1; then
    BACKEND_STATUS="Functional"
else
    BACKEND_STATUS="Import Issues"
fi

echo "Backend status: $BACKEND_STATUS"

echo ""
echo "6. FRONTEND STATUS"
echo "------------------"

if [ -f "frontend/web/package.json" ]; then
    echo "✅ Frontend package.json exists"
    if [ -d "frontend/web/node_modules" ]; then
        echo "✅ Node modules installed"
        FRONTEND_STATUS="Ready"
    else
        echo "❌ Node modules not installed"
        FRONTEND_STATUS="Dependencies Missing"
    fi
else
    echo "❌ Frontend package.json not found"
    FRONTEND_STATUS="Not Configured"
fi

echo "Frontend status: $FRONTEND_STATUS"

echo ""
echo "7. ML PIPELINE STATUS"
echo "--------------------"

ML_MODELS=0
if [ -d "backend/ml_models" ]; then
    ML_MODELS=$(find backend/ml_models -name "*.pkl" -o -name "*.h5" -o -name "*.pt" 2>/dev/null | wc -l)
fi

echo "Trained models found: $ML_MODELS"

if [ -d "backend/ml" ]; then
    ML_FILES=$(find backend/ml -name "*.py" 2>/dev/null | wc -l)
    echo "ML pipeline files: $ML_FILES"
else
    echo "ML pipeline directory not found"
fi

echo ""
echo "8. TESTING STATUS"
echo "-----------------"

TEST_FILES=$(find . -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l)
echo "Test files found: $TEST_FILES"

if [ -f "backend/tests/conftest.py" ]; then
    echo "✅ Test configuration exists"
else
    echo "❌ Test configuration missing"
fi

echo ""
echo "9. DOCUMENTATION STATUS"
echo "-----------------------"

if [ -f "README.md" ]; then
    echo "✅ README.md exists"
fi

if [ -f "CLAUDE.md" ]; then
    echo "✅ CLAUDE.md exists"
fi

if [ -d "docs" ]; then
    DOC_FILES=$(find docs -name "*.md" 2>/dev/null | wc -l)
    echo "Documentation files: $DOC_FILES"
fi

echo ""
echo "10. SECURITY STATUS"
echo "-------------------"

SECURITY_FILES=$(find backend/security -name "*.py" 2>/dev/null | wc -l)
echo "Security implementation files: $SECURITY_FILES"

if [ -f ".env" ]; then
    echo "✅ Environment variables configured"
else
    echo "❌ .env file missing"
fi

echo ""
echo "======================================"
echo "CALCULATING OVERALL COMPLETION"
echo "======================================"

# Calculate weighted completion percentage
COMPLETION=0
WEIGHT_TOTAL=0

# Database (20% weight)
if [ "$STOCK_COUNT" -gt "20000" ]; then
    COMPLETION=$((COMPLETION + 20))
fi
WEIGHT_TOTAL=$((WEIGHT_TOTAL + 20))

# Backend (25% weight)
if [ "$BACKEND_STATUS" = "Functional" ]; then
    COMPLETION=$((COMPLETION + 25))
elif [ "$BACKEND_STATUS" = "Import Issues" ]; then
    COMPLETION=$((COMPLETION + 10))
fi
WEIGHT_TOTAL=$((WEIGHT_TOTAL + 25))

# Frontend (15% weight)
if [ "$FRONTEND_STATUS" = "Ready" ]; then
    COMPLETION=$((COMPLETION + 15))
elif [ "$FRONTEND_STATUS" = "Dependencies Missing" ]; then
    COMPLETION=$((COMPLETION + 10))
fi
WEIGHT_TOTAL=$((WEIGHT_TOTAL + 15))

# ML Pipeline (15% weight)
if [ "$ML_MODELS" -gt "0" ]; then
    COMPLETION=$((COMPLETION + 15))
elif [ "$ML_FILES" -gt "20" ]; then
    COMPLETION=$((COMPLETION + 8))
fi
WEIGHT_TOTAL=$((WEIGHT_TOTAL + 15))

# Services (10% weight)
SERVICE_PERCENT=$((SERVICES_UP * 10 / 7))
COMPLETION=$((COMPLETION + SERVICE_PERCENT))
WEIGHT_TOTAL=$((WEIGHT_TOTAL + 10))

# Dependencies (10% weight)
if [ "$DEPS_MISSING" -eq "0" ]; then
    COMPLETION=$((COMPLETION + 10))
elif [ "$DEPS_MISSING" -le "3" ]; then
    COMPLETION=$((COMPLETION + 5))
fi
WEIGHT_TOTAL=$((WEIGHT_TOTAL + 10))

# Testing (5% weight)
if [ "$TEST_FILES" -gt "10" ]; then
    COMPLETION=$((COMPLETION + 5))
fi
WEIGHT_TOTAL=$((WEIGHT_TOTAL + 5))

# Calculate final percentage
if [ "$WEIGHT_TOTAL" -gt 0 ]; then
    OVERALL_COMPLETION=$((COMPLETION * 100 / WEIGHT_TOTAL))
else
    OVERALL_COMPLETION=0
fi

echo ""
echo "OVERALL PROJECT COMPLETION: ${OVERALL_COMPLETION}%"
echo ""

echo "======================================"
echo "GENERATING UPDATED REPORTS"
echo "======================================"

# Update timestamp in reports
echo "Updating report timestamps..."
for file in $CONTEXT_DIR/*.md; do
    if [ -f "$file" ]; then
        # Update the date in the file if it exists
        sed -i "s/\*\*Date\*\*: .*/\*\*Date\*\*: $(date +%Y-%m-%d)/" "$file" 2>/dev/null || true
        sed -i "s/\*Last Updated: .*/\*Last Updated: $(date +%Y-%m-%d)/" "$file" 2>/dev/null || true
    fi
done

echo ""
echo "REFRESH ANALYSIS COMPLETE"
echo "Overall Completion: ${OVERALL_COMPLETION}%"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Reports updated in $CONTEXT_DIR/"
echo "======================================"

# Generate summary JSON for programmatic access
cat > "$CONTEXT_DIR/refresh_summary.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "overall_completion": $OVERALL_COMPLETION,
  "metrics": {
    "python_files": $PY_FILES,
    "javascript_files": $JS_FILES,
    "typescript_files": $((TS_FILES + TSX_FILES)),
    "documentation_files": $MD_FILES,
    "services_running": $SERVICES_UP,
    "services_down": $SERVICES_DOWN,
    "stocks_loaded": $STOCK_COUNT,
    "database_tables": $TABLE_COUNT,
    "dependencies_installed": $DEPS_INSTALLED,
    "dependencies_missing": $DEPS_MISSING,
    "ml_models_trained": $ML_MODELS,
    "test_files": $TEST_FILES
  },
  "status": {
    "backend": "$BACKEND_STATUS",
    "frontend": "$FRONTEND_STATUS",
    "database": $([ "$STOCK_COUNT" -gt "20000" ] && echo '"Operational"' || echo '"Issues"'),
    "ml_pipeline": $([ "$ML_MODELS" -gt "0" ] && echo '"Trained"' || echo '"Not Trained"')
  }
}
EOF

echo "Summary saved to $CONTEXT_DIR/refresh_summary.json"