#!/bin/bash
set -e

echo "Setting up development environment..."

# Create Python virtual environment with optimizations
if [ ! -d "/workspace/venv" ]; then
    python -m venv /workspace/venv --upgrade-deps
    source /workspace/venv/bin/activate
    pip install --upgrade pip setuptools wheel
    
    # Install requirements with parallel processing
    if [ -f "/workspace/requirements.txt" ]; then
        pip install --no-cache-dir -r /workspace/requirements.txt
    fi
fi

# Install frontend dependencies with caching
if [ -d "/workspace/frontend/web" ]; then
    cd /workspace/frontend/web
    if [ ! -d "node_modules" ]; then
        npm ci --prefer-offline --no-audit
    fi
    cd /workspace
fi

# Set up git hooks for performance monitoring
cat > /workspace/.git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Performance check before commit

echo "Running performance checks..."

# Check Python imports are optimized
if git diff --cached --name-only | grep -q '\.py$'; then
    echo "Checking Python import optimization..."
    isort --check-only --diff $(git diff --cached --name-only | grep '\.py$')
fi

# Check for large files
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [ "$size" -gt 5242880 ]; then
            echo "Warning: $file is larger than 5MB ($size bytes)"
        fi
    fi
done
EOF
chmod +x /workspace/.git/hooks/pre-commit

# Create aliases for performance tools
cat >> ~/.zshrc << 'EOF'

# Performance aliases
alias lg='lazygit'
alias gd='git diff --color-moved'
alias rgf='rg --files | rg'
alias fdf='fd --type f'
alias ncdu='ncdu --color dark'
alias htop='htop -d 10'
alias pspy='py-spy top -- python'
alias profile='python -m cProfile -s cumulative'
alias memprofile='python -m memory_profiler'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up -d'
alias dcd='docker-compose down'
alias dcl='docker-compose logs -f'
alias dps='docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# Python environment
alias activate='source /workspace/venv/bin/activate'
alias pytest='python -m pytest'
alias black='python -m black'
alias isort='python -m isort'

# Database connections
alias psql-dev='PGPASSWORD=postgres psql -h localhost -U postgres -d investment_db'
alias redis-cli='redis-cli -h localhost'

# Project shortcuts
alias backend='cd /workspace/backend && activate'
alias frontend='cd /workspace/frontend/web'
alias logs='tail -f /workspace/logs/*.log'
EOF

# Set up ripgrep configuration for better performance
mkdir -p ~/.config
cat > ~/.config/ripgrep << 'EOF'
# Ignore patterns for ripgrep
--glob=!*.pyc
--glob=!*/__pycache__/*
--glob=!*/node_modules/*
--glob=!*/.git/*
--glob=!*/venv/*
--glob=!*/.venv/*
--glob=!*/dist/*
--glob=!*/build/*
--glob=!*.min.js
--glob=!*.min.css
--glob=!*/coverage/*
--glob=!*/.pytest_cache/*
--glob=!*/.mypy_cache/*

# Performance settings
--max-columns=150
--max-columns-preview
--smart-case
--threads=4
EOF

echo "Development environment setup complete!"
echo ""
echo "Quick start commands:"
echo "  activate       - Activate Python virtual environment"
echo "  backend        - Navigate to backend directory"
echo "  frontend       - Navigate to frontend directory"
echo "  dcu            - Start Docker services"
echo "  dcl            - View Docker logs"
echo "  psql-dev       - Connect to PostgreSQL"
echo "  redis-cli      - Connect to Redis"
echo ""
echo "Performance tools:"
echo "  rgf <pattern>  - Fast file search with ripgrep"
echo "  fdf <pattern>  - Fast file find with fd"
echo "  pspy           - Python performance profiler"
echo "  htop           - System resource monitor"
echo "  ncdu           - Disk usage analyzer"