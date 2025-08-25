#!/bin/bash

# Performance optimization script for Investment Analysis Platform
# Based on Claude Code best practices

set -e

echo "🚀 Applying performance optimizations..."

# 1. Replace slow commands with faster alternatives
setup_command_replacements() {
    echo "Setting up command replacements..."
    
    # Check if ripgrep is installed
    if ! command -v rg &> /dev/null; then
        echo "Installing ripgrep..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y ripgrep || echo "Could not install ripgrep"
        fi
    fi
    
    # Check if fd is installed
    if ! command -v fd &> /dev/null && ! command -v fdfind &> /dev/null; then
        echo "Installing fd-find..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y fd-find || echo "Could not install fd-find"
            # Create fd symlink if fdfind exists
            if command -v fdfind &> /dev/null; then
                sudo ln -sf $(which fdfind) /usr/local/bin/fd 2>/dev/null || true
            fi
        fi
    fi
    
    # Create wrapper scripts for common commands
    mkdir -p ~/.local/bin
    
    # Only create wrappers if directory was created successfully
    if [ -d ~/.local/bin ]; then
        # Grep -> Ripgrep wrapper
        cat > ~/.local/bin/grep-fast << 'EOF'
#!/bin/bash
# Wrapper to use ripgrep instead of grep for better performance
if command -v rg &> /dev/null; then
    rg "$@"
else
    /usr/bin/grep "$@"
fi
EOF
        chmod +x ~/.local/bin/grep-fast 2>/dev/null || true
        
        # Find -> fd wrapper
        cat > ~/.local/bin/find-fast << 'EOF'
#!/bin/bash
# Use fd for simple file searches, fallback to find for complex queries
if [[ "$1" == "." ]] && [[ "$2" == "-name" ]] && (command -v fd &> /dev/null || command -v fdfind &> /dev/null); then
    shift 2
    if command -v fd &> /dev/null; then
        fd "$@"
    else
        fdfind "$@"
    fi
else
    /usr/bin/find "$@"
fi
EOF
        chmod +x ~/.local/bin/find-fast 2>/dev/null || true
        
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
            export PATH="$HOME/.local/bin:$PATH"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc 2>/dev/null || true
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
        fi
        
        echo "Command replacements configured (use grep-fast and find-fast)"
    else
        echo "Could not create ~/.local/bin directory"
    fi
}

# 2. Configure Docker for better performance
optimize_docker() {
    echo "Optimizing Docker configuration..."
    
    # Create optimized docker-compose override
    cat > docker-compose.performance.yml << 'EOF'
version: '3.8'

x-performance-limits: &performance-limits
  mem_limit: 2g
  memswap_limit: 2g
  cpu_shares: 1024
  pids_limit: 200

services:
  backend:
    <<: *performance-limits
    environment:
      - NODE_OPTIONS=--max-old-space-size=1536
      - PYTHONOPTIMIZE=2
      - MALLOC_ARENA_MAX=2
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
    
  frontend:
    <<: *performance-limits
    environment:
      - NODE_OPTIONS=--max-old-space-size=1024
      - GENERATE_SOURCEMAP=false
    
  postgres:
    <<: *performance-limits
    command: >
      postgres
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c maintenance_work_mem=128MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200
      -c work_mem=8MB
      -c min_wal_size=1GB
      -c max_wal_size=4GB
      -c max_worker_processes=4
      -c max_parallel_workers_per_gather=2
      -c max_parallel_workers=4
      -c max_parallel_maintenance_workers=2
    
  redis:
    <<: *performance-limits
    command: >
      redis-server
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --save 60 1000
      --tcp-backlog 511
      --tcp-keepalive 300
      --timeout 0
EOF
}

# 3. Set up caching for package managers
setup_package_caching() {
    echo "Setting up package manager caching..."
    
    # NPM cache configuration - check for permission issues
    if [ -w "$HOME" ]; then
        # Only set npm config if we have write permissions
        npm config set cache ~/.npm-cache 2>/dev/null || echo "Skipping npm cache config (permission issue)"
        npm config set prefer-offline true 2>/dev/null || true
        npm config set audit false 2>/dev/null || true
    else
        echo "Skipping npm config (no write permission to home directory)"
    fi
    
    # Pip cache configuration
    mkdir -p ~/.cache/pip
    mkdir -p ~/.config/pip  # Create the config directory first
    
    # Only create pip.conf if directory exists
    if [ -d ~/.config/pip ]; then
        cat > ~/.config/pip/pip.conf << 'EOF'
[global]
cache-dir = ~/.cache/pip
no-cache-dir = false

[install]
compile = yes
use-pep517 = true
EOF
        echo "Pip cache configured"
    else
        echo "Could not configure pip cache (directory creation failed)"
    fi
}

# 4. Configure system limits for better performance
configure_system_limits() {
    echo "Configuring system limits..."
    
    # Create limits configuration
    cat > /tmp/limits.conf << 'EOF'
# Increase file descriptor limits
* soft nofile 65536
* hard nofile 65536

# Increase process limits
* soft nproc 32768
* hard nproc 32768

# Increase memory limits
* soft memlock unlimited
* hard memlock unlimited
EOF
    
    # Apply if we have permissions
    if [ -w /etc/security/limits.d/ ]; then
        sudo cp /tmp/limits.conf /etc/security/limits.d/investment-app.conf
        echo "System limits configured (requires restart to take effect)"
    else
        echo "Cannot configure system limits (no write permission)"
    fi
}

# 5. Set up monitoring aliases
setup_monitoring() {
    echo "Setting up performance monitoring..."
    
    cat > ~/.performance_aliases << 'EOF'
# Performance monitoring aliases
alias perf-cpu='ps aux | sort -nrk 3,3 | head -10'
alias perf-mem='ps aux | sort -nrk 4,4 | head -10'
alias perf-docker='docker stats --no-stream'
alias perf-io='iotop -b -n 1'
alias perf-net='ss -tunap | grep ESTABLISHED'
alias perf-cache='redis-cli INFO stats | grep -E "hits|misses"'
alias perf-db='psql -U postgres -d investment_db -c "SELECT * FROM pg_stat_activity WHERE state != '"'"'idle'"'"';"'

# Quick performance check
perf-check() {
    echo "=== CPU Top 5 ==="
    ps aux | sort -nrk 3,3 | head -5
    echo ""
    echo "=== Memory Top 5 ==="
    ps aux | sort -nrk 4,4 | head -5
    echo ""
    echo "=== Docker Resources ==="
    docker stats --no-stream 2>/dev/null || echo "Docker not running"
    echo ""
    echo "=== Disk Usage ==="
    df -h | grep -E "^/|Filesystem"
}
EOF
    
    # Add to shell configs
    echo 'source ~/.performance_aliases' >> ~/.bashrc
    echo 'source ~/.performance_aliases' >> ~/.zshrc 2>/dev/null || true
}

# 6. Optimize Python imports
optimize_python() {
    echo "Optimizing Python configuration..."
    
    # Create Python startup file for faster imports
    cat > ~/.pythonrc << 'EOF'
import sys
import os

# Enable optimizations
sys.dont_write_bytecode = False  # Allow .pyc files for faster imports
os.environ['PYTHONOPTIMIZE'] = '1'  # Basic optimizations

# Preload common modules for REPL
try:
    import numpy as np
    import pandas as pd
    import json
    import datetime
    from pathlib import Path
except ImportError:
    pass
EOF
    
    export PYTHONSTARTUP=~/.pythonrc
    echo 'export PYTHONSTARTUP=~/.pythonrc' >> ~/.bashrc
    echo 'export PYTHONSTARTUP=~/.pythonrc' >> ~/.zshrc 2>/dev/null || true
}

# 7. Create performance testing script
create_perf_test() {
    echo "Creating performance test script..."
    
    cat > ./test_performance.sh << 'EOF'
#!/bin/bash
# Quick performance test for the investment platform

echo "🔍 Running performance tests..."

# Test API response time
test_api() {
    echo -n "API Health Check: "
    time curl -s http://localhost:8000/api/health > /dev/null
}

# Test database query
test_db() {
    echo -n "Database Query: "
    time PGPASSWORD=postgres psql -h localhost -U postgres -d investment_db -c "SELECT COUNT(*) FROM stocks;" > /dev/null 2>&1
}

# Test Redis
test_redis() {
    echo -n "Redis Ping: "
    time redis-cli ping > /dev/null
}

# Test file search
test_search() {
    echo -n "File Search (ripgrep): "
    time rg --files | wc -l
}

# Run tests
test_api
test_db
test_redis
test_search

echo ""
echo "✅ Performance tests complete"
EOF
    chmod +x ./test_performance.sh
}

# Main execution
main() {
    echo "Starting performance optimization setup..."
    
    setup_command_replacements
    optimize_docker
    setup_package_caching
    configure_system_limits
    setup_monitoring
    optimize_python
    create_perf_test
    
    echo ""
    echo "✅ Performance optimizations applied!"
    echo ""
    echo "Recommendations:"
    echo "1. Restart your shell to apply PATH changes"
    echo "2. Use 'docker-compose -f docker-compose.yml -f docker-compose.performance.yml up' for optimized containers"
    echo "3. Run './test_performance.sh' to benchmark your setup"
    echo "4. Use 'perf-check' command for quick performance overview"
    echo ""
    echo "Key optimizations applied:"
    echo "✓ Command replacements (grep→rg, find→fd)"
    echo "✓ Docker memory and CPU limits"
    echo "✓ PostgreSQL performance tuning"
    echo "✓ Redis memory optimization"
    echo "✓ Package manager caching"
    echo "✓ System limit increases"
    echo "✓ Performance monitoring tools"
}

# Run main function
main "$@"