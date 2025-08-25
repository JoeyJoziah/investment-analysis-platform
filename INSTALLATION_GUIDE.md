# Investment Analysis Platform - Installation Guide

## 🚀 Quick Start

### One-Line Installation
```bash
# Development environment
python install_platform.py dev

# Production environment
python install_platform.py prod
```

### Prerequisites
- Python 3.12+
- 4GB RAM (8GB recommended for ML packages)
- 5GB free disk space
- Internet connection
- Admin/sudo privileges (Linux/macOS)

## 📋 Installation Components

### 🏗️ System Dependencies (`install_system_deps.sh`)
Installs native libraries and build tools:
- Python development headers
- C/C++ compilers and build tools
- Cryptography libraries (OpenSSL, libffi)
- XML processing libraries
- Database client libraries (PostgreSQL)
- Kafka client libraries
- TA-Lib for technical analysis

### 🐍 Python Dependencies (`install_dependencies.py`)
Installs Python packages with:
- Intelligent parallel installation
- Automatic retry with exponential backoff
- Wheel caching for faster installs
- Cross-platform compatibility
- Comprehensive error handling

### 🎯 Platform Orchestrator (`install_platform.py`)
Complete environment setup:
- Coordinates system and Python installations
- Environment-specific configurations
- Post-installation validation
- Comprehensive reporting

## 🌍 Environment Types

| Environment | Use Case | Components | Install Time | Size |
|-------------|----------|------------|--------------|------|
| `minimal` | Base functionality | Core framework only | ~2min | ~100MB |
| `dev` | Development | Base + dev tools | ~5min | ~300MB |
| `test` | CI/CD testing | Base + testing tools | ~5min | ~300MB |
| `staging` | Pre-production | Full production stack | ~15min | ~3GB |
| `prod` | Production | Full stack + optimizations | ~15min | ~3GB |

## 🔧 Installation Methods

### Method 1: Complete Platform Installation (Recommended)
```bash
# Install everything for your environment
python install_platform.py prod --verbose
```

### Method 2: Step-by-Step Installation
```bash
# 1. Install system dependencies
./install_system_deps.sh --verbose

# 2. Install Python dependencies  
python install_dependencies.py -r requirements/production.txt

# 3. Validate installation
python -c "import fastapi, torch, transformers; print('✅ Installation validated')"
```

### Method 3: Modular Installation
```bash
# Install specific components
python install_dependencies.py -r requirements/base.txt
python install_dependencies.py -r requirements/ml.txt
python install_dependencies.py -r requirements/financial.txt
```

## 🎛️ Advanced Options

### Performance Tuning
```bash
# Maximize parallel installation
python install_platform.py prod --max-workers 8

# Longer timeout for slow connections
python install_platform.py prod --timeout 600

# Use binary packages only (faster)
python install_platform.py prod --prefer-binary
```

### Security Options
```bash
# Enable hash verification
python install_platform.py prod --verify-hashes

# Air-gapped installation (pre-downloaded wheels)
python install_dependencies.py --air-gapped --wheel-dir ./wheels
```

### Development Options
```bash
# Dry run to see what would be installed
python install_platform.py dev --dry-run

# Continue despite failures
python install_platform.py dev --force

# Verbose output for debugging
python install_platform.py dev --verbose
```

## 🐧 Platform-Specific Instructions

### Ubuntu/Debian
```bash
# Update package cache
sudo apt update

# Install platform
python install_platform.py prod
```

### CentOS/RHEL/Fedora
```bash
# Enable EPEL repository (CentOS/RHEL)
sudo yum install epel-release

# Install platform
python install_platform.py prod
```

### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install platform
python install_platform.py prod
```

### Windows (WSL)
```bash
# Use Windows Subsystem for Linux
wsl --install
# Then follow Ubuntu instructions
```

## 🐳 Docker Installation

### Development Container
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN python install_platform.py dev
EXPOSE 8000
CMD ["./start.sh", "dev"]
```

### Production Container (Multi-stage)
```dockerfile
# Build stage
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements/ ./requirements/
COPY install_*.py install_*.sh ./
RUN ./install_system_deps.sh --verbose
RUN python install_dependencies.py -r requirements/production.txt

# Runtime stage
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .
EXPOSE 8000
CMD ["./start.sh", "prod"]
```

## ⚙️ CI/CD Integration

### GitHub Actions
```yaml
name: Install Dependencies
steps:
  - uses: actions/checkout@v3
  
  - name: Set up Python
    uses: actions/setup-python@v4
    with:
      python-version: '3.12'
  
  - name: Cache dependencies
    uses: actions/cache@v3
    with:
      path: |
        ~/.cache/pip
        ./wheels
      key: ${{ runner.os }}-deps-${{ hashFiles('requirements/*.txt') }}
  
  - name: Install platform
    run: python install_platform.py test --max-workers 2
```

### GitLab CI
```yaml
install_deps:
  stage: setup
  script:
    - python install_platform.py test
  cache:
    paths:
      - wheels/
      - ~/.cache/pip/
  artifacts:
    reports:
      junit: test-results.xml
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Permission Denied
```bash
# Make scripts executable
chmod +x install_system_deps.sh install_platform.py

# Check sudo access
sudo -v
```

#### 2. Python Version Issues
```bash
# Check Python version
python3 --version

# Use specific Python version
python3.12 install_platform.py dev
```

#### 3. System Dependencies Missing
```bash
# Install system deps separately
./install_system_deps.sh --verbose --force

# Check what's missing
./install_system_deps.sh --dry-run
```

#### 4. Compilation Failures
```bash
# Install build tools
sudo apt install build-essential python3-dev

# Use pre-compiled packages
python install_platform.py dev --prefer-binary
```

#### 5. Network/Timeout Issues
```bash
# Increase timeout
python install_platform.py dev --timeout 900

# Use different index
python install_dependencies.py --index-url https://pypi.org/simple/
```

#### 6. Disk Space Issues
```bash
# Check available space
df -h

# Clean pip cache
pip cache purge

# Install minimal environment first
python install_platform.py minimal
```

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=/path/to/project
python install_platform.py dev --verbose 2>&1 | tee debug.log

# Check installation logs
tail -f platform_install.log
tail -f installation.log
tail -f system_deps_install.log
```

### Validation
```bash
# Test core imports
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"

# Run test suite
pytest tests/ -v

# Check service startup
./start.sh dev
curl http://localhost:8000/health
```

## 📊 Installation Monitoring

### Real-time Progress
```bash
# Monitor installation in another terminal
tail -f platform_install.log | grep -E "(INFO|ERROR|SUCCESS)"

# Watch system resources
watch -n 1 "free -h && df -h"

# Monitor network usage
iftop  # or nethogs
```

### Performance Metrics
```bash
# Time installation
time python install_platform.py prod

# Monitor with detailed stats
/usr/bin/time -v python install_platform.py prod

# Profile Python package installation
python -m cProfile -o install.prof install_dependencies.py -r requirements/production.txt
```

## 🔄 Maintenance

### Updating Dependencies
```bash
# Update requirements files
pip-compile requirements/base.in
pip-compile requirements/production.in

# Reinstall with updates
python install_platform.py prod --force
```

### Cleaning Up
```bash
# Remove installation artifacts
rm -f *.log installation_*.json
rm -rf wheels/ __pycache__/

# Clean Python cache
find . -type d -name "__pycache__" -delete
find . -name "*.pyc" -delete
```

### Health Checks
```bash
# Verify all components
python -c """
import sys
components = ['fastapi', 'pandas', 'torch', 'transformers', 'sqlalchemy']
for comp in components:
    try:
        __import__(comp)
        print(f'✅ {comp}')
    except ImportError:
        print(f'❌ {comp}')
"""
```

## 📈 Performance Benchmarks

### Installation Times (AWS c5.xlarge)
| Environment | Packages | Time | Parallel | Sequential |
|-------------|----------|------|----------|------------|
| minimal | 15 | 1m 30s | 1m 30s | 2m 15s |
| dev | 35 | 4m 15s | 4m 15s | 7m 30s |
| prod | 130+ | 12m 30s | 12m 30s | 25m 45s |

### Resource Usage
| Phase | RAM Peak | Disk I/O | Network |
|-------|----------|----------|----------|
| System Deps | 200MB | Low | 50MB |
| Python Base | 500MB | Medium | 100MB |
| ML Packages | 4GB | High | 2GB |
| Total | 4GB | | 2.2GB |

---

**Need help?** Check the installation logs and run with `--verbose` for detailed output. For complex issues, create a minimal reproduction case and check our troubleshooting guide.
