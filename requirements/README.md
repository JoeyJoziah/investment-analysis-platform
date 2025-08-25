# Investment Analysis Platform - Requirements Structure

This directory contains modular requirement files for the investment analysis platform, designed for efficient dependency management across different environments and use cases.

## 📁 File Structure

```
requirements/
├── README.md                 # This documentation
├── base.txt                  # Core dependencies (always required)
├── database.txt              # Database drivers and ORM
├── ml.txt                    # Machine learning frameworks
├── financial.txt             # Financial data and analysis tools
├── data_pipeline.txt         # Orchestration and streaming
├── monitoring.txt            # Observability and metrics
├── development.txt           # Development and testing tools
├── production.txt            # Production-ready full stack
└── constraints.txt           # Version constraints for security/compatibility
```

## 🎯 Usage Patterns

### Full Production Installation
```bash
# Install everything needed for production
python install_dependencies.py -r requirements/production.txt -c requirements/constraints.txt
```

### Development Environment
```bash
# Install core + development tools
python install_dependencies.py -r requirements/base.txt -r requirements/development.txt
```

### Specific Component Installation
```bash
# Just ML components
python install_dependencies.py -r requirements/base.txt -r requirements/ml.txt

# Just database components
python install_dependencies.py -r requirements/database.txt

# Financial analysis only
python install_dependencies.py -r requirements/financial.txt
```

### Docker Multi-stage Builds
```dockerfile
# Stage 1: Base dependencies
FROM python:3.12-slim as base
COPY requirements/base.txt /tmp/
RUN python install_dependencies.py -r /tmp/base.txt

# Stage 2: Add ML dependencies
FROM base as ml
COPY requirements/ml.txt /tmp/
RUN python install_dependencies.py -r /tmp/ml.txt

# Stage 3: Production
FROM ml as production
COPY requirements/production.txt /tmp/
RUN python install_dependencies.py -r /tmp/production.txt
```

## 📦 Package Categories

### 🔧 Base Dependencies (`base.txt`)
- **Purpose**: Core application framework
- **Size**: ~50MB
- **Install Time**: ~30 seconds
- **Contains**: FastAPI, Pydantic, basic utilities
- **System Deps**: Python 3.12+, pip

### 🗄️ Database Dependencies (`database.txt`) 
- **Purpose**: Database connectivity and ORM
- **Size**: ~100MB
- **Install Time**: ~60 seconds
- **Contains**: SQLAlchemy, PostgreSQL drivers, Redis, Elasticsearch
- **System Deps**: libpq-dev, postgresql-client

### 🧠 ML Dependencies (`ml.txt`)
- **Purpose**: Machine learning and AI capabilities
- **Size**: ~2GB (largest component)
- **Install Time**: ~10 minutes
- **Contains**: PyTorch, Transformers, scikit-learn, Prophet
- **System Deps**: build-essential, python3-dev

### 💰 Financial Dependencies (`financial.txt`)
- **Purpose**: Financial data and technical analysis
- **Size**: ~200MB
- **Install Time**: ~90 seconds
- **Contains**: yfinance, TA-Lib, Alpha Vantage, statistical tools
- **System Deps**: ta-lib, libxml2-dev

### 🔄 Data Pipeline Dependencies (`data_pipeline.txt`)
- **Purpose**: Orchestration and streaming
- **Size**: ~300MB
- **Install Time**: ~2 minutes
- **Contains**: Apache Airflow, Kafka clients
- **System Deps**: librdkafka-dev

### 📊 Monitoring Dependencies (`monitoring.txt`)
- **Purpose**: Observability and metrics
- **Size**: ~30MB
- **Install Time**: ~20 seconds
- **Contains**: Prometheus, OpenTelemetry, Sentry
- **System Deps**: None

### 🛠️ Development Dependencies (`development.txt`)
- **Purpose**: Testing and code quality
- **Size**: ~80MB
- **Install Time**: ~45 seconds
- **Contains**: pytest, black, mypy, bandit
- **System Deps**: None

## ⚡ Performance Optimizations

### Parallel Installation
The install script automatically groups packages for parallel installation:
- **Group 1**: System-dependent packages (sequential)
- **Group 2**: C extensions (parallel, limited workers)
- **Group 3**: Pure Python packages (high parallelism)
- **Group 4**: Large ML frameworks (sequential with progress tracking)

### Caching Strategy
```bash
# Pre-download wheels for air-gapped installations
python install_dependencies.py --wheel-dir ./wheels -r requirements/production.txt

# Use cached wheels
python install_dependencies.py --air-gapped --wheel-dir ./wheels
```

### Binary Preferences
- Prefer binary wheels over source compilation
- Fallback to source build with optimized compiler flags
- Smart retry logic with exponential backoff

## 🔒 Security Features

### Hash Verification
```bash
# Enable hash verification (requires hashes in requirements)
python install_dependencies.py --verify-hashes -r requirements/production.txt
```

### Constraints File
The `constraints.txt` file pins critical packages for:
- Security vulnerability prevention
- API compatibility maintenance
- Reproducible builds
- Version conflict resolution

## 🐳 Container Optimization

### Minimal Base Images
```dockerfile
# Use specific component requirements for smaller images
FROM python:3.12-slim

# Only install what's needed
COPY requirements/base.txt requirements/financial.txt /tmp/
RUN python install_dependencies.py -r /tmp/base.txt -r /tmp/financial.txt
```

### Multi-stage Builds
Reduce final image size by using build stages:
- **Stage 1**: Install build dependencies and compile
- **Stage 2**: Copy compiled artifacts to clean runtime image
- **Result**: ~60% smaller final images

## 🚀 CI/CD Integration

### GitHub Actions
```yaml
steps:
  - name: Install system dependencies
    run: ./install_system_deps.sh
    
  - name: Install Python dependencies
    run: |
      python install_dependencies.py \
        -r requirements/base.txt \
        -r requirements/development.txt \
        --max-workers 2 \
        --timeout 300
```

### Dependency Caching
```yaml
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ./wheels
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements/*.txt') }}
```

## 📈 Monitoring Installation

### Progress Tracking
The installer provides detailed progress information:
- Real-time installation status
- Package-by-package timing
- Parallel worker utilization
- Memory and disk usage
- Error rates and retry attempts

### Installation Reports
After completion, you'll get:
- Success/failure statistics
- Performance metrics
- System compatibility warnings
- Optimization recommendations

## 🔧 Troubleshooting

### Common Issues

#### 1. System Dependencies Missing
```bash
# Run system dependency installer first
./install_system_deps.sh --verbose
```

#### 2. Compilation Failures
```bash
# Use binary packages when possible
python install_dependencies.py --prefer-binary
```

#### 3. Network Timeouts
```bash
# Increase timeout and retries
python install_dependencies.py --timeout 600 --retries 5
```

#### 4. Memory Issues
```bash
# Reduce parallel workers
python install_dependencies.py --max-workers 1
```

### Debug Mode
```bash
# Enable verbose logging
python install_dependencies.py --verbose -r requirements/production.txt

# Check installation logs
tail -f installation.log
```

## 📊 Resource Requirements

| Component | Disk Space | RAM | Install Time | System Deps |
|-----------|------------|-----|--------------|-------------|
| Base | 50MB | 100MB | 30s | Python 3.12+ |
| Database | 100MB | 200MB | 60s | libpq-dev |
| ML | 2GB | 4GB | 10m | build-essential |
| Financial | 200MB | 300MB | 90s | ta-lib |
| Pipeline | 300MB | 500MB | 2m | librdkafka-dev |
| Monitoring | 30MB | 50MB | 20s | - |
| Development | 80MB | 150MB | 45s | - |
| **Full Production** | **3GB** | **5GB** | **15m** | **All above** |

## 🎓 Best Practices

1. **Start Small**: Begin with `base.txt` and add components as needed
2. **Use Constraints**: Always use `constraints.txt` for reproducible builds
3. **System Dependencies First**: Run `install_system_deps.sh` before Python packages
4. **Environment Separation**: Use different requirement combinations for dev/staging/prod
5. **Regular Updates**: Keep requirements and constraints files updated
6. **Monitor Resources**: Watch disk space and memory during ML package installation
7. **Cache Wheels**: Pre-build wheels for faster repeated installations
8. **Test Thoroughly**: Validate installations in clean environments

## 📞 Support

For issues with the installation system:
1. Check the installation logs: `installation.log` and `system_deps_install.log`
2. Run with `--verbose` flag for detailed output
3. Verify system dependencies with `./install_system_deps.sh --dry-run`
4. Test individual components before full production installation

---

**Total Dependencies**: 130+ packages  
**Supported Python**: 3.12+  
**Supported Platforms**: Ubuntu 18.04+, Debian 9+, CentOS 7+, Fedora 30+, Arch Linux, macOS 10.15+  
**Installation Method**: Parallel with intelligent grouping  
**Average Install Time**: 15 minutes (full production)
