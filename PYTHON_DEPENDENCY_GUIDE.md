# Python Dependency Management Guide for WSL

## Quick Fix for Current Errors

### The Problem
- **ModuleNotFoundError: No module named 'aiohttp'** - Missing Python packages
- **Wrong commands used**: `apt install` instead of `pip` for Python packages  
- **Virtual environment not activated** - Running scripts outside project environment

### Immediate Solution

```bash
# 1. Navigate to project directory
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/7b51113a393465a37d4f1fda36b4d190088ac69ea8d5cf2f90400b3c14148ad3

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install core dependencies (includes aiohttp)
pip install -r requirements-core.txt

# 4. Verify aiohttp is installed
python -c "import aiohttp; print('✅ aiohttp installed:', aiohttp.__version__)"

# 5. Run your script
python background_loader_enhanced.py
```

Or run the automated fix script:
```bash
./fix_python_deps.sh
```

## Understanding pip vs apt

### ❌ NEVER Use apt for Python Packages

**Wrong commands that will fail:**
```bash
apt install -r requirements.txt          # apt doesn't understand -r flag
apt install python3-requirements.txt     # No such package exists  
apt install python3-aiohttp              # May install wrong version
```

### ✅ Always Use pip for Python Packages

**Correct commands:**
```bash
pip install -r requirements.txt          # Install from requirements file
pip install aiohttp                      # Install individual package
pip install aiohttp==3.9.1              # Install specific version
pip install --upgrade pip                # Upgrade pip itself
```

### When to Use Each Tool

| Tool | Use Case | Examples |
|------|----------|----------|
| **apt install** | System packages, Python interpreter, system libraries | `python3`, `python3-pip`, `python3-dev`, `build-essential` |
| **pip install** | Python libraries and packages | `fastapi`, `pandas`, `aiohttp`, packages from requirements.txt |

## Virtual Environment Best Practices

### Why Virtual Environments Matter
- **Isolation**: Project dependencies don't conflict with system packages
- **Version Control**: Different projects can use different package versions
- **Clean Environment**: Easy to replicate on different machines
- **WSL Compatibility**: Prevents permission issues with system Python

### Always Activate Before Installing

```bash
# Method 1: Manual activation (every time)
source venv/bin/activate

# Method 2: Create convenient alias
echo "alias activate-inv='cd $(pwd) && source venv/bin/activate'" >> ~/.bashrc
source ~/.bashrc

# Then use: activate-inv
```

### Verify Virtual Environment is Active

```bash
# Check these show venv paths:
which python    # Should show: /path/to/project/venv/bin/python
which pip       # Should show: /path/to/project/venv/bin/pip
echo $VIRTUAL_ENV  # Should show: /path/to/project/venv

# Visual indicator in prompt:
(venv) user@machine:~/project$
```

## Available Requirements Files

### Choose Based on Your Needs

| File | Dependencies | Use Case |
|------|--------------|----------|
| `requirements-minimal.txt` | 23 packages | Quick testing, basic functionality |
| `requirements-core.txt` | 16 packages | **Recommended for development** |
| `requirements.txt` | 131 packages | Full feature set, all analysis tools |
| `requirements.production.txt` | 202 packages | Production deployment |

### Installation Commands

```bash
# For development (recommended)
pip install -r requirements-core.txt

# For minimal testing
pip install -r requirements-minimal.txt

# For full features
pip install -r requirements.txt

# For production
pip install -r requirements.production.txt
```

## Troubleshooting Common Issues

### Issue 1: "ModuleNotFoundError" Despite Installing

**Cause**: Virtual environment not activated

**Solution**:
```bash
source venv/bin/activate  # Must do this first
pip install [package]     # Then install
```

### Issue 2: "Permission denied" Errors

**Cause**: Trying to install to system Python

**Solution**:
```bash
# Don't use sudo with pip in virtual environment
pip install [package]     # ✅ Correct

# Avoid these:
sudo pip install [package]  # ❌ Wrong
```

### Issue 3: "Package not found" with apt

**Cause**: Using apt for Python packages

**Solution**:
```bash
# Use pip instead
pip install aiohttp       # ✅ Correct
apt install python3-aiohttp  # ❌ May be wrong version
```

### Issue 4: Slow Installation in WSL

**Solutions**:
```bash
# Use faster mirror
pip install -i https://pypi.python.org/simple/ aiohttp

# Install with cache
pip install --cache-dir ~/.pip/cache aiohttp

# Upgrade pip first
python -m pip install --upgrade pip
```

## Development Workflow

### Daily Development Routine

1. **Start development session:**
   ```bash
   cd /path/to/project
   source venv/bin/activate
   ```

2. **Install new packages as needed:**
   ```bash
   pip install new_package
   ```

3. **Update requirements when adding packages:**
   ```bash
   pip freeze > requirements-dev.txt
   ```

4. **Run your scripts:**
   ```bash
   python background_loader_enhanced.py
   python -m backend.api.main
   ```

### VS Code Integration

**Setup automatic virtual environment detection:**

1. Open project in VS Code: `code .`
2. Press `Ctrl+Shift+P` → "Python: Select Interpreter"  
3. Choose `./venv/bin/python`
4. VS Code will automatically activate venv in integrated terminal

## Emergency Recovery

### If Virtual Environment is Broken

```bash
# Remove and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-core.txt
```

### If pip is Broken

```bash
# Reinstall pip
curl https://bootstrap.pypa.io/get-pip.py | python
```

### If Python Dependencies are Corrupted

```bash
# Clean install approach
python3 -m venv venv_clean
source venv_clean/bin/activate
pip install --no-cache-dir -r requirements-core.txt
mv venv venv_backup
mv venv_clean venv
```

## Automated Setup Script

Use the provided `fix_python_deps.sh` for automated setup:

```bash
./fix_python_deps.sh
```

This script will:
- ✅ Activate virtual environment
- ✅ Upgrade pip
- ✅ Install core dependencies (including aiohttp)
- ✅ Verify installation
- ✅ Test imports
- ✅ Create helper scripts

## Quick Reference Commands

```bash
# Essential commands to remember:
source venv/bin/activate                 # Activate environment
pip install -r requirements-core.txt    # Install dependencies
python background_loader_enhanced.py    # Run your script
deactivate                              # Exit virtual environment

# Check status:
which python                            # Verify using venv Python
pip list                               # Show installed packages
python -c "import aiohttp"             # Test specific import
```

## Platform-Specific Notes for WSL

### WSL2 Considerations
- **File permissions**: Virtual environments work better in WSL filesystem (`~/projects/`) vs Windows mounts (`/mnt/c/`)
- **Performance**: Installing packages is faster in WSL filesystem
- **Path issues**: Use forward slashes, avoid Windows path formats

### WSL1 vs WSL2
- **WSL2** (recommended): Better performance, full Linux compatibility
- **WSL1**: May have file permission issues with pip

Check your WSL version: `wsl -l -v`

---

**Remember**: Never use `apt install` for Python packages from requirements.txt files. Always use `pip install` within an activated virtual environment.