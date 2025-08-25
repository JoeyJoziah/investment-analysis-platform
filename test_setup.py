#!/usr/bin/env python3
"""Test the setup by installing a minimal set of packages."""

import subprocess
import sys

def test_minimal_install():
    """Test minimal package installation."""
    packages = ['requests', 'pip-tools']
    
    print("Testing minimal installation...")
    cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Minimal installation successful!")
        print(f"Installed: {', '.join(packages)}")
        return True
    else:
        print("❌ Installation failed:")
        print(result.stderr)
        return False

if __name__ == "__main__":
    sys.exit(0 if test_minimal_install() else 1)
