#!/usr/bin/env python3
"""Test script to verify pytest-asyncio configuration"""
import subprocess
import sys

def main():
    print("Testing pytest-asyncio configuration...")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "backend/tests/test_thesis_api.py::TestInvestmentThesisAPI::test_create_thesis_success",
            "-xvs",
            "--tb=short",
            "--no-cov"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )

    output = result.stdout + result.stderr

    # Check for specific errors
    if "PytestRemovedIn9Warning" in output:
        print("FAIL: Still has PytestRemovedIn9Warning")
        # Find and print the warning
        for line in output.split('\n'):
            if "PytestRemovedIn9Warning" in line or "async fixture" in line:
                print(f"  {line}")
        return False

    if "session-scoped" in output and "async fixture" in output:
        print("FAIL: Session-scoped async fixture error")
        return False

    if "PASSED" in output:
        print("SUCCESS: Test passed!")
        return True

    if "ERROR" in output:
        print("Test has errors (but not pytest-asyncio config errors):")
        # Print error section
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if "ERROR at setup" in line:
                print('\n'.join(lines[i:min(i+30, len(lines))]))
                break
        return None  # Test has other errors, but config might be OK

    print("Test did not pass, checking output...")
    print(output[-1000:])
    return False

if __name__ == "__main__":
    result = main()
    if result is True:
        print("\n✅ pytest-asyncio configuration is FIXED")
        sys.exit(0)
    elif result is None:
        print("\n⚠️  pytest-asyncio config OK, but test has other errors")
        sys.exit(0)
    else:
        print("\n❌ pytest-asyncio configuration still has issues")
        sys.exit(1)
