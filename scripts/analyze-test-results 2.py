#!/usr/bin/env python3
"""
Analyze Test Results - Phase 1 Infrastructure Impact
Compares baseline (407/846 = 48.1%) to Phase 1 results
"""

import re
import sys
from pathlib import Path
from datetime import datetime

def parse_test_output(file_path: str):
    """Parse pytest output and extract metrics"""

    with open(file_path, 'r') as f:
        content = f.read()

    # Extract summary line (e.g., "316 failed, 391 passed, 779 warnings, 139 errors in 183.22s")
    # Pattern handles various orderings of passed/failed/errors/skipped
    passed_match = re.search(r'(\d+)\s+passed', content)
    failed_match = re.search(r'(\d+)\s+failed', content)
    error_match = re.search(r'(\d+)\s+errors?', content)
    skipped_match = re.search(r'(\d+)\s+skipped', content)

    if not passed_match:
        print("❌ Could not parse test results")
        return None

    passed = int(passed_match.group(1))
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(error_match.group(1)) if error_match else 0
    skipped = int(skipped_match.group(1)) if skipped_match else 0

    total = passed + errors + failed + skipped
    pass_rate = (passed / total * 100) if total > 0 else 0

    return {
        'passed': passed,
        'errors': errors,
        'failed': failed,
        'skipped': skipped,
        'total': total,
        'pass_rate': pass_rate
    }

def analyze_improvement(baseline, phase1):
    """Compare baseline to Phase 1 results"""

    print("=" * 70)
    print("PHASE 1 TEST INFRASTRUCTURE IMPACT ANALYSIS")
    print("=" * 70)
    print()

    print("📊 BASELINE (Before Infrastructure Fixes)")
    print(f"   Passed:    {baseline['passed']}/{baseline['total']} ({baseline['pass_rate']:.1f}%)")
    print(f"   Errors:    {baseline['errors']}")
    print(f"   Failed:    {baseline['failed']}")
    print()

    print("📊 PHASE 1 (After PostgreSQL Infrastructure)")
    print(f"   Passed:    {phase1['passed']}/{phase1['total']} ({phase1['pass_rate']:.1f}%)")
    print(f"   Errors:    {phase1['errors']}")
    print(f"   Failed:    {phase1['failed']}")
    print()

    # Calculate deltas
    delta_passed = phase1['passed'] - baseline['passed']
    delta_errors = phase1['errors'] - baseline['errors']
    delta_failed = phase1['failed'] - baseline['failed']
    delta_rate = phase1['pass_rate'] - baseline['pass_rate']

    print("📈 IMPROVEMENT")
    print(f"   Tests Fixed:     {delta_passed:+d}")
    print(f"   Error Change:    {delta_errors:+d}")
    print(f"   Failure Change:  {delta_failed:+d}")
    print(f"   Pass Rate:       {delta_rate:+.1f}%")
    print()

    # Target analysis
    target_tests = 659  # 80% of 846
    remaining_gap = target_tests - phase1['passed']

    print("🎯 TARGET ANALYSIS")
    print(f"   Target (80%):    {target_tests}/846 tests")
    print(f"   Current:         {phase1['passed']}/846 tests")
    print(f"   Remaining Gap:   {remaining_gap} tests")
    print()

    # Status assessment
    if phase1['pass_rate'] >= 80:
        status = "✅ PASSED - Ready for staging deployment"
    elif phase1['pass_rate'] >= 70:
        status = "⚠️  CLOSE - Minor fixes needed (~10% gap)"
    elif phase1['pass_rate'] >= 60:
        status = "🔶 MODERATE - Significant work needed (~20% gap)"
    else:
        status = "🔴 CRITICAL - Major fixes required (>20% gap)"

    print("📋 STATUS")
    print(f"   {status}")
    print()

    # Time estimate for remaining work
    if remaining_gap > 0:
        # Estimate: ~5 minutes per test fix
        hours_needed = (remaining_gap * 5) / 60
        print("⏱️  ESTIMATED REMAINING WORK")
        print(f"   Tests to Fix:    {remaining_gap}")
        print(f"   Estimated Time:  {hours_needed:.1f} hours")
        print()

    print("=" * 70)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    # Baseline metrics from docs/deployment/PHASE6_TEST_BASELINE_REPORT.md
    baseline = {
        'passed': 407,
        'errors': 139,
        'failed': 300,
        'skipped': 0,
        'total': 846,
        'pass_rate': 48.1
    }

    if len(sys.argv) < 2:
        print("Usage: python analyze-test-results.py <test_output_file>")
        sys.exit(1)

    phase1 = parse_test_output(sys.argv[1])

    if phase1:
        analyze_improvement(baseline, phase1)
    else:
        print("Failed to parse test results")
        sys.exit(1)
