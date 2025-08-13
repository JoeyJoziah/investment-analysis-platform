#!/usr/bin/env python3
"""
Comprehensive Test Runner

This script runs all test suites for the Investment Analysis Application
with proper reporting and integration with the CI/CD pipeline.
"""

import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil

class TestRunner:
    """Comprehensive test runner for the investment analysis application"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = datetime.now()
        self.results = {
            'overall_status': 'pending',
            'test_suites': {},
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0,
            'total_errors': 0,
            'execution_time': 0,
            'coverage': {},
            'performance': {},
            'security': {}
        }
    
    def log(self, message: str, level: str = 'INFO'):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: List[str], cwd: Optional[str] = None, 
                   timeout: int = 300) -> Dict[str, Any]:
        """Run command and return result"""
        self.log(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(command)
            }
            
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout}s", 'ERROR')
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout}s',
                'command': ' '.join(command)
            }
        except Exception as e:
            self.log(f"Command failed: {e}", 'ERROR')
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'command': ' '.join(command)
            }
    
    def setup_environment(self):
        """Setup test environment"""
        self.log("Setting up test environment...")
        
        # Ensure required directories exist
        Path("reports").mkdir(exist_ok=True)
        Path("htmlcov").mkdir(exist_ok=True)
        
        # Check required tools
        required_tools = ['pytest', 'black', 'isort', 'flake8', 'mypy']
        
        for tool in required_tools:
            result = self.run_command(['which', tool])
            if not result['success']:
                result = self.run_command(['pip', 'show', tool])
                if not result['success']:
                    self.log(f"Installing {tool}...", 'WARN')
                    install_result = self.run_command(['pip', 'install', tool])
                    if not install_result['success']:
                        self.log(f"Failed to install {tool}", 'ERROR')
                        return False
        
        return True
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality checks"""
        self.log("Running code quality checks...")
        
        # Black formatting check
        self.log("Checking code formatting with Black...")
        result = self.run_command(['black', '--check', '--diff', 'backend/'])
        formatting_passed = result['success']
        
        if not formatting_passed:
            self.log("Code formatting issues found", 'WARN')
            # Auto-format if in fix mode
            if hasattr(self, 'fix_issues') and self.fix_issues:
                self.log("Auto-formatting code...")
                self.run_command(['black', 'backend/'])
        
        # Import sorting check
        self.log("Checking import sorting with isort...")
        result = self.run_command(['isort', '--check-only', '--diff', 'backend/'])
        imports_passed = result['success']
        
        if not imports_passed:
            self.log("Import sorting issues found", 'WARN')
            if hasattr(self, 'fix_issues') and self.fix_issues:
                self.log("Auto-sorting imports...")
                self.run_command(['isort', 'backend/'])
        
        # Linting with flake8
        self.log("Running linting with flake8...")
        result = self.run_command([
            'flake8', 'backend/', 
            '--max-line-length=88', 
            '--extend-ignore=E203,W503'
        ])
        linting_passed = result['success']
        
        # Type checking with mypy
        self.log("Running type checking with mypy...")
        result = self.run_command([
            'mypy', 'backend/', 
            '--ignore-missing-imports',
            '--no-strict-optional'
        ])
        typing_passed = result['success']
        
        code_quality_passed = all([
            formatting_passed, imports_passed, linting_passed, typing_passed
        ])
        
        self.results['test_suites']['code_quality'] = {
            'status': 'passed' if code_quality_passed else 'failed',
            'formatting': formatting_passed,
            'imports': imports_passed,
            'linting': linting_passed,
            'typing': typing_passed
        }
        
        return code_quality_passed
    
    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage"""
        self.log("Running unit tests...")
        
        result = self.run_command([
            'pytest', 
            'backend/tests/test_comprehensive_units.py',
            '--verbose',
            '--cov=backend',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml',
            '--cov-report=term-missing',
            '--cov-fail-under=85',
            '--junitxml=reports/unit-test-results.xml',
            '-m', 'not slow'
        ], timeout=600)
        
        unit_tests_passed = result['success']
        
        # Parse test results
        test_count = 0
        passed_count = 0
        failed_count = 0
        
        if result['stdout']:
            lines = result['stdout'].split('\n')
            for line in lines:
                if 'failed' in line and 'passed' in line:
                    # Parse pytest summary line
                    # Example: "5 failed, 95 passed in 30.5s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'failed,':
                            failed_count = int(parts[i-1])
                        elif part == 'passed':
                            passed_count = int(parts[i-1])
                    test_count = passed_count + failed_count
        
        self.results['test_suites']['unit_tests'] = {
            'status': 'passed' if unit_tests_passed else 'failed',
            'total_tests': test_count,
            'passed': passed_count,
            'failed': failed_count,
            'stdout': result['stdout'][-1000:],  # Last 1000 chars
            'stderr': result['stderr'][-1000:] if result['stderr'] else ''
        }
        
        return unit_tests_passed
    
    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        self.log("Running integration tests...")
        
        result = self.run_command([
            'pytest',
            'backend/tests/test_integration_comprehensive.py',
            '--verbose',
            '--junitxml=reports/integration-test-results.xml',
            '-m', 'not slow',
            '--tb=short'
        ], timeout=900)
        
        integration_tests_passed = result['success']
        
        self.results['test_suites']['integration_tests'] = {
            'status': 'passed' if integration_tests_passed else 'failed',
            'stdout': result['stdout'][-1000:],
            'stderr': result['stderr'][-1000:] if result['stderr'] else ''
        }
        
        return integration_tests_passed
    
    def run_security_tests(self) -> bool:
        """Run security and compliance tests"""
        self.log("Running security and compliance tests...")
        
        # Run security test suite
        result = self.run_command([
            'pytest',
            'backend/tests/test_security_compliance.py',
            '--verbose',
            '--junitxml=reports/security-test-results.xml',
            '--tb=short'
        ], timeout=600)
        
        security_tests_passed = result['success']
        
        # Run bandit security scan
        self.log("Running Bandit security scan...")
        bandit_result = self.run_command([
            'bandit', '-r', 'backend/', '-f', 'json', '-o', 'reports/bandit-report.json'
        ])
        
        # Run safety check
        self.log("Running Safety dependency check...")
        safety_result = self.run_command([
            'safety', 'check', '--json', '--output', 'reports/safety-report.json'
        ])
        
        self.results['test_suites']['security_tests'] = {
            'status': 'passed' if security_tests_passed else 'failed',
            'bandit_passed': bandit_result['success'],
            'safety_passed': safety_result['success'],
            'stdout': result['stdout'][-1000:],
            'stderr': result['stderr'][-1000:] if result['stderr'] else ''
        }
        
        return security_tests_passed
    
    def run_performance_tests(self) -> bool:
        """Run performance tests"""
        self.log("Running performance tests...")
        
        result = self.run_command([
            'pytest',
            'backend/tests/test_performance_load.py',
            '--verbose',
            '--junitxml=reports/performance-test-results.xml',
            '-m', 'performance',
            '--tb=short'
        ], timeout=1800)  # 30 minutes for performance tests
        
        performance_tests_passed = result['success']
        
        self.results['test_suites']['performance_tests'] = {
            'status': 'passed' if performance_tests_passed else 'failed',
            'stdout': result['stdout'][-1000:],
            'stderr': result['stderr'][-1000:] if result['stderr'] else ''
        }
        
        return performance_tests_passed
    
    def run_financial_model_tests(self) -> bool:
        """Run financial model validation tests"""
        self.log("Running financial model validation tests...")
        
        result = self.run_command([
            'pytest',
            'backend/tests/test_financial_model_validation.py',
            '--verbose',
            '--junitxml=reports/financial-model-test-results.xml',
            '--tb=short'
        ], timeout=900)
        
        financial_tests_passed = result['success']
        
        self.results['test_suites']['financial_model_tests'] = {
            'status': 'passed' if financial_tests_passed else 'failed',
            'stdout': result['stdout'][-1000:],
            'stderr': result['stderr'][-1000:] if result['stderr'] else ''
        }
        
        return financial_tests_passed
    
    def generate_coverage_report(self):
        """Generate comprehensive coverage report"""
        self.log("Generating coverage reports...")
        
        # Run coverage analysis script
        if Path('scripts/coverage_analysis.py').exists():
            result = self.run_command([
                'python', 'scripts/coverage_analysis.py',
                '--mode', 'coverage',
                '--coverage-xml', 'coverage.xml',
                '--output-dir', 'reports/coverage'
            ])
            
            if result['success']:
                self.log("Coverage report generated successfully")
            else:
                self.log("Failed to generate coverage report", 'WARN')
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.log("Generating comprehensive test report...")
        
        # Run test report generation script
        if Path('scripts/generate_test_report.py').exists():
            result = self.run_command([
                'python', 'scripts/generate_test_report.py',
                '--security-reports', 'reports/',
                '--coverage-reports', 'reports/',
                '--integration-reports', 'reports/',
                '--performance-reports', 'reports/',
                '--frontend-reports', 'reports/',
                '--output', 'reports/comprehensive-test-report.html'
            ])
            
            if result['success']:
                self.log("Comprehensive test report generated successfully")
            else:
                self.log("Failed to generate comprehensive test report", 'WARN')
    
    def determine_overall_status(self) -> str:
        """Determine overall test status"""
        
        critical_suites = ['unit_tests', 'integration_tests', 'security_tests']
        
        # Check for critical failures
        for suite_name in critical_suites:
            suite = self.results['test_suites'].get(suite_name, {})
            if suite.get('status') == 'failed':
                return 'failed'
        
        # Check for warnings
        warning_conditions = [
            self.results['test_suites'].get('code_quality', {}).get('status') == 'failed',
            self.results['test_suites'].get('performance_tests', {}).get('status') == 'failed',
        ]
        
        if any(warning_conditions):
            return 'warning'
        
        return 'passed'
    
    def run_all_tests(self, include_slow: bool = False, include_performance: bool = True):
        """Run all test suites"""
        
        self.log("Starting comprehensive test suite...")
        
        # Setup
        if not self.setup_environment():
            self.results['overall_status'] = 'error'
            return False
        
        # Track results
        test_results = []
        
        # 1. Code Quality
        try:
            result = self.run_code_quality_checks()
            test_results.append(('code_quality', result))
        except Exception as e:
            self.log(f"Code quality checks failed: {e}", 'ERROR')
            test_results.append(('code_quality', False))
        
        # 2. Unit Tests
        try:
            result = self.run_unit_tests()
            test_results.append(('unit_tests', result))
        except Exception as e:
            self.log(f"Unit tests failed: {e}", 'ERROR')
            test_results.append(('unit_tests', False))
        
        # 3. Integration Tests
        try:
            result = self.run_integration_tests()
            test_results.append(('integration_tests', result))
        except Exception as e:
            self.log(f"Integration tests failed: {e}", 'ERROR')
            test_results.append(('integration_tests', False))
        
        # 4. Security Tests
        try:
            result = self.run_security_tests()
            test_results.append(('security_tests', result))
        except Exception as e:
            self.log(f"Security tests failed: {e}", 'ERROR')
            test_results.append(('security_tests', False))
        
        # 5. Financial Model Tests
        try:
            result = self.run_financial_model_tests()
            test_results.append(('financial_model_tests', result))
        except Exception as e:
            self.log(f"Financial model tests failed: {e}", 'ERROR')
            test_results.append(('financial_model_tests', False))
        
        # 6. Performance Tests (optional)
        if include_performance:
            try:
                result = self.run_performance_tests()
                test_results.append(('performance_tests', result))
            except Exception as e:
                self.log(f"Performance tests failed: {e}", 'ERROR')
                test_results.append(('performance_tests', False))
        
        # Generate reports
        self.generate_coverage_report()
        self.generate_test_report()
        
        # Calculate final results
        self.results['execution_time'] = (datetime.now() - self.start_time).total_seconds()
        self.results['overall_status'] = self.determine_overall_status()
        
        # Summary
        passed_suites = sum(1 for _, result in test_results if result)
        total_suites = len(test_results)
        
        self.log(f"Test Summary: {passed_suites}/{total_suites} suites passed")
        self.log(f"Overall Status: {self.results['overall_status'].upper()}")
        self.log(f"Total Execution Time: {self.results['execution_time']:.1f}s")
        
        # Save results
        with open('reports/test-results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results['overall_status'] in ['passed', 'warning']


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--include-slow', action='store_true', 
                       help='Include slow running tests')
    parser.add_argument('--include-performance', action='store_true', default=True,
                       help='Include performance tests')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--fix-issues', action='store_true',
                       help='Auto-fix code quality issues')
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    runner.fix_issues = args.fix_issues
    
    success = runner.run_all_tests(
        include_slow=args.include_slow,
        include_performance=args.include_performance
    )
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Tests failed with status: {runner.results['overall_status']}")
        sys.exit(1)


if __name__ == '__main__':
    main()