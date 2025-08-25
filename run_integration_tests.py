#!/usr/bin/env python3
"""
Integration Test Runner for Investment Analysis Platform
Comprehensive test execution script with reporting, coverage, and CI/CD integration.
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Comprehensive test runner for integration tests."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.test_dir = self.project_root / "backend" / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test categories and their files
        self.test_categories = {
            "api": ["test_api_integration.py"],
            "database": ["test_database_integration.py"], 
            "data_pipeline": ["test_data_pipeline_integration.py"],
            "websocket": ["test_websocket_integration.py"],
            "security": ["test_security_integration.py"],
            "resilience": ["test_resilience_integration.py"],
            "all": [
                "test_api_integration.py",
                "test_database_integration.py",
                "test_data_pipeline_integration.py", 
                "test_websocket_integration.py",
                "test_security_integration.py",
                "test_resilience_integration.py"
            ]
        }
    
    def setup_environment(self, environment: str = "test"):
        """Setup test environment variables."""
        env_vars = {
            "test": {
                "TESTING": "true",
                "DEBUG": "false",
                "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
                "REDIS_URL": "redis://localhost:6379/1",
                "SECRET_KEY": "test-secret-key-for-testing-only",
                "SKIP_EXTERNAL_API_TESTS": "true",
                "LOG_LEVEL": "WARNING"
            },
            "integration": {
                "TESTING": "true", 
                "DEBUG": "false",
                "DATABASE_URL": os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://test_user:test_pass@localhost/test_investment_db"),
                "REDIS_URL": os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1"),
                "SECRET_KEY": "test-secret-key-for-integration-testing",
                "SKIP_EXTERNAL_API_TESTS": "false",
                "LOG_LEVEL": "INFO"
            },
            "ci": {
                "TESTING": "true",
                "DEBUG": "false", 
                "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
                "REDIS_URL": "redis://localhost:6379/1",
                "SECRET_KEY": "ci-test-secret-key",
                "SKIP_EXTERNAL_API_TESTS": "true",
                "SKIP_SLOW_TESTS": "true",
                "LOG_LEVEL": "ERROR"
            }
        }
        
        env_config = env_vars.get(environment, env_vars["test"])
        
        for key, value in env_config.items():
            os.environ[key] = value
        
        logger.info(f"Environment configured for: {environment}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        checks = []
        
        # Check if pytest is installed
        try:
            import pytest
            checks.append(("pytest", True, f"Version: {pytest.__version__}"))
        except ImportError:
            checks.append(("pytest", False, "Not installed"))
        
        # Check if coverage is available
        try:
            import coverage
            checks.append(("coverage", True, f"Version: {coverage.__version__}"))
        except ImportError:
            checks.append(("coverage", False, "Not installed"))
        
        # Check database connection (if using real database)
        db_url = os.getenv("DATABASE_URL", "")
        if "postgresql" in db_url:
            try:
                import asyncpg
                checks.append(("asyncpg", True, f"Version: {asyncpg.__version__}"))
            except ImportError:
                checks.append(("asyncpg", False, "Required for PostgreSQL"))
        
        # Check Redis connection (if using real Redis)
        redis_url = os.getenv("REDIS_URL", "")
        if "redis://" in redis_url:
            try:
                import redis
                checks.append(("redis", True, f"Version: {redis.__version__}"))
            except ImportError:
                checks.append(("redis", False, "Required for Redis"))
        
        # Report check results
        all_passed = True
        for name, passed, info in checks:
            status = "✓" if passed else "✗"
            logger.info(f"{status} {name}: {info}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def run_tests(
        self,
        categories: List[str] = None,
        markers: List[str] = None,
        coverage: bool = True,
        parallel: bool = False,
        verbose: bool = True,
        fail_fast: bool = False,
        output_format: str = "both"  # "json", "html", "both"
    ) -> Dict:
        """Run integration tests with specified parameters."""
        
        # Determine test files to run
        test_files = []
        categories = categories or ["all"]
        
        for category in categories:
            if category in self.test_categories:
                test_files.extend(self.test_categories[category])
            else:
                logger.warning(f"Unknown test category: {category}")
        
        # Remove duplicates
        test_files = list(set(test_files))
        
        if not test_files:
            logger.error("No test files selected")
            return {"success": False, "error": "No test files"}
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test files
        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                cmd.append(str(test_path))
            else:
                logger.warning(f"Test file not found: {test_path}")
        
        # Add options
        if verbose:
            cmd.append("-v")
        
        if fail_fast:
            cmd.extend(["--maxfail=1"])
        
        # Add markers
        if markers:
            marker_expr = " and ".join(markers)
            cmd.extend(["-m", marker_expr])
        
        # Add parallel execution
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(["-n", "auto"])
            except ImportError:
                logger.warning("pytest-xdist not available, running sequentially")
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=backend",
                "--cov-report=term-missing",
                f"--cov-report=html:{self.reports_dir}/coverage_html",
                f"--cov-report=xml:{self.reports_dir}/coverage.xml",
                "--cov-fail-under=75"
            ])
        
        # Add output formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format in ["json", "both"]:
            cmd.extend([
                f"--json-report",
                f"--json-report-file={self.reports_dir}/test_report_{timestamp}.json"
            ])
        
        if output_format in ["html", "both"]:
            cmd.extend([
                f"--html={self.reports_dir}/test_report_{timestamp}.html",
                "--self-contained-html"
            ])
        
        # Add JUnit XML for CI/CD
        cmd.extend([f"--junit-xml={self.reports_dir}/junit_{timestamp}.xml"])
        
        # Execute tests
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Test files: {', '.join(test_files)}")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            test_result = {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "test_files": test_files,
                "categories": categories
            }
            
            # Log results
            if result.returncode == 0:
                logger.info(f"✓ All tests passed in {duration:.2f} seconds")
            else:
                logger.error(f"✗ Tests failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
            
            # Print stdout for immediate feedback
            if result.stdout:
                print(result.stdout)
            
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out after 30 minutes")
            return {
                "success": False,
                "error": "timeout",
                "duration": 1800
            }
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_summary_report(self, test_results: Dict) -> Dict:
        """Generate summary report from test results."""
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_success": test_results.get("success", False),
            "total_duration": test_results.get("duration", 0),
            "categories_tested": test_results.get("categories", []),
            "test_files": test_results.get("test_files", [])
        }
        
        # Parse stdout for test statistics
        stdout = test_results.get("stdout", "")
        
        # Extract test counts (basic parsing)
        if "passed" in stdout:
            lines = stdout.split('\n')
            for line in lines:
                if "passed" in line and "failed" in line:
                    summary["test_summary_line"] = line.strip()
                    break
        
        # Add coverage information if available
        coverage_file = self.reports_dir / "coverage.xml"
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    summary["coverage"] = {
                        "line_rate": float(coverage_elem.get("line-rate", 0)) * 100,
                        "branch_rate": float(coverage_elem.get("branch-rate", 0)) * 100
                    }
            except Exception as e:
                logger.warning(f"Could not parse coverage data: {e}")
        
        # Save summary
        summary_file = self.reports_dir / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to: {summary_file}")
        return summary
    
    def run_specific_test_suite(self, suite: str) -> Dict:
        """Run specific test suites for different scenarios."""
        
        suites = {
            "smoke": {
                "categories": ["api"],
                "markers": ["not slow"],
                "coverage": False,
                "parallel": False
            },
            "regression": {
                "categories": ["all"],
                "markers": ["not slow"],
                "coverage": True,
                "parallel": True
            },
            "full": {
                "categories": ["all"],
                "markers": [],
                "coverage": True,
                "parallel": True
            },
            "security": {
                "categories": ["security"],
                "markers": ["security"],
                "coverage": True,
                "parallel": False
            },
            "performance": {
                "categories": ["all"],
                "markers": ["performance"],
                "coverage": False,
                "parallel": False
            }
        }
        
        suite_config = suites.get(suite)
        if not suite_config:
            logger.error(f"Unknown test suite: {suite}")
            return {"success": False, "error": f"Unknown suite: {suite}"}
        
        logger.info(f"Running {suite} test suite")
        return self.run_tests(**suite_config)


def main():
    """Main entry point for test runner."""
    
    parser = argparse.ArgumentParser(description="Integration Test Runner")
    
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        choices=list(IntegrationTestRunner(Path(__file__).parent).test_categories.keys()),
        help="Test categories to run"
    )
    
    parser.add_argument(
        "--environment", 
        choices=["test", "integration", "ci"],
        default="test",
        help="Test environment configuration"
    )
    
    parser.add_argument(
        "--suite",
        choices=["smoke", "regression", "full", "security", "performance"],
        help="Predefined test suite to run"
    )
    
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Pytest markers to filter tests"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true", 
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "html", "both"],
        default="both",
        help="Output report format"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites, don't run tests"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IntegrationTestRunner()
    
    # Setup environment
    runner.setup_environment(args.environment)
    
    # Check prerequisites
    if not runner.check_prerequisites():
        logger.error("Prerequisites not met")
        if args.check_only:
            sys.exit(1)
        else:
            logger.info("Continuing anyway...")
    
    if args.check_only:
        logger.info("Prerequisites check completed")
        sys.exit(0)
    
    # Run tests
    if args.suite:
        # Run predefined suite
        result = runner.run_specific_test_suite(args.suite)
    else:
        # Run with custom parameters
        result = runner.run_tests(
            categories=args.categories,
            markers=args.markers,
            coverage=not args.no_coverage,
            parallel=args.parallel,
            fail_fast=args.fail_fast,
            output_format=args.output_format
        )
    
    # Generate summary
    summary = runner.generate_summary_report(result)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Overall Result: {'PASSED' if result['success'] else 'FAILED'}")
    print(f"Duration: {result.get('duration', 0):.2f} seconds")
    print(f"Categories: {', '.join(result.get('categories', []))}")
    print(f"Test Files: {len(result.get('test_files', []))}")
    
    if 'coverage' in summary:
        print(f"Line Coverage: {summary['coverage']['line_rate']:.1f}%")
        print(f"Branch Coverage: {summary['coverage']['branch_rate']:.1f}%")
    
    if 'test_summary_line' in summary:
        print(f"Test Results: {summary['test_summary_line']}")
    
    print(f"Reports Directory: {runner.reports_dir}")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()