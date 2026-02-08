#!/usr/bin/env python3
"""
Documentation Health Validation Script

Purpose: Comprehensive Python-based documentation validation
Usage: python scripts/validate_doc_health.py [--output FORMAT] [--strict]
"""

import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys


class DocumentationHealthValidator:
    """Validates documentation health metrics and quality"""

    def __init__(self, docs_dir: str = "./docs"):
        self.docs_dir = Path(docs_dir)
        self.violations = []
        self.metrics = {}
        self.checks_run = 0
        self.total_score = 0.0

    def validate_all(self) -> Dict:
        """Run all validation checks and return results"""

        checks = {
            "coverage": self.check_coverage(),
            "recency": self.check_recency(),
            "completeness": self.check_completeness(),
            "links": self.check_links(),
            "code_examples": self.check_code_examples(),
            "formatting": self.check_formatting(),
            "metadata": self.check_metadata(),
            "readability": self.check_readability(),
        }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "violations": self.violations,
            "metrics": self.metrics,
            "health_score": self.calculate_health_score(checks),
            "summary": self.generate_summary(checks),
        }

    def check_coverage(self) -> Dict:
        """Check documentation coverage"""

        if not self.docs_dir.exists():
            return {
                "name": "Coverage",
                "status": "fail",
                "message": f"Docs directory not found: {self.docs_dir}",
                "value": 0,
            }

        doc_count = len(list(self.docs_dir.glob("**/*.md")))
        backend_files = len(list(Path("./backend").glob("**/*.py")))
        frontend_files = len(list(Path("./frontend").glob("**/*.{ts,tsx}")))
        total_modules = backend_files + frontend_files

        coverage_percent = 0
        if total_modules > 0:
            coverage_percent = (doc_count / total_modules) * 100

        status = "pass" if coverage_percent >= 95 else "warning"

        return {
            "name": "Coverage",
            "status": status,
            "doc_files": doc_count,
            "module_files": total_modules,
            "coverage_percent": round(coverage_percent, 1),
            "target": 95,
        }

    def check_recency(self) -> Dict:
        """Check documentation recency"""

        now = datetime.now().timestamp()
        ages = []
        stale_files = []

        docs = list(self.docs_dir.glob("**/*.md"))
        if not docs:
            return {
                "name": "Recency",
                "status": "warning",
                "message": "No documentation files found",
                "average_age_days": 0,
            }

        for doc_file in docs:
            mod_time = doc_file.stat().st_mtime
            age_days = (now - mod_time) / 86400
            ages.append(age_days)

            if age_days > 30:
                stale_files.append({
                    "file": str(doc_file.relative_to(self.docs_dir)),
                    "age_days": round(age_days, 1),
                })
                self.violations.append({
                    "file": str(doc_file),
                    "issue": f"Document is {age_days:.0f} days old",
                    "severity": "medium",
                })

        avg_age = sum(ages) / len(ages) if ages else 0
        max_age = max(ages) if ages else 0

        status = "pass" if avg_age < 30 else "warning" if avg_age < 45 else "fail"

        return {
            "name": "Recency",
            "status": status,
            "average_age_days": round(avg_age, 1),
            "max_age_days": round(max_age, 1),
            "stale_files": len(stale_files),
            "target_max_days": 30,
            "stale_files_detail": stale_files[:5],  # Show first 5
        }

    def check_completeness(self) -> Dict:
        """Check documentation completeness"""

        required_sections = [
            "Overview",
            "Installation",
            "Setup",
            "Usage",
            "Configuration",
            "Examples",
            "Troubleshooting",
        ]

        incomplete_items = []
        complete_count = 0
        total_count = 0

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text(encoding="utf-8", errors="ignore")
            total_count += 1

            missing_sections = []
            for section in required_sections:
                if f"## {section}" not in content and f"# {section}" not in content:
                    missing_sections.append(section)

            if not missing_sections:
                complete_count += 1
            else:
                incomplete_items.append({
                    "file": str(doc_file.relative_to(self.docs_dir)),
                    "missing_sections": missing_sections,
                })

        completeness = (complete_count / total_count * 100) if total_count > 0 else 0
        status = "pass" if completeness >= 90 else "warning"

        return {
            "name": "Completeness",
            "status": status,
            "complete_count": complete_count,
            "total_count": total_count,
            "completeness_percent": round(completeness, 1),
            "target": 90,
            "incomplete_items": incomplete_items[:10],  # Show first 10
        }

    def check_links(self) -> Dict:
        """Check for broken links"""

        broken_links = []
        external_links = []
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text(encoding="utf-8", errors="ignore")
            links = re.findall(link_pattern, content)

            for text, url in links:
                # Skip anchors
                if url.startswith("#"):
                    continue

                # Track external links
                if url.startswith(("http://", "https://", "ftp://")):
                    external_links.append(url)
                    continue

                # Check internal links
                target_path = self.docs_dir / url
                if not target_path.exists():
                    broken_links.append({
                        "file": str(doc_file.relative_to(self.docs_dir)),
                        "target": url,
                        "text": text,
                    })
                    self.violations.append({
                        "file": str(doc_file),
                        "issue": f"Broken link: {url}",
                        "severity": "high",
                    })

        status = "pass" if not broken_links else "fail"

        return {
            "name": "Link Integrity",
            "status": status,
            "broken_links": len(broken_links),
            "external_links": len(external_links),
            "broken_details": broken_links[:10],  # Show first 10
        }

    def check_code_examples(self) -> Dict:
        """Check for code examples"""

        files_with_examples = 0
        example_pattern = r"```[\w\-]*\n"
        example_count = 0

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text(encoding="utf-8", errors="ignore")

            if re.search(example_pattern, content):
                files_with_examples += 1
                example_count += len(re.findall(example_pattern, content))

        total_files = len(list(self.docs_dir.glob("**/*.md")))
        coverage = (files_with_examples / total_files * 100) if total_files > 0 else 0
        status = "pass" if coverage >= 80 else "warning"

        return {
            "name": "Code Examples",
            "status": status,
            "files_with_examples": files_with_examples,
            "total_files": total_files,
            "coverage_percent": round(coverage, 1),
            "total_examples": example_count,
            "target": 80,
        }

    def check_formatting(self) -> Dict:
        """Check markdown formatting"""

        formatting_issues = []

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text(encoding="utf-8", errors="ignore")

            # Check for proper heading levels
            if content.count("## ") == 0 and content.count("# ") <= 1:
                formatting_issues.append({
                    "file": str(doc_file.relative_to(self.docs_dir)),
                    "issue": "No proper heading structure",
                })

            # Check for improperly formatted lists
            if re.search(r"\n-(?! )", content):
                formatting_issues.append({
                    "file": str(doc_file.relative_to(self.docs_dir)),
                    "issue": "Improperly formatted list items",
                })

            # Check for code block language tags
            code_blocks = re.findall(r"```(\w*)\n", content)
            if "```\n" in content and "" in code_blocks:  # Empty language tag
                formatting_issues.append({
                    "file": str(doc_file.relative_to(self.docs_dir)),
                    "issue": "Code blocks missing language tags",
                })

        status = "pass" if not formatting_issues else "warning"

        return {
            "name": "Formatting",
            "status": status,
            "issues": len(formatting_issues),
            "details": formatting_issues[:10],  # Show first 10
        }

    def check_metadata(self) -> Dict:
        """Check document metadata"""

        missing_metadata = []

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text(encoding="utf-8", errors="ignore")

            required_meta = [
                ("Version", r"Version:"),
                ("Last Updated", r"Last Updated:|Updated:"),
                ("Status", r"Status:"),
            ]

            missing = []
            for meta_name, pattern in required_meta:
                if not re.search(pattern, content, re.IGNORECASE):
                    missing.append(meta_name)

            if missing:
                missing_metadata.append({
                    "file": str(doc_file.relative_to(self.docs_dir)),
                    "missing_fields": missing,
                })

        status = "pass" if not missing_metadata else "warning"

        return {
            "name": "Metadata",
            "status": status,
            "issues": len(missing_metadata),
            "missing_items": missing_metadata[:10],  # Show first 10
        }

    def check_readability(self) -> Dict:
        """Check documentation readability"""

        readability_issues = []
        avg_line_length = 0
        total_lines = 0

        for doc_file in self.docs_dir.glob("**/*.md"):
            content = doc_file.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            total_lines += len(lines)

            for i, line in enumerate(lines):
                if len(line) > 120:  # Line too long
                    readability_issues.append({
                        "file": str(doc_file.relative_to(self.docs_dir)),
                        "line": i + 1,
                        "length": len(line),
                        "issue": "Line exceeds 120 characters",
                    })

        avg_line_length = sum(
            len(line) for doc_file in self.docs_dir.glob("**/*.md")
            for line in doc_file.read_text(encoding="utf-8", errors="ignore").split("\n")
        ) / max(total_lines, 1)

        status = "pass" if len(readability_issues) == 0 else "warning"

        return {
            "name": "Readability",
            "status": status,
            "long_lines": len(readability_issues),
            "average_line_length": round(avg_line_length, 1),
            "target_max_line_length": 120,
            "issues": readability_issues[:10],  # Show first 10
        }

    def calculate_health_score(self, checks: Dict) -> float:
        """Calculate overall health score"""

        scores = []
        weights = {
            "coverage": 0.20,
            "recency": 0.20,
            "completeness": 0.20,
            "links": 0.15,
            "code_examples": 0.10,
            "formatting": 0.05,
            "metadata": 0.05,
            "readability": 0.05,
        }

        for check_name, check in checks.items():
            if check["status"] == "pass":
                score = 100
            elif check["status"] == "warning":
                score = 75
            else:  # fail
                score = 50

            weight = weights.get(check_name, 0.05)
            scores.append(score * weight)

        return round(sum(scores), 1)

    def generate_summary(self, checks: Dict) -> Dict:
        """Generate summary report"""

        passed = sum(1 for check in checks.values() if check["status"] == "pass")
        warnings = sum(1 for check in checks.values() if check["status"] == "warning")
        failed = sum(1 for check in checks.values() if check["status"] == "fail")

        return {
            "total_checks": len(checks),
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "total_violations": len(self.violations),
        }


def print_results(results: Dict, output_format: str = "json") -> None:
    """Print results in specified format"""

    if output_format == "json":
        print(json.dumps(results, indent=2))
    elif output_format == "table":
        print("\nDocumentation Health Check Results")
        print("=" * 60)
        for check_name, check in results["checks"].items():
            status_symbol = "✓" if check["status"] == "pass" else "⚠" if check["status"] == "warning" else "✗"
            print(f"{status_symbol} {check['name']:<20} {check['status']:<10}")
        print("=" * 60)
        print(f"Overall Health Score: {results['health_score']}/100")
        print(f"Total Violations: {len(results['violations'])}")
    elif output_format == "summary":
        summary = results["summary"]
        print(f"\nDocumentation Health Summary")
        print(f"Score: {results['health_score']}/100")
        print(f"Checks Passed: {summary['passed']}/{summary['total_checks']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Failures: {summary['failed']}")
        print(f"Total Violations: {summary['total_violations']}")


def save_results(results: Dict, output_dir: str = ".reports/doc-health") -> None:
    """Save results to file"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_path / f"validation_results_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Documentation Health Validation"
    )
    parser.add_argument(
        "--output",
        choices=["json", "table", "summary"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if score < 90",
    )
    parser.add_argument(
        "--docs-dir",
        default="./docs",
        help="Documentation directory path",
    )

    args = parser.parse_args()

    validator = DocumentationHealthValidator(args.docs_dir)
    results = validator.validate_all()

    print_results(results, args.output)

    if args.save:
        save_results(results)

    if args.strict and results["health_score"] < 90:
        print(
            f"\nError: Health score {results['health_score']} is below target of 90",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
