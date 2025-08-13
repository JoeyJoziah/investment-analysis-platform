#!/usr/bin/env python3
"""
Test Coverage Analysis and Performance Benchmarking

This script provides comprehensive analysis of test coverage and
performance benchmarking with trend analysis and reporting.
"""

import argparse
import json
import os
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import subprocess
import requests
import time

@dataclass
class CoverageMetrics:
    """Container for coverage metrics"""
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    statement_coverage: float
    total_lines: int
    covered_lines: int
    missing_lines: int
    excluded_lines: int
    timestamp: datetime
    commit_hash: str

@dataclass
class PerformanceBenchmark:
    """Container for performance benchmark"""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: Optional[float]
    timestamp: datetime
    commit_hash: str
    environment: str

@dataclass
class ModuleCoverage:
    """Coverage metrics for a specific module"""
    module_name: str
    line_coverage: float
    branch_coverage: float
    complexity_score: float
    lines_total: int
    lines_covered: int
    lines_missing: int
    critical_paths_covered: bool

class CoverageAnalyzer:
    """Analyzes test coverage and generates reports"""
    
    def __init__(self, db_path: str = "coverage_metrics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Coverage metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                commit_hash TEXT,
                line_coverage REAL,
                branch_coverage REAL,
                function_coverage REAL,
                statement_coverage REAL,
                total_lines INTEGER,
                covered_lines INTEGER,
                missing_lines INTEGER,
                excluded_lines INTEGER
            )
        ''')
        
        # Performance benchmarks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                commit_hash TEXT,
                test_name TEXT,
                execution_time REAL,
                memory_usage REAL,
                cpu_usage REAL,
                throughput REAL,
                environment TEXT
            )
        ''')
        
        # Module coverage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS module_coverage (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                commit_hash TEXT,
                module_name TEXT,
                line_coverage REAL,
                branch_coverage REAL,
                complexity_score REAL,
                lines_total INTEGER,
                lines_covered INTEGER,
                lines_missing INTEGER,
                critical_paths_covered INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def parse_coverage_xml(self, xml_path: Path) -> Tuple[CoverageMetrics, List[ModuleCoverage]]:
        """Parse coverage XML file and extract metrics"""
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Overall coverage metrics
        line_rate = float(root.get('line-rate', 0))
        branch_rate = float(root.get('branch-rate', 0))
        
        # Count totals
        total_lines = 0
        covered_lines = 0
        
        # Module-level metrics
        modules = []
        
        for package in root.findall('.//package'):
            package_name = package.get('name', 'unknown')
            
            for class_elem in package.findall('classes/class'):
                class_name = class_elem.get('name', 'unknown')
                filename = class_elem.get('filename', '')
                
                # Calculate class metrics
                class_lines = 0
                class_covered = 0
                
                for line in class_elem.findall('lines/line'):
                    hits = int(line.get('hits', 0))
                    class_lines += 1
                    if hits > 0:
                        class_covered += 1
                
                total_lines += class_lines
                covered_lines += class_covered
                
                # Create module coverage
                if class_lines > 0:
                    module_coverage = ModuleCoverage(
                        module_name=f"{package_name}.{class_name}",
                        line_coverage=(class_covered / class_lines) * 100,
                        branch_coverage=0.0,  # Would need more detailed parsing
                        complexity_score=self.calculate_complexity_score(filename),
                        lines_total=class_lines,
                        lines_covered=class_covered,
                        lines_missing=class_lines - class_covered,
                        critical_paths_covered=class_covered > class_lines * 0.8
                    )
                    modules.append(module_coverage)
        
        # Create overall metrics
        commit_hash = self.get_current_commit_hash()
        
        overall_metrics = CoverageMetrics(
            line_coverage=line_rate * 100,
            branch_coverage=branch_rate * 100,
            function_coverage=0.0,  # Not always available in XML
            statement_coverage=line_rate * 100,  # Approximation
            total_lines=total_lines,
            covered_lines=covered_lines,
            missing_lines=total_lines - covered_lines,
            excluded_lines=0,
            timestamp=datetime.now(),
            commit_hash=commit_hash
        )
        
        return overall_metrics, modules
    
    def calculate_complexity_score(self, filename: str) -> float:
        """Calculate complexity score for a file"""
        if not os.path.exists(filename):
            return 0.0
        
        try:
            # Use radon to calculate complexity
            result = subprocess.run(
                ['radon', 'cc', filename, '-j'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] in ['function', 'method']:
                            total_complexity += item['complexity']
                            function_count += 1
                
                return total_complexity / max(function_count, 1)
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError):
            pass
        
        return 0.0
    
    def get_current_commit_hash(self) -> str:
        """Get current Git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # Short hash
            
        except subprocess.SubprocessError:
            pass
        
        return 'unknown'
    
    def store_coverage_metrics(self, metrics: CoverageMetrics, modules: List[ModuleCoverage]):
        """Store coverage metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store overall metrics
        cursor.execute('''
            INSERT INTO coverage_metrics 
            (timestamp, commit_hash, line_coverage, branch_coverage, function_coverage,
             statement_coverage, total_lines, covered_lines, missing_lines, excluded_lines)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.commit_hash,
            metrics.line_coverage,
            metrics.branch_coverage,
            metrics.function_coverage,
            metrics.statement_coverage,
            metrics.total_lines,
            metrics.covered_lines,
            metrics.missing_lines,
            metrics.excluded_lines
        ))
        
        # Store module metrics
        for module in modules:
            cursor.execute('''
                INSERT INTO module_coverage
                (timestamp, commit_hash, module_name, line_coverage, branch_coverage,
                 complexity_score, lines_total, lines_covered, lines_missing, critical_paths_covered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.commit_hash,
                module.module_name,
                module.line_coverage,
                module.branch_coverage,
                module.complexity_score,
                module.lines_total,
                module.lines_covered,
                module.lines_missing,
                1 if module.critical_paths_covered else 0
            ))
        
        conn.commit()
        conn.close()
    
    def get_coverage_trend(self, days: int = 30) -> List[CoverageMetrics]:
        """Get coverage trend for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM coverage_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp
        ''', (cutoff_date,))
        
        results = []
        for row in cursor.fetchall():
            metrics = CoverageMetrics(
                line_coverage=row[3],
                branch_coverage=row[4],
                function_coverage=row[5],
                statement_coverage=row[6],
                total_lines=row[7],
                covered_lines=row[8],
                missing_lines=row[9],
                excluded_lines=row[10],
                timestamp=datetime.fromisoformat(row[1]),
                commit_hash=row[2]
            )
            results.append(metrics)
        
        conn.close()
        return results
    
    def generate_coverage_report(self, output_dir: Path):
        """Generate comprehensive coverage report"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get recent data
        recent_metrics = self.get_coverage_trend(30)
        
        if not recent_metrics:
            print("No coverage data available")
            return
        
        latest_metrics = recent_metrics[-1]
        
        # Generate trend plots
        self.plot_coverage_trends(recent_metrics, output_dir)
        self.plot_module_coverage(latest_metrics.commit_hash, output_dir)
        
        # Generate HTML report
        self.generate_coverage_html_report(recent_metrics, output_dir)
        
        print(f"Coverage report generated in {output_dir}")
    
    def plot_coverage_trends(self, metrics: List[CoverageMetrics], output_dir: Path):
        """Plot coverage trends over time"""
        
        if len(metrics) < 2:
            return
        
        # Prepare data
        timestamps = [m.timestamp for m in metrics]
        line_coverage = [m.line_coverage for m in metrics]
        branch_coverage = [m.branch_coverage for m in metrics]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, line_coverage, marker='o', label='Line Coverage')
        plt.plot(timestamps, branch_coverage, marker='s', label='Branch Coverage')
        plt.title('Coverage Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Coverage (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Coverage distribution
        plt.subplot(2, 2, 2)
        plt.hist([line_coverage, branch_coverage], bins=20, alpha=0.7, 
                label=['Line Coverage', 'Branch Coverage'])
        plt.title('Coverage Distribution')
        plt.xlabel('Coverage (%)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Lines covered over time
        plt.subplot(2, 2, 3)
        total_lines = [m.total_lines for m in metrics]
        covered_lines = [m.covered_lines for m in metrics]
        
        plt.plot(timestamps, total_lines, label='Total Lines', marker='o')
        plt.plot(timestamps, covered_lines, label='Covered Lines', marker='s')
        plt.title('Lines of Code Over Time')
        plt.xlabel('Date')
        plt.ylabel('Lines')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Coverage quality score
        plt.subplot(2, 2, 4)
        quality_scores = [
            (m.line_coverage + m.branch_coverage) / 2 
            for m in metrics
        ]
        plt.plot(timestamps, quality_scores, marker='o', color='green')
        plt.title('Coverage Quality Score')
        plt.xlabel('Date')
        plt.ylabel('Quality Score (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'coverage_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_module_coverage(self, commit_hash: str, output_dir: Path):
        """Plot module-level coverage analysis"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT module_name, line_coverage, complexity_score, lines_total
            FROM module_coverage 
            WHERE commit_hash = ?
            ORDER BY line_coverage ASC
        ''', (commit_hash,))
        
        modules = cursor.fetchall()
        conn.close()
        
        if not modules:
            return
        
        # Prepare data
        module_names = [m[0].split('.')[-1][:20] for m in modules]  # Truncate names
        coverages = [m[1] for m in modules]
        complexities = [m[2] for m in modules]
        line_counts = [m[3] for m in modules]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Module coverage bar chart
        colors = ['red' if c < 70 else 'yellow' if c < 85 else 'green' for c in coverages]
        ax1.barh(module_names, coverages, color=colors, alpha=0.7)
        ax1.set_xlabel('Coverage (%)')
        ax1.set_title('Module Coverage')
        ax1.axvline(x=80, color='orange', linestyle='--', label='Target (80%)')
        ax1.legend()
        
        # Coverage vs Complexity scatter
        ax2.scatter(complexities, coverages, s=[l/10 for l in line_counts], alpha=0.6)
        ax2.set_xlabel('Complexity Score')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_title('Coverage vs Complexity')
        ax2.grid(True, alpha=0.3)
        
        # Coverage distribution
        ax3.hist(coverages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=np.mean(coverages), color='red', linestyle='--', label=f'Mean: {np.mean(coverages):.1f}%')
        ax3.set_xlabel('Coverage (%)')
        ax3.set_ylabel('Number of Modules')
        ax3.set_title('Coverage Distribution')
        ax3.legend()
        
        # Top uncovered modules
        bottom_modules = sorted(modules, key=lambda x: x[1])[:10]
        bottom_names = [m[0].split('.')[-1][:15] for m in bottom_modules]
        bottom_coverages = [m[1] for m in bottom_modules]
        
        ax4.barh(bottom_names, bottom_coverages, color='red', alpha=0.7)
        ax4.set_xlabel('Coverage (%)')
        ax4.set_title('Modules Needing Attention (Bottom 10)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'module_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_coverage_html_report(self, metrics: List[CoverageMetrics], output_dir: Path):
        """Generate HTML coverage report"""
        
        latest = metrics[-1] if metrics else None
        if not latest:
            return
        
        # Calculate trends
        trend_direction = ""
        trend_color = "gray"
        
        if len(metrics) > 1:
            previous = metrics[-2]
            coverage_change = latest.line_coverage - previous.line_coverage
            
            if coverage_change > 0:
                trend_direction = f"↑ +{coverage_change:.1f}%"
                trend_color = "green"
            elif coverage_change < 0:
                trend_direction = f"↓ {coverage_change:.1f}%"
                trend_color = "red"
            else:
                trend_direction = "→ No change"
                trend_color = "gray"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Coverage Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .trend {{
            color: {trend_color};
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .coverage-excellent {{ color: #28a745; }}
        .coverage-good {{ color: #28a745; }}
        .coverage-warning {{ color: #ffc107; }}
        .coverage-poor {{ color: #dc3545; }}
        
        .charts-section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .chart-image {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        
        .coverage-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .coverage-fill {{
            height: 100%;
            background: linear-gradient(90deg, #dc3545, #ffc107, #28a745);
            transition: width 0.3s ease;
        }}
        
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .recommendations h3 {{
            color: #856404;
            margin-top: 0;
        }}
        
        .recommendation-item {{
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid #ffeaa7;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Test Coverage Report</h1>
        <p>Investment Analysis Application</p>
        <p>Generated: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Commit: {latest.commit_hash}</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Line Coverage</div>
            <div class="metric-value {'coverage-excellent' if latest.line_coverage >= 90 else 'coverage-good' if latest.line_coverage >= 80 else 'coverage-warning' if latest.line_coverage >= 70 else 'coverage-poor'}">
                {latest.line_coverage:.1f}%
            </div>
            <div class="coverage-bar">
                <div class="coverage-fill" style="width: {latest.line_coverage}%"></div>
            </div>
            <div class="trend">{trend_direction}</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Branch Coverage</div>
            <div class="metric-value {'coverage-excellent' if latest.branch_coverage >= 85 else 'coverage-good' if latest.branch_coverage >= 75 else 'coverage-warning' if latest.branch_coverage >= 65 else 'coverage-poor'}">
                {latest.branch_coverage:.1f}%
            </div>
            <div class="coverage-bar">
                <div class="coverage-fill" style="width: {latest.branch_coverage}%"></div>
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Total Lines</div>
            <div class="metric-value" style="color: #007bff;">
                {latest.total_lines:,}
            </div>
            <div class="metric-label">
                {latest.covered_lines:,} covered, {latest.missing_lines:,} missing
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Coverage Quality</div>
            <div class="metric-value {'coverage-excellent' if (latest.line_coverage + latest.branch_coverage)/2 >= 85 else 'coverage-good' if (latest.line_coverage + latest.branch_coverage)/2 >= 75 else 'coverage-warning' if (latest.line_coverage + latest.branch_coverage)/2 >= 65 else 'coverage-poor'}">
                {(latest.line_coverage + latest.branch_coverage)/2:.1f}%
            </div>
            <div class="metric-label">Combined Score</div>
        </div>
    </div>

    <div class="charts-section">
        <h3>📈 Coverage Trends</h3>
        <img src="coverage_trends.png" alt="Coverage Trends" class="chart-image">
    </div>

    <div class="charts-section">
        <h3>🎯 Module Analysis</h3>
        <img src="module_coverage.png" alt="Module Coverage" class="chart-image">
    </div>

    <div class="recommendations">
        <h3>💡 Recommendations</h3>
        
        {'<div class="recommendation-item">✅ Excellent line coverage! Maintain current testing practices.</div>' if latest.line_coverage >= 90 else ''}
        {'<div class="recommendation-item">⚠️ Line coverage below 90%. Consider adding more unit tests.</div>' if latest.line_coverage < 90 else ''}
        {'<div class="recommendation-item">🎯 Branch coverage could be improved. Add tests for edge cases and error conditions.</div>' if latest.branch_coverage < 80 else ''}
        {'<div class="recommendation-item">🔍 Focus on modules with low coverage (see module analysis above).</div>' if latest.line_coverage < 85 else ''}
        {'<div class="recommendation-item">📊 Set up coverage gates in CI/CD to prevent coverage regression.</div>'}
        {'<div class="recommendation-item">🧪 Consider mutation testing to validate test quality.</div>'}
        {'<div class="recommendation-item">📝 Add integration tests for critical business logic paths.</div>'}
    </div>

    <div class="footer">
        <p>Coverage analysis generated by Investment Analysis App CI/CD Pipeline</p>
        <p>Target: >90% line coverage, >80% branch coverage</p>
    </div>
</body>
</html>
        """
        
        with open(output_dir / 'coverage_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)


class PerformanceBenchmarker:
    """Performance benchmarking and trend analysis"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database for performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                commit_hash TEXT,
                test_suite TEXT,
                test_name TEXT,
                execution_time REAL,
                memory_usage REAL,
                cpu_usage REAL,
                throughput REAL,
                environment TEXT,
                baseline_execution_time REAL,
                baseline_memory_usage REAL,
                performance_regression REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def run_performance_benchmarks(self) -> List[PerformanceBenchmark]:
        """Run performance benchmarks"""
        
        benchmarks = []
        commit_hash = self.get_current_commit_hash()
        environment = os.environ.get('CI_ENVIRONMENT', 'local')
        
        # Benchmark 1: Stock analysis performance
        benchmark = self.benchmark_stock_analysis()
        benchmark.commit_hash = commit_hash
        benchmark.environment = environment
        benchmarks.append(benchmark)
        
        # Benchmark 2: Database query performance
        benchmark = self.benchmark_database_queries()
        benchmark.commit_hash = commit_hash
        benchmark.environment = environment
        benchmarks.append(benchmark)
        
        # Benchmark 3: API response time
        benchmark = self.benchmark_api_responses()
        benchmark.commit_hash = commit_hash
        benchmark.environment = environment
        benchmarks.append(benchmark)
        
        # Benchmark 4: Cache performance
        benchmark = self.benchmark_cache_operations()
        benchmark.commit_hash = commit_hash
        benchmark.environment = environment
        benchmarks.append(benchmark)
        
        return benchmarks
    
    def benchmark_stock_analysis(self) -> PerformanceBenchmark:
        """Benchmark stock analysis performance"""
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Simulate stock analysis workload
        try:
            # This would run actual stock analysis
            # For now, simulate with computational work
            import numpy as np
            
            for _ in range(100):  # Simulate 100 stocks
                # Simulate technical analysis calculations
                data = np.random.random(252)  # Year of daily data
                sma_20 = np.convolve(data, np.ones(20)/20, mode='valid')
                sma_50 = np.convolve(data, np.ones(50)/50, mode='valid')
                rsi = self.calculate_rsi(data)
            
            throughput = 100 / max(time.time() - start_time, 0.001)
            
        except Exception as e:
            print(f"Benchmark error: {e}")
            throughput = 0
        
        execution_time = (time.time() - start_time) * 1000  # ms
        memory_usage = self.get_memory_usage() - start_memory
        
        return PerformanceBenchmark(
            test_name="stock_analysis_performance",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # Would need actual CPU monitoring
            throughput=throughput,
            timestamp=datetime.now(),
            commit_hash="",
            environment=""
        )
    
    def benchmark_database_queries(self) -> PerformanceBenchmark:
        """Benchmark database query performance"""
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Simulate database queries
        try:
            # This would run actual database queries
            # For now, simulate with in-memory operations
            import sqlite3
            
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute('''
                CREATE TABLE test_stocks (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    price REAL,
                    volume INTEGER
                )
            ''')
            
            # Insert test data
            for i in range(10000):
                cursor.execute(
                    'INSERT INTO test_stocks (ticker, price, volume) VALUES (?, ?, ?)',
                    (f'STOCK{i}', 100 + i * 0.01, 1000000 + i * 1000)
                )
            
            # Run queries
            for _ in range(100):
                cursor.execute('SELECT * FROM test_stocks WHERE price > 150 ORDER BY volume DESC LIMIT 10')
                cursor.fetchall()
            
            conn.close()
            throughput = 100 / max(time.time() - start_time, 0.001)
            
        except Exception as e:
            print(f"Database benchmark error: {e}")
            throughput = 0
        
        execution_time = (time.time() - start_time) * 1000
        memory_usage = self.get_memory_usage() - start_memory
        
        return PerformanceBenchmark(
            test_name="database_query_performance",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,
            throughput=throughput,
            timestamp=datetime.now(),
            commit_hash="",
            environment=""
        )
    
    def benchmark_api_responses(self) -> PerformanceBenchmark:
        """Benchmark API response times"""
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Simulate API calls
        try:
            response_times = []
            
            # This would make actual API calls
            # For now, simulate network latency
            for _ in range(10):
                api_start = time.time()
                time.sleep(0.001)  # Simulate 1ms response time
                response_times.append((time.time() - api_start) * 1000)
            
            avg_response_time = sum(response_times) / len(response_times)
            throughput = len(response_times) / max(time.time() - start_time, 0.001)
            
        except Exception as e:
            print(f"API benchmark error: {e}")
            avg_response_time = 0
            throughput = 0
        
        execution_time = avg_response_time
        memory_usage = self.get_memory_usage() - start_memory
        
        return PerformanceBenchmark(
            test_name="api_response_performance",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,
            throughput=throughput,
            timestamp=datetime.now(),
            commit_hash="",
            environment=""
        )
    
    def benchmark_cache_operations(self) -> PerformanceBenchmark:
        """Benchmark cache operations"""
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            # Simulate cache operations
            cache = {}
            
            # Cache writes
            for i in range(1000):
                cache[f"key_{i}"] = {"data": f"value_{i}", "timestamp": time.time()}
            
            # Cache reads
            hit_count = 0
            for i in range(1000):
                if f"key_{i}" in cache:
                    hit_count += 1
                    _ = cache[f"key_{i}"]
            
            throughput = 2000 / max(time.time() - start_time, 0.001)  # 1000 writes + 1000 reads
            
        except Exception as e:
            print(f"Cache benchmark error: {e}")
            throughput = 0
        
        execution_time = (time.time() - start_time) * 1000
        memory_usage = self.get_memory_usage() - start_memory
        
        return PerformanceBenchmark(
            test_name="cache_operations_performance",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,
            throughput=throughput,
            timestamp=datetime.now(),
            commit_hash="",
            environment=""
        )
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI for benchmark"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rs = avg_gain / max(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_current_commit_hash(self) -> str:
        """Get current Git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except subprocess.SubprocessError:
            pass
        return 'unknown'
    
    def store_benchmarks(self, benchmarks: List[PerformanceBenchmark]):
        """Store benchmark results in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for benchmark in benchmarks:
            cursor.execute('''
                INSERT INTO performance_benchmarks
                (timestamp, commit_hash, test_suite, test_name, execution_time, 
                 memory_usage, cpu_usage, throughput, environment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                benchmark.timestamp.isoformat(),
                benchmark.commit_hash,
                'comprehensive_benchmarks',
                benchmark.test_name,
                benchmark.execution_time,
                benchmark.memory_usage,
                benchmark.cpu_usage,
                benchmark.throughput,
                benchmark.environment
            ))
        
        conn.commit()
        conn.close()
    
    def generate_performance_report(self, output_dir: Path):
        """Generate performance benchmark report"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get recent benchmarks
        conn = sqlite3.connect(self.db_path)
        
        # Get latest benchmarks
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM performance_benchmarks 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''')
        
        recent_benchmarks = cursor.fetchall()
        conn.close()
        
        if not recent_benchmarks:
            print("No performance benchmark data available")
            return
        
        # Group by test name
        benchmark_groups = {}
        for row in recent_benchmarks:
            test_name = row[4]  # test_name column
            if test_name not in benchmark_groups:
                benchmark_groups[test_name] = []
            
            benchmark = {
                'timestamp': datetime.fromisoformat(row[1]),
                'execution_time': row[5],
                'memory_usage': row[6],
                'throughput': row[8]
            }
            benchmark_groups[test_name].append(benchmark)
        
        # Generate plots
        self.plot_performance_trends(benchmark_groups, output_dir)
        
        # Generate HTML report
        self.generate_performance_html_report(benchmark_groups, output_dir)
        
        print(f"Performance report generated in {output_dir}")
    
    def plot_performance_trends(self, benchmark_groups: Dict, output_dir: Path):
        """Plot performance trends"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot execution time trends
        ax = axes[0, 0]
        for test_name, benchmarks in benchmark_groups.items():
            timestamps = [b['timestamp'] for b in benchmarks]
            exec_times = [b['execution_time'] for b in benchmarks]
            ax.plot(timestamps, exec_times, marker='o', label=test_name)
        
        ax.set_title('Execution Time Trends')
        ax.set_xlabel('Date')
        ax.set_ylabel('Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot memory usage trends
        ax = axes[0, 1]
        for test_name, benchmarks in benchmark_groups.items():
            timestamps = [b['timestamp'] for b in benchmarks]
            memory_usage = [b['memory_usage'] for b in benchmarks]
            ax.plot(timestamps, memory_usage, marker='s', label=test_name)
        
        ax.set_title('Memory Usage Trends')
        ax.set_xlabel('Date')
        ax.set_ylabel('Memory (MB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot throughput trends
        ax = axes[1, 0]
        for test_name, benchmarks in benchmark_groups.items():
            timestamps = [b['timestamp'] for b in benchmarks]
            throughputs = [b['throughput'] for b in benchmarks if b['throughput'] > 0]
            if throughputs:
                ax.plot(timestamps[-len(throughputs):], throughputs, marker='^', label=test_name)
        
        ax.set_title('Throughput Trends')
        ax.set_xlabel('Date')
        ax.set_ylabel('Operations/Second')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance distribution
        ax = axes[1, 1]
        all_exec_times = []
        labels = []
        
        for test_name, benchmarks in benchmark_groups.items():
            exec_times = [b['execution_time'] for b in benchmarks]
            all_exec_times.extend(exec_times)
            labels.extend([test_name] * len(exec_times))
        
        # Create box plot
        test_names = list(benchmark_groups.keys())
        exec_time_data = []
        for test_name in test_names:
            exec_times = [b['execution_time'] for b in benchmark_groups[test_name]]
            exec_time_data.append(exec_times)
        
        ax.boxplot(exec_time_data, labels=[name[:10] for name in test_names])
        ax.set_title('Performance Distribution')
        ax.set_ylabel('Execution Time (ms)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_html_report(self, benchmark_groups: Dict, output_dir: Path):
        """Generate HTML performance report"""
        
        # Calculate summary statistics
        total_tests = len(benchmark_groups)
        avg_execution_time = 0
        avg_memory_usage = 0
        
        if benchmark_groups:
            all_exec_times = []
            all_memory_usage = []
            
            for benchmarks in benchmark_groups.values():
                if benchmarks:
                    latest = benchmarks[0]  # Most recent
                    all_exec_times.append(latest['execution_time'])
                    all_memory_usage.append(latest['memory_usage'])
            
            avg_execution_time = sum(all_exec_times) / len(all_exec_times) if all_exec_times else 0
            avg_memory_usage = sum(all_memory_usage) / len(all_memory_usage) if all_memory_usage else 0
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Benchmark Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
            margin: 10px 0;
        }}
        
        .charts-section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .benchmark-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .benchmark-table th,
        .benchmark-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .benchmark-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        
        .performance-good {{ color: #28a745; }}
        .performance-warning {{ color: #ffc107; }}
        .performance-poor {{ color: #dc3545; }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ Performance Benchmark Report</h1>
        <p>Investment Analysis Application</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary-cards">
        <div class="summary-card">
            <h3>Total Benchmarks</h3>
            <div class="metric-value">{total_tests}</div>
        </div>
        
        <div class="summary-card">
            <h3>Avg Execution Time</h3>
            <div class="metric-value">{avg_execution_time:.1f}ms</div>
        </div>
        
        <div class="summary-card">
            <h3>Avg Memory Usage</h3>
            <div class="metric-value">{avg_memory_usage:.1f}MB</div>
        </div>
    </div>

    <div class="charts-section">
        <h3>📈 Performance Trends</h3>
        <img src="performance_trends.png" alt="Performance Trends" style="max-width: 100%; height: auto;">
    </div>

    <div class="charts-section">
        <h3>📊 Benchmark Results</h3>
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Execution Time (ms)</th>
                    <th>Memory Usage (MB)</th>
                    <th>Throughput (ops/s)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for test_name, benchmarks in benchmark_groups.items():
            if benchmarks:
                latest = benchmarks[0]
                exec_time = latest['execution_time']
                memory = latest['memory_usage']
                throughput = latest['throughput']
                
                # Determine status based on thresholds
                if exec_time < 100 and memory < 100:
                    status = "Excellent"
                    status_class = "performance-good"
                elif exec_time < 500 and memory < 500:
                    status = "Good"
                    status_class = "performance-good"
                elif exec_time < 1000 and memory < 1000:
                    status = "Warning"
                    status_class = "performance-warning"
                else:
                    status = "Poor"
                    status_class = "performance-poor"
                
                html_content += f"""
                <tr>
                    <td>{test_name.replace('_', ' ').title()}</td>
                    <td>{exec_time:.1f}</td>
                    <td>{memory:.1f}</td>
                    <td>{throughput:.1f}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
                """
        
        html_content += """
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>Performance benchmarks run automatically in CI/CD pipeline</p>
        <p>Targets: &lt;100ms execution time, &lt;100MB memory usage</p>
    </div>
</body>
</html>
        """
        
        with open(output_dir / 'performance_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description='Coverage and Performance Analysis')
    parser.add_argument('--mode', choices=['coverage', 'performance', 'both'], 
                       default='both', help='Analysis mode')
    parser.add_argument('--coverage-xml', help='Path to coverage XML file')
    parser.add_argument('--output-dir', default='reports', help='Output directory')
    parser.add_argument('--run-benchmarks', action='store_true', 
                       help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.mode in ['coverage', 'both']:
        print("📊 Running coverage analysis...")
        analyzer = CoverageAnalyzer()
        
        if args.coverage_xml:
            coverage_xml = Path(args.coverage_xml)
            if coverage_xml.exists():
                metrics, modules = analyzer.parse_coverage_xml(coverage_xml)
                analyzer.store_coverage_metrics(metrics, modules)
                print(f"✅ Coverage metrics stored: {metrics.line_coverage:.1f}% line coverage")
            else:
                print(f"❌ Coverage XML file not found: {coverage_xml}")
        
        analyzer.generate_coverage_report(output_dir / 'coverage')
        print("✅ Coverage report generated")
    
    if args.mode in ['performance', 'both']:
        print("⚡ Running performance analysis...")
        benchmarker = PerformanceBenchmarker()
        
        if args.run_benchmarks:
            print("Running performance benchmarks...")
            benchmarks = benchmarker.run_performance_benchmarks()
            benchmarker.store_benchmarks(benchmarks)
            print(f"✅ {len(benchmarks)} benchmarks completed")
        
        benchmarker.generate_performance_report(output_dir / 'performance')
        print("✅ Performance report generated")
    
    print(f"📁 Reports available in: {output_dir}")


if __name__ == '__main__':
    main()