#!/usr/bin/env python3
"""
Migration script to convert Apache Airflow DAGs to Prefect 2.x flows.
Designed for the Investment Analysis Application's data pipeline migration.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import textwrap

class AirflowToPrefectMigrator:
    """Converts Airflow DAGs to Prefect flows for Python 3.13 compatibility."""
    
    def __init__(self, airflow_dag_path: str, output_path: str):
        self.airflow_dag_path = Path(airflow_dag_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Mapping of Airflow operators to Prefect equivalents
        self.operator_mapping = {
            'PythonOperator': self._convert_python_operator,
            'PostgresOperator': self._convert_postgres_operator,
            'TriggerDagRunOperator': self._convert_trigger_dag_operator,
            'BashOperator': self._convert_bash_operator,
        }
        
    def migrate_dag(self, dag_file: Path) -> str:
        """Convert a single Airflow DAG file to Prefect flow."""
        print(f"Migrating {dag_file.name}...")
        
        with open(dag_file, 'r') as f:
            content = f.read()
        
        # Parse the Python AST
        tree = ast.parse(content)
        
        # Extract DAG configuration
        dag_config = self._extract_dag_config(tree)
        
        # Extract tasks
        tasks = self._extract_tasks(tree)
        
        # Generate Prefect flow
        prefect_code = self._generate_prefect_flow(dag_file.stem, dag_config, tasks, content)
        
        # Save the converted flow
        output_file = self.output_path / f"{dag_file.stem}_flow.py"
        with open(output_file, 'w') as f:
            f.write(prefect_code)
        
        print(f"✅ Migrated to {output_file}")
        return str(output_file)
    
    def _extract_dag_config(self, tree: ast.Module) -> Dict:
        """Extract DAG configuration from AST."""
        config = {
            'schedule': None,
            'description': '',
            'tags': [],
            'retries': 2,
            'retry_delay': 300,  # 5 minutes in seconds
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'dag':
                        if isinstance(node.value, ast.Call):
                            for keyword in node.value.keywords:
                                if keyword.arg == 'schedule_interval':
                                    config['schedule'] = self._extract_value(keyword.value)
                                elif keyword.arg == 'description':
                                    config['description'] = self._extract_value(keyword.value)
                                elif keyword.arg == 'tags':
                                    config['tags'] = self._extract_value(keyword.value)
                    
                    elif isinstance(target, ast.Name) and target.id == 'default_args':
                        if isinstance(node.value, ast.Dict):
                            for key, value in zip(node.value.keys, node.value.values):
                                key_str = self._extract_value(key)
                                if key_str == 'retries':
                                    config['retries'] = self._extract_value(value)
                                elif key_str == 'retry_delay':
                                    # Convert timedelta to seconds
                                    config['retry_delay'] = 300  # Default 5 minutes
        
        return config
    
    def _extract_tasks(self, tree: ast.Module) -> List[Dict]:
        """Extract task definitions from AST."""
        tasks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    # Check if it's an operator instantiation
                    operator_name = self._get_operator_name(node.value)
                    if operator_name in self.operator_mapping:
                        task_info = {
                            'name': node.targets[0].id if node.targets else 'unnamed_task',
                            'operator': operator_name,
                            'args': self._extract_operator_args(node.value),
                        }
                        tasks.append(task_info)
        
        return tasks
    
    def _get_operator_name(self, call_node: ast.Call) -> Optional[str]:
        """Get the operator name from a Call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None
    
    def _extract_operator_args(self, call_node: ast.Call) -> Dict:
        """Extract arguments from operator instantiation."""
        args = {}
        for keyword in call_node.keywords:
            if keyword.arg:
                args[keyword.arg] = self._extract_value(keyword.value)
        return args
    
    def _extract_value(self, node):
        """Extract Python value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._extract_value(k): self._extract_value(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
    
    def _generate_prefect_flow(self, flow_name: str, config: Dict, tasks: List[Dict], original_content: str) -> str:
        """Generate Prefect flow code from extracted information."""
        
        # Extract function definitions from original content
        functions = self._extract_functions(original_content)
        
        code = f'''"""
Prefect flow converted from Airflow DAG: {flow_name}
{config.get('description', '')}

Migrated for Python 3.13 compatibility.
Original Airflow DAG replaced with Prefect 2.x flow.
"""

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import asyncio
import os

# Import your backend modules
from backend.data_ingestion.enhanced_api_client import EnhancedAPIClient
from backend.utils.enhanced_cost_monitor import EnhancedCostMonitor
from backend.utils.data_quality import DataQualityChecker
from backend.analytics.technical.technical_analysis_engine import TechnicalAnalysisEngine
from backend.analytics.fundamental.fundamental_analysis_engine import FundamentalAnalysisEngine
from backend.analytics.sentiment.sentiment_analysis_engine import SentimentAnalysisEngine
from backend.ml.recommendation_engine import RecommendationEngine

# Configure logging
logger = get_run_logger()

'''

        # Add converted functions
        for func in functions:
            code += f"\n{self._convert_function_to_task(func)}\n"
        
        # Add converted tasks
        for task_info in tasks:
            converter = self.operator_mapping.get(task_info['operator'])
            if converter:
                code += f"\n{converter(task_info)}\n"
        
        # Add main flow
        code += self._generate_main_flow(flow_name, config, tasks)
        
        # Add deployment configuration
        code += self._generate_deployment(flow_name, config)
        
        return code
    
    def _extract_functions(self, content: str) -> List[str]:
        """Extract function definitions from original content."""
        functions = []
        lines = content.split('\n')
        in_function = False
        current_function = []
        indent_level = 0
        
        for line in lines:
            if line.strip().startswith('def ') and not in_function:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                current_function = [line]
            elif in_function:
                if line.strip() and not line.startswith(' ' * indent_level) and not line.strip().startswith('#'):
                    # End of function
                    functions.append('\n'.join(current_function))
                    current_function = []
                    in_function = False
                    if line.strip().startswith('def '):
                        in_function = True
                        indent_level = len(line) - len(line.lstrip())
                        current_function = [line]
                else:
                    current_function.append(line)
        
        if current_function:
            functions.append('\n'.join(current_function))
        
        return functions
    
    def _convert_function_to_task(self, func: str) -> str:
        """Convert a function to a Prefect task."""
        # Add @task decorator
        lines = func.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                # Extract function name
                func_name = re.search(r'def\s+(\w+)', line).group(1)
                
                # Add task decorator with caching for idempotency
                lines[i] = f'''@task(
    name="{func_name}",
    retries={2},
    retry_delay_seconds=300,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
{line}'''
                break
        
        # Replace Airflow-specific code
        result = '\n'.join(lines)
        result = result.replace('**context', '')
        result = result.replace('ti.xcom_push', 'return')
        result = result.replace('ti.xcom_pull', '')
        result = result.replace('get_current_context()', 'get_run_logger()')
        
        return result
    
    def _convert_python_operator(self, task_info: Dict) -> str:
        """Convert PythonOperator to Prefect task."""
        args = task_info['args']
        task_name = task_info['name']
        python_callable = args.get('python_callable', 'unnamed_function')
        
        return f'''
# Task: {task_name}
# Original: PythonOperator
# Converted to Prefect task call in flow
'''
    
    def _convert_postgres_operator(self, task_info: Dict) -> str:
        """Convert PostgresOperator to Prefect task."""
        return f'''
@task(name="{task_info['name']}")
async def {task_info['name']}():
    """Execute PostgreSQL query."""
    from sqlalchemy.ext.asyncio import create_async_engine
    
    engine = create_async_engine(os.getenv("DATABASE_URL"))
    async with engine.begin() as conn:
        # Execute your SQL here
        result = await conn.execute("{task_info['args'].get('sql', '')}")
        return result.fetchall()
'''
    
    def _convert_trigger_dag_operator(self, task_info: Dict) -> str:
        """Convert TriggerDagRunOperator to Prefect subflow."""
        return f'''
@task(name="{task_info['name']}")
async def {task_info['name']}():
    """Trigger another flow (originally DAG)."""
    from prefect.deployments import run_deployment
    
    # Trigger the converted flow deployment
    flow_run = await run_deployment(
        name="{task_info['args'].get('trigger_dag_id', 'subflow')}/production",
        parameters={{}},
        timeout=0,  # Don't wait for completion
    )
    return flow_run.id
'''
    
    def _convert_bash_operator(self, task_info: Dict) -> str:
        """Convert BashOperator to Prefect shell task."""
        return f'''
@task(name="{task_info['name']}")
async def {task_info['name']}():
    """Execute bash command."""
    import subprocess
    
    result = subprocess.run(
        "{task_info['args'].get('bash_command', 'echo Hello')}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Command failed: {{result.stderr}}")
    
    return result.stdout
'''
    
    def _generate_main_flow(self, flow_name: str, config: Dict, tasks: List[Dict]) -> str:
        """Generate the main Prefect flow function."""
        return f'''

@flow(
    name="{flow_name}",
    description="{config['description']}",
    retries={config['retries']},
    retry_delay_seconds={config['retry_delay']},
    persist_result=True,
)
async def {flow_name}_flow():
    """
    Main Prefect flow for {flow_name}.
    Processes 6000+ stocks with intelligent tiering and API management.
    """
    logger = get_run_logger()
    logger.info(f"Starting {{flow_name}} flow")
    
    # Initialize monitoring
    cost_monitor = EnhancedCostMonitor()
    quality_checker = DataQualityChecker()
    
    try:
        # Check if market is open
        market_status = await get_market_calendar()
        if not market_status['is_open']:
            logger.info(f"Market is closed: {{market_status['reason']}}")
            return {{"status": "skipped", "reason": "market_closed"}}
        
        # Prioritize stocks into tiers
        stock_tiers = await prioritize_stocks()
        logger.info(f"Prioritized {{sum(len(tier) for tier in stock_tiers.values())}} stocks into tiers")
        
        # Process each tier with appropriate update frequency
        results = {{}}
        for tier, stocks in stock_tiers.items():
            logger.info(f"Processing Tier {{tier}}: {{len(stocks)}} stocks")
            
            # Check cost budget before processing
            if cost_monitor.is_emergency_mode():
                logger.warning("Emergency mode activated - using cached data only")
                tier_results = await process_cached_stocks(stocks)
            else:
                tier_results = await process_stock_tier(tier, stocks)
            
            results[tier] = tier_results
            
            # Update cost metrics
            cost_monitor.update_metrics()
        
        # Generate recommendations
        recommendations = await generate_recommendations(results)
        
        # Store results
        await store_analysis_results(recommendations)
        
        logger.info("Flow completed successfully")
        return {{
            "status": "success",
            "stocks_processed": sum(len(tier) for tier in stock_tiers.values()),
            "recommendations": len(recommendations),
            "cost_status": cost_monitor.get_daily_summary()
        }}
        
    except Exception as e:
        logger.error(f"Flow failed: {{str(e)}}")
        raise

'''
    
    def _generate_deployment(self, flow_name: str, config: Dict) -> str:
        """Generate Prefect deployment configuration."""
        schedule = config.get('schedule', '0 6 * * 1-5')
        
        return f'''

# Create deployment for production
if __name__ == "__main__":
    # Create a deployment with a cron schedule
    deployment = Deployment.build_from_flow(
        flow={flow_name}_flow,
        name="production",
        version="1.0.0",
        tags={config.get('tags', [])},
        schedule=CronSchedule(cron="{schedule}", timezone="America/New_York"),
        work_queue_name="investment-analysis",
        infrastructure="process",  # Can be changed to kubernetes, docker, etc.
        parameters={{}},
        description="{config['description']}",
    )
    
    # Apply the deployment
    deployment.apply()
    
    print(f"✅ Deployment created: {flow_name}/production")
    print(f"📅 Schedule: {schedule}")
    print(f"🏷️  Tags: {config.get('tags', [])}")
    print()
    print("To run the flow manually:")
    print(f"  prefect deployment run '{flow_name}/production'")
    print()
    print("To start a worker:")
    print("  prefect agent start -q investment-analysis")
'''

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Airflow DAGs to Prefect flows')
    parser.add_argument('--input', '-i', 
                       default='data_pipelines/airflow/dags',
                       help='Path to Airflow DAGs directory')
    parser.add_argument('--output', '-o',
                       default='data_pipelines/prefect/flows', 
                       help='Output path for Prefect flows')
    parser.add_argument('--dag', '-d',
                       help='Specific DAG file to migrate (optional)')
    
    args = parser.parse_args()
    
    migrator = AirflowToPrefectMigrator(args.input, args.output)
    
    if args.dag:
        # Migrate specific DAG
        dag_file = Path(args.input) / args.dag
        if dag_file.exists():
            migrator.migrate_dag(dag_file)
        else:
            print(f"❌ DAG file not found: {dag_file}")
    else:
        # Migrate all DAGs
        dag_files = list(Path(args.input).glob('*.py'))
        print(f"Found {len(dag_files)} DAG files to migrate")
        
        for dag_file in dag_files:
            try:
                migrator.migrate_dag(dag_file)
            except Exception as e:
                print(f"❌ Failed to migrate {dag_file.name}: {str(e)}")
        
        print(f"\n✅ Migration complete! Prefect flows saved to: {args.output}")
        print("\nNext steps:")
        print("1. Review the generated flows for any manual adjustments")
        print("2. Update imports to use Prefect-specific modules")
        print("3. Test flows in development environment")
        print("4. Deploy to Prefect Cloud or self-hosted Prefect server")

if __name__ == "__main__":
    main()