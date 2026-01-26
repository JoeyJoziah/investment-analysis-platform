#!/usr/bin/env python3
"""
Setup Airflow pools for the investment analysis platform.
Run this script after Airflow is initialized to create the necessary pools.

Usage:
    python scripts/setup_airflow_pools.py

Or via Airflow CLI:
    airflow pools set stock_api_pool 8 "Pool for stock API rate limiting"
"""

import os
import sys
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pool configurations
POOLS = [
    {
        'name': 'stock_api_pool',
        'slots': 8,
        'description': 'Pool for stock API rate limiting - controls concurrent API connections'
    },
    {
        'name': 'data_processing_pool',
        'slots': 4,
        'description': 'Pool for CPU-intensive data processing tasks'
    },
    {
        'name': 'ml_inference_pool',
        'slots': 2,
        'description': 'Pool for ML model inference - memory intensive'
    }
]


def setup_pool_via_cli(name: str, slots: int, description: str) -> bool:
    """Setup pool using Airflow CLI"""
    try:
        cmd = ['airflow', 'pools', 'set', name, str(slots), description]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            logger.info(f"Created/updated pool '{name}' with {slots} slots")
            return True
        else:
            logger.error(f"Failed to create pool '{name}': {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout creating pool '{name}'")
        return False
    except FileNotFoundError:
        logger.warning("Airflow CLI not found, trying Python API method")
        return False


def setup_pool_via_api(name: str, slots: int, description: str) -> bool:
    """Setup pool using Airflow Python API"""
    try:
        from airflow.models import Pool
        from airflow.utils.db import create_session

        with create_session() as session:
            pool = session.query(Pool).filter(Pool.pool == name).first()

            if pool:
                # Update existing pool
                pool.slots = slots
                pool.description = description
                logger.info(f"Updated pool '{name}' with {slots} slots")
            else:
                # Create new pool
                pool = Pool(pool=name, slots=slots, description=description)
                session.add(pool)
                logger.info(f"Created pool '{name}' with {slots} slots")

            session.commit()
            return True

    except ImportError:
        logger.error("Airflow not installed - cannot use Python API")
        return False
    except Exception as e:
        logger.error(f"Failed to create pool '{name}' via API: {e}")
        return False


def list_pools_via_cli() -> bool:
    """List existing pools"""
    try:
        cmd = ['airflow', 'pools', 'list', '-o', 'json']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            import json
            pools = json.loads(result.stdout)
            logger.info("Existing pools:")
            for pool in pools:
                logger.info(f"  - {pool['pool']}: {pool['slots']} slots - {pool.get('description', '')}")
            return True
        else:
            logger.warning(f"Could not list pools: {result.stderr}")
            return False

    except Exception as e:
        logger.warning(f"Could not list pools: {e}")
        return False


def setup_all_pools():
    """Setup all required pools"""
    logger.info("Setting up Airflow pools for investment analysis platform...")

    success_count = 0
    fail_count = 0

    for pool_config in POOLS:
        name = pool_config['name']
        slots = pool_config['slots']
        description = pool_config['description']

        # Try CLI first, then API
        if setup_pool_via_cli(name, slots, description):
            success_count += 1
        elif setup_pool_via_api(name, slots, description):
            success_count += 1
        else:
            fail_count += 1
            logger.error(f"Could not create pool '{name}' via any method")

    # List final state
    list_pools_via_cli()

    logger.info(f"Pool setup complete: {success_count} successful, {fail_count} failed")

    return fail_count == 0


def create_pool_sql_script():
    """Generate SQL script to create pools directly in Airflow database"""
    sql_lines = [
        "-- SQL script to create Airflow pools",
        "-- Run this against your Airflow metadata database if CLI/API methods don't work",
        ""
    ]

    for pool_config in POOLS:
        name = pool_config['name']
        slots = pool_config['slots']
        description = pool_config['description']

        sql_lines.append(f"""
-- Pool: {name}
INSERT INTO slot_pool (pool, slots, description)
VALUES ('{name}', {slots}, '{description}')
ON CONFLICT (pool) DO UPDATE SET
    slots = EXCLUDED.slots,
    description = EXCLUDED.description;
""")

    sql_script = '\n'.join(sql_lines)

    # Write to file
    sql_file = os.path.join(os.path.dirname(__file__), 'airflow_pools.sql')
    with open(sql_file, 'w') as f:
        f.write(sql_script)

    logger.info(f"SQL script written to: {sql_file}")

    return sql_script


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Setup Airflow pools')
    parser.add_argument('--sql', action='store_true', help='Generate SQL script instead of executing')
    parser.add_argument('--list', action='store_true', help='List existing pools only')

    args = parser.parse_args()

    if args.sql:
        create_pool_sql_script()
    elif args.list:
        list_pools_via_cli()
    else:
        success = setup_all_pools()
        sys.exit(0 if success else 1)
