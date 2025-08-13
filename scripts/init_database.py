#!/usr/bin/env python3
"""
Database Initialization Script with TimescaleDB Support
"""
from backend.utils.db_init import DatabaseInitializer
from backend.utils.db_timescale_init import TimescaleDBInitializer

import sys
import os
from pathlib import Path


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))



def main():
    """Initialize database with schema, TimescaleDB, and data"""
    print("Initializing Investment Analysis Database...")
    
    # First, initialize base database
    initializer = DatabaseInitializer()
    success = initializer.initialize()
    
    if not success:
        print("❌ Base database initialization failed!")
        return 1
    
    print("✅ Base database initialized successfully!")
    
    # Now initialize TimescaleDB
    print("\nInitializing TimescaleDB for time-series optimization...")
    timescale_initializer = TimescaleDBInitializer()
    timescale_success = timescale_initializer.initialize_timescaledb()
    
    if timescale_success:
        print("✅ TimescaleDB initialization completed successfully!")
        
        # Get compression stats
        stats = timescale_initializer.get_compression_stats()
        if stats.get('compression_stats'):
            print("\nCompression Statistics:")
            for stat in stats['compression_stats']:
                print(f"  - {stat['table']}: {stat.get('compression_ratio', 0)}% compression")
        
        return 0
    else:
        print("⚠️  TimescaleDB initialization failed - continuing with standard PostgreSQL")
        return 0  # Don't fail completely if TimescaleDB isn't available

if __name__ == "__main__":
    sys.exit(main())