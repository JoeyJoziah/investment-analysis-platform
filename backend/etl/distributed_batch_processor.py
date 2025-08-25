"""
Distributed Batch Processor for Large-Scale Stock Data Extraction
Handles 6000+ stocks with intelligent load balancing and resumable processing
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import pickle
import random
import time
from pathlib import Path

from .multi_source_extractor import MultiSourceStockExtractor, ExtractionResult, extract_stocks_data

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    job_id: str
    tickers: List[str]
    status: str  # 'pending', 'running', 'completed', 'failed', 'paused'
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    total_tickers: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.total_tickers == 0:
            self.total_tickers = len(self.tickers)


@dataclass
class ProcessorConfig:
    max_concurrent_jobs: int = 3
    max_concurrent_per_job: int = 8
    batch_size: int = 20
    cache_dir: str = "/tmp/stock_cache"
    job_timeout_hours: int = 12
    retry_failed_jobs: bool = True
    max_retries: int = 3
    inter_batch_delay: Tuple[float, float] = (2.0, 5.0)  # min, max seconds
    priority_processing: bool = True


class DistributedBatchProcessor:
    """Manages large-scale batch processing of stock data extraction"""
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        
        # Setup directories
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.jobs_dir = self.cache_dir / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.cache_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize job database
        self.job_db_path = self.cache_dir / "jobs.db"
        self._init_job_db()
        
        # Job management
        self.active_jobs = {}
        self.job_queue = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_jobs_created': 0,
            'total_jobs_completed': 0,
            'total_tickers_processed': 0,
            'total_successful_extractions': 0,
            'total_failed_extractions': 0,
            'start_time': None
        }
        
        logger.info(f"Initialized DistributedBatchProcessor with config: {asdict(self.config)}")
    
    def _init_job_db(self):
        """Initialize job tracking database"""
        conn = sqlite3.connect(self.job_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                priority INTEGER,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                progress INTEGER DEFAULT 0,
                total_tickers INTEGER,
                successful_extractions INTEGER DEFAULT 0,
                failed_extractions INTEGER DEFAULT 0,
                error_message TEXT,
                config_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_tickers (
                job_id TEXT,
                ticker TEXT,
                status TEXT,
                extraction_source TEXT,
                extracted_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (job_id) REFERENCES batch_jobs (job_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processor_stats (
                timestamp TIMESTAMP PRIMARY KEY,
                total_jobs_running INTEGER,
                tickers_per_minute REAL,
                success_rate REAL,
                avg_time_per_ticker REAL,
                memory_usage_mb REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_job(self, tickers: List[str], priority: int = 1, job_id: str = None) -> str:
        """Create a new batch job"""
        if not job_id:
            job_id = f"job_{int(time.time())}_{random.randint(1000, 9999)}"
        
        job = BatchJob(
            job_id=job_id,
            tickers=tickers,
            status='pending',
            priority=priority,
            created_at=datetime.now(),
            total_tickers=len(tickers)
        )
        
        # Save job to database
        self._save_job_to_db(job)
        
        # Save ticker list to file
        job_file = self.jobs_dir / f"{job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(asdict(job), f, indent=2, default=str)
        
        self.job_queue.append(job)
        self.stats['total_jobs_created'] += 1
        
        logger.info(f"Created job {job_id} with {len(tickers)} tickers (priority {priority})")
        return job_id
    
    def create_jobs_from_ticker_list(self, all_tickers: List[str], 
                                    tickers_per_job: int = 200,
                                    priority: int = 1) -> List[str]:
        """Split large ticker list into multiple jobs"""
        job_ids = []
        
        for i in range(0, len(all_tickers), tickers_per_job):
            batch_tickers = all_tickers[i:i + tickers_per_job]
            job_id = self.create_job(
                tickers=batch_tickers,
                priority=priority,
                job_id=f"batch_{i//tickers_per_job + 1}_{int(time.time())}"
            )
            job_ids.append(job_id)
        
        logger.info(f"Created {len(job_ids)} jobs from {len(all_tickers)} tickers")
        return job_ids
    
    def _save_job_to_db(self, job: BatchJob):
        """Save job to database"""
        conn = sqlite3.connect(self.job_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO batch_jobs 
            (job_id, status, priority, created_at, started_at, completed_at, 
             progress, total_tickers, successful_extractions, failed_extractions, 
             error_message, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.job_id, job.status, job.priority,
            job.created_at.isoformat() if job.created_at else None,
            job.started_at.isoformat() if job.started_at else None,
            job.completed_at.isoformat() if job.completed_at else None,
            job.progress, job.total_tickers,
            job.successful_extractions, job.failed_extractions,
            job.error_message, json.dumps(asdict(self.config), default=str)
        ))
        
        conn.commit()
        conn.close()
    
    def _load_job_from_db(self, job_id: str) -> Optional[BatchJob]:
        """Load job from database"""
        conn = sqlite3.connect(self.job_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return BatchJob(
            job_id=row[0],
            tickers=[],  # Will be loaded separately if needed
            status=row[1],
            priority=row[2],
            created_at=datetime.fromisoformat(row[3]) if row[3] else None,
            started_at=datetime.fromisoformat(row[4]) if row[4] else None,
            completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
            progress=row[6],
            total_tickers=row[7],
            successful_extractions=row[8],
            failed_extractions=row[9],
            error_message=row[10]
        )
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current status of a job"""
        job = self._load_job_from_db(job_id)
        if not job:
            return None
        
        return {
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'total_tickers': job.total_tickers,
            'successful_extractions': job.successful_extractions,
            'failed_extractions': job.failed_extractions,
            'completion_percentage': (job.progress / job.total_tickers * 100) if job.total_tickers > 0 else 0,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'error_message': job.error_message
        }
    
    def list_jobs(self, status_filter: str = None) -> List[Dict]:
        """List all jobs, optionally filtered by status"""
        conn = sqlite3.connect(self.job_db_path)
        cursor = conn.cursor()
        
        if status_filter:
            cursor.execute(
                "SELECT job_id, status, priority, progress, total_tickers, created_at FROM batch_jobs WHERE status = ? ORDER BY priority, created_at",
                (status_filter,)
            )
        else:
            cursor.execute(
                "SELECT job_id, status, priority, progress, total_tickers, created_at FROM batch_jobs ORDER BY priority, created_at"
            )
        
        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                'job_id': row[0],
                'status': row[1],
                'priority': row[2],
                'progress': row[3],
                'total_tickers': row[4],
                'completion_percentage': (row[3] / row[4] * 100) if row[4] > 0 else 0,
                'created_at': row[5]
            })
        
        conn.close()
        return jobs
    
    async def _process_single_job(self, job: BatchJob) -> BatchJob:
        """Process a single job"""
        logger.info(f"Starting job {job.job_id} with {len(job.tickers)} tickers")
        
        job.status = 'running'
        job.started_at = datetime.now()
        self._save_job_to_db(job)
        
        try:
            # Load tickers from file if not in memory
            if not job.tickers:
                job_file = self.jobs_dir / f"{job.job_id}.json"
                if job_file.exists():
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                        job.tickers = job_data.get('tickers', [])
            
            # Create extractor for this job
            extractor = MultiSourceStockExtractor(
                cache_dir=str(self.cache_dir),
                max_concurrent=self.config.max_concurrent_per_job
            )
            
            # Process in batches with progress tracking
            results = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(job.tickers), batch_size):
                batch_tickers = job.tickers[i:i + batch_size]
                logger.debug(f"Job {job.job_id}: Processing batch {i//batch_size + 1} ({len(batch_tickers)} tickers)")
                
                # Extract batch
                batch_results = await extractor.batch_extract(
                    tickers=batch_tickers,
                    batch_size=len(batch_tickers),
                    max_concurrent=self.config.max_concurrent_per_job
                )
                
                # Update job progress
                for result in batch_results:
                    if result.success:
                        job.successful_extractions += 1
                    else:
                        job.failed_extractions += 1
                    
                    results.append(result)
                
                job.progress = i + len(batch_tickers)
                self._save_job_to_db(job)
                
                # Inter-batch delay
                if i + batch_size < len(job.tickers):
                    delay = random.uniform(*self.config.inter_batch_delay)
                    await asyncio.sleep(delay)
            
            # Save results
            results_file = self.results_dir / f"{job.job_id}_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Mark job as completed
            job.status = 'completed'
            job.completed_at = datetime.now()
            self._save_job_to_db(job)
            
            # Update global stats
            self.stats['total_jobs_completed'] += 1
            self.stats['total_tickers_processed'] += len(job.tickers)
            self.stats['total_successful_extractions'] += job.successful_extractions
            self.stats['total_failed_extractions'] += job.failed_extractions
            
            logger.info(f"Completed job {job.job_id}: {job.successful_extractions}/{len(job.tickers)} successful")
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._save_job_to_db(job)
        
        return job
    
    async def start_processing(self):
        """Start the job processing loop"""
        if self.is_running:
            logger.warning("Processor already running")
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info("Started distributed batch processor")
        
        try:
            while self.is_running:
                # Load pending jobs from database if queue is empty
                if not self.job_queue:
                    self._load_pending_jobs()
                
                # Get available job slots
                available_slots = self.config.max_concurrent_jobs - len(self.active_jobs)
                
                if available_slots > 0 and self.job_queue:
                    # Sort jobs by priority
                    if self.config.priority_processing:
                        self.job_queue.sort(key=lambda x: x.priority)
                    
                    # Start new jobs up to available slots
                    jobs_to_start = self.job_queue[:available_slots]
                    self.job_queue = self.job_queue[available_slots:]
                    
                    for job in jobs_to_start:
                        task = asyncio.create_task(self._process_single_job(job))
                        self.active_jobs[job.job_id] = task
                        logger.info(f"Started processing job {job.job_id}")
                
                # Check for completed jobs
                completed_job_ids = []
                for job_id, task in self.active_jobs.items():
                    if task.done():
                        completed_job_ids.append(job_id)
                        try:
                            job_result = await task
                            logger.info(f"Job {job_id} completed with status: {job_result.status}")
                        except Exception as e:
                            logger.error(f"Job {job_id} failed: {e}")
                
                # Remove completed jobs
                for job_id in completed_job_ids:
                    del self.active_jobs[job_id]
                
                # Sleep before next iteration
                await asyncio.sleep(5)
                
                # Update stats periodically
                self._update_stats()
        
        except Exception as e:
            logger.error(f"Processor error: {e}")
        finally:
            self.is_running = False
            logger.info("Stopped distributed batch processor")
    
    def _load_pending_jobs(self):
        """Load pending jobs from database"""
        conn = sqlite3.connect(self.job_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT job_id FROM batch_jobs 
            WHERE status = 'pending'
            ORDER BY priority, created_at
            LIMIT 20
        """)
        
        for row in cursor.fetchall():
            job_id = row[0]
            
            # Load job data from file
            job_file = self.jobs_dir / f"{job_id}.json"
            if job_file.exists():
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                        job = BatchJob(
                            job_id=job_data['job_id'],
                            tickers=job_data['tickers'],
                            status=job_data['status'],
                            priority=job_data['priority'],
                            created_at=datetime.fromisoformat(job_data['created_at']),
                            total_tickers=job_data['total_tickers']
                        )
                        self.job_queue.append(job)
                except Exception as e:
                    logger.error(f"Error loading job {job_id}: {e}")
        
        conn.close()
    
    def _update_stats(self):
        """Update processor statistics"""
        try:
            conn = sqlite3.connect(self.job_db_path)
            cursor = conn.cursor()
            
            # Calculate current metrics
            current_time = datetime.now()
            running_jobs = len(self.active_jobs)
            
            # Calculate tickers per minute (last hour)
            hour_ago = (current_time - timedelta(hours=1)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM job_tickers 
                WHERE extracted_at > ? AND status = 'success'
            """, (hour_ago,))
            
            tickers_last_hour = cursor.fetchone()[0] or 0
            tickers_per_minute = tickers_last_hour / 60.0 if tickers_last_hour > 0 else 0
            
            # Calculate success rate (last hour)
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM job_tickers 
                WHERE extracted_at > ?
            """, (hour_ago,))
            
            result = cursor.fetchone()
            success_rate = result[0] if result and result[0] else 0
            
            # Insert stats
            cursor.execute("""
                INSERT INTO processor_stats 
                (timestamp, total_jobs_running, tickers_per_minute, success_rate, avg_time_per_ticker)
                VALUES (?, ?, ?, ?, ?)
            """, (current_time.isoformat(), running_jobs, tickers_per_minute, success_rate, 0))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Error updating stats: {e}")
    
    def stop_processing(self):
        """Stop the job processing loop"""
        logger.info("Stopping distributed batch processor...")
        self.is_running = False
    
    def pause_job(self, job_id: str):
        """Pause a specific job"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].cancel()
            del self.active_jobs[job_id]
            
            # Update job status
            conn = sqlite3.connect(self.job_db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE batch_jobs SET status = 'paused' WHERE job_id = ?", (job_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Paused job {job_id}")
    
    def resume_job(self, job_id: str):
        """Resume a paused job"""
        conn = sqlite3.connect(self.job_db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE batch_jobs SET status = 'pending' WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"Resumed job {job_id}")
    
    def get_processor_stats(self) -> Dict:
        """Get processor performance statistics"""
        stats = self.stats.copy()
        
        if stats['start_time']:
            runtime_hours = (datetime.now() - stats['start_time']).total_seconds() / 3600
            stats['runtime_hours'] = runtime_hours
            
            if runtime_hours > 0:
                stats['avg_tickers_per_hour'] = stats['total_tickers_processed'] / runtime_hours
        
        stats['active_jobs'] = len(self.active_jobs)
        stats['queued_jobs'] = len(self.job_queue)
        
        if stats['total_tickers_processed'] > 0:
            stats['overall_success_rate'] = (
                stats['total_successful_extractions'] / stats['total_tickers_processed'] * 100
            )
        
        return stats
    
    async def process_all_stocks(self, stock_tickers: List[str],
                               tickers_per_job: int = 200,
                               priority: int = 1) -> str:
        """
        Convenience method to process all stocks in the database
        
        Returns the ID of the first job created
        """
        logger.info(f"Creating jobs for {len(stock_tickers)} stocks")
        
        # Create jobs
        job_ids = self.create_jobs_from_ticker_list(
            all_tickers=stock_tickers,
            tickers_per_job=tickers_per_job,
            priority=priority
        )
        
        # Start processing if not already running
        if not self.is_running:
            # Start processing in background
            asyncio.create_task(self.start_processing())
        
        return job_ids[0] if job_ids else None


# Convenience functions
async def process_stock_universe(stock_tickers: List[str],
                               cache_dir: str = "/tmp/stock_cache",
                               tickers_per_job: int = 200,
                               max_concurrent_jobs: int = 3,
                               max_concurrent_per_job: int = 8) -> DistributedBatchProcessor:
    """
    Process the entire stock universe using distributed batch processing
    
    Args:
        stock_tickers: List of all stock tickers to process
        cache_dir: Directory for caching and job storage
        tickers_per_job: Number of tickers per job
        max_concurrent_jobs: Maximum concurrent jobs
        max_concurrent_per_job: Maximum concurrent requests per job
    
    Returns:
        DistributedBatchProcessor instance
    """
    config = ProcessorConfig(
        max_concurrent_jobs=max_concurrent_jobs,
        max_concurrent_per_job=max_concurrent_per_job,
        batch_size=20,
        cache_dir=cache_dir,
        job_timeout_hours=12,
        retry_failed_jobs=True
    )
    
    processor = DistributedBatchProcessor(config)
    
    # Create and start processing jobs
    await processor.process_all_stocks(
        stock_tickers=stock_tickers,
        tickers_per_job=tickers_per_job
    )
    
    return processor


if __name__ == "__main__":
    # Example usage
    async def test():
        # Test with sample tickers
        test_tickers = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-A', 'AVGO', 'JPM',
            'JNJ', 'WMT', 'V', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'PYPL', 'BAC',
            'ADBE', 'CMCSA', 'NFLX', 'ABT', 'CRM', 'NKE', 'TMO', 'ACN', 'COST', 'ABBV'
        ] * 10  # Simulate 300 tickers
        
        processor = await process_stock_universe(
            stock_tickers=test_tickers,
            tickers_per_job=50,
            max_concurrent_jobs=2,
            max_concurrent_per_job=5
        )
        
        # Monitor progress
        while processor.is_running:
            stats = processor.get_processor_stats()
            print(f"Processing: {stats['active_jobs']} active jobs, "
                  f"{stats['total_tickers_processed']} tickers processed")
            await asyncio.sleep(10)
        
        final_stats = processor.get_processor_stats()
        print(f"Final stats: {final_stats}")
    
    asyncio.run(test())