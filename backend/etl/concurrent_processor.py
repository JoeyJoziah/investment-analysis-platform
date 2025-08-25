"""
Concurrent Processing Engine for Unlimited Stock Data Extraction
Handles 6000+ stocks with intelligent throttling, load balancing, and resource management
"""

import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Set
import time
import threading
import queue
import multiprocessing as mp
from dataclasses import dataclass, field
import psutil
import os
import signal
import json
from collections import defaultdict, deque
import weakref
import traceback
import random

logger = logging.getLogger(__name__)

@dataclass
class ProcessingTask:
    """Represents a data extraction task"""
    task_id: str
    ticker: str
    priority: int = 1  # Lower = higher priority
    created_at: datetime = field(default_factory=datetime.now)
    max_attempts: int = 3
    current_attempts: int = 0
    timeout_seconds: int = 30
    context: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"{self.ticker}_{int(time.time()*1000)}"

@dataclass
class ProcessingResult:
    """Result of a processing task"""
    task_id: str
    ticker: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    processor_id: str = ""
    attempts_used: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ProcessorStats:
    """Statistics for a processor"""
    processor_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time_ms: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_rate: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return (self.tasks_completed / total) if total > 0 else 0.0
    
    @property
    def avg_execution_time_ms(self) -> float:
        return (self.total_execution_time_ms / self.tasks_completed) if self.tasks_completed > 0 else 0.0

class ThrottleManager:
    """Manages request throttling to avoid overwhelming servers"""
    
    def __init__(self, max_requests_per_second: int = 10, burst_capacity: int = 50):
        self.max_rps = max_requests_per_second
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # Dynamic throttling based on success rates
        self.recent_failures = deque(maxlen=100)
        self.adaptive_throttling = True
        self.current_delay = 0.1  # Base delay between requests
        
    def acquire_permit(self) -> bool:
        """Try to acquire a request permit"""
        with self.lock:
            now = time.time()
            
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.max_rps
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    async def wait_for_permit(self) -> None:
        """Wait until a request permit is available"""
        while not self.acquire_permit():
            # Calculate wait time
            with self.lock:
                wait_time = max(0.01, (1 - self.tokens) / self.max_rps)
            
            await asyncio.sleep(wait_time)
    
    def record_result(self, success: bool) -> None:
        """Record request result for adaptive throttling"""
        if not self.adaptive_throttling:
            return
        
        self.recent_failures.append(not success)
        
        # Adjust throttling based on recent failure rate
        if len(self.recent_failures) >= 10:
            failure_rate = sum(self.recent_failures) / len(self.recent_failures)
            
            if failure_rate > 0.2:  # More than 20% failures
                self.current_delay = min(self.current_delay * 1.5, 5.0)
                logger.warning(f"High failure rate ({failure_rate:.1%}), increasing delay to {self.current_delay:.2f}s")
            elif failure_rate < 0.05:  # Less than 5% failures
                self.current_delay = max(self.current_delay * 0.8, 0.1)
    
    def get_current_delay(self) -> float:
        """Get current delay between requests"""
        return self.current_delay

class ResourceMonitor:
    """Monitors system resources to prevent overload"""
    
    def __init__(self, max_cpu_percent: int = 80, max_memory_percent: int = 80):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.monitoring = False
        self.monitor_thread = None
        
        # Resource history for trend analysis
        self.cpu_history = deque(maxlen=60)  # Last 60 measurements
        self.memory_history = deque(maxlen=60)
        
        # Alerts
        self.high_resource_usage = False
        self.last_alert_time = 0
        
    def start_monitoring(self):
        """Start resource monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                
                # Store in history
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_percent)
                
                # Check for high resource usage
                cpu_high = cpu_percent > self.max_cpu_percent
                memory_high = memory_percent > self.max_memory_percent
                
                if cpu_high or memory_high:
                    if not self.high_resource_usage:
                        self.high_resource_usage = True
                        logger.warning(f"High resource usage detected: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
                else:
                    if self.high_resource_usage:
                        self.high_resource_usage = False
                        logger.info("Resource usage returned to normal levels")
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)
    
    def should_throttle_processing(self) -> bool:
        """Check if processing should be throttled due to high resource usage"""
        return self.high_resource_usage
    
    def get_resource_stats(self) -> Dict:
        """Get current resource statistics"""
        cpu_avg = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        memory_avg = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'cpu_avg': cpu_avg,
            'memory_percent': psutil.virtual_memory().percent,
            'memory_avg': memory_avg,
            'high_resource_usage': self.high_resource_usage,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

class WorkerPool:
    """Pool of worker processes/threads for concurrent processing"""
    
    def __init__(self, 
                 worker_count: int = None,
                 use_processes: bool = False,
                 max_tasks_per_worker: int = 1000,
                 worker_timeout: int = 300):
        
        self.use_processes = use_processes
        self.max_tasks_per_worker = max_tasks_per_worker
        self.worker_timeout = worker_timeout
        
        # Auto-detect optimal worker count
        if worker_count is None:
            cpu_count = multiprocessing.cpu_count()
            if use_processes:
                # For CPU-bound tasks (like data processing)
                worker_count = cpu_count
            else:
                # For I/O-bound tasks (like web scraping)
                worker_count = min(cpu_count * 4, 50)
        
        self.worker_count = worker_count
        self.executor = None
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers_busy = set()
        self.worker_stats = {}
        
        # Worker lifecycle management
        self.shutdown_event = threading.Event()
        self.workers = []
        
        logger.info(f"Initializing worker pool: {worker_count} {'processes' if use_processes else 'threads'}")
    
    def start(self):
        """Start the worker pool"""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(
                max_workers=self.worker_count,
                initializer=self._init_worker_process
            )
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.worker_count)
        
        # Initialize worker statistics
        for i in range(self.worker_count):
            worker_id = f"worker_{i}"
            self.worker_stats[worker_id] = ProcessorStats(processor_id=worker_id)
        
        logger.info(f"Worker pool started with {self.worker_count} workers")
    
    def stop(self):
        """Stop the worker pool gracefully"""
        logger.info("Shutting down worker pool...")
        
        self.shutdown_event.set()
        
        if self.executor:
            self.executor.shutdown(wait=True, timeout=30)
        
        logger.info("Worker pool shut down complete")
    
    def submit_task(self, func: Callable, task: ProcessingTask) -> asyncio.Future:
        """Submit a task to the worker pool"""
        if not self.executor:
            raise RuntimeError("Worker pool not started")
        
        # Wrap the function to include error handling and stats
        wrapped_func = self._wrap_worker_function(func, task)
        
        # Submit to executor
        future = self.executor.submit(wrapped_func, task)
        return future
    
    def _wrap_worker_function(self, func: Callable, task: ProcessingTask) -> Callable:
        """Wrap worker function with error handling and statistics"""
        def wrapper(task_obj: ProcessingTask) -> ProcessingResult:
            worker_id = f"worker_{threading.current_thread().ident}"
            start_time = time.time()
            
            try:
                # Update worker stats
                if worker_id in self.worker_stats:
                    self.worker_stats[worker_id].last_activity = datetime.now()
                
                # Execute the actual function
                result_data = func(task_obj)
                
                execution_time = int((time.time() - start_time) * 1000)
                
                # Create successful result
                result = ProcessingResult(
                    task_id=task_obj.task_id,
                    ticker=task_obj.ticker,
                    success=True,
                    data=result_data,
                    execution_time_ms=execution_time,
                    processor_id=worker_id,
                    attempts_used=task_obj.current_attempts + 1
                )
                
                # Update stats
                if worker_id in self.worker_stats:
                    stats = self.worker_stats[worker_id]
                    stats.tasks_completed += 1
                    stats.total_execution_time_ms += execution_time
                
                return result
                
            except Exception as e:
                execution_time = int((time.time() - start_time) * 1000)
                error_msg = f"{type(e).__name__}: {str(e)}"
                
                # Log the error with full traceback
                logger.error(f"Worker {worker_id} task {task_obj.task_id} failed: {error_msg}")
                logger.debug(traceback.format_exc())
                
                # Create error result
                result = ProcessingResult(
                    task_id=task_obj.task_id,
                    ticker=task_obj.ticker,
                    success=False,
                    error=error_msg,
                    execution_time_ms=execution_time,
                    processor_id=worker_id,
                    attempts_used=task_obj.current_attempts + 1
                )
                
                # Update error stats
                if worker_id in self.worker_stats:
                    stats = self.worker_stats[worker_id]
                    stats.tasks_failed += 1
                    stats.error_rate = stats.tasks_failed / (stats.tasks_completed + stats.tasks_failed)
                
                return result
        
        return wrapper
    
    @staticmethod
    def _init_worker_process():
        """Initialize worker process (for ProcessPoolExecutor)"""
        # Set up signal handling to allow graceful shutdown
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    def get_worker_stats(self) -> Dict[str, ProcessorStats]:
        """Get statistics for all workers"""
        return self.worker_stats.copy()
    
    def get_pool_stats(self) -> Dict:
        """Get overall pool statistics"""
        if not self.worker_stats:
            return {}
        
        total_completed = sum(s.tasks_completed for s in self.worker_stats.values())
        total_failed = sum(s.tasks_failed for s in self.worker_stats.values())
        total_execution_time = sum(s.total_execution_time_ms for s in self.worker_stats.values())
        
        return {
            'worker_count': len(self.worker_stats),
            'total_tasks_completed': total_completed,
            'total_tasks_failed': total_failed,
            'overall_success_rate': total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0,
            'avg_execution_time_ms': total_execution_time / total_completed if total_completed > 0 else 0,
            'workers_active': len([s for s in self.worker_stats.values() if (datetime.now() - s.last_activity).seconds < 60])
        }

class ConcurrentProcessor:
    """Main concurrent processing engine for stock data extraction"""
    
    def __init__(self,
                 max_concurrent_requests: int = 50,
                 max_requests_per_second: int = 10,
                 use_processes: bool = False,
                 enable_resource_monitoring: bool = True,
                 retry_failed_tasks: bool = True):
        
        self.max_concurrent = max_concurrent_requests
        self.retry_failed_tasks = retry_failed_tasks
        
        # Initialize components
        self.throttle_manager = ThrottleManager(max_requests_per_second)
        self.resource_monitor = ResourceMonitor() if enable_resource_monitoring else None
        self.worker_pool = WorkerPool(worker_count=max_concurrent_requests, use_processes=use_processes)
        
        # Task management
        self.pending_tasks = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_results = []
        self.failed_tasks = []
        
        # Processing statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'total_processing_time_ms': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Control flags
        self.processing = False
        self.shutdown_requested = False
        
        logger.info(f"Initialized ConcurrentProcessor: {max_concurrent_requests} max concurrent, {max_requests_per_second} RPS")
    
    def start(self):
        """Start the concurrent processor"""
        if self.processing:
            return
        
        self.processing = True
        self.shutdown_requested = False
        
        # Start components
        self.worker_pool.start()
        
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        self.stats['start_time'] = datetime.now()
        logger.info("ConcurrentProcessor started")
    
    def stop(self):
        """Stop the concurrent processor gracefully"""
        logger.info("Stopping ConcurrentProcessor...")
        
        self.shutdown_requested = True
        self.processing = False
        
        # Stop components
        self.worker_pool.stop()
        
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        self.stats['end_time'] = datetime.now()
        logger.info("ConcurrentProcessor stopped")
    
    async def process_tasks(self, 
                           tasks: List[ProcessingTask], 
                           worker_function: Callable,
                           progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """Process a batch of tasks concurrently"""
        
        if not self.processing:
            self.start()
        
        logger.info(f"Starting concurrent processing of {len(tasks)} tasks")
        
        # Add tasks to queue
        for task in tasks:
            # Assign priority based on task characteristics
            priority = self._calculate_task_priority(task)
            self.pending_tasks.put((priority, task))
            self.stats['tasks_submitted'] += 1
        
        # Process tasks with concurrency control
        active_futures = {}
        all_results = []
        processed_count = 0
        
        try:
            while not self.pending_tasks.empty() or active_futures:
                # Check resource usage and throttle if needed
                if self.resource_monitor and self.resource_monitor.should_throttle_processing():
                    logger.info("Throttling processing due to high resource usage")
                    await asyncio.sleep(2)
                    continue
                
                # Submit new tasks if we have capacity
                while (len(active_futures) < self.max_concurrent and 
                       not self.pending_tasks.empty()):
                    
                    # Wait for throttling permit
                    await self.throttle_manager.wait_for_permit()
                    
                    try:
                        priority, task = self.pending_tasks.get_nowait()
                        
                        # Submit task to worker pool
                        future = self.worker_pool.submit_task(worker_function, task)
                        active_futures[future] = task
                        
                        logger.debug(f"Submitted task {task.task_id} (priority: {priority})")
                        
                    except queue.Empty:
                        break
                
                # Check for completed tasks
                if active_futures:
                    # Wait for at least one task to complete
                    completed_futures = []
                    
                    for future in list(active_futures.keys()):
                        if future.done():
                            completed_futures.append(future)
                    
                    # Process completed futures
                    for future in completed_futures:
                        task = active_futures.pop(future)
                        
                        try:
                            result = future.result()
                            
                            # Record throttling result
                            self.throttle_manager.record_result(result.success)
                            
                            if result.success:
                                all_results.append(result)
                                self.stats['tasks_completed'] += 1
                                processed_count += 1
                                
                                logger.debug(f"Task {task.task_id} completed successfully in {result.execution_time_ms}ms")
                                
                            else:
                                # Handle failed task
                                if self.retry_failed_tasks and task.current_attempts < task.max_attempts - 1:
                                    # Retry the task
                                    task.current_attempts += 1
                                    retry_priority = self._calculate_task_priority(task) + 10  # Lower priority for retries
                                    self.pending_tasks.put((retry_priority, task))
                                    self.stats['tasks_retried'] += 1
                                    
                                    logger.warning(f"Retrying task {task.task_id} (attempt {task.current_attempts + 1}/{task.max_attempts})")
                                else:
                                    # Task finally failed
                                    all_results.append(result)
                                    self.failed_tasks.append(task)
                                    self.stats['tasks_failed'] += 1
                                    processed_count += 1
                                    
                                    logger.error(f"Task {task.task_id} failed permanently: {result.error}")
                        
                        except Exception as e:
                            logger.error(f"Error processing future for task {task.task_id}: {e}")
                            self.stats['tasks_failed'] += 1
                            processed_count += 1
                    
                    # Call progress callback
                    if progress_callback and processed_count > 0:
                        try:
                            await progress_callback(processed_count, len(tasks), all_results[-10:])
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")
                
                # Small delay to prevent busy waiting
                if not completed_futures:
                    await asyncio.sleep(0.1)
                
                # Check for shutdown
                if self.shutdown_requested:
                    logger.warning("Shutdown requested during processing")
                    break
        
        finally:
            # Cancel any remaining futures
            for future in active_futures:
                if not future.done():
                    future.cancel()
        
        # Calculate final statistics
        successful_results = [r for r in all_results if r.success]
        failed_results = [r for r in all_results if not r.success]
        
        total_time = sum(r.execution_time_ms for r in all_results)
        self.stats['total_processing_time_ms'] += total_time
        
        logger.info(f"Concurrent processing completed:")
        logger.info(f"  Total: {len(all_results)}/{len(tasks)} processed")
        logger.info(f"  Successful: {len(successful_results)}")
        logger.info(f"  Failed: {len(failed_results)}")
        logger.info(f"  Success Rate: {len(successful_results)/len(all_results)*100:.1f}%" if all_results else "0%")
        logger.info(f"  Average Time: {total_time/len(all_results):.0f}ms per task" if all_results else "N/A")
        
        return all_results
    
    def _calculate_task_priority(self, task: ProcessingTask) -> int:
        """Calculate priority for task scheduling (lower = higher priority)"""
        base_priority = task.priority
        
        # Increase priority (lower number) for:
        # - Tasks that have been waiting longer
        wait_time_minutes = (datetime.now() - task.created_at).total_seconds() / 60
        wait_bonus = int(wait_time_minutes / 5)  # +1 priority per 5 minutes waiting
        
        # - Tasks that have failed before (give them another chance)
        retry_bonus = task.current_attempts * 2
        
        final_priority = max(1, base_priority - wait_bonus - retry_bonus)
        return final_priority
    
    def get_processing_stats(self) -> Dict:
        """Get comprehensive processing statistics"""
        worker_stats = self.worker_pool.get_pool_stats() if self.worker_pool else {}
        resource_stats = self.resource_monitor.get_resource_stats() if self.resource_monitor else {}
        
        processing_time = 0
        if self.stats['start_time']:
            end_time = self.stats['end_time'] or datetime.now()
            processing_time = (end_time - self.stats['start_time']).total_seconds()
        
        return {
            'processing': {
                'tasks_submitted': self.stats['tasks_submitted'],
                'tasks_completed': self.stats['tasks_completed'],
                'tasks_failed': self.stats['tasks_failed'],
                'tasks_retried': self.stats['tasks_retried'],
                'success_rate': self.stats['tasks_completed'] / max(self.stats['tasks_submitted'], 1),
                'processing_time_seconds': processing_time,
                'avg_task_time_ms': self.stats['total_processing_time_ms'] / max(self.stats['tasks_completed'], 1)
            },
            'workers': worker_stats,
            'resources': resource_stats,
            'throttling': {
                'current_delay_seconds': self.throttle_manager.get_current_delay(),
                'tokens_available': self.throttle_manager.tokens,
                'recent_failures': len([f for f in self.throttle_manager.recent_failures if f])
            }
        }
    
    async def process_stock_extraction(self, 
                                     tickers: List[str],
                                     extraction_function: Callable,
                                     max_concurrent: Optional[int] = None,
                                     progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """Convenience method for stock data extraction processing"""
        
        # Override max concurrent if specified
        if max_concurrent:
            original_max = self.max_concurrent
            self.max_concurrent = max_concurrent
        
        try:
            # Create tasks for each ticker
            tasks = []
            for i, ticker in enumerate(tickers):
                task = ProcessingTask(
                    task_id=f"extract_{ticker}_{int(time.time())}",
                    ticker=ticker,
                    priority=1,  # All extraction tasks have equal priority
                    max_attempts=3,
                    timeout_seconds=30,
                    context={'extraction_type': 'stock_data'}
                )
                tasks.append(task)
            
            # Process the tasks
            results = await self.process_tasks(
                tasks=tasks,
                worker_function=extraction_function,
                progress_callback=progress_callback
            )
            
            return results
        
        finally:
            # Restore original max concurrent
            if max_concurrent:
                self.max_concurrent = original_max

# Example usage and testing
async def test_concurrent_processor():
    """Test the concurrent processing engine"""
    
    # Mock extraction function for testing
    def mock_extraction_function(task: ProcessingTask) -> Dict:
        """Mock function that simulates stock data extraction"""
        import time
        import random
        
        # Simulate processing time
        processing_time = random.uniform(0.1, 2.0)
        time.sleep(processing_time)
        
        # Simulate occasional failures
        if random.random() < 0.1:  # 10% failure rate
            raise Exception(f"Mock extraction failed for {task.ticker}")
        
        # Return mock data
        return {
            'ticker': task.ticker,
            'price': round(random.uniform(10, 1000), 2),
            'volume': random.randint(100000, 10000000),
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_extractor'
        }
    
    # Progress callback for monitoring
    async def progress_callback(completed: int, total: int, recent_results: List[ProcessingResult]):
        success_count = sum(1 for r in recent_results if r.success)
        logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) - Recent success: {success_count}/{len(recent_results)}")
    
    # Initialize processor
    processor = ConcurrentProcessor(
        max_concurrent_requests=20,
        max_requests_per_second=15,
        use_processes=False,  # Use threads for I/O-bound tasks
        enable_resource_monitoring=True
    )
    
    try:
        # Test with sample tickers
        test_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
            'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'ADBE', 'CRM', 'NFLX',
            'CMCSA', 'ABT', 'COST', 'TMO', 'ACN', 'AVGO', 'DHR', 'NEE', 'TXN', 'LIN'
        ]
        
        logger.info(f"Testing concurrent processing with {len(test_tickers)} tickers")
        
        # Start processing
        start_time = time.time()
        
        results = await processor.process_stock_extraction(
            tickers=test_tickers,
            extraction_function=mock_extraction_function,
            progress_callback=progress_callback
        )
        
        processing_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        logger.info("=== Concurrent Processing Test Results ===")
        logger.info(f"Total tickers: {len(test_tickers)}")
        logger.info(f"Results returned: {len(results)}")
        logger.info(f"Successful: {len(successful_results)}")
        logger.info(f"Failed: {len(failed_results)}")
        logger.info(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        logger.info(f"Average time per ticker: {processing_time/len(test_tickers):.2f} seconds")
        
        # Show detailed statistics
        stats = processor.get_processing_stats()
        logger.info("=== Detailed Statistics ===")
        logger.info(f"Processing stats: {json.dumps(stats['processing'], indent=2)}")
        logger.info(f"Worker stats: {json.dumps(stats['workers'], indent=2)}")
        logger.info(f"Resource stats: {json.dumps(stats['resources'], indent=2)}")
        
        # Show sample results
        logger.info("=== Sample Results ===")
        for result in successful_results[:3]:
            logger.info(f"✓ {result.ticker}: {result.data} ({result.execution_time_ms}ms)")
        
        for result in failed_results[:2]:
            logger.info(f"✗ {result.ticker}: {result.error}")
        
        logger.info("Concurrent processor test completed successfully!")
        
    except Exception as e:
        logger.error(f"Concurrent processor test failed: {e}")
        raise
    
    finally:
        processor.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    asyncio.run(test_concurrent_processor())