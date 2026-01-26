"""
Batch Processing Utilities for Airflow DAGs
Provides optimized parallel processing for stock data operations.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 50
    max_workers: int = 8
    use_processes: bool = False  # False = threads (better for I/O), True = processes (better for CPU)
    retry_on_failure: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    inter_batch_delay: float = 0.1


@dataclass
class BatchResult:
    """Result of a batch operation"""
    batch_id: int
    success_count: int = 0
    error_count: int = 0
    processed_items: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_id': self.batch_id,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'errors': self.errors[:10],  # Limit errors for XCom
            'execution_time_ms': self.execution_time_ms
        }


@dataclass
class ProcessingStats:
    """Aggregated processing statistics"""
    total_items: int = 0
    total_success: int = 0
    total_errors: int = 0
    total_batches: int = 0
    elapsed_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.total_success / self.total_items * 100

    @property
    def throughput_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.total_items / self.elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_items': self.total_items,
            'total_success': self.total_success,
            'total_errors': self.total_errors,
            'total_batches': self.total_batches,
            'elapsed_seconds': self.elapsed_seconds,
            'success_rate': self.success_rate,
            'throughput_per_second': self.throughput_per_second
        }


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, max_rate: int = 60, burst_capacity: int = 10):
        """
        Initialize rate limiter.

        Args:
            max_rate: Maximum requests per minute
            burst_capacity: Allow burst of this many requests
        """
        self.max_rate = max_rate
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire a rate limit token.

        Args:
            blocking: If True, wait for token. If False, return immediately.

        Returns:
            True if token acquired, False if not available (non-blocking only)
        """
        while True:
            with self.lock:
                now = time.time()

                # Refill tokens based on time elapsed
                elapsed = now - self.last_refill
                tokens_to_add = elapsed * (self.max_rate / 60.0)
                self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
                self.last_refill = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

                if not blocking:
                    return False

            # Wait before retrying
            time.sleep(0.1)

    def wait(self):
        """Wait for a rate limit token (blocking)"""
        self.acquire(blocking=True)


class BatchProcessor:
    """
    Parallel batch processor for stock data operations.
    Supports both thread-based (I/O bound) and process-based (CPU bound) parallelism.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.rate_limiter = RateLimiter(max_rate=60, burst_capacity=10)
        self.stats = ProcessingStats()

    def create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Split items into batches"""
        batches = []
        for i in range(0, len(items), self.config.batch_size):
            batches.append(items[i:i + self.config.batch_size])
        logger.info(f"Created {len(batches)} batches of ~{self.config.batch_size} items each")
        return batches

    def process_parallel(
        self,
        items: List[Any],
        worker_func: Callable[[int, List[Any]], BatchResult],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[ProcessingStats, List[BatchResult]]:
        """
        Process items in parallel using batches.

        Args:
            items: List of items to process
            worker_func: Function that processes a batch: (batch_id, batch_items) -> BatchResult
            progress_callback: Optional callback for progress updates: (completed, total)

        Returns:
            Tuple of (ProcessingStats, list of BatchResults)
        """
        if not items:
            logger.warning("No items to process")
            return self.stats, []

        logger.info(f"Starting parallel processing of {len(items)} items with {self.config.max_workers} workers")
        start_time = time.time()

        # Create batches
        batches = self.create_batches(items)

        # Prepare batch arguments
        batch_args = [(i, batch) for i, batch in enumerate(batches)]

        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor

        all_results: List[BatchResult] = []
        completed_batches = 0

        with ExecutorClass(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(worker_func, batch_id, batch_items): batch_id
                for batch_id, batch_items in batch_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]

                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    all_results.append(result)

                    self.stats.total_success += result.success_count
                    self.stats.total_errors += result.error_count

                    logger.debug(f"Batch {batch_id} completed: {result.success_count} success, {result.error_count} errors")

                except Exception as e:
                    logger.error(f"Batch {batch_id} failed with exception: {e}")

                    # Create error result
                    error_result = BatchResult(
                        batch_id=batch_id,
                        error_count=self.config.batch_size,
                        errors=[str(e)]
                    )
                    all_results.append(error_result)
                    self.stats.total_errors += self.config.batch_size

                completed_batches += 1

                # Progress callback
                if progress_callback:
                    progress_callback(completed_batches, len(batches))

        # Calculate final stats
        elapsed_time = time.time() - start_time
        self.stats.total_items = len(items)
        self.stats.total_batches = len(batches)
        self.stats.elapsed_seconds = elapsed_time

        logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total: {self.stats.total_success} success, {self.stats.total_errors} errors")
        logger.info(f"Throughput: {self.stats.throughput_per_second:.1f} items/second")

        return self.stats, all_results

    def process_with_retry(
        self,
        batch_id: int,
        items: List[Any],
        process_func: Callable[[Any], bool],
        item_identifier: Callable[[Any], str] = str
    ) -> BatchResult:
        """
        Process a batch with retry logic for failed items.

        Args:
            batch_id: Batch identifier
            items: Items to process
            process_func: Function to process single item, returns True on success
            item_identifier: Function to get string identifier for item

        Returns:
            BatchResult with processing outcomes
        """
        start_time = time.time()
        result = BatchResult(batch_id=batch_id)

        # Track items to retry
        items_to_process = items.copy()
        attempt = 0

        while items_to_process and attempt < self.config.max_retries:
            attempt += 1
            failed_items = []

            for item in items_to_process:
                item_id = item_identifier(item)

                try:
                    # Apply rate limiting
                    self.rate_limiter.wait()

                    # Process item
                    success = process_func(item)

                    if success:
                        result.success_count += 1
                        result.processed_items.append(item_id)
                    else:
                        failed_items.append(item)

                except Exception as e:
                    failed_items.append(item)
                    result.errors.append(f"{item_id}: {str(e)[:100]}")

            # Prepare for retry
            items_to_process = failed_items

            if items_to_process and attempt < self.config.max_retries:
                logger.warning(f"Batch {batch_id} retry {attempt}: {len(items_to_process)} items failed")
                time.sleep(1.0 * attempt)  # Exponential backoff

        # Record final failures
        for item in items_to_process:
            item_id = item_identifier(item)
            result.error_count += 1
            result.failed_items.append(item_id)

        result.execution_time_ms = int((time.time() - start_time) * 1000)

        return result


def create_stock_batch_processor(
    batch_size: int = 50,
    max_workers: int = 8,
    api_rate_limit: int = 60
) -> BatchProcessor:
    """
    Factory function to create a batch processor optimized for stock data.

    Args:
        batch_size: Number of stocks per batch
        max_workers: Number of parallel workers
        api_rate_limit: API calls per minute limit

    Returns:
        Configured BatchProcessor instance
    """
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        use_processes=False,  # Use threads for I/O-bound API calls
        retry_on_failure=True,
        max_retries=3,
        timeout_seconds=300
    )

    processor = BatchProcessor(config)
    processor.rate_limiter = RateLimiter(max_rate=api_rate_limit, burst_capacity=10)

    return processor


# Example usage for testing
if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO)

    # Test the batch processor
    def mock_worker(batch_id: int, items: List[str]) -> BatchResult:
        """Mock worker that simulates processing"""
        result = BatchResult(batch_id=batch_id)

        for item in items:
            # Simulate some work
            time.sleep(0.05)

            # Simulate 10% failure rate
            if random.random() < 0.1:
                result.error_count += 1
                result.failed_items.append(item)
                result.errors.append(f"{item}: Random failure")
            else:
                result.success_count += 1
                result.processed_items.append(item)

        return result

    # Create test data
    test_items = [f"TICKER_{i}" for i in range(200)]

    # Create processor
    processor = create_stock_batch_processor(batch_size=20, max_workers=4)

    # Process
    def progress(completed, total):
        print(f"Progress: {completed}/{total} batches")

    stats, results = processor.process_parallel(
        items=test_items,
        worker_func=mock_worker,
        progress_callback=progress
    )

    print("\nFinal Stats:")
    print(f"  Total: {stats.total_items}")
    print(f"  Success: {stats.total_success}")
    print(f"  Errors: {stats.total_errors}")
    print(f"  Rate: {stats.success_rate:.1f}%")
    print(f"  Throughput: {stats.throughput_per_second:.1f}/sec")
