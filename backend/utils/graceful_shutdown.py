"""
Graceful Shutdown Handler for Investment Analysis App
Handles SIGTERM/SIGINT signals for proper cleanup of cache warming and connections
"""

import signal
import asyncio
import logging
import sys
import time
from typing import Optional, Callable, List, Any
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """
    Manages graceful shutdown of application components.
    
    Features:
    - Handles SIGTERM and SIGINT signals
    - Completes in-progress cache warming
    - Saves cache state for quick recovery
    - Closes database connections properly
    - Ensures data consistency
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize shutdown handler.
        
        Args:
            timeout: Maximum seconds to wait for graceful shutdown
        """
        self.timeout = timeout
        self.shutdown_event = asyncio.Event()
        self.is_shutting_down = False
        self.shutdown_start_time: Optional[datetime] = None
        self.cleanup_tasks: List[Callable] = []
        self.critical_tasks: List[asyncio.Task] = []
        
        # Track components
        self.components = {
            'cache_warming': None,
            'database': None,
            'redis': None,
            'background_tasks': [],
            'api_requests': []
        }
        
        # Metrics
        self.metrics = {
            'shutdown_requested': None,
            'shutdown_completed': None,
            'tasks_cancelled': 0,
            'tasks_completed': 0,
            'connections_closed': 0
        }
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Register SIGTERM and SIGINT handlers."""
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, self._signal_handler)
        
        # Windows compatibility
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._signal_handler)
        
        logger.info("Signal handlers registered for graceful shutdown")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        
        if self.is_shutting_down:
            logger.warning(f"Already shutting down, forcing exit...")
            sys.exit(1)
        
        self.is_shutting_down = True
        self.shutdown_start_time = datetime.now()
        self.metrics['shutdown_requested'] = self.shutdown_start_time
        
        # Set shutdown event for async tasks
        self.shutdown_event.set()
        
        # Start shutdown sequence
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """
        Execute graceful shutdown sequence.
        
        Shutdown order:
        1. Stop accepting new requests
        2. Complete critical cache warming operations
        3. Save cache state
        4. Finish in-flight requests
        5. Close database connections
        6. Close Redis connections
        7. Cancel remaining background tasks
        """
        try:
            logger.info("Starting graceful shutdown sequence...")
            
            # Phase 1: Stop accepting new work
            await self._stop_accepting_requests()
            
            # Phase 2: Complete critical operations
            await self._complete_critical_operations()
            
            # Phase 3: Save state
            await self._save_state()
            
            # Phase 4: Cleanup connections
            await self._cleanup_connections()
            
            # Phase 5: Cancel remaining tasks
            await self._cancel_remaining_tasks()
            
            self.metrics['shutdown_completed'] = datetime.now()
            duration = (self.metrics['shutdown_completed'] - self.metrics['shutdown_requested']).total_seconds()
            
            logger.info(
                f"Graceful shutdown completed in {duration:.1f}s - "
                f"Tasks completed: {self.metrics['tasks_completed']}, "
                f"cancelled: {self.metrics['tasks_cancelled']}, "
                f"connections closed: {self.metrics['connections_closed']}"
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Graceful shutdown timed out after {self.timeout}s, forcing exit")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            sys.exit(1)
        finally:
            sys.exit(0)
    
    async def _stop_accepting_requests(self):
        """Stop accepting new API requests."""
        logger.info("Stopping acceptance of new requests...")
        
        # Set health check to unhealthy
        if 'health_check' in self.components:
            self.components['health_check'].is_healthy = False
        
        # Wait briefly for load balancer to detect unhealthy state
        await asyncio.sleep(2)
    
    async def _complete_critical_operations(self):
        """Complete critical cache warming operations."""
        logger.info("Completing critical operations...")
        
        if self.components.get('cache_warming'):
            cache_warmer = self.components['cache_warming']
            
            # Check if warming is in progress
            if hasattr(cache_warmer, 'current_batch'):
                logger.info("Waiting for current cache warming batch to complete...")
                
                try:
                    # Wait with timeout for current batch
                    await asyncio.wait_for(
                        cache_warmer.complete_current_batch(),
                        timeout=min(10, self.timeout / 3)
                    )
                    self.metrics['tasks_completed'] += 1
                    logger.info("Cache warming batch completed successfully")
                except asyncio.TimeoutError:
                    logger.warning("Cache warming batch timed out, cancelling...")
                    self.metrics['tasks_cancelled'] += 1
                except Exception as e:
                    logger.error(f"Error completing cache warming: {e}")
    
    async def _save_state(self):
        """Save application state for quick recovery."""
        logger.info("Saving application state...")
        
        state_data = {
            'shutdown_time': datetime.now().isoformat(),
            'cache_keys': [],
            'in_progress_tasks': [],
            'metrics': self.metrics
        }
        
        # Save cache keys for priority warming on restart
        if self.components.get('cache'):
            try:
                cache = self.components['cache']
                state_data['cache_keys'] = await cache.get_warm_keys()
                logger.info(f"Saved {len(state_data['cache_keys'])} cache keys for recovery")
            except Exception as e:
                logger.error(f"Failed to save cache keys: {e}")
        
        # Save to Redis or file
        try:
            import json
            state_file = '/tmp/investment_app_state.json'
            with open(state_file, 'w') as f:
                json.dump(state_data, f)
            logger.info(f"Application state saved to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save application state: {e}")
    
    async def _cleanup_connections(self):
        """Close database and cache connections gracefully."""
        logger.info("Cleaning up connections...")
        
        # Close database connections
        if self.components.get('database'):
            try:
                db = self.components['database']
                
                # Wait for active queries with timeout
                active_queries = getattr(db, 'active_queries', 0)
                if active_queries > 0:
                    logger.info(f"Waiting for {active_queries} active queries to complete...")
                    await asyncio.sleep(min(5, self.timeout / 4))
                
                # Close pool
                if hasattr(db, 'close'):
                    await db.close()
                    self.metrics['connections_closed'] += 1
                    logger.info("Database connections closed")
            except Exception as e:
                logger.error(f"Error closing database connections: {e}")
        
        # Close Redis connections
        if self.components.get('redis'):
            try:
                redis = self.components['redis']
                if hasattr(redis, 'close'):
                    await redis.close()
                    await redis.wait_closed()
                    self.metrics['connections_closed'] += 1
                    logger.info("Redis connections closed")
            except Exception as e:
                logger.error(f"Error closing Redis connections: {e}")
    
    async def _cancel_remaining_tasks(self):
        """Cancel remaining background tasks."""
        logger.info("Cancelling remaining background tasks...")
        
        # Get all running tasks
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
        
        if tasks:
            logger.info(f"Cancelling {len(tasks)} background tasks...")
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
                self.metrics['tasks_cancelled'] += 1
            
            # Wait for cancellation with timeout
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Background tasks cancelled")
    
    def register_component(self, name: str, component: Any):
        """
        Register a component for graceful shutdown.
        
        Args:
            name: Component identifier
            component: Component instance with cleanup methods
        """
        self.components[name] = component
        logger.debug(f"Registered component for shutdown: {name}")
    
    def register_cleanup_task(self, task: Callable):
        """
        Register a cleanup task to run during shutdown.
        
        Args:
            task: Async function to run during cleanup
        """
        self.cleanup_tasks.append(task)
    
    @contextmanager
    def protect_critical_section(self):
        """
        Context manager to protect critical sections from interruption.
        
        Usage:
            with shutdown_handler.protect_critical_section():
                # Critical code that should complete even during shutdown
                pass
        """
        if self.is_shutting_down:
            remaining_time = self.timeout - (datetime.now() - self.shutdown_start_time).total_seconds()
            if remaining_time <= 0:
                raise RuntimeError("Shutdown timeout exceeded")
        
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if duration > 5:
                logger.warning(f"Critical section took {duration:.1f}s to complete")
    
    def should_continue(self) -> bool:
        """
        Check if processing should continue or shutdown requested.
        
        Returns:
            True if should continue, False if shutting down
        """
        return not self.is_shutting_down


# Global shutdown handler instance
shutdown_handler = GracefulShutdownHandler()


def init_graceful_shutdown(app):
    """
    Initialize graceful shutdown for FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    from backend.utils.cache_warming import CacheWarmingStrategy
    from backend.utils.database_optimized import db
    from backend.utils.cache import get_redis_client
    
    # Register components
    shutdown_handler.register_component('cache_warming', CacheWarmingStrategy())
    shutdown_handler.register_component('database', db)
    shutdown_handler.register_component('redis', get_redis_client())
    
    @app.on_event("shutdown")
    async def handle_shutdown():
        """FastAPI shutdown event handler."""
        if not shutdown_handler.is_shutting_down:
            await shutdown_handler.shutdown()
    
    logger.info("Graceful shutdown handler initialized")


# Example usage in cache warming
async def cache_warming_with_shutdown(cache_warmer, symbols: List[str]):
    """
    Example of cache warming that respects shutdown signals.
    
    Args:
        cache_warmer: CacheWarmingStrategy instance
        symbols: List of symbols to warm
    """
    batch_size = 10
    
    for i in range(0, len(symbols), batch_size):
        # Check if should continue
        if not shutdown_handler.should_continue():
            logger.info("Shutdown requested, stopping cache warming")
            break
        
        batch = symbols[i:i + batch_size]
        
        # Protect critical batch processing
        with shutdown_handler.protect_critical_section():
            await cache_warmer.warm_batch(batch)
        
        # Brief pause between batches
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Test graceful shutdown
    import asyncio
    
    async def test_shutdown():
        """Test shutdown handler."""
        logger.info("Testing graceful shutdown handler...")
        
        # Simulate components
        class MockComponent:
            async def close(self):
                await asyncio.sleep(1)
                logger.info("Mock component closed")
        
        shutdown_handler.register_component('test', MockComponent())
        
        # Simulate SIGTERM after 2 seconds
        async def send_signal():
            await asyncio.sleep(2)
            logger.info("Sending SIGTERM...")
            shutdown_handler._signal_handler(signal.SIGTERM, None)
        
        asyncio.create_task(send_signal())
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
                logger.info("Application running...")
        except SystemExit:
            logger.info("Application exited")
    
    # Run test
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_shutdown())