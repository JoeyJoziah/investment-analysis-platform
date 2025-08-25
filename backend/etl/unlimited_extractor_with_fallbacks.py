"""
Complete Unlimited Stock Data Extractor with Intelligent Fallback System
Integrates all components for 6000+ stock extraction without rate limits
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import random

# Import our custom modules
from .unlimited_data_extractor import (
    UnlimitedDataExtractor, 
    StockData, 
    ExtractionResult,
    YahooFinanceWebScraper,
    BulkDataDownloader,
    SECEdgarExtractor,
    IEXCloudFreeExtractor
)
from .intelligent_cache_system import IntelligentCacheManager
from .concurrent_processor import ConcurrentProcessor, ProcessingTask, ProcessingResult
from .data_validation_pipeline import (
    FinancialDataValidator, 
    ValidationLevel, 
    DataQualityScore,
    validate_extraction_results
)

logger = logging.getLogger(__name__)

@dataclass
class SourceHealth:
    """Health status of a data source"""
    source_name: str
    is_available: bool = True
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 0.0
    
    @property
    def is_healthy(self) -> bool:
        return (self.is_available and 
                self.consecutive_failures < 5 and 
                self.error_rate < 0.5)

@dataclass
class FallbackStrategy:
    """Defines a fallback strategy for data extraction"""
    name: str
    priority: int  # Lower = higher priority
    sources: List[str]  # Ordered list of sources to try
    max_attempts_per_source: int = 3
    timeout_seconds: int = 30
    min_success_rate: float = 0.3
    enabled: bool = True
    description: str = ""

class HealthMonitor:
    """Monitors health of data sources and manages availability"""
    
    def __init__(self, check_interval_minutes: int = 5):
        self.source_health = {}
        self.check_interval = check_interval_minutes * 60
        self.monitoring = False
        self.monitor_thread = None
        self.health_callbacks = []
        
        # Initialize known sources
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize health tracking for known sources"""
        sources = [
            'yahoo_scraper',
            'yahoo_bulk',
            'sec_edgar',
            'iex_free',
            'yahoo_selenium',
            'marketwatch_scraper'
        ]
        
        for source in sources:
            self.source_health[source] = SourceHealth(source_name=source)
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Health monitoring started (check interval: {self.check_interval/60:.1f}min)")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _perform_health_checks(self):
        """Perform health checks on all sources"""
        logger.debug("Performing health checks...")
        
        for source_name, health in self.source_health.items():
            try:
                # Simple availability check based on recent activity
                if health.last_success and health.last_failure:
                    time_since_success = (datetime.now() - health.last_success).total_seconds()
                    time_since_failure = (datetime.now() - health.last_failure).total_seconds()
                    
                    # Mark as unavailable if recent failures and no recent successes
                    if time_since_failure < 300 and time_since_success > 300:  # 5 minutes
                        if health.is_available:
                            health.is_available = False
                            logger.warning(f"Marking {source_name} as unavailable due to recent failures")
                    
                    # Mark as available if recent successes
                    elif time_since_success < 300:
                        if not health.is_available:
                            health.is_available = True
                            logger.info(f"Marking {source_name} as available due to recent success")
                
                health.last_health_check = datetime.now()
                
                # Update error rate
                total_attempts = health.success_count + health.failure_count
                if total_attempts > 0:
                    health.error_rate = health.failure_count / total_attempts
                
            except Exception as e:
                logger.error(f"Health check error for {source_name}: {e}")
        
        # Notify callbacks of health changes
        for callback in self.health_callbacks:
            try:
                callback(self.source_health)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    def record_success(self, source_name: str, response_time_ms: float = 0):
        """Record successful operation"""
        if source_name not in self.source_health:
            self.source_health[source_name] = SourceHealth(source_name=source_name)
        
        health = self.source_health[source_name]
        health.last_success = datetime.now()
        health.success_count += 1
        health.consecutive_failures = 0
        
        if response_time_ms > 0:
            # Update average response time
            if health.avg_response_time_ms == 0:
                health.avg_response_time_ms = response_time_ms
            else:
                health.avg_response_time_ms = (health.avg_response_time_ms + response_time_ms) / 2
        
        # Mark as available if it was down
        if not health.is_available:
            health.is_available = True
            logger.info(f"Source {source_name} recovered and marked as available")
    
    def record_failure(self, source_name: str, error: str = ""):
        """Record failed operation"""
        if source_name not in self.source_health:
            self.source_health[source_name] = SourceHealth(source_name=source_name)
        
        health = self.source_health[source_name]
        health.last_failure = datetime.now()
        health.failure_count += 1
        health.consecutive_failures += 1
        
        # Mark as unavailable after consecutive failures
        if health.consecutive_failures >= 5:
            health.is_available = False
            logger.warning(f"Source {source_name} marked as unavailable after {health.consecutive_failures} consecutive failures")
    
    def get_healthy_sources(self) -> List[str]:
        """Get list of currently healthy sources"""
        return [name for name, health in self.source_health.items() if health.is_healthy]
    
    def get_source_health(self, source_name: str) -> Optional[SourceHealth]:
        """Get health status for specific source"""
        return self.source_health.get(source_name)
    
    def get_health_summary(self) -> Dict:
        """Get comprehensive health summary"""
        healthy_sources = self.get_healthy_sources()
        total_sources = len(self.source_health)
        
        return {
            'total_sources': total_sources,
            'healthy_sources': len(healthy_sources),
            'unhealthy_sources': total_sources - len(healthy_sources),
            'health_rate': len(healthy_sources) / total_sources if total_sources > 0 else 0,
            'healthy_source_list': healthy_sources,
            'source_details': {
                name: {
                    'is_healthy': health.is_healthy,
                    'success_rate': health.success_rate,
                    'consecutive_failures': health.consecutive_failures,
                    'avg_response_time_ms': health.avg_response_time_ms,
                    'last_success': health.last_success.isoformat() if health.last_success else None
                }
                for name, health in self.source_health.items()
            }
        }
    
    def add_health_callback(self, callback: Callable[[Dict[str, SourceHealth]], None]):
        """Add callback to be notified of health changes"""
        self.health_callbacks.append(callback)

class FallbackManager:
    """Manages fallback strategies for data extraction"""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.strategies = self._initialize_strategies()
        self.strategy_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        
        logger.info(f"Initialized FallbackManager with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> List[FallbackStrategy]:
        """Initialize fallback strategies"""
        return [
            FallbackStrategy(
                name="primary_scraping",
                priority=1,
                sources=["yahoo_scraper", "yahoo_bulk"],
                max_attempts_per_source=2,
                timeout_seconds=15,
                description="Primary web scraping approach"
            ),
            FallbackStrategy(
                name="selenium_fallback",
                priority=2,
                sources=["yahoo_selenium", "marketwatch_scraper"],
                max_attempts_per_source=3,
                timeout_seconds=30,
                description="Selenium-based scraping for dynamic content"
            ),
            FallbackStrategy(
                name="official_data",
                priority=3,
                sources=["sec_edgar", "iex_free"],
                max_attempts_per_source=2,
                timeout_seconds=20,
                description="Official regulatory and free market data"
            ),
            FallbackStrategy(
                name="comprehensive_retry",
                priority=4,
                sources=["yahoo_scraper", "sec_edgar", "yahoo_selenium", "iex_free"],
                max_attempts_per_source=1,
                timeout_seconds=45,
                description="Final attempt using all available sources"
            )
        ]
    
    async def extract_with_fallback(self, 
                                   ticker: str, 
                                   extractor_function: Callable,
                                   max_strategies: int = None) -> ExtractionResult:
        """Extract data using fallback strategies"""
        
        max_strategies = max_strategies or len(self.strategies)
        last_error = None
        
        # Sort strategies by priority and filter by health
        available_strategies = []
        for strategy in sorted(self.strategies, key=lambda s: s.priority):
            if not strategy.enabled:
                continue
            
            # Check if strategy has healthy sources
            healthy_sources = [s for s in strategy.sources 
                             if self.health_monitor.get_source_health(s) and 
                                self.health_monitor.get_source_health(s).is_healthy]
            
            if healthy_sources:
                available_strategies.append(strategy)
            else:
                logger.debug(f"Skipping strategy {strategy.name} - no healthy sources")
        
        # Try strategies in order
        for i, strategy in enumerate(available_strategies[:max_strategies]):
            if i > 0:
                logger.info(f"Trying fallback strategy {strategy.name} for {ticker}")
            
            try:
                result = await self._try_strategy(ticker, strategy, extractor_function)
                
                if result.success:
                    self.strategy_stats[strategy.name]['successes'] += 1
                    logger.debug(f"Strategy {strategy.name} succeeded for {ticker}")
                    return result
                else:
                    last_error = result.error
                    logger.debug(f"Strategy {strategy.name} failed for {ticker}: {last_error}")
            
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Strategy {strategy.name} threw exception for {ticker}: {e}")
            
            finally:
                self.strategy_stats[strategy.name]['attempts'] += 1
        
        # All strategies failed
        return ExtractionResult(
            ticker=ticker,
            success=False,
            error=f"All fallback strategies failed. Last error: {last_error}",
            source="fallback_manager"
        )
    
    async def _try_strategy(self, 
                           ticker: str, 
                           strategy: FallbackStrategy, 
                           extractor_function: Callable) -> ExtractionResult:
        """Try a specific fallback strategy"""
        
        # Get healthy sources for this strategy
        healthy_sources = []
        for source_name in strategy.sources:
            health = self.health_monitor.get_source_health(source_name)
            if health and health.is_healthy:
                healthy_sources.append(source_name)
        
        if not healthy_sources:
            return ExtractionResult(
                ticker=ticker,
                success=False,
                error=f"No healthy sources available for strategy {strategy.name}",
                source=strategy.name
            )
        
        # Try each healthy source
        for source_name in healthy_sources:
            for attempt in range(strategy.max_attempts_per_source):
                start_time = time.time()
                
                try:
                    # Call extractor with specific source
                    result = await asyncio.wait_for(
                        extractor_function(ticker, source_name),
                        timeout=strategy.timeout_seconds
                    )
                    
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    if result and result.success:
                        self.health_monitor.record_success(source_name, response_time_ms)
                        result.source = f"{strategy.name}:{source_name}"
                        return result
                    else:
                        error_msg = result.error if result else "No result returned"
                        logger.debug(f"Source {source_name} failed for {ticker} (attempt {attempt+1}): {error_msg}")
                        
                        if attempt == strategy.max_attempts_per_source - 1:  # Last attempt
                            self.health_monitor.record_failure(source_name, error_msg)
                
                except asyncio.TimeoutError:
                    error_msg = f"Timeout after {strategy.timeout_seconds}s"
                    logger.debug(f"Source {source_name} timed out for {ticker} (attempt {attempt+1})")
                    
                    if attempt == strategy.max_attempts_per_source - 1:
                        self.health_monitor.record_failure(source_name, error_msg)
                
                except Exception as e:
                    error_msg = f"Exception: {str(e)}"
                    logger.debug(f"Source {source_name} exception for {ticker} (attempt {attempt+1}): {e}")
                    
                    if attempt == strategy.max_attempts_per_source - 1:
                        self.health_monitor.record_failure(source_name, error_msg)
                
                # Small delay between attempts
                if attempt < strategy.max_attempts_per_source - 1:
                    await asyncio.sleep(random.uniform(1, 3))
        
        return ExtractionResult(
            ticker=ticker,
            success=False,
            error=f"All sources failed for strategy {strategy.name}",
            source=strategy.name
        )
    
    def get_strategy_stats(self) -> Dict:
        """Get statistics for fallback strategies"""
        stats = {}
        for strategy_name, counts in self.strategy_stats.items():
            attempts = counts['attempts']
            successes = counts['successes']
            stats[strategy_name] = {
                'attempts': attempts,
                'successes': successes,
                'success_rate': successes / attempts if attempts > 0 else 0,
                'failures': attempts - successes
            }
        
        return stats

class UnlimitedStockDataExtractor:
    """Complete unlimited stock data extraction system with fallbacks"""
    
    def __init__(self, 
                 cache_dir: str = "/tmp/unlimited_stock_cache",
                 enable_validation: bool = True,
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 enable_caching: bool = True,
                 enable_health_monitoring: bool = True,
                 max_concurrent: int = 50):
        
        self.cache_dir = cache_dir
        self.enable_validation = enable_validation
        self.validation_level = validation_level
        self.enable_caching = enable_caching
        self.max_concurrent = max_concurrent
        
        # Initialize core components
        self.base_extractor = UnlimitedDataExtractor(cache_dir)
        
        # Initialize caching system
        if enable_caching:
            self.cache_manager = IntelligentCacheManager(
                cache_dir=cache_dir,
                memory_size_mb=256,
                disk_size_mb=2048,
                enable_analytics=True
            )
        else:
            self.cache_manager = None
        
        # Initialize validation system
        if enable_validation:
            self.validator = FinancialDataValidator(validation_level)
        else:
            self.validator = None
        
        # Initialize health monitoring and fallback system
        if enable_health_monitoring:
            self.health_monitor = HealthMonitor(check_interval_minutes=5)
            self.fallback_manager = FallbackManager(self.health_monitor)
            self.health_monitor.start_monitoring()
        else:
            self.health_monitor = None
            self.fallback_manager = None
        
        # Initialize concurrent processor
        self.processor = ConcurrentProcessor(
            max_concurrent_requests=max_concurrent,
            max_requests_per_second=20,
            enable_resource_monitoring=True
        )
        
        # Statistics
        self.extraction_stats = {
            'total_requests': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'fallback_uses': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Initialized UnlimitedStockDataExtractor with all components")
    
    async def extract_stock_data(self, ticker: str, use_cache: bool = True) -> ExtractionResult:
        """Extract stock data for a single ticker with full fallback support"""
        start_time = time.time()
        self.extraction_stats['total_requests'] += 1
        
        # Check cache first
        if use_cache and self.cache_manager:
            cached_data = await self.cache_manager.get(f"stock_data:{ticker}", "stocks")
            if cached_data:
                self.extraction_stats['cache_hits'] += 1
                
                # Convert cached data back to StockData object
                if isinstance(cached_data, dict):
                    stock_data = StockData(
                        ticker=ticker,
                        timestamp=datetime.fromisoformat(cached_data['timestamp']),
                        **{k: v for k, v in cached_data.items() if k not in ['ticker', 'timestamp']}
                    )
                else:
                    stock_data = cached_data
                
                return ExtractionResult(
                    ticker=ticker,
                    success=True,
                    data=stock_data,
                    source="cache",
                    cache_hit=True,
                    extraction_time_ms=int((time.time() - start_time) * 1000)
                )
        
        # Extract with fallback system
        if self.fallback_manager:
            result = await self.fallback_manager.extract_with_fallback(
                ticker=ticker,
                extractor_function=self._extract_with_source
            )
            
            if not result.success:
                self.extraction_stats['fallback_uses'] += 1
        else:
            # Direct extraction without fallbacks
            result = await self.base_extractor.extract_stock_data(ticker, use_cache=False)
        
        # Validate extracted data
        if result.success and result.data and self.validator:
            try:
                validation_score = await self.validator.validate_stock_data(
                    result.data.to_dict(), ticker
                )
                
                # Apply cleaning if validation found issues
                if validation_score.issues:
                    cleaned_data_dict = await self.validator.clean_and_correct_data(
                        result.data.to_dict(), validation_score
                    )
                    
                    # Create new StockData with cleaned data
                    result.data = StockData(
                        ticker=ticker,
                        timestamp=result.data.timestamp,
                        source=result.data.source,
                        data_quality_score=validation_score.overall_score,
                        **{k: v for k, v in cleaned_data_dict.items() 
                           if k not in ['ticker', 'timestamp', 'source', 'data_quality_score']}
                    )
                else:
                    result.data.data_quality_score = validation_score.overall_score
                
                # Reject low quality data
                if validation_score.overall_score < 30:
                    self.extraction_stats['validation_failures'] += 1
                    return ExtractionResult(
                        ticker=ticker,
                        success=False,
                        error=f"Data quality too low: {validation_score.overall_score}/100",
                        source=result.source
                    )
            
            except Exception as e:
                logger.warning(f"Validation failed for {ticker}: {e}")
        
        # Cache successful results
        if result.success and result.data and self.cache_manager:
            try:
                await self.cache_manager.set(
                    f"stock_data:{ticker}", 
                    result.data.to_dict(), 
                    "stocks",
                    ttl_hours=6
                )
            except Exception as e:
                logger.warning(f"Failed to cache data for {ticker}: {e}")
        
        # Update statistics
        if result.success:
            self.extraction_stats['successful_extractions'] += 1
        else:
            self.extraction_stats['failed_extractions'] += 1
        
        result.extraction_time_ms = int((time.time() - start_time) * 1000)
        return result
    
    async def _extract_with_source(self, ticker: str, source_name: str) -> ExtractionResult:
        """Extract data using a specific source (for fallback system)"""
        try:
            if source_name == "yahoo_scraper":
                scraper = YahooFinanceWebScraper()
                data = scraper.scrape_yahoo_summary(ticker)
                scraper.cleanup()
                
                if data and 'current_price' in data:
                    stock_data = StockData(
                        ticker=ticker,
                        timestamp=datetime.now(),
                        source=source_name,
                        **data
                    )
                    return ExtractionResult(ticker=ticker, success=True, data=stock_data, source=source_name)
            
            elif source_name == "yahoo_selenium":
                scraper = YahooFinanceWebScraper()
                data = scraper.scrape_with_selenium(ticker)
                scraper.cleanup()
                
                if data and 'current_price' in data:
                    stock_data = StockData(
                        ticker=ticker,
                        timestamp=datetime.now(),
                        source=source_name,
                        **data
                    )
                    return ExtractionResult(ticker=ticker, success=True, data=stock_data, source=source_name)
            
            elif source_name == "sec_edgar":
                extractor = SECEdgarExtractor()
                data = extractor.extract_company_facts(ticker)
                
                if data:
                    stock_data = StockData(
                        ticker=ticker,
                        timestamp=datetime.now(),
                        source=source_name,
                        **data
                    )
                    return ExtractionResult(ticker=ticker, success=True, data=stock_data, source=source_name)
            
            elif source_name == "iex_free":
                extractor = IEXCloudFreeExtractor()
                data = extractor.extract_company_info(ticker)
                
                if data:
                    stock_data = StockData(
                        ticker=ticker,
                        timestamp=datetime.now(),
                        source=source_name,
                        **data
                    )
                    return ExtractionResult(ticker=ticker, success=True, data=stock_data, source=source_name)
            
            elif source_name == "yahoo_bulk":
                downloader = BulkDataDownloader(self.cache_dir)
                bulk_data = downloader.download_yahoo_bulk_data([ticker])
                
                if ticker in bulk_data:
                    stock_data = StockData(
                        ticker=ticker,
                        timestamp=datetime.now(),
                        source=source_name,
                        **bulk_data[ticker]
                    )
                    return ExtractionResult(ticker=ticker, success=True, data=stock_data, source=source_name)
            
            # Source not implemented or failed
            return ExtractionResult(
                ticker=ticker,
                success=False,
                error=f"Source {source_name} not available or returned no data",
                source=source_name
            )
        
        except Exception as e:
            return ExtractionResult(
                ticker=ticker,
                success=False,
                error=f"Source {source_name} failed: {str(e)}",
                source=source_name
            )
    
    async def extract_bulk_data(self, 
                               tickers: List[str], 
                               progress_callback: Optional[Callable] = None,
                               batch_size: int = 100) -> List[ExtractionResult]:
        """Extract data for multiple tickers with full fallback support"""
        
        logger.info(f"Starting bulk extraction for {len(tickers)} tickers")
        self.processor.start()
        
        try:
            # Create extraction function for concurrent processor
            async def extraction_function(task: ProcessingTask) -> Dict:
                result = await self.extract_stock_data(task.ticker, use_cache=True)
                
                if result.success and result.data:
                    return result.data.to_dict()
                else:
                    raise Exception(result.error or "Extraction failed")
            
            # Process tickers in batches to manage memory
            all_results = []
            ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
            
            for batch_num, ticker_batch in enumerate(ticker_batches):
                logger.info(f"Processing batch {batch_num + 1}/{len(ticker_batches)} ({len(ticker_batch)} tickers)")
                
                # Process batch with concurrent processor
                batch_results = await self.processor.process_stock_extraction(
                    tickers=ticker_batch,
                    extraction_function=extraction_function,
                    max_concurrent=self.max_concurrent,
                    progress_callback=progress_callback
                )
                
                # Convert ProcessingResults to ExtractionResults
                for proc_result in batch_results:
                    if proc_result.success and proc_result.data:
                        stock_data = StockData(
                            ticker=proc_result.ticker,
                            timestamp=datetime.now(),
                            **proc_result.data
                        )
                        
                        extraction_result = ExtractionResult(
                            ticker=proc_result.ticker,
                            success=True,
                            data=stock_data,
                            source=proc_result.processor_id,
                            extraction_time_ms=proc_result.execution_time_ms
                        )
                    else:
                        extraction_result = ExtractionResult(
                            ticker=proc_result.ticker,
                            success=False,
                            error=proc_result.error,
                            source=proc_result.processor_id,
                            extraction_time_ms=proc_result.execution_time_ms
                        )
                    
                    all_results.append(extraction_result)
                
                # Brief pause between batches
                if batch_num < len(ticker_batches) - 1:
                    await asyncio.sleep(2)
            
            # Final statistics
            successful_results = [r for r in all_results if r.success]
            failed_results = [r for r in all_results if not r.success]
            
            logger.info(f"Bulk extraction completed:")
            logger.info(f"  Total: {len(all_results)}/{len(tickers)}")
            logger.info(f"  Successful: {len(successful_results)}")
            logger.info(f"  Failed: {len(failed_results)}")
            logger.info(f"  Success Rate: {len(successful_results)/len(all_results)*100:.1f}%" if all_results else "0%")
            
            return all_results
        
        finally:
            self.processor.stop()
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            'extraction': self.extraction_stats.copy(),
            'uptime_hours': (datetime.now() - self.extraction_stats['start_time']).total_seconds() / 3600
        }
        
        # Add success rate
        total_attempts = stats['extraction']['successful_extractions'] + stats['extraction']['failed_extractions']
        if total_attempts > 0:
            stats['extraction']['success_rate'] = stats['extraction']['successful_extractions'] / total_attempts
        else:
            stats['extraction']['success_rate'] = 0
        
        # Cache statistics
        if self.cache_manager:
            stats['cache'] = self.cache_manager.get_comprehensive_stats()
        
        # Health monitoring statistics
        if self.health_monitor:
            stats['health'] = self.health_monitor.get_health_summary()
        
        # Fallback statistics
        if self.fallback_manager:
            stats['fallback'] = self.fallback_manager.get_strategy_stats()
        
        # Processor statistics
        stats['processor'] = self.processor.get_processing_stats()
        
        # Validation statistics
        if self.validator:
            stats['validation'] = self.validator.get_validation_statistics()
        
        return stats
    
    def cleanup(self):
        """Cleanup all resources"""
        try:
            if self.health_monitor:
                self.health_monitor.stop_monitoring()
            
            if self.processor:
                self.processor.stop()
            
            if self.base_extractor:
                self.base_extractor.cleanup()
            
            logger.info("UnlimitedStockDataExtractor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Example usage and testing
async def test_unlimited_extractor_with_fallbacks():
    """Test the complete unlimited extraction system"""
    
    extractor = UnlimitedStockDataExtractor(
        cache_dir="/tmp/test_unlimited_cache",
        enable_validation=True,
        validation_level=ValidationLevel.STANDARD,
        enable_caching=True,
        enable_health_monitoring=True,
        max_concurrent=20
    )
    
    try:
        # Test single extraction
        logger.info("Testing single ticker extraction with fallbacks...")
        
        result = await extractor.extract_stock_data('AAPL')
        
        if result.success:
            logger.info(f"✓ Successfully extracted AAPL data:")
            logger.info(f"  Source: {result.data.source}")
            logger.info(f"  Quality Score: {result.data.data_quality_score}/100")
            logger.info(f"  Price: ${result.data.current_price}")
            logger.info(f"  Cache Hit: {result.cache_hit}")
            logger.info(f"  Time: {result.extraction_time_ms}ms")
        else:
            logger.error(f"✗ Failed to extract AAPL: {result.error}")
        
        # Test bulk extraction
        test_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
            'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'ADBE', 'CRM', 'NFLX'
        ]
        
        logger.info(f"Testing bulk extraction for {len(test_tickers)} tickers...")
        
        async def progress_callback(completed, total, recent):
            logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        results = await extractor.extract_bulk_data(
            tickers=test_tickers,
            progress_callback=progress_callback,
            batch_size=10
        )
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        logger.info("=== Bulk Extraction Results ===")
        logger.info(f"Total: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Success Rate: {len(successful)/len(results)*100:.1f}%")
        
        # Show sample successful results
        logger.info("Sample successful extractions:")
        for result in successful[:3]:
            logger.info(f"  ✓ {result.ticker}: ${result.data.current_price} (source: {result.data.source})")
        
        # Show failed results
        if failed:
            logger.info("Failed extractions:")
            for result in failed[:3]:
                logger.info(f"  ✗ {result.ticker}: {result.error}")
        
        # Show comprehensive statistics
        stats = extractor.get_comprehensive_stats()
        logger.info("=== System Statistics ===")
        logger.info(f"Total requests: {stats['extraction']['total_requests']}")
        logger.info(f"Success rate: {stats['extraction']['success_rate']:.1%}")
        logger.info(f"Cache hit rate: {stats['cache']['overview']['hit_rate']:.1%}" if 'cache' in stats else "Cache disabled")
        logger.info(f"Healthy sources: {stats['health']['healthy_sources']}/{stats['health']['total_sources']}" if 'health' in stats else "Health monitoring disabled")
        
        logger.info("Unlimited extractor with fallbacks test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    
    finally:
        extractor.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    asyncio.run(test_unlimited_extractor_with_fallbacks())