"""
Enhanced Bulk Operations with Granular Error Handling
Provides robust error tracking and recovery for bulk database operations.
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, DataError
from sqlalchemy.orm import Session
import pandas as pd

from backend.models.tables import Stock, PriceHistory, TechnicalIndicators, CostMetrics
from backend.models.database import get_db_session

logger = logging.getLogger(__name__)


class BulkOperationStatus(Enum):
    """Status of individual bulk operation records."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


@dataclass
class BulkOperationResult:
    """Result of a bulk operation with detailed tracking."""
    total_records: int
    successful: int
    failed: int
    skipped: int
    errors: List[Dict[str, Any]]
    warnings: List[str]
    execution_time_ms: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.successful / self.total_records) * 100


class EnhancedBulkOperations:
    """Enhanced bulk operations with granular error handling and recovery."""
    
    def __init__(self, batch_size: int = 1000, max_retries: int = 3):
        """
        Initialize bulk operations handler.
        
        Args:
            batch_size: Maximum records per batch
            max_retries: Maximum retry attempts for failed records
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
    
    def store_price_data_with_tracking(
        self, 
        data: List[Dict],
        validate: bool = True
    ) -> BulkOperationResult:
        """
        Store price data with detailed error tracking and recovery.
        
        Args:
            data: List of price data dictionaries
            validate: Whether to validate data before insertion
        
        Returns:
            BulkOperationResult with detailed metrics
        """
        if not data:
            return BulkOperationResult(0, 0, 0, 0, [], [], 0.0)
        
        start_time = datetime.now()
        result = BulkOperationResult(
            total_records=len(data),
            successful=0,
            failed=0,
            skipped=0,
            errors=[],
            warnings=[],
            execution_time_ms=0.0
        )
        
        with get_db_session() as session:
            try:
                # Get stock mappings
                symbols = list(set(item.get('symbol') for item in data if item.get('symbol')))
                stock_map = self._get_stock_mappings(session, symbols)
                
                if len(stock_map) < len(symbols):
                    missing = set(symbols) - set(stock_map.keys())
                    result.warnings.append(f"Missing stocks in database: {missing}")
                
                # Process in batches for better memory management
                for batch_start in range(0, len(data), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(data))
                    batch_data = data[batch_start:batch_end]
                    
                    batch_result = self._process_price_batch(
                        session, 
                        batch_data, 
                        stock_map,
                        validate
                    )
                    
                    result.successful += batch_result['successful']
                    result.failed += batch_result['failed']
                    result.skipped += batch_result['skipped']
                    result.errors.extend(batch_result['errors'])
                    
                    # Commit after each successful batch
                    if batch_result['successful'] > 0:
                        session.commit()
                        logger.info(
                            f"Batch {batch_start//self.batch_size + 1}: "
                            f"Processed {batch_result['successful']} records"
                        )
                
            except Exception as e:
                logger.error(f"Fatal error in bulk operation: {e}")
                session.rollback()
                result.errors.append({
                    'type': 'fatal',
                    'error': str(e),
                    'traceback': str(e.__traceback__)
                })
            
            finally:
                result.execution_time_ms = (
                    datetime.now() - start_time
                ).total_seconds() * 1000
        
        # Log summary
        logger.info(
            f"Bulk operation completed: "
            f"Success: {result.successful}/{result.total_records} "
            f"({result.success_rate:.1f}%), "
            f"Failed: {result.failed}, "
            f"Skipped: {result.skipped}, "
            f"Time: {result.execution_time_ms:.2f}ms"
        )
        
        return result
    
    def _get_stock_mappings(
        self, 
        session: Session, 
        symbols: List[str]
    ) -> Dict[str, int]:
        """Get stock ID mappings for symbols."""
        if not symbols:
            return {}
        
        stocks = session.query(Stock.symbol, Stock.id).filter(
            Stock.symbol.in_(symbols)
        ).all()
        
        return {s.symbol: s.id for s in stocks}
    
    def _process_price_batch(
        self,
        session: Session,
        batch_data: List[Dict],
        stock_map: Dict[str, int],
        validate: bool
    ) -> Dict:
        """Process a single batch of price data."""
        batch_result = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        # Prepare and validate records
        valid_records = []
        current_date = datetime.now().date()
        
        for idx, item in enumerate(batch_data):
            try:
                # Validate and prepare record
                record = self._prepare_price_record(item, stock_map, current_date)
                
                if record is None:
                    batch_result['skipped'] += 1
                    continue
                
                if validate:
                    validation_errors = self._validate_price_record(record)
                    if validation_errors:
                        batch_result['errors'].append({
                            'symbol': item.get('symbol'),
                            'errors': validation_errors,
                            'data': item
                        })
                        batch_result['failed'] += 1
                        continue
                
                valid_records.append(record)
                
            except Exception as e:
                batch_result['errors'].append({
                    'symbol': item.get('symbol'),
                    'error': str(e),
                    'index': idx
                })
                batch_result['failed'] += 1
        
        # Bulk insert valid records
        if valid_records:
            try:
                self._execute_bulk_upsert(session, valid_records)
                batch_result['successful'] = len(valid_records)
                
            except IntegrityError as e:
                # Handle constraint violations
                logger.warning(f"Integrity error, attempting record-by-record insertion: {e}")
                batch_result.update(
                    self._fallback_individual_insert(session, valid_records)
                )
                
            except DataError as e:
                # Handle data type errors
                logger.error(f"Data error in bulk insert: {e}")
                batch_result['errors'].append({
                    'type': 'data_error',
                    'error': str(e),
                    'record_count': len(valid_records)
                })
                batch_result['failed'] += len(valid_records)
                
            except SQLAlchemyError as e:
                logger.error(f"Database error in bulk insert: {e}")
                batch_result['errors'].append({
                    'type': 'database_error',
                    'error': str(e)
                })
                batch_result['failed'] += len(valid_records)
                session.rollback()
        
        return batch_result
    
    def _prepare_price_record(
        self,
        item: Dict,
        stock_map: Dict[str, int],
        current_date: datetime.date
    ) -> Optional[Dict]:
        """Prepare a single price record for insertion."""
        symbol = item.get('symbol')
        
        if not symbol or symbol not in stock_map:
            return None
        
        try:
            return {
                'stock_id': stock_map[symbol],
                'date': current_date,
                'open': float(item.get('open', 0)),
                'high': float(item.get('high', 0)),
                'low': float(item.get('low', 0)),
                'close': float(item.get('close', 0)),
                'volume': int(item.get('volume', 0)),
                'adjusted_close': float(
                    item.get('adjusted_close', item.get('close', 0))
                )
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid data for {symbol}: {e}")
            return None
    
    def _validate_price_record(self, record: Dict) -> List[str]:
        """Validate a price record for data quality."""
        errors = []
        
        # Price consistency checks
        if record['high'] < record['low']:
            errors.append("High price less than low price")
        
        if record['close'] > record['high'] or record['close'] < record['low']:
            errors.append("Close price outside high-low range")
        
        if record['open'] > record['high'] or record['open'] < record['low']:
            errors.append("Open price outside high-low range")
        
        # Volume check
        if record['volume'] < 0:
            errors.append("Negative volume")
        
        # Price sanity checks
        if record['close'] <= 0:
            errors.append("Invalid close price (zero or negative)")
        
        # Extreme value checks
        if record['high'] / record['low'] > 10:  # 1000% intraday range
            errors.append("Suspicious price range (>1000% intraday)")
        
        return errors
    
    def _execute_bulk_upsert(self, session: Session, records: List[Dict]):
        """Execute bulk upsert operation."""
        stmt = insert(PriceHistory).values(records)
        
        stmt = stmt.on_conflict_do_update(
            index_elements=['stock_id', 'date'],
            set_={
                'open': stmt.excluded.open,
                'high': stmt.excluded.high,
                'low': stmt.excluded.low,
                'close': stmt.excluded.close,
                'volume': stmt.excluded.volume,
                'adjusted_close': stmt.excluded.adjusted_close,
                'updated_at': datetime.utcnow()
            }
        )
        
        session.execute(stmt)
    
    def _fallback_individual_insert(
        self,
        session: Session,
        records: List[Dict]
    ) -> Dict:
        """Fallback to individual record insertion for error isolation."""
        result = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        for record in records:
            try:
                stmt = insert(PriceHistory).values(record)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['stock_id', 'date'],
                    set_={
                        'close': stmt.excluded.close,
                        'volume': stmt.excluded.volume,
                        'updated_at': datetime.utcnow()
                    }
                )
                session.execute(stmt)
                session.commit()
                result['successful'] += 1
                
            except Exception as e:
                session.rollback()
                result['failed'] += 1
                result['errors'].append({
                    'stock_id': record.get('stock_id'),
                    'error': str(e)
                })
                
                # Log specific error for debugging
                logger.debug(f"Failed to insert record: {record}, Error: {e}")
        
        return result
    
    def store_technical_indicators_bulk(
        self,
        indicators_data: List[Dict]
    ) -> BulkOperationResult:
        """Store technical indicators with error handling."""
        if not indicators_data:
            return BulkOperationResult(0, 0, 0, 0, [], [], 0.0)
        
        start_time = datetime.now()
        result = BulkOperationResult(
            total_records=len(indicators_data),
            successful=0,
            failed=0,
            skipped=0,
            errors=[],
            warnings=[],
            execution_time_ms=0.0
        )
        
        with get_db_session() as session:
            try:
                # Process in batches
                for batch_start in range(0, len(indicators_data), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(indicators_data))
                    batch = indicators_data[batch_start:batch_end]
                    
                    try:
                        stmt = insert(TechnicalIndicators).values(batch)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['stock_id', 'date'],
                            set_={
                                'rsi': stmt.excluded.rsi,
                                'macd': stmt.excluded.macd,
                                'macd_signal': stmt.excluded.macd_signal,
                                'updated_at': datetime.utcnow()
                            }
                        )
                        
                        session.execute(stmt)
                        session.commit()
                        result.successful += len(batch)
                        
                    except SQLAlchemyError as e:
                        logger.error(f"Error inserting technical indicators batch: {e}")
                        session.rollback()
                        result.failed += len(batch)
                        result.errors.append({
                            'batch': f"{batch_start}-{batch_end}",
                            'error': str(e)
                        })
                
            except Exception as e:
                logger.error(f"Fatal error storing technical indicators: {e}")
                result.errors.append({
                    'type': 'fatal',
                    'error': str(e)
                })
            
            finally:
                result.execution_time_ms = (
                    datetime.now() - start_time
                ).total_seconds() * 1000
        
        return result
    
    def retry_failed_operations(
        self,
        failed_records: List[Dict],
        operation_type: str = 'price'
    ) -> BulkOperationResult:
        """Retry failed bulk operations with exponential backoff."""
        import time
        
        if operation_type == 'price':
            operation_func = self.store_price_data_with_tracking
        elif operation_type == 'technical':
            operation_func = self.store_technical_indicators_bulk
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        retry_delay = 1.0  # Start with 1 second
        
        for attempt in range(self.max_retries):
            logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")
            
            result = operation_func(failed_records, validate=False)
            
            if result.success_rate > 90:  # 90% success threshold
                logger.info(f"Retry successful: {result.success_rate:.1f}% success rate")
                return result
            
            # Exponential backoff
            time.sleep(retry_delay)
            retry_delay *= 2
        
        logger.warning(f"Max retries reached. Final success rate: {result.success_rate:.1f}%")
        return result


# Global instance for use in DAGs
bulk_operations = EnhancedBulkOperations()