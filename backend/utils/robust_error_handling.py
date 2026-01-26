"""
Robust Error Handling and Logging System
Addresses all error handling issues identified in the debugging analysis
"""

import logging
import sys
import traceback
import functools
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import uuid
from contextlib import asynccontextmanager
import time

# Custom exception classes for better error categorization
class DatabaseSchemaError(Exception):
    """Raised when database schema issues are detected"""
    pass

class DataQualityError(Exception):
    """Raised when data quality validation fails"""
    pass

class APIRateLimitError(Exception):
    """Raised when API rate limits are exceeded"""
    pass

class AsyncOperationError(Exception):
    """Raised when async operations fail"""
    pass

class DataSourceError(Exception):
    """Raised when external data sources fail"""
    pass

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Structured error context information"""
    error_id: str
    timestamp: datetime
    function_name: str
    module_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context_data: Dict[str, Any]
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None

class RobustLogger:
    """
    Enhanced logging system with structured error tracking
    """
    
    def __init__(self, name: str = "investment_platform"):
        self.logger = logging.getLogger(name)
        self.error_history: List[ErrorContext] = []
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging with multiple handlers and formats"""
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Set base level
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            file_handler = logging.FileHandler('logs/investment_platform.log')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        except Exception:
            # If log directory doesn't exist, just use console
            pass
        
        # Error file handler for errors only
        try:
            error_handler = logging.FileHandler('logs/errors.log')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_format)
            self.logger.addHandler(error_handler)
        except Exception:
            pass
    
    def log_error(self, 
                  error: Exception, 
                  context_data: Dict[str, Any] = None,
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  user_id: str = None,
                  request_id: str = None) -> str:
        """
        Log error with full context and return error ID for tracking
        """
        
        error_id = str(uuid.uuid4())[:8]
        context_data = context_data or {}
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            function_name=sys._getframe(1).f_code.co_name,
            module_name=sys._getframe(1).f_globals.get('__name__', 'unknown'),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            context_data=context_data,
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id
        )
        
        # Store in error history
        self.error_history.append(error_context)
        
        # Log with appropriate level based on severity
        log_level = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        # Create structured log message
        log_message = f"ERROR[{error_id}] {error_context.error_type}: {error_context.error_message}"
        if context_data:
            log_message += f" | Context: {json.dumps(context_data, default=str)}"
        
        self.logger.log(log_level, log_message)
        
        # For critical errors, also log stack trace
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"ERROR[{error_id}] Stack trace:\n{error_context.stack_trace}")
        
        return error_id
    
    def log_database_error(self, error: Exception, query: str = None, params: Dict = None) -> str:
        """Specialized logging for database errors"""
        context_data = {}
        if query:
            context_data['query'] = query
        if params:
            context_data['params'] = params
        
        # Determine severity based on error type
        severity = ErrorSeverity.HIGH
        if "column" in str(error).lower() and "does not exist" in str(error).lower():
            severity = ErrorSeverity.CRITICAL
        
        return self.log_error(
            DatabaseSchemaError(f"Database error: {error}"),
            context_data=context_data,
            severity=severity
        )
    
    def log_async_error(self, error: Exception, operation: str = None) -> str:
        """Specialized logging for async operation errors"""
        context_data = {'operation': operation} if operation else {}
        
        # Check for specific async issues
        severity = ErrorSeverity.MEDIUM
        if "_asyncio.Future" in str(error) or "_condition" in str(error):
            severity = ErrorSeverity.HIGH
            context_data['async_issue'] = "Future object attribute error"
        
        return self.log_error(
            AsyncOperationError(f"Async operation failed: {error}"),
            context_data=context_data,
            severity=severity
        )
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        # Group by error type
        error_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'error_types': error_counts,
            'severity_distribution': severity_counts,
            'time_period_hours': hours,
            'most_recent_error': recent_errors[-1].error_message if recent_errors else None
        }

# Global logger instance
robust_logger = RobustLogger()

def handle_errors(
    exceptions: tuple = (Exception,),
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    return_value: Any = None,
    log_context: bool = True
):
    """
    Decorator for robust error handling with logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                context_data = {}
                if log_context:
                    context_data = {
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                
                error_id = robust_logger.log_error(e, context_data, severity)
                
                # For critical errors, re-raise
                if severity == ErrorSeverity.CRITICAL:
                    raise
                
                return return_value
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                context_data = {}
                if log_context:
                    context_data = {
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'is_async': True
                    }
                
                error_id = robust_logger.log_async_error(e, func.__name__)
                
                # For critical errors, re-raise
                if severity == ErrorSeverity.CRITICAL:
                    raise
                
                return return_value
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator

def safe_database_operation(func: Callable) -> Callable:
    """
    Decorator specifically for database operations with enhanced error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Log database error
                context_data = {
                    'attempt': attempt + 1,
                    'max_retries': max_retries,
                    'function': func.__name__
                }
                
                error_id = robust_logger.log_database_error(e, params=context_data)
                
                # Check for specific database issues
                if "column" in error_msg and "does not exist" in error_msg:
                    robust_logger.logger.critical(
                        f"ERROR[{error_id}] Database schema issue detected - immediate attention required"
                    )
                    raise DatabaseSchemaError(f"Schema error: {e}")
                
                # Check if retryable error
                retryable_errors = [
                    "connection",
                    "timeout", 
                    "temporary",
                    "lock"
                ]
                
                is_retryable = any(err in error_msg for err in retryable_errors)
                
                if attempt < max_retries - 1 and is_retryable:
                    robust_logger.logger.warning(
                        f"ERROR[{error_id}] Retrying database operation in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Final attempt failed or non-retryable error
                    raise
        
    return wrapper

@asynccontextmanager
async def safe_async_operation(operation_name: str):
    """
    Context manager for safe async operations with proper cleanup
    """
    start_time = time.time()
    robust_logger.logger.info(f"Starting async operation: {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        robust_logger.logger.info(f"Async operation '{operation_name}' completed in {duration:.2f}s")
        
    except asyncio.CancelledError:
        robust_logger.logger.warning(f"Async operation '{operation_name}' was cancelled")
        raise
        
    except Exception as e:
        duration = time.time() - start_time
        error_id = robust_logger.log_async_error(e, operation_name)
        robust_logger.logger.error(
            f"ERROR[{error_id}] Async operation '{operation_name}' failed after {duration:.2f}s"
        )
        raise
        
    finally:
        # Ensure proper cleanup
        try:
            # Force garbage collection for async objects
            import gc
            gc.collect()
        except Exception:
            pass

class BatchProcessingErrorHandler:
    """
    Specialized error handler for batch processing operations
    Addresses the specific asyncio Future object errors
    """
    
    def __init__(self, batch_name: str):
        self.batch_name = batch_name
        self.processed_count = 0
        self.error_count = 0
        self.errors: List[Dict[str, Any]] = []
    
    def record_success(self, item_id: str = None):
        """Record successful processing"""
        self.processed_count += 1
        
    def record_error(self, error: Exception, item_id: str = None, context: Dict = None):
        """Record processing error"""
        self.error_count += 1
        
        error_record = {
            'item_id': item_id or f"item_{self.error_count}",
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': context or {}
        }
        
        self.errors.append(error_record)
        
        # Log with robust logger
        robust_logger.log_error(
            error,
            context_data={
                'batch_name': self.batch_name,
                'item_id': item_id,
                'batch_progress': f"{self.processed_count + self.error_count} items processed",
                **(context or {})
            },
            severity=ErrorSeverity.MEDIUM
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch processing summary"""
        total_items = self.processed_count + self.error_count
        success_rate = (self.processed_count / total_items * 100) if total_items > 0 else 0
        
        return {
            'batch_name': self.batch_name,
            'total_processed': total_items,
            'successful': self.processed_count,
            'failed': self.error_count,
            'success_rate': round(success_rate, 2),
            'error_summary': [
                {
                    'error_type': error['error_type'],
                    'count': len([e for e in self.errors if e['error_type'] == error['error_type']])
                }
                for error in self.errors
            ] if self.errors else []
        }

# Example usage and testing
if __name__ == "__main__":
    # Test error handling
    
    @handle_errors(exceptions=(ValueError, TypeError), severity=ErrorSeverity.MEDIUM, return_value="default")
    def test_function_with_error():
        raise ValueError("Test error for demonstration")
    
    @safe_database_operation
    def test_database_operation():
        # Simulate database schema error
        raise Exception("column 'code' does not exist")
    
    # Test the decorators
    result = test_function_with_error()
    print(f"Function returned: {result}")
    
    try:
        test_database_operation()
    except DatabaseSchemaError as e:
        print(f"Caught database schema error: {e}")
    
    # Print error summary
    summary = robust_logger.get_error_summary(hours=1)
    print(f"Error summary: {json.dumps(summary, indent=2)}")