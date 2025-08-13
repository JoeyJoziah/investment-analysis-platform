"""
Comprehensive tests for Circuit Breaker implementation
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from backend.utils.circuit_breaker import (
    CircuitBreaker,
    APICircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    CircuitBreakerManager,
    get_api_circuit_breaker,
    with_circuit_breaker
)


class TestCircuitBreaker:
    """Test basic circuit breaker functionality"""
    
    def test_initial_state(self):
        """Test circuit breaker starts in closed state"""
        cb = CircuitBreaker(name="test_breaker")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
    
    def test_successful_calls(self):
        """Test circuit remains closed on successful calls"""
        cb = CircuitBreaker(name="test_breaker")
        mock_func = Mock(return_value="success")
        
        for _ in range(10):
            result = cb.call(mock_func)
            assert result == "success"
        
        assert cb.state == CircuitState.CLOSED
        assert mock_func.call_count == 10
    
    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures"""
        cb = CircuitBreaker(failure_threshold=3, name="test_breaker")
        mock_func = Mock(side_effect=Exception("test error"))
        
        # First 2 failures - circuit still closed
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(mock_func)
            assert cb.state == CircuitState.CLOSED
        
        # Third failure - circuit opens
        with pytest.raises(Exception):
            cb.call(mock_func)
        assert cb.state == CircuitState.OPEN
        
        # Further calls fail fast
        with pytest.raises(CircuitBreakerError):
            cb.call(mock_func)
        
        # Function not called when circuit is open
        assert mock_func.call_count == 3
    
    def test_circuit_recovery(self):
        """Test circuit moves to half-open and recovers"""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=2,
            name="test_breaker"
        )
        mock_func = Mock()
        
        # Open the circuit
        mock_func.side_effect = Exception("error")
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(mock_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Circuit should be half-open
        mock_func.side_effect = None
        mock_func.return_value = "success"
        
        # First successful call in half-open state
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second successful call closes circuit
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    def test_half_open_failure_reopens(self):
        """Test failure in half-open state reopens circuit"""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            name="test_breaker"
        )
        mock_func = Mock()
        
        # Open the circuit
        mock_func.side_effect = Exception("error")
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(mock_func)
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Failure in half-open state
        with pytest.raises(Exception):
            cb.call(mock_func)
        
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test circuit breaker with async functions"""
        cb = CircuitBreaker(failure_threshold=2, name="async_test")
        
        async def async_success():
            return "async_success"
        
        async def async_failure():
            raise Exception("async error")
        
        # Test successful async calls
        result = await cb.async_call(async_success)
        assert result == "async_success"
        assert cb.state == CircuitState.CLOSED
        
        # Test failing async calls
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.async_call(async_failure)
        
        assert cb.state == CircuitState.OPEN
        
        # Circuit open - fails fast
        with pytest.raises(CircuitBreakerError):
            await cb.async_call(async_success)
    
    def test_decorator_usage(self):
        """Test circuit breaker as decorator"""
        cb = CircuitBreaker(failure_threshold=2)
        
        @cb
        def protected_function(should_fail=False):
            if should_fail:
                raise Exception("Function failed")
            return "success"
        
        # Successful calls
        assert protected_function() == "success"
        assert protected_function() == "success"
        
        # Failing calls open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                protected_function(should_fail=True)
        
        # Circuit open
        with pytest.raises(CircuitBreakerError):
            protected_function()
    
    @pytest.mark.asyncio
    async def test_async_decorator_usage(self):
        """Test circuit breaker as async decorator"""
        cb = CircuitBreaker(failure_threshold=2)
        
        @cb
        async def protected_async_function(should_fail=False):
            if should_fail:
                raise Exception("Async function failed")
            return "async_success"
        
        # Successful calls
        assert await protected_async_function() == "async_success"
        
        # Failing calls
        for _ in range(2):
            with pytest.raises(Exception):
                await protected_async_function(should_fail=True)
        
        # Circuit open
        with pytest.raises(CircuitBreakerError):
            await protected_async_function()
    
    def test_metrics_collection(self):
        """Test metrics are collected correctly"""
        cb = CircuitBreaker(name="metrics_test")
        mock_func = Mock(return_value="success")
        
        # Make some successful calls
        for _ in range(5):
            cb.call(mock_func)
        
        # Make some failing calls
        mock_func.side_effect = Exception("error")
        for _ in range(3):
            try:
                cb.call(mock_func)
            except:
                pass
        
        metrics = cb.get_metrics()
        
        assert metrics['name'] == "metrics_test"
        assert metrics['state'] == CircuitState.OPEN.value
        assert metrics['failure_count'] == 3
        assert metrics['avg_response_time_ms'] >= 0
    
    def test_manual_reset(self):
        """Test manual circuit reset"""
        cb = CircuitBreaker(failure_threshold=2)
        mock_func = Mock(side_effect=Exception("error"))
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(mock_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Manual reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
    
    def test_callbacks(self):
        """Test on_open and on_close callbacks"""
        open_callback = Mock()
        close_callback = Mock()
        
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=1,
            on_open=open_callback,
            on_close=close_callback
        )
        
        mock_func = Mock()
        
        # Open circuit
        mock_func.side_effect = Exception("error")
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(mock_func)
        
        open_callback.assert_called_once()
        
        # Wait and recover
        time.sleep(1.1)
        mock_func.side_effect = None
        mock_func.return_value = "success"
        cb.call(mock_func)
        
        close_callback.assert_called_once()


class TestAPICircuitBreaker:
    """Test API-specific circuit breaker functionality"""
    
    def test_rate_limit_opens_circuit(self):
        """Test circuit opens on rate limit errors"""
        cb = APICircuitBreaker(
            api_name="test_api",
            rate_limit_threshold=2
        )
        
        assert cb.state == CircuitState.CLOSED
        
        # First rate limit
        cb.record_rate_limit()
        assert cb.state == CircuitState.CLOSED
        
        # Second rate limit opens circuit
        cb.record_rate_limit()
        assert cb.state == CircuitState.OPEN
    
    def test_timeout_opens_circuit(self):
        """Test circuit opens on timeout errors"""
        cb = APICircuitBreaker(
            api_name="test_api",
            timeout_threshold=3
        )
        
        # Record timeouts
        for i in range(2):
            cb.record_timeout()
            assert cb.state == CircuitState.CLOSED
        
        # Third timeout opens circuit
        cb.record_timeout()
        assert cb.state == CircuitState.OPEN
    
    def test_success_resets_counters(self):
        """Test successful call resets error counters"""
        cb = APICircuitBreaker(
            api_name="test_api",
            rate_limit_threshold=3,
            timeout_threshold=3
        )
        
        # Record some errors
        cb.record_rate_limit()
        cb.record_timeout()
        
        # Successful call
        mock_func = Mock(return_value="success")
        cb.call(mock_func)
        
        # Counters should be reset
        assert cb._rate_limit_count == 0
        assert cb._timeout_count == 0


class TestCircuitBreakerManager:
    """Test circuit breaker manager"""
    
    def test_register_and_get(self):
        """Test registering and retrieving circuit breakers"""
        manager = CircuitBreakerManager()
        cb = CircuitBreaker(name="test")
        
        manager.register("test", cb)
        retrieved = manager.get("test")
        
        assert retrieved is cb
    
    def test_get_or_create(self):
        """Test get_or_create functionality"""
        manager = CircuitBreakerManager()
        
        # Create new
        cb1 = manager.get_or_create("new_breaker", failure_threshold=5)
        assert cb1 is not None
        
        # Get existing
        cb2 = manager.get_or_create("new_breaker", failure_threshold=10)
        assert cb1 is cb2  # Same instance
    
    def test_get_all_metrics(self):
        """Test getting metrics for all breakers"""
        manager = CircuitBreakerManager()
        
        cb1 = CircuitBreaker(name="breaker1")
        cb2 = CircuitBreaker(name="breaker2")
        
        manager.register("breaker1", cb1)
        manager.register("breaker2", cb2)
        
        metrics = manager.get_all_metrics()
        
        assert "breaker1" in metrics
        assert "breaker2" in metrics
        assert metrics["breaker1"]["name"] == "breaker1"
        assert metrics["breaker2"]["name"] == "breaker2"
    
    def test_reset_all(self):
        """Test resetting all circuit breakers"""
        manager = CircuitBreakerManager()
        
        # Create and open some breakers
        for i in range(3):
            cb = CircuitBreaker(failure_threshold=1)
            mock_func = Mock(side_effect=Exception("error"))
            
            with pytest.raises(Exception):
                cb.call(mock_func)
            
            manager.register(f"breaker{i}", cb)
            assert cb.state == CircuitState.OPEN
        
        # Reset all
        manager.reset_all()
        
        # All should be closed
        for i in range(3):
            cb = manager.get(f"breaker{i}")
            assert cb.state == CircuitState.CLOSED
    
    def test_get_open_circuits(self):
        """Test getting list of open circuits"""
        manager = CircuitBreakerManager()
        
        # Create mixed state breakers
        cb_open = CircuitBreaker(failure_threshold=1)
        cb_closed = CircuitBreaker()
        
        # Open one breaker
        mock_func = Mock(side_effect=Exception("error"))
        with pytest.raises(Exception):
            cb_open.call(mock_func)
        
        manager.register("open", cb_open)
        manager.register("closed", cb_closed)
        
        open_circuits = manager.get_open_circuits()
        
        assert "open" in open_circuits
        assert "closed" not in open_circuits


class TestGetAPICircuitBreaker:
    """Test pre-configured API circuit breakers"""
    
    def test_finnhub_configuration(self):
        """Test Finnhub circuit breaker configuration"""
        cb = get_api_circuit_breaker("finnhub")
        
        assert cb.api_name == "finnhub"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30
        assert cb.rate_limit_threshold == 3
    
    def test_alpha_vantage_configuration(self):
        """Test Alpha Vantage circuit breaker configuration"""
        cb = get_api_circuit_breaker("alpha_vantage")
        
        assert cb.api_name == "alpha_vantage"
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 60
        assert cb.rate_limit_threshold == 2
    
    def test_polygon_configuration(self):
        """Test Polygon circuit breaker configuration"""
        cb = get_api_circuit_breaker("polygon")
        
        assert cb.api_name == "polygon"
        assert cb.failure_threshold == 4
        assert cb.recovery_timeout == 45
        assert cb.rate_limit_threshold == 2
    
    def test_yahoo_finance_configuration(self):
        """Test Yahoo Finance circuit breaker configuration"""
        cb = get_api_circuit_breaker("yahoo_finance")
        
        assert cb.api_name == "yahoo_finance"
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 20
        assert cb.rate_limit_threshold == 5
    
    def test_unknown_api_default_configuration(self):
        """Test default configuration for unknown API"""
        cb = get_api_circuit_breaker("unknown_api")
        
        assert cb.api_name == "unknown_api"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60
        assert cb.rate_limit_threshold == 3
    
    def test_singleton_behavior(self):
        """Test that same API returns same circuit breaker instance"""
        cb1 = get_api_circuit_breaker("finnhub")
        cb2 = get_api_circuit_breaker("finnhub")
        
        assert cb1 is cb2


class TestWithCircuitBreakerDecorator:
    """Test the with_circuit_breaker decorator"""
    
    def test_decorator_with_name(self):
        """Test decorator with custom name"""
        
        @with_circuit_breaker(name="custom_breaker", failure_threshold=2)
        def protected_function():
            return "success"
        
        assert protected_function() == "success"
    
    def test_decorator_without_name(self):
        """Test decorator uses function name when no name provided"""
        
        @with_circuit_breaker(failure_threshold=2)
        def my_function():
            return "success"
        
        assert my_function() == "success"
    
    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test decorator with async function"""
        
        @with_circuit_breaker(name="async_breaker")
        async def async_function():
            return "async_success"
        
        result = await async_function()
        assert result == "async_success"
    
    def test_decorator_opens_circuit(self):
        """Test decorator opens circuit on failures"""
        call_count = 0
        
        @with_circuit_breaker(failure_threshold=2)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Failed")
        
        # First two failures
        for _ in range(2):
            with pytest.raises(Exception):
                failing_function()
        
        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            failing_function()
        
        # Function called only twice (not on third attempt)
        assert call_count == 2


class TestThreadSafety:
    """Test thread safety of circuit breaker"""
    
    def test_concurrent_calls(self):
        """Test circuit breaker handles concurrent calls safely"""
        import threading
        
        cb = CircuitBreaker(failure_threshold=5)
        results = []
        errors = []
        
        def make_call(should_fail):
            try:
                if should_fail:
                    cb.call(Mock(side_effect=Exception("error")))
                else:
                    result = cb.call(Mock(return_value="success"))
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = []
        for i in range(10):
            should_fail = i % 2 == 0
            t = threading.Thread(target=make_call, args=(should_fail,))
            threads.append(t)
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify results
        assert len(results) > 0
        assert len(errors) > 0


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_api_retry_with_circuit_breaker(self):
        """Test API retry logic with circuit breaker"""
        from backend.utils.circuit_breaker import get_api_circuit_breaker
        
        cb = get_api_circuit_breaker("test_api")
        cb.reset()  # Ensure clean state
        
        api_calls = 0
        
        async def mock_api_call():
            nonlocal api_calls
            api_calls += 1
            
            if api_calls <= 3:
                raise Exception("API Error")
            return {"data": "success"}
        
        # First 3 calls fail
        for _ in range(3):
            with pytest.raises(Exception):
                await cb.async_call(mock_api_call)
        
        # Circuit should be open
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        await asyncio.sleep(cb.recovery_timeout + 0.1)
        
        # Should succeed now
        result = await cb.async_call(mock_api_call)
        assert result["data"] == "success"
    
    def test_cascading_failure_prevention(self):
        """Test prevention of cascading failures"""
        # Simulate multiple services
        service_a = CircuitBreaker(failure_threshold=2, name="ServiceA")
        service_b = CircuitBreaker(failure_threshold=2, name="ServiceB")
        
        def call_service_a():
            return service_a.call(Mock(side_effect=Exception("Service A down")))
        
        def call_service_b():
            try:
                # Service B depends on Service A
                call_service_a()
                return "success"
            except CircuitBreakerError:
                # Handle gracefully when Service A circuit is open
                return "fallback_response"
            except Exception:
                raise Exception("Service B error")
        
        # Service A fails
        for _ in range(2):
            with pytest.raises(Exception):
                call_service_a()
        
        # Service A circuit is open
        assert service_a.state == CircuitState.OPEN
        
        # Service B uses fallback instead of failing
        result = service_b.call(lambda: call_service_b())
        assert result == "fallback_response"
        
        # Service B circuit remains closed
        assert service_b.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])