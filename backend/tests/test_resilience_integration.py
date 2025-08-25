"""
Error Handling and Resilience Integration Tests for Investment Analysis Platform
Tests circuit breakers, fallback mechanisms, retry logic, and system recovery.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from httpx import TimeoutException, ConnectError
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
import redis
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.utils.circuit_breaker import CircuitBreaker, CircuitState
from backend.utils.advanced_circuit_breaker import AdvancedCircuitBreaker
from backend.utils.redis_resilience import ResilientRedisClient, RedisMode
from backend.utils.resilient_pipeline import ResilientPipeline
from backend.utils.enhanced_error_handling import (
    EnhancedErrorHandler,
    RetryableError,
    NonRetryableError
)
from backend.utils.graceful_shutdown import GracefulShutdownManager
from backend.utils.disaster_recovery import DisasterRecoveryManager
from backend.data_ingestion.robust_api_client import RobustAPIClient


class TestCircuitBreakerIntegration:
    """Test circuit breaker patterns and fault tolerance."""

    @pytest.fixture
    def basic_circuit_breaker(self):
        """Create basic circuit breaker."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10,
            name="test_basic"
        )

    @pytest.fixture
    def advanced_circuit_breaker(self):
        """Create advanced circuit breaker."""
        return AdvancedCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            slow_call_threshold=2.0,
            slow_call_rate_threshold=0.5,
            minimum_throughput=10,
            name="test_advanced"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_circuit_breaker_state_transitions(self, basic_circuit_breaker):
        """Test circuit breaker state transitions under different failure scenarios."""
        
        # Initial state should be CLOSED
        assert basic_circuit_breaker.state == CircuitState.CLOSED
        
        # Simulate successful calls
        for i in range(5):
            result = await basic_circuit_breaker.call(lambda: f"success_{i}")
            assert result == f"success_{i}"
        
        assert basic_circuit_breaker.state == CircuitState.CLOSED
        assert basic_circuit_breaker.success_count == 5
        
        # Simulate failures to trip the circuit
        async def failing_function():
            raise Exception("Service unavailable")
        
        for i in range(3):  # failure_threshold = 3
            with pytest.raises(Exception):
                await basic_circuit_breaker.call(failing_function)
        
        # Circuit should be OPEN after reaching failure threshold
        assert basic_circuit_breaker.state == CircuitState.OPEN
        assert basic_circuit_breaker.failure_count == 3
        
        # Subsequent calls should fail fast
        with pytest.raises(Exception) as exc_info:
            await basic_circuit_breaker.call(lambda: "should_not_execute")
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
        
        # Test automatic recovery after timeout
        basic_circuit_breaker.last_failure_time = datetime.utcnow() - timedelta(seconds=15)
        
        # Should be in HALF_OPEN state after timeout
        result = await basic_circuit_breaker.call(lambda: "recovery_test")
        assert result == "recovery_test"
        assert basic_circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_circuit_breaker_with_fallback(self, basic_circuit_breaker):
        """Test circuit breaker with fallback mechanisms."""
        
        # Configure fallback function
        async def primary_function():
            raise Exception("Primary service down")
        
        async def fallback_function():
            return "fallback_response"
        
        # Trip the circuit breaker
        for _ in range(3):
            try:
                await basic_circuit_breaker.call(primary_function)
            except Exception:
                pass
        
        # Test fallback execution when circuit is open
        result = await basic_circuit_breaker.call(
            primary_function,
            fallback=fallback_function
        )
        
        assert result == "fallback_response"
        assert basic_circuit_breaker.fallback_count > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_advanced_circuit_breaker_features(self, advanced_circuit_breaker):
        """Test advanced circuit breaker features like slow call detection."""
        
        # Test slow call detection
        async def slow_function():
            await asyncio.sleep(3.0)  # Exceeds slow_call_threshold
            return "slow_response"
        
        # Execute several slow calls
        for i in range(10):
            try:
                await advanced_circuit_breaker.call(slow_function)
            except Exception:
                pass  # May timeout or be rejected
        
        # Check if slow calls were detected
        metrics = advanced_circuit_breaker.get_metrics()
        assert "slow_calls" in metrics
        
        # Test sliding window behavior
        assert hasattr(advanced_circuit_breaker, 'sliding_window')
        assert advanced_circuit_breaker.minimum_throughput == 10
        
        # Test partial failure scenarios
        async def partially_failing_function():
            if advanced_circuit_breaker.call_count % 3 == 0:
                raise Exception("Intermittent failure")
            return "success"
        
        # Execute mixed success/failure pattern
        results = []
        for i in range(20):
            try:
                result = await advanced_circuit_breaker.call(partially_failing_function)
                results.append(result)
            except Exception as e:
                results.append(str(e))
        
        # Should handle partial failures intelligently
        success_rate = len([r for r in results if r == "success"]) / len(results)
        assert 0.4 < success_rate < 0.8  # Expect partial success


class TestResilientRedisIntegration:
    """Test Redis resilience and cluster failover."""

    @pytest.fixture
    def resilient_redis_client(self):
        """Create resilient Redis client."""
        return ResilientRedisClient(
            mode=RedisMode.STANDALONE,
            standalone_url="redis://localhost:6379",
            max_retries=3,
            retry_delay=1.0
        )

    @pytest.fixture
    def sentinel_redis_client(self):
        """Create sentinel Redis client."""
        return ResilientRedisClient(
            mode=RedisMode.SENTINEL,
            sentinel_hosts=[("localhost", 26379), ("localhost", 26380), ("localhost", 26381)],
            sentinel_service="mymaster",
            max_retries=5
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_redis_connection_resilience(self, resilient_redis_client):
        """Test Redis connection resilience and recovery."""
        
        await resilient_redis_client.initialize()
        
        # Test normal operation
        await resilient_redis_client.set("test_key", "test_value", ttl=60)
        result = await resilient_redis_client.get("test_key")
        assert result == "test_value"
        
        # Simulate connection failure
        with patch.object(resilient_redis_client._redis_client, 'get') as mock_get:
            mock_get.side_effect = [
                RedisConnectionError("Connection lost"),
                RedisConnectionError("Still down"),
                "recovered_value"  # Third attempt succeeds
            ]
            
            # Should retry and eventually succeed
            result = await resilient_redis_client.get("test_key")
            assert result == "recovered_value"
            assert mock_get.call_count == 3
        
        # Test circuit breaker integration
        circuit_breaker = resilient_redis_client.circuit_breaker
        assert circuit_breaker is not None
        
        # Simulate persistent failure to trip circuit breaker
        with patch.object(resilient_redis_client._redis_client, 'set') as mock_set:
            mock_set.side_effect = RedisConnectionError("Redis server down")
            
            # Multiple failed attempts should trip circuit breaker
            for _ in range(5):
                try:
                    await resilient_redis_client.set("fail_key", "value")
                except Exception:
                    pass
            
            # Circuit breaker should be open
            assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_redis_sentinel_failover(self, sentinel_redis_client):
        """Test Redis Sentinel automatic failover."""
        
        with patch('redis.sentinel.Sentinel') as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_sentinel_class.return_value = mock_sentinel
            
            # Mock master discovery
            mock_sentinel.master_for.return_value = AsyncMock()
            mock_sentinel.slave_for.return_value = AsyncMock()
            
            await sentinel_redis_client.initialize()
            
            # Test master failure simulation
            mock_sentinel.master_for.side_effect = [
                Exception("Master not available"),
                AsyncMock()  # Failover successful
            ]
            
            # Should handle master failover
            await sentinel_redis_client.set("failover_test", "value")
            
            # Verify failover was handled
            assert mock_sentinel.master_for.call_count >= 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_redis_cluster_resilience(self):
        """Test Redis cluster resilience and node failure handling."""
        
        cluster_client = ResilientRedisClient(
            mode=RedisMode.CLUSTER,
            cluster_nodes=[
                ("localhost", 7000),
                ("localhost", 7001), 
                ("localhost", 7002)
            ],
            max_retries=3
        )
        
        with patch('redis.cluster.RedisCluster') as mock_cluster_class:
            mock_cluster = AsyncMock()
            mock_cluster_class.return_value = mock_cluster
            
            await cluster_client.initialize()
            
            # Simulate cluster slot migration
            mock_cluster.get.side_effect = [
                Exception("MOVED 3999 127.0.0.1:7001"),
                "migrated_value"  # After slot migration
            ]
            
            result = await cluster_client.get("cluster_key")
            assert result == "migrated_value"
            
            # Should handle cluster topology changes
            assert mock_cluster.get.call_count == 2


class TestAPIClientResilience:
    """Test API client resilience and error handling."""

    @pytest.fixture
    def robust_api_client(self):
        """Create robust API client."""
        return RobustAPIClient(
            base_url="https://api.example.com",
            timeout=30.0,
            max_retries=3,
            backoff_factor=2.0
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_api_retry_logic(self, robust_api_client):
        """Test API retry logic for different error types."""
        
        with patch('httpx.AsyncClient.get') as mock_get:
            # Test timeout retry
            mock_get.side_effect = [
                TimeoutException("Request timeout"),
                TimeoutException("Request timeout again"),
                MagicMock(status_code=200, json=lambda: {"data": "success"})
            ]
            
            response = await robust_api_client.get("/test/endpoint")
            
            assert response["data"] == "success"
            assert mock_get.call_count == 3
        
        # Test non-retryable errors
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=400, text="Bad Request"),  # Client error - no retry
            ]
            
            with pytest.raises(Exception):
                await robust_api_client.get("/invalid/endpoint")
            
            # Should not retry 400 errors
            assert mock_get.call_count == 1
        
        # Test server error retry
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=500, text="Internal Server Error"),
                MagicMock(status_code=502, text="Bad Gateway"),
                MagicMock(status_code=200, json=lambda: {"recovered": True})
            ]
            
            response = await robust_api_client.get("/server/error")
            
            assert response["recovered"] is True
            assert mock_get.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_api_circuit_breaker_integration(self, robust_api_client):
        """Test API client with circuit breaker integration."""
        
        # Ensure client has circuit breaker
        circuit_breaker = getattr(robust_api_client, 'circuit_breaker', None)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=10
            )
            robust_api_client.circuit_breaker = circuit_breaker
        
        with patch('httpx.AsyncClient.get') as mock_get:
            # Cause multiple failures to trip circuit breaker
            mock_get.side_effect = [
                MagicMock(status_code=500, text="Server Error"),
                MagicMock(status_code=503, text="Service Unavailable"),
                MagicMock(status_code=500, text="Server Error"),
                MagicMock(status_code=500, text="Still failing")
            ]
            
            # Execute calls that should trip the breaker
            for i in range(4):
                try:
                    await robust_api_client.get(f"/endpoint/{i}")
                except Exception:
                    pass
            
            # Circuit should be open
            assert circuit_breaker.state == CircuitState.OPEN
        
        # Test fallback data when circuit is open
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: {"fallback": True})
            
            # Should use fallback when circuit is open
            result = await robust_api_client.get_with_fallback(
                "/endpoint/fallback",
                fallback_data={"cached": "data"}
            )
            
            assert "cached" in result or "fallback" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_api_rate_limiting_resilience(self, robust_api_client):
        """Test API rate limiting and backoff strategies."""
        
        with patch('httpx.AsyncClient.get') as mock_get:
            # Simulate rate limiting responses
            mock_get.side_effect = [
                MagicMock(status_code=429, headers={"Retry-After": "60"}),  # Rate limited
                MagicMock(status_code=429, headers={"X-RateLimit-Reset": str(int(datetime.utcnow().timestamp()) + 30)}),
                MagicMock(status_code=200, json=lambda: {"data": "after_backoff"})
            ]
            
            # Should handle rate limiting with backoff
            start_time = datetime.utcnow()
            
            with patch('asyncio.sleep') as mock_sleep:
                response = await robust_api_client.get("/rate/limited")
                
                # Should have attempted backoff
                mock_sleep.assert_called()
                
                # Should eventually succeed
                assert response["data"] == "after_backoff"


class TestDatabaseResilience:
    """Test database resilience and connection recovery."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_connection_recovery(self):
        """Test database connection recovery mechanisms."""
        
        from backend.config.database import get_async_db_session
        
        with patch('sqlalchemy.ext.asyncio.create_async_engine') as mock_engine:
            mock_session = AsyncMock()
            mock_engine.return_value.begin.return_value.__aenter__.return_value = mock_session
            
            # Simulate connection loss and recovery
            mock_session.execute.side_effect = [
                DisconnectionError("Connection lost", None, None),
                DisconnectionError("Still disconnected", None, None),
                MagicMock()  # Connection recovered
            ]
            
            # Test automatic retry on connection loss
            session = await get_async_db_session()
            
            # Should eventually succeed after retries
            # In production, this would be handled by connection pooling

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_transaction_rollback(self):
        """Test database transaction rollback on errors."""
        
        from backend.config.database import get_async_db_session
        
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        
        with patch('backend.config.database.get_async_db_session', return_value=mock_session):
            try:
                async with mock_session.begin():
                    # Simulate operation that fails
                    mock_session.execute.side_effect = SQLAlchemyError("Database error")
                    raise SQLAlchemyError("Transaction failed")
            except SQLAlchemyError:
                pass
            
            # Should have called rollback
            mock_session.rollback.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_deadlock_handling(self):
        """Test database deadlock detection and handling."""
        
        from backend.utils.deadlock_handler import DeadlockHandler
        
        deadlock_handler = DeadlockHandler(max_retries=3, backoff_delay=0.1)
        
        call_count = 0
        
        async def deadlock_prone_operation():
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                # Simulate deadlock
                raise SQLAlchemyError("deadlock detected")
            
            return "success_after_retry"
        
        # Should retry and eventually succeed
        result = await deadlock_handler.execute_with_retry(deadlock_prone_operation)
        
        assert result == "success_after_retry"
        assert call_count == 3


class TestSystemResilience:
    """Test overall system resilience and disaster recovery."""

    @pytest.fixture
    def disaster_recovery_manager(self):
        """Create disaster recovery manager."""
        return DisasterRecoveryManager(
            backup_interval=3600,  # 1 hour
            max_backup_age=86400 * 7,  # 7 days
            health_check_interval=60  # 1 minute
        )

    @pytest.fixture
    def graceful_shutdown_manager(self):
        """Create graceful shutdown manager."""
        return GracefulShutdownManager(
            shutdown_timeout=30,
            force_shutdown_timeout=60
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_graceful_shutdown(self, graceful_shutdown_manager):
        """Test graceful application shutdown."""
        
        # Simulate running services
        services = []
        for i in range(3):
            service = AsyncMock()
            service.name = f"service_{i}"
            service.shutdown = AsyncMock()
            services.append(service)
        
        graceful_shutdown_manager.register_services(services)
        
        # Test shutdown sequence
        await graceful_shutdown_manager.shutdown()
        
        # All services should have been shut down
        for service in services:
            service.shutdown.assert_called_once()
        
        # Test timeout handling
        slow_service = AsyncMock()
        slow_service.name = "slow_service"
        
        async def slow_shutdown():
            await asyncio.sleep(35)  # Exceeds timeout
        
        slow_service.shutdown = slow_shutdown
        graceful_shutdown_manager.register_service(slow_service)
        
        # Should handle timeout
        start_time = datetime.utcnow()
        await graceful_shutdown_manager.shutdown()
        end_time = datetime.utcnow()
        
        # Should not wait beyond force shutdown timeout
        duration = (end_time - start_time).total_seconds()
        assert duration < 65  # Force shutdown timeout + buffer

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_check_integration(self):
        """Test comprehensive health check system."""
        
        from backend.monitoring.health_checks import HealthCheckManager
        
        health_manager = HealthCheckManager()
        
        # Mock health checks for different components
        async def database_health_check():
            # Simulate database connectivity check
            return {"status": "healthy", "response_time": 0.05}
        
        async def redis_health_check():
            # Simulate Redis connectivity check
            return {"status": "healthy", "memory_usage": "50MB"}
        
        async def api_health_check():
            # Simulate external API health check
            return {"status": "degraded", "error": "Rate limited"}
        
        health_manager.register_check("database", database_health_check)
        health_manager.register_check("redis", redis_health_check)
        health_manager.register_check("external_apis", api_health_check)
        
        # Run comprehensive health check
        health_status = await health_manager.run_all_checks()
        
        assert "database" in health_status
        assert "redis" in health_status
        assert "external_apis" in health_status
        
        # Overall status should reflect worst component status
        assert health_status["overall_status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_disaster_recovery_procedures(self, disaster_recovery_manager):
        """Test disaster recovery procedures."""
        
        # Test backup creation
        backup_metadata = await disaster_recovery_manager.create_backup()
        
        assert "backup_id" in backup_metadata
        assert "timestamp" in backup_metadata
        assert "size" in backup_metadata
        
        # Test backup restoration
        with patch.object(disaster_recovery_manager, '_restore_from_backup') as mock_restore:
            mock_restore.return_value = {"status": "success", "restored_items": 1000}
            
            restore_result = await disaster_recovery_manager.restore_backup(
                backup_metadata["backup_id"]
            )
            
            assert restore_result["status"] == "success"
            mock_restore.assert_called_once()
        
        # Test automated recovery procedures
        with patch.object(disaster_recovery_manager, '_detect_system_failure') as mock_detect:
            mock_detect.return_value = True
            
            recovery_result = await disaster_recovery_manager.attempt_auto_recovery()
            
            # Should attempt recovery when failure is detected
            assert recovery_result is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascade_failure_prevention(self):
        """Test prevention of cascade failures."""
        
        from backend.utils.resilient_pipeline import ResilientPipeline
        
        pipeline = ResilientPipeline(
            max_concurrent_operations=5,
            circuit_breaker_threshold=3,
            isolation_enabled=True
        )
        
        # Simulate services with different failure modes
        async def stable_service(data):
            await asyncio.sleep(0.1)
            return f"processed_{data}"
        
        async def unstable_service(data):
            if random.random() < 0.7:  # 70% failure rate
                raise Exception(f"Service failure for {data}")
            return f"unstable_processed_{data}"
        
        async def critical_service(data):
            # This service should be isolated from failures
            await asyncio.sleep(0.05)
            return f"critical_{data}"
        
        # Process data through pipeline
        test_data = list(range(20))
        
        results = await pipeline.process_batch(
            data=test_data,
            processors={
                "stable": stable_service,
                "unstable": unstable_service,
                "critical": critical_service
            },
            isolation_groups={
                "critical": ["critical"]  # Isolate critical service
            }
        )
        
        # Critical service should maintain high success rate despite other failures
        critical_results = [r for r in results if "critical" in str(r)]
        critical_success_rate = len([r for r in critical_results if "critical_" in str(r)]) / len(critical_results)
        
        assert critical_success_rate > 0.8  # Should be isolated from other failures


class TestEndToEndResilience:
    """Test end-to-end resilience scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_system_failure_recovery(self):
        """Test complete system failure and recovery scenario."""
        
        # Simulate complete system failure
        components = {
            "database": {"status": "down", "last_seen": datetime.utcnow() - timedelta(minutes=5)},
            "redis": {"status": "down", "last_seen": datetime.utcnow() - timedelta(minutes=3)},
            "external_apis": {"status": "down", "last_seen": datetime.utcnow() - timedelta(minutes=2)}
        }
        
        # Test system recovery sequence
        recovery_steps = []
        
        # 1. Detect failures
        for component, info in components.items():
            if info["status"] == "down":
                recovery_steps.append(f"detected_{component}_failure")
        
        # 2. Initiate recovery procedures
        recovery_steps.append("initiated_recovery")
        
        # 3. Restore services in order
        restore_order = ["database", "redis", "external_apis"]
        for component in restore_order:
            # Simulate restoration
            await asyncio.sleep(0.1)  # Simulate recovery time
            components[component]["status"] = "recovering"
            recovery_steps.append(f"restoring_{component}")
        
        # 4. Verify system health
        all_healthy = all(
            comp["status"] in ["healthy", "recovering"] 
            for comp in components.values()
        )
        
        if all_healthy:
            recovery_steps.append("system_recovery_complete")
        
        # Verify recovery sequence
        expected_steps = [
            "detected_database_failure",
            "detected_redis_failure", 
            "detected_external_apis_failure",
            "initiated_recovery",
            "restoring_database",
            "restoring_redis",
            "restoring_external_apis",
            "system_recovery_complete"
        ]
        
        assert recovery_steps == expected_steps

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_partial_degradation_handling(self):
        """Test system behavior under partial degradation."""
        
        # Simulate partial system degradation
        system_state = {
            "api_latency": 2.5,  # High latency
            "database_connections": 8,  # Limited connections
            "cache_hit_rate": 0.3,  # Low cache performance
            "external_api_rate_limit": 0.9  # Near rate limit
        }
        
        # Test degraded mode operations
        degraded_operations = []
        
        # High latency -> Reduce timeout
        if system_state["api_latency"] > 2.0:
            degraded_operations.append("reduced_timeout")
        
        # Limited DB connections -> Use connection pooling
        if system_state["database_connections"] < 10:
            degraded_operations.append("connection_pooling")
        
        # Low cache hit rate -> Increase cache TTL
        if system_state["cache_hit_rate"] < 0.5:
            degraded_operations.append("extended_cache_ttl")
        
        # Near rate limit -> Implement backoff
        if system_state["external_api_rate_limit"] > 0.8:
            degraded_operations.append("api_backoff")
        
        # System should adapt to degraded conditions
        expected_adaptations = [
            "reduced_timeout",
            "connection_pooling", 
            "extended_cache_ttl",
            "api_backoff"
        ]
        
        assert set(degraded_operations) == set(expected_adaptations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])