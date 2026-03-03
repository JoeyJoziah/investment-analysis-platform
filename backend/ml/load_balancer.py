"""
Load balancer for distributed model serving.

Extracted from pipeline_optimization.py.  Contains LoadBalancer only.
Import via the original path (backend.ml.pipeline_optimization) or directly from here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from backend.ml.pipeline_types import LoadBalancingConfig
except ImportError:  # pragma: no cover
    from pipeline_types import LoadBalancingConfig  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class LoadBalancer:
    """Load balancer for distributed model serving"""

    def __init__(self, config: Optional[LoadBalancingConfig] = None) -> None:
        self.config = config or LoadBalancingConfig()

        # Worker registry
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Load balancing state
        self.current_worker_index = 0
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Health checking
        self.health_check_thread: Optional[threading.Thread] = None
        self.is_health_checking = False

        logger.info("Load balancer initialized")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_worker(
        self,
        worker_id: str,
        endpoint: str,
        weight: float = 1.0,
        max_connections: Optional[int] = None,
    ) -> None:
        """Register a model serving worker"""

        self.workers[worker_id] = {
            'endpoint': endpoint,
            'weight': weight,
            'max_connections': max_connections or self.config.max_connections_per_worker,
            'current_connections': 0,
            'is_healthy': True,
            'last_health_check': datetime.now(timezone.utc),
            'registered_at': datetime.now(timezone.utc),
        }

        self.worker_stats[worker_id] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_request_time': None,
        }

        self.circuit_breakers[worker_id] = {
            'failure_count': 0,
            'last_failure_time': None,
            'state': 'closed',  # closed, open, half_open
            'next_attempt_time': None,
        }

        logger.info(f"Registered worker {worker_id} at {endpoint}")

    def select_worker(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Select worker based on load balancing strategy"""

        available_workers: List[str] = [
            worker_id
            for worker_id, worker_info in self.workers.items()
            if (
                worker_info['is_healthy']
                and worker_info['current_connections'] < worker_info['max_connections']
                and self.circuit_breakers[worker_id]['state'] != 'open'
            )
        ]

        if not available_workers:
            half_open_workers = [
                wid
                for wid in self.workers.keys()
                if self.circuit_breakers[wid]['state'] == 'half_open'
            ]
            return half_open_workers[0] if half_open_workers else None

        strategy = self.config.strategy

        if strategy == "round_robin":
            worker_id = available_workers[self.current_worker_index % len(available_workers)]
            self.current_worker_index += 1
            return worker_id

        elif strategy == "weighted":
            weights = [self.workers[w]['weight'] for w in available_workers]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return str(np.random.choice(available_workers, p=weights))
            return available_workers[0]

        elif strategy == "least_connections":
            return min(
                available_workers, key=lambda w: self.workers[w]['current_connections']
            )

        return available_workers[0]

    def record_request(
        self,
        worker_id: str,
        success: bool,
        response_time: float,
    ) -> None:
        """Record request result for load balancing decisions"""

        if worker_id not in self.worker_stats:
            return

        stats = self.worker_stats[worker_id]
        circuit_breaker = self.circuit_breakers[worker_id]

        stats['total_requests'] += 1
        stats['last_request_time'] = datetime.now(timezone.utc)

        if success:
            stats['successful_requests'] += 1
            old_avg = stats['average_response_time']
            count = stats['successful_requests']
            stats['average_response_time'] = (old_avg * (count - 1) + response_time) / count

            if circuit_breaker['state'] == 'half_open':
                circuit_breaker['state'] = 'closed'
                circuit_breaker['failure_count'] = 0
                logger.info(f"Circuit breaker closed for worker {worker_id}")

        else:
            stats['failed_requests'] += 1
            circuit_breaker['failure_count'] += 1
            circuit_breaker['last_failure_time'] = datetime.now(timezone.utc)

            if (
                circuit_breaker['failure_count'] >= self.config.circuit_breaker_threshold
                and circuit_breaker['state'] == 'closed'
            ):
                circuit_breaker['state'] = 'open'
                circuit_breaker['next_attempt_time'] = datetime.now(timezone.utc) + timedelta(
                    seconds=60
                )
                logger.warning(f"Circuit breaker opened for worker {worker_id}")

    def start_health_checks(self) -> None:
        """Start health checking thread"""

        if self.is_health_checking:
            return

        self.is_health_checking = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self.health_check_thread.start()
        logger.info("Started health checking")

    def stop_health_checks(self) -> None:
        """Stop health checking thread"""

        self.is_health_checking = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("Stopped health checking")

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics"""

        result: Dict[str, Any] = {
            'total_workers': len(self.workers),
            'healthy_workers': sum(1 for w in self.workers.values() if w['is_healthy']),
            'workers': {},
        }

        for worker_id, worker_info in self.workers.items():
            worker_stats = self.worker_stats[worker_id].copy()
            circuit_breaker = self.circuit_breakers[worker_id]

            result['workers'][worker_id] = {
                'endpoint': worker_info['endpoint'],
                'weight': worker_info['weight'],
                'is_healthy': worker_info['is_healthy'],
                'current_connections': worker_info['current_connections'],
                'max_connections': worker_info['max_connections'],
                'circuit_breaker_state': circuit_breaker['state'],
                'circuit_breaker_failures': circuit_breaker['failure_count'],
                **worker_stats,
            }

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _health_check_loop(self) -> None:
        """Main health checking loop"""

        while self.is_health_checking:
            try:
                for worker_id, worker_info in self.workers.items():
                    is_healthy = self._check_worker_health(worker_id, worker_info)
                    worker_info['is_healthy'] = is_healthy
                    worker_info['last_health_check'] = datetime.now(timezone.utc)

                    circuit_breaker = self.circuit_breakers[worker_id]
                    if (
                        circuit_breaker['state'] == 'open'
                        and datetime.now(timezone.utc) >= circuit_breaker['next_attempt_time']
                    ):
                        circuit_breaker['state'] = 'half_open'
                        logger.info(f"Circuit breaker half-open for worker {worker_id}")

                # Blocking sleep OK: runs in dedicated daemon thread
                time.sleep(self.config.health_check_interval)

            except Exception as exc:
                logger.error(f"Error in health check loop: {exc}")
                time.sleep(10)  # Blocking sleep OK: runs in dedicated daemon thread

    def _check_worker_health(
        self, worker_id: str, worker_info: Dict[str, Any]
    ) -> bool:
        """Check health of a specific worker"""

        try:
            stats = self.worker_stats[worker_id]

            if stats['last_request_time'] is None:
                return True  # New worker, assume healthy

            last_request = stats['last_request_time']
            time_since_request = datetime.now(timezone.utc) - last_request

            if time_since_request.total_seconds() > 300:  # 5 minutes
                return True  # No recent requests, can't determine health

            total = stats['total_requests']
            successful = stats['successful_requests']

            if total > 10:
                success_rate = successful / total
                return success_rate > 0.8

            return True

        except Exception as exc:
            logger.error(f"Error checking health of worker {worker_id}: {exc}")
            return False


__all__ = ["LoadBalancer"]
