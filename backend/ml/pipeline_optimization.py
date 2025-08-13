"""
ML Pipeline Optimization System
Provides efficient model inference, caching, load balancing, and artifact management
"""

import os
import json
import pickle
import hashlib
import logging
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import time
import psutil
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import redis
from kubernetes import client, config as k8s_config
import docker

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Pipeline optimization strategies"""
    CACHING = "caching"
    PARALLELIZATION = "parallelization"
    QUANTIZATION = "quantization"
    BATCHING = "batching"
    LOAD_BALANCING = "load_balancing"
    PREPROCESSING_CACHE = "preprocessing_cache"


class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    SKLEARN_JOBLIB = "sklearn_joblib"
    XGBOOST = "xgboost"
    PICKLE = "pickle"


@dataclass
class InferenceMetrics:
    """Inference performance metrics"""
    model_name: str
    timestamp: datetime
    batch_size: int
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    throughput_samples_per_sec: float
    cache_hit_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LoadBalancingConfig:
    """Load balancing configuration"""
    strategy: str = "round_robin"  # round_robin, weighted, least_connections
    health_check_interval: int = 30  # seconds
    max_connections_per_worker: int = 100
    timeout_seconds: int = 30
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5  # failures before circuit break


class ModelArtifactManager:
    """Manages model artifacts and optimized versions"""
    
    def __init__(self, storage_path: str = "/app/model_artifacts"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Artifact registry
        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self.load_artifacts_registry()
        
        # Optimization cache
        self.optimization_cache: Dict[str, Any] = {}
        
        logger.info(f"Model artifact manager initialized with {len(self.artifacts)} artifacts")
    
    def store_artifact(self,
                      model_name: str,
                      model_version: str,
                      model_object: Any,
                      model_format: ModelFormat,
                      metadata: Dict[str, Any] = None) -> str:
        """Store model artifact with optimizations"""
        
        artifact_id = f"{model_name}_{model_version}_{model_format.value}"
        artifact_dir = self.storage_path / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original model
        original_path = artifact_dir / f"original.{model_format.value}"
        self._save_model_by_format(model_object, original_path, model_format)
        
        # Create optimized versions
        optimized_paths = self._create_optimized_versions(
            model_object, artifact_dir, model_format
        )
        
        # Store metadata
        artifact_metadata = {
            'model_name': model_name,
            'model_version': model_version,
            'format': model_format.value,
            'created_at': datetime.utcnow().isoformat(),
            'original_path': str(original_path),
            'original_size_mb': original_path.stat().st_size / (1024 * 1024),
            'optimized_versions': optimized_paths,
            'metadata': metadata or {}
        }
        
        self.artifacts[artifact_id] = artifact_metadata
        self._save_artifacts_registry()
        
        logger.info(f"Stored artifact {artifact_id} with {len(optimized_paths)} optimized versions")
        
        return artifact_id
    
    def load_artifact(self, 
                     artifact_id: str,
                     optimization: str = "original") -> Optional[Any]:
        """Load model artifact with specified optimization"""
        
        if artifact_id not in self.artifacts:
            logger.error(f"Artifact {artifact_id} not found")
            return None
        
        artifact_info = self.artifacts[artifact_id]
        
        try:
            if optimization == "original":
                model_path = Path(artifact_info['original_path'])
                model_format = ModelFormat(artifact_info['format'])
            else:
                optimized_versions = artifact_info.get('optimized_versions', {})
                if optimization not in optimized_versions:
                    logger.warning(f"Optimization {optimization} not available for {artifact_id}")
                    model_path = Path(artifact_info['original_path'])
                    model_format = ModelFormat(artifact_info['format'])
                else:
                    model_path = Path(optimized_versions[optimization])
                    # Determine format from path extension
                    if '.onnx' in str(model_path):
                        model_format = ModelFormat.ONNX
                    elif '.trt' in str(model_path):
                        model_format = ModelFormat.TENSORRT
                    else:
                        model_format = ModelFormat(artifact_info['format'])
            
            return self._load_model_by_format(model_path, model_format)
            
        except Exception as e:
            logger.error(f"Error loading artifact {artifact_id}: {e}")
            return None
    
    def _save_model_by_format(self, model: Any, path: Path, format: ModelFormat):
        """Save model in specified format"""
        
        if format == ModelFormat.PYTORCH:
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), path)
            else:
                torch.save(model, path)
                
        elif format == ModelFormat.SKLEARN_JOBLIB:
            joblib.dump(model, path)
            
        elif format in [ModelFormat.XGBOOST, ModelFormat.PICKLE]:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
                
        else:
            raise ValueError(f"Unsupported format for saving: {format}")
    
    def _load_model_by_format(self, path: Path, format: ModelFormat) -> Any:
        """Load model in specified format"""
        
        if format == ModelFormat.PYTORCH:
            return torch.load(path, map_location='cpu')
            
        elif format == ModelFormat.SKLEARN_JOBLIB:
            return joblib.load(path)
            
        elif format in [ModelFormat.XGBOOST, ModelFormat.PICKLE]:
            with open(path, 'rb') as f:
                return pickle.load(f)
                
        elif format == ModelFormat.ONNX:
            try:
                import onnxruntime as ort
                return ort.InferenceSession(str(path))
            except ImportError:
                logger.error("ONNX runtime not available")
                return None
                
        else:
            raise ValueError(f"Unsupported format for loading: {format}")
    
    def _create_optimized_versions(self,
                                  model: Any,
                                  artifact_dir: Path,
                                  format: ModelFormat) -> Dict[str, str]:
        """Create optimized versions of the model"""
        
        optimized_paths = {}
        
        try:
            # Quantization for PyTorch models
            if format == ModelFormat.PYTORCH and hasattr(model, 'eval'):
                quantized_path = artifact_dir / "quantized.pt"
                quantized_model = self._quantize_pytorch_model(model)
                if quantized_model:
                    torch.save(quantized_model, quantized_path)
                    optimized_paths['quantized'] = str(quantized_path)
            
            # ONNX export for PyTorch models
            if format == ModelFormat.PYTORCH:
                onnx_path = artifact_dir / "model.onnx"
                if self._export_to_onnx(model, onnx_path):
                    optimized_paths['onnx'] = str(onnx_path)
                    
                    # TensorRT optimization (if available)
                    tensorrt_path = artifact_dir / "model.trt"
                    if self._optimize_with_tensorrt(onnx_path, tensorrt_path):
                        optimized_paths['tensorrt'] = str(tensorrt_path)
            
            # Compressed versions
            if format in [ModelFormat.SKLEARN_JOBLIB, ModelFormat.XGBOOST]:
                compressed_path = artifact_dir / f"compressed.{format.value}"
                if self._compress_model(model, compressed_path):
                    optimized_paths['compressed'] = str(compressed_path)
            
        except Exception as e:
            logger.error(f"Error creating optimized versions: {e}")
        
        return optimized_paths
    
    def _quantize_pytorch_model(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Apply quantization to PyTorch model"""
        try:
            model.eval()
            
            # Post-training quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error quantizing PyTorch model: {e}")
            return None
    
    def _export_to_onnx(self, model: torch.nn.Module, onnx_path: Path) -> bool:
        """Export PyTorch model to ONNX"""
        try:
            model.eval()
            
            # Create dummy input (would need to be model-specific)
            dummy_input = torch.randn(1, 10)  # Adjust based on model
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            return False
    
    def _optimize_with_tensorrt(self, onnx_path: Path, tensorrt_path: Path) -> bool:
        """Optimize ONNX model with TensorRT"""
        try:
            # This would require TensorRT installation
            logger.info("TensorRT optimization not implemented (requires TensorRT SDK)")
            return False
            
        except Exception as e:
            logger.error(f"Error optimizing with TensorRT: {e}")
            return False
    
    def _compress_model(self, model: Any, compressed_path: Path) -> bool:
        """Compress model using various techniques"""
        try:
            # For sklearn models, try different compression levels
            if hasattr(model, 'predict'):
                joblib.dump(model, compressed_path, compress=3)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error compressing model: {e}")
            return False
    
    def load_artifacts_registry(self):
        """Load artifacts registry from disk"""
        registry_file = self.storage_path / "artifacts_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.artifacts = json.load(f)
            except Exception as e:
                logger.error(f"Error loading artifacts registry: {e}")
                self.artifacts = {}
    
    def _save_artifacts_registry(self):
        """Save artifacts registry to disk"""
        registry_file = self.storage_path / "artifacts_registry.json"
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.artifacts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving artifacts registry: {e}")


class InferenceCache:
    """Intelligent caching system for model predictions"""
    
    def __init__(self, 
                 max_size: int = 10000,
                 ttl_seconds: int = 3600,
                 redis_url: str = None):
        
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # In-memory cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.cache_lock = threading.Lock()
        
        # Redis cache (optional)
        self.redis_client = None
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached prediction"""
        
        with self.cache_lock:
            # Check in-memory cache first
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                
                # Check TTL
                if datetime.utcnow() - cache_entry['timestamp'] < timedelta(seconds=self.ttl_seconds):
                    self.access_times[cache_key] = datetime.utcnow()
                    self.hits += 1
                    return cache_entry['data']
                else:
                    # Expired
                    del self.cache[cache_key]
                    del self.access_times[cache_key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"prediction:{cache_key}")
                if cached_data:
                    data = pickle.loads(cached_data)
                    
                    # Store back in memory cache
                    self._store_in_memory(cache_key, data)
                    
                    self.hits += 1
                    return data
            except Exception as e:
                logger.error(f"Error reading from Redis cache: {e}")
        
        self.misses += 1
        return None
    
    def set(self, cache_key: str, data: Any):
        """Store prediction in cache"""
        
        # Store in memory
        self._store_in_memory(cache_key, data)
        
        # Store in Redis
        if self.redis_client:
            try:
                serialized_data = pickle.dumps(data)
                self.redis_client.setex(
                    f"prediction:{cache_key}",
                    self.ttl_seconds,
                    serialized_data
                )
            except Exception as e:
                logger.error(f"Error storing in Redis cache: {e}")
    
    def _store_in_memory(self, cache_key: str, data: Any):
        """Store data in in-memory cache"""
        
        with self.cache_lock:
            # Evict if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.utcnow()
            }
            self.access_times[cache_key] = datetime.utcnow()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        
        if not self.access_times:
            return
        
        # Find LRU entry
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Remove from cache
        del self.cache[lru_key]
        del self.access_times[lru_key]
        
        self.evictions += 1
    
    def generate_cache_key(self, 
                          model_name: str,
                          input_data: np.ndarray,
                          parameters: Dict[str, Any] = None) -> str:
        """Generate cache key for input"""
        
        # Hash input data
        input_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        
        # Include parameters
        param_str = json.dumps(parameters or {}, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return f"{model_name}:{input_hash}:{param_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': hit_ratio,
            'evictions': self.evictions,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self):
        """Clear all caches"""
        
        with self.cache_lock:
            self.cache.clear()
            self.access_times.clear()
        
        if self.redis_client:
            try:
                # Clear prediction keys
                keys = self.redis_client.keys("prediction:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")


class LoadBalancer:
    """Load balancer for distributed model serving"""
    
    def __init__(self, config: LoadBalancingConfig = None):
        self.config = config or LoadBalancingConfig()
        
        # Worker registry
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Load balancing state
        self.current_worker_index = 0
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Health checking
        self.health_check_thread = None
        self.is_health_checking = False
        
        logger.info("Load balancer initialized")
    
    def register_worker(self,
                       worker_id: str,
                       endpoint: str,
                       weight: float = 1.0,
                       max_connections: int = None):
        """Register a model serving worker"""
        
        self.workers[worker_id] = {
            'endpoint': endpoint,
            'weight': weight,
            'max_connections': max_connections or self.config.max_connections_per_worker,
            'current_connections': 0,
            'is_healthy': True,
            'last_health_check': datetime.utcnow(),
            'registered_at': datetime.utcnow()
        }
        
        # Initialize stats
        self.worker_stats[worker_id] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_request_time': None
        }
        
        # Initialize circuit breaker
        self.circuit_breakers[worker_id] = {
            'failure_count': 0,
            'last_failure_time': None,
            'state': 'closed',  # closed, open, half_open
            'next_attempt_time': None
        }
        
        logger.info(f"Registered worker {worker_id} at {endpoint}")
    
    def select_worker(self, request_context: Dict[str, Any] = None) -> Optional[str]:
        """Select worker based on load balancing strategy"""
        
        available_workers = [
            worker_id for worker_id, worker_info in self.workers.items()
            if (worker_info['is_healthy'] and 
                worker_info['current_connections'] < worker_info['max_connections'] and
                self.circuit_breakers[worker_id]['state'] != 'open')
        ]
        
        if not available_workers:
            # Try half-open circuit breakers
            half_open_workers = [
                worker_id for worker_id in self.workers.keys()
                if self.circuit_breakers[worker_id]['state'] == 'half_open'
            ]
            
            if half_open_workers:
                return half_open_workers[0]  # Try the first half-open worker
            
            return None
        
        strategy = self.config.strategy
        
        if strategy == "round_robin":
            worker_id = available_workers[self.current_worker_index % len(available_workers)]
            self.current_worker_index += 1
            return worker_id
            
        elif strategy == "weighted":
            # Weighted random selection
            weights = [self.workers[w]['weight'] for w in available_workers]
            total_weight = sum(weights)
            
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return np.random.choice(available_workers, p=weights)
            else:
                return available_workers[0]
                
        elif strategy == "least_connections":
            return min(available_workers, 
                      key=lambda w: self.workers[w]['current_connections'])
        
        else:
            # Default to round robin
            return available_workers[0]
    
    def record_request(self, 
                      worker_id: str, 
                      success: bool, 
                      response_time: float):
        """Record request result for load balancing decisions"""
        
        if worker_id not in self.worker_stats:
            return
        
        stats = self.worker_stats[worker_id]
        circuit_breaker = self.circuit_breakers[worker_id]
        
        # Update stats
        stats['total_requests'] += 1
        stats['last_request_time'] = datetime.utcnow()
        
        if success:
            stats['successful_requests'] += 1
            
            # Update average response time
            old_avg = stats['average_response_time']
            count = stats['successful_requests']
            stats['average_response_time'] = (old_avg * (count - 1) + response_time) / count
            
            # Reset circuit breaker on success
            if circuit_breaker['state'] == 'half_open':
                circuit_breaker['state'] = 'closed'
                circuit_breaker['failure_count'] = 0
                logger.info(f"Circuit breaker closed for worker {worker_id}")
                
        else:
            stats['failed_requests'] += 1
            
            # Update circuit breaker
            circuit_breaker['failure_count'] += 1
            circuit_breaker['last_failure_time'] = datetime.utcnow()
            
            if (circuit_breaker['failure_count'] >= self.config.circuit_breaker_threshold and 
                circuit_breaker['state'] == 'closed'):
                # Open circuit breaker
                circuit_breaker['state'] = 'open'
                circuit_breaker['next_attempt_time'] = (
                    datetime.utcnow() + timedelta(seconds=60)  # 60 second timeout
                )
                logger.warning(f"Circuit breaker opened for worker {worker_id}")
    
    def start_health_checks(self):
        """Start health checking thread"""
        
        if self.is_health_checking:
            return
        
        self.is_health_checking = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        logger.info("Started health checking")
    
    def stop_health_checks(self):
        """Stop health checking thread"""
        
        self.is_health_checking = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        logger.info("Stopped health checking")
    
    def _health_check_loop(self):
        """Main health checking loop"""
        
        while self.is_health_checking:
            try:
                for worker_id, worker_info in self.workers.items():
                    is_healthy = self._check_worker_health(worker_id, worker_info)
                    worker_info['is_healthy'] = is_healthy
                    worker_info['last_health_check'] = datetime.utcnow()
                    
                    # Update circuit breaker state
                    circuit_breaker = self.circuit_breakers[worker_id]
                    if (circuit_breaker['state'] == 'open' and 
                        datetime.utcnow() >= circuit_breaker['next_attempt_time']):
                        circuit_breaker['state'] = 'half_open'
                        logger.info(f"Circuit breaker half-open for worker {worker_id}")
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(10)
    
    def _check_worker_health(self, worker_id: str, worker_info: Dict[str, Any]) -> bool:
        """Check health of a specific worker"""
        
        try:
            # Simple health check (in practice would be HTTP request)
            # For now, just check if worker has been responding
            stats = self.worker_stats[worker_id]
            
            if stats['last_request_time'] is None:
                return True  # New worker, assume healthy
            
            # Check if worker has been responsive recently
            last_request = stats['last_request_time']
            time_since_request = datetime.utcnow() - last_request
            
            if time_since_request.total_seconds() > 300:  # 5 minutes
                return True  # No recent requests, can't determine health
            
            # Check success rate
            total = stats['total_requests']
            successful = stats['successful_requests']
            
            if total > 10:  # Only check if we have enough data
                success_rate = successful / total
                return success_rate > 0.8  # 80% success rate threshold
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking health of worker {worker_id}: {e}")
            return False
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics"""
        
        stats = {
            'total_workers': len(self.workers),
            'healthy_workers': sum(1 for w in self.workers.values() if w['is_healthy']),
            'workers': {}
        }
        
        for worker_id, worker_info in self.workers.items():
            worker_stats = self.worker_stats[worker_id].copy()
            circuit_breaker = self.circuit_breakers[worker_id]
            
            stats['workers'][worker_id] = {
                'endpoint': worker_info['endpoint'],
                'weight': worker_info['weight'],
                'is_healthy': worker_info['is_healthy'],
                'current_connections': worker_info['current_connections'],
                'max_connections': worker_info['max_connections'],
                'circuit_breaker_state': circuit_breaker['state'],
                'circuit_breaker_failures': circuit_breaker['failure_count'],
                **worker_stats
            }
        
        return stats


class MLPipelineOptimizer:
    """
    Main ML pipeline optimization system
    """
    
    def __init__(self,
                 storage_path: str = "/app/pipeline_optimization",
                 enable_caching: bool = True,
                 enable_load_balancing: bool = False):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.artifact_manager = ModelArtifactManager(str(self.storage_path / "artifacts"))
        
        self.cache = None
        if enable_caching:
            self.cache = InferenceCache()
        
        self.load_balancer = None
        if enable_load_balancing:
            self.load_balancer = LoadBalancer()
            self.load_balancer.start_health_checks()
        
        # Optimization strategies
        self.optimization_strategies: Dict[str, bool] = {
            OptimizationStrategy.CACHING.value: enable_caching,
            OptimizationStrategy.PARALLELIZATION.value: True,
            OptimizationStrategy.QUANTIZATION.value: True,
            OptimizationStrategy.BATCHING.value: True,
            OptimizationStrategy.LOAD_BALANCING.value: enable_load_balancing,
            OptimizationStrategy.PREPROCESSING_CACHE.value: True
        }
        
        # Performance tracking
        self.inference_metrics: List[InferenceMetrics] = []
        self.preprocessing_cache: Dict[str, Any] = {}
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        logger.info("ML Pipeline Optimizer initialized")
    
    async def optimized_inference(self,
                                model_name: str,
                                input_data: np.ndarray,
                                model_version: str = "latest",
                                optimization_level: str = "balanced",
                                batch_size: int = None,
                                use_cache: bool = True) -> Tuple[np.ndarray, InferenceMetrics]:
        """
        Perform optimized model inference
        
        Args:
            model_name: Name of the model
            input_data: Input data for inference
            model_version: Model version to use
            optimization_level: Level of optimization (fast, balanced, accurate)
            batch_size: Batch size for inference
            use_cache: Whether to use caching
            
        Returns:
            Predictions and inference metrics
        """
        
        start_time = time.time()
        
        # Generate cache key
        cache_key = None
        if use_cache and self.cache:
            cache_key = self.cache.generate_cache_key(model_name, input_data)
            
            # Check cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                # Return cached result with minimal metrics
                inference_time = (time.time() - start_time) * 1000
                
                metrics = InferenceMetrics(
                    model_name=model_name,
                    timestamp=datetime.utcnow(),
                    batch_size=len(input_data),
                    inference_time_ms=inference_time,
                    preprocessing_time_ms=0,
                    postprocessing_time_ms=0,
                    total_time_ms=inference_time,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    gpu_usage_percent=None,
                    throughput_samples_per_sec=len(input_data) / (inference_time / 1000),
                    cache_hit_ratio=1.0
                )
                
                return cached_result, metrics
        
        # Preprocessing
        preprocessing_start = time.time()
        processed_data = await self._optimized_preprocessing(input_data, model_name)
        preprocessing_time = (time.time() - preprocessing_start) * 1000
        
        # Model inference
        inference_start = time.time()
        
        # Select optimization strategy
        model_artifact_id = f"{model_name}_{model_version}_pytorch"  # Default format
        optimization = self._select_optimization(optimization_level)
        
        # Load optimized model
        model = self.artifact_manager.load_artifact(model_artifact_id, optimization)
        if model is None:
            raise ValueError(f"Could not load model {model_name}")
        
        # Batch processing if needed
        if batch_size and len(processed_data) > batch_size:
            predictions = await self._batch_inference(model, processed_data, batch_size)
        else:
            predictions = await self._single_inference(model, processed_data)
        
        inference_time = (time.time() - inference_start) * 1000
        
        # Postprocessing
        postprocessing_start = time.time()
        final_predictions = await self._optimized_postprocessing(predictions, model_name)
        postprocessing_time = (time.time() - postprocessing_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # Cache result
        if use_cache and self.cache and cache_key:
            self.cache.set(cache_key, final_predictions)
        
        # System metrics
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_usage = psutil.cpu_percent()
        
        # Create metrics
        metrics = InferenceMetrics(
            model_name=model_name,
            timestamp=datetime.utcnow(),
            batch_size=len(input_data),
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocessing_time,
            postprocessing_time_ms=postprocessing_time,
            total_time_ms=total_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_usage_percent=self._get_gpu_usage(),
            throughput_samples_per_sec=len(input_data) / (total_time / 1000),
            cache_hit_ratio=self.cache.get_stats()['hit_ratio'] if self.cache else 0.0
        )
        
        # Store metrics
        self.inference_metrics.append(metrics)
        
        return final_predictions, metrics
    
    async def _optimized_preprocessing(self, 
                                     input_data: np.ndarray,
                                     model_name: str) -> np.ndarray:
        """Optimized preprocessing with caching"""
        
        if not self.optimization_strategies[OptimizationStrategy.PREPROCESSING_CACHE.value]:
            return self._basic_preprocessing(input_data)
        
        # Generate preprocessing cache key
        data_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        cache_key = f"preprocessing:{model_name}:{data_hash}"
        
        if cache_key in self.preprocessing_cache:
            return self.preprocessing_cache[cache_key]
        
        # Perform preprocessing
        processed_data = self._basic_preprocessing(input_data)
        
        # Cache result
        self.preprocessing_cache[cache_key] = processed_data
        
        # Limit cache size
        if len(self.preprocessing_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.preprocessing_cache.keys())[:100]
            for key in oldest_keys:
                del self.preprocessing_cache[key]
        
        return processed_data
    
    def _basic_preprocessing(self, input_data: np.ndarray) -> np.ndarray:
        """Basic preprocessing operations"""
        
        # Standard scaling (example)
        if len(input_data.shape) == 2:
            mean = np.mean(input_data, axis=0)
            std = np.std(input_data, axis=0)
            return (input_data - mean) / (std + 1e-8)
        
        return input_data
    
    async def _single_inference(self, model: Any, input_data: np.ndarray) -> np.ndarray:
        """Single model inference"""
        
        # Handle different model types
        if hasattr(model, 'predict'):
            # Sklearn-like model
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, model.predict, input_data)
            
        elif hasattr(model, 'forward'):
            # PyTorch model
            import torch
            model.eval()
            with torch.no_grad():
                tensor_input = torch.FloatTensor(input_data)
                output = model(tensor_input)
                return output.numpy()
        
        else:
            # Generic prediction
            return np.array([0.0] * len(input_data))
    
    async def _batch_inference(self, 
                             model: Any, 
                             input_data: np.ndarray, 
                             batch_size: int) -> np.ndarray:
        """Batch model inference for large inputs"""
        
        num_samples = len(input_data)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        batch_results = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_data = input_data[start_idx:end_idx]
            batch_prediction = await self._single_inference(model, batch_data)
            batch_results.append(batch_prediction)
        
        return np.concatenate(batch_results)
    
    async def _optimized_postprocessing(self, 
                                      predictions: np.ndarray,
                                      model_name: str) -> np.ndarray:
        """Optimized postprocessing"""
        
        # Apply model-specific postprocessing
        if 'classification' in model_name.lower():
            # Apply softmax for classification
            exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        return predictions
    
    def _select_optimization(self, optimization_level: str) -> str:
        """Select optimization strategy based on level"""
        
        if optimization_level == "fast":
            return "quantized"  # Fastest, potentially less accurate
        elif optimization_level == "accurate":
            return "original"   # Most accurate, slower
        else:  # balanced
            return "onnx"      # Good balance of speed and accuracy
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(info.gpu)
        except:
            return None
    
    async def parallel_inference(self,
                               models: Dict[str, str],
                               input_data: np.ndarray,
                               combine_strategy: str = "average") -> Dict[str, np.ndarray]:
        """Run inference on multiple models in parallel"""
        
        if not self.optimization_strategies[OptimizationStrategy.PARALLELIZATION.value]:
            # Sequential execution
            results = {}
            for model_name, model_version in models.items():
                pred, _ = await self.optimized_inference(model_name, input_data, model_version)
                results[model_name] = pred
            return results
        
        # Parallel execution
        tasks = []
        for model_name, model_version in models.items():
            task = asyncio.create_task(
                self.optimized_inference(model_name, input_data, model_version)
            )
            tasks.append((model_name, task))
        
        results = {}
        for model_name, task in tasks:
            pred, _ = await task
            results[model_name] = pred
        
        return results
    
    def register_model_for_optimization(self,
                                      model_name: str,
                                      model_version: str,
                                      model_object: Any,
                                      model_format: ModelFormat) -> str:
        """Register model and create optimized versions"""
        
        return self.artifact_manager.store_artifact(
            model_name=model_name,
            model_version=model_version,
            model_object=model_object,
            model_format=model_format
        )
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'optimization_strategies': self.optimization_strategies,
            'cache_stats': self.cache.get_stats() if self.cache else {},
            'load_balancer_stats': self.load_balancer.get_worker_stats() if self.load_balancer else {},
            'recent_inference_metrics': [],
            'performance_summary': {},
            'artifact_summary': {}
        }
        
        # Recent inference metrics
        recent_metrics = self.inference_metrics[-100:]  # Last 100 inferences
        report['recent_inference_metrics'] = [m.to_dict() for m in recent_metrics]
        
        # Performance summary
        if recent_metrics:
            report['performance_summary'] = {
                'average_inference_time_ms': np.mean([m.inference_time_ms for m in recent_metrics]),
                'average_total_time_ms': np.mean([m.total_time_ms for m in recent_metrics]),
                'average_throughput': np.mean([m.throughput_samples_per_sec for m in recent_metrics]),
                'p95_inference_time_ms': np.percentile([m.inference_time_ms for m in recent_metrics], 95),
                'average_memory_usage_mb': np.mean([m.memory_usage_mb for m in recent_metrics]),
                'average_cpu_usage': np.mean([m.cpu_usage_percent for m in recent_metrics])
            }
        
        # Artifact summary
        report['artifact_summary'] = {
            'total_artifacts': len(self.artifact_manager.artifacts),
            'total_storage_mb': sum(
                artifact.get('original_size_mb', 0) 
                for artifact in self.artifact_manager.artifacts.values()
            )
        }
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        
        if self.load_balancer:
            self.load_balancer.stop_health_checks()
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("ML Pipeline Optimizer cleaned up")


# Global optimizer instance
_pipeline_optimizer: Optional[MLPipelineOptimizer] = None

def get_pipeline_optimizer() -> MLPipelineOptimizer:
    """Get global pipeline optimizer instance"""
    global _pipeline_optimizer
    if _pipeline_optimizer is None:
        _pipeline_optimizer = MLPipelineOptimizer()
    return _pipeline_optimizer