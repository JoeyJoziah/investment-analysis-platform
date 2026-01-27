"""
GPU Utilities for ML Training

Provides centralized GPU detection and configuration for all ML models.
Supports PyTorch, XGBoost, and LightGBM with graceful CPU fallback.

Usage:
    from backend.ml.gpu_utils import GPUConfig, get_gpu_config

    config = get_gpu_config()
    if config.cuda_available:
        # Use GPU-accelerated training
        ...
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """Configuration for GPU-accelerated ML training."""

    # PyTorch CUDA support
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: str = ""
    cuda_memory_total_gb: float = 0.0
    cuda_memory_free_gb: float = 0.0

    # XGBoost GPU support
    xgboost_gpu_available: bool = False
    xgboost_tree_method: str = "hist"  # 'hist' for CPU, 'gpu_hist' for GPU
    xgboost_device: str = "cpu"  # 'cpu' or 'cuda'

    # LightGBM GPU support
    lightgbm_gpu_available: bool = False
    lightgbm_device: str = "cpu"  # 'cpu' or 'gpu'
    lightgbm_gpu_platform_id: int = 0
    lightgbm_gpu_device_id: int = 0

    # General GPU info
    gpu_brand: str = ""
    compute_capability: str = ""

    # Feature flags
    use_mixed_precision: bool = False
    gpu_memory_fraction: float = 0.9  # Use 90% of GPU memory max

    def get_pytorch_device(self) -> str:
        """Get the PyTorch device string."""
        return "cuda" if self.cuda_available else "cpu"

    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost GPU-related parameters."""
        params = {
            "tree_method": self.xgboost_tree_method,
            "device": self.xgboost_device,
        }
        if self.xgboost_gpu_available:
            # Additional GPU-specific params for XGBoost 2.0+
            params["max_bin"] = 256  # GPU hist uses binned data
        return params

    def get_lightgbm_params(self) -> Dict[str, Any]:
        """Get LightGBM GPU-related parameters."""
        params = {
            "device": self.lightgbm_device,
        }
        if self.lightgbm_gpu_available:
            params["gpu_platform_id"] = self.lightgbm_gpu_platform_id
            params["gpu_device_id"] = self.lightgbm_gpu_device_id
        return params

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "cuda_memory_total_gb": self.cuda_memory_total_gb,
            "cuda_memory_free_gb": self.cuda_memory_free_gb,
            "xgboost_gpu_available": self.xgboost_gpu_available,
            "xgboost_tree_method": self.xgboost_tree_method,
            "xgboost_device": self.xgboost_device,
            "lightgbm_gpu_available": self.lightgbm_gpu_available,
            "lightgbm_device": self.lightgbm_device,
            "gpu_brand": self.gpu_brand,
            "compute_capability": self.compute_capability,
            "use_mixed_precision": self.use_mixed_precision,
        }


def detect_cuda() -> Dict[str, Any]:
    """Detect CUDA availability for PyTorch."""
    result = {
        "available": False,
        "device_count": 0,
        "device_name": "",
        "memory_total_gb": 0.0,
        "memory_free_gb": 0.0,
        "compute_capability": "",
    }

    try:
        import torch

        if torch.cuda.is_available():
            result["available"] = True
            result["device_count"] = torch.cuda.device_count()

            if result["device_count"] > 0:
                result["device_name"] = torch.cuda.get_device_name(0)

                # Get memory info
                total_memory = torch.cuda.get_device_properties(0).total_memory
                result["memory_total_gb"] = total_memory / (1024 ** 3)

                # Get free memory (requires allocation context)
                try:
                    free_memory, total = torch.cuda.mem_get_info(0)
                    result["memory_free_gb"] = free_memory / (1024 ** 3)
                except Exception:
                    result["memory_free_gb"] = result["memory_total_gb"] * 0.9  # Estimate

                # Get compute capability
                major, minor = torch.cuda.get_device_capability(0)
                result["compute_capability"] = f"{major}.{minor}"

                logger.info(f"CUDA detected: {result['device_name']} "
                           f"({result['memory_total_gb']:.1f}GB, "
                           f"compute capability {result['compute_capability']})")
    except ImportError:
        logger.debug("PyTorch not available, CUDA detection skipped")
    except Exception as e:
        logger.warning(f"Error detecting CUDA: {e}")

    return result


def detect_xgboost_gpu() -> Dict[str, Any]:
    """Detect GPU support for XGBoost."""
    result = {
        "available": False,
        "tree_method": "hist",
        "device": "cpu",
    }

    # First check if CUDA is available via PyTorch (most reliable)
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    if not cuda_available:
        logger.debug("CUDA not available (checked via PyTorch), XGBoost GPU disabled")
        return result

    try:
        import xgboost as xgb

        # Check XGBoost version for GPU support method
        xgb_version = tuple(int(x) for x in xgb.__version__.split('.')[:2])

        # Try to create a small GPU model to test availability
        try:
            # XGBoost 2.0+ uses 'device' parameter
            if xgb_version >= (2, 0):
                test_params = {
                    "tree_method": "hist",
                    "device": "cuda",
                    "n_estimators": 1,
                    "max_depth": 1,
                }
                model = xgb.XGBRegressor(**test_params)

                # Quick test with minimal data
                import numpy as np
                X_test = np.random.randn(10, 2)
                y_test = np.random.randn(10)
                model.fit(X_test, y_test, verbose=False)

                result["available"] = True
                result["tree_method"] = "hist"  # hist works for both CPU and GPU in 2.0+
                result["device"] = "cuda"
                logger.info("XGBoost GPU support detected (XGBoost 2.0+ with CUDA)")

            else:
                # XGBoost < 2.0 uses gpu_hist tree_method
                test_params = {
                    "tree_method": "gpu_hist",
                    "n_estimators": 1,
                    "max_depth": 1,
                }
                model = xgb.XGBRegressor(**test_params)

                import numpy as np
                X_test = np.random.randn(10, 2)
                y_test = np.random.randn(10)
                model.fit(X_test, y_test, verbose=False)

                result["available"] = True
                result["tree_method"] = "gpu_hist"
                result["device"] = "gpu"  # Legacy parameter
                logger.info("XGBoost GPU support detected (XGBoost < 2.0 with gpu_hist)")

        except Exception as e:
            logger.debug(f"XGBoost GPU not available: {e}")

    except ImportError:
        logger.debug("XGBoost not available")
    except Exception as e:
        logger.warning(f"Error detecting XGBoost GPU: {e}")

    return result


def detect_lightgbm_gpu() -> Dict[str, Any]:
    """Detect GPU support for LightGBM."""
    result = {
        "available": False,
        "device": "cpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }

    try:
        import lightgbm as lgb

        # Try to create a small GPU model to test availability
        try:
            test_params = {
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
                "n_estimators": 1,
                "max_depth": 1,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**test_params)

            import numpy as np
            X_test = np.random.randn(10, 2)
            y_test = np.random.randn(10)
            model.fit(X_test, y_test)

            result["available"] = True
            result["device"] = "gpu"
            logger.info("LightGBM GPU support detected")

        except Exception as e:
            logger.debug(f"LightGBM GPU not available: {e}")

    except ImportError:
        logger.debug("LightGBM not available")
    except Exception as e:
        logger.warning(f"Error detecting LightGBM GPU: {e}")

    return result


def get_gpu_config(
    force_cpu: bool = False,
    use_mixed_precision: bool = False,
    gpu_memory_fraction: float = 0.9
) -> GPUConfig:
    """
    Get GPU configuration for ML training.

    Args:
        force_cpu: Force CPU usage even if GPU is available
        use_mixed_precision: Enable mixed precision training for PyTorch
        gpu_memory_fraction: Maximum fraction of GPU memory to use

    Returns:
        GPUConfig with detected capabilities
    """
    config = GPUConfig(
        use_mixed_precision=use_mixed_precision,
        gpu_memory_fraction=gpu_memory_fraction,
    )

    # Check environment variable for forcing CPU
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        force_cpu = True

    if force_cpu:
        logger.info("GPU disabled by configuration, using CPU")
        return config

    # Detect CUDA for PyTorch
    cuda_info = detect_cuda()
    if cuda_info["available"]:
        config.cuda_available = True
        config.cuda_device_count = cuda_info["device_count"]
        config.cuda_device_name = cuda_info["device_name"]
        config.cuda_memory_total_gb = cuda_info["memory_total_gb"]
        config.cuda_memory_free_gb = cuda_info["memory_free_gb"]
        config.compute_capability = cuda_info["compute_capability"]

        # Determine GPU brand
        device_name = cuda_info["device_name"].lower()
        if "nvidia" in device_name or "geforce" in device_name or "rtx" in device_name or "gtx" in device_name:
            config.gpu_brand = "NVIDIA"
        elif "amd" in device_name or "radeon" in device_name:
            config.gpu_brand = "AMD"
        else:
            config.gpu_brand = "Unknown"

    # Detect XGBoost GPU
    xgb_info = detect_xgboost_gpu()
    if xgb_info["available"]:
        config.xgboost_gpu_available = True
        config.xgboost_tree_method = xgb_info["tree_method"]
        config.xgboost_device = xgb_info["device"]

    # Detect LightGBM GPU
    lgb_info = detect_lightgbm_gpu()
    if lgb_info["available"]:
        config.lightgbm_gpu_available = True
        config.lightgbm_device = lgb_info["device"]
        config.lightgbm_gpu_platform_id = lgb_info["gpu_platform_id"]
        config.lightgbm_gpu_device_id = lgb_info["gpu_device_id"]

    # Log summary
    gpu_summary = []
    if config.cuda_available:
        gpu_summary.append(f"PyTorch/CUDA ({config.cuda_device_name})")
    if config.xgboost_gpu_available:
        gpu_summary.append("XGBoost GPU")
    if config.lightgbm_gpu_available:
        gpu_summary.append("LightGBM GPU")

    if gpu_summary:
        logger.info(f"GPU support available: {', '.join(gpu_summary)}")
    else:
        logger.info("No GPU support detected, using CPU for all models")

    return config


def log_gpu_memory_usage(prefix: str = ""):
    """Log current GPU memory usage (for debugging)."""
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

                logger.info(
                    f"{prefix}GPU {i} Memory: "
                    f"Allocated: {allocated:.2f}GB, "
                    f"Reserved: {reserved:.2f}GB, "
                    f"Total: {total:.2f}GB"
                )
    except Exception as e:
        logger.debug(f"Could not log GPU memory: {e}")


def clear_gpu_memory():
    """Clear GPU memory cache (useful between training runs)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory cache")
    except Exception as e:
        logger.debug(f"Could not clear GPU memory: {e}")


def set_gpu_memory_fraction(fraction: float = 0.9):
    """
    Set the maximum GPU memory fraction PyTorch should use.

    Args:
        fraction: Fraction of GPU memory to use (0.0 to 1.0)
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Note: This should be called before any GPU operations
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"Set GPU memory fraction to {fraction:.0%}")
    except Exception as e:
        logger.debug(f"Could not set GPU memory fraction: {e}")


# Singleton instance for caching GPU config
_cached_config: Optional[GPUConfig] = None


def get_cached_gpu_config(refresh: bool = False, **kwargs) -> GPUConfig:
    """
    Get cached GPU configuration (singleton pattern).

    Args:
        refresh: Force refresh of cached config
        **kwargs: Additional arguments passed to get_gpu_config

    Returns:
        Cached GPUConfig instance
    """
    global _cached_config

    if _cached_config is None or refresh:
        _cached_config = get_gpu_config(**kwargs)

    return _cached_config
