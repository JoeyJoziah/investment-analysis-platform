"""
Model artifact manager for the ML pipeline optimization system.

Extracted from pipeline_optimization.py.  Contains ModelArtifactManager only.
Import via the original path (backend.ml.pipeline_optimization) or directly from here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from backend.ml.pipeline_types import ModelFormat
except ImportError:  # pragma: no cover
    from pipeline_types import ModelFormat  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class ModelArtifactManager:
    """Manages model artifacts and optimized versions"""

    def __init__(self, storage_path: str = "/app/model_artifacts") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self.load_artifacts_registry()

        self.optimization_cache: Dict[str, Any] = {}

        logger.info(f"Model artifact manager initialized with {len(self.artifacts)} artifacts")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def store_artifact(
        self,
        model_name: str,
        model_version: str,
        model_object: Any,
        model_format: ModelFormat,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store model artifact with optimizations"""

        artifact_id = f"{model_name}_{model_version}_{model_format.value}"
        artifact_dir = self.storage_path / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        original_path = artifact_dir / f"original.{model_format.value}"
        self._save_model_by_format(model_object, original_path, model_format)

        optimized_paths = self._create_optimized_versions(model_object, artifact_dir, model_format)

        artifact_metadata: Dict[str, Any] = {
            'model_name': model_name,
            'model_version': model_version,
            'format': model_format.value,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'original_path': str(original_path),
            'original_size_mb': original_path.stat().st_size / (1024 * 1024),
            'optimized_versions': optimized_paths,
            'metadata': metadata or {},
        }

        self.artifacts[artifact_id] = artifact_metadata
        self._save_artifacts_registry()

        logger.info(f"Stored artifact {artifact_id} with {len(optimized_paths)} optimized versions")
        return artifact_id

    def load_artifact(
        self,
        artifact_id: str,
        optimization: str = "original",
    ) -> Optional[Any]:
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
                    if '.onnx' in str(model_path):
                        model_format = ModelFormat.ONNX
                    elif '.trt' in str(model_path):
                        model_format = ModelFormat.TENSORRT
                    else:
                        model_format = ModelFormat(artifact_info['format'])

            return self._load_model_by_format(model_path, model_format)
        except Exception as exc:
            logger.error(f"Error loading artifact {artifact_id}: {exc}")
            return None

    def load_artifacts_registry(self) -> None:
        """Load artifacts registry from disk"""
        registry_file = self.storage_path / "artifacts_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as fh:
                    self.artifacts = json.load(fh)
            except Exception as exc:
                logger.error(f"Error loading artifacts registry: {exc}")
                self.artifacts = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_artifacts_registry(self) -> None:
        """Save artifacts registry to disk"""
        registry_file = self.storage_path / "artifacts_registry.json"
        try:
            with open(registry_file, 'w') as fh:
                json.dump(self.artifacts, fh, indent=2)
        except Exception as exc:
            logger.error(f"Error saving artifacts registry: {exc}")

    def _save_model_by_format(self, model: Any, path: Path, format: ModelFormat) -> None:
        """Save model in specified format"""

        if format == ModelFormat.PYTORCH:
            if torch is None:
                raise RuntimeError("torch is not installed")
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), path)
            else:
                torch.save(model, path)

        elif format == ModelFormat.SKLEARN_JOBLIB:
            if joblib is None:
                raise RuntimeError("joblib is not installed")
            joblib.dump(model, path)

        elif format in [ModelFormat.XGBOOST, ModelFormat.PICKLE]:
            # SECURITY: Use joblib instead of pickle for safer serialization
            if joblib is None:
                raise RuntimeError("joblib is not installed")
            joblib.dump(model, path)

        else:
            raise ValueError(f"Unsupported format for saving: {format}")

    def _load_model_by_format(self, path: Path, format: ModelFormat) -> Any:
        """Load model in specified format"""

        if format == ModelFormat.PYTORCH:
            if torch is None:
                raise RuntimeError("torch is not installed")
            # F-03-002: torch.load defaults to pickle deserialization which
            # is RCE-equivalent on untrusted artifacts. ``weights_only=True``
            # restricts deserialization to tensor primitives. Full-model
            # artifacts that need non-tensor classes must be allowlisted via
            # ``torch.serialization.add_safe_globals([Cls, ...])`` before
            # the load — see workpaper §9 for the safe_globals escalation
            # path.
            return torch.load(path, map_location='cpu', weights_only=True)

        elif format == ModelFormat.SKLEARN_JOBLIB:
            if joblib is None:
                raise RuntimeError("joblib is not installed")
            return joblib.load(path)

        elif format in [ModelFormat.XGBOOST, ModelFormat.PICKLE]:
            # SECURITY: Use joblib instead of pickle for safer deserialization
            if joblib is None:
                raise RuntimeError("joblib is not installed")
            return joblib.load(path)

        elif format == ModelFormat.ONNX:
            try:
                import onnxruntime as ort
                return ort.InferenceSession(str(path))
            except ImportError:
                logger.error("ONNX runtime not available")
                return None

        else:
            raise ValueError(f"Unsupported format for loading: {format}")

    def _create_optimized_versions(
        self,
        model: Any,
        artifact_dir: Path,
        format: ModelFormat,
    ) -> Dict[str, str]:
        """Create optimized versions of the model"""

        optimized_paths: Dict[str, str] = {}

        try:
            if torch is not None:
                if format == ModelFormat.PYTORCH and hasattr(model, 'eval'):
                    quantized_path = artifact_dir / "quantized.pt"
                    quantized_model = self._quantize_pytorch_model(model)
                    if quantized_model is not None:
                        torch.save(quantized_model, quantized_path)
                        optimized_paths['quantized'] = str(quantized_path)

                if format == ModelFormat.PYTORCH:
                    onnx_path = artifact_dir / "model.onnx"
                    if self._export_to_onnx(model, onnx_path):
                        optimized_paths['onnx'] = str(onnx_path)

                        tensorrt_path = artifact_dir / "model.trt"
                        if self._optimize_with_tensorrt(onnx_path, tensorrt_path):
                            optimized_paths['tensorrt'] = str(tensorrt_path)

            if format in [ModelFormat.SKLEARN_JOBLIB, ModelFormat.XGBOOST]:
                compressed_path = artifact_dir / f"compressed.{format.value}"
                if self._compress_model(model, compressed_path):
                    optimized_paths['compressed'] = str(compressed_path)

        except Exception as exc:
            logger.error(f"Error creating optimized versions: {exc}")

        return optimized_paths

    def _quantize_pytorch_model(
        self, model: Any
    ) -> Optional[Any]:
        """Apply quantization to PyTorch model"""
        try:
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8,
            )
            return quantized_model
        except Exception as exc:
            logger.error(f"Error quantizing PyTorch model: {exc}")
            return None

    def _export_to_onnx(self, model: Any, onnx_path: Path) -> bool:
        """Export PyTorch model to ONNX"""
        try:
            model.eval()
            dummy_input = torch.randn(1, 10)
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            )
            return True
        except Exception as exc:
            logger.error(f"Error exporting to ONNX: {exc}")
            return False

    def _optimize_with_tensorrt(self, onnx_path: Path, tensorrt_path: Path) -> bool:
        """Optimize ONNX model with TensorRT"""
        try:
            logger.info("TensorRT optimization not implemented (requires TensorRT SDK)")
            return False
        except Exception as exc:
            logger.error(f"Error optimizing with TensorRT: {exc}")
            return False

    def _compress_model(self, model: Any, compressed_path: Path) -> bool:
        """Compress model using various techniques"""
        try:
            if hasattr(model, 'predict') and joblib is not None:
                joblib.dump(model, compressed_path, compress=3)
                return True
            return False
        except Exception as exc:
            logger.error(f"Error compressing model: {exc}")
            return False


__all__ = ["ModelArtifactManager"]
