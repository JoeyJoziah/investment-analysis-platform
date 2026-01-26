"""
Hugging Face Hub Client for Model Storage
Provides upload/download capabilities with caching and versioning for ML models.

This module handles:
- Model upload to HuggingFace Hub
- Model download with local caching
- Version management and registry synchronization
- Private repository support
"""

import os
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class HFModelInfo:
    """Model information for HF Hub operations"""
    model_name: str
    version: str
    model_type: str  # lstm, xgboost, prophet
    local_path: str
    hf_path: str
    uploaded_at: Optional[str] = None
    commit_hash: Optional[str] = None
    sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HFModelInfo':
        """Create from dictionary"""
        return cls(**data)


class HuggingFaceHubClient:
    """
    Client for Hugging Face Hub model storage operations.
    Supports upload, download, versioning, and caching.
    """

    def __init__(
        self,
        repo_id: str = None,
        token: str = None,
        cache_dir: str = None,
        auto_create_repo: bool = True,
        private: bool = True
    ):
        """
        Initialize HuggingFace Hub client.

        Args:
            repo_id: Repository ID (e.g., "username/repo-name")
            token: HuggingFace API token
            cache_dir: Local cache directory for downloaded models
            auto_create_repo: Create repository if it doesn't exist
            private: Whether to create private repositories
        """
        self.token = token or os.getenv("HF_TOKEN")
        self.repo_id = repo_id or os.getenv("HF_MODEL_REPO", "investment-analysis-models")

        # Handle cache directory - use relative path for local dev, /app for Docker
        cache_dir_env = cache_dir or os.getenv("HF_HOME", "./ml_models/.hf_cache")
        # Resolve relative paths from current working directory
        if cache_dir_env.startswith("./"):
            self.cache_dir = Path.cwd() / cache_dir_env[2:]
        elif cache_dir_env.startswith("/app") and not Path("/app").exists():
            # Docker path but not in Docker - use local equivalent
            self.cache_dir = Path.cwd() / "ml_models" / ".hf_cache"
        else:
            self.cache_dir = Path(cache_dir_env)

        self.auto_create_repo = auto_create_repo
        self.private = private if os.getenv("HF_PRIVATE_REPOS", "true").lower() == "true" else False
        self.lock = threading.Lock()

        # Lazy import huggingface_hub to avoid import errors if not installed
        self._api = None
        self._hf_hub_available = None

        # Ensure cache directory exists
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Fallback to a temp directory if cache_dir creation fails
            import tempfile
            self.cache_dir = Path(tempfile.gettempdir()) / "hf_models_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using fallback cache directory: {self.cache_dir}")

        # Initialize repository if needed
        if auto_create_repo and self._check_hf_hub_available():
            self._ensure_repo_exists()

        logger.info(f"HuggingFace Hub client initialized for repo: {self.repo_id}")

    def _check_hf_hub_available(self) -> bool:
        """Check if huggingface_hub library is available"""
        if self._hf_hub_available is not None:
            return self._hf_hub_available

        try:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
            self._hf_hub_available = True
            return True
        except ImportError:
            logger.warning("huggingface_hub not installed. HF Hub features disabled.")
            self._hf_hub_available = False
            return False

    @property
    def api(self):
        """Get HuggingFace API client (lazy loading)"""
        if self._api is None:
            if self._check_hf_hub_available():
                from huggingface_hub import HfApi
                self._api = HfApi(token=self.token)
            else:
                raise ImportError("huggingface_hub is not installed")
        return self._api

    def _ensure_repo_exists(self) -> bool:
        """Create repository if it doesn't exist"""
        if not self._check_hf_hub_available():
            return False

        try:
            self.api.repo_info(repo_id=self.repo_id, token=self.token)
            logger.debug(f"Repository {self.repo_id} exists")
            return True
        except Exception:
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=self.private,
                    repo_type="model"
                )
                logger.info(f"Created HF Hub repository: {self.repo_id} (private={self.private})")

                # Create initial README
                self._create_model_card()
                return True
            except Exception as e:
                logger.error(f"Failed to create repository {self.repo_id}: {e}")
                return False

    def _create_model_card(self):
        """Create initial model card README for the repository"""
        readme_content = """---
language:
- en
tags:
- finance
- investment
- stock-prediction
- time-series
license: mit
---

# Investment Analysis ML Models

This repository contains machine learning models for the Investment Analysis Platform.

## Models

| Model | Type | Description |
|-------|------|-------------|
| LSTM | PyTorch | Deep learning model for price prediction |
| XGBoost | sklearn | Gradient boosting classifier for direction prediction |
| Prophet | Meta | Time-series forecasting for individual stocks |

## Usage

These models are designed to be used with the Investment Analysis Platform.
They are automatically downloaded and loaded during inference.

## Versioning

Models are versioned using semantic versioning (major.minor.patch).
Each version includes:
- Model weights/artifacts
- Scaler (if applicable)
- Configuration JSON
- Training results

## Privacy

This is a private repository for internal use.
"""
        try:
            from huggingface_hub import upload_file
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(readme_content)
                temp_path = f.name

            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="README.md",
                repo_id=self.repo_id,
                token=self.token,
                commit_message="Initialize repository with model card"
            )

            os.unlink(temp_path)
            logger.info("Created model card README")
        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")

    def upload_model(
        self,
        model_name: str,
        version: str,
        local_dir: Path,
        commit_message: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[HFModelInfo]:
        """
        Upload model files to HuggingFace Hub.

        Args:
            model_name: Name of the model (lstm, xgboost, prophet)
            version: Semantic version (e.g., "1.0.0")
            local_dir: Local directory containing model files
            commit_message: Optional commit message
            metadata: Optional metadata to include

        Returns:
            HFModelInfo if upload successful, None otherwise
        """
        if not self._check_hf_hub_available():
            logger.error("HuggingFace Hub not available")
            return None

        try:
            from huggingface_hub import upload_folder

            local_dir = Path(local_dir)
            if not local_dir.exists():
                logger.error(f"Local directory does not exist: {local_dir}")
                return None

            hf_path = f"models/{model_name}/v{version}"
            commit_msg = commit_message or f"Upload {model_name} v{version}"

            # Add metadata file if provided
            if metadata:
                metadata_path = local_dir / "hf_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            # Upload entire folder
            with self.lock:
                self.api.upload_folder(
                    folder_path=str(local_dir),
                    path_in_repo=hf_path,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=commit_msg
                )

            # Get commit hash
            try:
                commits = list(self.api.list_repo_commits(self.repo_id))
                commit_hash = commits[0].commit_id if commits else None
            except Exception:
                commit_hash = None

            model_info = HFModelInfo(
                model_name=model_name,
                version=version,
                model_type=model_name,
                local_path=str(local_dir),
                hf_path=hf_path,
                uploaded_at=datetime.utcnow().isoformat(),
                commit_hash=commit_hash
            )

            logger.info(f"Uploaded {model_name} v{version} to {self.repo_id}/{hf_path}")
            return model_info

        except Exception as e:
            logger.error(f"Failed to upload {model_name} v{version}: {e}")
            return None

    def download_model(
        self,
        model_name: str,
        version: str = "latest",
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download model from HuggingFace Hub with caching.

        Args:
            model_name: Name of the model
            version: Version to download ("latest" resolves to newest)
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded model directory, or None if failed
        """
        if not self._check_hf_hub_available():
            logger.error("HuggingFace Hub not available")
            return None

        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            # Resolve "latest" to actual version
            if version == "latest":
                version = self._get_latest_version(model_name)
                if not version:
                    logger.error(f"No versions found for {model_name}")
                    return None

            hf_path = f"models/{model_name}/v{version}"
            local_path = self.cache_dir / model_name / f"v{version}"

            # Check cache
            if not force_download and local_path.exists() and any(local_path.iterdir()):
                logger.info(f"Using cached {model_name} v{version} from {local_path}")
                return local_path

            # Download files
            local_path.mkdir(parents=True, exist_ok=True)
            files = self._list_model_files(model_name, version)

            if not files:
                logger.error(f"No files found for {model_name} v{version}")
                return None

            for file_path in files:
                try:
                    downloaded = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=file_path,
                        local_dir=str(self.cache_dir),
                        token=self.token,
                        force_download=force_download
                    )
                    logger.debug(f"Downloaded {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to download {file_path}: {e}")

            logger.info(f"Downloaded {model_name} v{version} to {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download {model_name} v{version}: {e}")
            return None

    def _get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version for a model"""
        if not self._check_hf_hub_available():
            return None

        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(repo_id=self.repo_id, token=self.token)
            prefix = f"models/{model_name}/v"

            versions = set()
            for f in files:
                if f.startswith(prefix):
                    # Extract version from path
                    parts = f[len(prefix):].split("/")
                    if parts:
                        versions.add(parts[0])

            if not versions:
                return None

            # Sort by semantic version
            versions = list(versions)
            versions.sort(key=lambda x: tuple(map(int, x.split("."))), reverse=True)
            return versions[0]

        except Exception as e:
            logger.error(f"Failed to get latest version for {model_name}: {e}")
            return None

    def _list_model_files(self, model_name: str, version: str) -> List[str]:
        """List all files for a specific model version"""
        if not self._check_hf_hub_available():
            return []

        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(repo_id=self.repo_id, token=self.token)
            prefix = f"models/{model_name}/v{version}/"
            return [f for f in files if f.startswith(prefix)]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def list_versions(self, model_name: str) -> List[str]:
        """List all available versions for a model"""
        if not self._check_hf_hub_available():
            return []

        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(repo_id=self.repo_id, token=self.token)
            prefix = f"models/{model_name}/v"

            versions = set()
            for f in files:
                if f.startswith(prefix):
                    parts = f[len(prefix):].split("/")
                    if parts:
                        versions.add(parts[0])

            # Sort by semantic version (newest first)
            versions = list(versions)
            versions.sort(key=lambda x: tuple(map(int, x.split("."))), reverse=True)
            return versions

        except Exception as e:
            logger.error(f"Failed to list versions for {model_name}: {e}")
            return []

    def list_models(self) -> List[str]:
        """List all models in the repository"""
        if not self._check_hf_hub_available():
            return []

        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(repo_id=self.repo_id, token=self.token)
            prefix = "models/"

            models = set()
            for f in files:
                if f.startswith(prefix):
                    # Extract model name from path
                    parts = f[len(prefix):].split("/")
                    if parts:
                        models.add(parts[0])

            return sorted(list(models))

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def sync_registry(self, local_registry_path: Path) -> bool:
        """
        Sync local model registry with HF Hub.

        Args:
            local_registry_path: Path to local registry.json

        Returns:
            True if sync successful
        """
        if not self._check_hf_hub_available():
            return False

        try:
            from huggingface_hub import upload_file

            self.api.upload_file(
                path_or_fileobj=str(local_registry_path),
                path_in_repo="registry.json",
                repo_id=self.repo_id,
                token=self.token,
                commit_message="Sync model registry"
            )
            logger.info("Registry synced to HF Hub")
            return True
        except Exception as e:
            logger.error(f"Failed to sync registry: {e}")
            return False

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version from the Hub"""
        if not self._check_hf_hub_available():
            return False

        try:
            hf_path = f"models/{model_name}/v{version}"
            files = self._list_model_files(model_name, version)

            for file_path in files:
                self.api.delete_file(
                    path_in_repo=file_path,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"Delete {model_name} v{version}"
                )

            logger.info(f"Deleted {model_name} v{version} from HF Hub")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {model_name} v{version}: {e}")
            return False

    def get_model_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model version"""
        if not self._check_hf_hub_available():
            return None

        try:
            from huggingface_hub import hf_hub_download

            # Try to download hf_metadata.json
            metadata_path = f"models/{model_name}/v{version}/hf_metadata.json"

            try:
                local_file = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=metadata_path,
                    token=self.token
                )
                with open(local_file, 'r') as f:
                    return json.load(f)
            except Exception:
                # No metadata file, return basic info
                return {
                    "model_name": model_name,
                    "version": version,
                    "files": self._list_model_files(model_name, version)
                }

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None


# Singleton instance for easy access
_hf_client: Optional[HuggingFaceHubClient] = None


def get_hf_hub_client(
    repo_id: str = None,
    token: str = None,
    cache_dir: str = None
) -> HuggingFaceHubClient:
    """
    Get or create the singleton HuggingFace Hub client.

    Args:
        repo_id: Override default repository ID
        token: Override default token
        cache_dir: Override default cache directory

    Returns:
        HuggingFaceHubClient instance
    """
    global _hf_client

    # Check if HF Hub is enabled
    if os.getenv("HF_HUB_ENABLED", "true").lower() != "true":
        logger.info("HuggingFace Hub disabled via HF_HUB_ENABLED=false")
        return None

    if _hf_client is None:
        _hf_client = HuggingFaceHubClient(
            repo_id=repo_id,
            token=token,
            cache_dir=cache_dir
        )

    return _hf_client
