"""
Hugging Face Datasets Hub Integration
Manages versioned ML training datasets on HF Hub.

This module handles:
- Dataset upload to HuggingFace Datasets Hub
- Dataset download with local caching
- Version management with semantic versioning
- Metadata tracking for reproducibility
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    """Dataset version metadata"""
    version: str
    commit_hash: str
    created_at: str
    total_samples: int
    stock_count: int
    feature_count: int
    date_range_start: str
    date_range_end: str
    train_samples: int
    val_samples: int
    test_samples: int
    feature_columns: List[str]
    label_columns: List[str]
    data_hash: str
    parent_version: Optional[str] = None
    model_versions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create from dictionary"""
        return cls(**data)


class HuggingFaceDatasetManager:
    """Manages ML datasets on Hugging Face Hub"""

    def __init__(
        self,
        repo_id: str = None,
        token: str = None,
        local_cache_dir: str = "data/ml_training",
        auto_create_repo: bool = True,
        private: bool = True
    ):
        """
        Initialize HuggingFace Dataset Manager.

        Args:
            repo_id: Repository ID (e.g., "username/dataset-name")
            token: HuggingFace API token
            local_cache_dir: Local directory for dataset caching
            auto_create_repo: Create repository if it doesn't exist
            private: Whether to create private repositories
        """
        self.repo_id = repo_id or os.getenv("HF_DATASET_REPO", "investment-ml-datasets")
        self.token = token or os.getenv("HF_TOKEN")

        # Handle cache directory - use relative path for local dev, /app for Docker
        if local_cache_dir.startswith("./"):
            self.local_cache_dir = Path.cwd() / local_cache_dir[2:]
        elif local_cache_dir.startswith("/app") and not Path("/app").exists():
            # Docker path but not in Docker - use local equivalent
            self.local_cache_dir = Path.cwd() / "data" / "ml_training"
        else:
            self.local_cache_dir = Path(local_cache_dir)

        self.auto_create_repo = auto_create_repo
        self.private = private if os.getenv("HF_PRIVATE_REPOS", "true").lower() == "true" else False
        self.lock = threading.Lock()

        # Lazy imports
        self._api = None
        self._hf_available = None

        # Ensure cache directory exists
        try:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Fallback to temp directory if creation fails
            import tempfile
            self.local_cache_dir = Path(tempfile.gettempdir()) / "hf_datasets_cache"
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using fallback cache directory: {self.local_cache_dir}")

        if auto_create_repo and self._check_available():
            self._ensure_repo_exists()

        logger.info(f"HuggingFace Dataset Manager initialized for repo: {self.repo_id}")

    def _check_available(self) -> bool:
        """Check if huggingface_hub and datasets libraries are available"""
        if self._hf_available is not None:
            return self._hf_available

        try:
            from huggingface_hub import HfApi
            from datasets import Dataset, DatasetDict
            self._api = HfApi(token=self.token)
            self._hf_available = True
            return True
        except ImportError as e:
            logger.warning(f"HuggingFace libraries not fully installed: {e}")
            self._hf_available = False
            return False

    @property
    def api(self):
        """Get HuggingFace API client (lazy loading)"""
        if self._api is None:
            if self._check_available():
                from huggingface_hub import HfApi
                self._api = HfApi(token=self.token)
            else:
                raise ImportError("huggingface_hub is not installed")
        return self._api

    def _ensure_repo_exists(self) -> bool:
        """Create dataset repository if it doesn't exist"""
        if not self._check_available():
            return False

        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="dataset", token=self.token)
            logger.debug(f"Dataset repository {self.repo_id} exists")
            return True
        except Exception:
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=self.private,
                    repo_type="dataset"
                )
                logger.info(f"Created HF dataset repository: {self.repo_id} (private={self.private})")
                self._create_dataset_card()
                return True
            except Exception as e:
                logger.error(f"Failed to create dataset repository {self.repo_id}: {e}")
                return False

    def _create_dataset_card(self):
        """Create initial dataset card README for the repository"""
        readme_content = """---
language:
- en
task_categories:
- time-series-forecasting
- tabular-classification
tags:
- finance
- stock-market
- investment
- technical-indicators
size_categories:
- 10K<n<100K
license: mit
---

# Investment Analysis ML Training Dataset

This dataset contains processed stock market data for training ML models
in the Investment Analysis Platform.

## Dataset Structure

- **train**: Training data (70% of samples)
- **validation**: Validation data (15% of samples)
- **test**: Test data (15% of samples)

## Features

The dataset includes 80+ features:
- Price data (OHLCV)
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Fundamental metrics (P/E, Market Cap, etc.)
- Derived features (returns, volatility, momentum)

## Labels

- Future returns at multiple horizons (1d, 5d, 10d, 20d)
- Direction signals
- Risk-adjusted returns

## Versioning

Datasets are versioned using semantic versioning (major.minor.patch).
Each version includes metadata about:
- Date range covered
- Number of stocks
- Feature columns
- Data hash for reproducibility

## Privacy

This is a private repository for internal use.
"""
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(readme_content)
                temp_path = f.name

            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message="Initialize repository with dataset card"
            )

            os.unlink(temp_path)
            logger.info("Created dataset card README")
        except Exception as e:
            logger.warning(f"Failed to create dataset card: {e}")

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of DataFrame content"""
        # Use a sample of the data for faster hashing of large datasets
        if len(df) > 10000:
            sample = df.sample(n=10000, random_state=42)
        else:
            sample = df

        return hashlib.sha256(
            sample.to_csv(index=False).encode()
        ).hexdigest()[:16]

    def _get_next_version(self, version_type: str = "patch") -> str:
        """Generate next semantic version"""
        versions = self.list_versions()
        if not versions:
            return "1.0.0"

        latest = max(versions, key=lambda v: tuple(map(int, v.split('.'))))
        major, minor, patch = map(int, latest.split('.'))

        if version_type == "major":
            return f"{major + 1}.0.0"
        elif version_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"

    def upload_dataset(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metadata: Dict[str, Any],
        version_type: str = "patch",
        commit_message: str = None
    ) -> Optional[DatasetVersion]:
        """
        Upload dataset splits to Hugging Face Hub.

        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            test_df: Test data DataFrame
            metadata: Additional metadata (stocks, date_range, etc.)
            version_type: "major", "minor", or "patch"
            commit_message: Custom commit message

        Returns:
            DatasetVersion with upload details, or None if failed
        """
        if not self._check_available():
            logger.error("HuggingFace libraries not available")
            return None

        try:
            from datasets import Dataset, DatasetDict

            version = self._get_next_version(version_type)

            # Create HF Dataset objects
            dataset_dict = DatasetDict({
                "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
                "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
                "test": Dataset.from_pandas(test_df.reset_index(drop=True))
            })

            # Compute data hash for reproducibility
            combined_hash = hashlib.sha256(
                (self._compute_data_hash(train_df) +
                 self._compute_data_hash(val_df) +
                 self._compute_data_hash(test_df)).encode()
            ).hexdigest()[:32]

            # Prepare version metadata
            date_range = metadata.get('date_range', {})
            version_metadata = DatasetVersion(
                version=version,
                commit_hash="",  # Will be filled after push
                created_at=datetime.utcnow().isoformat(),
                total_samples=len(train_df) + len(val_df) + len(test_df),
                stock_count=metadata.get('stocks_processed', 0),
                feature_count=len(train_df.columns),
                date_range_start=date_range.get('start', ''),
                date_range_end=date_range.get('end', ''),
                train_samples=len(train_df),
                val_samples=len(val_df),
                test_samples=len(test_df),
                feature_columns=list(train_df.columns),
                label_columns=metadata.get('label_columns', []),
                data_hash=combined_hash
            )

            # Push to Hub
            commit_msg = commit_message or f"Dataset version {version}"

            with self.lock:
                dataset_dict.push_to_hub(
                    self.repo_id,
                    token=self.token,
                    commit_message=commit_msg,
                    private=self.private
                )

            # Get commit hash
            try:
                commits = list(self.api.list_repo_commits(self.repo_id, repo_type="dataset"))
                version_metadata.commit_hash = commits[0].commit_id if commits else ""
            except Exception:
                pass

            # Upload version metadata
            self._upload_version_metadata(version_metadata)

            # Create version tag
            try:
                self.api.create_tag(
                    self.repo_id,
                    repo_type="dataset",
                    tag=f"v{version}",
                    revision=version_metadata.commit_hash or "main"
                )
            except Exception as e:
                logger.warning(f"Failed to create version tag: {e}")

            logger.info(f"Uploaded dataset version {version} to {self.repo_id}")
            return version_metadata

        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            return None

    def _upload_version_metadata(self, version_info: DatasetVersion):
        """Upload version metadata to Hub"""
        try:
            import tempfile

            metadata_content = json.dumps(version_info.to_dict(), indent=2, default=str)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(metadata_content)
                temp_path = f.name

            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=f"versions/v{version_info.version}/metadata.json",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=f"Add metadata for version {version_info.version}"
            )

            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to upload version metadata: {e}")

    def download_dataset(
        self,
        version: str = None,
        split: str = None
    ) -> Optional[Any]:
        """
        Download dataset from Hugging Face Hub.

        Args:
            version: Specific version/tag (None for latest)
            split: Specific split ("train", "validation", "test") or None for all

        Returns:
            DatasetDict or Dataset with requested splits
        """
        if not self._check_available():
            logger.error("HuggingFace libraries not available")
            return None

        try:
            from datasets import load_dataset

            revision = f"v{version}" if version else "main"

            dataset = load_dataset(
                self.repo_id,
                revision=revision,
                split=split,
                token=self.token
            )

            logger.info(f"Downloaded dataset from {self.repo_id} (revision: {revision})")
            return dataset

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return None

    def download_as_dataframe(
        self,
        version: str = None,
        split: str = "train"
    ) -> Optional[pd.DataFrame]:
        """Download dataset split as pandas DataFrame"""
        dataset = self.download_dataset(version=version, split=split)
        if dataset is None:
            return None

        return dataset.to_pandas()

    def list_versions(self) -> List[str]:
        """List all dataset versions (tags)"""
        if not self._check_available():
            return []

        try:
            refs = self.api.list_repo_refs(self.repo_id, repo_type="dataset")
            versions = [tag.name.lstrip('v') for tag in refs.tags if tag.name.startswith('v')]
            # Sort by semantic version (newest first)
            versions.sort(key=lambda x: tuple(map(int, x.split('.'))), reverse=True)
            return versions
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []

    def get_version_metadata(self, version: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific version"""
        if not self._check_available():
            return None

        try:
            from huggingface_hub import hf_hub_download

            local_file = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=f"versions/v{version}/metadata.json",
                token=self.token
            )

            with open(local_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to get version metadata: {e}")
            return None

    def upload_from_local(
        self,
        local_processed_dir: Path = None,
        version_type: str = "patch"
    ) -> Optional[DatasetVersion]:
        """
        Upload dataset from local processed directory.

        Args:
            local_processed_dir: Path to processed data directory
            version_type: Version increment type

        Returns:
            DatasetVersion if successful
        """
        if local_processed_dir is None:
            local_processed_dir = self.local_cache_dir / "processed"

        local_processed_dir = Path(local_processed_dir)

        if not local_processed_dir.exists():
            logger.error(f"Processed data directory not found: {local_processed_dir}")
            return None

        try:
            # Load data files
            train_path = local_processed_dir / "train_data.parquet"
            val_path = local_processed_dir / "val_data.parquet"
            test_path = local_processed_dir / "test_data.parquet"
            metadata_path = local_processed_dir / "metadata.json"

            if not all(p.exists() for p in [train_path, val_path, test_path]):
                logger.error("Missing required data files")
                return None

            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)

            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            return self.upload_dataset(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                metadata=metadata,
                version_type=version_type
            )

        except Exception as e:
            logger.error(f"Failed to upload from local: {e}")
            return None


# Singleton instance for easy access
_dataset_manager: Optional[HuggingFaceDatasetManager] = None


def get_dataset_manager(
    repo_id: str = None,
    token: str = None,
    cache_dir: str = None
) -> Optional[HuggingFaceDatasetManager]:
    """
    Get or create the singleton HuggingFace Dataset Manager.

    Args:
        repo_id: Override default repository ID
        token: Override default token
        cache_dir: Override default cache directory

    Returns:
        HuggingFaceDatasetManager instance or None if disabled
    """
    global _dataset_manager

    # Check if HF Hub is enabled
    if os.getenv("HF_HUB_ENABLED", "true").lower() != "true":
        logger.info("HuggingFace Hub disabled via HF_HUB_ENABLED=false")
        return None

    if _dataset_manager is None:
        _dataset_manager = HuggingFaceDatasetManager(
            repo_id=repo_id,
            token=token,
            local_cache_dir=cache_dir or "data/ml_training"
        )

    return _dataset_manager
