"""
Memory Sync Adapter - Bidirectional sync between Claude Flow memory and ML Pipeline
Integrates .swarm/memory.db with model registry and training state
"""

import os
import json
import sqlite3
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a memory entry for synchronization"""
    key: str
    value: Any
    namespace: str
    timestamp: datetime
    vector_embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    ttl: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "value": self.value if isinstance(self.value, str) else json.dumps(self.value),
            "namespace": self.namespace,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
            "ttl": self.ttl
        }


@dataclass
class SyncResult:
    """Result of a sync operation"""
    success: bool
    entries_synced: int
    entries_failed: int
    direction: str  # "to_memory" or "from_memory"
    duration_ms: float
    errors: List[str]


class ClaudeFlowMemoryAdapter:
    """
    Adapter for Claude Flow's AgentDB memory system
    Provides bidirectional sync with ML Pipeline components
    """

    # ML-specific namespaces for the memory system
    NAMESPACES = {
        "ml-models": "Model metadata and versions",
        "training-jobs": "Training job state and results",
        "predictions": "Prediction results and metrics",
        "feature-store": "Feature embeddings and transformations",
        "ml-patterns": "Learned ML patterns and optimizations",
        "pipeline-state": "Pipeline execution state",
    }

    def __init__(
        self,
        memory_db_path: Optional[str] = None,
        registry_path: Optional[str] = None
    ):
        # Default paths from environment or project structure
        self.memory_db_path = memory_db_path or os.getenv(
            "CLAUDE_FLOW_MEMORY_PATH",
            ".swarm/memory.db"
        )
        self.registry_path = registry_path or "backend/ml/models/registry.json"

        # Ensure paths are absolute
        project_root = Path(__file__).parent.parent.parent.parent
        if not Path(self.memory_db_path).is_absolute():
            self.memory_db_path = str(project_root / self.memory_db_path)
        if not Path(self.registry_path).is_absolute():
            self.registry_path = str(project_root / self.registry_path)

        # Connection pool
        self._connection: Optional[sqlite3.Connection] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the memory adapter and ensure schema exists"""
        try:
            # Check if memory database exists
            if not Path(self.memory_db_path).exists():
                logger.warning(f"Memory database not found at {self.memory_db_path}")
                logger.info("Run 'npx claude-flow memory init' to initialize")
                return False

            # Connect to database
            self._connection = sqlite3.connect(
                self.memory_db_path,
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row

            # Initialize ML namespaces if they don't exist
            await self._ensure_namespaces()

            self._initialized = True
            logger.info(f"Memory adapter initialized: {self.memory_db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory adapter: {e}")
            return False

    async def _ensure_namespaces(self):
        """Ensure ML-specific namespaces exist in the memory system"""
        cursor = self._connection.cursor()

        for namespace, description in self.NAMESPACES.items():
            # Check if namespace exists in entries
            cursor.execute("""
                SELECT COUNT(*) FROM memory_entries
                WHERE namespace = ?
            """, (namespace,))

            # Initialize namespace with metadata entry if empty
            if cursor.fetchone()[0] == 0:
                await self.store(
                    key=f"_namespace_meta_{namespace}",
                    value={"description": description, "created": datetime.utcnow().isoformat()},
                    namespace=namespace
                )

        self._connection.commit()

    async def store(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        metadata: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Store a value in Claude Flow memory"""
        if not self._initialized:
            await self.initialize()

        try:
            cursor = self._connection.cursor()

            # Serialize value
            value_str = value if isinstance(value, str) else json.dumps(value)
            metadata_str = json.dumps(metadata) if metadata else None

            # Generate unique ID
            entry_id = f"entry_{int(datetime.utcnow().timestamp() * 1000)}_{hashlib.md5(key.encode()).hexdigest()[:8]}"

            # Insert or update
            cursor.execute("""
                INSERT OR REPLACE INTO memory_entries
                (id, key, value, namespace, metadata, ttl, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                key,
                value_str,
                namespace,
                metadata_str,
                ttl,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat()
            ))

            self._connection.commit()
            logger.debug(f"Stored {key} in namespace {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to store {key}: {e}")
            return False

    async def retrieve(
        self,
        key: str,
        namespace: str = "default"
    ) -> Optional[Any]:
        """Retrieve a value from Claude Flow memory"""
        if not self._initialized:
            await self.initialize()

        try:
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT value, metadata FROM memory_entries
                WHERE key = ? AND namespace = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (key, namespace))

            row = cursor.fetchone()
            if row:
                value = row["value"]
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve {key}: {e}")
            return None

    async def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search memory entries (text-based, vector search requires CLI)"""
        if not self._initialized:
            await self.initialize()

        try:
            cursor = self._connection.cursor()

            if namespace:
                cursor.execute("""
                    SELECT key, value, namespace, metadata, updated_at
                    FROM memory_entries
                    WHERE namespace = ? AND (key LIKE ? OR value LIKE ?)
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (namespace, f"%{query}%", f"%{query}%", limit))
            else:
                cursor.execute("""
                    SELECT key, value, namespace, metadata, updated_at
                    FROM memory_entries
                    WHERE key LIKE ? OR value LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))

            results = []
            for row in cursor.fetchall():
                value = row["value"]
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

                results.append({
                    "key": row["key"],
                    "value": value,
                    "namespace": row["namespace"],
                    "updated_at": row["updated_at"]
                })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def sync_model_to_memory(
        self,
        model_name: str,
        model_metadata: Dict
    ) -> bool:
        """Sync a model's metadata to Claude Flow memory"""
        key = f"model_{model_name}"

        # Enrich metadata
        enriched = {
            **model_metadata,
            "synced_at": datetime.utcnow().isoformat(),
            "source": "ml_pipeline",
            "sync_version": "1.0"
        }

        return await self.store(
            key=key,
            value=enriched,
            namespace="ml-models",
            metadata={"model_name": model_name, "sync_type": "model_registration"}
        )

    async def sync_training_job(
        self,
        job_id: str,
        job_state: Dict
    ) -> bool:
        """Sync training job state to memory"""
        key = f"job_{job_id}"

        return await self.store(
            key=key,
            value=job_state,
            namespace="training-jobs",
            metadata={"job_id": job_id}
        )

    async def sync_prediction_result(
        self,
        prediction_id: str,
        result: Dict
    ) -> bool:
        """Sync prediction result to memory for learning"""
        key = f"prediction_{prediction_id}"

        return await self.store(
            key=key,
            value=result,
            namespace="predictions",
            metadata={"prediction_id": prediction_id},
            ttl=86400 * 7  # 7 days TTL for predictions
        )

    async def get_model_from_memory(
        self,
        model_name: str
    ) -> Optional[Dict]:
        """Retrieve model metadata from Claude Flow memory"""
        return await self.retrieve(
            key=f"model_{model_name}",
            namespace="ml-models"
        )

    async def sync_registry_to_memory(self) -> SyncResult:
        """Sync entire model registry to Claude Flow memory"""
        start_time = datetime.utcnow()
        synced = 0
        failed = 0
        errors = []

        try:
            # Load registry
            if not Path(self.registry_path).exists():
                return SyncResult(
                    success=False,
                    entries_synced=0,
                    entries_failed=0,
                    direction="to_memory",
                    duration_ms=0,
                    errors=["Registry file not found"]
                )

            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            # Sync each model
            models = registry.get("models", [])
            for model in models:
                model_name = model.get("name", model.get("id", "unknown"))
                success = await self.sync_model_to_memory(model_name, model)
                if success:
                    synced += 1
                else:
                    failed += 1
                    errors.append(f"Failed to sync {model_name}")

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SyncResult(
                success=failed == 0,
                entries_synced=synced,
                entries_failed=failed,
                direction="to_memory",
                duration_ms=duration,
                errors=errors
            )

        except Exception as e:
            logger.error(f"Registry sync failed: {e}")
            return SyncResult(
                success=False,
                entries_synced=synced,
                entries_failed=failed + 1,
                direction="to_memory",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                errors=[str(e)]
            )

    async def sync_memory_to_registry(self) -> SyncResult:
        """Sync Claude Flow memory models back to registry"""
        start_time = datetime.utcnow()
        synced = 0
        failed = 0
        errors = []

        try:
            # Get all models from memory
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT key, value FROM memory_entries
                WHERE namespace = 'ml-models' AND key LIKE 'model_%'
            """)

            memory_models = []
            for row in cursor.fetchall():
                try:
                    value = json.loads(row["value"])
                    memory_models.append(value)
                    synced += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"Parse error: {e}")

            # Update registry
            if Path(self.registry_path).exists():
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {"models": [], "version": "1.0"}

            # Merge models (memory takes precedence for matching names)
            existing_names = {m.get("name") for m in registry.get("models", [])}
            for mem_model in memory_models:
                name = mem_model.get("name")
                if name and name not in existing_names:
                    registry["models"].append(mem_model)

            # Save registry
            Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SyncResult(
                success=failed == 0,
                entries_synced=synced,
                entries_failed=failed,
                direction="from_memory",
                duration_ms=duration,
                errors=errors
            )

        except Exception as e:
            logger.error(f"Memory to registry sync failed: {e}")
            return SyncResult(
                success=False,
                entries_synced=synced,
                entries_failed=failed + 1,
                direction="from_memory",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                errors=[str(e)]
            )

    async def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._initialized = False


# Singleton instance for use across the application
_memory_adapter: Optional[ClaudeFlowMemoryAdapter] = None


async def get_memory_adapter() -> ClaudeFlowMemoryAdapter:
    """Get or create the memory adapter singleton"""
    global _memory_adapter
    if _memory_adapter is None:
        _memory_adapter = ClaudeFlowMemoryAdapter()
        await _memory_adapter.initialize()
    return _memory_adapter
