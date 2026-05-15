"""
API Versioning System

`/api/v1/` is the current stable prefix for this platform. There is no v2
migration planned; future major versions would mount at `/api/v2/`. The
historical V1DeprecationMiddleware, V1_TO_V2_ENDPOINT_MAP, and
create_versioned_router() factory were removed in PRD audit 2026-04 /
Workstream F because they incorrectly treated the current prefix as legacy
and emitted Sunset/Deprecation/Warning headers plus log spam on every
production request.

This module now provides:
- Version detection from requests (header, URL path, query param)
- Per-version registry metadata (status, dates, change log)
- Optional response transformers between versions (kept for forward use)
- V1MigrationMetrics + admin router (kept inert; harmless if unused)
"""

from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime, timezone
from functools import wraps
from enum import Enum
from collections import defaultdict
import logging
import warnings
import asyncio

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# V1 MIGRATION METRICS TRACKING
# =============================================================================

class V1MigrationMetrics:
    """
    Tracks V1 API usage to monitor migration progress.

    This helps identify:
    - Which V1 endpoints are still being used
    - Which clients haven't migrated
    - Traffic patterns for V1 vs V2
    """

    def __init__(self):
        self._endpoint_usage: Dict[str, int] = defaultdict(int)
        self._client_usage: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "last_seen": None, "endpoints": set()}
        )
        self._hourly_usage: Dict[str, int] = defaultdict(int)
        self._total_v1_requests = 0
        self._total_v2_requests = 0
        self._total_v3_requests = 0
        self._redirects_issued = 0
        self._lock = asyncio.Lock()

    async def record_v1_request(
        self,
        endpoint: str,
        client_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Record a V1 API request for tracking."""
        async with self._lock:
            self._total_v1_requests += 1
            self._endpoint_usage[endpoint] += 1

            # Track hourly usage
            hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
            self._hourly_usage[hour_key] += 1

            # Track client usage
            if client_id:
                self._client_usage[client_id]["count"] += 1
                self._client_usage[client_id]["last_seen"] = datetime.now(timezone.utc)
                self._client_usage[client_id]["endpoints"].add(endpoint)
                if user_agent:
                    self._client_usage[client_id]["user_agent"] = user_agent

            # Log high-frequency V1 usage
            if self._total_v1_requests % 100 == 0:
                logger.warning(
                    f"V1 API usage milestone: {self._total_v1_requests} total requests. "
                    f"Top endpoints: {self.get_top_endpoints(3)}"
                )

    async def record_version_request(self, version: str) -> None:
        """Record request by API version."""
        async with self._lock:
            if version == "v1":
                self._total_v1_requests += 1
            elif version == "v2":
                self._total_v2_requests += 1
            elif version == "v3":
                self._total_v3_requests += 1

    async def record_redirect(self) -> None:
        """Record when a V1 redirect is issued."""
        async with self._lock:
            self._redirects_issued += 1

    def get_top_endpoints(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the most frequently used V1 endpoints."""
        sorted_endpoints = sorted(
            self._endpoint_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_endpoints[:n]

    def get_migration_progress(self) -> Dict[str, Any]:
        """Get overall migration progress statistics."""
        total = self._total_v1_requests + self._total_v2_requests + self._total_v3_requests
        return {
            "total_requests": total,
            "v1_requests": self._total_v1_requests,
            "v2_requests": self._total_v2_requests,
            "v3_requests": self._total_v3_requests,
            "v1_percentage": (self._total_v1_requests / total * 100) if total > 0 else 0,
            "v2_percentage": (self._total_v2_requests / total * 100) if total > 0 else 0,
            "v3_percentage": (self._total_v3_requests / total * 100) if total > 0 else 0,
            "redirects_issued": self._redirects_issued,
            "unique_v1_clients": len(self._client_usage),
            "top_v1_endpoints": self.get_top_endpoints(10),
            "migration_complete": self._total_v1_requests == 0 and total > 0
        }

    def get_client_report(self) -> List[Dict[str, Any]]:
        """Get report of clients still using V1 API."""
        clients = []
        for client_id, data in self._client_usage.items():
            clients.append({
                "client_id": client_id,
                "request_count": data["count"],
                "last_seen": data["last_seen"].isoformat() if data["last_seen"] else None,
                "endpoints_used": list(data["endpoints"]),
                "user_agent": data.get("user_agent")
            })
        return sorted(clients, key=lambda x: x["request_count"], reverse=True)


# Global metrics instance
v1_migration_metrics = V1MigrationMetrics()


class APIVersion(Enum):
    """API version definitions."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = V3  # Current latest version


class VersionStatus(Enum):
    """Version lifecycle status."""
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class VersionInfo(BaseModel):
    """API version information."""
    version: str
    status: VersionStatus
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    changes: List[str] = Field(default_factory=list)
    breaking_changes: List[str] = Field(default_factory=list)


# Version registry
VERSION_REGISTRY: Dict[APIVersion, VersionInfo] = {
    APIVersion.V1: VersionInfo(
        version="v1",
        status=VersionStatus.SUNSET,  # V1 is now sunset as of 2025-07-01
        release_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        deprecation_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        sunset_date=datetime(2025, 7, 1, tzinfo=timezone.utc),
        changes=[
            "Initial API release",
            "Basic stock data endpoints",
            "Simple authentication"
        ],
        breaking_changes=[]
    ),
    APIVersion.V2: VersionInfo(
        version="v2",
        status=VersionStatus.STABLE,
        release_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
        deprecation_date=datetime(2025, 7, 1, tzinfo=timezone.utc),
        changes=[
            "Added WebSocket support",
            "Enhanced rate limiting",
            "Batch operations",
            "Improved error responses"
        ],
        breaking_changes=[
            "Changed authentication to OAuth2",
            "Modified response structure for /stocks endpoint",
            "Renamed 'ticker' to 'symbol' in all endpoints"
        ]
    ),
    APIVersion.V3: VersionInfo(
        version="v3",
        status=VersionStatus.STABLE,
        release_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        changes=[
            "GraphQL support",
            "Real-time streaming",
            "Advanced analytics endpoints",
            "Machine learning predictions"
        ],
        breaking_changes=[
            "New pagination format",
            "Changed date format to ISO 8601",
            "Restructured error codes"
        ]
    )
}


class APIVersionManager:
    """Manages API versioning across the application."""
    
    def __init__(self, default_version: APIVersion = APIVersion.LATEST):
        """
        Initialize version manager.
        
        Args:
            default_version: Default API version to use
        """
        self.default_version = default_version
        self.routers: Dict[APIVersion, APIRouter] = {}
        self.transformers: Dict[tuple, Callable] = {}  # (from_version, to_version) -> transformer
        
        # Metrics
        self._metrics = {
            'requests_by_version': {v.value: 0 for v in APIVersion},
            'deprecated_version_usage': 0,
            'version_errors': 0
        }
    
    def register_transformer(
        self,
        from_version: APIVersion,
        to_version: APIVersion,
        transformer: Callable
    ) -> None:
        """
        Register a data transformer between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            transformer: Function to transform data
        """
        self.transformers[(from_version, to_version)] = transformer
        logger.info(f"Registered transformer from {from_version.value} to {to_version.value}")
    
    def get_version_from_request(self, request: Request) -> APIVersion:
        """
        Extract API version from request.
        
        Priority:
        1. Header: X-API-Version
        2. URL path: /api/v1/...
        3. Query parameter: ?version=v1
        4. Default version
        """
        # Check header
        version_header = request.headers.get("X-API-Version")
        if version_header:
            try:
                return APIVersion(version_header)
            except ValueError:
                logger.warning(f"Invalid version in header: {version_header}")
        
        # Check URL path
        path_parts = request.url.path.split('/')
        for part in path_parts:
            if part in [v.value for v in APIVersion]:
                return APIVersion(part)
        
        # Check query parameter
        version_param = request.query_params.get("version")
        if version_param:
            try:
                return APIVersion(version_param)
            except ValueError:
                logger.warning(f"Invalid version in query: {version_param}")
        
        # Return default
        return self.default_version
    
    def check_version_status(self, version: APIVersion) -> None:
        """
        Check version status and emit warnings if needed.
        
        Args:
            version: API version to check
        
        Raises:
            HTTPException: If version is sunset
        """
        info = VERSION_REGISTRY.get(version)
        if not info:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown API version: {version.value}"
            )
        
        if info.status == VersionStatus.SUNSET:
            raise HTTPException(
                status_code=410,
                detail=f"API version {version.value} is no longer supported. "
                       f"Please upgrade to {APIVersion.LATEST.value}"
            )
        
        if info.status == VersionStatus.DEPRECATED:
            warnings.warn(
                f"API version {version.value} is deprecated and will be sunset on "
                f"{info.sunset_date}. Please upgrade to {APIVersion.LATEST.value}",
                DeprecationWarning
            )
            self._metrics['deprecated_version_usage'] += 1
    
    def transform_response(
        self,
        data: Any,
        from_version: APIVersion,
        to_version: APIVersion
    ) -> Any:
        """
        Transform response data between versions.
        
        Args:
            data: Response data
            from_version: Current data version
            to_version: Target version
        
        Returns:
            Transformed data
        """
        if from_version == to_version:
            return data
        
        transformer = self.transformers.get((from_version, to_version))
        if transformer:
            return transformer(data)
        
        # Try to find a path through intermediate versions
        path = self._find_transformation_path(from_version, to_version)
        if path:
            result = data
            for i in range(len(path) - 1):
                transformer = self.transformers.get((path[i], path[i + 1]))
                if transformer:
                    result = transformer(result)
            return result
        
        logger.warning(f"No transformer from {from_version.value} to {to_version.value}")
        return data
    
    def _find_transformation_path(
        self,
        from_version: APIVersion,
        to_version: APIVersion
    ) -> Optional[List[APIVersion]]:
        """Find transformation path between versions."""
        # Simple BFS to find path
        from collections import deque
        
        queue = deque([(from_version, [from_version])])
        visited = {from_version}
        
        while queue:
            current, path = queue.popleft()
            
            if current == to_version:
                return path
            
            # Check all possible transformations from current
            for (f, t), _ in self.transformers.items():
                if f == current and t not in visited:
                    visited.add(t)
                    queue.append((t, path + [t]))
        
        return None
    
    def version_route(
        self,
        supported_versions: List[APIVersion] = None,
        deprecated_in: Optional[APIVersion] = None,
        removed_in: Optional[APIVersion] = None
    ):
        """
        Decorator for versioned API endpoints.
        
        Args:
            supported_versions: List of versions supporting this endpoint
            deprecated_in: Version where endpoint is deprecated
            removed_in: Version where endpoint is removed
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                version = self.get_version_from_request(request)
                
                # Check if endpoint is supported in this version
                if supported_versions and version not in supported_versions:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Endpoint not available in API {version.value}"
                    )
                
                # Check if endpoint is removed
                if removed_in and version.value >= removed_in.value:
                    raise HTTPException(
                        status_code=410,
                        detail=f"Endpoint removed in API {removed_in.value}"
                    )
                
                # Warn if deprecated
                if deprecated_in and version.value >= deprecated_in.value:
                    warnings.warn(
                        f"Endpoint is deprecated in API {deprecated_in.value}",
                        DeprecationWarning
                    )
                
                # Update metrics
                self._metrics['requests_by_version'][version.value] += 1
                
                # Execute endpoint
                result = await func(request, *args, **kwargs)
                
                # Transform response if needed
                client_version = self.get_version_from_request(request)
                if hasattr(result, '__api_version__'):
                    result = self.transform_response(
                        result,
                        result.__api_version__,
                        client_version
                    )
                
                return result
            
            return wrapper
        return decorator
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get versioning metrics."""
        return self._metrics.copy()


# Data transformers between versions
def transform_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform V1 response to V2 format."""
    transformed = data.copy()
    
    # Rename 'ticker' to 'symbol'
    if 'ticker' in transformed:
        transformed['symbol'] = transformed.pop('ticker')
    
    # Update response structure
    if 'data' in transformed:
        transformed['result'] = transformed.pop('data')
    
    # Add metadata
    transformed['_metadata'] = {
        'version': 'v2',
        'transformed_from': 'v1',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    return transformed


def transform_v2_to_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform V2 response to V3 format."""
    transformed = data.copy()
    
    # Update pagination format
    if 'page' in transformed and 'per_page' in transformed:
        transformed['pagination'] = {
            'current_page': transformed.pop('page'),
            'items_per_page': transformed.pop('per_page'),
            'total_items': transformed.get('total', 0)
        }
    
    # Convert dates to ISO 8601
    for key in ['created_at', 'updated_at', 'date']:
        if key in transformed and transformed[key]:
            if isinstance(transformed[key], str):
                try:
                    dt = datetime.fromisoformat(transformed[key])
                    transformed[key] = dt.isoformat()
                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Could not parse date {transformed[key]}: {e}")
    
    # Update error codes
    if 'error_code' in transformed:
        old_code = transformed['error_code']
        # Map old codes to new structure
        code_mapping = {
            'ERR001': 'VALIDATION_ERROR',
            'ERR002': 'NOT_FOUND',
            'ERR003': 'UNAUTHORIZED',
            'ERR004': 'RATE_LIMITED'
        }
        transformed['error'] = {
            'code': code_mapping.get(old_code, old_code),
            'message': transformed.get('error_message', ''),
            'details': transformed.get('error_details', {})
        }
        transformed.pop('error_code', None)
        transformed.pop('error_message', None)
        transformed.pop('error_details', None)
    
    return transformed


def transform_v1_to_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform V1 response directly to V3 format."""
    # First transform to V2
    v2_data = transform_v1_to_v2(data)
    # Then transform to V3
    return transform_v2_to_v3(v2_data)


# Global version manager
version_manager = APIVersionManager()

# Register transformers
version_manager.register_transformer(APIVersion.V1, APIVersion.V2, transform_v1_to_v2)
version_manager.register_transformer(APIVersion.V2, APIVersion.V3, transform_v2_to_v3)
version_manager.register_transformer(APIVersion.V1, APIVersion.V3, transform_v1_to_v3)


# =============================================================================
# V1 MIGRATION ROUTER (Admin endpoints for monitoring migration)
# =============================================================================

v1_migration_router = APIRouter(
    prefix="/api/v1/admin/v1-migration",
    tags=["v1-migration", "admin"]
)


@v1_migration_router.get("/metrics")
async def get_v1_migration_metrics():
    """
    Get V1 API migration metrics and progress.

    Returns statistics on V1 usage to help monitor migration progress.
    """
    return {
        "status": "success",
        "data": v1_migration_metrics.get_migration_progress(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@v1_migration_router.get("/clients")
async def get_v1_clients():
    """
    Get list of clients still using V1 API.

    Returns detailed information about which clients are still making
    V1 API requests, useful for targeted migration outreach.
    """
    return {
        "status": "success",
        "data": v1_migration_metrics.get_client_report(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# NOTE: GET /endpoint-mapping was removed in PRD audit 2026-04 / Workstream F.
# It exposed the now-deleted V1_TO_V2_ENDPOINT_MAP, which was always wrong
# (mapped current /api/v1/* to non-existent /api/* targets) and had zero
# call sites.


@v1_migration_router.get("/version-info")
async def get_all_version_info():
    """
    Get information about all API versions.
    """
    return {
        "status": "success",
        "data": {
            version.value: info.model_dump()
            for version, info in VERSION_REGISTRY.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }