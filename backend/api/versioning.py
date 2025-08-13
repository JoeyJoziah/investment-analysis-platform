"""
API Versioning System
Provides backward compatibility and smooth migration paths for API changes.
"""

from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
import logging
import warnings

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        status=VersionStatus.DEPRECATED,
        release_date=datetime(2024, 1, 1),
        deprecation_date=datetime(2025, 1, 1),
        sunset_date=datetime(2025, 7, 1),
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
        release_date=datetime(2024, 7, 1),
        deprecation_date=datetime(2025, 7, 1),
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
        release_date=datetime(2025, 1, 1),
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
    
    def register_router(self, version: APIVersion, router: APIRouter) -> None:
        """Register a router for a specific API version."""
        self.routers[version] = router
        logger.info(f"Registered router for API {version.value}")
    
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
        'timestamp': datetime.utcnow().isoformat()
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


# Versioned router factory
def create_versioned_router(version: APIVersion) -> APIRouter:
    """Create a router for a specific API version."""
    router = APIRouter(
        prefix=f"/api/{version.value}",
        tags=[f"API {version.value}"],
        responses={
            410: {"description": "API version no longer supported"},
            426: {"description": "Upgrade required"}
        }
    )
    
    # Add version info endpoint
    @router.get("/version")
    async def get_version_info():
        """Get information about this API version."""
        info = VERSION_REGISTRY.get(version)
        if info:
            return info.dict()
        return {"error": "Version information not found"}
    
    # Add deprecation headers middleware
    @router.middleware("http")
    async def add_version_headers(request: Request, call_next):
        response = await call_next(request)
        
        info = VERSION_REGISTRY.get(version)
        if info:
            response.headers["X-API-Version"] = version.value
            response.headers["X-API-Status"] = info.status.value
            
            if info.status == VersionStatus.DEPRECATED:
                response.headers["X-API-Deprecation-Date"] = info.deprecation_date.isoformat()
                response.headers["Sunset"] = info.sunset_date.isoformat()
                response.headers["Link"] = f'</api/{APIVersion.LATEST.value}>; rel="successor-version"'
        
        return response
    
    version_manager.register_router(version, router)
    return router