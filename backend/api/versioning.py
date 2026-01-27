"""
API Versioning System
Provides backward compatibility and smooth migration paths for API changes.

This module handles:
- Version detection from requests (header, URL path, query param)
- Deprecation warnings and sunset enforcement
- Automatic redirects from V1 to V2 endpoints
- V1 usage metrics tracking for migration monitoring
- Response transformation between API versions
"""

from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime, timedelta, timezone
from functools import wraps
from enum import Enum
from collections import defaultdict
import logging
import warnings
import asyncio
import time

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
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


# =============================================================================
# V1 TO V2 ENDPOINT MAPPING
# =============================================================================

# Maps V1 endpoints to their V2 equivalents
V1_TO_V2_ENDPOINT_MAP: Dict[str, str] = {
    # Stock endpoints
    "/api/v1/stocks": "/api/stocks",
    "/api/v1/stocks/search": "/api/stocks/search",
    "/api/v1/stocks/sectors": "/api/stocks/sectors",
    "/api/v1/stock/{symbol}": "/api/stocks/{symbol}",
    "/api/v1/stock/{symbol}/quote": "/api/stocks/{symbol}/quote",
    "/api/v1/stock/{symbol}/history": "/api/stocks/{symbol}/history",
    "/api/v1/stock/{symbol}/statistics": "/api/stocks/{symbol}/statistics",

    # Analysis endpoints
    "/api/v1/analysis/analyze": "/api/analysis/analyze",
    "/api/v1/analysis/{symbol}": "/api/analysis/analyze",
    "/api/v1/analysis/batch": "/api/analysis/batch",
    "/api/v1/analysis/compare": "/api/analysis/compare",
    "/api/v1/analysis/indicators/{symbol}": "/api/analysis/indicators/{symbol}",
    "/api/v1/analysis/sentiment/{symbol}": "/api/analysis/sentiment/{symbol}",

    # Portfolio endpoints
    "/api/v1/portfolio": "/api/portfolio",
    "/api/v1/portfolio/{id}": "/api/portfolio/{id}",
    "/api/v1/portfolio/{id}/holdings": "/api/portfolio/{id}/holdings",
    "/api/v1/portfolio/{id}/performance": "/api/portfolio/{id}/performance",

    # Auth endpoints (changed significantly in V2)
    "/api/v1/auth/login": "/api/auth/login",
    "/api/v1/auth/register": "/api/auth/register",
    "/api/v1/auth/token": "/api/auth/token",
    "/api/v1/auth/refresh": "/api/auth/refresh",
    "/api/v1/auth/me": "/api/auth/me",

    # Recommendations
    "/api/v1/recommendations": "/api/recommendations",
    "/api/v1/recommendations/{symbol}": "/api/recommendations/{symbol}",

    # Watchlist (V1 used different structure)
    "/api/v1/watchlist": "/api/watchlists/default",
    "/api/v1/watchlist/add/{symbol}": "/api/watchlists/default/symbols/{symbol}",
    "/api/v1/watchlist/remove/{symbol}": "/api/watchlists/default/symbols/{symbol}",
}

# Parameter mapping from V1 to V2
V1_TO_V2_PARAM_MAP: Dict[str, Dict[str, str]] = {
    "ticker": "symbol",  # V1 used 'ticker', V2 uses 'symbol'
    "stock_id": "symbol",
    "page_size": "limit",
    "page_num": "offset",  # Needs transformation: offset = (page_num - 1) * page_size
}


def map_v1_endpoint_to_v2(v1_path: str) -> Optional[str]:
    """
    Map a V1 endpoint path to its V2 equivalent.

    Args:
        v1_path: The V1 API endpoint path

    Returns:
        The equivalent V2 path, or None if no mapping exists
    """
    # Direct match
    if v1_path in V1_TO_V2_ENDPOINT_MAP:
        return V1_TO_V2_ENDPOINT_MAP[v1_path]

    # Pattern matching for parameterized routes
    for v1_pattern, v2_pattern in V1_TO_V2_ENDPOINT_MAP.items():
        if "{" in v1_pattern:
            # Convert pattern to regex-like matching
            v1_parts = v1_pattern.split("/")
            path_parts = v1_path.split("/")

            if len(v1_parts) != len(path_parts):
                continue

            match = True
            params = {}
            for v1_part, path_part in zip(v1_parts, path_parts):
                if v1_part.startswith("{") and v1_part.endswith("}"):
                    param_name = v1_part[1:-1]
                    params[param_name] = path_part
                elif v1_part != path_part:
                    match = False
                    break

            if match:
                # Substitute parameters into V2 pattern
                v2_path = v2_pattern
                for param_name, param_value in params.items():
                    v2_path = v2_path.replace(f"{{{param_name}}}", param_value)
                return v2_path

    return None


def transform_v1_params_to_v2(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform V1 query parameters to V2 format.

    Args:
        params: V1 query parameters

    Returns:
        Transformed parameters for V2 API
    """
    transformed = {}

    for key, value in params.items():
        # Check if parameter needs renaming
        new_key = V1_TO_V2_PARAM_MAP.get(key, key)
        transformed[new_key] = value

    # Handle pagination transformation
    if "page_num" in params and "page_size" in params:
        page_num = int(params["page_num"])
        page_size = int(params["page_size"])
        transformed["offset"] = (page_num - 1) * page_size
        transformed["limit"] = page_size
        transformed.pop("page_num", None)
        transformed.pop("page_size", None)

    return transformed


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
            return info.model_dump()
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


# =============================================================================
# V1 DEPRECATION MIDDLEWARE
# =============================================================================

class V1DeprecationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle V1 API deprecation, redirects, and usage tracking.

    This middleware:
    1. Detects V1 API requests from the URL path
    2. Tracks V1 usage metrics for migration monitoring
    3. Adds deprecation/sunset headers to V1 responses
    4. Optionally redirects V1 requests to V2 equivalents
    5. Returns 410 Gone for sunset endpoints with migration guidance

    Configuration:
    - enable_redirects: If True, automatically redirect V1 requests to V2
    - grace_period_days: Days after sunset to still allow V1 (with warnings)
    - strict_mode: If True, return 410 immediately after sunset date
    """

    def __init__(
        self,
        app: ASGIApp,
        enable_redirects: bool = False,
        grace_period_days: int = 30,
        strict_mode: bool = False
    ):
        super().__init__(app)
        self.enable_redirects = enable_redirects
        self.grace_period_days = grace_period_days
        self.strict_mode = strict_mode
        self.v1_info = VERSION_REGISTRY.get(APIVersion.V1)

    async def dispatch(self, request: Request, call_next):
        """Process the request and handle V1 deprecation logic."""
        path = request.url.path

        # Check if this is a V1 API request
        if "/api/v1/" in path or path.startswith("/api/v1"):
            return await self._handle_v1_request(request, call_next)

        # Check for V1 version header
        version_header = request.headers.get("X-API-Version")
        if version_header == "v1":
            return await self._handle_v1_request(request, call_next)

        # Not a V1 request, pass through normally
        # Track V2/V3 usage
        if "/api/v2/" in path or "/api/" in path:
            await v1_migration_metrics.record_version_request("v2")
        elif "/api/v3/" in path:
            await v1_migration_metrics.record_version_request("v3")

        response = await call_next(request)
        return response

    async def _handle_v1_request(self, request: Request, call_next):
        """Handle a V1 API request with deprecation logic."""
        path = request.url.path
        method = request.method
        client_id = request.headers.get("X-Client-ID") or request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")

        # Track V1 usage
        await v1_migration_metrics.record_v1_request(
            endpoint=path,
            client_id=client_id,
            user_agent=user_agent
        )

        # Check sunset status
        now = datetime.now(timezone.utc)
        is_past_sunset = self.v1_info and self.v1_info.sunset_date and now > self.v1_info.sunset_date
        grace_period_end = (
            self.v1_info.sunset_date + timedelta(days=self.grace_period_days)
            if self.v1_info and self.v1_info.sunset_date
            else None
        )
        is_past_grace_period = grace_period_end and now > grace_period_end

        # Strict mode: Return 410 immediately after sunset
        if self.strict_mode and is_past_sunset:
            return await self._return_sunset_response(request, path)

        # Past grace period: Return 410
        if is_past_grace_period:
            return await self._return_sunset_response(request, path)

        # Check if we should redirect
        if self.enable_redirects:
            v2_path = map_v1_endpoint_to_v2(path)
            if v2_path:
                await v1_migration_metrics.record_redirect()
                return await self._redirect_to_v2(request, v2_path)

        # Allow the request but add deprecation headers
        response = await call_next(request)

        # Add deprecation headers
        response = await self._add_deprecation_headers(response, is_past_sunset)

        # Log warning for V1 usage
        logger.warning(
            f"V1 API request: {method} {path} from client={client_id}. "
            f"V1 is {'SUNSET' if is_past_sunset else 'DEPRECATED'}. "
            f"Please migrate to V2/V3."
        )

        return response

    async def _return_sunset_response(self, request: Request, path: str) -> JSONResponse:
        """Return a 410 Gone response with migration guidance."""
        v2_path = map_v1_endpoint_to_v2(path)

        response_body = {
            "error": "API version no longer supported",
            "code": "API_VERSION_SUNSET",
            "message": (
                f"API V1 was sunset on {self.v1_info.sunset_date.strftime('%Y-%m-%d') if self.v1_info else 'N/A'}. "
                f"Please migrate to API V2 or V3."
            ),
            "migration": {
                "current_endpoint": path,
                "suggested_endpoint": v2_path or "See migration guide",
                "migration_guide": "/api/docs/migration/v1-to-v2",
                "latest_version": APIVersion.LATEST.value,
                "documentation": "/api/docs"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return JSONResponse(
            status_code=410,
            content=response_body,
            headers={
                "X-API-Version": "v1",
                "X-API-Status": "sunset",
                "Sunset": self.v1_info.sunset_date.isoformat() if self.v1_info else "",
                "Link": f'</api/{APIVersion.LATEST.value}>; rel="successor-version"',
                "X-Migration-Guide": "/api/docs/migration/v1-to-v2"
            }
        )

    async def _redirect_to_v2(self, request: Request, v2_path: str) -> RedirectResponse:
        """Redirect a V1 request to its V2 equivalent."""
        # Transform query parameters
        v2_params = transform_v1_params_to_v2(dict(request.query_params))

        # Build the redirect URL
        query_string = "&".join(f"{k}={v}" for k, v in v2_params.items()) if v2_params else ""
        redirect_url = f"{v2_path}?{query_string}" if query_string else v2_path

        logger.info(f"Redirecting V1 request {request.url.path} to V2: {redirect_url}")

        return RedirectResponse(
            url=redirect_url,
            status_code=308,  # Permanent redirect, preserves method
            headers={
                "X-Redirect-Reason": "V1 API sunset - automatically redirected to V2",
                "X-Original-Path": request.url.path,
                "X-Migration-Guide": "/api/docs/migration/v1-to-v2"
            }
        )

    async def _add_deprecation_headers(self, response, is_past_sunset: bool):
        """Add deprecation headers to a V1 response."""
        if self.v1_info:
            # RFC 8594 Sunset header
            if self.v1_info.sunset_date:
                response.headers["Sunset"] = self.v1_info.sunset_date.strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                )

            # RFC 8288 Link header for successor version
            response.headers["Link"] = f'</api/{APIVersion.LATEST.value}>; rel="successor-version"'

            # Deprecation header (draft-ietf-httpapi-deprecation-header)
            if self.v1_info.deprecation_date:
                response.headers["Deprecation"] = self.v1_info.deprecation_date.strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                )

            # Custom headers for additional context
            response.headers["X-API-Version"] = "v1"
            response.headers["X-API-Status"] = "sunset" if is_past_sunset else "deprecated"
            response.headers["X-Migration-Guide"] = "/api/docs/migration/v1-to-v2"

            # Warning header (RFC 7234)
            warning_msg = (
                f'299 - "API V1 is {"sunset" if is_past_sunset else "deprecated"}. '
                f'Please migrate to V2 or V3. See /api/docs/migration/v1-to-v2"'
            )
            response.headers["Warning"] = warning_msg

        return response


# =============================================================================
# V1 MIGRATION ROUTER (Admin endpoints for monitoring migration)
# =============================================================================

v1_migration_router = APIRouter(
    prefix="/api/admin/v1-migration",
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


@v1_migration_router.get("/endpoint-mapping")
async def get_endpoint_mapping():
    """
    Get the complete V1 to V2 endpoint mapping.

    Useful for clients to understand how to migrate their API calls.
    """
    return {
        "status": "success",
        "data": {
            "endpoint_map": V1_TO_V2_ENDPOINT_MAP,
            "parameter_map": V1_TO_V2_PARAM_MAP,
            "breaking_changes": VERSION_REGISTRY[APIVersion.V2].breaking_changes
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


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