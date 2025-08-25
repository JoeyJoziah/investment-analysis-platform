"""
Comprehensive WebSocket Security System
Provides authentication, authorization, rate limiting, and monitoring for WebSocket connections
"""

import os
import json
import jwt
import time
import uuid
import asyncio
import hashlib
import hmac
from typing import Dict, List, Optional, Set, Any, Union, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, field
import logging
from urllib.parse import parse_qs, urlparse
from collections import defaultdict, deque

# WebSocket imports
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status, Query
from starlette.websockets import WebSocketState

# Redis for rate limiting and session management
import redis.asyncio as aioredis

# Authentication imports
from .enhanced_auth import get_auth_manager, UserSession, UserRole, AuthenticationError
from .audit_logging import get_audit_logger, AuditEventType, AuditSeverity
from .advanced_rate_limiter import RateLimitStrategy, ThreatLevel

logger = logging.getLogger(__name__)


class WebSocketEventType(str, Enum):
    """WebSocket event types for audit logging"""
    CONNECTION_OPENED = "ws_connection_opened"
    CONNECTION_CLOSED = "ws_connection_closed"
    CONNECTION_FAILED = "ws_connection_failed"
    MESSAGE_SENT = "ws_message_sent"
    MESSAGE_RECEIVED = "ws_message_received"
    SUBSCRIPTION_ADDED = "ws_subscription_added"
    SUBSCRIPTION_REMOVED = "ws_subscription_removed"
    RATE_LIMITED = "ws_rate_limited"
    SECURITY_VIOLATION = "ws_security_violation"


class WebSocketMessageType(str, Enum):
    """WebSocket message types"""
    AUTHENTICATE = "authenticate"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SYSTEM = "system"
    CHAT = "chat"
    ALERT = "alert"


class WebSocketSecurityViolation(str, Enum):
    """Types of WebSocket security violations"""
    INVALID_TOKEN = "invalid_token"
    EXPIRED_TOKEN = "expired_token"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_MESSAGE = "malicious_message"
    INVALID_MESSAGE_FORMAT = "invalid_message_format"
    EXCESSIVE_CONNECTIONS = "excessive_connections"


@dataclass
class WebSocketClient:
    """WebSocket client information with security context"""
    client_id: str
    websocket: WebSocket
    ip_address: str
    user_agent: str
    connected_at: datetime
    last_activity: datetime
    
    # Authentication
    user_session: Optional[UserSession] = None
    is_authenticated: bool = False
    authentication_token: Optional[str] = None
    
    # Rate limiting
    message_count: int = 0
    message_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    subscription_count: int = 0
    
    # Subscriptions and permissions
    subscriptions: Set[str] = field(default_factory=set)
    allowed_actions: Set[str] = field(default_factory=set)
    
    # Security monitoring
    security_violations: List[Dict[str, Any]] = field(default_factory=list)
    threat_level: ThreatLevel = ThreatLevel.LOW
    is_suspicious: bool = False
    is_blocked: bool = False
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
    
    def add_message(self):
        """Track message for rate limiting"""
        now = datetime.now(timezone.utc)
        self.message_count += 1
        self.message_timestamps.append(now)
        self.update_activity()
    
    def get_message_rate(self, window_seconds: int = 60) -> int:
        """Get message rate within time window"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        recent_messages = [ts for ts in self.message_timestamps if ts > cutoff]
        return len(recent_messages)
    
    def add_security_violation(self, violation_type: WebSocketSecurityViolation, details: Dict[str, Any] = None):
        """Add security violation to client record"""
        violation = {
            "type": violation_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }
        self.security_violations.append(violation)
        
        # Escalate threat level based on violations
        if len(self.security_violations) >= 10:
            self.threat_level = ThreatLevel.CRITICAL
        elif len(self.security_violations) >= 5:
            self.threat_level = ThreatLevel.HIGH
        elif len(self.security_violations) >= 2:
            self.threat_level = ThreatLevel.MEDIUM


@dataclass
class WebSocketRateLimitConfig:
    """Rate limiting configuration for WebSocket connections"""
    max_connections_per_ip: int = 10
    max_connections_per_user: int = 5
    max_messages_per_minute: int = 60
    max_subscriptions_per_client: int = 100
    message_size_limit: int = 65536  # 64KB
    heartbeat_interval: int = 30  # seconds
    connection_timeout: int = 3600  # 1 hour


class WebSocketAuthenticator:
    """WebSocket authentication handler"""
    
    def __init__(self):
        self.auth_manager = get_auth_manager()
    
    async def authenticate_connection(
        self,
        websocket: WebSocket,
        token: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[UserSession]:
        """Authenticate WebSocket connection"""
        
        if token:
            try:
                # Verify JWT token
                payload = await self.auth_manager.jwt_manager.verify_token(token)
                
                # Get session
                session_id = payload.get("session_id")
                if session_id:
                    session = await self.auth_manager.session_manager.get_session(session_id)
                    if session and session.status.value == "active":
                        return session
                
            except Exception as e:
                logger.warning(f"WebSocket token authentication failed: {e}")
                
        elif api_key:
            # Handle API key authentication
            # This would integrate with your API key management system
            pass
        
        return None
    
    def extract_token_from_query(self, query_params: Dict[str, List[str]]) -> Optional[str]:
        """Extract authentication token from query parameters"""
        if "token" in query_params:
            return query_params["token"][0]
        
        if "access_token" in query_params:
            return query_params["access_token"][0]
        
        return None
    
    def extract_token_from_headers(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract authentication token from WebSocket headers"""
        auth_header = headers.get("authorization", "").lower()
        if auth_header.startswith("bearer "):
            return auth_header[7:]
        
        return headers.get("sec-websocket-protocol")


class WebSocketRateLimiter:
    """Rate limiter for WebSocket connections"""
    
    def __init__(self, config: WebSocketRateLimitConfig, redis_url: str = None):
        self.config = config
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self._redis: Optional[aioredis.Redis] = None
        
        # In-memory rate limiting state
        self.ip_connections: defaultdict[str, int] = defaultdict(int)
        self.user_connections: defaultdict[str, int] = defaultdict(int)
    
    async def get_redis(self) -> Optional[aioredis.Redis]:
        """Get Redis connection"""
        if not self.redis_url:
            return None
        
        if not self._redis:
            self._redis = aioredis.from_url(self.redis_url)
        return self._redis
    
    async def check_connection_limits(
        self,
        ip_address: str,
        user_id: Optional[str] = None
    ) -> tuple[bool, str]:
        """Check if connection should be allowed based on rate limits"""
        
        # Check IP-based connection limit
        if self.ip_connections[ip_address] >= self.config.max_connections_per_ip:
            return False, f"Too many connections from IP {ip_address}"
        
        # Check user-based connection limit
        if user_id and self.user_connections[user_id] >= self.config.max_connections_per_user:
            return False, f"Too many connections for user {user_id}"
        
        # Check Redis-based limits if available
        redis = await self.get_redis()
        if redis:
            try:
                # Check IP connections in Redis
                ip_key = f"ws_ip_connections:{ip_address}"
                ip_count = await redis.incr(ip_key)
                await redis.expire(ip_key, 3600)  # 1 hour TTL
                
                if ip_count > self.config.max_connections_per_ip:
                    await redis.decr(ip_key)  # Revert increment
                    return False, f"Too many connections from IP {ip_address} (Redis)"
                
                # Check user connections in Redis
                if user_id:
                    user_key = f"ws_user_connections:{user_id}"
                    user_count = await redis.incr(user_key)
                    await redis.expire(user_key, 3600)  # 1 hour TTL
                    
                    if user_count > self.config.max_connections_per_user:
                        await redis.decr(user_key)  # Revert increment
                        return False, f"Too many connections for user {user_id} (Redis)"
                        
            except Exception as e:
                logger.error(f"Redis rate limiting error: {e}")
        
        return True, "Connection allowed"
    
    async def register_connection(self, ip_address: str, user_id: Optional[str] = None):
        """Register a new connection for rate limiting"""
        self.ip_connections[ip_address] += 1
        if user_id:
            self.user_connections[user_id] += 1
    
    async def unregister_connection(self, ip_address: str, user_id: Optional[str] = None):
        """Unregister a connection from rate limiting"""
        if self.ip_connections[ip_address] > 0:
            self.ip_connections[ip_address] -= 1
        
        if user_id and self.user_connections[user_id] > 0:
            self.user_connections[user_id] -= 1
        
        # Update Redis counters if available
        redis = await self.get_redis()
        if redis:
            try:
                await redis.decr(f"ws_ip_connections:{ip_address}")
                if user_id:
                    await redis.decr(f"ws_user_connections:{user_id}")
            except Exception as e:
                logger.error(f"Redis connection cleanup error: {e}")
    
    def check_message_rate(self, client: WebSocketClient) -> bool:
        """Check if client's message rate is within limits"""
        message_rate = client.get_message_rate(60)  # messages per minute
        return message_rate <= self.config.max_messages_per_minute
    
    def check_message_size(self, message_size: int) -> bool:
        """Check if message size is within limits"""
        return message_size <= self.config.message_size_limit


class WebSocketSecurityManager:
    """Comprehensive WebSocket security management system"""
    
    def __init__(self, rate_limit_config: WebSocketRateLimitConfig = None):
        self.rate_limit_config = rate_limit_config or WebSocketRateLimitConfig()
        self.authenticator = WebSocketAuthenticator()
        self.rate_limiter = WebSocketRateLimiter(self.rate_limit_config)
        self.audit_logger = get_audit_logger()
        
        # Active connections
        self.connections: Dict[str, WebSocketClient] = {}
        
        # Security monitoring
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: List[str] = [
            r"<script.*?>",
            r"javascript:",
            r"eval\s*\(",
            r"document\.",
            r"window\.",
            r"\bselect\b.*\bfrom\b",
            r"\bunion\b.*\bselect\b",
        ]
        
        # Heartbeat management
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
    
    async def authenticate_websocket(
        self,
        websocket: WebSocket,
        client_id: str,
        ip_address: str,
        user_agent: str
    ) -> WebSocketClient:
        """Authenticate and create WebSocket client"""
        
        # Extract authentication information
        query_params = dict(websocket.query_params)
        headers = dict(websocket.headers)
        
        token = (
            self.authenticator.extract_token_from_query(query_params) or
            self.authenticator.extract_token_from_headers(headers)
        )
        
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            await self.audit_logger.log_event(
                AuditEventType.SECURITY_VIOLATION,
                ip_address=ip_address,
                action="websocket_connection_blocked",
                severity=AuditSeverity.HIGH,
                details={"reason": "blocked_ip", "client_id": client_id}
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        
        # Check connection rate limits
        allowed, reason = await self.rate_limiter.check_connection_limits(ip_address)
        if not allowed:
            await self.audit_logger.log_event(
                AuditEventType.SECURITY_VIOLATION,
                ip_address=ip_address,
                action="websocket_rate_limited",
                severity=AuditSeverity.MEDIUM,
                details={"reason": reason, "client_id": client_id}
            )
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=reason)
        
        # Attempt authentication
        user_session = None
        if token:
            user_session = await self.authenticator.authenticate_connection(websocket, token=token)
        
        # Create client object
        client = WebSocketClient(
            client_id=client_id,
            websocket=websocket,
            ip_address=ip_address,
            user_agent=user_agent,
            connected_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            user_session=user_session,
            is_authenticated=user_session is not None,
            authentication_token=token
        )
        
        # Set allowed actions based on authentication
        if user_session:
            client.allowed_actions = self._get_allowed_actions(user_session.role)
        else:
            client.allowed_actions = {"heartbeat", "authenticate"}
        
        # Register connection for rate limiting
        user_id = user_session.user_id if user_session else None
        await self.rate_limiter.register_connection(ip_address, user_id)
        
        # Store connection
        self.connections[client_id] = client
        
        # Start heartbeat monitoring
        self.heartbeat_tasks[client_id] = asyncio.create_task(
            self._monitor_heartbeat(client_id)
        )
        
        # Log successful connection
        await self.audit_logger.log_event(
            WebSocketEventType.CONNECTION_OPENED,
            user_id=user_session.user_id if user_session else None,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint="websocket",
            action="connect",
            result="success",
            severity=AuditSeverity.LOW,
            details={
                "client_id": client_id,
                "authenticated": client.is_authenticated
            }
        )
        
        return client
    
    async def disconnect_client(self, client_id: str):
        """Disconnect and clean up WebSocket client"""
        
        if client_id not in self.connections:
            return
        
        client = self.connections[client_id]
        
        try:
            # Cancel heartbeat monitoring
            if client_id in self.heartbeat_tasks:
                self.heartbeat_tasks[client_id].cancel()
                del self.heartbeat_tasks[client_id]
            
            # Unregister from rate limiting
            user_id = client.user_session.user_id if client.user_session else None
            await self.rate_limiter.unregister_connection(client.ip_address, user_id)
            
            # Log disconnection
            await self.audit_logger.log_event(
                WebSocketEventType.CONNECTION_CLOSED,
                user_id=user_id,
                ip_address=client.ip_address,
                endpoint="websocket",
                action="disconnect",
                result="success",
                severity=AuditSeverity.LOW,
                details={
                    "client_id": client_id,
                    "connection_duration": (datetime.now(timezone.utc) - client.connected_at).total_seconds(),
                    "message_count": client.message_count
                }
            )
            
            # Remove from connections
            del self.connections[client_id]
            
        except Exception as e:
            logger.error(f"Error disconnecting client {client_id}: {e}")
    
    async def validate_message(
        self,
        client_id: str,
        message_data: Union[str, bytes],
        message_type: WebSocketMessageType = None
    ) -> tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Validate incoming WebSocket message"""
        
        if client_id not in self.connections:
            return False, None, "Client not found"
        
        client = self.connections[client_id]
        
        # Check message rate limiting
        if not self.rate_limiter.check_message_rate(client):
            client.add_security_violation(
                WebSocketSecurityViolation.RATE_LIMIT_EXCEEDED,
                {"message_rate": client.get_message_rate()}
            )
            
            await self.audit_logger.log_event(
                WebSocketEventType.RATE_LIMITED,
                user_id=client.user_session.user_id if client.user_session else None,
                ip_address=client.ip_address,
                action="message_rate_limited",
                severity=AuditSeverity.MEDIUM,
                details={"client_id": client_id, "message_rate": client.get_message_rate()}
            )
            
            return False, None, "Rate limit exceeded"
        
        # Check message size
        message_size = len(message_data.encode() if isinstance(message_data, str) else message_data)
        if not self.rate_limiter.check_message_size(message_size):
            client.add_security_violation(
                WebSocketSecurityViolation.MALICIOUS_MESSAGE,
                {"message_size": message_size, "limit": self.rate_limit_config.message_size_limit}
            )
            
            return False, None, "Message too large"
        
        # Parse message
        try:
            if isinstance(message_data, str):
                message = json.loads(message_data)
            else:
                message = json.loads(message_data.decode())
        except json.JSONDecodeError as e:
            client.add_security_violation(
                WebSocketSecurityViolation.INVALID_MESSAGE_FORMAT,
                {"error": str(e)}
            )
            return False, None, "Invalid JSON format"
        
        # Validate message structure
        if not isinstance(message, dict) or "type" not in message:
            client.add_security_violation(
                WebSocketSecurityViolation.INVALID_MESSAGE_FORMAT,
                {"message": "Missing type field"}
            )
            return False, None, "Invalid message structure"
        
        # Check for malicious content
        message_str = json.dumps(message)
        for pattern in self.suspicious_patterns:
            import re
            if re.search(pattern, message_str, re.IGNORECASE):
                client.add_security_violation(
                    WebSocketSecurityViolation.MALICIOUS_MESSAGE,
                    {"pattern": pattern, "message": message_str[:200]}
                )
                
                await self.audit_logger.log_event(
                    WebSocketEventType.SECURITY_VIOLATION,
                    user_id=client.user_session.user_id if client.user_session else None,
                    ip_address=client.ip_address,
                    action="malicious_message_detected",
                    severity=AuditSeverity.HIGH,
                    details={"client_id": client_id, "pattern": pattern}
                )
                
                return False, None, "Malicious content detected"
        
        # Validate message type and permissions
        msg_type = message.get("type")
        if msg_type not in client.allowed_actions:
            client.add_security_violation(
                WebSocketSecurityViolation.UNAUTHORIZED_ACTION,
                {"action": msg_type, "allowed_actions": list(client.allowed_actions)}
            )
            
            return False, None, f"Action '{msg_type}' not allowed"
        
        # Update client activity
        client.add_message()
        
        # Log message reception
        await self.audit_logger.log_event(
            WebSocketEventType.MESSAGE_RECEIVED,
            user_id=client.user_session.user_id if client.user_session else None,
            ip_address=client.ip_address,
            action="message_received",
            severity=AuditSeverity.LOW,
            details={
                "client_id": client_id,
                "message_type": msg_type,
                "message_size": message_size
            }
        )
        
        return True, message, None
    
    async def send_secure_message(
        self,
        client_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """Send message to client with security logging"""
        
        if client_id not in self.connections:
            return False
        
        client = self.connections[client_id]
        
        try:
            # Add timestamp and security headers
            secure_message = {
                **message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_id": os.getenv("SERVER_ID", "unknown")
            }
            
            # Send message
            await client.websocket.send_json(secure_message)
            
            # Log message sending
            await self.audit_logger.log_event(
                WebSocketEventType.MESSAGE_SENT,
                user_id=client.user_session.user_id if client.user_session else None,
                ip_address=client.ip_address,
                action="message_sent",
                severity=AuditSeverity.LOW,
                details={
                    "client_id": client_id,
                    "message_type": message.get("type", "unknown")
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            return False
    
    async def broadcast_secure_message(
        self,
        message: Dict[str, Any],
        role_filter: Optional[UserRole] = None,
        exclude_client: Optional[str] = None
    ) -> int:
        """Broadcast message to multiple clients with role filtering"""
        
        sent_count = 0
        
        for client_id, client in self.connections.items():
            if client_id == exclude_client:
                continue
            
            # Apply role filter
            if role_filter and (not client.user_session or client.user_session.role != role_filter):
                continue
            
            if await self.send_secure_message(client_id, message):
                sent_count += 1
        
        return sent_count
    
    def _get_allowed_actions(self, role: UserRole) -> Set[str]:
        """Get allowed WebSocket actions based on user role"""
        
        base_actions = {"heartbeat", "subscribe", "unsubscribe"}
        
        role_permissions = {
            UserRole.SUPER_ADMIN: base_actions | {"admin", "broadcast", "manage_connections"},
            UserRole.ADMIN: base_actions | {"admin", "broadcast"},
            UserRole.ANALYST: base_actions | {"analysis", "recommendations"},
            UserRole.TRADER: base_actions | {"portfolio", "trade"},
            UserRole.VIEWER: base_actions | {"data"},
            UserRole.SERVICE_ACCOUNT: base_actions | {"data", "system"}
        }
        
        return role_permissions.get(role, base_actions)
    
    async def _monitor_heartbeat(self, client_id: str):
        """Monitor client heartbeat and timeout inactive connections"""
        
        while client_id in self.connections:
            try:
                client = self.connections[client_id]
                
                # Check if connection is still active
                if client.websocket.client_state != WebSocketState.CONNECTED:
                    break
                
                # Check heartbeat timeout
                now = datetime.now(timezone.utc)
                timeout_duration = timedelta(seconds=self.rate_limit_config.heartbeat_interval * 3)
                
                if now - client.last_activity > timeout_duration:
                    logger.info(f"WebSocket client {client_id} timed out")
                    
                    # Send timeout message
                    try:
                        await client.websocket.send_json({
                            "type": "error",
                            "message": "Connection timeout",
                            "code": "TIMEOUT"
                        })
                        await client.websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                    except:
                        pass
                    
                    break
                
                # Send heartbeat ping
                try:
                    await client.websocket.ping()
                except:
                    logger.info(f"Failed to ping client {client_id}")
                    break
                
                # Wait for next heartbeat check
                await asyncio.sleep(self.rate_limit_config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitoring error for {client_id}: {e}")
                break
        
        # Clean up connection
        await self.disconnect_client(client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        
        authenticated_count = sum(1 for c in self.connections.values() if c.is_authenticated)
        suspicious_count = sum(1 for c in self.connections.values() if c.is_suspicious)
        
        role_stats = defaultdict(int)
        for client in self.connections.values():
            if client.user_session:
                role_stats[client.user_session.role.value] += 1
            else:
                role_stats["anonymous"] += 1
        
        return {
            "total_connections": len(self.connections),
            "authenticated_connections": authenticated_count,
            "anonymous_connections": len(self.connections) - authenticated_count,
            "suspicious_connections": suspicious_count,
            "blocked_ips": len(self.blocked_ips),
            "connections_by_role": dict(role_stats),
            "active_heartbeat_monitors": len(self.heartbeat_tasks)
        }
    
    async def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Block an IP address from making WebSocket connections"""
        
        self.blocked_ips.add(ip_address)
        
        # Disconnect all clients from this IP
        clients_to_disconnect = [
            client_id for client_id, client in self.connections.items()
            if client.ip_address == ip_address
        ]
        
        for client_id in clients_to_disconnect:
            client = self.connections[client_id]
            try:
                await client.websocket.send_json({
                    "type": "error",
                    "message": f"Access denied: {reason}",
                    "code": "BLOCKED"
                })
                await client.websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            except:
                pass
            
            await self.disconnect_client(client_id)
        
        # Log IP blocking
        await self.audit_logger.log_event(
            AuditEventType.SECURITY_VIOLATION,
            ip_address=ip_address,
            action="ip_blocked",
            severity=AuditSeverity.CRITICAL,
            details={
                "reason": reason,
                "disconnected_clients": len(clients_to_disconnect)
            }
        )
        
        logger.warning(f"Blocked IP {ip_address}: {reason}")


# Global WebSocket security manager
_websocket_security: Optional[WebSocketSecurityManager] = None


def get_websocket_security() -> WebSocketSecurityManager:
    """Get global WebSocket security manager"""
    global _websocket_security
    if _websocket_security is None:
        _websocket_security = WebSocketSecurityManager()
    return _websocket_security


# Security decorator for WebSocket endpoints
def secure_websocket(
    require_auth: bool = False,
    allowed_roles: List[UserRole] = None,
    rate_limit_override: Optional[WebSocketRateLimitConfig] = None
):
    """Decorator for securing WebSocket endpoints"""
    
    def decorator(func: Callable):
        async def wrapper(websocket: WebSocket, *args, **kwargs):
            security_manager = get_websocket_security()
            
            # Extract client information
            client_id = kwargs.get("client_id") or str(uuid.uuid4())
            ip_address = websocket.client.host
            user_agent = websocket.headers.get("user-agent", "Unknown")
            
            try:
                # Authenticate connection
                client = await security_manager.authenticate_websocket(
                    websocket, client_id, ip_address, user_agent
                )
                
                # Check authentication requirements
                if require_auth and not client.is_authenticated:
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
                    return
                
                # Check role requirements
                if allowed_roles and client.user_session:
                    if client.user_session.role not in allowed_roles:
                        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Insufficient permissions")
                        return
                
                # Accept connection
                await websocket.accept()
                
                # Call original function with security context
                kwargs['security_manager'] = security_manager
                kwargs['client'] = client
                
                return await func(websocket, *args, **kwargs)
                
            except HTTPException as e:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=e.detail)
            except Exception as e:
                logger.error(f"WebSocket security error: {e}")
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Internal error")
            finally:
                # Clean up connection
                if client_id in security_manager.connections:
                    await security_manager.disconnect_client(client_id)
        
        return wrapper
    return decorator


# Utility functions for secure WebSocket message handling
async def send_error_message(websocket: WebSocket, error_code: str, message: str):
    """Send error message to WebSocket client"""
    try:
        await websocket.send_json({
            "type": "error",
            "code": error_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")


async def validate_subscription_permissions(
    client: WebSocketClient,
    symbols: List[str]
) -> tuple[List[str], List[str]]:
    """Validate client permissions for symbol subscriptions"""
    allowed_symbols = []
    denied_symbols = []
    
    for symbol in symbols:
        # Basic validation - in production, this would check against
        # user's subscription level, data permissions, etc.
        if len(symbol) <= 10 and symbol.isalpha():
            allowed_symbols.append(symbol.upper())
        else:
            denied_symbols.append(symbol)
    
    # Role-based filtering
    if client.user_session:
        if client.user_session.role in [UserRole.VIEWER]:
            # Limit viewers to basic symbols
            allowed_symbols = allowed_symbols[:10]  # Max 10 symbols
    else:
        # Anonymous users get very limited access
        allowed_symbols = allowed_symbols[:3]  # Max 3 symbols
    
    return allowed_symbols, denied_symbols