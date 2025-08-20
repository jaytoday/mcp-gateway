"""
Session management for MCP Gateway
Provides session storage and management with database-ready abstraction
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a user session with isolated MCP server connections"""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime] = None
    connections: Dict[str, Any] = field(default_factory=dict)  # server_uuid -> connection data
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for storage"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "connections": self.connections,
            "metadata": self.metadata,
            "active": self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from stored dictionary"""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            connections=data.get("connections", {}),
            metadata=data.get("metadata", {}),
            active=data.get("active", True)
        )
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        if not self.active:
            return True
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def touch(self) -> None:
        """Update last accessed time"""
        self.last_accessed = datetime.now()


class SessionStore(ABC):
    """Abstract base class for session storage implementations"""
    
    @abstractmethod
    async def create_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Create a new session"""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data by ID"""
        pass
    
    @abstractmethod
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update existing session data"""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a session"""
        pass
    
    @abstractmethod
    async def list_sessions(self) -> List[str]:
        """List all session IDs"""
        pass
    
    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count of deleted sessions"""
        pass


class InMemorySessionStore(SessionStore):
    """In-memory implementation of SessionStore for development/testing"""
    
    def __init__(self):
        """Initialize in-memory storage"""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Create a new session in memory"""
        async with self._lock:
            self.sessions[session_id] = data
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data from memory"""
        async with self._lock:
            return self.sessions.get(session_id)
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data in memory"""
        async with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id] = data
            else:
                raise KeyError(f"Session {session_id} not found")
    
    async def delete_session(self, session_id: str) -> None:
        """Remove session from memory"""
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    async def list_sessions(self) -> List[str]:
        """List all session IDs in memory"""
        async with self._lock:
            return list(self.sessions.keys())
    
    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from memory"""
        async with self._lock:
            expired_sessions = []
            for session_id, session_data in self.sessions.items():
                try:
                    session = Session.from_dict(session_data)
                    if session.is_expired():
                        expired_sessions.append(session_id)
                except Exception as e:
                    logger.error(f"Error checking session {session_id} expiration: {e}")
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            return len(expired_sessions)


class SessionManager:
    """Manages session lifecycle and operations"""
    
    def __init__(self, store: SessionStore, default_ttl_minutes: int = 60):
        """
        Initialize session manager
        
        Args:
            store: SessionStore implementation
            default_ttl_minutes: Default session TTL in minutes
        """
        self.store = store
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_callbacks: List[callable] = []
    
    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session with generated UUID"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + self.default_ttl
        
        session = Session(
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
            metadata=metadata or {},
            connections={},
            active=True
        )
        
        await self.store.create_session(session_id, session.to_dict())
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve and validate a session"""
        session_data = await self.store.get_session(session_id)
        if not session_data:
            return None
        
        session = Session.from_dict(session_data)
        
        # Check if expired
        if session.is_expired():
            await self.delete_session(session_id)
            return None
        
        # Touch the session to update last accessed time
        session.touch()
        await self.update_session(session)
        
        return session
    
    async def update_session(self, session: Session) -> None:
        """Save session changes to store"""
        await self.store.update_session(session.session_id, session.to_dict())
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and return success status"""
        try:
            await self.store.delete_session(session_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def list_active_sessions(self) -> List[Session]:
        """List all active (non-expired) sessions"""
        session_ids = await self.store.list_sessions()
        active_sessions = []
        
        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session and not session.is_expired():
                active_sessions.append(session)
        
        return active_sessions
    
    async def add_server_connection(self, session_id: str, server_uuid: str, connection_data: Dict[str, Any]) -> None:
        """Add a server connection to a session"""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.connections[server_uuid] = connection_data
        await self.update_session(session)
    
    async def remove_server_connection(self, session_id: str, server_uuid: str) -> None:
        """Remove a server connection from a session"""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if server_uuid in session.connections:
            del session.connections[server_uuid]
            await self.update_session(session)
    
    async def get_session_connections(self, session_id: str) -> Dict[str, Any]:
        """Get all server connections for a session"""
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        return session.connections
    
    async def start_cleanup_task(self, interval_seconds: int = 300) -> None:
        """Start background task to cleanup expired sessions"""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.warning("Cleanup task already running")
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval_seconds))
        logger.info(f"Started session cleanup task with {interval_seconds}s interval")
    
    async def stop_cleanup_task(self) -> None:
        """Stop the cleanup background task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped session cleanup task")
    
    def register_cleanup_callback(self, callback: callable) -> None:
        """Register a callback to be called when sessions are cleaned up"""
        self._cleanup_callbacks.append(callback)
    
    async def _cleanup_loop(self, interval_seconds: int) -> None:
        """Background loop for cleaning up expired sessions"""
        while True:
            try:
                # Get expired sessions before cleanup
                session_ids = await self.store.list_sessions()
                expired_sessions = []
                
                for session_id in session_ids:
                    session_data = await self.store.get_session(session_id)
                    if session_data:
                        try:
                            session = Session.from_dict(session_data)
                            if session.is_expired():
                                expired_sessions.append(session_id)
                        except Exception as e:
                            logger.error(f"Error checking session {session_id}: {e}")
                
                # Call cleanup callbacks for each expired session
                for session_id in expired_sessions:
                    for callback in self._cleanup_callbacks:
                        try:
                            await callback(session_id)
                        except Exception as e:
                            logger.error(f"Error in cleanup callback for session {session_id}: {e}")
                
                # Now cleanup the sessions from store
                deleted_count = await self.store.cleanup_expired_sessions()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired sessions")
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(interval_seconds)