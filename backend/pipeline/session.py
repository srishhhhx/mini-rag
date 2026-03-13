from dataclasses import dataclass, field
from typing import Literal, Optional
import time
import asyncio
import logging
import faiss
from rank_bm25 import BM25Okapi

from models import ChunkMeta
from pipeline.memory import ChatMemory

logger = logging.getLogger(__name__)

class SessionNotFoundError(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found or expired.")


@dataclass
class SessionData:
    faiss_index: Optional[faiss.Index] = None
    bm25_index: Optional[BM25Okapi] = None
    chunks: list[ChunkMeta] = field(default_factory=list)
    chunk_count: int = 0
    chat_memory: ChatMemory = field(default_factory=ChatMemory)
    status: Literal["processing", "ready", "error"] = "processing"
    page_count: int = 0
    doc_title: Optional[str] = None
    doc_topic: Optional[str] = None
    error_message: Optional[str] = None
    last_accessed: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_minutes: int = 30):
        self._store: dict[str, SessionData] = {}
        self._ttl_seconds = ttl_minutes * 60
        self._lock = asyncio.Lock()

    async def create(self, session_id: str) -> SessionData:
        """Create a new session (or reset an existing one)."""
        async with self._lock:
            session = SessionData()
            self._store[session_id] = session
            return session

    async def get(self, session_id: str) -> Optional[SessionData]:
        """Retrieve a session and update its last_accessed timestamp."""
        async with self._lock:
            session = self._store.get(session_id)
            if session:
                session.last_accessed = time.time()
            return session

    async def get_or_raise(self, session_id: str) -> SessionData:
        """Retrieve a session or raise SessionNotFoundError."""
        session = await self.get(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        return session

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        async with self._lock:
            session = self._store.pop(session_id, None)
            if session:
                session.faiss_index = None
                session.bm25_index = None
                session.chunks = []
            return session is not None

    def exists(self, session_id: str) -> bool:
        return session_id in self._store

    @property
    def active_count(self) -> int:
        return len(self._store)

    async def start_eviction_loop(self):
        """Background task: evict sessions idle longer than TTL."""
        while True:
            await asyncio.sleep(600)  # run every 10 minutes
            await self._evict_stale()

    async def _evict_stale(self) -> None:
        now = time.time()
        async with self._lock:
            expired = [
                sid
                for sid, session in self._store.items()
                if now - session.last_accessed > self._ttl_seconds
            ]
            for sid in expired:
                session = self._store.pop(sid)
                session.faiss_index = None
                session.bm25_index = None
                session.chunks = []
        if expired:
            logger.info(f"Evicted {len(expired)} expired session(s): {expired}")


# Singleton instance used throughout the app
session_store = SessionStore()
