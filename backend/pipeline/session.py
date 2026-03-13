from dataclasses import dataclass, field
from typing import Literal, Optional
import time
import asyncio
import faiss
from rank_bm25 import BM25Okapi

from models import ChunkMeta
from pipeline.memory import ChatMemory


@dataclass
class SessionData:
    faiss_index: Optional[faiss.Index] = None
    bm25_index: Optional[BM25Okapi] = None
    chunks: list[ChunkMeta] = field(default_factory=list)
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

    def create(self, session_id: str) -> SessionData:
        """Create a new session (or reset an existing one)."""
        session = SessionData()
        self._store[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[SessionData]:
        """Retrieve a session and update its last_accessed timestamp."""
        session = self._store.get(session_id)
        if session:
            session.last_accessed = time.time()
        return session

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        return self._store.pop(session_id, None) is not None

    def exists(self, session_id: str) -> bool:
        return session_id in self._store

    async def start_eviction_loop(self):
        """Background task: evict sessions idle longer than TTL."""
        while True:
            await asyncio.sleep(600)  # run every 10 minutes
            now = time.time()
            expired = [
                sid
                for sid, session in self._store.items()
                if now - session.last_accessed > self._ttl_seconds
            ]
            for sid in expired:
                del self._store[sid]


# Singleton instance used throughout the app
session_store = SessionStore()
