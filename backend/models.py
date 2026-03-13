from pydantic import BaseModel
from typing import Literal, Optional


# ── Request models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str


# ── Response models ─────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    status: Literal["processing"]
    session_id: str
    message: str


class SessionStatusResponse(BaseModel):
    status: Literal["processing", "ready", "error"]
    chunk_count: int
    page_count: int
    doc_title: Optional[str] = None
    doc_topic: Optional[str] = None
    error_message: Optional[str] = None


class SourceChunk(BaseModel):
    page: int
    text: str
    section_header: Optional[str] = None
    chunk_type: str  # "paragraph" | "list" | "table"
    score: float


class DeleteSessionResponse(BaseModel):
    status: Literal["deleted"]
    session_id: str


# ── Internal data structures ─────────────────────────────────────────────────

class ChunkMeta(BaseModel):
    text: str
    page: int
    chunk_idx: int
    section_header: Optional[str] = None
    chunk_type: str = "paragraph"
    word_count: int = 0
    char_count: int = 0
