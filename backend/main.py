"""
FastAPI application — all routes and startup logic.

Routes:
  POST   /upload                    — ingest a PDF (async background task)
  GET    /session/{session_id}/status — poll ingestion progress
  POST   /chat                      — SSE streaming chat
  DELETE /session/{session_id}      — clean up session
"""

import asyncio
import logging
import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq, Groq

from config import get_settings
from models import (
    ChatRequest,
    DeleteSessionResponse,
    SessionStatusResponse,
    UploadResponse,
)
from pipeline.generation import generate_streaming
from pipeline.ingestion import ingest_pdf
from pipeline.retrieval import retrieve
from pipeline.session import session_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF Chat API",
    description="Mini RAG system — upload a PDF and chat with it.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients & Observability (shared, thread-safe) ───────────────────────────
groq_sync = Groq(api_key=settings.groq_api_key)
groq_async = AsyncGroq(api_key=settings.groq_api_key)


# ── Startup: warm up models + start eviction loop ─────────────────────────────

@app.on_event("startup")
async def startup():
    # embedding model, reranking via Cohere
    logger.info("Starting session eviction loop...")
    asyncio.create_task(session_store.start_eviction_loop())
    logger.info("Backend ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Accept a PDF upload and start the ingestion pipeline in the background.
    Frontend should poll GET /session/{session_id}/status to track progress.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read file bytes
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Create/reset session — clears previous document and history
    session = session_store.create(session_id)

    # Run ingestion in background so we can return immediately
    background_tasks.add_task(
        _run_ingestion_sync, file_bytes=file_bytes, session_id=session_id
    )

    return UploadResponse(
        status="processing",
        session_id=session_id,
        message="Document received. Processing started.",
    )


async def _run_ingestion_sync(file_bytes: bytes, session_id: str):
    """Wrapper to run async ingestion and handle top-level errors."""
    session = session_store.get(session_id)
    if not session:
        logger.warning(f"Session {session_id} disappeared before ingestion started.")
        return
    await ingest_pdf(file_bytes=file_bytes, session=session, groq_client=groq_sync)


@app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Poll this endpoint to check ingestion progress."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    return SessionStatusResponse(
        status=session.status,
        chunk_count=len(session.chunks),
        page_count=session.page_count,
        doc_title=session.doc_title,
        doc_topic=session.doc_topic,
        error_message=session.error_message,
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream an answer to a question about the uploaded document.
    Response is Server-Sent Events (text/event-stream).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session = session_store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    if session.status == "processing":
        raise HTTPException(status_code=409, detail="Document is still being processed. Please wait.")

    if session.status == "error":
        raise HTTPException(
            status_code=422,
            detail=session.error_message or "Document processing failed.",
        )

    if session.status != "ready" or not session.faiss_index:
        raise HTTPException(status_code=404, detail="No document found in this session. Please upload a PDF first.")

    # Retrieval (sync, fast)
    try:
        retrieval_result = retrieve(query=request.question, session=session)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Stream generation
    return StreamingResponse(
        generate_streaming(
            query=request.question,
            retrieval=retrieval_result,
            memory=session.chat_memory,
            groq_client=groq_async,
            doc_title=session.doc_title,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


@app.delete("/session/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str):
    """Explicitly clean up a session and free memory."""
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return DeleteSessionResponse(status="deleted", session_id=session_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
