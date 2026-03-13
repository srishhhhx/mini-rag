"""
Generation pipeline:
  Build system prompt + context + history → stream Groq LLM response via SSE
  Handles low-confidence (no-hallucination) case.
  LangSmith @traceable for observability.
"""

import json
import logging
from typing import AsyncGenerator

from groq import AsyncGroq
from langsmith import traceable

from config import get_settings
from models import ChunkMeta, SourceChunk
from pipeline.memory import ChatMemory
from pipeline.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document analyst. Answer questions using ONLY the document context provided. Never use outside knowledge.

RESPONSE RULES:
1. Base every claim on the provided context chunks. 
2. Cite page numbers inline using (Page N) format — not at the end, but immediately after the claim.
3. For factual or numerical questions: answer in 1-3 sentences maximum.
4. For broad questions (summarise, explain, describe): answer in a structured paragraph.
5. For comparisons: use the exact figures from the context, do not approximate.
6. Never fabricate numbers, names, dates, or statistics.
7. If the context contains partial information, state what is available and explicitly note what is missing.
8. When the context contains lists or multiple items, include ALL relevant items in your answer. Do not truncate or pick and choose.
9. Maintain a polite, friendly, and professional tone.
"""

def _build_prompt(
    query: str,
    chunks: list[ChunkMeta],
    history: str,
    doc_title: str | None,
) -> list[dict]:
    """Build the message list for the chat completion call."""
    context_parts = []
    for chunk in chunks:
        header = f"[Page {chunk.page}]"
        if chunk.section_header:
            header += f" — {chunk.section_header}"
        context_parts.append(f"{header}\n{chunk.text}")
    context = "\n\n---\n\n".join(context_parts)

    doc_line = f"Document: {doc_title}\n\n" if doc_title else ""
    
    # History BEFORE context — follow-up resolution happens before chunk reading
    history_block = f"Conversation so far:\n{history}\n\n" if history else ""
    
    user_content = (
        f"{doc_line}"
        f"{history_block}"
        f"Relevant document excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite page numbers inline):"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── Low-confidence fallback ────────────────────────────────────────────────────

LOW_CONFIDENCE_RESPONSE = (
    "I couldn't find relevant information about this in the document. "
    "Please try rephrasing your question or ask about topics covered in the uploaded PDF."
)


# ── Streaming generator ────────────────────────────────────────────────────────

@traceable(name="generate_answer")
async def generate_streaming(
    query: str,
    retrieval: RetrievalResult,
    memory: ChatMemory,
    groq_client: AsyncGroq,
    doc_title: str | None = None,
    max_tokens: int = 1024,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields Server-Sent Events (SSE) strings.

    SSE format:
      data: {"token": "..."}\n\n          ← for each token
      data: {"done": true, "sources": [...], "confident": bool}\n\n   ← final event
    """
    # Low-confidence: skip LLM entirely
    is_greeting = query.strip().lower() in [
        "hi", "hello", "hey", "greetings", "good morning", 
        "good afternoon", "good evening", "hi there", "hey there", "hola"
    ]
    
    if not retrieval.confident and not is_greeting:
        yield f"data: {json.dumps({'token': LOW_CONFIDENCE_RESPONSE})}\n\n"
        sources = _build_sources(retrieval)
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'confident': False})}\n\n"
        memory.add(query, LOW_CONFIDENCE_RESPONSE)
        return

    # Build prompt
    history_str = memory.format()
    messages = _build_prompt(query, retrieval.chunks, history_str, doc_title)
    
    # Extract user content from the prompt
    user_content = next(m["content"] for m in messages if m["role"] == "user")

    full_response = ""
    settings = get_settings()

    try:
        # Use Groq primary
        model = settings.groq_model_primary
        stream = await groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield f"data: {json.dumps({'token': delta})}\n\n"

    except Exception as e:
        logger.warning(f"Primary model failed ({e}), falling back to {settings.groq_model_fallback}")
        # Fallback to smaller model on Groq
        try:
            stream = await groq_client.chat.completions.create(
                model=settings.groq_model_fallback,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'token': delta})}\n\n"
        except Exception as e2:
            error_msg = "An error occurred while generating the response. Please try again."
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
            full_response = error_msg
            logger.error(f"Fallback model also failed: {e2}")

    # Store in memory
    if full_response:
        memory.add(query, full_response)

    # Final SSE event with sources
    sources = _build_sources(retrieval)
    yield f"data: {json.dumps({'done': True, 'sources': sources, 'confident': True})}\n\n"


def _build_sources(retrieval: RetrievalResult) -> list[dict]:
    """Convert retrieved chunks to JSON-serializable source list."""
    sources = []
    for chunk, score in zip(retrieval.chunks, retrieval.scores):
        sources.append(
            SourceChunk(
                page=chunk.page,
                text=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                section_header=chunk.section_header,
                chunk_type=chunk.chunk_type,
                score=round(float(score), 4),
            ).model_dump()
        )
    return sources
