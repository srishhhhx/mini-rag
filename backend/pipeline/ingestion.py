"""
Ingestion pipeline:
  PDF parse → sentence-aware chunk → regex metadata → doc-level Groq metadata
  → embed (HuggingFace Inference API) → FAISS IndexFlatIP + BM25Okapi
"""

import io
import json
import re
import logging
from typing import Optional

import fitz  # PyMuPDF
import faiss
import nltk
import numpy as np
from groq import Groq
# from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langsmith import traceable

from config import get_settings
from models import ChunkMeta
from pipeline.session import SessionData
from utils.pdf_utils import (
    is_scanned_pdf,
    get_page_count,
    extract_section_header,
    detect_chunk_type,
)

# NLTK punkt_tab is pre-downloaded via setup; attempt download only if missing
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    import ssl
    try:
        _ctx = ssl._create_unverified_context
        ssl._create_default_https_context = _ctx
    except AttributeError:
        pass
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)

# ── Local Embeddings client singleton ─────────────────────────────────────────
_embed_client: Optional[SentenceTransformer] = None


def get_embed_client() -> SentenceTransformer:
    global _embed_client
    if _embed_client is None:
        settings = get_settings()
        logger.info(f"Loading local embedding model: {settings.embed_model}")
        _embed_client = SentenceTransformer(settings.embed_model)
    return _embed_client


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using local SentenceTransformers.
    Returns an L2-normalised float32 numpy array of shape (len(texts), dim).
    """
    client = get_embed_client()
    
    # encode() returns a numpy array. normalize_embeddings=True applies L2 norm.
    arr = client.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    return np.array(arr, dtype="float32")


# ── Sentence-aware chunker ────────────────────────────────────────────────────

def sentence_aware_chunk(
    pages: list[tuple[str, int]],
    chunk_size: int = 500,
) -> list[dict]:
    """
    Split document text into chunks that never break mid-sentence.
    Each chunk inherits the page number of the first sentence in it.
    """
    all_sentences: list[tuple[str, int]] = []
    for page_text, page_num in pages:
        cleaned = page_text.strip()
        if not cleaned:
            continue
        sentences = nltk.sent_tokenize(cleaned)
        for sent in sentences:
            s = sent.strip()
            if s:
                all_sentences.append((s, page_num))

    if not all_sentences:
        return []

    chunks: list[dict] = []
    current_sentences: list[str] = []
    current_page: int = all_sentences[0][1]
    current_len: int = 0

    for sent, page_num in all_sentences:
        sent_len = len(sent)
        if current_len + sent_len + 1 > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({"text": chunk_text, "page": current_page})
            # 1-sentence overlap
            current_sentences = [current_sentences[-1]]
            current_page = page_num
            current_len = len(current_sentences[0])

        current_sentences.append(sent)
        if page_num > current_page:
            current_page = page_num
        current_len += sent_len + 1

    if current_sentences:
        chunks.append({"text": " ".join(current_sentences), "page": current_page})

    return chunks


# ── Doc-level metadata via Groq ───────────────────────────────────────────────

def _extract_doc_metadata(first_text: str, groq_client: Groq, settings) -> dict:
    """
    Single Groq call to extract title, topic, and document type.
    Returns {"title": ..., "topic": ..., "doc_type": ...}
    """
    snippet = first_text[:300].strip()
    prompt = (
        "You are an assistant that extracts document metadata. "
        "From the following document excerpt, extract the title, topic, and document type. "
        "Reply ONLY with a JSON object like: "
        '{"title": "...", "topic": "...", "doc_type": "..."}\n\n'
        f"Excerpt:\n{snippet}"
    )
    try:
        response = groq_client.chat.completions.create(
            model=settings.groq_model_primary,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning(f"Doc metadata extraction failed: {e}")
    return {"title": None, "topic": None, "doc_type": None}


# ── Main ingestion function ───────────────────────────────────────────────────

@traceable(name="ingest_document")
async def ingest_pdf(
    file_bytes: bytes,
    session: SessionData,
    groq_client: Groq,
) -> None:
    """
    Full ingestion pipeline. Mutates `session` in-place.
    Sets session.status to 'ready' on success or 'error' on failure.
    """
    settings = get_settings()
    try:
        # 1. Parse PDF
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")

        # 2. Scanned PDF check
        if is_scanned_pdf(doc):
            session.status = "error"
            session.error_message = (
                "PDF appears to be scanned or image-only. "
                "Text extraction failed. Please upload a text-based PDF."
            )
            return

        session.page_count = get_page_count(doc)

        # 3. Extract text per page
        pages: list[tuple[str, int]] = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((text, page_num))
        doc.close()

        if not pages:
            session.status = "error"
            session.error_message = "No extractable text found in this PDF."
            return

        # 4. Sentence-aware chunking
        raw_chunks = sentence_aware_chunk(pages, chunk_size=settings.chunk_size)

        if not raw_chunks:
            session.status = "error"
            session.error_message = "Could not extract any text chunks from the document."
            return

        # 5. Extract per-chunk regex metadata
        chunk_metas: list[ChunkMeta] = []
        for idx, rc in enumerate(raw_chunks):
            text = rc["text"]
            chunk_metas.append(
                ChunkMeta(
                    text=text,
                    page=rc["page"],
                    chunk_idx=idx,
                    section_header=extract_section_header(text),
                    chunk_type=detect_chunk_type(text),
                    word_count=len(text.split()),
                    char_count=len(text),
                )
            )

        # 6. Extract document-level metadata (single Groq call)
        first_text = pages[0][0] if pages else ""
        meta = _extract_doc_metadata(first_text, groq_client, settings)
        session.doc_title = meta.get("title")
        session.doc_topic = meta.get("topic")

        # 7. Embed all chunks via HuggingFace Inference API (batched, L2-normalised)
        texts = [c.text for c in chunk_metas]
        logger.info(f"Embedding {len(texts)} chunks via HF API...")
        embeddings = embed_texts(texts)

        # 8. Build FAISS IndexFlatIP (cosine similarity via normalised vectors)
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)

        # 9. Build BM25 index (keyword search)
        tokenized_texts = [text.lower().split() for text in texts]
        bm25_index = BM25Okapi(tokenized_texts)

        # 10. Commit to session
        session.chunks = chunk_metas
        session.faiss_index = faiss_index
        session.bm25_index = bm25_index
        session.chat_memory.clear()
        session.status = "ready"

        logger.info(
            f"Ingestion complete: {len(chunk_metas)} chunks, "
            f"{session.page_count} pages, title='{session.doc_title}'"
        )

    except Exception as e:
        logger.exception("Ingestion failed")
        session.status = "error"
        session.error_message = f"Ingestion failed: {str(e)}"
