"""
Retrieval pipeline:
  query embed → FAISS cosine search (top-k)
                        + BM25 keyword search (top-k)
  → Reciprocal Rank Fusion (RRF)
  → confidence check
  → Cohere Rerank API
  → top-3 ChunkMeta with scores
"""

import logging
from dataclasses import dataclass

import cohere
import numpy as np
from langsmith import traceable

from config import get_settings
from models import ChunkMeta
from pipeline.session import SessionData
from pipeline.ingestion import embed_texts

logger = logging.getLogger(__name__)

# ── Cohere client singleton ────────────────────────────────────────────────────
_cohere_client: cohere.Client | None = None


def get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        settings = get_settings()
        logger.info("Initialising Cohere client")
        _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
    return _cohere_client


# ── RRF fusion ─────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    faiss_indices: list[int],
    bm25_indices: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.
    Returns list of (chunk_index, rrf_score) sorted descending.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(faiss_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Main retrieval function ────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunks: list[ChunkMeta]
    scores: list[float]
    confident: bool


def build_retrieval_query(question: str, session: SessionData) -> str:
    last_answer = session.chat_memory.get_last_assistant()
    if not last_answer:
        return question
    return f"{question} {last_answer}"[:512]  # cap length


@traceable(name="hybrid_retrieve")
def retrieve(query: str, session: SessionData) -> RetrievalResult:
    """
    Hybrid FAISS + BM25 retrieval with RRF fusion + Cohere reranking.
    Returns top-K reranked chunks and a confidence flag.
    """
    if session.status != "ready" or session.faiss_index is None:
        raise ValueError("Session is not ready for retrieval.")

    settings = get_settings()
    top_k = settings.top_k_retrieval
    top_rerank = settings.top_k_rerank

    expanded_query = build_retrieval_query(query, session)

    # 1. Embed query via HF API (returns L2-normalised array)
    query_vec = embed_texts([expanded_query])  # shape: (1, dim)

    # 2. FAISS cosine search
    n_candidates = min(top_k, session.faiss_index.ntotal)
    faiss_scores, faiss_idxs = session.faiss_index.search(query_vec, n_candidates)
    faiss_idxs = [i for i in faiss_idxs[0].tolist() if i >= 0]

    # 3. BM25 keyword search
    from pipeline.ingestion import tokenize_for_bm25
    tokenized_query = tokenize_for_bm25(expanded_query)
    bm25_scores_all = session.bm25_index.get_scores(tokenized_query)
    bm25_ranked = sorted(
        range(len(bm25_scores_all)), key=lambda i: bm25_scores_all[i], reverse=True
    )[:top_k]

    # 4. RRF fusion → top_k merged candidates
    fused = reciprocal_rank_fusion(faiss_idxs, bm25_ranked, k=settings.rrf_k)
    seen: set[int] = set()
    merged: list[tuple[int, float]] = []
    for idx, rrf_score in fused:
        if idx not in seen and 0 <= idx < len(session.chunks):
            seen.add(idx)
            merged.append((idx, rrf_score))
        if len(merged) >= top_k:
            break

    # 5. Cohere reranking + confidence via Cohere relevance score
    candidates = [(idx, session.chunks[idx]) for idx, _ in merged]

    if not candidates:
        return RetrievalResult(chunks=[], scores=[], confident=False)

    try:
        co = get_cohere_client()
        rerank_response = co.rerank(
            query=expanded_query,
            documents=[chunk.text for _, chunk in candidates],
            model="rerank-v3.5",
            top_n=min(top_rerank, len(candidates)),
        )
        # Map back from rerank result indices to original candidates
        top_chunks = [candidates[r.index][1] for r in rerank_response.results]
        top_scores = [r.relevance_score for r in rerank_response.results]
        # Confidence = best Cohere score above threshold
        # Cohere scores: ~0.02 for irrelevant, 0.3–0.9 for relevant
        confident = top_scores[0] >= settings.confidence_threshold
    except Exception as e:
        logger.warning(f"Cohere rerank failed: {e}")
        top_chunks = [chunk for _, chunk in candidates[:top_rerank]]
        top_scores = [score for _, score in merged[:top_rerank]]
        confident = False  # honest fallback

    return RetrievalResult(chunks=top_chunks, scores=top_scores, confident=confident)
