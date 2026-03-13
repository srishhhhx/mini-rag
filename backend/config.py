from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

# Project root is two levels up from this file (backend/config.py → pdf-reader/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

# Inject .env directly into os.environ early so that @traceable and other decorators have access to it at import time.
load_dotenv(_ENV_FILE)

class Settings(BaseSettings):
    # Groq LLM
    groq_api_key: str
    groq_model_primary: str = "llama-3.3-70b-versatile"
    groq_model_fallback: str = "llama-3.1-8b-instant"

    # Embedding (HuggingFace Inference API)
    hf_api_key: str
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Reranking (Cohere API)
    cohere_api_key: str

    # LangSmith observability
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "pdf-chat-rag"

    # Session management
    session_ttl_minutes: int = 30

    # Retrieval config
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    rrf_k: int = 60
    confidence_threshold: float = 0.4

    # Chunking
    chunk_size: int = 2000  # characters, ~500 tokens

    # CORS
    frontend_url: str = "http://localhost:3000"

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
