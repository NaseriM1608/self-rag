"""Application settings.

Secrets are read from the environment / .env only. Every other field has an
in-code default and can be overridden by an environment variable of the same
name (case-insensitive), e.g. LLM_MODEL, N_RESULTS, INDEX_DIR.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _default_index_dir() -> Path:
    """Keep indexes out of cloud-synced folders (OneDrive/Dropbox):

    SQLite-backed stores (ChromaDB today, Neo4j volumes tomorrow) corrupt
    under file-locking sync engines, so default to a per-user data dir.
    """
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "self-rag" / "index"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Secrets — environment / .env only, never hardcoded here ---
    # Accepts OPENROUTER_API_KEY or the legacy GROQ_API_KEY name.
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or ""
    )
    langsmith_api_key: str = ""

    # --- LLM (OpenRouter, OpenAI-compatible endpoint) ---
    llm_model: str = "stealth/ox-alpha"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    # Temperature escalation across grounding-failure retries: regeneration
    # at temperature 0 reproduces the same ungrounded answer forever.
    retry_temperature_step: float = 0.35
    max_retry_temperature: float = 0.7

    # --- LangSmith tracing (LANGSMITH_TRACING=true + api key to enable) ---
    langsmith_tracing: bool = False
    langsmith_project: str = "self-rag"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- Retrieval / index ---
    # Runtime retrieval backend (see retrievers.get_retriever); 'dense' is
    # the frozen Chroma baseline used only by evals.
    default_retriever: str = "hybrid"
    documents_path: str = "documents"
    index_dir: Path = Field(default_factory=_default_index_dir)
    collection_name: str = "my_collection"
    embedding_model_name: str = "BAAI/bge-m3"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    n_results: int = 5

    # --- Neo4j graph store (chunks + vector index today, KG later) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    vector_index_name: str = "chunk_embeddings"
    # Dense output dimension of BAAI/bge-m3.
    embedding_dims: int = 1024

    # --- Graph control ---
    # True upper bound on chain invocations per run. Grading costs one call
    # per candidate document, so the typical happy path already spends ~8.
    max_llm_calls: int = 25

    def require_llm_key(self) -> str:
        if not self.llm_api_key:
            raise RuntimeError(
                "No API key found — set OPENROUTER_API_KEY in .env "
                "(see .env.example)."
            )
        return self.llm_api_key


settings = Settings()
