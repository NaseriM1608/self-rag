"""Corpus ingestion utilities shared by every retrieval backend.

Loads documents, splits them with deterministic content-hash chunk ids, and
produces embeddings. Storage-specific sync/search live with their backends
(neo4j_store.py for the runtime store; retrievers.DenseRetriever keeps a
Chromadb baseline purely for eval comparisons).
"""

import hashlib
import logging
from functools import lru_cache

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def embedding_model() -> SentenceTransformer:
    logger.info("Loading embedding model %s", settings.embedding_model_name)
    return SentenceTransformer(settings.embedding_model_name)


def load_documents() -> list[Document]:
    loader = DirectoryLoader(
        settings.documents_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No .txt documents found in {settings.documents_path}")
    logger.info("Loaded %d documents from %s", len(documents), settings.documents_path)
    return documents


def split_text(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No chunks created from documents")
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic ID from source + content — stable across re-syncs."""
    source = str(chunk.metadata.get("source", ""))
    digest = hashlib.sha1(f"{source}::{chunk.page_content}".encode()).hexdigest()
    return f"chunk_{digest[:24]}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model().encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 16,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def clean_metadata(metadata: dict) -> dict:
    # Store metadata values must be scalars; drop anything None/complex.
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


# Historical name kept so existing imports/tests keep working.
_clean_metadata = clean_metadata
