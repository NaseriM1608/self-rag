"""Retriever abstraction — pluggable search backends behind one protocol.

Every retrieval stack implements `Retriever`. Consumers — the LangGraph
nodes and the eval runners — depend on the protocol and select
implementations by name, so adding a backend never touches calling code.
The runtime default is Neo4j (`settings.default_retriever`);
`DenseRetriever` (Chromadb) is kept as the frozen eval baseline.
"""

import re
from functools import lru_cache
from typing import Any, Protocol

from langchain_core.documents import Document

from config import settings


class Retriever(Protocol):
    """Anything that can answer a query with ranked documents."""

    name: str

    def search(self, query: str, k: int = 0) -> list[Document]:
        """Return up to k documents ranked by relevance (k=0 → default)."""
        ...


@lru_cache(maxsize=1)
def _chroma_store() -> tuple[Any, Any]:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(settings.index_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "ip"}
    )
    return client, col


class DenseRetriever:
    """Dense vector retrieval over ChromaDB — frozen eval-time baseline."""

    name = "dense"

    def search(self, query: str, k: int = 0) -> list[Document]:
        from retriever import embed_texts

        _, col = _chroma_store()
        embedding = embed_texts([query])[0]
        results = col.query(
            query_embeddings=[embedding], n_results=k or settings.n_results
        )
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"][0], results["metadatas"][0], strict=True)
        ]


class Neo4jDenseRetriever:
    """Dense vector retrieval via Neo4j's native vector index."""

    name = "neo4j-dense"

    def search(self, query: str, k: int = 0) -> list[Document]:
        from neo4j_store import vector_search
        from retriever import embed_texts

        embedding = embed_texts([query])[0]
        return vector_search(embedding, k or settings.n_results)


class FulltextRetriever:
    """Lexical retrieval via the Lucene full-text index (BM25 scoring)."""

    name = "fulltext"

    def search(self, query: str, k: int = 0) -> list[Document]:
        from neo4j_store import fulltext_search

        return fulltext_search(_sanitize_lucene(query), k or settings.n_results)


def _sanitize_lucene(query: str) -> str:
    # Strip Lucene operator characters so raw questions parse as plain terms.
    return " ".join(re.sub(r"[^\w\s]", " ", query).split())


def rrf_fuse(ranked_lists: list[list[Document]], final_k: int) -> list[Document]:
    """Reciprocal Rank Fusion across result lists (rrf_k = 60).

    Documents are keyed by chunk id when present, else content prefix.
    Fused metadata records which retrievers contributed and the fused score.
    """
    rrf_k = 60
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    contributors: dict[str, set[str]] = {}

    for results in ranked_lists:
        for rank, doc in enumerate(results, start=1):
            key = str(doc.metadata.get("id") or doc.page_content[:80])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            contributors.setdefault(key, set()).add(
                str(doc.metadata.get("retriever", "unknown"))
            )
            if key not in docs_by_key:
                docs_by_key[key] = doc

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:final_k]
    out = []
    for key, score in ranked:
        base = docs_by_key[key]
        metadata = dict(base.metadata)
        metadata["rrf_score"] = round(score, 5)
        metadata["retrievers"] = sorted(contributors[key])
        out.append(Document(page_content=base.page_content, metadata=metadata))
    return out


class HybridRetriever:
    """Vector + lexical retrieval fused with Reciprocal Rank Fusion."""

    name = "hybrid"

    def __init__(self, per_retriever_k: int = 10):
        self.per_retriever_k = per_retriever_k

    def search(self, query: str, k: int = 0) -> list[Document]:
        final_k = k or settings.n_results
        fetch_k = max(self.per_retriever_k, final_k)
        vector = Neo4jDenseRetriever().search(query, fetch_k)
        lexical = FulltextRetriever().search(query, fetch_k)
        return rrf_fuse([vector, lexical], final_k)


class KGRetriever:
    """Graph-context retrieval: one-hop relations around matched entities."""

    name = "kg"

    def search(self, query: str, k: int = 0) -> list[Document]:
        from kg import kg_context

        docs = kg_context(query)
        limit = k or settings.n_results
        return docs[:limit]


class HybridKGRetriever:
    """Hybrid chunk retrieval with KG triple-documents appended.

    Chunks come first (they carry the prose the generator cites); triples
    follow as supplementary context. Both pass through the same grading
    and grounding gates downstream.
    """

    name = "hybrid+kg"

    def __init__(self, kg_docs: int = 5):
        self.kg_docs = kg_docs

    def search(self, query: str, k: int = 0) -> list[Document]:
        final_k = k or settings.n_results
        chunks = HybridRetriever().search(query, final_k)
        return chunks + KGRetriever().search(query, self.kg_docs)


class GraphExpandRetriever:
    """Hybrid retrieval with graph-bridged chunks competing for the same slots.

    Unlike hybrid+kg (which appends triple text restating the chunks it came
    from), this walks the entity graph out to prose chunks the similarity
    search never ranked, preferring source files absent from the hybrid
    result. That targets the measured failure mode: multi-hop questions whose
    second source never enters the context window.

    The context budget is held fixed at k: bridged chunks take `bridge_slots`
    of it rather than being appended on top, so the comparison against hybrid
    is like-for-like in both ranking window and LLM grading cost. When the
    graph finds nothing to bridge, the slots are backfilled with hybrid hits.
    """

    name = "graph-expand"

    def __init__(self, bridge_slots: int = 2):
        self.bridge_slots = bridge_slots

    def search(self, query: str, k: int = 0) -> list[Document]:
        from kg import graph_expanded_chunks

        final_k = k or settings.n_results
        slots = min(self.bridge_slots, max(final_k - 1, 0))
        # Over-fetch so unused bridge slots can be backfilled with real hits.
        chunks = HybridRetriever().search(query, final_k)
        kept = chunks[: final_k - slots]

        exclude = {
            str(doc.metadata.get("id"))
            for doc in chunks
            if doc.metadata.get("id") is not None
        }
        seen_sources = {
            str(doc.metadata.get("source"))
            for doc in kept
            if doc.metadata.get("source")
        }
        bridged = graph_expanded_chunks(
            query, exclude_ids=exclude, seen_sources=seen_sources, limit=slots
        )
        merged = kept + bridged
        if len(merged) < final_k:
            backfill = chunks[final_k - slots :]
            merged += backfill[: final_k - len(merged)]
        return merged[:final_k]


def get_retriever(name: str = "dense") -> Retriever:
    """Resolve a retriever implementation by registered name."""
    if name == "dense":
        return DenseRetriever()
    if name == "neo4j-dense":
        return Neo4jDenseRetriever()
    if name == "fulltext":
        return FulltextRetriever()
    if name == "hybrid":
        return HybridRetriever()
    if name == "kg":
        return KGRetriever()
    if name == "hybrid+kg":
        return HybridKGRetriever()
    if name == "graph-expand":
        return GraphExpandRetriever()
    known = [
        "dense", "neo4j-dense", "fulltext", "hybrid", "kg", "hybrid+kg",
        "graph-expand",
    ]
    raise KeyError(f"Unknown retriever variant {name!r}; known: {known}")
