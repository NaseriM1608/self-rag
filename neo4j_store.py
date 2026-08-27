"""Neo4j persistence layer.

Stores document chunks as (:Chunk) nodes under a native vector index plus a
Lucene full-text index; the knowledge-graph layer (kg.py) adds Entity nodes
and MENTIONS/RELATES relationships onto the same database. All writes are
idempotent syncs keyed by content-hash chunk ids so re-ingesting an edited
corpus updates exactly the changed pieces.
"""

import logging
import time
from functools import lru_cache

from langchain_core.documents import Document
from neo4j import Driver, GraphDatabase

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def driver() -> Driver:
    if not settings.neo4j_password:
        raise RuntimeError("NEO4J_PASSWORD not set - add it to .env (see .env.example)")
    drv = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    drv.verify_connectivity()
    logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
    return drv


def _session():
    return driver().session(database=settings.neo4j_database)


def ensure_vector_index() -> None:
    # Index names cannot be Cypher parameters; the value is trusted config.
    with _session() as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {settings.vector_index_name} IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: $dims,
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            dims=settings.embedding_dims,
        )


def ensure_fulltext_index() -> None:
    with _session() as session:
        session.run(
            """
            CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """
        )


def wait_for_index_online(name: str, timeout_s: float = 120.0) -> bool:
    """Block until a newly created index finishes populating (or time out).

    Querying a vector index mid-population fails with 51N63, so callers
    should wait right after CREATE ... INDEX statements.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with _session() as session:
            rows = session.run(
                "SHOW INDEXES YIELD name, state "
                "WHERE name = $name RETURN state",
                name=name,
            ).data()
        if rows and rows[0]["state"] == "ONLINE":
            return True
        time.sleep(1.0)
    logger.warning("Index %s still not online after %.0fs", name, timeout_s)
    return False


def vector_search(query_embedding: list[float], k: int) -> list[Document]:
    """Top-k chunks via the native vector index (cosine similarity).

    Uses the Cypher 25 SEARCH clause — db.index.vector.queryNodes is
    deprecated as of Neo4j 2026.04.
    """
    with _session() as session:
        result = session.run(
            f"""
            MATCH (c:Chunk)
              SEARCH c IN (
                VECTOR INDEX `{settings.vector_index_name}`
                FOR $emb
                LIMIT $k
              )
              SCORE AS score
            RETURN c.id AS id, c.text AS text,
                   c.source AS source, score
            """,
            k=k,
            emb=query_embedding,
        )
        return [
            Document(
                page_content=r["text"],
                metadata={
                    "id": r["id"], "source": r["source"],
                    "score": r["score"], "retriever": "vector",
                },
            )
            for r in result
        ]


def fulltext_search(query: str, k: int) -> list[Document]:
    """Top-k chunks via the Lucene full-text index (BM25 scoring).

    The Cypher SEARCH clause does not support text indexes yet, so this uses
    db.index.fulltext.queryNodes. Malformed Lucene queries degrade to an
    empty result rather than raising into the pipeline.
    """
    ensure_fulltext_index()
    try:
        with _session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_fulltext', $q)
                YIELD node, score
                RETURN node.id AS id, node.text AS text,
                       node.source AS source, score
                LIMIT $k
                """,
                q=query,
                k=k,
            )
            return [
                Document(
                    page_content=r["text"],
                    metadata={
                        "id": r["id"], "source": r["source"],
                        "score": r["score"], "retriever": "fulltext",
                    },
                )
                for r in result
            ]
    except Exception as exc:
        logger.warning("Full-text search failed for %r: %s", query[:60], exc)
        return []


def sync_neo4j_index() -> dict:
    """Full pipeline: documents -> chunks -> embeddings -> Neo4j sync.

    Embeds only chunk ids not yet stored, so an unchanged corpus re-syncs
    without touching the embedding model.
    """
    from retriever import chunk_id, embed_texts, load_documents, split_text

    ensure_vector_index()
    chunks = split_text(load_documents())

    with _session() as session:
        existing = {r["id"] for r in session.run("MATCH (c:Chunk) RETURN c.id AS id")}

    new_chunks = [c for c in chunks if chunk_id(c) not in existing]
    stale_ids = existing - {chunk_id(c) for c in chunks}

    if stale_ids:
        with _session() as session:
            session.run(
                "UNWIND $ids AS id MATCH (c:Chunk {id: id}) DETACH DELETE c",
                ids=list(stale_ids),
            )
    if new_chunks:
        embeddings = embed_texts([c.page_content for c in new_chunks])
        rows = [
            {
                "id": chunk_id(c),
                "text": c.page_content,
                "source": str(c.metadata.get("source", "")),
                "embedding": emb,
            }
            for c, emb in zip(new_chunks, embeddings, strict=True)
        ]
        with _session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c.text = row.text,
                    c.source = row.source,
                    c.embedding = row.embedding
                """,
                rows=rows,
            )

    stats = {"added": len(new_chunks), "removed": len(stale_ids), "total": len(chunks)}
    logger.info(
        "Neo4j chunk sync: +%d new, -%d removed, %d total",
        stats["added"], stats["removed"], stats["total"],
    )
    return stats
