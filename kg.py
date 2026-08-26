"""Knowledge-graph construction and retrieval over Neo4j.

Build: every indexed chunk is sent to the LLM for (subject, predicate,
object) extraction with structured output. Results are cached on disk keyed
by content-hash chunk id, so re-running only pays for new/changed chunks.
Entities merge on normalized name; every relation keeps provenance back to
the originating chunk(s). Entity names are embedded so questions can find
relevant nodes without an extra LLM call at query time.

Retrieve: embed the question -> nearest Entity nodes -> expand one hop of
relations -> serialize triples as context Documents. KG-derived context is
NOT trusted automatically: it flows through the same grading/grounding gates
as ordinary retrieved chunks.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from chains import llm
from config import settings

logger = logging.getLogger(__name__)

# Triple-extraction cache — per-user location (settings.kg_cache_path) so
# the synced repo doesn't churn on every extraction checkpoint.
CACHE_PATH = settings.kg_cache_path
MAX_RELATIONS_PER_CHUNK = 8
EXTRACTION_WORKERS = 4


class ExtractedRelation(BaseModel):
    """One factual relation stated in the source text."""

    subject: str = Field(description="canonical name of the subject entity")
    predicate: str = Field(description="short lowercase verb phrase connecting them")
    object: str = Field(description="canonical name of the object entity")


class ChunkRelations(BaseModel):
    relations: list[ExtractedRelation] = Field(
        default_factory=list, description="relations stated in the chunk"
    )


extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You extract knowledge-graph relations from technical text.

Rules:
- Extract only facts explicitly stated in the text; never use outside knowledge.
- Use canonical entity names ("Transformer", not "the transformer architecture was").
- Predicates are short lowercase verb phrases ("introduced", "uses", "stores").
- Skip vague or trivial statements; prefer definitional, causal, and taxonomic facts.
- Return at most {max_relations} relations; fewer is better than noise.""",
        ),
        (
            "human",
            """Example text:
"RAG converts documents into embeddings which are stored in a vector database."
Example output:
{{"relations": [
  {{"subject": "RAG", "predicate": "converts documents into", "object": "embeddings"}},
  {{"subject": "RAG", "predicate": "stores embeddings in", "object": "vector database"}}
]}}

Text to process:
{text}""",
        ),
    ]
)


@lru_cache(maxsize=1)
def _extraction_chain():
    return extraction_prompt | llm.with_structured_output(ChunkRelations)


def normalize_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).title()[:80]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8"
    )


def extract_chunk(chunk_id: str, text: str, cache: dict) -> None:
    """Extract relations for one chunk; fills the cache entry in place."""
    try:
        result: ChunkRelations = _extraction_chain().invoke(
            {"text": text, "max_relations": MAX_RELATIONS_PER_CHUNK}
        )
        cache[chunk_id] = [r.model_dump() for r in result.relations]
    except Exception as exc:
        logger.warning("Extraction failed for %s: %s", chunk_id, exc)
        cache[chunk_id] = []


def write_graph(chunk_relations: dict[str, list[dict]]) -> dict:
    """Rebuild (:Entity)/[:RELATES]/[:MENTIONS] structures from extractions."""
    from neo4j_store import _session

    mentions = [
        {
            "chunk_id": cid,
            "entities": sorted(
                {normalize_entity(r["subject"]) for r in rels}
                | {normalize_entity(r["object"]) for r in rels}
            ),
        }
        for cid, rels in chunk_relations.items()
        if rels
    ]
    rows = [
        {
            "subject": normalize_entity(r["subject"]),
            "predicate": r["predicate"].lower(),
            "object": normalize_entity(r["object"]),
            "chunk_id": cid,
        }
        for cid, rels in chunk_relations.items()
        for r in rels
    ]

    with _session() as session:
        # Preserve name->embedding across the rebuild so unchanged entities
        # do not get re-embedded on every sync.
        existing_embeddings = {
            r["name"]: r["emb"]
            for r in session.run(
                "MATCH (e:Entity) WHERE e.embedding IS NOT NULL "
                "RETURN e.name AS name, e.embedding AS emb"
            )
        }
        session.run("MATCH ()-[r:RELATES]->() DELETE r")
        session.run("MATCH ()-[m:MENTIONS]->() DELETE m")
        session.run("MATCH (e:Entity) DETACH DELETE e")

        if mentions:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.chunk_id})
                UNWIND row.entities AS name
                MERGE (e:Entity {name: name})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                rows=mentions,
            )
        if rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (s:Entity {name: row.subject})
                MATCH (o:Entity {name: row.object})
                MERGE (s)-[r:RELATES {predicate: row.predicate}]->(o)
                  ON CREATE SET r.chunk_ids = [row.chunk_id]
                  ON MATCH SET r.chunk_ids =
                    CASE WHEN row.chunk_id IN r.chunk_ids
                         THEN r.chunk_ids ELSE r.chunk_ids + row.chunk_id END
                """,
                rows=rows,
            )
        restored = [
            {"name": name, "embedding": emb}
            for name, emb in existing_embeddings.items()
        ]
        for i in range(0, len(restored), 500):
            session.run(
                """
                UNWIND $pairs AS pair
                MATCH (e:Entity {name: pair.name})
                SET e.embedding = pair.embedding
                """,
                pairs=restored[i : i + 500],
            )

    with _session() as session:
        entity_count = session.run(
            "MATCH (e:Entity) RETURN count(e) AS n"
        ).single()["n"]
        edge_count = session.run(
            "MATCH ()-[r:RELATES]->() RETURN count(r) AS n"
        ).single()["n"]

    logger.info("KG write: %d entities, %d relation edges", entity_count, edge_count)
    return {"entities": entity_count, "relations": edge_count}


def build_knowledge_graph(force: bool = False) -> dict:
    """Extract relations for all indexed chunks (cached) and write the graph."""
    from neo4j_store import ensure_vector_index
    from retriever import chunk_id, load_documents, split_text

    ensure_vector_index()
    cache = {} if force else load_cache()

    chunks = split_text(load_documents())
    todo: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        cid = chunk_id(chunk)
        if cid in seen:
            continue
        seen.add(cid)
        if cid not in cache:
            todo.append((cid, chunk.page_content))

    logger.info(
        "KG extraction: %d chunks total, %d cached, %d to extract",
        len(seen), len(seen) - len(todo), len(todo),
    )
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as pool:
            futures = {
                pool.submit(extract_chunk, cid, text, cache): cid
                for cid, text in todo
            }
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 25 == 0:
                    save_cache(cache)
                    logger.info("KG extraction progress: %d/%d", done, len(todo))
        save_cache(cache)

    chunk_relations = {cid: cache[cid] for cid in seen if cid in cache}
    stats = write_graph(chunk_relations)
    embed_entities()
    return stats


def embed_entities() -> None:
    """Store one embedding per Entity name (query-time matching needs these)."""
    from neo4j_store import _session
    from retriever import embed_texts

    with _session() as session:
        names = [
            r["name"]
            for r in session.run(
                "MATCH (e:Entity) WHERE e.embedding IS NULL RETURN e.name AS name"
            )
        ]
    if not names:
        logger.info("Entity embeddings already up to date")
        return

    embeddings = embed_texts(names)
    pairs = [{"name": n, "embedding": e} for n, e in zip(names, embeddings, strict=True)]
    with _session() as session:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (e:Entity {name: pair.name})
            SET e.embedding = pair.embedding
            """,
            pairs=pairs,
        )
        session.run(
            f"""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (e:Entity) ON e.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {settings.embedding_dims},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
    # A freshly created index is not queryable until population finishes.
    from neo4j_store import wait_for_index_online

    wait_for_index_online("entity_embeddings")
    logger.info("Embedded %d entity names", len(names))


_MATCH_STOPWORDS = frozenset(
    """a an and are as at by did do does for from how in into is of on or that the
    their them then there these this to was were what when where which who whom
    whose why will with""".split()
)


def _match_entities(question: str, limit: int) -> list[str]:
    """Find graph entities named in the question via token overlap.

    Deterministic and explainable; falls back to embedding similarity over
    entity names only when no entity name shares a token with the question.
    """
    from neo4j_store import _session

    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", question.lower())
        if tok not in _MATCH_STOPWORDS
    }
    with _session() as session:
        names = [
            r["name"] for r in session.run("MATCH (e:Entity) RETURN e.name AS name")
        ]

    scored = []
    for name in names:
        lowered = name.lower()
        hits = sum(1 for tok in tokens if tok in lowered)
        if hits:
            scored.append((hits, -len(lowered), name))
    if scored:
        scored.sort(reverse=True)
        return [name for _, _, name in scored[:limit]]

    # Fallback: semantic nearest entity names.
    from retriever import embed_texts

    embedding = embed_texts([question])[0]
    with _session() as session:
        rows = session.run(
            """
            MATCH (e:Entity)
              SEARCH e IN (
                VECTOR INDEX `entity_embeddings`
                FOR $emb LIMIT $m
              )
              SCORE AS score
            RETURN e.name AS name
            """,
            emb=embedding,
            m=limit,
        )
        return [r["name"] for r in rows]


def kg_context(query: str, max_entities: int = 3, per_entity: int = 6) -> list[Document]:
    """Serialize one-hop graph relations around the question's entities."""
    from neo4j_store import _session

    entities = _match_entities(query, max_entities)
    with _session() as session:
        if not entities:
            return []
        rows = session.run(
            """
            UNWIND $names AS name
            MATCH (s:Entity {name: name})-[r:RELATES]->(o:Entity)
            RETURN s.name AS subject, r.predicate AS predicate,
                   o.name AS object, r.chunk_ids AS chunk_ids
            """,
            names=entities,
        ).data()

    documents = []
    for row in rows[: max_entities * per_entity]:
        text = f"{row['subject']} --{row['predicate']}--> {row['object']}"
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"knowledge_graph ({', '.join(row['chunk_ids'][:2])})",
                    "retriever": "kg",
                    "chunk_ids": row["chunk_ids"],
                },
            )
        )
    return documents


def graph_expanded_chunks(
    query: str,
    exclude_ids: set[str],
    seen_sources: set[str],
    limit: int,
    max_entities: int = 4,
) -> list[Document]:
    """Reach chunks that similarity never ranked, via shared entities.

    This is the bridge kg_context does not provide: instead of serializing
    triples (which restate the already-retrieved chunks they were extracted
    from), walk (:Chunk)-[:MENTIONS]->(:Entity) from the question's entities
    back out to *other* chunks — preferring source documents missing from the
    similarity result, which is exactly the second-source gap that multi-hop
    questions fail on. Returned documents are ordinary prose chunks, so the
    relevance grader and the citation-bound generator handle them normally.
    """
    from neo4j_store import _session

    entities = _match_entities(query, max_entities)
    if not entities:
        return []

    with _session() as session:
        rows = session.run(
            """
            UNWIND $names AS name
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: name})
            WHERE NOT c.id IN $exclude
            WITH c, count(DISTINCT e) AS matched
            RETURN c.id AS id, c.text AS text, c.source AS source, matched
            ORDER BY
              CASE WHEN c.source IN $seen THEN 1 ELSE 0 END ASC,
              matched DESC,
              c.id ASC
            LIMIT $k
            """,
            names=entities,
            exclude=sorted(exclude_ids),
            seen=sorted(seen_sources),
            k=limit,
        ).data()

    return [
        Document(
            page_content=row["text"],
            metadata={
                "id": row["id"],
                "source": row["source"],
                "retriever": "graph-expand",
                "entity_matches": row["matched"],
                "bridged_entities": entities,
            },
        )
        for row in rows
    ]
