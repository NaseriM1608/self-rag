"""Offline unit tests — no model loads, no network, no index access."""

from langchain_core.documents import Document

from evals.run_retrieval import is_hit, normalize
from retriever import _clean_metadata, chunk_id


def test_chunk_id_is_deterministic_per_content():
    a = Document(page_content="hello world", metadata={"source": "documents/x.txt"})
    b = Document(page_content="hello world", metadata={"source": "documents/x.txt"})
    assert chunk_id(a) == chunk_id(b)


def test_chunk_id_differs_for_different_content_or_source():
    base = Document(page_content="hello world", metadata={"source": "a.txt"})
    other_text = Document(page_content="goodbye", metadata={"source": "a.txt"})
    other_src = Document(page_content="hello world", metadata={"source": "b.txt"})
    assert chunk_id(base) != chunk_id(other_text)
    assert chunk_id(base) != chunk_id(other_src)


def test_clean_metadata_drops_non_scalars():
    dirty = {"source": "x.txt", "start_index": 0, "note": None, "tags": ["a"]}
    cleaned = _clean_metadata(dirty)
    assert cleaned == {"source": "x.txt", "start_index": 0}


def test_normalize_collapses_whitespace_and_case():
    assert normalize("The   Transformer\nArchitecture ") == "the transformer architecture"


def test_is_hit_matches_snippet_inside_chunk():
    chunk = "RAG enables LLMs to retrieve new information from external sources."
    assert is_hit(chunk, ["retrieve new information"])
    assert not is_hit(chunk, ["quantum computing"])


def test_is_hit_ignores_case_and_whitespace():
    assert is_hit("Fixed Length WITH overlap", ["fixed length with overlap"])


def test_rrf_fuse_fuses_and_ranks():
    """A doc appearing in both lists outranks one appearing in a single list."""
    from langchain_core.documents import Document

    from retrievers import rrf_fuse

    a_both = Document(page_content="in both", metadata={"id": "a", "retriever": "vector"})
    a_lex = Document(page_content="in both", metadata={"id": "a", "retriever": "fulltext"})
    b_vec = Document(page_content="vector only", metadata={"id": "b", "retriever": "vector"})
    c_lex = Document(page_content="lexical only", metadata={"id": "c", "retriever": "fulltext"})

    fused = rrf_fuse([[a_both, b_vec], [a_lex, c_lex]], final_k=3)
    assert [d.metadata["id"] for d in fused] == ["a", "b", "c"]
    assert fused[0].metadata["retrievers"] == ["fulltext", "vector"]
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]


def test_sanitize_lucene_strips_operators():
    from retrievers import _sanitize_lucene

    assert _sanitize_lucene('what is "prompt stuffing"? (RAG)') == "what is prompt stuffing RAG"
