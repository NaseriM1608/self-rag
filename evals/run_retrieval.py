"""Retrieval quality evaluation: Recall@k and MRR against the golden set.

Usage:
    python -m evals.run_retrieval --variant dense

Variants are registered in RETRIEVERS; new retriever stacks (hybrid, KG)
plug in there without touching the metric code. Results land in
evals/results/retrieval_<variant>.json for the report generator.
"""

import argparse
import json
import re
import time
from pathlib import Path

RESULTS_DIR = Path("evals/results")
GOLDEN_FILE = Path("evals/golden/retrieval_golden.json")

# Retriever variants under test. Each maps (query, k) -> list[Document];
# new stacks (neo4j-dense, hybrid, hybrid+kg) register here as they land.
def _dense(query: str, k: int):
    from retrievers import get_retriever

    return get_retriever("dense").search(query, k)


def _neo4j_dense(query: str, k: int):
    from retrievers import get_retriever

    return get_retriever("neo4j-dense").search(query, k)


def _fulltext(query: str, k: int):
    from retrievers import get_retriever

    return get_retriever("fulltext").search(query, k)


def _hybrid(query: str, k: int):
    from retrievers import get_retriever

    return get_retriever("hybrid").search(query, k)


def _graph_expand(query: str, k: int):
    from retrievers import get_retriever

    return get_retriever("graph-expand").search(query, k)


RETRIEVERS = {
    "dense": _dense,
    "neo4j-dense": _neo4j_dense,
    "fulltext": _fulltext,
    "hybrid": _hybrid,
    "graph-expand": _graph_expand,
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


_CORPUS_CACHE: dict[str, str] | None = None


def _corpus_norm() -> dict[str, str]:
    """Normalized full text of every document file (loaded once)."""
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        _CORPUS_CACHE = {
            p.stem: normalize(p.read_text(encoding="utf-8"))
            for p in Path("documents").glob("*.txt")
        }
    return _CORPUS_CACHE


def snippet_hosts(snippets: list[str]) -> set[str]:
    """Which document files contain each snippet — re-derived from disk so
    labels cannot silently rot."""
    hosts: set[str] = set()
    for snippet in snippets:
        s = normalize(snippet)
        for name, text in _corpus_norm().items():
            if s in text:
                hosts.add(name)
    return hosts


def is_hit(chunk_text: str, snippets: list[str]) -> bool:
    normalized = normalize(chunk_text)
    return any(normalize(snippet) in normalized for snippet in snippets)


def evaluate_question(
    retriever, question: str, snippets: list[str], k: int
) -> dict:
    start = time.perf_counter()
    docs = retriever(question, k)
    latency_ms = (time.perf_counter() - start) * 1000

    first_hit_rank = None
    hits_at_5 = 0
    covered_hosts: set[str] = set()
    for rank, doc in enumerate(docs, start=1):
        if is_hit(doc.page_content, snippets):
            if first_hit_rank is None:
                first_hit_rank = rank
            if rank <= 5:
                hits_at_5 += 1

    # Which source documents contributed relevant chunks within top-k?
    # Multi-hop answers need evidence from >= 2 distinct sources.
    for doc in docs[:5]:
        matched = [s for s in snippets if normalize(s) in normalize(doc.page_content)]
        covered_hosts |= snippet_hosts(matched)

    # A question counts as recalled@k once any relevant chunk appears in top-k.
    recalled_at_5 = first_hit_rank is not None and first_hit_rank <= 5
    recalled_at_k = first_hit_rank is not None
    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0

    return {
        "latency_ms": round(latency_ms, 1),
        "first_hit_rank": first_hit_rank,
        "recalled_at_5": recalled_at_5,
        "recalled_at_k": recalled_at_k,
        "mrr": mrr,
        "hits_in_top5": hits_at_5,
        "covered_sources_top5": sorted(covered_hosts),
    }


def _aggregate(rows: dict, variant: str, k: int) -> dict:
    n = len(rows) or 1
    return {
        "variant": variant,
        "n_questions": n,
        "recall@5": sum(r["recalled_at_5"] for r in rows.values()) / n,
        f"recall@{k}": sum(r["recalled_at_k"] for r in rows.values()) / n,
        "mrr": sum(r["mrr"] for r in rows.values()) / n,
        "p50_latency_ms": sorted(r["latency_ms"] for r in rows.values())[n // 2],
    }


def run_variant(variant: str, k: int = 10) -> dict:
    retriever = RETRIEVERS[variant]
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))

    answerable = [q for q in golden["questions"] if not q.get("expect_unanswerable")]
    unanswerable = [q for q in golden["questions"] if q.get("expect_unanswerable")]

    rows = {}
    for q in answerable:
        rows[q["id"]] = evaluate_question(retriever, q["question"], q["relevant_snippets"], k)
        status = "hit" if rows[q["id"]]["first_hit_rank"] else "MISS"
        print(f"  {q['id']}: {status} rank={rows[q['id']]['first_hit_rank']} "
              f"{rows[q['id']]['latency_ms']:.0f}ms")

    metrics = _aggregate(rows, variant, k)
    metrics["per_question"] = rows

    # The headline slice: questions whose relevant evidence genuinely spans
    # multiple documents — where graph traversal should differentiate from
    # flat vector similarity.
    golden_by_id = {q["id"]: q for q in answerable}
    mh_rows = {qid: row for qid, row in rows.items() if golden_by_id[qid].get("multi_hop")}
    if mh_rows:
        mh_agg = _aggregate(mh_rows, variant, k)

        # Strict metric: did EVERY required source document contribute a chunk
        # to top-5? Only questions whose labelled snippets actually live in 2+
        # distinct files can test this; a question with single-file evidence
        # would silently degrade the metric into plain recall@5, so those are
        # excluded from the denominator and reported separately.
        cross_doc, single_doc = {}, {}
        for qid, row in mh_rows.items():
            needed = len(snippet_hosts(golden_by_id[qid]["relevant_snippets"]))
            (cross_doc if needed >= 2 else single_doc)[qid] = (row, needed)

        fully_covered = sum(
            1
            for row, needed in cross_doc.values()
            if len(row.get("covered_sources_top5", [])) >= needed
        )
        n_cross = len(cross_doc)

        metrics["multi_hop"] = {
            key: mh_agg[key] for key in ("n_questions", "recall@5", "mrr")
        }
        metrics["multi_hop"]["cross_document_n"] = n_cross
        metrics["multi_hop"]["single_document_excluded"] = sorted(single_doc)
        metrics["multi_hop"]["all_sources_in_top5"] = (
            fully_covered / n_cross if n_cross else None
        )
        print(
            f"  [multi-hop] recall@5={mh_agg['recall@5']:.3f} "
            f"mrr={mh_agg['mrr']:.3f} "
            f"all-required-sources-in-top5={fully_covered}/{n_cross} "
            f"(cross-document questions only; "
            f"{len(single_doc)} single-document excluded)"
        )

    # Unanswerables must not be "answered" confidently — record what comes back
    # so the e2e eval can verify the graph exits via its no-documents path.
    unanswerable_rows = {}
    for q in unanswerable:
        docs = retriever(q["question"], settings_n_results())
        unanswerable_rows[q["id"]] = len(docs)
    metrics["unanswerable_top_returned"] = unanswerable_rows

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{variant}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n[{variant}] recall@5={metrics['recall@5']:.3f} "
          f"recall@{k}={metrics[f'recall@{k}']:.3f} mrr={metrics['mrr']:.3f} "
          f"p50={metrics['p50_latency_ms']:.0f}ms -> {out}")
    return metrics


def settings_n_results() -> int:
    from config import settings

    return settings.n_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="dense", choices=sorted(RETRIEVERS))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_variant(args.variant, args.k)
