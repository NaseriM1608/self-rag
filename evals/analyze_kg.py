"""Diagnose the KG multi-hop gap: what does graph-expand actually change?

Compares hybrid vs graph-expand retrieval on the golden multi-hop slice and
answers, per question:

  1. Did the bridge slots fill at all?
  2. Did bridged chunks bring in golden source files hybrid's top-5 missed?
  3. Are bridged chunks relevant (golden snippet hits) or noise?

Needs the live Neo4j index (embeddings + MENTIONS graph) but no LLM —
retrieval-side only. Usage:

    python -m evals.analyze_kg [--k 5]

Writes evals/results/kg_gap_analysis.json and prints a per-question table.
Interpretation guide:
  - slots filled ~0            -> query-time entity matching is the bottleneck
  - filled but not hits        -> bridged chunks are noise (entity linking too
                                 loose, or one-hop neighborhood too generic)
  - hits but sources unchanged -> bridge restates already-covered sources
  - new golden sources + hits  -> retrieval did its job; the remaining e2e gap
                                 is in generation (context not synthesized)
"""

import argparse
import json
from pathlib import Path

from evals.run_retrieval import GOLDEN_FILE, is_hit, normalize
from retrievers import GraphExpandRetriever, HybridRetriever, retrieval_diagnostics

RESULTS_FILE = Path("evals/results/kg_gap_analysis.json")


def _stems(sources) -> set[str]:
    return {normalize(Path(str(s)).stem) for s in sources if s}


def _doc_source_stem(doc) -> str:
    return normalize(Path(str(doc.metadata.get("source", ""))).stem)


def analyze_question(question: dict, k: int) -> dict:
    snippets = question["relevant_snippets"]
    golden_sources = _stems(question.get("snippet_source_files", []))

    hybrid_docs = HybridRetriever().search(question["question"], k)
    retrieval_diagnostics.set(None)  # discard hybrid's (empty) snapshot
    expand_docs = GraphExpandRetriever().search(question["question"], k)
    diag = retrieval_diagnostics.get() or {}

    hybrid_sources = {_doc_source_stem(d) for d in hybrid_docs[:5]}
    expand_sources = {_doc_source_stem(d) for d in expand_docs[:5]}
    bridged = diag.get("bridged", [])
    bridged_by_id = {str(b["id"]) for b in bridged}
    bridged_docs = [d for d in expand_docs if str(d.metadata.get("id")) in bridged_by_id]
    bridged_hits = sum(1 for d in bridged_docs if is_hit(d.page_content, snippets))

    golden_sources_hybrid_missed = golden_sources - hybrid_sources
    bridged_sources = _stems([b["source"] for b in bridged])

    return {
        "id": question["id"],
        "question": question["question"],
        "golden_sources": sorted(golden_sources),
        "hybrid_top5_sources": sorted(hybrid_sources),
        "graph_expand_top5_sources": sorted(expand_sources),
        "golden_sources_hybrid_missed": sorted(golden_sources_hybrid_missed),
        "bridge_slots_reserved": diag.get("bridge_slots_reserved", 0),
        "bridge_slots_filled": diag.get("bridge_slots_filled", 0),
        "bridged_sources": sorted(bridged_sources),
        "bridged_hits": bridged_hits,
        "bridged_total": len(bridged_docs),
        # The money metric: did bridging recover a golden source hybrid missed?
        "recovered_golden_sources": sorted(
            golden_sources_hybrid_missed & bridged_sources
        ),
        "matched_entities": diag.get("matched_entities", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    multi_hop = [q for q in golden["questions"] if q.get("multi_hop")]
    print(f"Analyzing {len(multi_hop)} multi-hop questions at k={args.k}\n")

    rows = [analyze_question(q, args.k) for q in multi_hop]

    header = (
        f"{'id':8s} {'slots':>6s} {'filled':>6s} {'b-hits':>6s} "
        f"{'missed(src)':>14s} {'recovered':>10s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        missed = ",".join(s[:9] for s in r["golden_sources_hybrid_missed"]) or "-"
        recovered = ",".join(s[:9] for s in r["recovered_golden_sources"]) or "-"
        print(
            f"{r['id']:8s} {r['bridge_slots_reserved']:>6d} "
            f"{r['bridge_slots_filled']:>6d} {r['bridged_hits']:>6d} "
            f"{missed:>14s} {recovered:>10s}"
        )

    filled = sum(r["bridge_slots_filled"] for r in rows)
    reserved = sum(r["bridge_slots_reserved"] for r in rows)
    summary = {
        "n_questions": len(rows),
        "k": args.k,
        "slot_fill_rate": filled / reserved if reserved else 0.0,
        "questions_with_bridged_hits": sum(1 for r in rows if r["bridged_hits"] > 0),
        "questions_with_recovered_golden_source": sum(
            1 for r in rows if r["recovered_golden_sources"]
        ),
        "questions_where_hybrid_missed_a_golden_source": sum(
            1 for r in rows if r["golden_sources_hybrid_missed"]
        ),
    }
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps({"summary": summary, "questions": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
