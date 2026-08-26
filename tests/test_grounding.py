"""
Mini-benchmark for `grounding_chain` (chains.py): measures how often it
flags a valid logical inference as an unsupported claim ("YES/NO" ->
here treated as grounded/ungrounded), the exact failure mode named in
the README under "Strict grounding" and in the paper.

Lives in tests/ but marked `live` (real LLM calls) — deselected by the
default pytest addopts and by CI; run explicitly with:

    pytest tests/test_grounding.py -v -s -m live

or, for the standalone printed report:

    python tests/test_grounding.py

Requires OPENROUTER_API_KEY (or legacy GROQ_API_KEY) in your environment
/ .env since grounding_chain makes a real LLM call. Tests are skipped
automatically if the key isn't set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from dotenv import load_dotenv

load_dotenv()

from chains import GroundingVerdict, grounding_chain  # noqa: E402 (needs dotenv first)

# Real LLM calls — excluded from the default suite via `-m 'not live'`.
pytestmark = pytest.mark.live

HAS_LLM_KEY = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY"))
skip_no_key = pytest.mark.skipif(
    not HAS_LLM_KEY,
    reason="No LLM API key set — grounding_chain needs a live call",
)


# ---------------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------------
# category:
#   "fact"        -> answer states something directly present in documents
#                    (should be grounded)
#   "inference"    -> answer states a valid logical/numeric/temporal/
#                    transitive inference from documents, not verbatim
#                    (should be grounded — this is the case the README
#                    says the checker currently over-flags)
#   "unsupported"  -> answer states something with no support in the
#                    documents at all (should be ungrounded)

@dataclass(frozen=True)
class Example:
    id: str
    documents: str
    answer: str
    category: str
    expected_grounded: bool


EXAMPLES: list[Example] = [
    # facts
    Example(
        "fact-1",
        "The Inverse Cloze Task (ICT) is a pre-training method where a sentence is used as a pseudo-query and the rest of the passage as the pseudo-document.",
        "ICT is a pre-training technique that treats a sentence as a query against the rest of its passage.",
        "fact", True,
    ),
    Example(
        "fact-2",
        "Self-RAG uses reflection tokens to decide when to retrieve and to critique its own generations.",
        "Self-RAG relies on reflection tokens for both retrieval decisions and self-critique.",
        "fact", True,
    ),
    Example(
        "fact-3",
        "BAAI/bge-m3 is an embedding model that supports dense, sparse, and multi-vector retrieval.",
        "bge-m3 can be used for dense, sparse, and multi-vector retrieval.",
        "fact", True,
    ),

    # inferences (should still be grounded)
    Example(
        "infer-1",
        "ICT is used for pre-training the retriever to recognize which passage a sentence was drawn from.",
        "ICT improves the retriever's ability to fetch relevant documents.",
        "inference", True,
    ),
    Example(
        "infer-2",
        "Model A scored 92% on the eval; Model B scored 87% on the same eval.",
        "Model A outperformed Model B on the evaluation.",
        "inference", True,
    ),
    Example(
        "infer-3",
        "LangGraph checkpoints let a graph resume from any saved state after an interruption.",
        "LangGraph checkpoints support recovering a run after a crash.",
        "inference", True,
    ),
    Example(
        "infer-4",
        "The grounding node runs after generate and before check_usefulness in the graph.",
        "Grounding is checked before the usefulness check.",
        "inference", True,
    ),
    Example(
        "infer-5",
        "ChromaDB persists embeddings to disk so they don't need to be recomputed on every run.",
        "Restarting the app does not require re-embedding the documents.",
        "inference", True,
    ),
    Example(
        "infer-6",
        "MAX_LLM_CALLS caps the total number of LLM calls per run; the graph exits once the cap is hit.",
        "The system will not loop forever even if grounding keeps failing.",
        "inference", True,
    ),

    # unsupported (should be ungrounded)
    Example(
        "unsup-1",
        "The Inverse Cloze Task (ICT) is a pre-training method where a sentence is used as a pseudo-query and the rest of the passage as the pseudo-document.",
        "ICT was introduced in the original Self-RAG paper.",
        "unsupported", False,
    ),
    Example(
        "unsup-2",
        "Self-RAG uses reflection tokens to decide when to retrieve and to critique its own generations.",
        "Self-RAG requires a fine-tuned 13B parameter model to work at all.",
        "unsupported", False,
    ),
    Example(
        "unsup-3",
        "BAAI/bge-m3 is an embedding model that supports dense, sparse, and multi-vector retrieval.",
        "bge-m3 was trained exclusively on English text.",
        "unsupported", False,
    ),
    Example(
        "unsup-4",
        "LangGraph checkpoints let a graph resume from any saved state after an interruption.",
        "LangGraph checkpoints are stored encrypted by default.",
        "unsupported", False,
    ),
    Example(
        "unsup-5",
        "MAX_LLM_CALLS caps the total number of LLM calls per run; the graph exits once the cap is hit.",
        "MAX_LLM_CALLS is set to 10 by default.",
        "unsupported", False,
    ),
]


def _by_category(category: str) -> list[Example]:
    return [e for e in EXAMPLES if e.category == category]


def run_grounding_check(documents: str, answer: str) -> bool:
    """Calls the real chain exactly as nodes.check_grounding does."""
    verdict: GroundingVerdict = grounding_chain.invoke(
        {"answer": answer, "documents": documents}
    )
    return verdict.is_grounded


# ---------------------------------------------------------------------------
# pytest tests
# ---------------------------------------------------------------------------

@skip_no_key
@pytest.mark.parametrize("example", _by_category("fact"), ids=lambda e: e.id)
def test_facts_are_grounded(example: Example):
    got = run_grounding_check(example.documents, example.answer)
    assert got == example.expected_grounded, f"[{example.id}] expected grounded, got ungrounded"


@skip_no_key
@pytest.mark.parametrize("example", _by_category("unsupported"), ids=lambda e: e.id)
def test_unsupported_claims_are_flagged(example: Example):
    got = run_grounding_check(example.documents, example.answer)
    assert got == example.expected_grounded, f"[{example.id}] expected ungrounded, got grounded"


@skip_no_key
@pytest.mark.parametrize("example", _by_category("inference"), ids=lambda e: e.id)
def test_nuanced_inference_not_flagged_as_hallucination(example: Example):
    """
    The regression test the README's 'Strict grounding' limitation is
    about: a valid inference shouldn't be treated the same as an
    unsupported claim just because it's not verbatim in the documents.
    """
    got = run_grounding_check(example.documents, example.answer)
    assert got == example.expected_grounded, (
        f"[{example.id}] nuanced inference wrongly flagged as ungrounded: "
        f"documents={example.documents!r} answer={example.answer!r}"
    )


@skip_no_key
def test_inference_false_positive_rate_reported(capsys):
    """No hard threshold asserted — prints the rate so it's tracked in
    CI output / can be pasted into the README as a numeric result."""
    inference_examples = _by_category("inference")
    wrong = [e.id for e in inference_examples if run_grounding_check(e.documents, e.answer) != e.expected_grounded]
    fp_rate = len(wrong) / len(inference_examples)
    print(f"\nnuanced-inference-flagged-as-hallucination rate: {fp_rate:.1%} "
          f"({len(wrong)}/{len(inference_examples)})")
    if wrong:
        print(f"  wrongly flagged: {wrong}")


# ---------------------------------------------------------------------------
# Standalone mini-benchmark report
# ---------------------------------------------------------------------------

def run_benchmark() -> dict:
    categories = ["fact", "inference", "unsupported"]
    report: dict = {}

    for cat in categories:
        examples = _by_category(cat)
        wrong_ids = []
        correct = 0
        for e in examples:
            got = run_grounding_check(e.documents, e.answer)
            if got == e.expected_grounded:
                correct += 1
            else:
                wrong_ids.append(e.id)
        report[cat] = {"n": len(examples), "correct": correct, "wrong_ids": wrong_ids}

    labels = {
        "fact": "Facts (verbatim support)",
        "inference": "Nuanced inference (the key case)",
        "unsupported": "Unsupported (true hallucinations)",
    }

    print("=" * 60)
    print("SELF-RAG check_grounding — MINI-BENCHMARK")
    print("=" * 60)
    total_n = total_correct = 0
    for cat in categories:
        r = report[cat]
        total_n += r["n"]
        total_correct += r["correct"]
        acc = r["correct"] / r["n"]
        print(f"{labels[cat]:38s} {r['correct']}/{r['n']}  ({acc:.1%})")
        if r["wrong_ids"]:
            print(f"    misclassified: {r['wrong_ids']}")

    inf = report["inference"]
    inf_fp_rate = 1 - inf["correct"] / inf["n"]
    print("-" * 60)
    print(f"Overall accuracy: {total_correct}/{total_n} ({total_correct/total_n:.1%})")
    print(f"Inference false-positive rate: {inf_fp_rate:.1%}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    if not HAS_LLM_KEY:
        raise SystemExit(
            "Set OPENROUTER_API_KEY (or GROQ_API_KEY) in your environment or .env first."
        )
    run_benchmark()
