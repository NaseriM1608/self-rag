"""End-to-end QA evaluation: full graph per question, LLM-judged answers.

This is the metric that can see GraphRAG's contribution: retrieval-level
recall cannot, because KG triple-documents don't contain golden snippets
verbatim — but a multi-hop answer either has both facts or it doesn't.

Usage:
    python -m evals.run_e2e --variant hybrid --slice multi_hop
    python -m evals.run_e2e --variant hybrid+kg --slice multi_hop

Results land in evals/results/e2e_<variant>_<slice>.json.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from metrics import track_query

RESULTS_DIR = Path("evals/results")
GOLDEN_FILE = Path("evals/golden/retrieval_golden.json")


class AnswerJudge(BaseModel):
    """Correctness of a generated answer against a reference answer."""

    score: int = Field(
        description="2 = states the key facts correctly and completely; "
        "1 = partially correct or incomplete; 0 = wrong, unsupported, or "
        "refuses despite the reference containing an answer"
    )
    reasoning: str = Field(description="one sentence justifying the score")


_judge_chain = None


def judge_chain():
    global _judge_chain
    if _judge_chain is None:
        from chains import llm

        prompt = ChatPromptTemplate.from_template(
            """You grade a generated answer against a reference answer.

Scoring rules:
- 2: the candidate states the reference's key facts correctly (extra detail is fine)
- 1: partially correct, or correct but missing an important part of the reference
- 0: factually wrong, unsupported, or refuses to answer although the reference has one

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}"""
        )
        # function_calling like every other grader: json_mode chokes when the
        # model wraps its answer in markdown instead of emitting raw JSON.
        _judge_chain = prompt | llm.with_structured_output(
            AnswerJudge, method="function_calling"
        )
    return _judge_chain


_TRANSIENT_BACKOFF_S = (20, 45, 90)


def _is_daily_pool_exhausted(exc: Exception) -> bool:
    return "free-models-per-day" in str(exc)


def run_e2e(variant: str, slice_name: str = "multi_hop", limit: int | None = None) -> dict:
    from graph import build_graph
    from retrievers import get_retriever

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    questions = [
        q
        for q in golden["questions"]
        if not q.get("expect_unanswerable") and q.get("reference_answer")
    ]
    if slice_name == "multi_hop":
        questions = [q for q in questions if q.get("multi_hop")]
    elif slice_name == "single":
        questions = [q for q in questions if not q.get("multi_hop")]
    if limit:
        questions = questions[:limit]

    agent = build_graph(get_retriever(variant))
    rows = {}
    aborted = None
    for q in questions:
        start = time.perf_counter()
        attempt = 0
        while True:  # transient-fault retry loop (daily-pool 429s abort)
            try:
                record = track_query(agent, q["question"], variant=variant)
                verdict: AnswerJudge = judge_chain().invoke(
                    {
                        "question": q["question"],
                        "reference": q["reference_answer"],
                        "candidate": record.generation,
                    }
                )
                break
            except Exception as exc:
                if _is_daily_pool_exhausted(exc):
                    aborted = f"daily pool exhausted at {q['id']}: {str(exc)[:120]}"
                    print(f"  ABORTED: {aborted}", flush=True)
                    break
                retryable = (
                    "RateLimit" in type(exc).__name__
                    or "429" in str(exc)
                    or "ValidationError" in type(exc).__name__
                )
                if retryable and attempt < len(_TRANSIENT_BACKOFF_S):
                    wait = _TRANSIENT_BACKOFF_S[attempt]
                    attempt += 1
                    print(
                        f"  {q['id']}: transient fault ({type(exc).__name__}), "
                        f"retry {attempt}/{len(_TRANSIENT_BACKOFF_S)} in {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                aborted = f"{type(exc).__name__} at {q['id']}: {str(exc)[:140]}"
                print(f"  ABORTED at {q['id']}: {aborted}", flush=True)
                break
        if aborted:
            break
        rows[q["id"]] = {
            "score": verdict.score,
            "reasoning": verdict.reasoning,
            "is_grounded": record.is_grounded,
            "is_useful": record.is_useful,
            "duration_s": record.duration_s,
            "llm_calls": record.llm_calls,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost_usd": record.cost_usd,
        }
        print(
            f"  {q['id']}: score={verdict.score} grounded={record.is_grounded} "
            f"{time.perf_counter() - start:.0f}s",
            flush=True,
        )

    n = len(rows) or 1
    durations = sorted(r["duration_s"] for r in rows.values())
    metrics = {
        "variant": variant,
        "slice": slice_name,
        "n_questions": len(rows),
        "aborted": aborted,
        "avg_score": sum(r["score"] for r in rows.values()) / n,
        "pct_correct": sum(1 for r in rows.values() if r["score"] >= 2) / n,
        "grounded_rate": sum(1 for r in rows.values() if r["is_grounded"]) / n,
        "p50_duration_s": statistics.median(durations) if durations else 0,
        "avg_llm_calls": sum(r["llm_calls"] for r in rows.values()) / n,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows.values()), 6),
        "per_question": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"e2e_{variant}_{slice_name}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"\n[e2e {variant}/{slice_name}] avg_score={metrics['avg_score']:.2f} "
        f"correct={metrics['pct_correct']:.1%} grounded={metrics['grounded_rate']:.1%} "
        f"p50={metrics['p50_duration_s']:.1f}s -> {out}"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--slice", default="multi_hop", choices=["all", "multi_hop", "single"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_e2e(args.variant, args.slice, args.limit)
