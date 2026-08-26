"""Aggregate eval results into docs/METRICS.md — the project's numbers page.

Combines:
  - evals/results/retrieval_<variant>.json   (Recall@k / MRR / latency)
  - evals/results/grounding_judge.json       (judge P/R/F1)
  - evals/results/runs.jsonl                 (end-to-end run telemetry)

Usage:
    python -m evals.report
"""

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("evals/results")
OUT_FILE = Path("docs/METRICS.md")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def retrieval_section(lines: list[str]) -> None:
    lines.append("## Retrieval quality\n")
    result_files = sorted(RESULTS_DIR.glob("retrieval_*.json"))
    if not result_files:
        lines.append("_No retrieval results yet — run `python -m evals.run_retrieval`._\n")
        return

    has_mh = any(
        "multi_hop" in json.loads(p.read_text(encoding="utf-8")) for p in result_files
    )
    if has_mh:
        lines.append("| Variant | n | Recall@5 | Recall@10 | MRR | p50 latency | Multi-hop R@5 |")
        lines.append("|---|---|---|---|---|---|---|")
    else:
        lines.append("| Variant | n | Recall@5 | Recall@10 | MRR | p50 latency |")
        lines.append("|---|---|---|---|---|---|")
    for path in result_files:
        m = json.loads(path.read_text(encoding="utf-8"))
        mh = m.get("multi_hop")
        # Emit "—" for variants without multi-hop data so every row keeps the
        # full column count when the header includes the multi-hop column.
        mh_cell = f" {pct(mh['recall@5'])} |" if mh else " — |"
        lines.append(
            f"| {m['variant']} | {m['n_questions']} "
            f"| {pct(m['recall@5'])} | {pct(m.get('recall@10', 0))} "
            f"| {m['mrr']:.3f} | {m['p50_latency_ms']:.0f} ms |{mh_cell}"
        )
    lines.append("")


def grounding_section(lines: list[str]) -> None:
    lines.append("## Grounding-judge self-accuracy\n")
    path = RESULTS_DIR / "grounding_judge.json"
    if not path.exists():
        lines.append("_No results yet — run `python -m evals.run_grounding_judge`._\n")
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    c = m["confusion"]
    lines.append(f"n = {m['n_examples']} labeled claims\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {pct(m['accuracy'])} |")
    lines.append(f"| Ungrounded precision | {pct(m['ungrounded_precision'])} |")
    lines.append(f"| Ungrounded recall | {pct(m['ungrounded_recall'])} |")
    lines.append(f"| Ungrounded F1 | {m['ungrounded_f1']:.3f} |")
    lines.append(
        f"| Valid inferences wrongly flagged | "
        f"{pct(m['inference_false_positive_rate'])} |"
    )
    lines.append("")
    lines.append(
        f"Missed hallucinations (unsupported accepted): "
        f"{c['false_negatives_missed_hallucinations']}; "
        f"valid claims flagged as hallucinations: "
        f"{c['false_positives_valid_claims_flagged']}.\n"
    )


def runs_section(lines: list[str]) -> None:
    lines.append("## End-to-end runs (telemetry)\n")
    from metrics import load_records

    records = load_records()
    if not records:
        lines.append("_No recorded runs yet — telemetry lands in evals/results/runs.jsonl._\n")
        return

    by_variant: dict[str, list[dict]] = {}
    for r in records:
        by_variant.setdefault(r["variant"], []).append(r)

    lines.append("| Variant | n | success | p50 s | p95 s | llm calls/query | tok in/out | $/query |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for variant, rs in sorted(by_variant.items()):
        durations = sorted(r["duration_s"] for r in rs)
        p50 = statistics.median(durations)
        p95 = durations[max(0, int(len(durations) * 0.95) - 1)]
        success = sum(1 for r in rs if r["is_grounded"] and r["is_useful"]) / len(rs)
        avg_calls = statistics.mean(r["llm_calls"] for r in rs)
        tok_in = sum(r["input_tokens"] for r in rs) // len(rs)
        tok_out = sum(r["output_tokens"] for r in rs) // len(rs)
        cost = statistics.mean(r["cost_usd"] for r in rs)
        lines.append(
            f"| {variant} | {len(rs)} | {pct(success)} | {p50:.2f} | {p95:.2f} "
            f"| {avg_calls:.1f} | {tok_in}/{tok_out} | ${cost:.4f} |"
        )
    lines.append("")

    failures: dict[str, int] = {}
    for r in records:
        if not (r["is_grounded"] and r["is_useful"]):
            key = "ungrounded" if not r["is_grounded"] else "not_useful"
            failures[key] = failures.get(key, 0) + 1
        else:
            failures["success"] = failures.get("success", 0) + 1
    taxonomy = " · ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
    lines.append(f"Outcome distribution — {taxonomy}\n")


def e2e_section(lines: list[str]) -> None:
    lines.append("## End-to-end answer quality (LLM-judged)\n")
    result_files = sorted(RESULTS_DIR.glob("e2e_*.json"))
    if not result_files:
        lines.append("_No e2e results yet — run `python -m evals.run_e2e`._\n")
        return

    lines.append("| Variant | slice | n | avg score (0-2) | correct | grounded | p50 s |")
    lines.append("|---|---|---|---|---|---|---|")
    for path in result_files:
        m = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"| {m['variant']} | {m['slice']} | {m['n_questions']} "
            f"| {m['avg_score']:.2f} | {pct(m['pct_correct'])} "
            f"| {pct(m['grounded_rate'])} | {m['p50_duration_s']:.1f} |"
        )
    lines.append("")
    lines.append("_Judge = the pipeline's own model scoring against golden reference answers; self-judge bias applies._\n")


def dataset_counts() -> tuple[int, int, int]:
    """(answerable, unanswerable, grounding examples) from the golden files."""
    golden = json.loads(
        Path("evals/golden/retrieval_golden.json").read_text(encoding="utf-8")
    )
    qs = golden["questions"]
    answerable = sum(1 for q in qs if not q.get("expect_unanswerable"))
    unanswerable = sum(1 for q in qs if q.get("expect_unanswerable"))
    judge = json.loads(
        Path("evals/golden/grounding_judge_set.json").read_text(encoding="utf-8")
    )
    return answerable, unanswerable, len(judge["examples"])


def build() -> None:
    answerable, unanswerable, judge_n = dataset_counts()
    lines = [
        "# Measured Performance",
        "",
        (
            f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — "
            "every number below is produced by `python -m evals.*` against the live "
            'index and LLM; regenerate to refresh._'
        ),
        "",
        (
            f"- Golden retrieval set: {answerable} answerable questions "
            f"+ {unanswerable} unanswerable controls"
        ),
        f"- Grounding-judge set: {judge_n} labeled claims",
        "",
    ]
    retrieval_section(lines)
    grounding_section(lines)
    e2e_section(lines)
    runs_section(lines)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    build()
