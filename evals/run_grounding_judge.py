"""Grounding-judge self-evaluation: how accurate is check_grounding itself?

Runs the live grounding chain over a labeled set and reports accuracy,
precision/recall/F1 for the 'ungrounded' class, plus the key number from
the README's strict-grounding limitation: the false-positive rate on valid
inferences.

Usage:
    python -m evals.run_grounding_judge
"""

import json
from pathlib import Path

from chains import GroundingVerdict, grounding_chain

RESULTS_DIR = Path("evals/results")
GOLDEN_FILE = Path("evals/golden/grounding_judge_set.json")

INFERENCE_CATEGORIES = {
    "inference",
    "numeric_inference",
    "temporal_inference",
    "transitive_inference",
    "paraphrase_inference",
}


def run() -> dict:
    examples = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["examples"]

    tp = fp = tn = fn = 0
    per_example = {}
    category_counts: dict[str, dict[str, int]] = {}

    for ex in examples:
        verdict: GroundingVerdict = grounding_chain.invoke(
            {"answer": ex["answer"], "documents": ex["documents"]}
        )
        got_grounded = verdict.is_grounded
        expected_grounded = ex["expected_grounded"]

        if not expected_grounded and not got_grounded:
            tp += 1  # correctly flagged ungrounded
        elif expected_grounded and not got_grounded:
            fn += 1  # valid claim wrongly flagged (false positive on grounding)
        elif not expected_grounded and got_grounded:
            fp += 1  # unsupported claim slipped through
        else:
            tn += 1

        cat = ex["category"]
        counts = category_counts.setdefault(cat, {"n": 0, "correct": 0})
        counts["n"] += 1
        counts["correct"] += int(got_grounded == expected_grounded)

        per_example[ex["id"]] = {
            "category": cat,
            "expected_grounded": expected_grounded,
            "judged_grounded": got_grounded,
            "correct": got_grounded == expected_grounded,
        }
        print(f"  {ex['id']}: {'OK' if per_example[ex['id']]['correct'] else 'WRONG'} "
              f"(expected grounded={expected_grounded}, judged={got_grounded})")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn

    inference_n = sum(category_counts[c]["n"] for c in INFERENCE_CATEGORIES if c in category_counts)
    inference_wrong = sum(
        category_counts[c]["n"] - category_counts[c]["correct"]
        for c in INFERENCE_CATEGORIES if c in category_counts
    )

    metrics = {
        "n_examples": total,
        "accuracy": (tp + tn) / total,
        "ungrounded_precision": precision,
        "ungrounded_recall": recall,
        "ungrounded_f1": f1,
        "confusion": {
            "true_positives_flagged": tp,
            "false_negatives_missed_hallucinations": fp,
            "false_positives_valid_claims_flagged": fn,
            "true_negatives_correctly_grounded": tn,
        },
        "inference_false_positive_rate": inference_wrong / inference_n if inference_n else 0.0,
        "per_category": category_counts,
        "per_example": per_example,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "grounding_judge.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\naccuracy={metrics['accuracy']:.3f}  "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}  "
          f"inference-FPR={metrics['inference_false_positive_rate']:.3f} -> {out}")
    return metrics


if __name__ == "__main__":
    run()
