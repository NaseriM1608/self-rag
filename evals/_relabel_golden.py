"""Derive cross-document labels for the golden set mechanically.

A question is cross-document only when its `relevant_snippets` appear
verbatim in 2+ distinct files under documents/. Hand labels drift (an
earlier pass over-counted six single-file questions as multi-hop and
missed one genuine cross-document case), so the flag is recomputed from
the corpus instead of trusted.

Usage:
    python -m evals._relabel_golden
"""

import glob
import json
import os
import re
from pathlib import Path

GOLDEN_FILE = Path("evals/golden/retrieval_golden.json")

LABELING_NOTE = (
    "multi_hop/cross_document are derived mechanically from which documents/ "
    "files contain each relevant_snippet verbatim - a question counts as "
    "cross-document only when its evidence spans 2+ distinct files. An earlier "
    "hand-authored multi_hop label over-counted 6 questions whose evidence is "
    "single-file; those are now false. Regenerate with evals/_relabel_golden.py."
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def snippet_host_files(snippets: list[str], corpus: dict[str, str]) -> list[str]:
    return sorted(
        {
            name
            for snippet in snippets
            for name, text in corpus.items()
            if _normalize(snippet) in text
        }
    )


def relabel() -> list[tuple[str, bool, bool]]:
    corpus = {
        os.path.basename(path): _normalize(
            Path(path).read_text(encoding="utf-8", errors="replace")
        )
        for path in glob.glob("documents/*.txt")
    }
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))

    changed = []
    for question in golden["questions"]:
        if question.get("expect_unanswerable"):
            continue
        hosts = snippet_host_files(question["relevant_snippets"], corpus)
        question["snippet_source_files"] = hosts
        question["cross_document"] = len(hosts) >= 2
        previous = bool(question.get("multi_hop"))
        if previous != question["cross_document"]:
            changed.append((question["id"], previous, question["cross_document"]))
        question["multi_hop"] = question["cross_document"]

    golden["_labeling"] = LABELING_NOTE
    GOLDEN_FILE.write_text(
        json.dumps(golden, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return changed


if __name__ == "__main__":
    for qid, was, now in relabel():
        print(f"  {qid}: multi_hop {was} -> {now}")
