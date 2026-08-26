"""CLI entry point: sync the index, then answer one question."""

import logging

from config import settings
from graph import build_graph
from neo4j_store import sync_neo4j_index

logger = logging.getLogger(__name__)


def main() -> None:
    settings.require_llm_key()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    stats = sync_neo4j_index()
    print(
        f"Index ready: {stats['total']} chunks "
        f"(+{stats['added']} new, -{stats['removed']} removed)"
    )

    agent = build_graph()
    initial_state = {
        "question": "What is the Inverse Cloze Task and how is it used in RAG?",
        "documents": [],
        "generation": "",
        "generation_attempts": 0,
        "is_grounded": False,
        "is_useful": False,
        "llm_calls": 0,
    }
    result = agent.invoke(initial_state)

    if not result["is_grounded"]:
        print("Answer could not be verified")
    elif not result["is_useful"]:
        print("Answer did not address the question")
    else:
        print(result["generation"])

    print(
        f"[run] llm_calls={result['llm_calls']} "
        f"generation_attempts={result['generation_attempts']}"
    )


if __name__ == "__main__":
    main()
