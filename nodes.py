import logging

from chains import (
    GroundingVerdict,
    RelevanceVerdict,
    UsefulnessVerdict,
    grounding_chain,
    make_generation_chain,
    relevance_chain,
    usefulness_chain,
)
from config import settings
from state import AgentState

logger = logging.getLogger(__name__)


def make_retrieve_node(retriever):
    """Build a retrieve node bound to a specific Retriever implementation."""

    def retrieve(state: AgentState) -> dict:
        return {"documents": retriever.search(state["question"])}

    return retrieve


def grade_relevance(state: AgentState) -> dict:
    candidates = state["documents"]
    filtered = []
    for doc in candidates:
        verdict: RelevanceVerdict = relevance_chain.invoke(
            {"question": state["question"], "chunk": doc.page_content}
        )
        if verdict.is_relevant:
            filtered.append(doc)

    dropped = len(candidates) - len(filtered)
    if dropped:
        logger.info("Grading kept %d of %d candidate documents", len(filtered), len(candidates))

    # True call accounting: grading is one LLM invocation per candidate.
    return {"documents": filtered, "llm_calls": state["llm_calls"] + len(candidates)}


def generate(state: AgentState) -> dict:
    attempt = state.get("generation_attempts", 0)
    # Escalate temperature on retries so regeneration is not a deterministic
    # repeat of the same ungrounded answer.
    temperature = min(
        settings.llm_temperature + settings.retry_temperature_step * attempt,
        settings.max_retry_temperature,
    )
    documents = "\n\n".join(
        f"[{i + 1}] {doc.page_content} (source: {doc.metadata.get('source', 'unknown')})"
        for i, doc in enumerate(state["documents"])
    )
    result = make_generation_chain(temperature).invoke(
        {"question": state["question"], "documents": documents}
    )

    return {
        "generation": str(result),
        "generation_attempts": attempt + 1,
        "llm_calls": state["llm_calls"] + 1,
    }


def check_grounding(state: AgentState) -> dict:
    documents = "\n\n".join(doc.page_content for doc in state["documents"])
    verdict: GroundingVerdict = grounding_chain.invoke(
        {"answer": state["generation"], "documents": documents}
    )
    return {"is_grounded": verdict.is_grounded, "llm_calls": state["llm_calls"] + 1}


def check_usefulness(state: AgentState) -> dict:
    verdict: UsefulnessVerdict = usefulness_chain.invoke(
        {"question": state["question"], "answer": state["generation"]}
    )
    return {"is_useful": verdict.is_useful, "llm_calls": state["llm_calls"] + 1}
