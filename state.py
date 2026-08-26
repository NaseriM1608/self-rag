from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    # Number of times generate has run for this question; drives temperature
    # escalation so failed-grounding retries are not deterministic repeats.
    generation_attempts: int
    is_grounded: bool
    is_useful: bool
    # True count of LLM chain invocations (grading is one call per document).
    llm_calls: int
