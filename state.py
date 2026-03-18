from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    should_retrieve: bool
    relevance_scores: List[str]
    is_grounded: bool
    is_useful: bool

