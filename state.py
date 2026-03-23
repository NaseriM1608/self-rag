from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    is_grounded: bool
    is_useful: bool
    llm_calls: int

