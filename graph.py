from langgraph.graph import END, StateGraph

from config import settings
from nodes import (
    check_grounding,
    check_usefulness,
    generate,
    grade_relevance,
    make_retrieve_node,
)
from retrievers import Retriever, get_retriever
from state import AgentState

MAX_LLM_CALLS = settings.max_llm_calls


def route_after_grading(state: AgentState) -> str:
    if not state["documents"]:
        return END
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return END
    return "generate"


def route_after_generation(state: AgentState) -> str:
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return END
    return "check_grounding"


def route_after_checking_grounding(state: AgentState) -> str:
    if not state["is_grounded"]:
        if state["llm_calls"] >= MAX_LLM_CALLS:
            return END
        return "generate"
    return "check_usefulness"


def build_graph(retriever: Retriever | None = None):
    retriever = retriever or get_retriever(settings.default_retriever)
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("generate", generate)
    graph.add_node("check_grounding", check_grounding)
    graph.add_node("check_usefulness", check_usefulness)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_edge("check_usefulness", END)
    graph.add_conditional_edges("grade_relevance", route_after_grading)
    graph.add_conditional_edges("generate", route_after_generation)
    graph.add_conditional_edges("check_grounding", route_after_checking_grounding)

    agent = graph.compile()
    return agent
