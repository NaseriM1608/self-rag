from langgraph.graph import StateGraph, END
from nodes import *


MAX_LLM_CALLS = 10


def route_after_retrieval(state):
    if state["should_retrieve"]:
        return "retrieve"
    return "generate"


def route_after_grading(state):
    if not state["documents"]:
        return END
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return END
    return "generate"


def route_after_generation(state):
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return END
    if not state["documents"]:
        return "check_usefulness"
    return "check_grounding"


def route_after_checking_grounding(state):
    if not state["is_grounded"]:
        if state["llm_calls"] >= MAX_LLM_CALLS:
            return END
        return "generate"
    return "check_usefulness"


graph = StateGraph(AgentState) # type: ignore

graph.add_node("decide_retrieval", decide_retrieval) # type: ignore
graph.add_node("retrieve", retrieve) # type: ignore
graph.add_node("grade_relevance", grade_relevance) # type: ignore
graph.add_node("generate", generate) # type: ignore
graph.add_node("check_grounding", check_grounding) # type: ignore
graph.add_node("check_usefulness", check_usefulness) # type: ignore

graph.set_entry_point("decide_retrieval")
graph.add_edge("retrieve", "grade_relevance")
graph.add_edge("check_usefulness", END)
graph.add_conditional_edges("decide_retrieval", route_after_retrieval)
graph.add_conditional_edges("grade_relevance", route_after_grading)
graph.add_conditional_edges("generate", route_after_generation)
graph.add_conditional_edges("check_grounding", route_after_checking_grounding)

agent = graph.compile()

