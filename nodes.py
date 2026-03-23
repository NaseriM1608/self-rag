from chains import *
from state import AgentState
from retriever import search


def retrieve(state: AgentState) -> dict:
    documents = search(query=state["question"])
    return {
        "documents": documents
    }


def grade_relevance(state: AgentState) -> dict:
    filtered_documents = []

    for doc in state["documents"]:
        result = relevance_chain.invoke({
            "question": state["question"],
            "chunk": doc.page_content
        })

        if result.strip().lower() == "yes":
            filtered_documents.append(doc)

    return {
        "documents": filtered_documents,
        "llm_calls": state["llm_calls"] + 1
    }


def generate(state: AgentState) -> dict:
    documents = "\n\n".join(
        f"[{i + 1}] {doc.page_content} (source: {doc.metadata.get('source', 'unknown')})"
        for i, doc in enumerate(state["documents"])
    )
    result = generation_chain.invoke({
        "question": state["question"],
        "documents": documents
    })

    return {
        "generation": str(result),
        "llm_calls": state["llm_calls"] + 1
    }


def check_grounding(state: AgentState) -> dict:
    documents = '\n\n'.join(f"{doc.page_content}" for doc in state["documents"])

    result = grounding_chain.invoke({
        "answer": state["generation"],
        "documents": documents
    })

    decision = result.strip().lower()

    return {
        "is_grounded": decision == "yes",
        "llm_calls": state["llm_calls"] + 1
    }


def check_usefulness(state: AgentState) -> dict:
    result = usefulness_chain.invoke({
        "question": state["question"],
        "answer": state["generation"]
    })

    decision = result.strip().lower()

    return {
        "is_useful": decision == "yes",
        "llm_calls": state["llm_calls"] + 1
    }


