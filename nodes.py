from langchain_core.output_parsers import StrOutputParser
from prompts import *
from state import AgentState
from llm import llm
from retriever import search


def decide_retrieval(state: AgentState) -> AgentState:
    retrieval_chain = retrieval_prompt | llm | StrOutputParser()

    result = retrieval_chain.invoke({
        "question": state["question"]
    })

    decision = result.strip().lower()

    return {
        "should_retrieve": decision == "yes"
    }


def retrieve(state: AgentState) -> AgentState:
    documents = search(query=state["question"])
    return {
        "documents": documents
    }


def grade_relevance(state: AgentState) -> AgentState:
    relevance_chain = relevance_prompt | llm | StrOutputParser()

    filtered_documents = []

    for doc in state["documents"]:
        result = relevance_chain.invoke({
            "question": state["question"],
            "chunk": doc.page_content
        })

        if result.strip().lower() == "yes":
            filtered_documents.append(doc)

    return {
        "documents": filtered_documents
    }


def generate(state: AgentState) -> AgentState:
    generation_chain = generation_prompt | llm | StrOutputParser()

    documents = "\n\n".join(
        f"[{i + 1}] {doc.page_content} (source: {doc.metadata.get('source', 'unknown')})"
        for i, doc in enumerate(state["documents"])
    )
    result = generation_chain.invoke({
        "question": state["question"],
        "documents": documents
    })

    return {
        "generation": str(result)
    }


def check_grounding(state: AgentState) -> AgentState:
    grounding_chain = grounding_prompt | llm | StrOutputParser()

    documents = '\n\n'.join(f"{doc.page_content}" for doc in state["documents"])

    result = grounding_chain.invoke({
        "answer": state["generation"],
        "documents": documents
    })

    decision = result.strip().lower()

    return {
        "is_grounded": decision == "yes"
    }


def check_usefulness(state: AgentState) -> AgentState:
    usefulness_chain = usefulness_prompt | llm | StrOutputParser()

    result = usefulness_chain.invoke({
        "question": state["question"],
        "answer": state["generation"]
    })

    decision = result.strip().lower()

    return {
        "is_useful": decision == "yes"
    }


