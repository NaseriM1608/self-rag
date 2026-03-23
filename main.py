from retriever import build_vectorstore
from graph import build_graph

build_vectorstore()
agent = build_graph()

initial_state = {
    "question": "What is the Inverse Cloze Task and how is it used in RAG?",
    "documents": [],
    "generation": "",
    "is_grounded": False,
    "is_useful": False,
    "llm_calls": 0
}

result = agent.invoke(initial_state)


if not result["is_grounded"]:
    print("Answer could not be verified")

elif not result["is_useful"]:
    print("Answer did not address the question")

else:
    print(result["generation"])