# Self-RAG with LangGraph

A Self-Reflective Retrieval-Augmented Generation pipeline built with LangGraph, ChromaDB, and Groq. The system retrieves documents, grades their relevance, generates answers, and verifies grounding and usefulness before returning a response — looping back to regenerate when verification fails.

---

## What is Self-RAG

Standard RAG always retrieves and always trusts what it generates. Self-RAG adds a layer of self-criticism: the system grades each retrieved document for relevance, checks whether the generated answer is actually supported by the documents, and checks whether the answer addresses the question. If any check fails, the system loops back or exits with a specific failure reason rather than returning a bad answer.

---

## Architecture

```
START
  |
retrieve
  |
grade_relevance  -->  [no relevant documents]  -->  END
  |
generate
  |
check_grounding  -->  [not grounded, retries < MAX]  -->  generate
  |              -->  [not grounded, retries >= MAX]  -->  END
check_usefulness -->  [not useful]  -->  END
  |
END (return generation)
```

Every grading decision is a conditional edge in the LangGraph graph. The graph never proceeds past a failed check without either looping back or exiting cleanly.

---

## Design Decisions

**Always retrieve**
The system always retrieves from the vector store rather than deciding whether retrieval is needed. This avoids the risk of the model answering from internal knowledge when document-grounded answers are required.

**Two-model setup**
All nodes use llama-3.3-70b-versatile. For a portfolio project, consistency and grading accuracy matter more than generation speed.

**Strict grounding**
The grounding check is designed to catch not just hallucinations but also reasonable inferences presented as facts. If the answer contains any claim not directly stated in the retrieved documents, it is marked as ungrounded and regenerated.

**Loop protection**
A `llm_calls` counter in the state tracks the total number of LLM calls. If the counter exceeds `MAX_LLM_CALLS`, the graph exits regardless of grading results to prevent infinite loops.

**Specific exit conditions**
The system distinguishes between three failure modes: no relevant documents found, answer could not be grounded after maximum retries, and answer did not address the question. Each produces a different message rather than a generic failure.

---

## Project Structure

```
self-rag/
├── state.py         # AgentState TypedDict
├── retriever.py     # ChromaDB vector store and search
├── prompts.py       # All ChatPromptTemplate definitions
├── llm.py           # Groq LLM instances
├── nodes.py         # LangGraph node functions
├── graph.py         # Graph construction and conditional routing
├── main.py          # Entry point
└── documents/       # .txt files to index
```

---

## Stack

- LangGraph — graph construction and stateful execution
- ChromaDB — vector store with persistent storage
- BAAI/bge-m3 — local embeddings via sentence-transformers
- Groq — LLM inference (llama-3.1-8b-instant, llama-3.3-70b-versatile)
- LangChain — prompt templates and LCEL chains

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/self-rag.git
cd self-rag
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free API key at console.groq.com.

**4. Add documents**

Place `.txt` files in the `documents/` folder. These will be chunked, embedded, and stored in ChromaDB on first run.

**5. Run**

Open `main.py`, set your question:

```python
initial_state = {
    "question": "your question here",
    ...
}
```

Then run:

```bash
python main.py
```

---

## Example Output

**Question:** What is the Inverse Cloze Task and how is it used in RAG?

**Answer:** The Inverse Cloze Task (ICT) is a technique that helps the model learn retrieval patterns by predicting masked text within documents. It is used in RAG as a method for pre-training the retriever.

**Grounding check:** Passed — every claim is directly supported by the source document.

---

**Question:** The ICT improves the retriever's ability to fetch relevant documents.

**Grounding check:** Failed — the document states ICT is used for pre-training but does not claim it improves retrieval ability. That is an inference, not a stated fact. The system loops back to regenerate.

---

## Limitations

- Inference detection is imperfect even with stronger models. Plausible but unsupported conclusions can sometimes pass the grounding check.
- The system always retrieves, so questions answerable from general knowledge will still query the vector store. This is intentional but less efficient than a retrieval decision step.
- BGE-M3 runs on CPU by default. Embedding speed depends on machine resources.
