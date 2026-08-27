# Self-RAG with LangGraph + Neo4j

![CI](https://github.com/NaseriM1608/self-rag/actions/workflows/ci.yml/badge.svg)

A Self-Reflective Retrieval-Augmented Generation pipeline built with LangGraph,
OpenRouter, and Neo4j. The system retrieves documents (hybrid vector + BM25,
optionally expanded through a knowledge graph), grades their relevance,
generates cited answers, and verifies grounding and usefulness before
returning a response — looping back to regenerate when verification fails.

Measured numbers live in [docs/METRICS.md](docs/METRICS.md); every figure
there is produced by the evaluation harness in `evals/`.

---

## What is Self-RAG

Standard RAG always retrieves and always trusts what it generates. Self-RAG
adds a layer of self-criticism: the system grades each retrieved document for
relevance, checks whether the generated answer is actually supported by the
documents, and checks whether the answer addresses the question. If any check
fails, the system loops back or exits with a specific failure reason rather
than returning a bad answer.

Reflection here is implemented as separate grader nodes with structured-output
LLM verdicts (relevance / grounding / usefulness) that drive conditional edges
in the graph — not as literal reflection tokens as in the original paper.

---

## Architecture

```
START
  |
retrieve  (Neo4j: vector + fulltext, RRF fusion; optional KG expansion)
  |
grade_relevance  -->  [no relevant documents]  -->  END
  |
generate  (citations required: [1: Source Name]; retry temp escalates)
  |
check_grounding  -->  [not grounded, retries < MAX]  -->  generate
  |              -->  [not grounded, retries >= MAX]  -->  END
check_usefulness -->  [not useful]  -->  END
  |
END (return generation)
```

Every grading decision is a conditional edge in the LangGraph graph. The graph
never proceeds past a failed check without either looping back or exiting
cleanly. A `llm_calls` counter in the state enforces a hard budget
(`MAX_LLM_CALLS`, default 25) so the loop can never run away.

### Retrieval variants (`retrievers.py`)

| Name | What it does |
|---|---|
| `dense` | ChromaDB vectors — frozen eval baseline only |
| `neo4j-dense` | Neo4j native vector index (bge-m3 embeddings) |
| `fulltext` | Neo4j Lucene fulltext (BM25-style lexical) |
| `hybrid` | vector + lexical fused with Reciprocal Rank Fusion (default) |
| `kg` | one-hop knowledge-graph triples as documents |
| `hybrid+kg` | hybrid chunks plus appended KG triples |
| `graph-expand` | hybrid, with slots reserved for graph-bridged chunks from unseen sources (targets multi-hop) |

The knowledge graph (`kg.py`) is LLM-extracted from the indexed chunks:
`(:Entity)-[:RELATES]->(:Entity)` and `(:Chunk)-[:MENTIONS]->(:Entity)`.
KG-derived context is never trusted blindly — it passes through the same
relevance/grounding gates as any other retrieved document.

### FastAPI service (`service.py`)

`GET /health` reports readiness; `POST /query` runs the full graph and returns
the answer plus verification flags and telemetry (LLM calls, tokens, duration,
cost). Prometheus metrics at `/metrics` when `prometheus-client` is installed.

---

## Design Decisions

**Always retrieve**
The system always retrieves rather than deciding whether retrieval is needed.
This avoids the risk of the model answering from internal knowledge when
document-grounded answers are required.

**Single model, structured verdicts**
All nodes use one OpenRouter model (`stealth/ox-alpha` by default) with
`with_structured_output` for grader verdicts. For a portfolio project,
consistency and grading accuracy matter more than generation speed.

**Strict grounding**
The grounding check is designed to catch not just hallucinations but also
reasonable inferences presented as facts. If the answer contains any claim not
directly stated in the retrieved documents, it is marked as ungrounded and
regenerated. Known trade-off: valid inferences are sometimes over-flagged —
the harness measures this rate (`evals/run_grounding_judge.py`).

**Retry temperature escalation**
Regenerating at temperature 0 reproduces the same ungrounded answer forever,
so each grounding-failure retry raises the temperature
(`0.0 → +0.35/attempt, cap 0.7`).

**Loop protection**
The `llm_calls` counter bounds total chain invocations per run. Grading costs
one call per candidate document, so the typical happy path already spends ~8.

**Specific exit conditions**
The system distinguishes between three failure modes: no relevant documents
found, answer could not be grounded after maximum retries, and answer did not
address the question. Each produces a different message rather than a generic
failure.

---

## Project Structure

```
self-rag/
├── main.py            # CLI entry: sync index, answer one question
├── graph.py           # LangGraph assembly and conditional routing
├── nodes.py           # Graph node functions (grade/generate/check)
├── chains.py          # Prompts + structured-output LCEL chains
├── state.py           # AgentState TypedDict
├── llm.py             # OpenRouter LLM instances (ChatOpenAI)
├── config.py          # pydantic-settings configuration
├── retrievers.py      # Retriever protocol + 7 backends (see table)
├── retriever.py       # Shared ingestion utils (load/split/embed/chunk ids)
├── neo4j_store.py     # Neo4j persistence: vector/fulltext index, sync
├── kg.py              # Knowledge-graph extraction + graph retrieval
├── metrics.py         # Token/cost telemetry → evals/results/runs.jsonl
├── service.py         # FastAPI service (/health, /query, /metrics)
├── evals/             # Eval harness + golden sets + results
│   ├── run_retrieval.py        # Recall@k / MRR / latency per variant
│   ├── run_e2e.py              # full-graph runs, LLM-judged 0-2 scores
│   ├── run_grounding_judge.py  # grounding-checker self-accuracy
│   └── report.py               # aggregates results → docs/METRICS.md
├── tests/             # offline unit + service tests; live-marked benchmark
├── documents/         # .txt corpus to index
└── docs/METRICS.md    # measured performance
```

---

## Stack

- LangGraph — graph construction and stateful execution
- Neo4j 5 — chunk store, native vector index, Lucene fulltext, knowledge graph
- BAAI/bge-m3 — local embeddings via sentence-transformers (1024-dim)
- OpenRouter — LLM inference (`stealth/ox-alpha` by default)
- FastAPI — HTTP service; LangChain/LCEL — prompts and chains
- pytest / ruff / GitHub Actions — offline tests, lint, CI (live evals on demand)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/NaseriM1608/self-rag.git
cd self-rag
```

**2. Install dependencies** (Python 3.10)

```bash
pip install -r requirements-dev.txt   # runtime + service + evals + tests
```

**2. Configure secrets** — copy `.env.example` to `.env` and fill in:

```
OPENROUTER_API_KEY=...   # https://openrouter.ai (GROQ_API_KEY also accepted)
NEO4J_PASSWORD=...       # from your Neo4j instance
```

**3. Run Neo4j** — Neo4j Desktop, plain Docker:

```bash
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j:5
```

—or the whole stack (app + Neo4j + one-shot ingestion) with compose:

```bash
docker compose up -d
docker compose run --rm ingest   # sync chunks + build the knowledge graph
```

**4. Build the index and knowledge graph**

```bash
python main.py                        # syncs chunks + answers a demo question
python -c "import kg; kg.build_knowledge_graph()"   # one-time KG extraction
```

Chunk sync is incremental (content-hash ids); KG extraction is cached, so
rebuilds only process new chunks.

**5. Ask questions**

```bash
python main.py
# or serve the API:
uvicorn service:app --reload
```

Switch retrieval backends with `DEFAULT_RETRIEVER=graph-expand` (or any name
from the table above).

---

## Evaluation

```bash
python -m evals.run_retrieval --variant hybrid   # Recall@k / MRR / latency
python -m evals.run_e2e --variant hybrid --slice multi_hop
python -m evals.run_grounding_judge              # checker self-accuracy
python -m evals.report                           # regenerate docs/METRICS.md
```

Offline tests (`pytest`) need no network or index; the live grounding
benchmark runs with `pytest -m live` when an API key is set. CI runs lint +
offline tests on every push, with a manual/nightly live-eval job.

---

## Limitations

- KG retrievers currently match but do not beat plain hybrid on multi-hop
  end-to-end questions (see METRICS.md). Diagnosed in
  [docs/KG_GAP_ANALYSIS.md](docs/KG_GAP_ANALYSIS.md): hybrid already covers
  every needed source for 5 of 6 multi-hop questions, so the remaining gap is
  generation-side synthesis — two-hop expansion was evaluated and deferred.
- Inference detection is imperfect even with stronger models. Plausible but
  unsupported conclusions can sometimes pass the grounding check, and valid
  inferences are sometimes over-flagged.
- The system always retrieves, so questions answerable from general knowledge
  will still query the store. This is intentional but less efficient than a
  retrieval decision step.
- BGE-M3 runs on CPU by default. Embedding speed depends on machine resources.
