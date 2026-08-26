"""FastAPI service exposing the Self-RAG pipeline.

POST /query runs the full graph (retrieve -> grade -> generate -> verify)
with per-run telemetry. GET /health reports readiness. The retriever stack
follows settings.default_retriever ('hybrid' today).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class QueryResponse(BaseModel):
    answer: str | None
    is_grounded: bool
    is_useful: bool
    llm_calls: int
    generation_attempts: int
    duration_s: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    variant: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.require_llm_key()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from graph import build_graph

    app.state.agent = build_graph()
    logger.info("Self-RAG ready (retriever=%s)", settings.default_retriever)
    yield
    app.state.agent = None


app = FastAPI(title="Self-RAG", version="1.0.0", lifespan=lifespan)

try:
    from prometheus_client import make_asgi_app

    app.mount("/metrics", make_asgi_app())
except ImportError:  # pragma: no cover - metrics are optional
    logger.warning("prometheus-client not installed; /metrics disabled")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retriever": settings.default_retriever}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    from metrics import track_query

    if getattr(app.state, "agent", None) is None:
        raise HTTPException(status_code=503, detail="pipeline not initialized")

    try:
        record = track_query(
            app.state.agent, request.question, variant=settings.default_retriever
        )
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(status_code=502, detail=str(exc)[:300]) from exc

    # A failed verification still returns the draft plus its flags so callers
    # can decide; ungrounded answers are explicitly marked.
    return QueryResponse(
        answer=record.generation or None,
        is_grounded=record.is_grounded,
        is_useful=record.is_useful,
        llm_calls=record.llm_calls,
        generation_attempts=record.generation_attempts,
        duration_s=record.duration_s,
        cost_usd=record.cost_usd,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        variant=record.variant,
    )
