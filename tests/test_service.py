"""Service-layer tests with a fake pipeline — no network, no model loads."""

from fastapi.testclient import TestClient

import graph
import service
from metrics import RunRecord


class FakeAgent:
    def invoke(self, state, config=None):
        state["generation"] = "RAG stores embeddings in a vector database."
        state["is_grounded"] = True
        state["is_useful"] = True
        state["llm_calls"] = 8
        state["generation_attempts"] = 1
        return state


def _make_record(**overrides) -> RunRecord:
    base = {
        "question": "q",
        "variant": "hybrid",
        "duration_s": 1.2,
        "llm_calls": 8,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.0,
        "generation_attempts": 1,
        "is_grounded": True,
        "is_useful": True,
        "documents_kept": 3,
        "timestamp_utc": "2026-08-26T00:00:00+00:00",
        "generation": "an answer",
    }
    base.update(overrides)
    return RunRecord(**base)


def test_health_reports_retriever():
    with TestClient(service.app) as client:
        res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_query_returns_answer_with_telemetry(monkeypatch):
    # graph.build_graph is imported inside the lifespan, so patch its source.
    monkeypatch.setattr(graph, "build_graph", lambda *a: FakeAgent())
    monkeypatch.setattr(
        "metrics.track_query",
        lambda agent, q, variant: _make_record(question=q),
    )
    with TestClient(service.app) as client:
        res = client.post(
            "/query", json={"question": "Where does RAG store embeddings?"}
        )
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "an answer"
    assert body["is_grounded"] and body["is_useful"]
    assert body["llm_calls"] == 8
    assert body["variant"] == "hybrid"


def test_query_validates_input():
    with TestClient(service.app) as client:
        res = client.post("/query", json={"question": "hi"})
    assert res.status_code == 422
