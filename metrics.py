"""Run telemetry: JSONL records capturing latency, token usage, and cost.

Every graph invocation can be recorded as one JSON line in
``evals/results/runs.jsonl`` so reports aggregate real numbers instead of
descriptions. Token usage is captured through a LangChain callback handler;
cost uses a price table that must be kept in sync with Groq's pricing page.
"""

import json
import logging
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# USD per million tokens (input, output) — keep in sync with provider pricing
# (openrouter.ai/models). stealth/ox-alpha is currently listed at $0/$0.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "stealth/ox-alpha": (0.0, 0.0),
}

RUNS_FILE = Path("evals/results/runs.jsonl")

# Per-query accumulation slot; contextvar keeps concurrent API requests
# isolated. Default is None — track_query() sets a fresh list per run.
_token_usage: ContextVar[list[dict[str, int]] | None] = ContextVar(
    "token_usage", default=None
)


class TokenUsageHandler(BaseCallbackHandler):
    """Accumulates usage_metadata from every LLM call into the context slot."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage_list = _token_usage.get()
        if usage_list is None:
            return  # LLM call outside track_query — nothing to attribute.
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    usage_list.append(
                        {
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        }
                    )


@dataclass
class RunRecord:
    """One end-to-end query execution with measured numbers."""

    question: str
    variant: str
    duration_s: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    generation_attempts: int
    is_grounded: bool
    is_useful: bool
    documents_kept: int
    timestamp_utc: str
    generation: str = ""
    error: str | None = None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES_PER_MTOK.get(model)
    if prices is None:
        logger.warning("No price entry for model %s; cost reported as 0", model)
        return 0.0
    in_price, out_price = prices
    return input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price


def track_query(agent: Any, question: str, variant: str = "baseline") -> RunRecord:
    """Invoke the graph once and return a fully-populated RunRecord.

    Also appends the record to RUNS_FILE so reports can aggregate history.
    """
    from config import settings

    token_slot: list[dict[str, int]] = []
    token_var = _token_usage.set(token_slot)

    start = time.perf_counter()
    try:
        result = agent.invoke(
            {
                "question": question,
                "documents": [],
                "generation": "",
                "generation_attempts": 0,
                "is_grounded": False,
                "is_useful": False,
                "llm_calls": 0,
            },
            config={"callbacks": [TokenUsageHandler()]},
        )
    finally:
        _token_usage.reset(token_var)
    duration = time.perf_counter() - start

    input_tokens = sum(u["input_tokens"] for u in token_slot)
    output_tokens = sum(u["output_tokens"] for u in token_slot)

    record = RunRecord(
        question=question,
        variant=variant,
        duration_s=round(duration, 3),
        llm_calls=result.get("llm_calls", 0),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(
            estimate_cost(settings.llm_model, input_tokens, output_tokens), 6
        ),
        generation_attempts=result.get("generation_attempts", 0),
        is_grounded=result.get("is_grounded", False),
        is_useful=result.get("is_useful", False),
        documents_kept=len(result.get("documents", [])),
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generation=str(result.get("generation", "")),
    )
    append_record(record)
    return record


def append_record(record: RunRecord) -> None:
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_records(path: Path = RUNS_FILE) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
