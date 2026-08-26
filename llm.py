import logging
import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)


def _configure_langsmith() -> None:
    """LangChain/LangGraph auto-detect tracing once these env vars exist —

    exporting them here (before any chain/graph runs) is enough, no callback
    wiring needed.
    """
    if not settings.langsmith_tracing:
        return
    if not settings.langsmith_api_key:
        logger.warning(
            "langsmith_tracing=true but LANGSMITH_API_KEY is empty; tracing stays off"
        )
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


_configure_langsmith()


@lru_cache(maxsize=8)
def _build_llm(temperature: float) -> ChatOpenAI:
    # OpenRouter exposes an OpenAI-compatible API; the key is read from the
    # environment (loaded from .env by config). Entry points call
    # settings.require_llm_key() to fail fast when absent.
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def default_llm() -> ChatOpenAI:
    """Deterministic grader LLM (settings temperature, default 0)."""
    return _build_llm(settings.llm_temperature)


def make_llm(temperature: float) -> ChatOpenAI:
    """LLM at an explicit temperature (used for retry escalation)."""
    return _build_llm(temperature)


# Shared deterministic instance used by the grading chains.
llm = default_llm()
