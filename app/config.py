"""Centralized environment variable management.

All settings are read from environment variables (or a ``.env`` file in the project root).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "DATA"
    processed_data_dir: Path = PROJECT_ROOT / "processed_data"

    # --- Jina AI (embeddings + reranker) ---
    jina_api_key: SecretStr | None = None
    jina_base_url: str = "https://api.jina.ai/v1"
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_reranker_model: str = "jina-reranker-v2-base-multilingual"
    jina_timeout_seconds: float = 30.0
    jina_max_retries: int = 3
    embedding_dim: int = 1024
    embedding_batch_size: int = 64

    # --- Local embedding fallback ---
    local_embeddings_enabled: bool = True
    local_embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"

    # --- Qdrant ---
    # Set qdrant_url (+ api key) for Qdrant Cloud, or qdrant_path for embedded local mode
    # (a directory, or ":memory:"). qdrant_url takes precedence.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str | None = None
    qdrant_collection: str = "enterprise_rag"
    qdrant_timeout_seconds: int = 60

    # --- Portkey LLM gateway ---
    portkey_api_key: SecretStr | None = None
    portkey_base_url: str | None = None  # only for a self-hosted gateway
    # Saved Portkey config slug ("pc-..."). Required when the workspace blocks inline configs;
    # it then defines routing/fallback instead of LLM_MODELS (see DOCS/SETUP.md).
    portkey_config_id: str | None = None
    # Comma-separated Portkey Model Catalog ids in fallback order. The slug after "@" is the name of
    # your OpenAI integration in Portkey; the second model is the fallback when the first is
    # rate-limited or erroring.
    llm_models: str = "@openai/gpt-4o,@openai/gpt-4o-mini"
    # Model for short, latency-sensitive calls (guardrail checks, planning), called directly through
    # Portkey; falls back to the LLM_MODELS route on failure. Empty = use LLM_MODELS for everything.
    llm_fast_model: str | None = "@openai/gpt-4o-mini"
    llm_fallback_on_status_codes: str = "429,500,502,503,504"  # empty = fall back on any non-2xx
    llm_retry_attempts: int = 2  # per target, before falling back
    llm_timeout_seconds: int = 60
    llm_temperature: float = 0.0
    # USD per 1M tokens as [input, output], matched by longest model-name prefix. Used for cost per
    # query in API responses, metrics and evals. Override with JSON, e.g.
    # LLM_PRICES='{"gpt-4o": [2.5, 10], "gpt-4o-mini": [0.15, 0.6]}'
    llm_prices: dict[str, tuple[float, float]] = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    }
    llm_max_tokens: int = 1024

    # --- API security ---
    api_keys: SecretStr | None = None  # comma-separated bearer tokens; unset = auth disabled
    rate_limit_per_minute: int = 30  # per API key (or client IP when auth is disabled); 0 = off
    redis_url: str | None = None  # e.g. redis://localhost:6379/0; unset = in-memory limiter
    max_question_chars: int = 4000

    # --- Observability ---
    # Pydantic Logfire: enabled when a write token is set (or logfire_console for local debugging).
    logfire_token: SecretStr | None = None
    logfire_service_name: str = "enterprise-rag"
    logfire_environment: str = "development"
    logfire_console: bool = False
    # LangSmith: enabled when langsmith_tracing is true and an API key is set.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "enterprise-rag"
    langsmith_endpoint: str | None = None  # e.g. https://eu.api.smith.langchain.com

    # --- Guardrails ---
    guardrails_enabled: bool = True
    guardrails_output_enabled: bool = True
    # If a rail check itself errors (e.g. LLM outage): False = block the request (fail closed).
    guardrails_fail_open: bool = False
    guardrails_max_input_chars: int = 4000
    guardrails_allowed_topics: str = (
        "Kubernetes workloads and operations (Jobs, CronJobs, Pods, autoscaling, parallel work queues, "
        "cluster architecture, monitoring) and Databricks job automation (CLI, SDKs, REST API)."
    )

    # --- Agent (LangGraph) ---
    # Run the planner concurrently with the input guardrail (plan discarded if the input is blocked).
    agent_parallel_planning: bool = True
    agent_max_iterations: int = 2  # planner -> retriever loops before answering with what we have
    agent_max_search_queries: int = 3
    agent_max_context_chunks: int = 6  # chunks passed to the responder
    agent_history_turns: int = 6  # past user/assistant turns kept in conversation memory
    agent_answer_max_tokens: int = 1024

    # --- Evaluation (RAGAS) ---
    # Judge LLM + embeddings, routed through Portkey to your OpenAI provider.
    eval_judge_model: str = "@openai/gpt-4o-mini"
    eval_embedding_model: str = "@openai/text-embedding-3-small"
    eval_concurrency: int = 4  # samples scored in parallel
    # Judge output limit. RAGAS's default (1024) truncates statement extraction for long answers.
    eval_judge_max_tokens: int = 4096

    # --- Retrieval ---
    retrieval_top_k: int = 20  # candidates fetched from Qdrant
    rerank_top_n: int = 5  # chunks returned after reranking
    rerank_max_candidates: int = 40  # merged multi-query candidates sent to the reranker (best vector scores)
    rerank_min_score: float = 0.2  # drop reranked chunks below this relevance (noise filter)

    @field_validator("*", mode="before")
    @classmethod
    def _blank_means_unset(cls, value, info: ValidationInfo):
        # `KEY=` in .env yields "" — treat it as "not configured" for optional settings.
        if isinstance(value, str) and not value.strip() and cls.model_fields[info.field_name].default is None:
            return None
        return value

    @property
    def api_key_list(self) -> list[str]:
        if self.api_keys is None:
            return []
        return [k.strip() for k in self.api_keys.get_secret_value().split(",") if k.strip()]

    @property
    def llm_model_list(self) -> list[str]:
        return [m.strip() for m in self.llm_models.split(",") if m.strip()]

    @property
    def llm_fallback_status_code_list(self) -> list[int]:
        return [int(c) for c in self.llm_fallback_on_status_codes.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
