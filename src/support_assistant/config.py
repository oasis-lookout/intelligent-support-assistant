"""Environment-driven configuration. Stdlib only so it is trivially testable."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

DEV_SECRET = "dev-only-secret-change-me-please-0123456789"

# One pipeline, three wiring profiles. Every value can still be overridden individually.
MODE_DEFAULTS = {
    "local":      dict(embedding="hash",    llm="extractive", store="memory", dim=512,  min_conf=0.25),
    "aws":        dict(embedding="bedrock", llm="bedrock",    store="qdrant", dim=1024, min_conf=0.35),
    "selfhosted": dict(embedding="openai",  llm="openai",     store="qdrant", dim=1024, min_conf=0.40),
}
EMBEDDING_PROVIDERS = {"hash", "bedrock", "openai"}
LLM_PROVIDERS = {"extractive", "bedrock", "openai"}
STORES = {"memory", "qdrant"}


@dataclass(frozen=True)
class Settings:
    app_env: str = "dev"
    deployment_mode: str = "local"
    embedding_provider: str = "hash"
    llm_provider: str = "extractive"
    vector_store: str = "memory"
    vector_store_path: str = "var/vectors"
    embedding_dim: int = 512

    aws_region: str = "ap-south-1"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_llm_model_id: str = ""
    bedrock_guardrail_id: str = ""
    bedrock_guardrail_version: str = ""

    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = ""
    llm_model: str = ""
    embed_model: str = "BAAI/bge-m3"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "support_chunks"

    database_url: str = "sqlite:///var/conversations.db"
    session_secret: str = DEV_SECRET
    session_ttl_seconds: int = 3600
    api_keys_file: str = "var/api_keys.json"

    top_k: int = 8
    context_k: int = 3
    min_confidence: float = 0.25
    min_hits: int = 1
    history_turns: int = 4
    max_question_chars: int = 1000
    max_answer_tokens: int = 400
    temperature: float = 0.1
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            v = e.get(name)
            return default if v is None or v == "" else v

        mode = get("DEPLOYMENT_MODE", "local").lower()
        if mode not in MODE_DEFAULTS:
            raise ValueError(f"DEPLOYMENT_MODE must be one of {sorted(MODE_DEFAULTS)}, got {mode!r}")
        d = MODE_DEFAULTS[mode]

        s = cls(
            app_env=get("APP_ENV", "dev").lower(),
            deployment_mode=mode,
            embedding_provider=get("EMBEDDING_PROVIDER", d["embedding"]).lower(),
            llm_provider=get("LLM_PROVIDER", d["llm"]).lower(),
            vector_store=get("VECTOR_STORE", d["store"]).lower(),
            vector_store_path=get("VECTOR_STORE_PATH", "var/vectors"),
            embedding_dim=int(get("EMBEDDING_DIM", str(d["dim"]))),
            aws_region=get("AWS_REGION", "ap-south-1"),
            bedrock_embed_model_id=get("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            bedrock_llm_model_id=get("BEDROCK_LLM_MODEL_ID"),
            bedrock_guardrail_id=get("BEDROCK_GUARDRAIL_ID"),
            bedrock_guardrail_version=get("BEDROCK_GUARDRAIL_VERSION"),
            openai_base_url=get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/"),
            openai_api_key=get("OPENAI_API_KEY"),
            llm_model=get("LLM_MODEL"),
            embed_model=get("EMBED_MODEL", "BAAI/bge-m3"),
            qdrant_url=get("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=get("QDRANT_API_KEY"),
            qdrant_collection=get("QDRANT_COLLECTION", "support_chunks"),
            database_url=get("DATABASE_URL", "sqlite:///var/conversations.db"),
            session_secret=get("SESSION_SECRET", DEV_SECRET),
            session_ttl_seconds=int(get("SESSION_TTL_SECONDS", "3600")),
            api_keys_file=get("API_KEYS_FILE", "var/api_keys.json"),
            top_k=int(get("RETRIEVAL_TOP_K", "8")),
            context_k=int(get("CONTEXT_K", "3")),
            min_confidence=float(get("MIN_CONFIDENCE", str(d["min_conf"]))),
            min_hits=int(get("MIN_HITS", "1")),
            history_turns=int(get("HISTORY_TURNS", "4")),
            max_question_chars=int(get("MAX_QUESTION_CHARS", "1000")),
            max_answer_tokens=int(get("MAX_ANSWER_TOKENS", "400")),
            temperature=float(get("LLM_TEMPERATURE", "0.1")),
            log_level=get("LOG_LEVEL", "INFO").upper(),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {sorted(EMBEDDING_PROVIDERS)}")
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(LLM_PROVIDERS)}")
        if self.vector_store not in STORES:
            raise ValueError(f"VECTOR_STORE must be one of {sorted(STORES)}")
        if self.app_env != "dev" and (self.session_secret == DEV_SECRET or len(self.session_secret) < 32):
            raise ValueError("SESSION_SECRET must be set to a random value of >= 32 chars outside dev")
        if self.llm_provider == "bedrock" and not self.bedrock_llm_model_id:
            raise ValueError("BEDROCK_LLM_MODEL_ID is required when LLM_PROVIDER=bedrock "
                             "(choose a model enabled in your account/region)")
        if self.llm_provider == "openai" and not self.llm_model:
            raise ValueError("LLM_MODEL is required when LLM_PROVIDER=openai (the model name served by vLLM)")
        if self.context_k > self.top_k:
            raise ValueError("CONTEXT_K must be <= RETRIEVAL_TOP_K")
