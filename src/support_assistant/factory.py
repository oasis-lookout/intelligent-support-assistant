"""Wires concrete providers from Settings. Heavy imports (boto3, qdrant) stay lazy."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .memory.conversations import ConversationStore, Database
from .providers.base import LLM, Embedder
from .rag.service import AssistantService
from .security.apikeys import ApiKeyRegistry
from .stores.base import VectorStore


def build_embedder(s: Settings) -> Embedder:
    if s.embedding_provider == "hash":
        from .providers.local import HashEmbedder
        return HashEmbedder(s.embedding_dim)
    if s.embedding_provider == "bedrock":
        from .providers.bedrock import BedrockEmbedder
        return BedrockEmbedder(s.aws_region, s.bedrock_embed_model_id, s.embedding_dim)
    from .providers.openai_compat import OpenAICompatEmbedder
    return OpenAICompatEmbedder(s.openai_base_url, s.embed_model, s.embedding_dim, s.openai_api_key)


def build_llm(s: Settings) -> LLM:
    if s.llm_provider == "extractive":
        from .providers.local import ExtractiveLLM
        return ExtractiveLLM()
    if s.llm_provider == "bedrock":
        from .providers.bedrock import BedrockLLM
        return BedrockLLM(s.aws_region, s.bedrock_llm_model_id, s.bedrock_guardrail_id, s.bedrock_guardrail_version)
    from .providers.openai_compat import OpenAICompatLLM
    return OpenAICompatLLM(s.openai_base_url, s.llm_model, s.openai_api_key)


def build_store(s: Settings) -> VectorStore:
    if s.vector_store == "memory":
        from .stores.memory import InMemoryVectorStore
        return InMemoryVectorStore(path=s.vector_store_path, dim=s.embedding_dim)
    from .stores.qdrant import QdrantVectorStore
    return QdrantVectorStore(s.qdrant_url, s.qdrant_collection, s.embedding_dim, s.qdrant_api_key)


@dataclass
class Container:
    settings: Settings
    service: AssistantService
    conversations: ConversationStore
    registry: ApiKeyRegistry


def build_container(s: Settings) -> Container:
    conversations = ConversationStore(Database(s.database_url))
    service = AssistantService(s, build_embedder(s), build_llm(s), build_store(s), conversations)
    return Container(s, service, conversations, ApiKeyRegistry.load(s.api_keys_file))
