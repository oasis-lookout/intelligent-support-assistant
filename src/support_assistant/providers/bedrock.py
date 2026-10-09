"""Amazon Bedrock providers: Titan Text Embeddings V2 + Converse API for generation."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import LLM, Embedder, alternate_roles, normalize


def _client(region: str):
    import boto3
    from botocore.config import Config

    return boto3.client("bedrock-runtime", region_name=region,
                        config=Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=60))


class BedrockEmbedder(Embedder):
    """Titan V2 supports output dimensions 256 / 512 / 1024. It has no batch API, so documents are
    embedded concurrently."""

    def __init__(self, region: str, model_id: str, dim: int = 1024, max_workers: int = 8) -> None:
        if dim not in (256, 512, 1024):
            raise ValueError("Titan Text Embeddings V2 supports dim in {256, 512, 1024}")
        self._client = _client(region)
        self.model_id, self.dim, self._workers = model_id, dim, max_workers
        self.name = f"bedrock:{model_id}:{dim}"

    def _one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
        resp = self._client.invoke_model(modelId=self.model_id, body=body,
                                         contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["embedding"]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if len(texts) == 1:
            return normalize(np.array([self._one(texts[0])]))
        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            return normalize(np.array(list(ex.map(self._one, texts))))


class BedrockLLM(LLM):
    def __init__(self, region: str, model_id: str, guardrail_id: str = "", guardrail_version: str = "") -> None:
        self._client = _client(region)
        self.model_id = model_id
        self._guardrail = ({"guardrailIdentifier": guardrail_id, "guardrailVersion": guardrail_version or "DRAFT"}
                           if guardrail_id else None)
        self.name = f"bedrock:{model_id}"

    def generate(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in alternate_roles(messages)]
        kwargs = dict(modelId=self.model_id, system=[{"text": system}], messages=msgs,
                      inferenceConfig={"maxTokens": max_tokens, "temperature": temperature})
        if self._guardrail:
            kwargs["guardrailConfig"] = self._guardrail
        resp = self._client.converse(**kwargs)
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts).strip()
