"""Real Ollama calls only. No canned responses or fallback embeddings."""
from __future__ import annotations

import json
import logging
import math
import os
from typing import TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


def normalized(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(x) for x in vector):
        raise ValueError("Embedding must contain finite values")
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("Embedding cannot be a zero vector")
    return [x / norm for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions differ. Rebuild the index.")
    return sum(x * y for x, y in zip(normalized(a), normalized(b)))


class Model:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
        # Bump this when replacing weights under the same embedding model tag.
        self.embedding_space = self.embedding_model + ":" + os.getenv("EMBEDDING_VERSION", "1")
        self.client = httpx.AsyncClient(
            base_url=(base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        await self.client.aclose()

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            response = await self.client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as error:
            raise RuntimeError("Cannot reach Ollama. Check OLLAMA_URL and start Ollama.") from error
        except httpx.TimeoutException as error:
            raise RuntimeError("Ollama timed out. Check available RAM and model size.") from error
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Ollama returned HTTP {error.response.status_code} for {path}. "
                "Check model installation; embedding context errors require smaller CHUNK_CHARS "
                "or a shorter question. No text was silently truncated."
            ) from error

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = await self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": texts, "truncate": False},
        )
        vectors = result.get("embeddings", [])
        if len(vectors) != len(texts):
            raise ValueError("Ollama returned an unexpected embedding count")
        values = [normalized(vector) for vector in vectors]
        if len({len(vector) for vector in values}) != 1:
            raise ValueError("Ollama returned inconsistent embedding dimensions")
        return values

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def json(self, schema: type[T], system: str, text: str, attempts: int = 3) -> T:
        if not 1 <= attempts <= 5:
            raise ValueError("attempts must be between 1 and 5")
        shape = schema.model_json_schema()
        messages = [
            {"role": "system", "content": system + "\nReturn JSON matching: " + json.dumps(shape)},
            {"role": "user", "content": text},
        ]
        for attempt in range(1, attempts + 1):
            result = await self._post("/api/chat", {
                "model": self.model,
                "messages": messages,
                "format": shape,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 1800},
            })
            content = result["message"]["content"]
            try:
                return schema.model_validate_json(content)
            except ValidationError as error:
                issues = [{"location": item["loc"], "type": item["type"]}
                          for item in error.errors(include_input=False, include_url=False)]
                logger.warning("structured_output_invalid attempt=%s issues=%s", attempt, issues)
                if attempt == attempts:
                    raise ValueError("Model output failed schema validation after all attempts") from error
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Correct these errors: " + json.dumps(issues)},
                ])
        raise RuntimeError("No model response")
